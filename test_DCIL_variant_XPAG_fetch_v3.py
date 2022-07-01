#!/usr/bin python -w

import os

os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = "0.5"

import time
import jax

print(jax.lib.xla_bridge.get_backend().platform)

#jax.config.update('jax_platform_name', "cpu")

from datetime import datetime
import argparse

import flax

import xpag
# from xpag.wrappers import gym_vec_env

from xpag.buffers import DefaultEpisodicBuffer
# from xpag.samplers import DefaultEpisodicSampler, HER
from xpag.tools import mujoco_notebook_replay

from xpag.tools.eval import single_rollout_eval
from xpag.tools.utils import hstack
from xpag.tools.logging import eval_log_reset
from xpag.tools.timing import timing_reset
import matplotlib.pyplot as plt
from matplotlib import collections as mc
import numpy as np
import copy

import gym_gfetch

## DCIL versions
from wrappers.gym_vec_env_mujoco import gym_vec_env
from skill_extractor import skills_extractor_Mj
from samplers import HER_DCIL_variant_v2
from goalsetters import DCILGoalSetterMj_variant_v3
from agents import SAC_variant

import pdb

def visu_success_zones(eval_env, skill_sequence, ax):
	"""
	Visualize success zones as sphere of radius eps_success around skill-goals
	"""

	for skill in skill_sequence:
		starting_state, _, _ = skill
		obs, full_state = starting_state
		goal = eval_env.project_to_goal_space(obs)

		u, v = np.mgrid[0:2*np.pi:20j, 0:np.pi:10j]

		x = goal[0] + 0.075*np.cos(u)*np.sin(v)
		y = goal[1] + 0.075*np.sin(u)*np.sin(v)
		z = goal[2] + 0.075*np.cos(v)
		ax.plot_wireframe(x, y, z, color="blue", alpha = 0.1)

	return

def plot_traj(eval_env, trajs, traj_eval, skill_sequence, save_dir, it=0):
	fig = plt.figure()
	ax = fig.add_subplot(projection='3d')

	for traj in trajs:
		# print("traj = ", traj)
		for i in range(traj[0].shape[0]):
			X = [eval_env.project_to_goal_space(state[i])[0] for state in traj]
			Y = [eval_env.project_to_goal_space(state[i])[1] for state in traj]
			Z = [eval_env.project_to_goal_space(state[i])[2] for state in traj]
			ax.plot(X,Y,Z, c="lightsteelblue", alpha = 0.4)
			X_obj = [eval_env.project_to_goal_space(state[i])[3] for state in traj]
			Y_obj = [eval_env.project_to_goal_space(state[i])[4] for state in traj]
			Z_obj = [eval_env.project_to_goal_space(state[i])[5] for state in traj]
			ax.plot(X_obj,Y_obj,Z_obj, c="darkseagreen", alpha = 0.8)

	X_eval = [eval_env.project_to_goal_space(state[0])[0] for state in traj_eval]
	Y_eval = [eval_env.project_to_goal_space(state[0])[1] for state in traj_eval]
	Z_eval = [eval_env.project_to_goal_space(state[0])[2] for state in traj_eval]
	ax.plot(X_eval, Y_eval, Z_eval, c = "blue")

	X_eval_obj = [eval_env.project_to_goal_space(state[0])[3] for state in traj_eval]
	Y_eval_obj = [eval_env.project_to_goal_space(state[0])[4] for state in traj_eval]
	Z_eval_obj = [eval_env.project_to_goal_space(state[0])[5] for state in traj_eval]
	ax.plot(X_eval_obj, Y_eval_obj, Z_eval_obj, c = "red")

	visu_success_zones(eval_env, skill_sequence, ax)

	for _azim in range(45, 360, 90):
		ax.view_init(azim=_azim)
		plt.savefig(save_dir + "/trajs_azim_" + str(_azim) + "_it_" + str(it) + ".png")
	# plt.savefig(save_dir + "/trajs_it_"+str(it)+".png")
	plt.close(fig)
	return

import torch


def eval_traj(env, eval_env, agent, goalsetter):
	traj = []
	observation = goalsetter.reset(eval_env, eval_env.reset())
	eval_done = False
	nb_skills_success = 0

	while goalsetter.curr_indx[0] <= goalsetter.nb_skills and not eval_done:
		# skill_success = False
		# print("curr_indx = ", goalsetter.curr_indx)
		max_steps = eval_env.get_max_episode_steps()
		# print("max_steps = ", max_steps)
		for i_step in range(0,int(max_steps[0])):
			#print("eval_env.skill_manager.indx_goal = ", eval_env.skill_manager.indx_goal)
			traj.append(observation["observation"].copy())
			if hasattr(env, "obs_rms"):
				action = agent.select_action(np.hstack((env._normalize_shape(observation["observation"],env.obs_rms["observation"]),
													env._normalize_shape(observation["desired_goal"],env.obs_rms["achieved_goal"]),
													observation["bin_skill_indx"])),
					deterministic=True,
				)
			else:
				action = agent.select_action(np.hstack((observation["observation"], observation["desired_goal"], observation["bin_skill_indx"])),
				deterministic=True,
				)

			# print("action = ", action)
			observation, _, done, info = goalsetter.step(
				eval_env, observation, action, *eval_env.step(action)
			)

			# print("observation eval = ", observation["observation"][0][:15])
			# print("observation.shape = ", observation["observation"].shape)
			# print("observation = ", eval_env.project_to_goal_space(observation["observation"].reshape(268,)))
			# print("done = ", done)
			if done.max():
				if info["is_success"]==1:
					nb_skills_success+=1
				observation, next_skill_avail = goalsetter.shift_skill(eval_env)
				break
		if not next_skill_avail:
			eval_done = True
	print("nb skills success = ", nb_skills_success)
	return traj, nb_skills_success

def visu_transitions(eval_env, transitions, it=0):
	fig, ax = plt.subplots()
	eval_env.plot(ax)
	for i in range(transitions["observation.achieved_goal"].shape[0]):
		obs_ag = transitions["observation.achieved_goal"][i,:2]
		next_obs_ag = transitions["next_observation.achieved_goal"][i,:2]
		X = [obs_ag[0], next_obs_ag[0]]
		Y = [obs_ag[1], next_obs_ag[1]]
		# if transitions["is_success"][i].max():
		if transitions["reward"][i].max():
			ax.plot(X,Y, c="green",marker = "o")
		else:
			ax.plot(X,Y,c="black",marker = "o")

		obs_dg = transitions["observation.desired_goal"][i,:2]
		next_obs_dg = transitions["next_observation.desired_goal"][i,:2]
		X = [obs_dg[0], next_obs_dg[0]]
		Y = [obs_dg[1], next_obs_dg[1]]
		# if transitions["is_success"][i].max():
		if transitions["reward"][i].max():
			ax.plot(X,Y, c="grey")
		else:
			ax.plot(X,Y,c="brown")

	plt.savefig(save_dir + "/transitions_"+str(it)+".png")
	plt.close(fig)
	return

if (__name__=='__main__'):

	parser = argparse.ArgumentParser(description='Argument for DCIL')
	parser.add_argument('--demo_path', help='path to demonstration file')
	parser.add_argument('--save_path', help='path to save directory')
	parsed_args = parser.parse_args()

	env_args = {}
	env_args["demo_path"] = str(parsed_args.demo_path)

	num_envs = 1  # the number of rollouts in parallel during training
	env, eval_env, env_info = gym_vec_env('GFetchGoal-v0', num_envs)
	print("env = ", env)

	s_extractor = skills_extractor_Mj(parsed_args.demo_path, eval_env, eps_state=0.5, beta=2.)
	print("nb_skills (remember to adjust value clipping in sac_from_jaxrl)= ", len(s_extractor.skills_sequence))

	goalsetter = DCILGoalSetterMj_variant_v3()
	goalsetter.set_skills_sequence(s_extractor.skills_sequence, env)#, n_skills=2)
	eval_goalsetter = DCILGoalSetterMj_variant_v3()
	eval_goalsetter.set_skills_sequence(s_extractor.skills_sequence, eval_env)#, n_skills=2)

	# print(goalsetter.skills_observations)
	# print(goalsetter.skills_full_states)
	# print(goalsetter.skills_max_episode_steps)
	# print("goalsetter.skills_sequence = ", goalsetter.skills_sequence)

	batch_size = 256
	gd_steps_per_step = 1.5
	start_training_after_x_steps = env_info['max_episode_steps'] * 50
	max_steps = 500_000
	evaluate_every_x_steps = 2_000
	save_agent_every_x_steps = 50_000

	## create log dir
	now = datetime.now()
	dt_string = 'DCIL_fetch_v3_%s_%s' % (datetime.now().strftime('%Y%m%d'), str(os.getpid()))
	# save_dir = os.path.join('/gpfswork/rech/kcr/ubj56je', 'results', 'xpag', 'DCIL_XPAG_dubins', dt_string)
	# save_dir = os.path.join(os.path.expanduser('~'), 'results', 'xpag', 'DCIL_XPAG_dubins', dt_string)
	save_dir = str(parsed_args.save_path) + dt_string
	os.mkdir(save_dir)
	## log file for success ratio
	f_ratio = open(save_dir + "/ratio.txt", "w")
	f_critic_loss = open(save_dir + "/critic_loss.txt", "w")
	f_nb_skills_success = open(save_dir + "/nb_skills_success.txt", "w")

	save_episode = True
	plot_projection = None

	params = {
		"actor_lr": 0.001,
		"backup_entropy": False,
		"critic_lr": 0.001,
		"discount": 0.98,
		# "hidden_dims": (512, 512, 512),
		"hidden_dims": (400,300),
		"init_temperature": 0.0001,
		"target_entropy": None,
		"target_update_period": 1,
		"tau": 0.005,
		"temp_lr": 0.0003,
	}

	with open(save_dir + "/sac_params.txt", "w") as f:
		print(params, file=f)

	agent = SAC_variant(
		env_info['observation_dim'] if not env_info['is_goalenv']
		else env_info['observation_dim'] + env_info['desired_goal_dim'] + len(bin(len(s_extractor.skills_sequence))[2:]),
		env_info['action_dim'],
		params = params
	)
	sampler = DefaultEpisodicSampler() if not env_info['is_goalenv'] else HER_DCIL_variant_v2(env.envs[0].compute_reward, env)
	buffer_ = DefaultEpisodicBuffer(
		max_episode_steps=env_info['max_episode_steps'],
		buffer_size=1_000_000,
		sampler=sampler
	)

	eval_log_reset()
	timing_reset()
	observation = goalsetter.reset(env, env.reset())
	print("observation = ", observation)
	trajs = []
	traj = []
	info_train = None
	num_success = 0
	num_rollouts = 0

	for i in range(max_steps // env_info["num_envs"]):
		# print("learn: ", eval_env.project_to_goal_space(observation["observation"][0]))
		traj.append(observation["observation"].copy())
		# print("\n")

		if not i % max(evaluate_every_x_steps // env_info["num_envs"], 1):
			print("i : ", i)
			# t1_logs = time.time()
			print("")
			if hasattr(env, "obs_rms"):
				print("do update ? ", env.do_update)
				print("RMS = ", env.obs_rms["observation"].mean[0][:10])
			# single_rollout_eval(
			# 	i * env_info["num_envs"],
			# 	eval_env,
			# 	env_info,
			# 	agent,
			# 	save_dir=save_dir,
			# 	plot_projection=plot_projection,
			# 	save_episode=save_episode,
			# )
			traj_eval, nb_skills_success = eval_traj(env, eval_env, agent, eval_goalsetter)
			# print("traj_eval = ", traj_eval)
			f_nb_skills_success.write(str(nb_skills_success) + "\n")
			plot_traj(eval_env, trajs, traj_eval, s_extractor.skills_sequence, save_dir, it=i)
			# visu_value(env, eval_env, agent, s_extractor.skills_sequence, save_dir, it=i)
			# visu_value_maze(env, eval_env, agent, s_extractor.skills_sequence, save_dir, it=i)

			# t2_logs = time.time()
			# print("logs time = ", t2_logs - t1_logs)

			if i > 2000:
				# visu_transitions(eval_env, transitions, it = i)
				print("info_train = ", info_train)
			trajs = []
			traj = []
			# if info_train is not None:
			# 	print("rewards = ", max(info_train["rewards"]))

			if num_rollouts > 0:
				print("ratio = ", float(num_success/num_rollouts))
				f_ratio.write(str(float(num_success/num_rollouts)) + "\n")
				num_success = 0
				num_rollouts = 0

		if not i % max(save_agent_every_x_steps // env_info["num_envs"], 1):
			if save_dir is not None:
				agent.save(os.path.join(save_dir, "agent"))

		if i * env_info["num_envs"] < start_training_after_x_steps:
			action = env_info["action_space"].sample()
		else:
			env.do_update = False
			t1_a_select = time.time()
			if hasattr(eval_env, "obs_rms"):
				action = agent.select_action(
					observation
					if not env_info["is_goalenv"]
					else np.hstack((env._normalize(observation["observation"], env.obs_rms["observation"]),
									env._normalize(observation["desired_goal"], env.obs_rms["achieved_goal"]),
									observation["bin_skill_indx"])),
					deterministic=False,
				)
			else:
				action = agent.select_action(
					observation
					if not env_info["is_goalenv"]
					else np.hstack((observation["observation"], observation["desired_goal"], observation["bin_skill_indx"])),
					deterministic=False,
				)
			t2_a_select = time.time()
			# print("action selection time = ", t2_a_select - t1_a_select)

			t1_train = time.time()
			for _ in range(max(round(gd_steps_per_step * env_info["num_envs"]), 1)):
				transitions = buffer_.sample(batch_size)
				info_train = agent.train_on_batch(transitions)
			t2_train = time.time()
			# print("training time = ", t2_train - t1_train)

			if i % 100 == 0:
				f_critic_loss.write(str(info_train["critic_loss"]) + "\n")
				f_critic_loss.flush()

		t1_step = time.time()
		next_observation, reward, done, info = goalsetter.step(
			env, observation, action, *env.step(action)
		)
		t2_step = time.time()
		# print("step time = ", t2_step - t1_step)

		# print("done = ", done)

		# pdb.set_trace()

		step = {
			"observation": observation,
			"action": action,
			"reward": reward,
			"truncation": info["truncation"],
			"done": done,
			"next_observation": next_observation,
		}

		# print("step = ", step)

		if env_info["is_goalenv"]:
			step["done_from_env"] = info["done_from_env"]
			step["is_success"] = info["is_success"]
			step["last_skill"] = (info["skill_indx"] == info["next_skill_indx"]).reshape(observation["desired_goal"].shape[0], 1)
			step["skill_indx"] = observation["bin_skill_indx"].reshape(observation["desired_goal"].shape[0], len(bin(len(s_extractor.skills_sequence))[2:]))
			step["next_skill_indx"] = observation["bin_next_skill_indx"].reshape(observation["desired_goal"].shape[0], len(bin(len(s_extractor.skills_sequence))[2:]))
			step["next_skill_goal"] = info["next_skill_goal"].reshape(observation["desired_goal"].shape)

		buffer_.insert(step)

		observation = next_observation.copy()

		t1_reset_time = time.time()
		if done.max():
			traj.append(observation["observation"].copy())
			num_rollouts += 1
			if info["is_success"].max() == 1:
				num_success += 1
			# use store_done() if the buffer is an episodic buffer
			if hasattr(buffer_, "store_done"):
				buffer_.store_done()
			observation = goalsetter.reset_done(env, env.reset_done())
			if len(traj) > 0:
				trajs.append(traj)
				traj = []
		t2_reset_time = time.time()
		# print("reset time = ", t2_reset_time - t1_reset_time)

	f_ratio.close()
	f_critic_loss.close()
	f_nb_skills_success.close()
