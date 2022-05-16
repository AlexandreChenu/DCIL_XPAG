#!/usr/bin python -w

import os

#os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
# os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = "0.5"

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

import gym_gmazes

## DCIL versions
from wrappers.gym_vec_env import gym_vec_env
from skill_extractor import *
from samplers import HER_DCIL
from goalsetters import DCILGoalSetter
from agents import SAC, SACTORCH

import pdb

def plot_traj(trajs, traj_eval, skill_sequence, save_dir, it=0):
	fig, ax = plt.subplots()
	env.plot(ax)
	for traj in trajs:
		for i in range(traj[0].shape[0]):
			X = [state[i][0] for state in traj]
			Y = [state[i][1] for state in traj]
			Theta = [state[i][2] for state in traj]
			ax.plot(X,Y, marker=".")

			for x, y, t in zip(X,Y,Theta):
				dx = np.cos(t)
				dy = np.sin(t)
				#arrow = plt.arrow(x,y,dx*0.1,dy*0.1,alpha = 0.6,width = 0.01, zorder=6)

	X_eval = [state[0][0] for state in traj_eval]
	Y_eval = [state[0][1] for state in traj_eval]
	ax.plot(X_eval, Y_eval, c = "red")

	circles = []
	for skill in skill_sequence:
		starting_state, _, _ = skill
		obs, full_state = starting_state
		# print("obs = ", obs)
		circle = plt.Circle((obs[0][0], obs[0][1]), 0.1, color='m', alpha = 0.6)
		circles.append(circle)
		# ax.add_patch(circle)
	coll = mc.PatchCollection(circles, color="plum", zorder = 4)
	ax.add_collection(coll)

	plt.savefig(save_dir + "/trajs_it_"+str(it)+".png")
	plt.close(fig)
	return

import torch
@torch.no_grad()
def visu_value(env, eval_env, agent, skill_sequence, save_dir, it=0):

	thetas = np.linspace(-torch.pi/2.,torch.pi/2.,100)

	values = []
	obs = eval_env.reset()
	#obs["observation"][0] = torch.tensor([ 0.33      ,  0.5       , -0.17363015])
	skill = skill_sequence[0]
	starting_state, _, goal = skill
	observation, full_state = starting_state
	obs["observation"][0][:] = observation[0][:]
	obs["desired_goal"][0][:] = goal[0][:2]
	for theta in list(thetas):
		obs["observation"][0][2] = theta
		#print("obs = ", obs["observation"])
		#print("dg = ", obs["desired_goal"])
		#print("stack = ", hstack(obs["observation"], obs["desired_goal"]))

		if hasattr(env, "obs_rms"):
			# print("normalization visu_value 1")
			norm_obs = env.normalize(obs)
			action = agent.select_action(hstack(norm_obs["observation"], norm_obs["desired_goal"]),
				deterministic=True,
			)
			value = agent.value(hstack(norm_obs["observation"], norm_obs["desired_goal"]), action)
		else:
			action = agent.select_action(hstack(obs["observation"], obs["desired_goal"]),
				deterministic=True,
			)
			value = agent.value(hstack(obs["observation"], obs["desired_goal"]), action)
		values.append(value)

	fig, ax = plt.subplots()
	plt.plot(list(thetas), values,label="learned V(s,g')")
	plt.plot()
	plt.xlabel("theta")
	plt.ylabel("value")
	plt.legend()
	plt.savefig(save_dir + "/value_skill_1_it_"+str(it)+".png")
	plt.close(fig)

	values = []
	#thetas = np.linspace(0.,torch.pi,100)
	thetas = np.linspace(torch.pi/2.,3*torch.pi/2.,100)
	obs = eval_env.reset()

	skill = skill_sequence[8]
	starting_state, _, goal = skill
	observation, full_state = starting_state
	obs["observation"][0][:] = observation[0][:]
	obs["desired_goal"][0][:] = goal[0][:2]

	for theta in list(thetas):
		obs["observation"][0][2] = theta
		if hasattr(env, "obs_rms"):
			# print("normalization visu_value 2")
			norm_obs = env.normalize(obs)
			action = agent.select_action(hstack(norm_obs["observation"], norm_obs["desired_goal"]),
				deterministic=True,
			)
			value = agent.value(hstack(norm_obs["observation"], norm_obs["desired_goal"]), action)
		else:
			action = agent.select_action(hstack(obs["observation"], obs["desired_goal"]),
				deterministic=True,
			)
			value = agent.value(hstack(obs["observation"], obs["desired_goal"]), action)
		values.append(value)

	fig, ax = plt.subplots()
	plt.plot(list(thetas), values,label="learned V(s,g')")
	plt.plot()
	plt.xlabel("theta")
	plt.ylabel("value")
	plt.legend()
	plt.savefig(save_dir + "/value_skill_2_it_"+str(it)+".png")
	plt.close(fig)

	return values

def eval_traj(env, eval_env, agent, goalsetter):
	traj = []
	eval_env.reset()
	init_indx = torch.ones((eval_env.num_envs,1)).int()
	observation = goalsetter.reset(eval_env, eval_env.reset())
	eval_done = False

	while goalsetter.curr_indx[0] <= goalsetter.nb_skills and not eval_done:
		# skill_success = False
		# print("curr_indx = ", goalsetter.curr_indx)
		for i_step in range(0,eval_env.max_episode_steps.int()):
			#print("eval_env.skill_manager.indx_goal = ", eval_env.skill_manager.indx_goal)
			traj.append(observation["observation"])
			if hasattr(env, "obs_rms"):
				norm_observation = env.normalize(observation)
				action = agent.select_action(hstack(norm_observation["observation"], norm_observation["desired_goal"]),
				deterministic=True,
				)
			else:
				action = agent.select_action(hstack(observation["observation"], observation["desired_goal"]),
				deterministic=True,
				)
			observation, _, done, info = goalsetter.step(
	            env, observation, action, *eval_env.step(action)
	        )
			# print("done = ", done)
			if done.max():
				observation = goalsetter.shift_skill(eval_env)
				break
		if goalsetter.curr_indx[0] == goalsetter.nb_skills -1:
			eval_done = True
	return traj

def visu_transitions(eval_env, transitions, it=0):
	fig, ax = plt.subplots()
	eval_env.plot(ax)
	for i in range(transitions["observation.achieved_goal"].shape[0]):
		obs_ag = transitions["observation.achieved_goal"][i,:2]
		next_obs_ag = transitions["next_observation.achieved_goal"][i,:2]
		X = [obs_ag[0], next_obs_ag[0]]
		Y = [obs_ag[1], next_obs_ag[1]]
		if transitions["is_success"][i].max():
			ax.plot(X,Y, c="green")
		else:
			ax.plot(X,Y,c="red")

		obs_dg = transitions["observation.desired_goal"][i,:2]
		next_obs_dg = transitions["next_observation.desired_goal"][i,:2]
		X = [obs_dg[0], next_obs_dg[0]]
		Y = [obs_dg[1], next_obs_dg[1]]
		if transitions["is_success"][i].max():
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
	env, eval_env, env_info = gym_vec_env('GMazeGoalDubins-v0', num_envs)
	# print("env = ", env)

	s_extractor = skills_extractor(parsed_args.demo_path, eval_env)

	goalsetter = DCILGoalSetter()
	goalsetter.set_skills_sequence(s_extractor.skills_sequence, env)
	eval_goalsetter = DCILGoalSetter()
	eval_goalsetter.set_skills_sequence(s_extractor.skills_sequence, eval_env)

	# print(goalsetter.skills_observations)
	# print(goalsetter.skills_full_states)
	# print(goalsetter.skills_max_episode_steps)
	# print("goalsetter.skills_sequence = ", goalsetter.skills_sequence)

	batch_size = 256
	gd_steps_per_step = 1.5
	start_training_after_x_steps = env_info['max_episode_steps'] * 10
	max_steps = 100_000
	evaluate_every_x_steps = 2_000
	save_agent_every_x_steps = 100_000

	## create log dir
	now = datetime.now()
	dt_string = '%s_%s' % (datetime.now().strftime('%Y%m%d'), str(os.getpid()))
	# save_dir = os.path.join('/gpfswork/rech/kcr/ubj56je', 'results', 'xpag', 'DCIL_XPAG_dubins', dt_string)
	# save_dir = os.path.join(os.path.expanduser('~'), 'results', 'xpag', 'DCIL_XPAG_dubins', dt_string)
	save_dir = str(parsed_args.save_path) + dt_string
	os.mkdir(save_dir)
	## log file for success ratio
	f_ratio = open(save_dir + "/ratio.txt", "w")
	f_critic_loss = open(save_dir + "/critic_loss.txt", "w")

	save_episode = True
	plot_projection = None

	agent = SACTORCH(
		env_info['observation_dim'] if not env_info['is_goalenv']
		else env_info['observation_dim'] + env_info['desired_goal_dim'],
		env_info['action_dim'],
		1.,
		{}
	)
	sampler = DefaultEpisodicSampler() if not env_info['is_goalenv'] else HER_DCIL(env.compute_reward, env)
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
		traj.append(observation["observation"])

		if not i % max(evaluate_every_x_steps // env_info["num_envs"], 1):
			print("i : ", i)
			# single_rollout_eval(
			# 	i * env_info["num_envs"],
			# 	eval_env,
			# 	env_info,
			# 	agent,
			# 	save_dir=save_dir,
			# 	plot_projection=plot_projection,
			# 	save_episode=save_episode,
			# )
			traj_eval = eval_traj(env, eval_env, agent, eval_goalsetter)
			plot_traj(trajs, traj_eval, s_extractor.skills_sequence, save_dir, it=i)
			visu_value(env, eval_env, agent, s_extractor.skills_sequence, save_dir, it=i)
			if i > 2000:
				visu_transitions(eval_env, transitions, it = i)
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

		# if not i % max(save_agent_every_x_steps // env_info["num_envs"], 1):
		# 	if save_dir is not None:
		# 		agent.save(os.path.join(save_dir, "agent"))

		if i * env_info["num_envs"] < start_training_after_x_steps:
			action = env_info["action_space"].sample()
		else:
			if hasattr(eval_env, "obs_rms"):
				norm_observation = env.normalize(observation)
				action = agent.select_action(
					norm_observation
					if not env_info["is_goalenv"]
					else hstack(norm_observation["observation"], norm_observation["desired_goal"]),
					deterministic=False,
				)
				value = agent.value(hstack(norm_observation["observation"], norm_observation["desired_goal"]), action)
				# print("value = ", value)
			else:
				action = agent.select_action(
					observation
					if not env_info["is_goalenv"]
					else hstack(observation["observation"], observation["desired_goal"]),
					deterministic=False,
				)
			for _ in range(max(round(gd_steps_per_step * env_info["num_envs"]), 1)):
				transitions = buffer_.sample(batch_size)
				info_train = agent.train_on_batch(transitions)
				# print("info_train = ", info_train)
			if i % 100 == 0:
				f_critic_loss.write(str(info_train["critic_loss"]) + "\n")
				f_critic_loss.flush()

		next_observation, reward, done, info = goalsetter.step(
            env, observation, action, *env.step(action)
        )

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
		if env_info["is_goalenv"]:
			step["is_success"] = info["is_success"]
			step["next_skill_goal"] = info["next_skill_goal"].reshape(observation["desired_goal"].shape)
			step["next_skill_avail"] = info["next_skill_avail"].reshape(info["is_success"].shape)

		buffer_.insert(step)

		observation = next_observation

		if done.max():
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

	f_ratio.close()
	f_critic_loss.close()
