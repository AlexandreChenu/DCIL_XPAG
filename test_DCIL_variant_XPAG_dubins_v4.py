#!/usr/bin python -w

import os

#os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
# os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = "0.5"

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
import math
import copy

import matplotlib.pyplot as plt
import matplotlib.colors as colors
import matplotlib.cm as cmx
import matplotlib as mpl

import gym_gmazes

## DCIL versions
from wrappers.gym_vec_env import gym_vec_env
from skill_extractor import *

from samplers import HER_DCIL_variant_v2 as HER_DCIL_variant ## state = obs + goal + index
from samplers import HER_no_index ## state = obs + goal
from samplers import EpisodicSampler_index ## state = obs + index

from goalsetters import DCILGoalSetter_variant_v4 as DCILGoalSetter_variant
from agents import SAC_variant

import seaborn
seaborn.set()
seaborn.set_style("whitegrid")

import pdb

def plot_car(state, ax, alpha, cabcolor="-r", truckcolor="-k"):  # pragma: no cover
        x = state[0] #self.pose[0]
        y = state[1] #self.pose[1]
        yaw = state[2] #self.pose[2]

        length = 0.2  # [m]
        width = 0.1  # [m]
        backtowheel = 0.05  # [m]
        # WHEEL_LEN = 0.03  # [m]
        # WHEEL_WIDTH = 0.02  # [m]
        # TREAD = 0.07  # [m]
        wb = 0.45  # [m]

        outline = np.array([[-backtowheel, (length - backtowheel), (length - backtowheel), -backtowheel, -backtowheel],
        					[width / 2, width / 2, - width / 2, -width / 2, width / 2]])

        Rot1 = np.array([[math.cos(yaw), math.sin(yaw)],
        				 [-math.sin(yaw), math.cos(yaw)]])

        outline = (outline.T.dot(Rot1)).T

        outline[0, :] += x
        outline[1, :] += y

        ax.plot(np.array(outline[0, :]).flatten(),
        		 np.array(outline[1, :]).flatten(), color=truckcolor, alpha = alpha)



def plot_traj(trajs, traj_eval, skill_sequence, save_dir, it=0):
	fig, ax = plt.subplots()

	env.plot(ax)

	# ax.set_xlim((-0.1, 4.))
	# ax.set_ylim((-0.1, 1.1))

	for traj in trajs:
		for i in range(traj[0].shape[0]):
			X = [state[i][0] for state in traj]
			Y = [state[i][1] for state in traj]
			Theta = [state[i][2] for state in traj]
			ax.plot(X,Y, marker=".", c="blue", alpha = 0.4)

			for x, y, t in zip(X,Y,Theta):
				dx = np.cos(t)
				dy = np.sin(t)
				#arrow = plt.arrow(x,y,dx*0.1,dy*0.1,alpha = 0.6,width = 0.01, zorder=6)

	X_eval = [state[0][0] for state in traj_eval]
	Y_eval = [state[0][1] for state in traj_eval]
	ax.plot(X_eval, Y_eval, alpha=0.6, c = "red")

	for state in traj_eval:
		# print("state = ", state)
		plot_car(state[0], ax, 0.6, truckcolor="red")

	circles = []
	for skill in skill_sequence:
		_, _, goal = skill
		# print("obs = ", obs)
		circle = plt.Circle((goal[0][0], goal[0][1]), 0.1, color='m', alpha = 0.4)
		circles.append(circle)
		# ax.add_patch(circle)
	coll = mc.PatchCollection(circles, color="plum", zorder = 4, alpha=0.4)
	ax.add_collection(coll)

	plt.savefig(save_dir + "/trajs_it_"+str(it)+".png")
	plt.close(fig)
	return

import torch
@torch.no_grad()
def visu_value(env, eval_env, agent, skill_sequence, save_dir, it=0):

	convert_table = np.eye(len(skill_sequence))

	thetas = np.linspace(-torch.pi/2.,torch.pi/2.,100)

	values = []
	obs = eval_env.reset()
	#obs["observation"][0] = torch.tensor([ 0.33      ,  0.5       , -0.17363015])
	skill = skill_sequence[0]
	# print("skill_0 = ", skill)
	starting_state, _, goal = skill
	observation, full_state = starting_state
	oh_skill_indx = convert_table[0]

	next_skill = skill_sequence[1]
	# print("skill_0 = ", skill)
	_, _, next_goal = next_skill
	oh_next_skill_indx = convert_table[1]

	obs["observation"][0][:] = observation[0][:]
	obs["desired_goal"][0][:] = goal[0][:2]
	obs["skill_indx"] = np.array([oh_skill_indx])
	for theta in list(thetas):
		obs["observation"][0][2] = theta
		#print("obs = ", obs["observation"])
		#print("dg = ", obs["desired_goal"])
		#print("stack = ", hstack(obs["observation"], obs["desired_goal"]))

		if hasattr(env, "obs_rms"):
			if agent.full_state: ## obs + index + goal (DCIL-II)
				action = agent.select_action(np.hstack((env._normalize_shape(obs["observation"],env.obs_rms["observation"]),
													env._normalize_shape(obs["desired_goal"],env.obs_rms["achieved_goal"]),
													obs["skill_indx"])),
					deterministic=True,
				)
				value = agent.value(np.hstack((env._normalize_shape(obs["observation"],env.obs_rms["observation"]),
										   env._normalize_shape(obs["desired_goal"],env.obs_rms["achieved_goal"]),
										   obs["skill_indx"])), action)

			elif not agent.goal: ## obs + index (option-like)
				action = agent.select_action(np.hstack((env._normalize_shape(obs["observation"],env.obs_rms["observation"]),
													obs["skill_indx"])),
					deterministic=True,
				)
				value = agent.value(np.hstack((env._normalize_shape(obs["observation"],env.obs_rms["observation"]),
										   obs["skill_indx"])), action)
			else: ## obs + goal (classic GCRL)
				action = agent.select_action(np.hstack((env._normalize_shape(obs["observation"],env.obs_rms["observation"]),
													env._normalize_shape(obs["desired_goal"],env.obs_rms["achieved_goal"]))),
					deterministic=True,
				)
				value = agent.value(np.hstack((env._normalize_shape(obs["observation"],env.obs_rms["observation"]),
										   env._normalize_shape(obs["desired_goal"],env.obs_rms["achieved_goal"]))), action)

		else:
			if agent.full_state:
				action = agent.select_action(np.hstack((obs["observation"], obs["desired_goal"], obs["skill_indx"])),
				deterministic=True,
				)
				value = agent.value(np.hstack((obs["observation"], obs["desired_goal"], obs["skill_indx"])), action)
			elif not agent.goal:
				action = agent.select_action(np.hstack((obs["observation"], obs["skill_indx"])),
				deterministic=True,
				)
				value = agent.value(np.hstack((obs["observation"], obs["skill_indx"])), action)
			else:
				action = agent.select_action(np.hstack((obs["observation"], obs["desired_goal"])),
				deterministic=True,
				)
				value = agent.value(np.hstack((obs["observation"], obs["desired_goal"])), action)
		values.append(value[0])

	fig, ax = plt.subplots()
	plt.plot(list(thetas), values,label="learned V(s,g')")
	plt.plot()
	plt.xlabel("theta")
	plt.ylabel("value")
	plt.legend()
	plt.savefig(save_dir + "/value_skill_1_it_"+str(it)+".png")
	plt.close(fig)


	return values



def eval_traj(env, eval_env, agent, goalsetter):
	traj = []
	init_indx = torch.ones((eval_env.num_envs,1)).int()
	observation = goalsetter.reset(eval_env, eval_env.reset())
	eval_done = False

	max_zone = 0

	while goalsetter.curr_indx[0] <= goalsetter.nb_skills and not eval_done:
		# skill_success = False
		# print("curr_indx = ", goalsetter.curr_indx)
		for i_step in range(0,eval_env.max_episode_steps.int()):
			#print("eval_env.skill_manager.indx_goal = ", eval_env.skill_manager.indx_goal)
			# print("observation = ", observation)

			## update max zone
			zone = eval_zone(observation["observation"][0])
			if zone > max_zone:
				max_zone = zone

			traj.append(observation["observation"])
			if hasattr(env, "obs_rms"):
				if agent.full_state: ## obs + index + goal (DCIL-II)
					action = agent.select_action(np.hstack((env._normalize_shape(observation["observation"],env.obs_rms["observation"]),
														env._normalize_shape(observation["desired_goal"],env.obs_rms["achieved_goal"]),
														observation["oh_skill_indx"])),
						deterministic=True,
					)
				elif not agent.goal: ## obs + index (option-like)
					action = agent.select_action(np.hstack((env._normalize_shape(observation["observation"],env.obs_rms["observation"]),
														observation["oh_skill_indx"])),
						deterministic=True,
					)
				else: ## obs + goal (classic GCRL)
					action = agent.select_action(np.hstack((env._normalize_shape(observation["observation"],env.obs_rms["observation"]),
														env._normalize_shape(observation["desired_goal"],env.obs_rms["achieved_goal"]))),
						deterministic=True,
					)

			else:
				if agent.full_state:
					action = agent.select_action(np.hstack((observation["observation"], observation["desired_goal"], observation["oh_skill_indx"])),
					deterministic=True,
					)
				elif not agent.goal:
					action = agent.select_action(np.hstack((observation["observation"], observation["oh_skill_indx"])),
					deterministic=True,
					)
				else:
					action = agent.select_action(np.hstack((observation["observation"], observation["desired_goal"])),
					deterministic=True,
					)
			# print("action = ", action)
			observation, _, done, info = goalsetter.step(
	            eval_env, observation, action, *eval_env.step(action)
	        )
			# print("done = ", done)
			if done.max():
				observation, next_skill_avail = goalsetter.shift_skill(eval_env)
				break
		if not next_skill_avail:
			eval_done = True
	return traj, max_zone

def eval_zone(state):
    x = state[0]
    y = state[1]
    if y < 1.:
        if x < 1.:
            return 1
        elif  x < 2.:
            return 2
        elif  x < 3.:
            return 3
        elif  x < 4.:
            return 4
        else:
            return 5
    elif y < 2.:
        if  x > 4.:
            return 6
        elif  x > 3.:
            return 7
        elif x > 2.:
            return 8
        else:
            return 11
    elif y < 3.:
        if x < 1.:
            return 11
        elif x < 2.:
            return 10
        elif x < 3.:
            return 9
        elif x < 4.:
            return 20
        else :
            return 21

    elif y < 4.:
        if x < 1.:
            return 12
        elif x < 2.:
            return 15
        elif x < 3.:
            return 16
        elif x < 4:
            return 19
        else :
            return 22
    else:
        if x < 1.:
            return 13
        elif x < 2.:
            return 14
        elif x < 3.:
            return 17
        elif x < 4:
            return 18
        else :
            return 23


if (__name__=='__main__'):

	parser = argparse.ArgumentParser(description='Argument for DCIL')
	parser.add_argument('--demo_path', help='path to demonstration file')
	parser.add_argument('--save_path', help='path to save directory')
	parser.add_argument('--goal', help='remove goal from obs')
	parser.add_argument('--index', help='remove index from obs')
	parsed_args = parser.parse_args()

	env_args = {}

	num_envs = 1  # the number of rollouts in parallel during training
	env, eval_env, env_info = gym_vec_env('GMazeGoalDubins-v0', num_envs)
	# print("env = ", env)

	print("env.num_envs = ", env.num_envs)
	print("eval_env.num_envs = ", eval_env.num_envs)

	s_extractor = skills_extractor(parsed_args.demo_path, eval_env)
	num_skills = len(s_extractor.skills_sequence)
	# s_extractor = skills_extractor(parsed_args.demo_path, eval_env)

	goalsetter = DCILGoalSetter_variant(env)
	goalsetter.set_skills_sequence(s_extractor.skills_sequence, env)
	eval_goalsetter = DCILGoalSetter_variant(eval_env)
	eval_goalsetter.set_skills_sequence(s_extractor.skills_sequence, eval_env)

	# print(goalsetter.skills_observations)
	# print(goalsetter.skills_full_states)
	# print(goalsetter.skills_max_episode_steps)
	# print("goalsetter.skills_sequence = ", goalsetter.skills_sequence)

	batch_size = 256
	gd_steps_per_step = 1.5
	start_training_after_x_steps = 500
	print("start_training_after_x_steps = ", start_training_after_x_steps)
	max_steps = 150_000
	evaluate_every_x_steps = 1000
	save_agent_every_x_steps = 100_000

	## create log dir
	now = datetime.now()
	dt_string = 'DCIL_XPAG_variant_' + str(bool(int(parsed_args.goal))) + '_' + str(bool(int(parsed_args.index))) + '_%s_%s' % (datetime.now().strftime('%Y%m%d'), str(os.getpid()))
	# save_dir = os.path.join('/gpfswork/rech/kcr/ubj56je', 'results', 'xpag', 'DCIL_XPAG_dubins', dt_string)
	# save_dir = os.path.join(os.path.expanduser('~'), 'results', 'xpag', 'DCIL_XPAG_dubins', dt_string)
	save_dir = str(parsed_args.save_path) + dt_string
	os.mkdir(save_dir)
	## log file for success ratio
	f_ratio = open(save_dir + "/ratio.txt", "w")
	f_critic_loss = open(save_dir + "/critic_loss.txt", "w")
	f_max_zone = open(save_dir + "/max_zone.txt", "w")

	save_episode = True
	plot_projection = None

	if bool(int(parsed_args.goal)) and bool(int(parsed_args.index)): ## full extended state
		observation_dim = env_info['observation_dim'] + env_info['desired_goal_dim'] + num_skills
	elif bool(int(parsed_args.index)): ## no goal (obs + index)
		observation_dim = env_info['observation_dim'] + num_skills
	else: ## no index (obs + goal)
		observation_dim = env_info['observation_dim'] + env_info['desired_goal_dim']

	agent = SAC_variant(
		env_info['observation_dim'] if not env_info['is_goalenv']
		else observation_dim,
		env_info['action_dim'],
		params = {
			"actor_lr": 0.001,
			"backup_entropy": False,
			"critic_lr": 0.001,
			"discount": 0.9,
			# "hidden_dims": (512, 512, 512),
			"hidden_dims": (400,300),
			"init_temperature": 0.001,
			"target_entropy": None,
			"target_update_period": 1,
			"tau": 0.005,
			"temp_lr": 0.0003,
		}
	)

	agent.goal = bool(int(parsed_args.goal))
	agent.index = bool(int(parsed_args.index))
	print("agent.goal = ", agent.goal)
	print("agent.index = ", agent.index)
	if (agent.goal and agent.index):
		agent.full_state = True
	else:
		agent.full_state = False
	print("agent.full_state = ", agent.full_state)

	if not env_info["is_goalenv"]:
		sampler = DefaultEpisodicSampler()
	elif agent.full_state:
		sampler = HER_DCIL_variant(env.compute_reward, env)
	elif not agent.goal:
		print("\n\n\n NO GOAL \n\n\n")
		sampler = EpisodicSampler_index(env)
	else:
		print("\n\n\n NO INDEX \n\n\n")
		sampler = HER_no_index(env.compute_reward, env)

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
		# print("\n")
		# print("observation = ", observation)
		if not i % max(evaluate_every_x_steps // env_info["num_envs"], 1):
			print("i : ", i)
			# t1_logs = time.time()
			print("")
			print("RMS = ", env.obs_rms["observation"].mean)

			traj_eval, max_zone = eval_traj(env, eval_env, agent, eval_goalsetter)

			with open(save_dir + "/traj_eval_it_" + str(i) + '.npy', 'wb') as f:
				np.save(f, np.array(traj_eval))

			plot_traj(trajs, traj_eval, s_extractor.skills_sequence, save_dir, it=i)

			# visu_value(env, eval_env, agent, skills_sequence, save_dir, it=i)
			# visu_value_maze(env, eval_env, agent, skills_sequence, save_dir, it=i)

			f_max_zone.write(str(max_zone) + "\n")
			f_max_zone.flush()

			if i > 300:
				print("info_train = ", info_train)

			trajs = []
			traj = []

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
			# t1_a_select = time.time()


			if hasattr(env, "obs_rms"):
				if agent.full_state: ## obs + index + goal (DCIL-II)
					action = agent.select_action(
						observation
						if not env_info["is_goalenv"]
						else np.hstack((env._normalize(observation["observation"], env.obs_rms["observation"]),
										env._normalize(observation["desired_goal"], env.obs_rms["achieved_goal"]),
										observation["oh_skill_indx"])),
						deterministic=False,
					)
				elif not agent.goal: ## obs + index (option-like)
					action = agent.select_action(
						observation
						if not env_info["is_goalenv"]
						else np.hstack((env._normalize(observation["observation"], env.obs_rms["observation"]),
										observation["oh_skill_indx"])),
						deterministic=False,
					)
				else: ## obs + goal (classic GCRL)
					action = agent.select_action(
						observation
						if not env_info["is_goalenv"]
						else np.hstack((env._normalize(observation["observation"], env.obs_rms["observation"]),
										env._normalize(observation["desired_goal"], env.obs_rms["achieved_goal"]))),
						deterministic=False,
					)
			else:
				if agent.full_state:
					action = agent.select_action(
						observation
						if not env_info["is_goalenv"]
						else np.hstack((observation["observation"], observation["desired_goal"], observation["oh_skill_indx"])),
						deterministic=False,
					)
				elif not agent.goal:
					action = agent.select_action(
						observation
						if not env_info["is_goalenv"]
						else np.hstack((observation["observation"], observation["oh_skill_indx"])),
						deterministic=False,
					)
				else:
					action = agent.select_action(
						observation
						if not env_info["is_goalenv"]
						else np.hstack((observation["observation"], observation["desired_goal"])),
						deterministic=False,
					)

			# t1_train = time.time()
			for _ in range(max(round(gd_steps_per_step * env_info["num_envs"]), 1)):
				transitions = buffer_.sample(batch_size)
				info_train = agent.train_on_batch(transitions)

			if i % 100 == 0:
				f_critic_loss.write(str(info_train["critic_loss"]) + "\n")
				f_critic_loss.flush()

		# t1_step = time.time()
		next_observation, reward, done, info = goalsetter.step(
            env, observation, action, *env.step(action)
        )


		step = {
			"observation": observation,
			"action": action,
			"reward": reward,
			"truncation": info["truncation"],
			"done": done,
			"next_observation": next_observation,
		}
		if env_info["is_goalenv"]:
			step["done_from_env"] = info["done_from_env"]
			step["is_success"] = info["is_success"]
			step["last_skill"] = (info["skill_indx"] == info["next_skill_indx"]).reshape(observation["desired_goal"].shape[0], 1)
			step["skill_indx"] = observation["oh_skill_indx"].reshape(observation["desired_goal"].shape[0], num_skills)
			step["next_skill_indx"] = observation["oh_next_skill_indx"].reshape(observation["desired_goal"].shape[0], num_skills)
			step["next_skill_goal"] = info["next_skill_goal"].reshape(observation["desired_goal"].shape)

		buffer_.insert(step)

		observation = next_observation.copy()

		# t1_reset_time = time.time()
		if done.max():
			# print("\n reset \n")
			traj.append(observation["observation"])
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
		# t2_reset_time = time.time()
		# print("reset time = ", t2_reset_time - t1_reset_time)

	f_ratio.close()
	f_critic_loss.close()
	f_max_zone.close()
