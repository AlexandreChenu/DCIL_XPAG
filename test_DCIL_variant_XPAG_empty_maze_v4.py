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
			ax.plot(X,Y, marker=".", c="blue", alpha = 0.7)

			for x, y, t in zip(X,Y,Theta):
				dx = np.cos(t)
				dy = np.sin(t)
				#arrow = plt.arrow(x,y,dx*0.1,dy*0.1,alpha = 0.6,width = 0.01, zorder=6)

	X_eval = [state[0][0] for state in traj_eval]
	Y_eval = [state[0][1] for state in traj_eval]
	ax.plot(X_eval, Y_eval, c = "red")

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


def plot_car(state, ax, alpha , truckcolor="-k"):  # pragma: no cover
        x = state[0] #self.pose[0]
        y = state[1] #self.pose[1]
        yaw = state[2] #self.pose[2]

        length = 0.2  # [m]
        width = 0.1  # [m]
        backtowheel = 0.05  # [m]
        wb = 0.45  # [m]

        outline = np.array([[-backtowheel, (length - backtowheel), (length - backtowheel), -backtowheel, -backtowheel],
        					[width / 2, width / 2, - width / 2, -width / 2, width / 2]])
        Rot1 = np.array([[np.cos(yaw), np.sin(yaw)],
        				 [-np.sin(yaw), np.cos(yaw)]])
        outline = (outline.T.dot(Rot1)).T

        outline[0, :] += x
        outline[1, :] += y

        ax.plot(np.array(outline[0, :]).flatten(),
        		 np.array(outline[1, :]).flatten(), color=truckcolor, alpha = alpha, linewidth=3.5, zorder=2)


def visu_value_maze(env, eval_env, agent, skill_sequence, save_dir, it=0):

		goals = [np.array([[1., 1.8]]), np.array([[1., 1.4]]), np.array([[1., 1.]]), np.array([[1., 0.6]]), np.array([[1., 0.2]])]
		plot_goals = [np.array([[1., 1.8]]), np.array([[1., 1.4]]), np.array([[1., 1.]]), np.array([[1., 0.6]]), np.array([[1., 0.2]])]
		goals_colors = ["m", "blue", "crimson",  "violet", "orange"]
		convert_table = np.eye(len(skill_sequence))

		skill_indx = 0
		final_skill_indx = 1
		frame_skip = 4

		fig = plt.figure()
		ax  = fig.add_axes([0.1, 0.1, 0.7, 0.85]) # [left, bottom, width, height]
		axc = fig.add_axes([0.85, 0.10, 0.05, 0.85])
		eval_env.plot(ax)

		circles = []
		for i in range(0,len(skill_sequence)):
			skl = skill_sequence[i]
			st, _, dg = skl
			obs, full_st = st
			circle = plt.Circle((dg[0][0], dg[0][1]), 0.05, color='m', alpha = 0.6)
			circles.append(circle)
		coll = mc.PatchCollection(circles, color="crimson", alpha = 1., zorder = 1)
		ax.add_collection(coll)

		ax.scatter(0.25,1.,c="black")


		for goal, goal_color in zip(plot_goals, goals_colors):
			circles_r = []
			circle_r = plt.Circle((goal[0][0], goal[0][1]), 0.05, alpha = 0.6)
			circles_r.append(circle_r)
			coll_r = mc.PatchCollection(circles_r, color=goal_color, alpha = 0.5, zorder = 1)
			ax.add_collection(coll_r)

		values = []
		states = []


		## intermediate states
		for goal_indx, desired_goal in enumerate(goals[1:-1]):

			obs = eval_env.reset()

			A = np.array([[0.2,1.]])
			B = desired_goal.copy() - np.array([[4*0.1*0.5, 0.]])
			theta = np.arctan2(B[0][1]-A[0][1],B[0][0]-A[0][0])

			for n_frame_skip in range(6,7):
				print("n_frame_skip = ", n_frame_skip)
				obs["observation"][0][:] = np.array([A[0][0] + n_frame_skip*0.1*0.5*np.cos(theta) ,A[0][1]+ n_frame_skip*0.1*0.5*np.sin(theta),theta])
				obs["achieved_goal"][0][:] = np.array([A[0][0] + n_frame_skip*0.1*0.5*np.cos(theta), A[0][1] + n_frame_skip*0.1*0.5*np.sin(theta)])
				obs["desired_goal"][0][:] = desired_goal[0][:2]
				obs["skill_indx"] = np.array([convert_table[skill_indx]])

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
				states.append(obs["observation"].copy())

		## intermediate states
		for goal_indx, desired_goal in enumerate(goals):

			obs = eval_env.reset()

			B = desired_goal.copy()
			theta = 0.

			obs["observation"][0][:] = np.array([B[0][0] - frame_skip*0.1*0.5*np.cos(theta) ,B[0][1]- frame_skip*0.1*0.5*np.sin(theta),theta])
			obs["achieved_goal"][0][:] = np.array([B[0][0] - frame_skip*0.1*0.5*np.cos(theta), B[0][1] - frame_skip*0.1*0.5*np.sin(theta)])
			obs["desired_goal"][0][:] = desired_goal[0][:2]
			obs["skill_indx"] = np.array([convert_table[skill_indx]])

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
			states.append(obs["observation"].copy())

		## final states
		for goal_indx, desired_goal in enumerate(goals):

			obs = eval_env.reset()

			A = desired_goal.copy()
			B = np.array([[1.8,1.]])
			theta = np.arctan2(B[0][1]-A[0][1],B[0][0]-A[0][0])

			obs["observation"][0][:] = np.array([B[0][0] - 2*frame_skip*0.1*0.5*np.cos(theta) ,B[0][1]- 2*frame_skip*0.1*0.5*np.sin(theta),theta])
			obs["achieved_goal"][0][:] = np.array([B[0][0] - 2*frame_skip*0.1*0.5*np.cos(theta), B[0][1] - 2*frame_skip*0.1*0.5*np.sin(theta)])
			obs["desired_goal"][0][:] = np.array([[1.8,1.]])
			obs["skill_indx"] = np.array([convert_table[final_skill_indx]])

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
			states.append(obs["observation"].copy())

		cmap = plt.cm.coolwarm
		# cNorm  = colors.Normalize(vmin=min(values), vmax=max(values))
		cNorm  = colors.Normalize(vmin=1., vmax=2.)
		scalarMap = cmx.ScalarMappable(norm=cNorm,cmap=cmap)

		for state, value in zip(states, values):

			colorVal = scalarMap.to_rgba(value)
			# print("colorVal = ", colorVal)
			plot_car(state[0,:], ax, 0.8, colorVal)
			# ax.arrow(state[0][0],state[0][1],dx*0.075,dy*0.075,alpha = 0.8,width = 0.015, color=colorVal, zorder=2)

		cb = mpl.colorbar.ColorbarBase(axc, cmap=cmap,
                                norm=cNorm,orientation='vertical')

		plt.savefig(save_dir + "/visu_value_landscape_" + str(goal_indx) + "_it_" + str(it) + ".png")
		plt.close(fig)

		with open(save_dir + '/values_landscape_goal_' + str(goal_indx) + "_it_" + str(it) + '.npy', 'wb') as f:
			np.save(f, np.array(values))

		return

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


def eval_relabelling(env, eval_env, agent, goalsetter):


	trajs = []
	goals = [np.array([[1., 1.8]]), np.array([[1., 1.4]]), np.array([[1., 1.]]),
			np.array([[1., 0.6]]), np.array([[1., 0.2]])]

	for goal in goals:
		traj = []
		observation = goalsetter.reset(eval_env, eval_env.reset())
		eval_done = False

		max_zone = 0

		for i_step in range(0,eval_env.max_episode_steps.int()):
			#print("eval_env.skill_manager.indx_goal = ", eval_env.skill_manager.indx_goal)
			# print("observation = ", observation)

			traj.append(observation["observation"])
			observation["desired_goal"][:,:] = goal[:,:]
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
		trajs.append(traj)
		traj = []

	return trajs

def plot_trajs_relabelling(trajs, save_dir, it=0):
	fig, ax = plt.subplots()

	env.plot(ax)

	# ax.set_xlim((-0.1, 4.))
	# ax.set_ylim((-0.1, 1.1))

	for traj in trajs:
		for i in range(traj[0].shape[0]):
			X = [state[i][0] for state in traj]
			Y = [state[i][1] for state in traj]
			Theta = [state[i][2] for state in traj]
			ax.plot(X,Y, marker=".", c="blue", alpha = 0.7)

			for x, y, t in zip(X,Y,Theta):
				dx = np.cos(t)
				dy = np.sin(t)
				#arrow = plt.arrow(x,y,dx*0.1,dy*0.1,alpha = 0.6,width = 0.01, zorder=6)

	X_eval = [state[0][0] for state in traj_eval]
	Y_eval = [state[0][1] for state in traj_eval]
	ax.plot(X_eval, Y_eval, c = "red")

	goals = [np.array([[1., 1.8]]), np.array([[1., 1.4]]), np.array([[1., 1.]]),
			np.array([[1., 0.6]]), np.array([[1., 0.2]])]

	circles = []
	for goal in goals:
		# print("obs = ", obs)
		circle = plt.Circle((goal[0][0], goal[0][1]), 0.1, color='m', alpha = 0.4)
		circles.append(circle)
		# ax.add_patch(circle)
	coll = mc.PatchCollection(circles, color="crimson", zorder = 4, alpha=0.4)
	ax.add_collection(coll)

	plt.savefig(save_dir + "/HER_trajs_it_"+str(it)+".png")
	plt.close(fig)
	return




if (__name__=='__main__'):

	parser = argparse.ArgumentParser(description='Argument for DCIL')
	# parser.add_argument('--demo_path', help='path to demonstration file')
	parser.add_argument('--save_path', help='path to save directory')
	parser.add_argument('--goal', help='remove goal from obs')
	parser.add_argument('--index', help='remove index from obs')
	parsed_args = parser.parse_args()

	env_args = {}

	num_envs = 1  # the number of rollouts in parallel during training
	env, eval_env, env_info = gym_vec_env('GMazeGoalEmptyDubins-v0', num_envs)
	# print("env = ", env)

	print("env.num_envs = ", env.num_envs)
	print("eval_env.num_envs = ", eval_env.num_envs)

	# s_extractor = skills_extractor(parsed_args.demo_path, eval_env)

	skills_sequence = [((np.array([[0.2, 1., 0.]]), np.array([[0.2, 1., 0.]])), 25, np.array([[1., 1., 0.]])),
	 					((np.array([[1., 1., 0.]]), np.array([[1., 1., 0.]])), 25, np.array([[1.8, 1., 0.]]))]

	num_skills = len(skills_sequence)

	goalsetter = DCILGoalSetter_variant(env)
	goalsetter.set_skills_sequence(skills_sequence, env)
	eval_goalsetter = DCILGoalSetter_variant(eval_env)
	eval_goalsetter.set_skills_sequence(skills_sequence, eval_env)

	# print(goalsetter.skills_observations)
	# print(goalsetter.skills_full_states)
	# print(goalsetter.skills_max_episode_steps)
	# print("goalsetter.skills_sequence = ", goalsetter.skills_sequence)

	batch_size = 256
	gd_steps_per_step = 1.5
	start_training_after_x_steps = 10000
	print("start_training_after_x_steps = ", start_training_after_x_steps)
	max_steps = 50_000
	evaluate_every_x_steps = 1000
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
			"discount": 0.99,
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
			# single_rollout_eval(
			# 	i * env_info["num_envs"],
			# 	eval_env,
			# 	env_info,
			# 	agent,
			# 	save_dir=save_dir,
			# 	plot_projection=plot_projection,
			# 	save_episode=save_episode,
			# )
			traj_eval, max_zone = eval_traj(env, eval_env, agent, eval_goalsetter)
			trajs_eval_relabelling = eval_relabelling(env, eval_env, agent, eval_goalsetter)

			with open(save_dir + "/traj_eval_it_" + str(i) + '.npy', 'wb') as f:
				np.save(f, np.array(traj_eval))

			plot_traj(trajs, traj_eval, skills_sequence, save_dir, it=i)
			plot_trajs_relabelling(trajs_eval_relabelling, save_dir, it=i)

			visu_value(env, eval_env, agent, skills_sequence, save_dir, it=i)
			visu_value_maze(env, eval_env, agent, skills_sequence, save_dir, it=i)

			f_max_zone.write(str(max_zone) + "\n")
			f_max_zone.flush()

			# t2_logs = time.time()
			# print("logs time = ", t2_logs - t1_logs)

			if i > 300:
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
			# t2_a_select = time.time()
			# print("action selection time = ", t2_a_select - t1_a_select)

			# t1_train = time.time()
			for _ in range(max(round(gd_steps_per_step * env_info["num_envs"]), 1)):
				transitions = buffer_.sample(batch_size)
				info_train = agent.train_on_batch(transitions)
			# t2_train = time.time()
			# print("training time = ", t2_train - t1_train)

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
