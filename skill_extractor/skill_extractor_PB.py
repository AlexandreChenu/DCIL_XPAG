import sys
import os
sys.path.append(os.getcwd())

import numpy as np
import math
import random
import copy
from collections import OrderedDict
import torch
import pickle

import matplotlib.pyplot as plt

import gym

class skills_extractor_PB():
	def __init__(self, demo_path, env, eval_env, eps_state=0.5, beta=1.25):

		self.env = env
		self.eval_env = eval_env

		# self.device = env.device

		self.eps_state = eps_state ## threshold distance in goal space for skill construction
		self.beta = beta

		self.L_observations, self.L_inner_states_env, self.L_inner_states_eval_env = self.get_demo(demo_path)
		self.L_observations = self.L_observations[:200]
		self.L_inner_states_env = self.L_inner_states_env[:200]
		self.L_inner_states_eval_env = self.L_inner_states_eval_env[:200]
		#
		self.demo_length = len(self.L_observations)
		print("self.demo_length = ", self.demo_length)
		#
		# print("demo_length = ", self.demo_length)

		# print("len(L_obs) = ", len(self.L_observations))
		# print("len(L_sim) = ", len(self.L_sim_states))
		# print("demo_length = ", self.demo_length)
		# print("self.L_observations = ", self.L_observations)
		self.skills_sequence_env, self.skills_sequence_eval_env = self.get_skills(self.L_observations, self.L_inner_states_env, self.L_inner_states_eval_env)
		assert len(self.skills_sequence_env) == len(self.skills_sequence_eval_env)

		# self.test_visu()
		print("nb_skills = ", len(self.skills_sequence_env))
		# print("demo_length = ", self.demo_length)

		## visual test
		# self.test_visu()

	def test_visu(self):
		fig = plt.figure()
		ax = fig.add_subplot(projection='3d')
		# self.env.plot(ax)
		X = [obs[0] for obs in self.L_observations]
		Y = [obs[1] for obs in self.L_observations]
		Z = [obs[2] for obs in self.L_observations]
		ax.plot(X,Y)

		X_start = []
		Y_start = []
		Z_start = []
		for skill in self.skills_sequence_env:
			state, _, _ = skill
			obs, _ = state
			X_start.append(obs[0])
			Y_start.append(obs[1])
			Z_start.append(obs[2])
		ax.scatter(X_start, Y_start, c="red", marker="o")
		# X_goal = []
		# Y_goal = []
		# for skill in self.skills_sequence:
		# 	# print("skill = ", skill)
		# 	_, _, goal = skill
		# 	# print("state = ", state[0])
		# 	X_goal.append(goal[0][0])
		# 	Y_goal.append(goal[0][1])
		# ax.scatter(X_goal, Y_goal, c="green", marker="o")
		plt.show()
		return

	def get_demo(self, demo_path, verbose=0):
		"""
		Extract demo from pickled file
		"""
		L_observations = []

		## different sets of inner states as they are linked to the instance
		L_inner_states_env = []
		L_inner_states_eval_env = []

		L_demo_states = os.listdir(demo_path)

		print("L_demo_states = ", L_demo_states)

		# for i in range(1,len(L_demo_states)):
		for i in range(1,400):
			state_file_path = [demo_path + "state" + str(i) + ".bullet"]
			# print("state_file_path = ", state_file_path)
			self.env.set_inner_state(state_file_path)
			obs = self.env.get_observation()
			# print("obs = ", obs["observation"][0][:3])
			L_observations.append(obs["observation"][0])
			inner_state = self.env.get_inner_state()
			L_inner_states_env.append(inner_state[0])

			self.eval_env.set_inner_state(state_file_path)
			inner_state_eval = self.eval_env.get_inner_state()
			L_inner_states_eval_env.append(inner_state_eval[0])

		return L_observations, L_inner_states_env, L_inner_states_eval_env

	def get_skills(self, L_observations, L_inner_states_env, L_inner_states_eval_env):
		"""
		Clean the demonstration trajectory according to hyperparameter eps_dist.
		"""
		skills_sequence_env = []
		skills_sequence_eval_env = []

		curr_obs = L_observations[0].copy()
		curr_starting_state_env = (L_observations[0].copy(), L_inner_states_env[0])
		curr_starting_state_eval_env = (L_observations[0].copy(), L_inner_states_eval_env[0])

		demo_length = 0

		i = 0
		while i < len(L_observations)-1:
			k = 0
			sum_dist = 0

			# cumulative distance
			while sum_dist <= self.eps_state and i + k < len(L_observations) - 1:
				shifted_obs = L_observations[i+k]

				sum_dist += self.env.goal_distance(self.env.project_to_goal_space(shifted_obs), self.env.project_to_goal_space(curr_obs))
				curr_obs = shifted_obs.copy()
				k += 1
				demo_length += 1

			# skills_sequence.append((curr_starting_state, int(self.beta*k), shifted_state.copy()))
			# skills_sequence_env.append((curr_starting_state_env, max(int(self.beta*k), 100), shifted_obs.copy()))
			# skills_sequence_eval_env.append((curr_starting_state_eval_env, max(int(self.beta*k), 100), shifted_obs.copy()))
			skills_sequence_env.append((curr_starting_state_env, 100, shifted_obs.copy()))
			skills_sequence_eval_env.append((curr_starting_state_eval_env, 100, shifted_obs.copy()))

			i = i + k

			print("k = ", k)

			# print("\ncurr_state[:10] = ", curr_state[:50])
			# print("L_observations[i][:10] = ", L_observations[i-1][:50])
			# print("test eq = ", curr_state[:100] == L_observations[i-1][:100])

			## check that curr_state corresponds to observation
			# assert (curr_state == L_observations[i-1]).all()
			# print("curr_state == ")
			# print("\ncurr_state[:10] = ", curr_state[:10])
			# print("L_observations[i][:10] = ", L_observations[i][:10])
			# print("L_sim_states[i][3][:10] = ", L_sim_states[i][3][:10])
			# assert (L_observations[i] == L_sim_states[i][3]).all()

			curr_starting_state_env = (curr_obs, L_inner_states_env[i-1])
			curr_starting_state_eval_env = (curr_obs, L_inner_states_eval_env[i-1])

		# print("len(skills_sequence) = ", len(skills_sequence))
		return skills_sequence_env, skills_sequence_eval_env


if (__name__=='__main__'):

	demo_path = "../demos/fetch_convert/1.demo"

	eval_env = gym.make("GFetchGoal-v0")

	## missing variables because we miss a wrapper here
	eval_env.num_envs = 1
	eval_env.device = "cpu"

	s_extractor = skills_extractor(demo_path, eval_env)

	print(s_extractor)
