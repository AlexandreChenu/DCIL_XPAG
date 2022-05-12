# Copyright 2022 Nicolas Perrin-Gilbert.
#
# Licensed under the BSD 3-Clause License.

from abc import ABC, abstractmethod
from typing import Tuple

import xpag
from xpag.goalsetters import GoalSetter

import numpy as np

class DCILGoalSetter(GoalSetter, ABC):
	def __init__(self, do_overshoot = False):
		super().__init__("DCILGoalSetter")

		self.skills_sequence = []
		self.nb_skills = 0

		self.curr_indx = None

		self.do_overshoot = do_overshoot

	def step(
		self,
		env,
		observation,
		action,
		new_observation,
		reward,
		done,
		info,
		eval_mode=False,
	):

		## get next skill goal for bonus reward computation
		next_skill_indices, info['next_skill_avail'] = self.next_skill_indx()
		next_skill_indices = np.where(info['next_skill_avail'] == 1, next_skill_indices, self.curr_indx)
		info["next_skill_goal"] = self.skills_goals[next_skill_indices.reshape(-1),0,:]

		self.last_info = info.copy()
		self.last_done = done

		assert reward.max() <= 1
		assert reward.min() >= 0

		return new_observation, reward, done, info

	def write_config(self, output_file: str):
		pass

	def save(self, directory: str):
		pass

	def load(self, directory: str):
		pass

	def set_skills_sequence(self, sseq, env):
		"""
		sseq = [skill_1, skill_2, ...]
		skill_i = ((starting_observation, starting_full_state), skill_length, skill_goal)
		"""
		self.skills_sequence = sseq
		self.nb_skills = len(self.skills_sequence)
		self.curr_indx = np.zeros((env.num_envs,1)).astype(np.intc)

		self.skills_observations = np.zeros((self.nb_skills, env.num_envs, env.get_obs_dim()))
		self.skills_full_states = np.zeros((self.nb_skills, env.num_envs, env.get_full_state_dim()))
		self.skills_max_episode_steps = np.zeros((self.nb_skills, env.num_envs, 1))
		self.skills_goals = np.zeros((self.nb_skills, env.num_envs, env.get_goal_dim()))

		for i,skill in enumerate(self.skills_sequence):
			skill_starting_state, skill_length, skill_goal_state = skill
			observation, full_state = skill_starting_state

			self.skills_observations[i,:,:] = np.tile(np.array(observation).flatten(), (env.num_envs, 1))
			self.skills_full_states[i,:,:] = np.tile(np.array(full_state).flatten(), (env.num_envs, 1))
			self.skills_max_episode_steps[i,:,:] = np.tile(np.array(skill_length).flatten(), (env.num_envs, 1))
			self.skills_goals[i,:,:] = np.tile(env.project_to_goal_space(skill_goal_state).flatten(), (env.num_envs, 1))

		return 0

	def sample_skill_indx(self, n_envs):
		"""
		Sample a skill indx (weighted sampling of skill according to skill success rate or uniform sampling)
		"""
		# weights_available = True
		# for i in range(self.delta_step,len(self.L_skills_results)):
		# 	if len(self.L_skills_results[i]) == 0:
		# 		weights_available = False
		#
		# # print("self.L_skills_results = ", self.L_skills_results)
		# #print("weights available = ", weights_available)
		#
		# ## fitness based selection
		# if self.weighted_sampling and weights_available:
		#
		# 	L_rates = self.get_skills_success_rates()
		#
		# 	assert len(L_rates) == len(self.L_states) - self.delta_step
		#
		# 	## weighted sampling
		# 	total_rate = sum(L_rates)
		#
		#
		# 	L_new_skill_indx = []
		# 	for i in range(self.num_envs):
		# 		pick = random.uniform(0, total_rate)
		#
		# 		current = 0
		# 		for i in range(0,len(L_rates)):
		# 			s_r = L_rates[i]
		# 			current += s_r
		# 			if current > pick:
		# 				break
		#
		# 		i += self.delta_step
		# 		L_new_skill_indx.append([i])
		#
		# 	## TODO: switch for tensor version
		# 	new_skill_indx = torch.tensor(L_new_skill_indx)
		#
		# 	assert new_skill_indx.shape == (self.num_envs, 1)
		#
		# ## uniform sampling
		# else:
		new_skill_indx = np.random.randint(0, self.nb_skills, (n_envs,))

		return new_skill_indx.astype(np.intc)

	def shift_skill(self, env):
		is_done = self.last_done
		next_skill_indices, next_skill_avail = self.next_skill_indx()

		## if overshoot possible, choose next skill indx, otherwise, sample new skill indx
		self.curr_indx = np.where(next_skill_avail == 1, next_skill_indices, self.curr_indx)

		## recover skill
		reset_max_episode_steps = self.skills_max_episode_steps[self.curr_indx.reshape(-1),0,:]
		reset_goals = self.skills_goals[self.curr_indx.reshape(-1),0,:]

		## set skill
		do_reset_max_episode_steps = is_done.copy()
		env.set_max_episode_steps(reset_max_episode_steps, do_reset_max_episode_steps)
		do_reset_goal = is_done.copy()
		env.set_goal(reset_goals, do_reset_goal)

		return env.get_observation()

	def next_skill_indx(self):
		curr_indx = self.curr_indx.copy()
		next_indx = curr_indx + 1
		next_skill_avail = (next_indx < self.nb_skills).astype(np.intc)
		return next_indx, next_skill_avail

	def _select_skill_indx(self, is_success, n_envs):
		"""
		Select skills (starting state, budget and goal)
		"""
		sampled_skill_indices = self.sample_skill_indx(n_envs) ## return tensor of new indices
		next_skill_indices, next_skill_avail = self.next_skill_indx() ## return tensor of next skills indices

		overshoot_possible = np.logical_and(is_success, next_skill_avail).astype(np.intc) * int(self.do_overshoot)

		## if overshoot possible, choose next skill indx, otherwise, sample new skill indx
		selected_skill_indices = np.where(overshoot_possible == 1, next_skill_indices, sampled_skill_indices)

		return selected_skill_indices, overshoot_possible

	def reset(self, env, observation, eval_mode=False):
		## reset to first skill
		self.curr_indx = np.zeros((env.num_envs,)).astype(np.intc)

		## recover skill
		reset_observations = self.skills_observations[self.curr_indx.reshape(-1),0,:]
		reset_full_states = self.skills_full_states[self.curr_indx.reshape(-1),0,:]
		reset_max_episode_steps = self.skills_max_episode_steps[self.curr_indx.reshape(-1),0,:]
		reset_goals = self.skills_goals[self.curr_indx.reshape(-1),0,:]

		## set skill
		do_reset_state = np.ones((env.num_envs,))
		env.set_state(reset_full_states, do_reset_state)
		do_reset_max_episode_steps = np.ones((env.num_envs,))
		env.set_max_episode_steps(reset_max_episode_steps, do_reset_max_episode_steps)
		do_reset_goal = np.ones((env.num_envs,))
		env.set_goal(reset_goals, do_reset_goal)

		return env.get_observation()

	def reset_done(self, env, observation, eval_mode=False):

		is_done = self.last_done
		is_success = self.last_info["is_success"] ## array of booleans
		selected_skill_indices, overshoot_possible = self._select_skill_indx(is_success, env.num_envs)
		self.curr_indx = np.where(is_done==1, selected_skill_indices, self.curr_indx)

		## recover skill
		reset_observations = self.skills_observations[self.curr_indx.reshape(-1),0,:]
		reset_full_states = self.skills_full_states[self.curr_indx.reshape(-1),0,:]
		reset_max_episode_steps = self.skills_max_episode_steps[self.curr_indx.reshape(-1),0,:]
		reset_goals = self.skills_goals[self.curr_indx.reshape(-1),0,:]

		## set skill
		do_reset_state = np.logical_and(is_done, np.logical_not(overshoot_possible)).astype(np.intc)
		env.set_state(reset_full_states, do_reset_state)
		do_reset_max_episode_steps = is_done.copy()
		env.set_max_episode_steps(reset_max_episode_steps, do_reset_max_episode_steps)
		do_reset_goal = is_done.copy()
		env.set_goal(reset_goals, do_reset_goal)

		return env.get_observation()
