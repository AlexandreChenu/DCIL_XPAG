# Copyright 2022 Nicolas Perrin-Gilbert.
#
# Licensed under the BSD 3-Clause License.

import sys
import inspect
import numpy as np
import gym
from gym.vector.utils import (
	write_to_shared_memory,
	concatenate,
	create_empty_array
)
from gym.vector import VectorEnv, AsyncVectorEnv

from wrappers.reset_done_sync_vector_env import SyncVectorEnv

from xpag.wrappers.reset_done import ResetDoneWrapper
from xpag.tools.utils import get_env_dimensions

from gym import envs

from collections import OrderedDict
from typing import Tuple, Union, Optional
import copy

import torch




def check_goalenv(env) -> bool:
	"""
	Checks if an environment is of type 'GoalEnv'.
	The migration of GoalEnv from gym (0.22) to gym-robotics makes this verification
	non-trivial. Here we just verify that the observation_space has a structure
	that is compatible with the GoalEnv class.
	"""
	if isinstance(env, VectorEnv):
		obs_space = env.single_observation_space
	else:
		obs_space = env.observation_space
	if not isinstance(obs_space, gym.spaces.Dict):
		return False
	else:
		for key in ["observation", "achieved_goal", "desired_goal"]:
			if key not in obs_space.spaces:
				return False
	return True


def gym_vec_env_(env_name, num_envs):
	# env = NormalizationWrapper(gym.make(env_name))
	dummy_env = gym.make(env_name)
	# We force the env to have either a standard gym time limit (with the max number
	# of steps defined in .spec.max_episode_steps), or that the max number of steps
	# is stored in .max_episode_steps (and in this case we assume that the
	# environment appropriately prevents episodes from exceeding max_episode_steps
	# steps).
	assert (
		hasattr(dummy_env.spec, "max_episode_steps")
		and dummy_env.spec.max_episode_steps is not None
	) or (
		hasattr(dummy_env, "max_episode_steps")
		and dummy_env.max_episode_steps is not None
	), "Only allowing gym envs with time limit (spec.max_episode_steps)."

	env = NormalizationWrapper(DCILVecWrapper(
			SyncVectorEnv(
				[
					(lambda: gym.make(env_name))
					if hasattr(dummy_env, "reset_done")
					else (lambda: ResetDoneWrapper(gym.make(env_name)))
				]
				* num_envs,
			)
		))

	env.device = "cpu"
	env._spec = dummy_env.spec

	## Note (Alex): dummy_env.spec.max_episode_steps may not exist with the new version of assert
	if hasattr(dummy_env, "max_episode_steps"):
		max_episode_steps = dummy_env.max_episode_steps
	else:
		max_episode_steps = dummy_env.spec.max_episode_steps

	env_type = "Gym"

	is_goalenv = check_goalenv(env)
	env_info = {
		"env_type": env_type,
		"name": env_name,
		"is_goalenv": is_goalenv,
		"num_envs": num_envs,
		"max_episode_steps": max_episode_steps,
		"action_space": env.action_space,
		"single_action_space": env.single_action_space,
	}
	get_env_dimensions(env_info, is_goalenv, env)
	return env, env_info


def gym_vec_env(env_name, num_envs):
	env, env_info = gym_vec_env_(env_name, num_envs)
	eval_env, _ = gym_vec_env_(env_name, 1)
	return env, eval_env, env_info


class ResetDoneVecWrapper(gym.Wrapper):
	def __init__(self, env: VectorEnv):
		super().__init__(env)

	def reset(self, **kwargs):
		return self.env.reset(**kwargs)

	## Note (Alex): pass result through concatenate function to obtain a OrderedDict as output
	def reset_done(self, **kwargs):
		results = self.env.call("reset_done", **kwargs)
		observations = create_empty_array(
			self.env.single_observation_space, n=self.num_envs, fn=np.empty
		)
		return concatenate(self.env.single_observation_space, results, observations)

	# def reset_done(self, **kwargs):
	#     return self.env.reset_done(**kwargs)

	def step(self, action):
		obs, reward, done, info = self.env.step(action)
		# info = {
		#     "info_tuple": info_,
		#     "truncation": np.array(
		#         [
		#             [elt["TimeLimit.truncated"] if "TimeLimit.truncated" in elt else 0]
		#             for elt in info_
		#         ]
		#     ).reshape((self.env.num_envs, -1)),
		# }

		return (
			obs,
			reward,
			done,
			info,
		)


class DCILVecWrapper(gym.Wrapper):
	def __init__(self, env: VectorEnv):
		super().__init__(env)

	def reset(self, **kwargs):
		return self.env.reset(**kwargs)

	## Note (Alex): pass result through concatenate function to obtain a OrderedDict as output
	def reset_done(self, **kwargs):
		results = self.env.call("reset_done", **kwargs)
		observations = create_empty_array(
			self.env.single_observation_space, n=self.num_envs, fn=np.empty
		)
		return concatenate(self.env.single_observation_space, results, observations)

	## TODO: switch to asynchronous
	def set_state(self,reset_sim_states, do_reset_states):
		for i, (env, reset_sim_state, do_reset_state) in enumerate(zip(self.envs, reset_sim_states, do_reset_states.flatten())):
			env.set_state(reset_sim_state, do_reset_state)

	def reset_before_reset_done(self, do_reset_states):
		for i, (env, do_reset_state) in enumerate(zip(self.envs, do_reset_states.flatten())):
			env.reset_before_reset_done(do_reset_state)

	## set simulator state from bullet state file
	def set_inner_state(self, inner_states):
		# print("self.envs = ", self.envs)
		# print("self.envs[0].env = ", self.envs[0].env)
		for i, (env, inner_state) in enumerate(zip(self.envs, inner_states)):
			# print("inner_state = ", inner_state)
			env.env.set_inner_state(inner_state, from_file=True)

	## recover simulator file and save it memory (not hard disk as with bullet state files)
	def get_inner_state(self):
		inner_states = []
		for i, env in enumerate(self.envs):
			inner_state = env.env.get_inner_state()
			inner_states.append(copy.deepcopy(inner_state))
		return inner_states

	## TODO: switch to asynchronous
	def set_goal(self,reset_goals, do_reset_goals):

		for i, (env, reset_goal, do_reset_goal) in enumerate(zip(self.envs, reset_goals, do_reset_goals.flatten())):
			env.set_goal(reset_goal, do_reset_goal)

	## TODO: switch to asynchronous
	def set_max_episode_steps(self,reset_max_episode_steps, do_reset_max_episode_steps):

		for i, (env, reset_max_episode_step, do_reset_max_episode_step) in enumerate(zip(self.envs, reset_max_episode_steps, do_reset_max_episode_steps.flatten())):
			env.set_max_episode_steps(reset_max_episode_step, do_reset_max_episode_step)

	def get_max_episode_steps(self, **kwargs):
		results = self.env.call("get_max_episode_steps", **kwargs)

		return np.concatenate(results)

	def get_observation(self, **kwargs):

		results = self.env.call("get_observation", **kwargs)
		observations = create_empty_array(
			self.env.single_observation_space, n=self.num_envs, fn=np.empty
		)
		return concatenate(self.env.single_observation_space, results, observations)

	def state_vector(self, **kwargs):
		results = self.env.call("state_vector", **kwargs)

		return np.concatenate(results)

	def goal_distance(self, goal1, goal2, **kwargs):
		results = self.env.call("goal_distance", goal1, goal2, **kwargs)
		return np.array(results)

	def project_to_goal_space(self, state, **kwargs):
		results = self.env.call("project_to_goal_space", state, **kwargs)
		return np.concatenate(results)

	def get_obs_dim(self, **kwargs):
		return self.env.call("get_obs_dim", **kwargs)[0]

	def get_goal_dim(self, **kwargs):
		return self.env.call("get_goal_dim", **kwargs)[0]


	# def reset_done(self, **kwargs):
	#     return self.env.reset_done(**kwargs)

	def step(self, action):
		obs, reward, done, info = self.env.step(action)

		# print("observation vec_env = ", obs["observation"][0][:15])

		# info = {
		#     "info_tuple": info_,
		#     "truncation": np.array(
		#         [
		#             [elt["TimeLimit.truncated"] if "TimeLimit.truncated" in elt else 0]
		#             for elt in info_
		#         ]
		#     ).reshape((self.env.num_envs, -1)),
		# }

		dict_info = {}

		## concatenate info keys
		for key in info[0].keys():
			if isinstance(info[0][key], np.ndarray):
				dict_info[key] = np.concatenate([_info[key] for _info in info]).reshape(self.env.num_envs, info[0][key].shape[0])
			elif torch.is_tensor(info[0][key]):
				dict_info[key] = torch.cat([_info[key] for _info in info]).reshape(self.env.num_envs, info[0][key].shape[0])
			else:
				dict_info[key] = [_info[key] for _info in info]

		return (
			obs,
			reward.reshape(self.env.num_envs,1),
			done.reshape(self.env.num_envs,1),
			dict_info,
		)


class RunningMeanStd:
	def __init__(self, epsilon: float = 1e-4, shape: Tuple[int, ...] = ()):
		"""
		Calulates the running mean and std of a data stream
		https://en.wikipedia.org/wiki/Algorithms_for_calculating_variance#Parallel_algorithm
		:param epsilon: helps with arithmetic issues
		:param shape: the shape of the data stream's output
		"""
		self.mean = np.zeros(shape, np.float64)
		self.var = np.ones(shape, np.float64)
		self.count = epsilon

	def update(self, arr: np.ndarray) -> None:
		batch_mean = np.mean(arr, axis=0)
		batch_var = np.var(arr, axis=0)
		batch_count = arr.shape[0]
		self.update_from_moments(batch_mean, batch_var, batch_count)

	def update_from_moments(self, batch_mean: np.ndarray, batch_var: np.ndarray, batch_count: Union[int, float]) -> None:
		delta = batch_mean - self.mean
		tot_count = self.count + batch_count

		new_mean = self.mean + delta * batch_count / tot_count
		m_a = self.var * self.count
		m_b = batch_var * batch_count
		m_2 = m_a + m_b + np.square(delta) * self.count * batch_count / (self.count + batch_count)
		new_var = m_2 / (self.count + batch_count)

		new_count = batch_count + self.count

		self.mean = new_mean
		self.var = new_var
		self.count = new_count


class NormalizationWrapper(gym.Wrapper):
	def __init__(self, env):
		super().__init__(env)
		## init observation running mean and std
		observation = self.reset()
		self.obs_rms = {key: RunningMeanStd(shape=observation[key].shape) for key in observation.keys()}
		for key in self.obs_rms.keys():
			self.obs_rms[key].update(observation[key])
		self.epsilon = 1e-8
		self.min_obs = -500.
		self.max_obs = 500.

		self.do_normalize = True
		self.do_update = True

	def step(self, action):
		obs, reward, done, info = self._step(action)

		## update RMS
		if self.do_update:
			for key in self.obs_rms.keys():
				self.obs_rms[key].update(obs[key])
		# print("mean achieved goal = ", self.obs_rms["achieved_goal"].mean)

		return (
			obs,
			reward,
			done,
			info,
		)

	def _step(self, action):
		obs, reward, done, info = self.env.step(action)

		return (
			obs,
			reward,
			done,
			info,
		)

	def _normalize(self, obs, rms):
		return np.clip((obs - rms.mean) / np.sqrt(rms.var + self.epsilon), self.min_obs, self.max_obs)

	def _unnormalize(self, norm_obs, rms):
		return norm_obs * np.sqrt(rms.var + self.epsilon) + rms.mean

	def _normalize_shape(self, obs, rms, logs=False):
		if logs:
			print("obs = ", obs)
			print("tile = ", np.tile(rms.mean[0],(obs.shape[0],1)))
		return np.clip((obs - np.tile(rms.mean[0],(obs.shape[0],1))) / np.sqrt(np.tile(rms.var[0],(obs.shape[0],1)) + self.epsilon), self.min_obs, self.max_obs)


	def normalize(self, obs):
		norm_obs = copy.deepcopy(obs)
		for key in obs.keys():
			norm_obs[key] = self._normalize(norm_obs[key], self.obs_rms[key])
		return norm_obs

	def unormalize(self, norm_obs):
		obs = copy.deepcopy(norm_obs)
		for key in obs.keys():
			obs[key] = self._unnormalize(obs[key], self.obs_rms[key])
		return norm_obs
