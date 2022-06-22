# Copyright 2022 Nicolas Perrin-Gilbert.
#
# Licensed under the BSD 3-Clause License.

import numpy as np
import torch
from xpag.tools.utils import DataType
from xpag.samplers import HER


class HER_DCIL_variant(HER):
	def __init__(
		self,
		compute_reward,
		env,
		replay_strategy: str = "future",
		datatype: DataType = DataType.NUMPY,
	):
		super().__init__(compute_reward, replay_strategy, datatype)

		self.env = env
		if hasattr(self.env, "obs_rms"):
			self.do_normalize = True
		else:
			self.do_normalize = False

	def sample(self, buffers, batch_size_in_transitions):
		rollout_batch_size = buffers["episode_length"].shape[0]
		batch_size = batch_size_in_transitions
		# select rollouts and steps
		episode_idxs = np.random.choice(
			np.arange(rollout_batch_size),
			size=batch_size,
			replace=True,
			p=buffers["episode_length"][:, 0, 0]
			/ buffers["episode_length"][:, 0, 0].sum(),
		)
		t_max_episodes = buffers["episode_length"][episode_idxs, 0].flatten()

		if self.datatype == DataType.TORCH_CPU or self.datatype == DataType.TORCH_CUDA:
			t_samples = (torch.rand_like(t_max_episodes) * t_max_episodes).long()
		else:
			t_samples = np.random.randint(t_max_episodes)
		transitions = {
			key: buffers[key][episode_idxs, t_samples] for key in buffers.keys()
		}
		# HER indexes
		her_indexes = np.where(np.random.uniform(size=batch_size) < self.future_p)

		if self.datatype == DataType.TORCH_CPU or self.datatype == DataType.TORCH_CUDA:
			future_offset = (
				torch.rand_like(t_max_episodes) * (t_max_episodes - t_samples)
			).long()
		else:
			future_offset = np.random.uniform(size=batch_size) * (
				t_max_episodes - t_samples
			)
			future_offset = future_offset.astype(int)
		future_t = (t_samples + future_offset)[her_indexes]
		# replace desired goal with achieved goal
		future_ag = buffers["next_observation.achieved_goal"][
			episode_idxs[her_indexes], future_t
		]

		transitions["observation.desired_goal"][her_indexes] = future_ag

		# recomputing rewards
		if self.datatype == DataType.TORCH_CPU or self.datatype == DataType.TORCH_CUDA:
			transitions["reward"] = torch.unsqueeze(
				self.reward_func(
					transitions["next_observation.achieved_goal"],
					transitions["observation.desired_goal"],
					{},
				),
				-1,
			)
		else:
			transitions["reward"] = np.expand_dims(
				self.reward_func(
					transitions["next_observation.achieved_goal"],
					transitions["observation.desired_goal"],
					{},
				),
				1,
			)
		transitions = {
			k: transitions[k].reshape(batch_size, *transitions[k].shape[1:])
			for k in transitions.keys()
		}
		if self.datatype == DataType.TORCH_CPU or self.datatype == DataType.TORCH_CUDA:
			## TODO (Alex): adapt normalization to torch
			if self.do_normalize:
				transitions["observation"] = torch.hstack(
					(
						self.env._normalize(transitions["observation.observation"], self.env.obs_rms["observation"]) ,
						self.env._normalize(transitions["observation.desired_goal"], self.env.obs_rms["achieved_goal"]) ,
					)
				)
				transitions["next_observation"] = torch.hstack(
					(
						self.env._normalize(transitions["next_observation.observation"], self.env.obs_rms["observation"]) ,
						self.env._normalize(transitions["observation.desired_goal"], self.env.obs_rms["achieved_goal"]),
					)
				)
			else:
				transitions["observation"] = torch.hstack(
					(
						transitions["observation.observation"],
						transitions["observation.desired_goal"],
					)
				)
				transitions["next_observation"] = torch.hstack(
					(
						transitions["next_observation.observation"],
						transitions["observation.desired_goal"],
					)
				)

		else:
			# print("mean achieved_goal HER = ", self.env.obs_rms["achieved_goal"].mean)
			if self.do_normalize:
				transitions["observation"] = np.concatenate(
					[
						self.env._normalize(transitions["observation.observation"], self.env.obs_rms["observation"]) ,
						self.env._normalize(transitions["observation.desired_goal"], self.env.obs_rms["achieved_goal"]) ,
						self.env._normalize(transitions["next_skill_goal"], self.env.obs_rms["achieved_goal"]),
					],
					axis=1,
				)
				transitions["next_observation"] = np.concatenate(
					[
						self.env._normalize(transitions["next_observation.observation"], self.env.obs_rms["observation"]),
						self.env._normalize(transitions["observation.desired_goal"], self.env.obs_rms["achieved_goal"]),
						self.env._normalize(transitions["next_skill_goal"], self.env.obs_rms["achieved_goal"]),
					],
					axis=1,
				)

			else:
				transitions["observation"] = np.concatenate(
					[
						transitions["observation.observation"],
						transitions["observation.desired_goal"],
						transitions["next_skill_goal"],
					],
					axis=1,
				)

				s_transitions = transitions.copy()
				s_transitions["next_observation"] = np.concatenate(
					[
						transitions["next_observation.observation"],
						transitions["next_skill_goal"],
						transitions["next_next_skill_goal"],
					],
					axis=1,
				)

				f_transitions = transitions.copy()
				f_transitions["next_observation"] = np.concatenate(
					[
						transitions["next_observation.observation"],
						transitions["observation.desired_goal"],
						transitions["next_skill_goal"],
					],
					axis=1,
				)

				transitions["next_observation"] = np.where(transitions["reward"]==1., s_transitions["next_observation"], f_transitions["next_observation"])

		# transitions["true_done"] = np.zeros(transitions["reward"].shape) + np.logical_and(transitions["reward"],np.logical_not(transitions["next_skill_avail"]))
		transitions["true_done"] = np.logical_or(transitions["done_from_env"], np.logical_and(transitions["reward"],np.logical_not(transitions["next_skill_avail"])))

		# print("true_done = ", transitions["true_done"][:10])
		# print("reward = ", transitions["reward"][:10])

		return transitions
