# Copyright 2022 Nicolas Perrin-Gilbert.
#
# Licensed under the BSD 3-Clause License.

from abc import ABC, abstractmethod
import numpy as np
import torch
from jaxlib.xla_extension import DeviceArray
from typing import Union, Dict
from xpag.tools.utils import DataType

from xpag.samplers import Sampler

class EpisodicSampler_index(Sampler):
	def __init__(self, env, datatype: DataType = DataType.NUMPY):
		assert (
			datatype == DataType.TORCH_CPU
			or datatype == DataType.TORCH_CUDA
			or datatype == DataType.NUMPY
		), (
			"datatype must be DataType.TORCH_CPU, "
			"DataType.TORCH_CUDA or DataType.NUMPY."
		)
		self.datatype = datatype
		super().__init__()

		self.env = env
		if hasattr(self.env, "obs_rms"):
			self.do_normalize = True
		else:
			self.do_normalize = False

	@staticmethod
	def sum(transitions) -> float:
		return sum([transitions[key].sum() for key in transitions.keys()])

	def sample(
		self,
		buffers: Dict[str, Union[torch.Tensor, np.ndarray]],
		batch_size: int,
	) -> Dict[str, Union[torch.Tensor, np.ndarray]]:
		rollout_batch_size = buffers["episode_length"].shape[0]
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

		if self.do_normalize:
			transitions["observation"] = np.concatenate(
				[
					self.env._normalize_shape(transitions["observation.observation"], self.env.obs_rms["observation"]),
					transitions["skill_indx"],
				],
				axis=1,
			)

			s_transitions = transitions.copy()
			s_transitions["next_observation"] = np.concatenate(
				[
					self.env._normalize_shape(transitions["next_observation.observation"], self.env.obs_rms["observation"]),
					transitions["next_skill_indx"],
				],
				axis=1,
			)

			f_transitions = transitions.copy()
			f_transitions["next_observation"] = np.concatenate(
				[
					self.env._normalize_shape(transitions["next_observation.observation"], self.env.obs_rms["observation"]),
					transitions["skill_indx"],
				],
				axis=1,
			)

			transitions["next_observation"] = np.where(transitions["reward"]==1., s_transitions["next_observation"], f_transitions["next_observation"])


		else:
			transitions["observation"] = np.concatenate(
				[
					transitions["observation.observation"],
					transitions["skill_indx"],
				],
				axis=1,
			)

			s_transitions = transitions.copy()
			s_transitions["next_observation"] = np.concatenate(
				[
					transitions["next_observation.observation"],
					transitions["next_skill_indx"],
				],
				axis=1,
			)

			f_transitions = transitions.copy()
			f_transitions["next_observation"] = np.concatenate(
				[
					transitions["next_observation.observation"],
					transitions["skill_indx"],
				],
				axis=1,
			)

			transitions["next_observation"] = np.where(transitions["reward"]==1., s_transitions["next_observation"], f_transitions["next_observation"])

		transitions["true_done"] = np.zeros(transitions["reward"].shape) + np.logical_and(transitions["reward"],transitions["last_skill"])

		return transitions
