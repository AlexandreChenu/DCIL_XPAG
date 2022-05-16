# Copyright 2022 Nicolas Perrin-Gilbert.
#
# Licensed under the BSD 3-Clause License.
#
# This file is an implementation of the SAC (Soft-Actor Critic) algorithm.
# It is partially derived from the implementation of SAC in brax
# [https://github.com/google/brax/blob/main/brax/training/sac.py]
# which contains the following copyright notice:
#
# Copyright 2022 The Brax Authors.
# Licensed under the Apache License, Version 2.0.

from abc import ABC
from typing import Any, Tuple, Callable, NewType
from functools import partial
import dataclasses
from dataclasses import dataclass
import haiku as hk
import torch
import flax
import math

# from flax import linen
import jax
import jax.numpy as jnp
import optax
from xpag.agents.agent import Agent

# import os
# import joblib
from acme.jax.networks import FeedForwardNetwork
import acme.jax.networks as networks

Params = Any
PRNGKey = jnp.ndarray

Observation = NewType("Observation", jnp.ndarray)
Action = NewType("Action", jnp.ndarray)
Reward = NewType("Reward", jnp.ndarray)
Done = NewType("Done", jnp.ndarray)
StateDescriptor = NewType("StateDescriptor", jnp.ndarray)
Skill = NewType("Skill", jnp.ndarray)
TrainingState = NewType("TrainingState", Any)


@dataclasses.dataclass
class FeedForwardModel:
    init: Any
    apply: Any


@flax.struct.dataclass
class TrainingState:
    """Contains training state for the learner."""

    policy_optimizer_state: optax.OptState
    policy_params: Params
    critic_optimizer_state: optax.OptState
    critic_params: Params
    alpha_optimizer_state: Params
    alpha_params: Params
    target_critic_params: Params
    key: PRNGKey
    steps: jnp.ndarray


def make_sac_networks(
    action_size: int,
    obs_size: int,
    hidden_layer_sizes: Tuple[int, ...] = (256, 256),
) -> Tuple:
    """Creates networks used by the SAC agent."""

    def _actor_fn(obs):
        network = hk.Sequential(
            [
                hk.nets.MLP(
                    list(hidden_layer_sizes),
                    w_init=hk.initializers.VarianceScaling(1.0, "fan_in", "uniform"),
                    activation=jax.nn.relu,
                    activate_final=True,
                ),
                networks.NormalTanhDistribution(action_size),
            ]
        )
        return network(obs)

    def _critic_fn(obs, action):
        network1 = hk.Sequential(
            [
                hk.nets.MLP(
                    list(hidden_layer_sizes) + [1],
                    w_init=hk.initializers.VarianceScaling(1.0, "fan_in", "uniform"),
                    activation=jax.nn.relu,
                ),
            ]
        )
        network2 = hk.Sequential(
            [
                hk.nets.MLP(
                    list(hidden_layer_sizes) + [1],
                    w_init=hk.initializers.VarianceScaling(1.0, "fan_in", "uniform"),
                    activation=jax.nn.relu,
                ),
            ]
        )
        input_ = jnp.concatenate([obs, action], axis=-1)
        value1 = network1(input_)
        value2 = network2(input_)
        return jnp.concatenate([value1, value2], axis=-1)

    policy = hk.without_apply_rng(hk.transform(_actor_fn))
    critic = hk.without_apply_rng(hk.transform(_critic_fn))
    dummy_obs = jnp.zeros((1, obs_size))
    dummy_action = jnp.zeros((1, action_size))

    policy_network = networks.FeedForwardNetwork(
        lambda key: policy.init(key, dummy_obs), policy.apply
    )
    critic_network = networks.FeedForwardNetwork(
        lambda key: critic.init(key, dummy_obs, dummy_action), critic.apply
    )

    return policy_network, critic_network


def make_sac_loss_fn(
    policy: FeedForwardNetwork,
    critic: FeedForwardNetwork,
    reward_scaling: float,
    discount: float,
    action_size: int,
):
    """Creates the loss functions for SAC"""

    @jax.jit
    def _policy_loss_fn(
        policy_params: Params,
        critic_params: Params,
        alpha: jnp.ndarray,
        observations,
        actions,
        rewards,
        new_observations,
        done,
        key: PRNGKey,
    ) -> jnp.ndarray:

        action_distribution = policy.apply(policy_params, observations)
        action = action_distribution.sample(seed=key)
        log_prob = action_distribution.log_prob(action)
        q_action = critic.apply(critic_params, observations, action)
        min_q = jnp.min(q_action, axis=-1)
        actor_loss = alpha * log_prob - min_q

        return jnp.mean(actor_loss)

    @jax.jit
    def _critic_loss_fn(
        critic_params: Params,
        policy_params: Params,
        target_critic_params: Params,
        alpha: jnp.ndarray,
        observations,
        actions,
        rewards,
        new_observations,
        done,
        key: PRNGKey,
    ) -> jnp.ndarray:

        q_old_action = critic.apply(critic_params, observations, actions)
        next_action_distribution = policy.apply(policy_params, new_observations)
        next_action = next_action_distribution.sample(seed=key)
        next_log_prob = next_action_distribution.log_prob(next_action)
        next_q = critic.apply(target_critic_params, new_observations, next_action)

        next_v = jnp.min(next_q, axis=-1) - alpha * next_log_prob

        target_q = jax.lax.stop_gradient(
            rewards * reward_scaling + (1.0 - done) * discount * next_v
        )

        q_error = q_old_action - jnp.expand_dims(target_q, -1)
        q_loss = 0.5 * jnp.mean(jnp.square(q_error))

        return q_loss

    target_entropy = -0.5 * action_size

    @jax.jit
    def _alpha_loss_fn(
        log_alpha: jnp.ndarray,
        policy_params: Params,
        observations,
        actions,
        rewards,
        new_observations,
        done,
        key: PRNGKey,
    ) -> jnp.ndarray:
        """Eq 18 from https://arxiv.org/pdf/1812.05905.pdf."""

        action_distribution = policy.apply(policy_params, observations)
        action = action_distribution.sample(seed=key)
        log_prob = action_distribution.log_prob(action)
        alpha = jnp.exp(log_alpha)
        alpha_loss = alpha * jax.lax.stop_gradient(-log_prob - target_entropy)

        loss = jnp.mean(alpha_loss)
        return loss

    return _alpha_loss_fn, _policy_loss_fn, _critic_loss_fn


@dataclass
class SacConfig:
    """Configuration for the SAC algorithm"""

    learning_rate: float = 3e-4
    alpha_init: float = 1.0
    discount: float = 0.99
    reward_scaling: float = 1.0
    tau: float = 0.005
    _alg_name: str = "SAC2"


@dataclass
class SacNetworksApply:
    """Networks forward functions for the SAC algorithm"""

    policy_fn: Callable[[Params, Observation], jnp.ndarray]
    critic_fn: Callable[[Params, Observation, Action], jnp.ndarray]


@dataclass
class SacLosses:
    """Losses for the SAC algorithm"""

    policy_loss_fn: Callable[
        [
            Params,
            Params,
            jnp.ndarray,
            jnp.ndarray,
            jnp.ndarray,
            jnp.ndarray,
            jnp.ndarray,
            jnp.ndarray,
            PRNGKey,
        ],
        jnp.ndarray,
    ]
    critic_loss_fn: Callable[
        [
            Params,
            Params,
            Params,
            jnp.ndarray,
            jnp.ndarray,
            jnp.ndarray,
            jnp.ndarray,
            jnp.ndarray,
            jnp.ndarray,
            PRNGKey,
        ],
        jnp.ndarray,
    ]
    alpha_loss_fn: Callable[
        [
            jnp.ndarray,
            Params,
            jnp.ndarray,
            jnp.ndarray,
            jnp.ndarray,
            jnp.ndarray,
            jnp.ndarray,
            PRNGKey,
        ],
        jnp.ndarray,
    ]


@dataclass
class SacOptimizers:
    """Optimizers for the SAC algorithm"""

    policy_optimizer: optax.GradientTransformation
    critic_optimizer: optax.GradientTransformation
    alpha_optimizer: optax.GradientTransformation


class SACparl:
    """
    A collection of functions that define the algorithm Soft Actor Critic
    (SAC), ref: https://arxiv.org/abs/1801.01290
    """

    def __init__(
        self,
        config: SacConfig,
        networks_apply: SacNetworksApply,
        losses: SacLosses,
        optimizers: SacOptimizers,
    ):

        self._config = config
        self._networks_apply = networks_apply
        self._losses = losses
        self._optimizers = optimizers

    @partial(jax.jit, static_argnames=("self", "deterministic"))
    def _select_action_fn(
        self,
        obs: Observation,
        policy_params: Params,
        random_key: PRNGKey,
        deterministic: bool = False,
    ) -> (Action, PRNGKey):
        """Selects an action acording to SAC policy."""
        action_distribution = self._networks_apply.policy_fn(policy_params, obs)
        if not deterministic:
            key, key_sample = jax.random.split(random_key)
            actions = action_distribution.sample(seed=key_sample)

        else:
            actions = action_distribution.mode()

        return actions, random_key

    @partial(jax.jit, static_argnames="self")
    def update_fn(
        self,
        training_state: TrainingState,
        observations,
        actions,
        rewards,
        new_observations,
        done,
    ) -> (TrainingState, dict):
        """Performs a training step, updates the training parameters and training
        state. Returns the updated training step
        """

        key = training_state.key

        # update alpha
        key, subkey = jax.random.split(key)
        alpha_loss, alpha_gradient = jax.value_and_grad(self._losses.alpha_loss_fn)(
            training_state.alpha_params,
            training_state.policy_params,
            observations,
            actions,
            rewards,
            new_observations,
            done,
            key=subkey,
        )

        alpha_updates, alpha_optimizer_state = self._optimizers.alpha_optimizer.update(
            alpha_gradient, training_state.alpha_optimizer_state
        )
        alpha_params = optax.apply_updates(training_state.alpha_params, alpha_updates)

        alpha = jnp.exp(training_state.alpha_params)

        # update critic
        key, subkey = jax.random.split(key)
        critic_loss, critic_gradient = jax.value_and_grad(self._losses.critic_loss_fn)(
            training_state.critic_params,
            training_state.policy_params,
            training_state.target_critic_params,
            alpha,
            observations,
            actions,
            rewards,
            new_observations,
            done,
            key=subkey,
        )

        (
            critic_updates,
            critic_optimizer_state,
        ) = self._optimizers.critic_optimizer.update(
            critic_gradient, training_state.critic_optimizer_state
        )
        critic_params = optax.apply_updates(
            training_state.critic_params, critic_updates
        )
        target_critic_params = jax.tree_multimap(
            lambda x1, x2: (1.0 - self._config.tau) * x1 + self._config.tau * x2,
            training_state.target_critic_params,
            critic_params,
        )

        # update actor
        key, subkey = jax.random.split(key)

        policy_loss, policy_gradient = jax.value_and_grad(self._losses.policy_loss_fn)(
            training_state.policy_params,
            training_state.critic_params,
            alpha,
            observations,
            actions,
            rewards,
            new_observations,
            done,
            key=subkey,
        )
        (
            policy_updates,
            policy_optimizer_state,
        ) = self._optimizers.policy_optimizer.update(
            policy_gradient, training_state.policy_optimizer_state
        )
        policy_params = optax.apply_updates(
            training_state.policy_params, policy_updates
        )

        # create new training state
        new_training_state = TrainingState(
            policy_optimizer_state=policy_optimizer_state,
            policy_params=policy_params,
            critic_optimizer_state=critic_optimizer_state,
            critic_params=critic_params,
            alpha_optimizer_state=alpha_optimizer_state,
            alpha_params=alpha_params,
            target_critic_params=target_critic_params,
            key=key,
            steps=training_state.steps + 1,
        )
        metrics = {
            "actor_loss": policy_loss,
            "critic_loss": critic_loss,
            "alpha_loss": alpha_loss,
        }
        return new_training_state, metrics


class SAC2(Agent, ABC):
    def __init__(self, observation_dim, action_dim, params=None):
        """
        Jax implementation of SAC (https://arxiv.org/abs/1812.05905).
        """

        if "backend" in params:
            self.backend = params["backend"]
        else:
            self.backend = "cpu"

        self._config_string = str(list(locals().items())[1:])
        super().__init__("SAC", observation_dim, action_dim, params)

        if "seed" in self.params:
            start_seed = self.params["seed"]
        else:
            start_seed = 42

        self.key, local_key, key_models = jax.random.split(
            jax.random.PRNGKey(start_seed), 3
        )

        alg_config = SacConfig()
        # from IPython import embed
        # embed()

        # Initialize networks and optimizers

        policy, critic = make_sac_networks(
            action_size=action_dim,
            obs_size=observation_dim,
        )

        sac_networks_apply = SacNetworksApply(
            policy_fn=policy.apply,
            critic_fn=critic.apply,
        )

        self.key, subkey = jax.random.split(self.key)
        policy_params = policy.init(subkey)

        self.key, subkey = jax.random.split(self.key)
        critic_params = critic.init(subkey)

        target_critic_params = jax.tree_map(
            lambda x: jnp.asarray(x.copy()), critic_params
        )

        policy_optimizer = optax.adam(learning_rate=alg_config.learning_rate)
        policy_optimizer_state = policy_optimizer.init(policy_params)
        critic_optimizer = optax.adam(learning_rate=alg_config.learning_rate)
        critic_optimizer_state = critic_optimizer.init(critic_params)

        log_alpha = jnp.asarray(math.log(alg_config.alpha_init), dtype=jnp.float32)
        alpha_optimizer = optax.adam(learning_rate=alg_config.learning_rate)
        alpha_optimizer_state = alpha_optimizer.init(log_alpha)

        sac_optimizers = SacOptimizers(
            policy_optimizer=policy_optimizer,
            critic_optimizer=critic_optimizer,
            alpha_optimizer=alpha_optimizer,
        )

        self.training_state = TrainingState(
            policy_optimizer_state=policy_optimizer_state,
            policy_params=policy_params,
            critic_optimizer_state=critic_optimizer_state,
            critic_params=critic_params,
            alpha_optimizer_state=alpha_optimizer_state,
            alpha_params=log_alpha,
            target_critic_params=target_critic_params,
            key=local_key,
            steps=jnp.array(0),
        )

        # Initialize losses

        (alpha_loss_fn, policy_loss_fn, critic_loss_fn,) = make_sac_loss_fn(
            policy=policy,
            critic=critic,
            reward_scaling=alg_config.reward_scaling,
            discount=alg_config.discount,
            action_size=action_dim,
        )

        sac_losses = SacLosses(
            policy_loss_fn=policy_loss_fn,
            critic_loss_fn=critic_loss_fn,
            alpha_loss_fn=alpha_loss_fn,
        )

        # Initialize algorithm

        self.sac = SACparl(
            config=alg_config,
            losses=sac_losses,
            optimizers=sac_optimizers,
            networks_apply=sac_networks_apply,
        )

    def select_action(self, observation, deterministic=True):
        action, self.key = self.sac._select_action_fn(
            observation, self.training_state.policy_params, self.key
        )
        return action

    def train(self, pre_sample, sampler, batch_size):
        batch = sampler.sample(pre_sample, batch_size)
        return self.train_on_batch(batch)

    def train_on_batch(self, batch):
        if torch.is_tensor(batch["r"]):
            version = "torch"
        else:
            version = "numpy"
        if version == "numpy":
            observations = jnp.array(batch["obs"])
            actions = jnp.array(batch["actions"])
            rewards = jnp.array(batch["r"])
            new_observations = jnp.array(batch["obs_next"])
            # SAC seems unstable when dealing with terminal transitions
            done = jnp.array(batch["terminals"])
        else:
            observations = jnp.array(batch["obs"].detach().cpu().numpy())
            actions = jnp.array(batch["actions"].detach().cpu().numpy())
            rewards = jnp.array(batch["r"].detach().cpu().numpy())
            new_observations = jnp.array(batch["obs_next"].detach().cpu().numpy())
            # SAC seems unstable when dealing with terminal transitions
            done = jnp.array(batch["terminals"].detach().cpu().numpy())

        self.training_state, metrics = self.sac.update_fn(
            self.training_state, observations, actions, rewards, new_observations, done
        )

        # from IPython import embed
        # embed()

        return metrics

    def save(self, directory):
        pass

    def load(self, directory):
        pass

    def write_config(self, output_file: str):
        pass
