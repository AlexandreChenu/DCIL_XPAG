from abc import ABC
import torch.nn.functional as F
import numpy as np
import os
import torch
from torch import nn as nn
from torch.distributions import Distribution, Normal
from xpag.agents.agent import Agent

# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

LOG_SIG_MAX = 2
LOG_SIG_MIN = -20


def fanin_init(tensor):
    size = tensor.size()
    if len(size) == 2:
        fan_in = size[0]
    elif len(size) > 2:
        fan_in = np.prod(size[1:])
    else:
        raise Exception("Shape must be have dimension at least 2.")
    bound = 1.0 / np.sqrt(fan_in)
    return tensor.data.uniform_(-bound, bound)


def initialize_hidden_layer(layer, b_init_value=0.1):
    fanin_init(layer.weight)
    layer.bias.data.fill_(b_init_value)


def initialize_last_layer(layer, init_w=1e-3):
    layer.weight.data.uniform_(-init_w, init_w)
    layer.bias.data.uniform_(-init_w, init_w)


class TanhNormal(Distribution):
    """
    Represent distribution of X where
        X ~ tanh(Z)
        Z ~ N(mean, std)
    Note: this is not very numerically stable.
    """

    def __init__(self, normal_mean, normal_std, device, epsilon=1e-6):
        """
        :param normal_mean: Mean of the normal distribution
        :param normal_std: Std of the normal distribution
        :param epsilon: Numerical stability epsilon when computing log-prob.
        """
        self.normal_mean = normal_mean
        self.normal_std = normal_std
        self.normal = Normal(normal_mean, normal_std)
        self.epsilon = epsilon
        self.device = device

    def sample_n(self, n, return_pre_tanh_value=False):
        z = self.normal.sample_n(n)
        if return_pre_tanh_value:
            return torch.tanh(z), z
        else:
            return torch.tanh(z)

    def log_prob(self, value, pre_tanh_value=None):
        """
        :param value: some value, x
        :param pre_tanh_value: arctanh(x)
        :return:
        """
        if pre_tanh_value is None:
            pre_tanh_value = torch.log((1 + value) / (1 - value)) / 2
        return self.normal.log_prob(pre_tanh_value) - torch.log(
            1 - value * value + self.epsilon
        )

    def sample(self, return_pretanh_value=False):
        """
        Gradients will and should *not* pass through this operation.
        See https://github.com/pytorch/pytorch/issues/4620 for discussion.
        """
        z = self.normal.sample().detach()

        if return_pretanh_value:
            return torch.tanh(z), z
        else:
            return torch.tanh(z)

    def rsample(self, return_pretanh_value=False):
        """
        Sampling in the reparameterization case.
        """
        z = (
            self.normal_mean
            + self.normal_std
            * Normal(
                torch.zeros(self.normal_mean.size()).to(self.device),
                torch.ones(self.normal_std.size()).to(self.device),
            ).sample()
        )
        z.requires_grad_()

        if return_pretanh_value:
            return torch.tanh(z), z
        else:
            return torch.tanh(z)


class Actor(nn.Module):
    def __init__(self, observation_dim, action_dim, max_action, device):
        super().__init__()

        self.device = device
        # hidden layers definition
        self.l1 = nn.Linear(observation_dim, 256)
        self.l2 = nn.Linear(256, 256)
        self.l3_mean = nn.Linear(256, action_dim)

        # std layer definition
        self.l3_log_std = nn.Linear(256, action_dim)

        # weights initialization
        initialize_hidden_layer(self.l1)
        initialize_hidden_layer(self.l2)
        initialize_last_layer(self.l3_mean)
        initialize_last_layer(self.l3_log_std)

        # print("max_action = ", max_action)
        # print("device = ", device)
        self.max_action = torch.tensor(max_action, device=self.device)

    def forward(self, x, deterministic=False, return_log_prob=False):
        # forward pass
        x = F.relu(self.l1(x))
        x = F.relu(self.l2(x))
        mean, log_std = self.l3_mean(x), self.l3_log_std(x)

        # compute std
        log_std = torch.clamp(log_std, LOG_SIG_MIN, LOG_SIG_MAX)
        std = torch.exp(log_std)

        # compute other relevant quantities
        log_prob, entropy, mean_action_log_prob, pre_tanh_value = None, None, None, None
        if deterministic:
            action = torch.tanh(mean)
        else:
            tanh_normal = TanhNormal(mean, std, device=self.device)
            if return_log_prob:
                action, pre_tanh_value = tanh_normal.rsample(return_pretanh_value=True)
                log_prob = tanh_normal.log_prob(action, pre_tanh_value=pre_tanh_value)
                log_prob = log_prob.sum(dim=1, keepdim=True)
            else:
                action = tanh_normal.rsample()

        action = action * self.max_action
        return (
            action,
            mean,
            log_std,
            log_prob,
            entropy,
            std,
            mean_action_log_prob,
            pre_tanh_value,
        )


class Critic(nn.Module):
    def __init__(self, observation_dim, action_dim, max_action, device):
        super().__init__()

        self.device = device
        # Q1 architecture
        self.l1 = nn.Linear(observation_dim + action_dim, 256)
        self.l2 = nn.Linear(256, 256)
        self.l3 = nn.Linear(256, 1)

        # Q2 architecture
        self.l4 = nn.Linear(observation_dim + action_dim, 256)
        self.l5 = nn.Linear(256, 256)
        self.l6 = nn.Linear(256, 1)

        # weights initialization
        initialize_hidden_layer(self.l1)
        initialize_hidden_layer(self.l2)
        initialize_last_layer(self.l3)

        initialize_hidden_layer(self.l4)
        initialize_hidden_layer(self.l5)
        initialize_last_layer(self.l6)

        self.max_action = torch.tensor(max_action, device=self.device)

    def forward(self, x, u):
        xu = torch.cat([x, u / self.max_action], 1)

        x1 = F.relu(self.l1(xu))
        x1 = F.relu(self.l2(x1))
        x1 = self.l3(x1)

        x2 = F.relu(self.l4(xu))
        x2 = F.relu(self.l5(x2))
        x2 = self.l6(x2)
        return x1, x2

    def Q1(self, x, u):
        xu = torch.cat([x, u / self.max_action], 1)

        x1 = F.relu(self.l1(xu))
        x1 = F.relu(self.l2(x1))
        x1 = self.l3(x1)
        return x1


class LogAlpha(nn.Module):
    def __init__(self):
        super().__init__()
        self.value = torch.nn.Parameter(torch.zeros(1, requires_grad=True))


class SACTORCH(Agent, ABC):
    def __init__(
        self,
        observation_dim,
        action_dim,
        max_action=1.0,
        params={},
        discount=0.99,
        reward_scale=1.0,
        policy_lr=1e-3,
        critic_lr=1e-3,
        alpha_lr=3e-4,
        soft_target_tau=0.005,
        target_update_period=1,
        use_automatic_entropy_tuning=False,
        target_entropy=None,
    ):
        self._config_string = str(list(locals().items())[1:])
        self.params = params
        if "device" not in self.params:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = self.params["device"]
        self.actor = Actor(observation_dim, action_dim, max_action, self.device).to(
            self.device
        )
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=policy_lr)

        self.critic = Critic(observation_dim, action_dim, max_action, self.device).to(
            self.device
        )
        self.critic_target = Critic(
            observation_dim, action_dim, max_action, self.device
        ).to(self.device)
        self.critic_target.load_state_dict(self.critic.state_dict())
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=critic_lr)

        super(SACTORCH, self).__init__("SACTORCH", observation_dim, action_dim, params)

        self.soft_target_tau = soft_target_tau
        self.target_update_period = target_update_period

        self.action_dim = action_dim

        self.use_automatic_entropy_tuning = use_automatic_entropy_tuning
        if self.use_automatic_entropy_tuning:
            if target_entropy:
                self.target_entropy = target_entropy
            else:
                self.target_entropy = -np.prod(
                    self.action_dim
                ).item()  # heuristic value
            self.log_alpha = LogAlpha().to(self.device)
            self.alpha_optimizer = torch.optim.Adam(
                self.log_alpha.parameters(), lr=alpha_lr
            )

        self.discount = discount
        self.reward_scale = reward_scale
        self._n_train_steps_total = 0

    @staticmethod
    def soft_update(source, target, tau):
        for param, target_param in zip(source.parameters(), target.parameters()):
            target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)

    def write_config(self, output_file):
        print(self._config_string, file=output_file)

    def save_nets(self, nets, directory):
        sdir = directory + "/" + self.__class__.__name__
        for i, net in enumerate(nets):
            filename = "net_" + str(i + 1)
            os.makedirs(sdir, exist_ok=True)
            torch.save(net.state_dict(), "%s/%s.pth" % (sdir, filename))

    def load_nets(self, nets, directory):
        sdir = directory + "/" + self.__class__.__name__
        for i, net in enumerate(nets):
            filename = "net_" + str(i + 1)
            net.load_state_dict(
                torch.load("%s/%s.pth" % (sdir, filename), map_location=self.device)
            )

    def save(self, directory):
        self.save_nets(
            [
                self.actor,
                self.actor_optimizer,
                self.critic,
                self.critic_target,
                self.critic_optimizer,
                self.log_alpha,
                self.alpha_optimizer,
            ],
            directory,
        )

    def load(self, directory):
        self.load_nets(
            [
                self.actor,
                self.actor_optimizer,
                self.critic,
                self.critic_target,
                self.critic_optimizer,
                self.log_alpha,
                self.alpha_optimizer,
            ],
            directory,
        )

    def select_action(self, observation, deterministic=True):
        observation = torch.FloatTensor(observation.reshape(1, -1)).to(self.device)
        return (
            self.actor(observation, deterministic=deterministic)[0]
            .cpu()
            .data.numpy()
            #.flatten()
        )

    def value(self, observation, action):
        observation = torch.FloatTensor(observation.reshape(1, -1)).to(self.device)
        action = torch.FloatTensor(action.reshape(1, -1)).to(self.device)
        qval = self.critic(observation, action)[0].cpu().data.numpy().flatten()
        return qval

    def train(self, buffer, sampler, batch_size):
        batch = sampler.sample(buffer, batch_size)
        self.train_on_batch(batch)

    def train_on_batch(self, batch):
        # increment iteration counter
        self._n_train_steps_total += 1

        rewards = torch.FloatTensor(batch["reward"]).view(-1, 1).to(self.device)
        # done = torch.FloatTensor(1.0 - batch["terminals"]).view(-1, 1).to(self.device)
        done = torch.FloatTensor(1.0 - batch["reward"]).view(-1, 1).to(self.device)
        observations = torch.FloatTensor(batch["observation"]).to(self.device)
        actions = torch.FloatTensor(batch["action"]).to(self.device)
        new_observations = torch.FloatTensor(batch["next_observation"]).to(self.device)

        # replay_data.observations
        # replay_data.next_observations
        # replay_data.dones
        # replay_data.actions
        # replay_data.rewards

        # compute actor and alpha losses
        new_obs_actions, policy_mean, policy_log_std, log_pi, *_ = self.actor(
            observations, return_log_prob=True
        )
        if self.use_automatic_entropy_tuning:
            alpha_loss = -(
                self.log_alpha.value * (log_pi + self.target_entropy).detach()
            ).mean()
            self.alpha_optimizer.zero_grad()
            alpha_loss.backward(retain_graph=True)
            self.alpha_optimizer.step()
            alpha = self.log_alpha.value.exp()
        else:
            alpha = 0.001

        Q1_new_actions, Q2_new_actions = self.critic_target(
            observations, new_obs_actions
        )
        Q_new_actions = torch.min(Q1_new_actions, Q2_new_actions)
        actor_loss = (alpha * log_pi - Q_new_actions).mean()

        # compute critic losses
        current_Q1, current_Q2 = self.critic(observations, actions)

        # Make sure policy accounts for squashing functions like tanh correctly!
        new_next_actions, _, _, new_log_pi, *_ = self.actor(
            new_observations, return_log_prob=True
        )
        target_Q1, target_Q2 = self.critic_target(new_observations, new_next_actions)
        target_Q_values = torch.min(target_Q1, target_Q2) - alpha * new_log_pi

        target_Q = self.reward_scale * rewards + (
            done * self.discount * target_Q_values
        )
        critic_loss = F.mse_loss(current_Q1, target_Q.detach()) + F.mse_loss(
            current_Q2, target_Q.detach()
        )

        # optimization
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        # soft updates
        if self._n_train_steps_total % self.target_update_period == 0:
            self.soft_update(self.critic, self.critic_target, self.soft_target_tau)

        metrics = {
                "actor_loss": actor_loss,
                "critic_loss": critic_loss,
        }

        return metrics
