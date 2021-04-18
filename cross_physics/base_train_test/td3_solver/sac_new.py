import copy
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.utils.tensorboard import SummaryWriter
writer = SummaryWriter()

from torch.distributions import Normal

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Implementation of Twin Delayed Deep Deterministic Policy Gradients (TD3)
# Paper: https://arxiv.org/abs/1802.09477

class Actor(nn.Module):
	def __init__(self, state_dim, action_dim, max_action):
		super(Actor, self).__init__()

		self.l1 = nn.Linear(state_dim, 256)
		self.l2 = nn.Linear(256, 256)
		self.l3 = nn.Linear(256, action_dim)
		
		self.max_action = max_action
		

	def forward(self, state):
		a = F.relu(self.l1(state))
		a = F.relu(self.l2(a))
		return self.max_action * torch.tanh(self.l3(a))

class Critic(nn.Module):
	def __init__(self, state_dim, action_dim):
		super(Critic, self).__init__()

		# Q1 architecture
		self.l1 = nn.Linear(state_dim + action_dim, 256)
		self.l2 = nn.Linear(256, 256)
		self.l3 = nn.Linear(256, 1)

		# Q2 architecture
		self.l4 = nn.Linear(state_dim + action_dim, 256)
		self.l5 = nn.Linear(256, 256)
		self.l6 = nn.Linear(256, 1)


	def forward(self, state, action):
		sa = torch.cat([state, action], 1)

		q1 = F.relu(self.l1(sa))
		q1 = F.relu(self.l2(q1))
		q1 = self.l3(q1)

		q2 = F.relu(self.l4(sa))
		q2 = F.relu(self.l5(q2))
		q2 = self.l6(q2)
		return q1, q2


	def Q1(self, state, action):
		sa = torch.cat([state, action], 1)

		q1 = F.relu(self.l1(sa))
		q1 = F.relu(self.l2(q1))
		q1 = self.l3(q1)
		return q1

class SAC(object):
	def __init__(
		self,
		state_dim,
		action_dim,
		max_action,
		discount=0.99,
		tau=0.005,
		):

		plr=3e-4
		qlr=3e-4
		policy_std_reg_weight=1e-3
		policy_mean_reg_weight=1e-3

		self.pf = Actor(state_dim, action_dim, max_action).to(device)
		self.pf_optimizer = torch.optim.Adam(self.pf.parameters(), lr=plr)

		self.stdf = Actor(state_dim, action_dim, max_action).to(device)
		self.stdf_optimizer = torch.optim.Adam(self.stdf.parameters(), lr=plr)

		self.critic = Critic(state_dim, action_dim).to(device)
		self.critic_target = copy.deepcopy(self.critic)
		self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=qlr)

		self.max_action = max_action
		self.tau = tau
		self.discount = discount

		self.total_it = 0

		self.target_entropy = -np.prod(action_dim).item()
		self.log_alpha = torch.zeros(1).to(device)
		self.log_alpha.requires_grad_()
		self.alpha_optimzier = torch.optim.Adam([self.log_alpha], lr = plr)

		self.critic_criterion = nn.MSELoss()
		self.policy_std_reg_weight = policy_std_reg_weight
		self.policy_mean_reg_weight = policy_mean_reg_weight

		self.reparameterization = True

	def select_action(self, state):
		state = torch.FloatTensor(state.reshape(1, -1)).to(device)
		return self.pf(state).cpu().data.numpy().flatten()

	def train(self, replay_buffer, batch_size=100):
		self.total_it += 1
		state, action, next_state, reward, not_done = replay_buffer.sample(batch_size)

		# Policy operations
		mean = self.pf.forward(state)
		log_std = self.stdf.forward(state)
		std = torch.exp(log_std)
		curr_distribution = Normal(mean, std)
		new_actions = curr_distribution.sample()
		log_probs = curr_distribution.log_prob(new_actions)
		log_probs = log_probs.sum(dim = -1, keepdim = True)

		q1_pred, q2_pred = self.critic(state, action)

		alpha = 1
		alpha_loss = 0

		with torch.no_grad():
			target_mean = self.pf.forward(next_state)
			target_log_std = self.stdf(next_state)
			target_std = torch.exp(target_log_std)
			target_distribution = Normal(target_mean, target_std)
			target_actions = target_distribution.sample()
			target_log_probs = target_distribution.log_prob(target_actions)
			target_log_probs = target_log_probs.sum(dim = -1, keepdim=True)

			target_q1_pred, target_q2_pred = self.critic_target(next_state, target_actions)
			min_target_q = torch.min(target_q1_pred, target_q2_pred)
			target_v_values = min_target_q - alpha * target_log_probs


		# QF Loss
		q_target = reward + not_done * self.discount * target_v_values
		assert q1_pred.shape == q_target.shape
		assert q2_pred.shape == q_target.shape
		qf1_loss = self.critic_criterion(q1_pred, q_target.detach())
		qf2_loss = self.critic_criterion(q2_pred, q_target.detach())
		critic_loss = qf1_loss + qf2_loss

		critic_1, critic_2 = self.critic(state, new_actions)

		q_new_actions = torch.min(critic_1, critic_2)


		# Policy Loss
		assert log_probs.shape == q_new_actions.shape
		policy_loss = (alpha * log_probs - q_new_actions).mean()

		std_reg_loss = self.policy_std_reg_weight * (log_std**2).mean()
		mean_reg_loss = self.policy_mean_reg_weight * (mean**2).mean()

		policy_loss += std_reg_loss + mean_reg_loss


		# Update Networks
		self.pf_optimizer.zero_grad()
		policy_loss.backward()
		self.pf_optimizer.step()

		self.critic_optimizer.zero_grad()
		critic_loss.backward()
		self.critic_optimizer.step()

		for net, target_net in self.target_networks:
			for target_param, param in zip(target_net.parameters(), net.parameters()):
				target_param.data.copy_(target_param.data * (1.0 - self.tau) + param.data * self.tau)


		# Information For Logger
		info = {}
		info['Reward_Mean'] = reward.mean().item()
		info['Training/policy_loss'] = policy_loss.item()
		info['Training/qf1_loss'] = qf1_loss.item()
		info['Training/qf2_loss'] = qf2_loss.item()

		info['log_std/mean'] = log_std.mean().item()
		info['log_std/std'] = log_std.std().item()
		info['log_std/max'] = log_std.max().item()
		info['log_std/min'] = log_std.min().item()

		info['log_probs/mean'] = log_probs.mean().item()
		info['log_probs/std'] = log_probs.std().item()
		info['log_probs/max'] = log_probs.max().item()
		info['log_probs/min'] = log_probs.min().item()

		info['mean/mean'] = mean.mean().item()
		info['mean/std'] = mean.std().item()
		info['mean/max'] = mean.max().item()
		info['mean/min'] = mean.min().item()

		return info

	def save(self, filename):
		torch.save(self.critic.state_dict(), filename + "_critic")
		torch.save(self.critic_optimizer.state_dict(), filename + "_critic_optimizer")
		
		torch.save(self.pf.state_dict(), filename + "_actor")
		torch.save(self.pf_optimizer.state_dict(), filename + "_actor_optimizer")


	def load(self, filename):
		self.critic.load_state_dict(torch.load(filename + "_critic"))
		self.critic_optimizer.load_state_dict(torch.load(filename + "_critic_optimizer"))
		self.critic_target = copy.deepcopy(self.critic)

		self.pf.load_state_dict(torch.load(filename + "_actor"))
		self.pf_optimizer.load_state_dict(torch.load(filename + "_actor_optimizer"))

	@property
	def networks(self):
		return [self.pf, self.critic, self.critic_target]

	@property
	def snapshot_networks(self):
		return [
			["pf", self.pf],
			["critic", self.critic],
		]

	@property
	def target_networks(self):
		return [(self.critic, self.critic_target)]


class TD3(object):
	def __init__(
		self,
		state_dim,
		action_dim,
		max_action,
		discount=0.99,
		tau=0.005,
		policy_noise=0.2,
		noise_clip=0.5,
		policy_freq=2
	):

		self.actor = Actor(state_dim, action_dim, max_action).to(device)
		self.actor_target = copy.deepcopy(self.actor)
		self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=3e-4)

		self.critic = Critic(state_dim, action_dim).to(device)
		self.critic_target = copy.deepcopy(self.critic)
		self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=3e-4)

		self.max_action = max_action
		self.discount = discount
		self.tau = tau
		self.policy_noise = policy_noise
		self.noise_clip = noise_clip
		self.policy_freq = policy_freq

		self.total_it = 0


	def select_action(self, state):
		state = torch.FloatTensor(state.reshape(1, -1)).to(device)
		return self.actor(state).cpu().data.numpy().flatten()


	def train(self, replay_buffer, batch_size=100):
		self.total_it += 1

		# Sample replay buffer 
		state, action, next_state, reward, not_done = replay_buffer.sample(batch_size)

		with torch.no_grad():
			# Select action according to policy and add clipped noise
			noise = (
				torch.randn_like(action) * self.policy_noise
			).clamp(-self.noise_clip, self.noise_clip)
			
			next_action = (
				self.actor_target(next_state) + noise
			).clamp(-self.max_action, self.max_action)

			# Compute the target Q value
			target_Q1, target_Q2 = self.critic_target(next_state, next_action)
			target_Q = torch.min(target_Q1, target_Q2)
			target_Q = reward + not_done * self.discount * target_Q

		# Get current Q estimates
		current_Q1, current_Q2 = self.critic(state, action)

		# Compute critic loss
		critic_loss = F.mse_loss(current_Q1, target_Q) + F.mse_loss(current_Q2, target_Q)
		writer.add_scalar("critic_loss/train", critic_loss, self.total_it)

		# Optimize the critic
		self.critic_optimizer.zero_grad()
		critic_loss.backward()
		self.critic_optimizer.step()

		# Delayed policy updates
		if self.total_it % self.policy_freq == 0:

			# Compute actor losse
			actor_loss = -self.critic.Q1(state, self.actor(state)).mean()
			writer.add_scalar("actor_loss/train", actor_loss, self.total_it)
			
			# Optimize the actor 
			self.actor_optimizer.zero_grad()
			actor_loss.backward()
			self.actor_optimizer.step()

			# Update the frozen target models
			for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
				target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

			for param, target_param in zip(self.actor.parameters(), self.actor_target.parameters()):
				target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

		writer.flush()

	def save(self, filename):
		torch.save(self.critic.state_dict(), filename + "_critic")
		torch.save(self.critic_optimizer.state_dict(), filename + "_critic_optimizer")
		
		torch.save(self.actor.state_dict(), filename + "_actor")
		torch.save(self.actor_optimizer.state_dict(), filename + "_actor_optimizer")


	def load(self, filename):
		self.critic.load_state_dict(torch.load(filename + "_critic"))
		self.critic_optimizer.load_state_dict(torch.load(filename + "_critic_optimizer"))
		self.critic_target = copy.deepcopy(self.critic)

		self.actor.load_state_dict(torch.load(filename + "_actor"))
		self.actor_optimizer.load_state_dict(torch.load(filename + "_actor_optimizer"))
		self.actor_target = copy.deepcopy(self.actor)
		