import copy
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import *

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Implementation of Twin Delayed Deep Deterministic Policy Gradients (TD3)
# Paper: https://arxiv.org/abs/1802.09477


class Actor(nn.Module):
	def __init__(self, state_dim, action_dim, max_action):
		super(Actor, self).__init__()

		self.l1 = nn.Linear(state_dim, 256)
		self.l2 = nn.Linear(256, 256)
		self.l3 = nn.Linear(256, action_dim)
		#self.logstd = nn.Linear(256, action_dim)

		self.max_action = max_action
		

	def forward(self, state):
		a = F.relu(self.l1(state))
		a = F.relu(self.l2(a))
		return self.max_action * torch.tanh(self.l3(a))

	"""
	def _clip_but_pass_gradient(self, x, lower=0., upper=1.):
        	clip_up = (x > upper).float()
        	clip_low = (x < lower).float()
        	with torch.no_grad():
            		clip = ((upper - x) * clip_up + (lower - x) * clip_low)
        	return x + clip

	def log_prob(self, state, action, epsilon=1e-6):
		action = action/self.max_action
		a = F.relu(self.l1(state))
		a = F.relu(self.l2(a))
		mean = self.l3(a)
		logstd = self.logstd(a)
		logstd = torch.clamp(logstd, min=-11, max=2)
		normal_dist = Normal(mean, logstd.exp())
		pre_tanh_action = torch.log((1 + epsilon + action) / (1 + epsilon - action)) / 2
		norm_lp = normal_dist.log_prob(pre_tanh_action)
		ret = (norm_lp - torch.sum(
			torch.log(self._clip_but_pass_gradient((1. - action.square())) + epsilon), axis=-1, keepdim=True))	
		ret = torch.sum(ret, axis=-1, keepdim=True)
		return ret
        """

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


class TD3(object):
	def __init__(
		self,
		state_dim,
		action_dim,
		context_dim,
		max_action,
		discount=0.99,
		tau=0.005,
		policy_noise=0.2,
		noise_clip=0.5,
		policy_freq=2
	):

		self.actor = Actor(state_dim+context_dim, action_dim, max_action).to(device)
		self.actor_target = copy.deepcopy(self.actor)
		self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=3e-4)

		self.critic = Critic(state_dim+context_dim, action_dim).to(device)
		self.critic_target = copy.deepcopy(self.critic)
		self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=3e-4)

		self.max_action = max_action
		self.discount = discount
		self.tau = tau
		self.policy_noise = policy_noise
		self.noise_clip = noise_clip
		self.policy_freq = policy_freq

		self.total_it = 0
		self.sil_total_it = 0

	def select_action(self, state, context):
		state = torch.FloatTensor(state.reshape(1, -1)).to(device)
		state = torch.cat((state, context), dim=1)
		return self.actor(state).cpu().data.numpy().flatten()


	def train(self, replay_buffer, cmodel, batch_size=100):
		self.total_it += 1
			
		# Sample replay buffer 
		state, action, next_state, reward, not_done, cp_obs, cp_act = replay_buffer.sample(batch_size)
		context_tensor = cmodel(cp_obs, cp_act).detach()
		state = torch.cat((state, context_tensor), dim=1)
		next_state = torch.cat((next_state, context_tensor), dim=1)
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

		# Optimize the critic
		self.critic_optimizer.zero_grad()
		critic_loss.backward()
		self.critic_optimizer.step()

		# Delayed policy updates
		if self.total_it % self.policy_freq == 0:

			# Compute actor losse
			actor_loss = -self.critic.Q1(state, self.actor(state)).mean()
			
			# Optimize the actor 
			self.actor_optimizer.zero_grad()
			actor_loss.backward()
			self.actor_optimizer.step()
		
		if self.total_it%100 == 0:
			print('critic loss', critic_loss.cpu().data.numpy(), 'actor loss', actor_loss.cpu().data.numpy())
	
	def sil_train(self, sil_replay_buffer, cmodel, batch_size=100, sil_update=1, sil_value=0.01, sil_clip=1):
		if sil_update==0:
			return
		self.sil_total_it += 1
		# train actor
		#import ipdb; ipdb.set_trace()
		for _ in range(sil_update):
			state, action, R, cp_obs, cp_act, weight, idx = sil_replay_buffer.sample(batch_size)
			context_tensor = cmodel(cp_obs, cp_act).detach()
			state = torch.cat((state, context_tensor), dim=1)
			value1, value2 = self.critic_target(state, self.actor_target(state))
			value = torch.min(value1, value2)
			adv = torch.clamp(R-value, min=0, max=sil_clip)
			actor_diff = (self.actor(state)-action).square().mean()
			#nlogp = - self.actor.log_prob(state, action)
			#clipped_nlogp = (torch.clamp(nlogp, max=5) - nlogp).detach() + nlogp
			#actor_loss = clipped_nlogp*adv.detach()
			actor_loss = (self.actor(state)-action).square()
			#loss_fn = torch.nn.L1Loss(reduce=False)
			#actor_loss = loss_fn(self.actor(state), action)
			actor_loss = actor_loss.mean(1)*weight*torch.clamp(adv.detach().squeeze()*1e8, min=1e-3)
			actor_loss = actor_loss.mean()
			self.actor_optimizer.zero_grad()
			actor_loss.backward()
			self.actor_optimizer.step()

			# train critic
			value1, value2 = self.critic(state, self.actor_target(state).detach())
			#critic_loss = (torch.clamp(R-value1, min=0, max=sil_clip)).square()+(torch.clamp(R-value2, min=0, max=sil_clip)).square()
			critic_loss = -value1 * (torch.clamp(R-value1, min=0, max=sil_clip).detach()) - value2 * (torch.clamp(R-value2, min=0, max=sil_clip).detach()) 
			critic_loss = critic_loss.mean(1)
			critic_loss = sil_value * (critic_loss*weight).mean()
		
			self.critic_optimizer.zero_grad()
			critic_loss.backward()
			self.critic_optimizer.step()

			self.update_parameters()
			value1, value2 = self.critic_target(state, self.actor_target(state))
			value = torch.min(value1, value2)
			adv = torch.clamp(R-value, min=0, max=sil_clip)
			sil_replay_buffer.update_priorities(idx.squeeze().cpu().detach().numpy(), adv.squeeze().cpu().detach().numpy())

		if self.sil_total_it%100 == 0:
			print('sil actor diff', actor_diff.cpu().data.numpy(), 'sil actor loss', actor_loss.cpu().data.numpy(), 'sil critic loss', critic_loss.cpu().data.numpy())
	
	"""
	def sil_train(self, sil_replay_buffer, batch_size=100, sil_update=1, sil_value=0.01, sil_clip=1):
		self.sil_total_it += 1
		state, action, R, weight, idx = sil_replay_buffer.sample(batch_size)
		for _ in range(sil_update):
			value1, value2 = self.critic_target(state, self.actor_target(state))
			value = torch.min(value1, value2)
			adv = torch.clamp(R-value, min=0, max=sil_clip)
			actor_diff = (self.actor(state)-action).square().mean()
			
			value1, value2 = self.critic(state, action)
			critic_loss = -value1 * (torch.clamp(R-value1, min=0, max=sil_clip).detach()) - value2 * (torch.clamp(R-value2, min=0, max=sil_clip).detach())
			critic_loss = critic_loss.mean(1)
			critic_loss = sil_value * (critic_loss*weight).mean()

			self.critic_optimizer.zero_grad()
			critic_loss.backward()
			self.critic_optimizer.step()
		if self.sil_total_it%100 == 0:
			print('sil actor diff', actor_diff.cpu().data.numpy(), 'sil critic loss', critic_loss.cpu().data.numpy())
		sil_replay_buffer.update_priorities(idx.squeeze().cpu().detach().numpy(), adv.squeeze().cpu().detach().numpy())
	"""	

	def sil_train_updateA(self, sil_replay_buffer, batch_size=100, sil_update=1, sil_value=0.01, sil_clip=1):
		self.sil_total_it += 1
		for _ in range(sil_update):
			state, action, R, weight, idx = sil_replay_buffer.sample(batch_size)
			value1, value2 = self.critic_target(state, self.actor_target(state))
			value = torch.min(value1, value2)
			adv = torch.clamp(R-value, min=0, max=sil_clip)
			actor_diff = (self.actor(state)-action).square().mean()
			actor_loss = (self.actor(state)-action).square()
			actor_loss = actor_loss.mean(1)*weight*torch.clamp(adv.detach().squeeze(), min=1e-10)
			actor_loss = actor_loss.mean()
			self.actor_optimizer.zero_grad()
			actor_loss.backward()
			self.actor_optimizer.step()
			sil_replay_buffer.update_priorities(idx.squeeze().cpu().detach().numpy(), adv.squeeze().cpu().detach().numpy())
			for param, target_param in zip(self.actor.parameters(), self.actor_target.parameters()):
                        	target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)                            

		if self.sil_total_it%100 == 0:
			print('sil actor diff', actor_diff.cpu().data.numpy(), 'sil actor loss', actor_loss.cpu().data.numpy())
			
	def update_parameters(self, tau=None):
		if tau is None:
			tau=self.tau
		# Update the frozen target models
		for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
			target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)

		for param, target_param in zip(self.actor.parameters(), self.actor_target.parameters()):
			target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)	

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
		
