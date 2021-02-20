import numpy as np
import torch
from segment_tree import SumSegmentTree, MinSegmentTree
import random

class ReplayBuffer(object):
	def __init__(self, state_dim, action_dim, history_length, max_size=int(1e6)):
		self.max_size = max_size
		self.ptr = 0
		self.size = 0

		self.state = np.zeros((max_size, state_dim))
		self.action = np.zeros((max_size, action_dim))
		self.next_state = np.zeros((max_size, state_dim))
		self.reward = np.zeros((max_size, 1))
		self.not_done = np.zeros((max_size, 1))
		self.cp_obs = np.zeros((max_size, history_length, state_dim))
		self.cp_act = np.zeros((max_size, history_length, action_dim))

		self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


	def add(self, state, action, next_state, reward, done, cp_obs, cp_act):
		self.state[self.ptr] = state
		self.action[self.ptr] = action
		self.next_state[self.ptr] = next_state
		self.reward[self.ptr] = reward
		self.not_done[self.ptr] = 1. - done
		self.cp_obs[self.ptr] = cp_obs
		self.cp_act[self.ptr] = cp_act
		
		self.ptr = (self.ptr + 1) % self.max_size
		self.size = min(self.size + 1, self.max_size)


	def sample(self, batch_size):
		ind = np.random.randint(0, self.size, size=batch_size)

		return (
			torch.FloatTensor(self.state[ind]).to(self.device),
			torch.FloatTensor(self.action[ind]).to(self.device),
			torch.FloatTensor(self.next_state[ind]).to(self.device),
			torch.FloatTensor(self.reward[ind]).to(self.device),
			torch.FloatTensor(self.not_done[ind]).to(self.device),
			torch.FloatTensor(self.cp_obs[ind]).reshape(batch_size, -1).to(self.device),
			torch.FloatTensor(self.cp_act[ind]).reshape(batch_size, -1).to(self.device),
		)


class SILReplayBuffer(object):
	def __init__(self, state_dim, action_dim, gamma, max_size=int(1e6)):
		self.max_size = max_size
		self.gamma = gamma
		self.ptr = 0
		self.size = 0
		self.state = np.zeros((max_size, state_dim))
		self.action = np.zeros((max_size, action_dim))
		self.R = np.zeros((max_size, 1))
		self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
		self.running_episode = []

	def add(self, state, action, next_state, reward, done):
		self.running_episode.append([state, action, reward])
		if done:
			self.add_episode(self.running_episode)
			self.running_episode = []

	def add_episode(self, trajectory):
		states, actions, rewards, dones = [], [], [], []
		for (state, action, reward) in trajectory:
			states.append(state)
			actions.append(action)
			rewards.append(reward)
			dones.append(False)
		dones[len(dones)-1] = True
		returns = discount_with_dones(rewards, dones, self.gamma)
		for (state, action, R) in list(zip(states, actions, returns)):
			self.state[self.ptr] = state
			self.action[self.ptr] = action
			self.R[self.ptr] = R
			self.ptr = (self.ptr + 1) % self.max_size
			self.size = min(self.size + 1, self.max_size)

	def sample(self, batch_size):
		ind = np.random.randint(0, self.size, size=batch_size)
		return (
                        torch.FloatTensor(self.state[ind]).to(self.device),
                        torch.FloatTensor(self.action[ind]).to(self.device),
                        torch.FloatTensor(self.R[ind]).to(self.device),
                )


class SILPrioritizedReplayBuffer(object):
        def __init__(self, state_dim, action_dim, history_length, gamma, alpha, beta, max_size=int(1e6)):
                self.max_size = max_size
                self.gamma = gamma
                self.ptr = 0
                self.size = 0
                self.state = np.zeros((max_size, state_dim))
                self.action = np.zeros((max_size, action_dim))
                self.cp_obs = np.zeros((max_size, history_length, state_dim))
                self.cp_act = np.zeros((max_size, history_length, action_dim))
                self.R = np.zeros((max_size, 1))
               	self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
               	self.running_episode = []
                self.alpha = alpha
                self.beta = beta
                it_capacity = 1
                while it_capacity < max_size:
                        it_capacity *= 2

                self._it_sum = SumSegmentTree(it_capacity)
                self._it_min = MinSegmentTree(it_capacity)
                self._max_priority = 1.0

        def add(self, state, action, next_state, reward, done, cp_obs, cp_act):
                self.running_episode.append([state, action, reward, cp_obs, cp_act])
                if done:
                        self.add_episode(self.running_episode)
                        self.running_episode = []

        def add_episode(self, trajectory):
                states, actions, rewards, dones, cp_obses, cp_acts = [], [], [], [], [], []
                for (state, action, reward, cp_obs, cp_act) in trajectory:
                        states.append(state)
                        actions.append(action)
                        rewards.append(reward)
                        dones.append(False)
                        cp_obses.append(cp_obs)
                        cp_acts.append(cp_act)
                dones[len(dones)-1] = True
                returns = discount_with_dones(rewards, dones, self.gamma)
                for (state, action, R, cp_obs, cp_act) in list(zip(states, actions, returns, cp_obses, cp_acts)):
                        self.state[self.ptr] = state
                        self.action[self.ptr] = action
                        self.R[self.ptr] = R
                        self.cp_obs[self.ptr] = cp_obs
                        self.cp_act[self.ptr] = cp_act
                        self._it_sum[self.ptr] = self._max_priority ** self.alpha
                        self._it_min[self.ptr] = self._max_priority ** self.alpha
                        self.ptr = (self.ptr + 1) % self.max_size
                        self.size = min(self.size + 1, self.max_size)

        def _sample_proportional(self, batch_size):
                res = []
                for _ in range(batch_size):
                        mass = random.random() * self._it_sum.sum(0, self.size - 1)
                        idx = self._it_sum.find_prefixsum_idx(mass)
                        res.append(idx)
                return res


        def sample(self, batch_size):
                idxes = self._sample_proportional(batch_size)

                if self.beta > 0:
                        weights = []
                        p_min = self._it_min.min() / self._it_sum.sum()
                        max_weight = (p_min * self.size) ** (-self.beta)
                        for idx in idxes:
                                p_sample = self._it_sum[idx] / self._it_sum.sum()
                                weight = (p_sample * self.size) ** (-self.beta)
                                weights.append(weight / max_weight)
                        weights = np.array(weights)
                else:
                        weights = np.ones_like(idxes, dtype=np.float32)
                return (
                        torch.FloatTensor(self.state[idxes]).to(self.device),
                        torch.FloatTensor(self.action[idxes]).to(self.device),
                        torch.FloatTensor(self.R[idxes]).to(self.device),
                        torch.FloatTensor(self.cp_obs[idxes]).reshape(batch_size, -1).to(self.device),
                        torch.FloatTensor(self.cp_act[idxes]).reshape(batch_size, -1).to(self.device),
                        torch.FloatTensor(weights).to(self.device),
                        torch.FloatTensor(idxes).to(self.device),
                )

        def update_priorities(self, idxes, priorities):
                assert len(idxes) == len(priorities)
                for idx, priority in zip(idxes, priorities):
                        priority = max(priority, 1e-6)
                        assert priority > 0
                        assert 0 <= idx < self.size
                        idx = int(idx)
                        self._it_sum[idx] = priority ** self.alpha
                        self._it_min[idx] = priority ** self.alpha
                        self._max_priority = max(self._max_priority, priority) 

def discount_with_dones(rewards, dones, gamma):
	discounted = []
	r = 0
	for reward, done in zip(rewards[::-1], dones[::-1]):
		r = reward + gamma*r*(1.-done) # fixed off by one bug
		discounted.append(r)
	return discounted[::-1]

