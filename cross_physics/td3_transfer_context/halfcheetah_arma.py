import gym
import numpy as np

class HalfCheetahARMAEnv(object):
    def __init__(self, armas=[0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5], reward_delay=1):
        self.armas = armas
        self.env = gym.make('HalfCheetah-v2')
        self.set_seed(0)
 
        self.arma = armas[0]
        self.observation_space = self.env.observation_space
        self.action_space = self.env.action_space
        self.reward_range = self.env.reward_range
        self.metadata = self.env.metadata
        self.spec = self.env.spec        
        self._max_episode_steps = self.env._max_episode_steps

        self.t = 0
        self.cum_reward = 0
        self.reward_delay = reward_delay

    def set_seed(self, seed):
        self.env.seed(seed)
        self.rng = np.random.RandomState(seed)
        self.seed = seed

    def reset(self):
        self.t = 0
        self.cum_reward = 0
        self.arma = self.rng.choice(self.armas)
        self.env.unwrapped.model.dof_armature[3:] = self.arma
        ob = self.env.reset()
        return ob

    def step(self, action):
        ob, r, done, info = self.env.step(action)
        self.cum_reward += r
        self.t += 1
        if self.t%self.reward_delay==0:
            r = self.cum_reward
            self.cum_reward = 0
        else:
            r = 0
        return ob, r, done, info

    def get_sim_parameters(self):
        return np.array([self.arma])
