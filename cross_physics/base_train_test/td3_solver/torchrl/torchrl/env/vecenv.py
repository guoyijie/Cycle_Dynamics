import numpy as np
from .base_wrapper import BaseWrapper
from toolz.dicttoolz import merge_with


class VecEnv(BaseWrapper):
    """
    Vector Env
        Each env should have
        1. same observation space shape
        2. same action space shape
    """
    def __init__(self, env_nums, env_funcs, env_args):
        self.env_nums = env_nums
        self.env_funcs = env_funcs
        self.env_args = env_args
        if isinstance(env_funcs, list):
            assert len(env_funcs) == env_nums
            assert len(env_args) == env_args
        else:
            self.env_funcs = [env_funcs for _ in range(env_nums)]
            self.env_args = [env_args for _ in range(env_nums)]

        self.set_up_envs()

    def set_up_envs(self):
        self.envs = [env_func(*env_arg) for env_func, env_arg
                     in zip(self.env_funcs, self.env_args)]

    def train(self):
        for env in self.envs:
            env.train()

    def eval(self):
        for env in self.envs:
            env.eval()

    def reset(self, **kwargs):
        import metaworld
        import random
        for each_env in self.envs:
            # ml1 = metaworld.ML1('coffee-push-v1')
            # ml1 = metaworld.ML1('push-back-v1')
            ml1 = metaworld.ML1('push-v1')
            # ml1 = metaworld.ML1('door-open-v1')
            # ml1 = metaworld.ML1('drawer-open-v1')
            # ml1 = metaworld.ML1('pick-place-v1')
            task = random.choice(ml1.train_tasks[0:50])
            each_env.set_task(task)
        obs = [env.reset() for env in self.envs]
        self._obs = np.stack(obs)
        return self._obs

    def partial_reset(self, index_mask, **kwargs):
        indexs = np.argwhere(index_mask == 1).reshape((-1))
        reset_obs = [self.envs[index].reset() for index in indexs]
        self._obs[index_mask] = reset_obs
        return self._obs

    def step(self, actions):
        actions = np.split(actions, self.env_nums)
        result = [env.step(np.squeeze(action)) for env, action in
                  zip(self.envs, actions)]
        obs, rews, dones, infos = zip(*result)
        self._obs = np.stack(obs)
        infos = merge_with(np.array, *infos)
        return self._obs, np.stack(rews)[:, np.newaxis], \
            np.stack(dones)[:, np.newaxis], infos

    def seed(self, seed):
        for env in self.envs:
            env.seed(seed)

    @property
    def observation_space(self):
        return self.envs[0].observation_space

    @property
    def action_space(self):
        return self.envs[0].action_space
