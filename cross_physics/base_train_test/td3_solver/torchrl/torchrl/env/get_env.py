from .atari_wrapper import *
from .continuous_wrapper import *
from .base_wrapper import *
from .vecenv import VecEnv
from .subproc_vecenv import SubProcVecEnv


def wrap_deepmind(env, frame_stack=False, scale=False, clip_rewards=False):
    assert 'NoFrameskip' in env.spec.id
    env = EpisodicLifeEnv(env)
    env = NoopResetEnv(env, noop_max=30)
    env = MaxAndSkipEnv(env, skip=4)
    if 'FIRE' in env.unwrapped.get_action_meanings():
        env = FireResetEnv(env)
    env = WarpFrame(env)
    if scale:
        env = ScaledFloatFrame(env)
    if clip_rewards:
        env = ClipRewardEnv(env)
    if frame_stack:
        env = FrameStack(env, 4)
    return env


def wrap_continuous_env(env, obs_norm, reward_scale):
    env = RewardShift(env, reward_scale)
    if obs_norm:
        return NormObs(env)
    return env


def get_env(env_id, env_param):
    if env_id == 'Push':
        import metaworld
        import random
        random.seed(1)
        torch.manual_seed(1)
        np.random.seed(1)
        ml1 = metaworld.ML1('push-v1')
        env = ml1.train_classes['push-v1']()
        task = random.choice(ml1.train_tasks[0:50])
        env.set_task(task)
        env.reset()
        env.seed(1)
        env = BaseWrapper(env)
        return env
    else:
        env = gym.make(env_id)
        if str(env.__class__.__name__).find('TimeLimit') >= 0:
            env = TimeLimitAugment(env)
        env = BaseWrapper(env)
        if "rew_norm" in env_param:
            env = NormRet(env, **env_param["rew_norm"])
            del env_param["rew_norm"]

        ob_space = env.observation_space
        if len(ob_space.shape) == 3:
            env = wrap_deepmind(env, **env_param)
        else:
            env = wrap_continuous_env(env, **env_param)

        if isinstance(env.action_space, gym.spaces.Box):
            return NormAct(env)
        return env


def get_single_env(env_id, env_param):
    if env_id in ['Push', 'Door_open', 'Drawer_open', 'Pick_place', 'Coffee_push', 'Push_back']:
        import metaworld
        import random
        random.seed(1)
        torch.manual_seed(1)
        np.random.seed(1)
        if env_id == 'Push':
            ml1 = metaworld.ML1('push-v1')
            env = ml1.train_classes['push-v1']()
        elif env_id == 'Door_open':
            ml1 = metaworld.ML1('door-open-v1')
            env = ml1.train_classes['door-open-v1']()
        elif env_id == 'Drawer_open':
            ml1 = metaworld.ML1('drawer-open-v1')
            env = ml1.train_classes['drawer-open-v1']()
        elif env_id == 'Pick_place':
            ml1 = metaworld.ML1('pick-place-v1')
            env = ml1.train_classes['pick-place-v1']()
        elif env_id == 'Coffee_push':
            ml1 = metaworld.ML1('coffee-push-v1')
            env = ml1.train_classes['coffee-push-v1']()
        elif env_id == 'Push_back':
            ml1 = metaworld.ML1('push-back-v1')
            env = ml1.train_classes['push-back-v1']()
        task = random.choice(ml1.train_tasks[0:50])
        env.set_task(task)
        env.reset()
        env.seed(1)
        env = BaseWrapper(env)
        return env
    else:
        env = gym.make(env_id)
        if str(env.__class__.__name__).find('TimeLimit') >= 0:
            env = TimeLimitAugment(env)
        env = BaseWrapper(env)

        ob_space = env.observation_space
        if len(ob_space.shape) == 3:
            env = wrap_deepmind(env, **env_param)

        if "reward_scale" in env_param:
            env = RewardShift(env, env_param["reward_scale"])

        if isinstance(env.action_space, gym.spaces.Box):
            return NormAct(env)
        return env


def get_vec_env(env_id, env_param, vec_env_nums):
    vec_env = VecEnv(
        vec_env_nums, get_single_env,
        [env_id, env_param])

    if "obs_norm" in env_param and env_param["obs_norm"]:
        vec_env = NormObs(vec_env)
    return vec_env


def get_subprocvec_env(env_id, env_param, vec_env_nums, proc_nums):
    vec_env = SubProcVecEnv(
        proc_nums, vec_env_nums, get_single_env,
        [env_id, env_param])

    if "obs_norm" in env_param and env_param["obs_norm"]:
        vec_env = NormObs(vec_env)
    return vec_env
