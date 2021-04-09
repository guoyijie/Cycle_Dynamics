import numpy as np
import torch
import gym
import argparse
import os

import utils
import TD3
import sac_new

import metaworld
import random

# import torchvision
from torch.utils.tensorboard import SummaryWriter
writer = SummaryWriter()
# from torchvision import datasets, transforms

def safe_path(path):
	if not os.path.exists(path):
		os.mkdir(path)
	return path

# Runs policy for X episodes and returns average reward
# A fixed seed is used for the eval environment
def eval_policy(policy, env_name, seed, t, eval_episodes=10):
	ml1 = metaworld.ML1('push-v1')
	eval_env = ml1.train_classes['push-v1']()
	task = random.choice(ml1.train_tasks[0:50])
	eval_env.set_task(task) 
	obs = eval_env.reset()

	eval_env.seed(seed + 100)

	avg_reward = 0.
	success_eval_cnt = 0
	for _ in range(eval_episodes):
		state, done = eval_env.reset(), False
		curr_episode = 0
		success_bool = False
		while not done:
			curr_episode += 1
			action = policy.select_action(np.array(state))
			if curr_episode < 150:
				next_state, reward, done, info = eval_env.step(action)
				if int(info['success']) != 0:
					success_bool = True
			else:
				done = 1
			avg_reward += reward
		if success_bool:
			success_eval_cnt += 1

	avg_reward /= eval_episodes
	success_rate_eval = success_eval_cnt / eval_episodes

	print(success_rate_eval)

	writer.add_scalar("success_rate/val", success_rate_eval, t)
	writer.add_scalar("reward/eval", avg_reward, t)

	print("---------------------------------------")
	print(f"Evaluation over {eval_episodes} episodes: {avg_reward:.3f}")
	print("---------------------------------------")
	return avg_reward

def main(args):
	import metaworld
	import random
	file_name = f"{args.policy}_{args.env}_{args.seed}"
	print("---------------------------------------")
	print(f"Policy: {args.policy}, Env: {args.env}, Seed: {args.seed}")
	print("---------------------------------------")

	log_path = safe_path(os.path.join(args.log_root, '{}_base'.format(args.env)))
	# log_path = safe_path("/home/wuqiuche/Cycle_Dynamics/logs/cross_physics/HalfCheetah-v2_base")
	result_path = safe_path(os.path.join(log_path, 'results'))
	model_path = safe_path(os.path.join(log_path, 'models'))

	random.seed(args.seed)
	torch.manual_seed(args.seed)
	np.random.seed(args.seed)

	print(metaworld.ML1.ENV_NAMES)
	ml1 = metaworld.ML1('push-v1')
	env = ml1.train_classes['push-v1']()
	# print(ml1.train_tasks)
	# for each in ml1.train_tasks:
	# 	print(each.env_name)
	task = random.choice(ml1.train_tasks[0:50])
	print(task)
	env.set_task(task) 
	obs = env.reset()
	env.seed(args.seed)
	print(obs)
	# print(task)
	# print(env.action_space)
	env.action_space.seed(args.seed)
	a = env.action_space.sample()
	print(a)
	obs, reward, done, info = env.step(a)

	# env = gym.make(args.env)

	# Set seeds

	state_dim = env.observation_space.shape[0]
	action_dim = env.action_space.shape[0]
	max_action = float(env.action_space.high[0])

	kwargs = {
		"state_dim": state_dim,
		"action_dim": action_dim,
		"max_action": max_action,
		"discount": args.discount,
		"tau": args.tau,
	}

	# # Initialize policy
	if args.policy == "TD3":
		# Target policy smoothing is scaled wrt the action scale
		kwargs["policy_noise"] = args.policy_noise * max_action
		kwargs["noise_clip"] = args.noise_clip * max_action
		kwargs["policy_freq"] = args.policy_freq
		policy = TD3.TD3(**kwargs)

	if args.policy == 'SAC':
		policy = sac_new.SAC(**kwargs)

	replay_buffer = utils.ReplayBuffer(state_dim, action_dim)

	# # Evaluate untrained policy
	evaluations = [eval_policy(policy, args.env, args.seed, 0)]

	state, done = env.reset(), False
	episode_reward = 0
	episode_timesteps = 0
	curr_episode = 0
	episode_num = 0
	arr_train_reward = []
	arr_success = []
	cnt_success = 0
	episode_cnt = 0
	success_episode_cnt = 0
	success_bool = False
	episode_cnt_arr = []

	for t in range(int(args.max_timesteps)):

		episode_timesteps += 1
		curr_episode += 1

		# Select action randomly or according to policy
		if t < args.start_timesteps:
			action = env.action_space.sample()
		else:
			action = (
					policy.select_action(np.array(state))
					+ np.random.normal(0, max_action * args.expl_noise, size=action_dim)
			).clip(-max_action, max_action)

		# Perform action
		if curr_episode < 150:
			next_state, reward, done, info = env.step(action)
			writer.add_scalar("reward/train", reward, t)
			if int(info['success']) != 0:
				success_bool = True
			cnt_success += int(info['success'])
			# writer.add_scalar("success/train", cnt_success, t)
		else:
			done = 1
			curr_episode = 0
		# try:
			# next_state, reward, done, info = env.step(action)
			# arr_train_reward.append(reward)
			# writer.add_scalar("reward/train", reward, t)
			# arr_success.append(info['success'])
		# except:
		# 	done = 1
		done_bool = float(done) if episode_timesteps < 149 else 0

		# Store data in replay buffer
		replay_buffer.add(state, action, next_state, 0.001*reward, done_bool)

		state = next_state
		episode_reward += reward

		# Train agent after collecting sufficient data
		if t >= args.start_timesteps:
			# print("Start training")
			policy_info = policy.train(replay_buffer, args.batch_size)
			for each_loginfo in policy_info:
				writer.add_scalar(each_loginfo, policy_info[each_loginfo], t)
			# writer.add_scalar("actor_loss/train", actor_loss, t)
			# writer.add_scalar("critic_loss/train", critic_loss, t)
			# writer.flush()

		if done:
			# +1 to account for 0 indexing. +0 on ep_timesteps since it will increment +1 even if done=True
			print(
				f"Total T: {t + 1} Episode Num: {episode_num + 1} Episode T: {episode_timesteps} Reward: {episode_reward:.3f}")
			writer.add_scalar("episode_reward/train", episode_reward, t)
			# print(cnt_reward_pos)
			# Reset environment
			# episode_cnt += 1
			# if success_bool:
			# 	success_episode_cnt += 1
			if success_bool:
				episode_cnt_arr.append(1)
			else:
				episode_cnt_arr.append(0)
			if len(episode_cnt_arr) == 100:
				writer.add_scalar("success_rate/train", sum(episode_cnt_arr)/len(episode_cnt_arr), t)
				episode_cnt_arr.pop(0)
			# print(len(episode_cnt_arr))
			# if episode_cnt % 100 == 0:
			# 	print(success_episode_cnt)
			# 	print(episode_cnt)
			# 	writer.add_scalar("success_rate/train", success_episode_cnt/episode_cnt, t)
			# 	episode_cnt = 0
			# 	success_episode_cnt = 0
			task = random.choice(ml1.train_tasks[0:50])
			env.set_task(task) 
			state, done = env.reset(), False
			env.seed(args.seed)
			env.action_space.seed(args.seed)
			episode_reward = 0
			episode_timesteps = 0
			curr_episode = 0
			episode_num += 1
			success_bool = False


		# Evaluate episode
		if (t + 1) % args.eval_freq == 0:
			evaluations.append(eval_policy(policy, args.env, args.seed, t))
			np.save(os.path.join(result_path, '{}'.format(file_name)), evaluations)
			if args.save_model: policy.save(os.path.join(model_path, '{}'.format(file_name)))

		# if t == 1e5:
		# 	np.save('reward', arr_train_reward, allow_pickle = True)
		# 	np.save('success', arr_success, allow_pickle = True)



if __name__ == "__main__":
	
	parser = argparse.ArgumentParser()
	# parser.add_argument("--policy", default="TD3")                  # Policy name (TD3, DDPG or OurDDPG)
	parser.add_argument("--policy", default="SAC")                  # Policy name (TD3, DDPG or OurDDPG)
	parser.add_argument("--env", default="pick-and-place")          # OpenAI gym environment name
	parser.add_argument("--seed", default=0, type=int)              # Sets Gym, PyTorch and Numpy seeds
	parser.add_argument("--start_timesteps", default=25e3, type=int)# Time steps initial random policy is used
	parser.add_argument("--eval_freq", default=5e3, type=int)       # How often (time steps) we evaluate
	parser.add_argument("--max_timesteps", default=1e6, type=int)   # Max time steps to run environment
	# Possible Adjust
	parser.add_argument("--expl_noise", default=0.01)                # Std of Gaussian exploration noise
	parser.add_argument("--batch_size", default=256, type=int)      # Batch size for both actor and critic
	parser.add_argument("--discount", default=0.99)                 # Discount factor
	parser.add_argument("--tau", default=0.005)                     # Target network update rate
	parser.add_argument("--policy_noise", default=0.02)              # Noise added to target policy during critic update
	parser.add_argument("--noise_clip", default=0.05)                # Range to clip target policy noise
	parser.add_argument("--policy_freq", default=20, type=int)       # Frequency of delayed policy updates
	parser.add_argument("--save_model", default=True)               # Save model and optimizer parameters
	parser.add_argument("--load_model", default="")                 # Model load file name, "" doesn't load, "default" uses file_name

	parser.add_argument("--log_root", default="../../../logs/cross_physics")
	args = parser.parse_args()

	main(args)

# 02/09: 1. Push: train with 50 tasks, val with 50 tasks, observe succ_rate on train/val(especiall val since val is without noise)
# 2. Push is ready, then train "pick-and-place"
# 3. Final goal: pick-and-place as context, move to "basketball" on meta-world. This means we have trained
# a trainsition model that works within meta-world setting.



#td3 - # episodes, reward tendency (increasing?), train_loss (train environment different from validation?), 

# Check initial states after we "env.reset()"
# Train environment = validation environment, validation is no noise - check how the noise influences result.
# Tensorboard - loss tendency
# Push, reach environment
# td3 - # episodes, reward tendency (increasing?), train_loss (train environment different from validation?), 