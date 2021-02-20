import numpy as np
import torch
import torch.nn as nn
import gym
import argparse
import os

import utils
import TD3
from tensorboardX import SummaryWriter
from halfcheetah_arma import HalfCheetahARMAEnv
from cycle_transfer_contrastive import ActionTransferAgent
from models import ImagePool, MultiPairAxmodel, MultiDmodel, MultiGANLoss, Forwardmodel
import copy

def safe_path(path):
	if not os.path.exists(path):
		os.mkdir(path)
	return path



# Runs policy for X episodes and returns average reward
# A fixed seed is used for the eval environment
#eval env should be created in the main function, instead of in this eval function
def eval_policy(policy, cmodel, eval_env, data_type, history_length, eval_episodes=10):

	avg_reward = 0.
	avg_reward_context = [0. for _ in range(len(data_type))]
	num_episode_context = [0. for _ in range(len(data_type))]

	for _ in range(eval_episodes):
		state, done = eval_env.reset(), False
		cp_obs = np.zeros((history_length, eval_env.observation_space.shape[0]))
		cp_act = np.zeros((history_length, eval_env.action_space.shape[0]))
		context = eval_env.get_sim_parameters()
		context_index = data_type.index(context[0])
		num_episode_context[context_index] += 1

		while not done:
			cp_obs_tensor = torch.FloatTensor(cp_obs).reshape(1,-1).cuda()
			cp_act_tensor = torch.FloatTensor(cp_act).reshape(1,-1).cuda()
			context_tensor = cmodel(cp_obs_tensor, cp_act_tensor)
			action = policy.select_action(np.array(state), context_tensor)
			next_state, reward, done, _ = eval_env.step(action)
			cp_obs[:-1] = cp_obs[1:]
			cp_obs[-1] = copy.deepcopy(state)
			cp_act[:-1] = cp_act[1:]
			cp_act[-1] = copy.deepcopy(action)
			context = eval_env.get_sim_parameters()
			avg_reward += reward
			avg_reward_context[context_index] += reward
			state = next_state

	avg_reward /= eval_episodes

	for i in range(len(data_type)):
		if num_episode_context[i] == 0:
			num_episode_context[i] = 1
			avg_reward_context[i] = 0.0
	avg_reward_context = np.array(avg_reward_context)/np.array(num_episode_context)
	print("---------------------------------------")
	print(f"Evaluation over {eval_episodes} episodes: {avg_reward:.3f}")
	print("---------------------------------------")
	return avg_reward, avg_reward_context



def main(args):
	print("---------------------------------------")
	print(f"Policy: {args.policy}, Env: {args.env}, Seed: {args.seed}")
	print("---------------------------------------")

	if args.env == 'HalfCheetah':
		data_type = [0.1, 0.2, 0.3, 0.4, 0.5]
		data_type_str = '0.1+0.2+0.3+0.4+0.5'
		env = HalfCheetahARMAEnv(armas=data_type, reward_delay=args.reward_delay)
		env.set_seed(args.seed)
		env.action_space.seed(args.seed)
		eval_env = HalfCheetahARMAEnv(armas=data_type, reward_delay=args.reward_delay)
		eval_env.set_seed(args.seed)
		eval_env.action_space.seed(args.seed)

	log_path = safe_path(os.path.join(args.log_root, '{}_{}_delay{}_s{}_sil_update{}_value{}_clip{}_noise{}_th{}_history{}_future{}_context{}_{}_n{}'.format(args.env, data_type_str, args.reward_delay, args.seed, args.sil_update, args.sil_value, args.sil_clip, args.expl_noise, args.threshold, args.history_length, args.future_length, args.context_dim, args.context_weight, args.pair_n)))
	model_path = safe_path(os.path.join(log_path, 'models'))
	writer = SummaryWriter(log_path)

	# Set seeds
	torch.manual_seed(args.seed)
	torch.cuda.manual_seed_all(args.seed)
	torch.backends.cudnn.deterministic = True
	torch.backends.cudnn.benchmark = False
	np.random.seed(args.seed)

	context_dim = args.context_dim
	state_dim = env.observation_space.shape[0]
	action_dim = env.action_space.shape[0]
	max_action = float(env.action_space.high[0])

	kwargs = {
		"state_dim": state_dim,
		"action_dim": action_dim,
		"context_dim": context_dim,
		"max_action": max_action,
		"discount": args.discount,
		"tau": args.tau,
	}

	# Initialize policy
	if args.policy == "TD3":
		# Target policy smoothing is scaled wrt the action scale
		kwargs["policy_noise"] = args.policy_noise * max_action
		kwargs["noise_clip"] = args.noise_clip * max_action
		kwargs["policy_freq"] = args.policy_freq
		policy = TD3.TD3(**kwargs)

	replay_buffer = utils.ReplayBuffer(state_dim, action_dim, history_length=args.history_length)
	sil_replay_buffer = utils.SILPrioritizedReplayBuffer(state_dim, action_dim, history_length=args.history_length, gamma=args.discount, alpha=args.sil_alpha, beta=args.sil_beta)
	action_transfer_agent = ActionTransferAgent(env_name=args.env, seed=args.seed, data_type=data_type, reward_delay=args.reward_delay, history_length=args.history_length, future_length=args.future_length, context_dim=args.context_dim, context_weight=args.context_weight, pair_n=args.pair_n)

	state, done = env.reset(), False
	cp_obs = np.zeros((args.history_length, env.observation_space.shape[0]))
	cp_act = np.zeros((args.history_length, env.action_space.shape[0]))
	#print('initial state', state)
	context = env.get_sim_parameters()
	episode_reward = 0
	episode_timesteps = 0
	episode_num = 0

	source_context = None
	avg_reward_context = np.zeros(len(data_type))
	for t in range(int(args.max_timesteps)):

		episode_timesteps += 1

		# Select action randomly or according to policy
		if t < args.start_timesteps:
			action = env.action_space.sample()
		else:
			cp_obs_tensor = torch.FloatTensor(cp_obs).reshape(1,-1).cuda()
			cp_act_tensor = torch.FloatTensor(cp_act).reshape(1,-1).cuda()
			context_tensor = action_transfer_agent.cmodel(cp_obs_tensor, cp_act_tensor)
			if source_context is not None:
				#action = action_transfer_agent.select_action(state, source_context=source_context, target_context=context, shared_policy=policy)
				action = (action_transfer_agent.select_action(state, source_context=source_context, target_context_tensor=context_tensor, shared_policy=policy) + np.random.normal(0, max_action * args.expl_noise, size=action_dim)).clip(-max_action, max_action)
			else:
				action = (policy.select_action(np.array(state), context_tensor) + np.random.normal(0, max_action * args.expl_noise, size=action_dim)).clip(-max_action, max_action)

		# Perform action
		next_state, reward, done, _ = env.step(action)
		next_context = env.get_sim_parameters()
		done_bool = float(done) if episode_timesteps < env._max_episode_steps else 0

		# Store data in replay buffer
		replay_buffer.add(state, action, next_state, reward, done_bool, cp_obs, cp_act)
		sil_replay_buffer.add(state, action, next_state, reward, done, cp_obs, cp_act)
		# Store data in transfer agent
		action_transfer_agent.add(state, action, done, context, cp_obs, cp_act)

		cp_obs[:-1] = cp_obs[1:]
		cp_obs[-1] = copy.deepcopy(state)
		cp_act[:-1] = cp_act[1:]
		cp_act[-1] = copy.deepcopy(action)
		state = next_state
		context = next_context
		episode_reward += reward

		# Train agent after collecting sufficient data
		if t >= args.start_timesteps:
			policy.train(replay_buffer, action_transfer_agent.cmodel, args.batch_size)
			policy.update_parameters()
			if sil_replay_buffer.size > args.batch_size and source_context is not None:
				policy.sil_train(sil_replay_buffer, action_transfer_agent.cmodel, args.batch_size, sil_update=args.sil_update, sil_value=args.sil_value, sil_clip=args.sil_clip)
				#policy.update_parameters()
		if done:
			# +1 to account for 0 indexing. +0 on ep_timesteps since it will increment +1 even if done=True
			writer.add_scalar('train_reward/arma%g'%context, episode_reward, t+1)
			if source_context is None:
				writer.add_scalar('train_source_context/arma%g'%context, 0, t+1)
			else:
				writer.add_scalar('train_source_context/arma%g'%context, source_context, t+1)
			print(
				f"Total T: {t + 1} Episode Num: {episode_num + 1} Episode T: {episode_timesteps} Reward: {episode_reward:.3f} Source: {source_context}")
			# Reset environment
			state, done = env.reset(), False
			cp_obs = np.zeros((args.history_length, env.observation_space.shape[0]))
			cp_act = np.zeros((args.history_length, env.action_space.shape[0]))
			context = env.get_sim_parameters()
			episode_reward = 0
			episode_timesteps = 0
			episode_num += 1
			source_context = action_transfer_agent.good_to_transfer(target_context=context, target_rewards=avg_reward_context, threshold=args.threshold)

		# Evaluate episode
		if (t + 1) % args.eval_freq == 0:
			avg_reward, avg_reward_context = eval_policy(policy, action_transfer_agent.cmodel, eval_env, history_length=args.history_length, data_type=data_type)
			writer.add_scalar("eval_reward/avg", avg_reward, t+1)
			for i in range(len(data_type)):
				if not avg_reward_context[i]==0:
					writer.add_scalar("eval_reward/arma%g"%data_type[i], avg_reward_context[i], t+1)
                        # train transfer agent
			action_transfer_agent.train_ax(writer, t, policy)
			print('avg reward context', avg_reward_context)
			print('transfer rewards', action_transfer_agent.transfer_rewards)


if __name__ == "__main__":
	
	parser = argparse.ArgumentParser()
	parser.add_argument("--policy", default='TD3', type=str)
	parser.add_argument("--env", default='HalfCheetah', type=str)
	parser.add_argument("--reward_delay", default=500, type=int) 
	parser.add_argument("--seed", default=0, type=int)              # Sets Gym, PyTorch and Numpy seeds
	parser.add_argument("--start_timesteps", default=5e3, type=int)# Time steps initial random policy is used
	parser.add_argument("--eval_freq", default=5e3, type=int)       # How often (time steps) we evaluate
	parser.add_argument("--max_timesteps", default=10e6, type=int)   # Max time steps to run environment
	parser.add_argument("--expl_noise", default=0.1)                # Std of Gaussian exploration noise
	parser.add_argument("--batch_size", default=256, type=int)      # Batch size for both actor and critic
	parser.add_argument("--discount", default=0.99)                 # Discount factor
	parser.add_argument("--tau", default=0.005)                     # Target network update rate
	parser.add_argument("--policy_noise", default=0.2)              # Noise added to target policy during critic update
	parser.add_argument("--noise_clip", default=0.5)                # Range to clip target policy noise
	parser.add_argument("--policy_freq", default=2, type=int)       # Frequency of delayed policy updates
	parser.add_argument("--save_model", default=True)               # Save model and optimizer parameters
	parser.add_argument("--load_model", default="")                 # Model load file name, "" doesn't load, "default" uses file_name

	parser.add_argument("--log_root", default="results_sil_transfer_contrastive_raresil_divide1/")
	parser.add_argument("--sil_update", default=0, type=int)
	parser.add_argument("--sil_value", default=1., type=float)
	parser.add_argument("--sil_clip", default=1., type=float)
	parser.add_argument("--sil_alpha", default=0.6, type=float)
	parser.add_argument("--sil_beta", default=0.1, type=float)
	parser.add_argument("--threshold", default=0.5, type=float)
	parser.add_argument("--history_length", default=10, type=int)
	parser.add_argument("--future_length", default=20, type=int)
	parser.add_argument("--context_dim", default=1, type=int)
	parser.add_argument("--context_weight", default=0.01, type=float)
	parser.add_argument("--pair_n", default=5000, type=int)
	args = parser.parse_args()

	main(args)
