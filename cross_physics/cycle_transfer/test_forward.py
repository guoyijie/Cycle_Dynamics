from models import Forwardmodel
import torch
import argparse
import metaworld
import random
import numpy as np
import torch.nn as nn


if __name__ == '__main__':
	parser = argparse.ArgumentParser(description='control dataset analyzer')
	parser.add_argument("--env", default="HalfCheetah-v2")
	parser.add_argument("--force", type=bool, default=True)
	parser.add_argument("--log_root", default="../../../logs/cross_physics")
	parser.add_argument('--data_type', type=str, default='arma3', help='data type')
	parser.add_argument('--data_id', type=int, default=2, help='data id')
	parser.add_argument('--episode_n', type=int, default=1000, help='episode number')
	opt = parser.parse_args()

	ml1 = metaworld.ML1('push-v1')
	env = ml1.train_classes['push-v1']()
	task = random.choice(ml1.train_tasks[0:50])
	env.set_task(task)

	opt.state_dim = env.observation_space.shape[0]
	opt.action_dim = env.action_space.shape[0]

	forward_model_push = Forwardmodel(opt).cuda().float()
	# forward_model_push.load_state_dict(torch.load("../../logs/cross_physics/push_data/push_2003/forward.pth"))

	# forward_model_coffee_push = Forwardmodel(opt).cuda()
	forward_model_push.load_state_dict(torch.load("../../logs/cross_physics/coffee_push_data/coffee_push_2003/forward.pth"))

	push_now_obs = torch.tensor(np.load("../../logs/cross_physics/push_data/push_2003/now.npy")).cuda().float()
	push_nxt_obs = torch.tensor(np.load("../../logs/cross_physics/push_data/push_2003/nxt.npy")).cuda().float()
	push_action = torch.tensor(np.load("../../logs/cross_physics/push_data/push_2003/act.npy")).cuda().float()

	coffee_push_now_obs = torch.tensor(np.load("../../logs/cross_physics/coffee_push_data/coffee_push_2003/now.npy")).cuda().float()
	coffee_push_nxt_obs = torch.tensor(np.load("../../logs/cross_physics/coffee_push_data/coffee_push_2003/nxt.npy")).cuda().float()
	coffee_push_action = torch.tensor(np.load("../../logs/cross_physics/coffee_push_data/coffee_push_2003/act.npy")).cuda().float()

	whole_data = (coffee_push_now_obs, coffee_push_action, coffee_push_nxt_obs)
	whole_data2 = (push_now_obs, push_action, push_nxt_obs)

	print(len(coffee_push_now_obs))

	loss_fn = nn.L1Loss()
	loss_total = 0.0

	for i in range(100):
		index = random.choice(range(150 * 1000))
		curr = whole_data2[0][i].reshape(1,-1)
		action = whole_data2[1][i].reshape(1,-1)
		target = whole_data2[2][i].reshape(1,-1)
		# sample = (curr, action, target)
		# print(curr)
		# print(action)
		ours = forward_model_push(curr, action)
		loss_total += loss_fn(ours, target)
	loss_total /= 100
	print(loss_total.item())




