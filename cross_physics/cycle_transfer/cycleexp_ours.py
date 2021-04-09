import os
import torch
import random
import argparse
import numpy as np
import torch.nn as nn
from tqdm import tqdm
import matplotlib.pyplot as plt
import torch.utils.data as Data
import matplotlib.pyplot as plt

# from collect_data import CycleData
from collect_data_push_dooropen import CycleData
from models import Forwardmodel,Axmodel,Dmodel,GANLoss,ImagePool,net_init,Stmodel, DStmodel

from torch.utils.tensorboard import SummaryWriter
writer = SummaryWriter()

def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True

setup_seed(0)


def show_points(gt_data, pred_data):
    ncols = int(np.sqrt(gt_data.shape[1])) + 1
    nrows = int(np.sqrt(gt_data.shape[1])) + 1
    assert (ncols * nrows >= gt_data.shape[1])
    _, axes = plt.subplots(ncols, nrows, figsize=(nrows * 3, ncols * 3))
    axes = axes.flatten()

    for ax_i, ax in enumerate(axes):
        if ax_i >= gt_data.shape[1]:
            continue
        ax.scatter(gt_data[:, ax_i], pred_data[:, ax_i], s=3, label='xyz_{}'.format(ax_i))

class Agent():
    def __init__(self,opt):
        self.opt = opt
        self.dataset = CycleData(opt)
        self.model = self.dataset.model

import sys
sys.path.append("/mnt/brain7/scratch/wuqiuche/Cycle_Dynamics/cross_physics/base_train_test/td3_solver/torchrl/")
import torchrl.policies as policies
import torchrl.networks as networks
import metaworld
import random

class ActionAgent:
    def __init__(self,opt):
        self.opt = opt
        opt.data_type = opt.data_type1
        opt.data_id = opt.data_id1
        opt.env = opt.env1
        self.agent1 = Agent(opt)
        opt.data_type = opt.data_type2
        opt.data_id = opt.data_id2
        opt.env = opt.env2
        self.agent2 = Agent(opt)
        opt.state_dim = self.agent1.dataset.state_dim
        opt.action_dim = self.agent1.dataset.action_dim
        self.env_logs = self.agent1.dataset.env_logs

        self.netG_B = Stmodel(opt).cuda()
        self.netD_B = DStmodel(opt).cuda()
        self.netF_A = self.agent1.model

        self.criterionCycle = nn.L1Loss()
        # self.model = Axmodel(opt).cuda()
        # self.back_model = Axmodel(opt).cuda()
        # self.dmodel = Dmodel(opt).cuda()
        # if self.opt.env == 'Walker2d-v2':
        #     net_init(self.model)
        #     net_init(self.back_model)
        #     net_init(self.dmodel)

        self.criterionGAN = GANLoss().cuda()
        self.fake_pool = ImagePool(256)
        self.real_pool = ImagePool(256)
        self.weight_path = os.path.join(self.env_logs,
                'model_{}_{}.pth'.format(opt.data_type1,opt.data_type2))

    def get_optim(self,lr):
        optimizer_g = torch.optim.Adam([{'params': self.agent1.model.parameters(), 'lr': 0.0},
                                      {'params': self.netG_B.parameters(), 'lr': lr}])
        self.optimizer_d = torch.optim.Adam(self.netD_B.parameters(),lr=lr)
        return optimizer_g,self.optimizer_d

    def online_test_state(self, G_B, itera):
        print("In online test, env", self.opt.env2)
        with torch.no_grad():
            now_buffer, action_buffer, nxt_buffer = [], [], []
            reward_buffer = []
            agent1_policy_model = policies.GuassianContPolicy(input_shape=self.opt.state_dim, output_shape=2 * self.opt.action_dim, **{"hidden_shapes": [400,400,400,400],"append_hidden_shapes": [], "base_type": networks.MLPBase, "activation_func": torch.nn.ReLU},**{"tanh_action": True}).cuda()
            if self.opt.env1 == 'push':
                agent1_policy_model.load_state_dict(torch.load("/mnt/brain7/scratch/wuqiuche/Cycle_Dynamics/cross_physics/base_train_test/td3_solver/torchrl/log/sac_push_400_4/Push/0/model/model_pf_best.pth"))
                # self.policy.load_state_dict(torch.load("/mnt/brain7/scratch/wuqiuche/Cycle_Dynamics/cross_physics/base_train_test/td3_solver/torchrl/log/push_to_dooropen_400/Push/0/model/model_pf_best.pth"))
                # self.policy.load_state_dict(torch.load("/mnt/brain7/scratch/wuqiuche/Cycle_Dynamics/cross_physics/base_train_test/td3_solver/torchrl/log/push_to_dooropen_400_full_observ/Push/0/model/model_pf_best.pth"))
                # self.policy.load_state_dict(torch.load("/mnt/brain7/scratch/wuqiuche/Cycle_Dynamics/cross_physics/base_train_test/td3_solver/torchrl/log/push_obj_goal_to_dooropen_400/Push/0/model/model_pf_best.pth"))
            elif self.opt.env1 == 'door_open':
                # self.policy.load_state_dict(torch.load("/mnt/brain7/scratch/wuqiuche/Cycle_Dynamics/cross_physics/base_train_test/td3_solver/torchrl/log/door_open_400_4/Door_open/0/model/model_pf_best.pth"))
                agent1_policy_model.load_state_dict(torch.load("/mnt/brain7/scratch/wuqiuche/Cycle_Dynamics/cross_physics/base_train_test/td3_solver/torchrl/log/door_open_fullobserv_400/Door_open/0/model/model_pf_best.pth"))
            elif self.opt.env1 == 'push_back':
                agent1_policy_model.load_state_dict(torch.load("/mnt/brain7/scratch/wuqiuche/Cycle_Dynamics/cross_physics/base_train_test/td3_solver/torchrl/log/push_back_400/Push_back/0/model/model_pf_best.pth"))
            elif self.opt.env1 == 'coffee_push':
                agent1_policy_model.load_state_dict(torch.load("/mnt/brain7/scratch/wuqiuche/Cycle_Dynamics/cross_physics/base_train_test/td3_solver/torchrl/log/coffee_push_400/Coffee_push/0/model/model_pf_best.pth"))
            for episode in range(itera):
                if self.opt.env2 == 'push':
                    ml1 = metaworld.ML1('push-v1')
                    curr_env = ml1.train_classes['push-v1']()
                elif self.opt.env2 == 'door_open':
                    ml1 = metaworld.ML1('door-open-v1')
                    curr_env = ml1.train_classes['door-open-v1']()
                elif self.opt.env2 == 'coffee_push':
                    ml1 = metaworld.ML1('coffee-push-v1')
                    curr_env = ml1.train_classes['coffee-push-v1']()
                elif self.opt.env2 == 'push_back':
                    ml1 = metaworld.ML1('push-back-v1')
                    curr_env = ml1.train_classes['push-back-v1']()
                task = random.choice(ml1.train_tasks[0:50])
                curr_env.set_task(task)
                obs = curr_env.reset()
                done = False
                episode_r = 0.
                count = 0
                cnt_step = 0
                
                while not done:
                    cnt_step += 1
                    if cnt_step > 150:
                        break
                    obs_1 = G_B(torch.FloatTensor(obs).cuda())
                    # obs_1 = torch.FloatTensor(obs_1)
                    act_1_2 = agent1_policy_model(obs_1)[0]
                    # print(act_1_2)
                    obs, r, done, info = curr_env.step(act_1_2.cpu().numpy())
                    episode_r += r
                reward_buffer.append(episode_r)
            return np.array(reward_buffer)



    def train_ax(self):
        lr = 1e-3
        optimizer_g,optimizer_d = self.get_optim(lr)
        loss_fn = nn.L1Loss()

        print('----------initial test as baseline--------------')
        ref_reward = self.online_test_state(lambda x:x, 10)
        # ref_reward = self.agent2.dataset.online_test(lambda x,y:y, self.agent1.dataset, 10)

        ours,baseline = [],[]
        self.opt.istrain = True
        last_back_loss = 100.
        max_reward = 0.

        lambda_G_B0 = 10.
        lambda_G_B1 = 10.
        lambda_G_B2 = 0.
        lambda_F = 200.

        for epoch in range(self.opt.epoch_n):
            # print("------")
            # print("\n"*20)
            # print(epoch)
            epoch_loss, cmp_loss = 0, 0
            
            for i in (range(self.opt.pair_n)):
                # print("\n"*10)
                # print(i)
                item1 = self.agent1.dataset.sample()
                real_A_state = item1[0]

                item2 = self.agent2.dataset.sample()
                now_state, action, nxt_state = item2

                fake_At0 = self.netG_B(now_state)
                pred_fake = self.netD_B(fake_At0)
                loss_G_Bt0 = self.criterionGAN(pred_fake, True) * lambda_G_B0

                fake_At1 = self.agent1.model(fake_At0, action)
                pred_fake = self.netD_B(fake_At1)
                loss_G_Bt1 = self.criterionGAN(pred_fake, True) * lambda_G_B1

                pred_At1 = self.netG_B(nxt_state)
                cycle_label = torch.zeros_like(fake_At1).float().cuda()
                diff = fake_At1-pred_At1
                loss_cycle = self.criterionCycle(diff,cycle_label) * lambda_F

                pred_fake = self.netD_B(pred_At1)
                loss_G_Bt2 = self.criterionGAN(pred_fake, True) * lambda_G_B2

                loss_G = loss_G_Bt0 + loss_G_Bt1 + loss_G_Bt2 + loss_cycle

                if self.opt.istrain:
                    optimizer_g.zero_grad()
                    loss_G.backward()
                    optimizer_g.step()
                epoch_loss += loss_cycle.item()

                optimizer_d.zero_grad()
                fake_At0 = self.netG_B(now_state)
                pred_fake = self.netD_B(fake_At0)
                loss_d_fake = self.criterionGAN(pred_fake, False)

                pred_real = self.netD_B(real_A_state)
                loss_d_real = self.criterionGAN(pred_real, True)

                loss_d = (loss_d_fake + loss_d_real) * 0.5
                loss_d.backward()
                optimizer_d.step()

                if i % 100 == 0:
                    writer.add_scalar("loss_d_real", loss_d_real, i + epoch * self.opt.pair_n)
                    writer.add_scalar("loss_d_fake", loss_d_fake, i + epoch * self.opt.pair_n)
                    writer.add_scalar("loss_d", loss_d, i + epoch * self.opt.pair_n)
                    writer.add_scalar("loss_G_Bt0", loss_G_Bt0, i + epoch * self.opt.pair_n)
                    writer.add_scalar("loss_G_Bt1", loss_G_Bt1, i + epoch * self.opt.pair_n)
                    writer.add_scalar("loss_G", loss_G, i + epoch * self.opt.pair_n)


            print('epoch:{} cycle_loss:{:.3f}'.format(epoch, epoch_loss / self.opt.pair_n))

            reward_ours = self.online_test_state(self.netG_B, 10)

            if reward_ours.mean() > max_reward:
                max_reward = reward_ours.mean()
                torch.save(self.back_model.state_dict(), self.weight_path)
            print('ours_cur:{:.2f}  ours_max:{:.2f}  ref_baseline:{:.2f}\n'
                  .format(reward_ours.mean(), max_reward, ref_reward.mean()))
            writer.add_scalar("ours_cur", reward_ours.mean(), epoch)
            writer.add_scalar("ours_max", max_reward, epoch)
            writer.add_scalar("ref_baseline", ref_reward.mean(), epoch)


    def eval_ax(self):
        self.back_model.load_state_dict(torch.load(self.weight_path))
        img_path_ours = os.path.join(self.agent2.dataset.data_folder,'ours_eval')
        img_path_base = os.path.join(self.agent2.dataset.data_folder,'base_eval')
        reward_ours = self.agent2.dataset.online_test(self.back_model, 10,img_path_ours)
        reward_base = self.agent2.dataset.online_test(lambda x, y: y, 10,img_path_base)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='control dataset analyzer')
    parser.add_argument('--istrain', type=bool, default=True, help='train or eval')
    # parser.add_argument('--pretrainF', type=bool, default=False, help='train or eval')
    parser.add_argument('--pair_n', type=int, default=3000, help='dataset sample number')

    parser.add_argument("--env", default="Walker2d-v2")
    parser.add_argument("--force", type=bool, default=False)
    parser.add_argument("--log_root", default="../../../logs/cross_physics")
    parser.add_argument('--episode_n', type=int, default=100, help='episode number')
    parser.add_argument('--state_dim', type=int, default=0, help='state dim')
    parser.add_argument('--action_dim', type=int, default=0, help='action dim')
    parser.add_argument('--eval_n', type=int, default=100, help='evaluation episode number')
    parser.add_argument('--epoch_n', type=int, default=30, help='training epoch number')

    parser.add_argument('--data_type', type=str, default='base', help='data type')
    parser.add_argument('--data_type1', type=str, default='base', help='data type')
    parser.add_argument('--data_type2', type=str, default='arma3', help='data type')

    parser.add_argument('--data_id', type=int, default=0, help='data id')
    parser.add_argument('--data_id1', type=int, default=3, help='data id')
    parser.add_argument('--data_id2', type=int, default=3, help='data id')

    opt = parser.parse_args()

    opt.env1 = opt.data_type1
    opt.env2 = opt.data_type2
    opt.eval_n = 5
    opt.pair_n = 3000
    opt.istrain = True
    opt.epoch_n = 30

    # # *****************************
    # #        halfcheetah
    # # *****************************
    # opt.data_id1 = 1
    # opt.data_id2 = 1
    # opt.lambda_F = 200
    # opt.lambda_G0 = 10
    # opt.lambda_G1 = 10
    # # opt.state_dim = 8
    # opt.pretrain_f = False

    agent = ActionAgent(opt)
    agent.train_ax()
    # train(opt)
    # opt.istrain = False
    # with torch.no_grad():
    #     eval(opt)







