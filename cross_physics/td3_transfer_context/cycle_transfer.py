import torch.nn as nn
import torch.nn.functional as F
from halfcheetah_arma import HalfCheetahARMAEnv
import torch
import numpy as np
import os
import numpy as np
from TD3 import TD3
from models import ImagePool, MultiPairAxmodel, MultiDmodel, MultiGANLoss, ContextModel, ContextForwardModel
import copy

class CycleData(object):
    def __init__(self, env_name, seed, data_type, reward_delay, history_length, sample_max=1e6):
        if env_name == 'HalfCheetah':
            self.env = HalfCheetahARMAEnv(armas=data_type, reward_delay=reward_delay)
            self.env.set_seed(seed)
            self.env.action_space.seed(seed)
        self.data_type=data_type[0]
        self.state_dim = self.env.observation_space.shape[0]
        self.action_dim = self.env.action_space.shape[0]
        self.max_action = float(self.env.action_space.high[0])
        self.history_length = history_length

        self.now_buffer = np.array([], dtype=np.float32).reshape(0,self.env.observation_space.shape[0])
        self.action_buffer = np.array([], dtype=np.float32).reshape(0,self.env.action_space.shape[0])
        self.nxt_buffer = np.array([], dtype=np.float32).reshape(0,self.env.observation_space.shape[0])
        self.cp_obs_buffer = np.array([], dtype=np.float32).reshape(0,self.history_length*self.env.observation_space.shape[0])
        self.cp_act_buffer = np.array([], dtype=np.float32).reshape(0,self.history_length*self.env.action_space.shape[0])
        
        self.sample_n = 0
        self.sample_max = int(sample_max)
        self.rng = np.random.RandomState(seed)

    def sample(self, batch_size=1024):
        if self.sample_n>0:
            idx = self.rng.random_integers(low=0, high=self.sample_n-1, size=batch_size)
            sample = (self.to_device(self.now_buffer[idx]),
                   self.to_device(self.action_buffer[idx]),
                   self.to_device(self.nxt_buffer[idx]),
                   self.to_device(self.cp_obs_buffer[idx]),
                   self.to_device(self.cp_act_buffer[idx]))
        else:
            sample = (None, None, None, None, None)
        return sample


    def to_device(self,data):
        return torch.tensor(data).float().cuda()

    def add(self, seg):
        data_obs, data_act, data_obs_next, data_cp_obs, data_cp_act = seg
        self.now_buffer = np.concatenate((self.now_buffer, data_obs), axis=0)
        self.action_buffer = np.concatenate((self.action_buffer, data_act), axis=0)
        self.nxt_buffer = np.concatenate((self.nxt_buffer, data_obs_next), axis=0)
        self.cp_obs_buffer = np.concatenate((self.cp_obs_buffer, data_cp_obs), axis=0)
        self.cp_act_buffer = np.concatenate((self.cp_act_buffer, data_cp_act), axis=0)
        if self.now_buffer.shape[0] > self.sample_max:
            self.now_buffer = self.now_buffer[:self.sample_max]
            self.action_buffer = self.action_buffer[:self.sample_max]
            self.nxt_buffer = self.nxt_buffer[:self.sample_max]
            self.cp_obs_buffer = self.cp_obs_buffer[:self.sample_max]
            self.cp_act_buffer = self.cp_act_buffer[:self.sample_max]
        self.sample_n = self.now_buffer.shape[0]

    def online_test(self, axmodel, cmodel, episode_n=100, source_context_tensor=None, shared_policy=None):
        with torch.no_grad():
            now_buffer, action_buffer, nxt_buffer = [], [], []
            reward_buffer = []
            for episode in (range(episode_n)):
                now_obs, action, nxt_obs = [], [], []
                obs = self.env.reset()
                cp_obs = np.zeros((self.history_length, self.env.observation_space.shape[0]))
                cp_act = np.zeros((self.history_length, self.env.action_space.shape[0]))
                done = False
                episode_r = 0.
                while not done:
                    now_obs.append(obs)
                    obs_tensor = torch.FloatTensor(obs.reshape(1, -1)).cuda()
                    cp_obs_tensor = torch.FloatTensor(cp_obs).reshape((-1, self.history_length*self.env.observation_space.shape[0])).cuda()
                    cp_act_tensor = torch.FloatTensor(cp_act).reshape((-1, self.history_length*self.env.action_space.shape[0])).cuda()
                    idx = np.random.choice(np.arange(source_context_tensor.shape[0]))
                    source_context = source_context_tensor[idx].reshape(1, -1).cuda()
                    target_context = cmodel(cp_obs_tensor, cp_act_tensor).cuda()
                    good_action =shared_policy.select_action(obs, source_context)
                    good_action = torch.FloatTensor(good_action).reshape(1,-1).cuda()
                    act = axmodel(obs_tensor,good_action,source_context, target_context).cpu().data.numpy().flatten()
                    new_obs, r, done, info = self.env.step(act)
                    cp_obs[:-1] = cp_obs[1:]
                    cp_obs[-1] = obs
                    cp_act[:-1] = cp_act[1:]
                    cp_act[-1] = act
                    action.append(act)
                    nxt_obs.append(new_obs)
                    obs = new_obs
                    episode_r += r
                reward_buffer.append(episode_r)
                now_buffer.extend(now_obs)
                action_buffer.extend(action)
                nxt_buffer.extend(nxt_obs)
                # print(episode_r/(episode+1))
            episode_r = sum(reward_buffer)
            # print('average reward: {:.2f}'.format(episode_r/episode_n))
            return np.array(reward_buffer)

class ActionTransferAgent(object):
    def __init__(self, env_name, seed, data_type, reward_delay, history_length, future_length, context_dim, context_weight, pair_n):
        self.data_type = data_type
        self.reward_delay = reward_delay
        self.history_length = history_length
        self.future_length = future_length
        self.context_dim = context_dim
        self.context_weight = context_weight
        self.pair_n = pair_n

        self.nested_agents = [None for _ in data_type]
        for j, target in enumerate(data_type):
            self.nested_agents[j] = CycleData(env_name, seed, data_type=[target], reward_delay=reward_delay, history_length=history_length)
        self.pools = [ImagePool(256) for _ in range(len(data_type))]

        self.state_dim = state_dim = self.nested_agents[0].state_dim
        self.action_dim = action_dim = self.nested_agents[0].action_dim

        self.cmodel = ContextModel(state_dim=state_dim, action_dim=action_dim, history_length=history_length, context_dim=context_dim).cuda()
        self.fmodel = ContextForwardModel(state_dim=state_dim, action_dim=action_dim, context_dim=context_dim).cuda()
        self.model = MultiPairAxmodel(state_dim, action_dim, context_dim).cuda()
        self.back_model = MultiPairAxmodel(state_dim, action_dim, context_dim).cuda()
        self.dmodel = MultiDmodel(state_dim, action_dim, context_dim).cuda()
        self.criterionGAN = MultiGANLoss().cuda()

        lr = 3e-4
        self.optimizer_g = torch.optim.Adam([{'params': self.back_model.parameters(), 'lr': lr}, {'params': self.model.parameters(), 'lr': lr}])
        self.optimizer_d = torch.optim.Adam(self.dmodel.parameters(),lr=lr)
        self.optimizer_f = torch.optim.Adam([{'params': self.cmodel.parameters(), 'lr': lr}, {'params': self.fmodel.parameters(), 'lr': lr}])

        self.transfer_rewards = np.array([[0 for target in data_type] for source in data_type])

        self.initialize_running_path()

    def initialize_running_path(self):
        self.running_path = {}
        self.running_path['observations'] = []
        self.running_path['actions'] = []
        self.running_path['cp_obs'] = []
        self.running_path['cp_act'] = []
        self.running_context = None

    def add(self, obs, action, done, context, cp_obs, cp_act):
        if self.running_context is not None:
            assert context == self.running_context
        self.running_context = context
        self.running_path['observations'].append(copy.deepcopy(obs))
        self.running_path['actions'].append(copy.deepcopy(action))
        self.running_path['cp_obs'].append(copy.deepcopy(cp_obs))
        self.running_path['cp_act'].append(copy.deepcopy(cp_act))
        if done:
            self.running_path['observations'] = np.array(self.running_path['observations'])
            self.running_path['actions'] = np.array(self.running_path['actions'])
            self.running_path['cp_obs'] = np.array(self.running_path['cp_obs'])
            self.running_path['cp_act'] = np.array(self.running_path['cp_act'])
            context_index = self.data_type.index(context)
            seg = self.process_traj(self.running_path)
            self.nested_agents[context_index].add(seg)
            self.initialize_running_path()

    def process_traj(self, path):
        history_length = self.history_length
        future_length = self.future_length

        obs_dim = self.state_dim
        act_dim = self.action_dim

        concat_obs_list, concat_act_list, concat_next_obs_list, concat_bool_list = [], [], [], []
        path_len = path["observations"].shape[0]
        concat_bool = np.ones((path["observations"][:-1].shape[0], future_length))
        for i in range(future_length):
            if i == 0:
                concat_obs = path["observations"][:-1]
                concat_act = path["actions"][:-1]
                concat_next_obs = path["observations"][1:]
                temp_next_act = path["actions"][1:]
            else:
                temp_next_obs = np.concatenate([path["observations"][1+i:], np.zeros((i, obs_dim))], axis=0)
                concat_obs = np.concatenate([concat_obs, concat_next_obs[:, -obs_dim:]], axis=1)
                concat_next_obs = np.concatenate([concat_next_obs, temp_next_obs], axis=1)

                concat_act = np.concatenate([concat_act, temp_next_act], axis=1)
                temp_next_act = np.concatenate([path["actions"][1+i:], np.zeros((i, act_dim))], axis=0)

            start_idx = max(i, 0)
            concat_bool[-i][start_idx:] = 0

        concat_obs_list.append(concat_obs)
        concat_act_list.append(concat_act)
        concat_next_obs_list.append(concat_next_obs)
        concat_bool_list.append(concat_bool)

        obs_next = np.concatenate(concat_next_obs_list, axis=0)
        obs = np.concatenate(concat_obs_list, axis=0)
        act = np.concatenate(concat_act_list, axis=0)
        future_bool = np.concatenate(concat_bool_list, axis=0)
        cp_obs = path["cp_obs"][:-1]
        cp_act = path["cp_act"][:-1]

        cp_obs = np.tile(cp_obs, (1, future_length))
        cp_obs = cp_obs.reshape((-1, obs_dim*history_length))
        cp_act = np.tile(cp_act, (1, future_length))
        cp_act = cp_act.reshape((-1, act_dim*history_length))
        obs = obs.reshape((-1, obs_dim))
        act = act.reshape((-1, act_dim))
        obs_next = obs_next.reshape((-1, obs_dim))
        future_bool = future_bool.reshape(-1)

        obs = obs[future_bool>0, :]
        act = act[future_bool>0, :]
        obs_next = obs_next[future_bool>0, :]
        cp_obs = cp_obs[future_bool>0, :]
        cp_act = cp_act[future_bool>0, :]

        return np.array(obs), np.array(act), np.array(obs_next), np.array(cp_obs), np.array(cp_act)

    def train_context_forward(self, writer, t, source_agent, target_agent):
        loss_fn = nn.L1Loss()
        triplet_loss_fn = nn.TripletMarginLoss(margin=1.0, p=2)
        epoch_loss, cmp_loss = 0, 0
        for i in range(self.pair_n):
            item = source_agent.sample()
            state, action, result, cp_state, cp_action = item
            if state is None:
                continue
            context = self.cmodel(cp_state, cp_action)
            out = self.fmodel(state, action, context)
            loss = loss_fn(out, result)
            epoch_loss += loss.item()
            if not source_agent.data_type==target_agent.data_type and self.context_weight>0:
                positive = source_agent.sample()
                _, _, _, cp_state_positive, cp_action_positive = positive
                positive_context = self.cmodel(cp_state_positive, cp_action_positive)
                negative = target_agent.sample()
                _, _, _, cp_state_negative, cp_action_negative = negative
                negative_context = self.cmodel(cp_state_negative, cp_action_negative)
                context_loss = triplet_loss_fn(context, positive_context, negative_context)*self.context_weight
                cmp_loss += context_loss.item()
                loss += context_loss
            self.optimizer_f.zero_grad()
            loss.backward()
            self.optimizer_f.step()
            if i==self.pair_n-1 and (not source_agent.data_type==target_agent.data_type):
                context = context.detach().cpu().data.numpy()
                #positive_context = positive_context.detach().cpu().data.numpy()
                #negative_context = negative_context.detach().cpu().data.numpy()
                print(source_agent.data_type, 'context', np.mean(context, axis=0), np.std(context, axis=0))
                #print(source_agent.data_type, 'context', np.mean(positive_context, axis=0), np.std(positive_context, axis=0))
                #print(target_agent.data_type, 'context', np.mean(negative_context, axis=0), np.std(negative_context, axis=0))
        print('forward loss:{:.7f}'.format(epoch_loss / self.pair_n))
        print('context loss', cmp_loss / self.pair_n)
        return epoch_loss / self.pair_n, cmp_loss / self.pair_n

    def cal_gan(self, real_action, trans_action, source_pool, target_pool, source_context, target_context):
        ######################
        # (1) Update D network
        ######################

        self.optimizer_d.zero_grad()

        fake_b = trans_action
        fake_ab = target_pool.query(fake_b.detach())
        pred_fake = self.dmodel(fake_ab)
        loss_d_fake = self.criterionGAN(pred_fake, target_context)

        real_b = real_action
        real_ab = source_pool.query(real_b)
        pred_real = self.dmodel(real_ab)
        loss_d_real = self.criterionGAN(pred_real, source_context)

        # Combined D loss
        loss_d = (loss_d_fake + loss_d_real) * 0.5
        loss_d.backward()
        self.optimizer_d.step()

        ######################
        # (2) Update G network
        ######################
        fake_ab = fake_b
        pred_fake = self.dmodel(fake_ab)
        loss_g_gan = self.criterionGAN(pred_fake, source_context)

        return loss_g_gan

    def train_agent(self, source_agent, target_agent, source_pool, target_pool):
        loss_fn = nn.L1Loss()
        epoch_loss, cmp_loss, gan_loss = 0, 0, 0
        for i in range(self.pair_n):
            item1 = source_agent.sample()
            if item1[0] is None:
                continue
            real_action, cp_obs, cp_act = item1[1], item1[3], item1[4]
            source_context_tensor = self.cmodel(cp_obs, cp_act).detach()

            item2 = target_agent.sample()
            if item2[0] is None:
                continue

            now_state, action, nxt_state, cp_obs, cp_act = item2
            target_context_tensor = self.cmodel(cp_obs, cp_act).detach()

            trans_action = self.model(now_state, action, source_context_tensor, target_context_tensor)

            out = self.fmodel(now_state, trans_action, source_context_tensor)
            loss_cycle = loss_fn(out, nxt_state)*20
            loss = loss_cycle

            back_action = self.back_model(now_state, trans_action, source_context_tensor, target_context_tensor)
            loss_back = loss_fn(back_action, action)
            loss += loss_back

            loss_g_gan = self.cal_gan(real_action, trans_action, source_pool, target_pool, source_context_tensor, target_context_tensor)
            loss_all = loss + loss_g_gan

            self.optimizer_g.zero_grad()
            loss_all.backward()
            self.optimizer_g.step()

            epoch_loss += loss_cycle.item()
            cmp_loss += loss_back.item()
            gan_loss += loss_g_gan.item()
            print('cycle_loss:{:.3f}  back_loss:{:.3f}  gan_loss:{:.3f}'.format(epoch_loss/self.pair_n, cmp_loss/self.pair_n, gan_loss/self.pair_n))
            return epoch_loss/self.pair_n, cmp_loss/self.pair_n, gan_loss/self.pair_n

    def train_ax(self, writer, t, shared_policy):
        for i, source in enumerate(self.data_type):
            for j, target in enumerate(self.data_type):
                forward_loss, context_loss = self.train_context_forward(writer, t, source_agent=self.nested_agents[i], target_agent=self.nested_agents[j])
                name = 'arma%s-arma%s'%(source, target)
                writer.add_scalar('%s/forward_loss'%name, forward_loss, t+1)
                writer.add_scalar('%s/context_loss'%name, context_loss, t+1)

        for i, source in enumerate(self.data_type):
            for j, target in enumerate(self.data_type):
                cycle_loss, back_loss, gan_loss = self.train_agent(self.nested_agents[i], self.nested_agents[j], source_pool=self.pools[i], target_pool=self.pools[j])
                name = 'arma%g-arma%g'%(source, target)
                writer.add_scalar('%s/cycle_loss'%name, cycle_loss, t+1)
                writer.add_scalar('%s/back_loss'%name, back_loss, t+1)
                writer.add_scalar('%s/gan_generator_loss'%name, gan_loss, t+1)

                item = self.nested_agents[i].sample()
                cp_obs, cp_act = item[3], item[4]
                source_context_tensor = self.cmodel(cp_obs, cp_act).detach()
                reward = self.nested_agents[j].online_test(self.back_model, self.cmodel, 10, source_context_tensor, shared_policy=shared_policy)
                writer.add_scalar('%s/shared_policy_reward'%name, reward.mean(), t+1)
                self.transfer_rewards[i][j] = reward.mean()

    def good_to_transfer(self, target_context, target_rewards, threshold):
        target_context_index = self.data_type.index(target_context[0])
        max_transfer_reward = np.max(self.transfer_rewards[:,target_context_index])
        if max_transfer_reward > target_rewards[target_context_index]+abs(target_rewards[target_context_index])*threshold:
            source_context_index = np.argmax(self.transfer_rewards[:,target_context_index])
            source_context = self.data_type[source_context_index]
            return source_context
        else:
            return None

    def select_action(self, state, source_context, target_context_tensor, shared_policy):
        state_tensor = torch.FloatTensor(state.reshape(1, -1)).cuda()
        source_context_index = self.data_type.index(source_context)
        
        item = self.nested_agents[source_context_index].sample(batch_size=1)
        cp_obs, cp_act = item[3], item[4]
        source_context_tensor = self.cmodel(cp_obs, cp_act).detach()
        good_action =shared_policy.select_action(state, source_context_tensor)
        good_action = torch.FloatTensor(good_action).reshape(1,-1).cuda()
        action = self.back_model(state_tensor, good_action, source_context_tensor, target_context_tensor).cpu().data.numpy().flatten()
        return action


