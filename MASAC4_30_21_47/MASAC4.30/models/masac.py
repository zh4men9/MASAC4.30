import torch
import torch.nn as nn
import numpy as np
from utilities.util import *
from models.model import Model
from collections import namedtuple



class MASAC(Model):

    def __init__(self, args, target_net=None):
        super(MASAC, self).__init__(args)
        self.construct_model()
        self.apply(self.init_weights)
        if target_net != None:
            self.target_net = target_net
            self.reload_params_to_target()
        self.sample_size = self.args.sample_size
        self.Transition = namedtuple('Transition', ('state', 'action', 'reward', 'next_state', 'done', 'last_step'))

    def unpack_data(self, batch):
        batch_size = len(batch.state)
        rewards = cuda_wrapper(torch.tensor(batch.reward, dtype=torch.float), self.cuda_)
        last_step = cuda_wrapper(torch.tensor(batch.last_step, dtype=torch.float).contiguous().view(-1, 1), self.cuda_)
        done = cuda_wrapper(torch.tensor(batch.done, dtype=torch.float).contiguous().view(-1, 1), self.cuda_)
        actions = cuda_wrapper(torch.tensor(np.stack(list(zip(*batch.action))[0], axis=0), dtype=torch.float), self.cuda_)
        state = cuda_wrapper(prep_obs(list(zip(batch.state))), self.cuda_)
        next_state = cuda_wrapper(prep_obs(list(zip(batch.next_state))), self.cuda_)
        return (rewards, last_step, done, actions, state, next_state)

    def construct_policy_net(self):
        action_dicts = []
        if self.args.shared_parameters:
            l1 = nn.Linear(self.obs_dim, self.hid_dim)
            l2 = nn.Linear(self.hid_dim, self.hid_dim)
            a = nn.Linear(self.hid_dim, self.act_dim)
            for i in range(self.n_):
                action_dicts.append(nn.ModuleDict( {'layer_1': l1,\
                                                    'layer_2': l2,\
                                                    'action_head': a
                                                    }
                                                 )
                                   )
        else:
            for i in range(self.n_):
                action_dicts.append(nn.ModuleDict( {'layer_1': nn.Linear(self.obs_dim, self.hid_dim),\
                                                    'layer_2': nn.Linear(self.hid_dim, self.hid_dim),\
                                                    'action_head': nn.Linear(self.hid_dim, self.act_dim)
                                                    }
                                                  )
                                   )
        self.action_dicts = nn.ModuleList(action_dicts)

    def construct_value_net(self):
        value_dicts = []
        if self.args.shared_parameters:
            l1 = nn.Linear( (self.obs_dim+self.act_dim)*self.n_, self.hid_dim )
            l2 = nn.Linear(self.hid_dim, self.hid_dim)
            v = nn.Linear(self.hid_dim, 1)
            for i in range(self.n_):
                value_dicts.append(nn.ModuleDict( {'layer_1': l1,\
                                                   'layer_2': l2,\
                                                   'value_head': v
                                                  }
                                                )
                                  )
        else:
            for i in range(self.n_):
                value_dicts.append(nn.ModuleDict( {'layer_1': nn.Linear( (self.obs_dim+self.act_dim)*self.n_, self.hid_dim ),\
                                                   'layer_2': nn.Linear(self.hid_dim, self.hid_dim),\
                                                   'value_head': nn.Linear(self.hid_dim, 1)
                                                  }
                                                )
                                  )
        self.value_dicts = nn.ModuleList(value_dicts)

    def construct_model(self):
        self.construct_value_net()
        self.construct_policy_net()

    def policy(self, obs, schedule=None, last_act=None, last_hid=None, info={}, stat={}):
        actions = []
        for i in range(self.n_):
            h = torch.relu( self.action_dicts[i]['layer_1'](obs[:, i, :]) )
            h = torch.relu( self.action_dicts[i]['layer_2'](h) )
            a = self.action_dicts[i]['action_head'](h)
            actions.append(a)
        actions = torch.stack(actions, dim=1)
        return actions

    # def sample_grandcoalitions(self, batch_size):
    #     seq_set = cuda_wrapper(torch.tril(torch.ones(self.n_, self.n_), diagonal=0, out=None), self.cuda_)
    #     grand_coalitions = cuda_wrapper(torch.multinomial(torch.ones(batch_size*self.sample_size, self.n_)/self.n_, self.n_, replacement=False), self.cuda_)
    #     individual_map = cuda_wrapper(torch.zeros(batch_size*self.sample_size*self.n_, self.n_), self.cuda_)
    #     individual_map.scatter_(1, grand_coalitions.contiguous().view(-1, 1), 1)
    #     individual_map = individual_map.contiguous().view(batch_size, self.sample_size, self.n_, self.n_)
    #     subcoalition_map = torch.matmul(individual_map, seq_set)
    #     grand_coalitions = grand_coalitions.unsqueeze(1).expand(batch_size*self.sample_size, self.n_, self.n_).contiguous().view(batch_size, self.sample_size, self.n_, self.n_) # shape = (b, n_s, n, n)
    #     return subcoalition_map, grand_coalitions
    def value(self, obs, act):
        # TODO: policy params update
        values = []
        for i in range(self.n_):
            h = torch.relu( self.value_dicts[i]['layer_1']( torch.cat( ( obs.contiguous().view( -1, np.prod(obs.size()[1:]) ), act.contiguous().view( -1, np.prod(act.size()[1:]) ) ), dim=-1 ) ) )
            h = torch.relu( self.value_dicts[i]['layer_2'](h) )
            v = self.value_dicts[i]['value_head'](h)
            values.append(v)
        values = torch.stack(values, dim=1)
        return values

    def get_loss(self, batch, alphas):  # batch是一个包含了状态、动作、奖励等信息的数据集
        # 计算策略损失和价值损失、这里的关键是使用了Shapley值
        batch_size = len(batch.state)
        n = self.args.agent_num
        action_dim = self.args.action_dim
        rewards, last_step, done, actions, state, next_state = self.unpack_data(batch)
        # (32, 3),(32,1),(32,1),(32,3,5),(32,3,18),(32,3,18)
        # do torche argmax action on torche action loss
        action_out = self.policy(state) # 根据当前观测state得到一个动作均值
        # 得到动作分布和确定动作
        action_dist, actions_ = select_act(self.args, action_out, status='train') # 归一化后的动作概率分布
        # 计算当前动作在其分布中的对数概率
        # log_pi_a (32,3)
        log_pi_a = action_dist.log_prob(actions_).sum(-1)
        # shapley_Q(32,3)
        values_ = self.value(state, actions_).contiguous().view(-1, self.n_)
        advantages = values_-alphas * log_pi_a
        if self.args.normalize_advantages:
            advantages = batchnorm(advantages)
        action_loss = -advantages
        action_loss = action_loss.mean(dim=0)
        # do torche exploration action on torche value loss
        values = self.value(state, actions).contiguous().view(-1, self.n_)
        # do torche argmax action on torche next value loss
        if self.args.target:
            next_action_out = self.target_net.policy(next_state)
        else:
            next_action_out = self.policy(next_state)
        next_action_dist, next_actions_ = select_act(self.args, next_action_out, status='train')
        log_pi_a_next = next_action_dist.log_prob(next_actions_).sum(-1)
        if self.args.target:
            next_values_ = self.target_net.value(next_state, next_actions_.detach()).contiguous().view(-1, self.n_)
        else:
            next_values_ = self.value(next_state, next_actions_.detach()).contiguous().view(-1, self.n_)
        returns = cuda_wrapper(torch.zeros((batch_size, n), dtype=torch.float), self.cuda_) # 一个形状为(batch_size,n)的全零张量
        assert values.size() == next_values_.size()
        assert returns.size() == values.size()
        for i in reversed(range(rewards.size(0))):
            if last_step[i]:
                next_return1 = 0 if done[i] else next_values_[i].detach()
                next_return2 = 0 if done[i] else log_pi_a_next[i].detach()
            else:
                next_return1 = next_values_[i].detach()
                next_return2 = 0 if done[i] else log_pi_a_next[i].detach()
            returns[i] = rewards[i] + self.args.gamma * (next_return1-alphas * next_return2)
        deltas = returns - values
        value_loss = deltas.pow(2).mean(dim=0)
        return action_loss, value_loss, log_pi_a

    def train_process(self, stat, trainer):
        info = {}
        state = trainer.env.reset()
        if self.args.reward_record_type is 'episode_mean_step':
            trainer.mean_reward = 0
            trainer.mean_success = 0
        for t in range(self.args.max_steps):
            state_ = cuda_wrapper(prep_obs(state).contiguous().view(1, self.n_, self.obs_dim), self.cuda_)
            start_step = True if t == 0 else False
            state_ = cuda_wrapper(prep_obs(state).contiguous().view(1, self.n_, self.obs_dim), self.cuda_)
            action_out = self.policy(state_, info=info, stat=stat)
            _, action = select_act(self.args, action_out, status='train', info=info)
            _, actual = translate_action(self.args, action, trainer.env)
            next_state, reward, done, debug = trainer.env.step(actual)
            if isinstance(done, list): done = np.sum(done)
            done_ = done or t==self.args.max_steps-1
            trans = self.Transition(state,
                                    action.cpu().numpy(),
                                    np.array(reward),
                                    next_state,
                                    done,
                                    done_
                                   )
            self.transition_update(trainer, trans, stat)
            success = debug['success'] if 'success' in debug else 0.0
            trainer.steps += 1
            if self.args.reward_record_type is 'mean_step':
                trainer.mean_reward = trainer.mean_reward + 1/trainer.steps*(np.mean(reward) - trainer.mean_reward)
                trainer.mean_success = trainer.mean_success + 1/trainer.steps*(success - trainer.mean_success)
            elif self.args.reward_record_type is 'episode_mean_step':
                trainer.mean_reward = trainer.mean_reward + 1/(t+1)*(np.mean(reward) - trainer.mean_reward)
                trainer.mean_success = trainer.mean_success + 1/(t+1)*(success - trainer.mean_success)
            else:
                raise RuntimeError('Please enter a correct reward record type, e.g. mean_step or episode_mean_step.')
            stat['mean_reward'] = trainer.mean_reward
            stat['mean_success'] = trainer.mean_success
            if done_:
                break
            state = next_state
        stat['turn'] = t+1
        trainer.episodes += 1
