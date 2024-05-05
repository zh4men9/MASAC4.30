from collections import namedtuple
import numpy as np
import torch
from torch import optim
import torch.nn as nn
from utilities.util import *
from utilities.replay_buffer import *
from utilities.inspector import *
from arguments import *
from utilities.logger import Logger



class PGTrainer(object):

    def __init__(self, args, model, env, logger, online):
        self.args = args
        self.cuda_ = self.args.cuda and torch.cuda.is_available()
        self.logger = logger
        self.online = online
        inspector(self.args)
        if self.args.target:
            target_net = model(self.args).cuda() if self.cuda_ else model(self.args)
            self.behaviour_net = model(self.args, target_net).cuda() if self.cuda_ else model(self.args, target_net)
        else:
            self.behaviour_net = model(self.args).cuda() if self.cuda_ else model(self.args)
        if self.args.replay:
            if self.online:
                self.replay_buffer = TransReplayBuffer(int(self.args.replay_buffer_size))
            else:
                self.replay_buffer = EpisodeReplayBuffer(int(self.args.replay_buffer_size))
        self.env = env
        self.action_optimizers = []
        for action_dict in self.behaviour_net.action_dicts:
            self.action_optimizers.append(optim.Adam(action_dict.parameters(), lr=args.policy_lrate))
        self.value_optimizers = []
        for value_dict in self.behaviour_net.value_dicts:
            self.value_optimizers.append(optim.Adam(value_dict.parameters(), lr=args.value_lrate))
        initial_values = [[np.log(self.args.init_temperature) for _ in range(self.args.agent_num)]]
        self.log_alphas = cuda_wrapper(torch.tensor(initial_values, dtype=torch.float32), cuda=self.cuda_)
        self.log_alphas.requires_grad = True
        #self.alphas = self.log_alphas.exp()
        self.log_alpha_optimizers = torch.optim.Adam([self.log_alphas], lr=args.alpha_lrate)
        self.init_action = cuda_wrapper( torch.zeros(1, self.args.agent_num, self.args.action_dim), cuda=self.cuda_ )
        self.target_entropy = -self.args.action_dim
        self.steps = 0
        self.episodes = 0
        self.mean_reward = 0
        self.mean_success = 0
        self.entr = self.args.entr
        self.entr_inc = self.args.entr_inc

    def get_loss(self, batch):
        action_loss, value_loss, log_p_a = self.behaviour_net.get_loss(batch, self.log_alphas.exp())
        return action_loss, value_loss, log_p_a

    def action_compute_grad(self, stat, loss, retain_graph):
        action_loss, log_p_a = loss
        action_loss.backward(retain_graph=retain_graph)

    def value_compute_grad(self, value_loss, retain_graph):
        value_loss.backward(retain_graph=retain_graph)

    def grad_clip(self, params):
        for param in params:
            param.grad.data.clamp_(-1, 1)

    def action_replay_process(self, stat):
        batch = self.replay_buffer.get_batch(self.args.batch_size)
        batch = self.behaviour_net.Transition(*zip(*batch))
        self.action_transition_process(stat, batch)

    def value_replay_process(self, stat):
        batch = self.replay_buffer.get_batch(self.args.batch_size)
        batch = self.behaviour_net.Transition(*zip(*batch))
        self.value_transition_process(stat, batch)
        
    def alpha_replay_process(self, stat):
        batch = self.replay_buffer.get_batch(self.args.batch_size)
        batch = self.behaviour_net.Transition(*zip(*batch))
        self.alpha_transition_process(stat, batch)
        
    def action_transition_process(self, stat, trans):
        action_loss, value_loss, log_p_a = self.get_loss(trans)
        policy_grads = []
        for i in range(self.args.agent_num):
            retain_graph = False if i == self.args.agent_num-1 else True
            action_optimizer = self.action_optimizers[i]
            action_optimizer.zero_grad()
            self.action_compute_grad(stat, (action_loss[i], log_p_a[:, i]), retain_graph)
            grad = []
            for pp in action_optimizer.param_groups[0]['params']:
                grad.append(pp.grad.clone())
            policy_grads.append(grad)
        policy_grad_norms = []
        for action_optimizer, grad in zip(self.action_optimizers, policy_grads):
            param = action_optimizer.param_groups[0]['params']
            for i in range(len(param)):
                param[i].grad = grad[i]
            if self.args.grad_clip:
                self.grad_clip(param)
            policy_grad_norms.append(get_grad_norm(param))
            action_optimizer.step()
        stat['policy_grad_norm'] = np.array(policy_grad_norms).mean()
        stat['action_loss'] = action_loss.mean().item()

    def value_transition_process(self, stat, trans):
        action_loss, value_loss, log_p_a = self.get_loss(trans)
        value_grads = []
        for i in range(self.args.agent_num):
            retain_graph = False if i == self.args.agent_num-1 else True
            value_optimizer = self.value_optimizers[i]
            value_optimizer.zero_grad()
            self.value_compute_grad(value_loss[i], retain_graph)
            grad = []
            for pp in value_optimizer.param_groups[0]['params']:
                grad.append(pp.grad.clone())
            value_grads.append(grad)
        value_grad_norms = []
        for value_optimizer, grad in zip(self.value_optimizers, value_grads):
            param = value_optimizer.param_groups[0]['params']
            for i in range(len(param)):
                param[i].grad = grad[i]
            if self.args.grad_clip:
                self.grad_clip(param)
            value_grad_norms.append(get_grad_norm(param))
            value_optimizer.step()
        stat['value_grad_norm'] = np.array(value_grad_norms).mean()
        stat['value_loss'] = value_loss.mean().item()
        
    def alpha_transition_process(self, stat, trans):
        # 计算alpha_loss
        action_loss, value_loss, log_p_a = self.get_loss(trans)
        alpha_loss = (self.log_alphas.exp() *
                      (-log_p_a - self.target_entropy).detach()).mean()

        # 对每个alpha进行优化
        self.log_alpha_optimizers.zero_grad()
        alpha_loss.backward()

        # 如果你需要梯度裁剪，可以调用下面的函数
        if self.args.grad_clip:
            self.grad_clip(self.log_alpha_optimizers.param_groups[0]['params'])

        # 计算梯度范数 (如果需要的话)
        alpha_grad_norm = get_grad_norm(self.log_alpha_optimizers.param_groups[0]['params'])

        # 更新alpha参数
        self.log_alpha_optimizers.step()
        # 更新统计数据
        stat['alpha_grad_norm'] = alpha_grad_norm
        stat['alpha_loss'] = alpha_loss.item()
        
    def run(self, stat):
        self.behaviour_net.train_process(stat, self)
        self.entr += self.entr_inc

    def logging(self, stat):
        for tag, value in stat.items():
            if isinstance(value, np.ndarray):
                self.logger.image_summary(tag, value, self.episodes)
            else:
                self.logger.scalar_summary(tag, value, self.episodes)

    def print_info(self, stat):
        action_loss = stat.get('action_loss', 0)
        value_loss = stat.get('value_loss', 0)
        entropy = stat.get('alpha_loss', 0)
        print('Episode: {:4d}, Mean Reward: {:2.4f}, Action Loss: {:2.4f}, Value Loss is: {:2.4f}, alpha_loss: {:2.4f}\n' \
              .format(self.episodes, stat['mean_reward'], action_loss , value_loss, stat['alpha_loss']))