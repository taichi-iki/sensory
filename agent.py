# coding: utf-8

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import numpy as np
from collections import deque

import chainer
import chainer.functions as F
import chainer.links as L
from chainer import serializers
from chainer import optimizers
from chainer.optimizer import WeightDecay

from environment import Environment

class Incident(object):
    def __init__(self, s=None, a=None, r=None, ns=None):
        self.s = s
        self.a = a
        self.r = r
        self.ns = ns

class QNet(chainer.Chain):
    def __init__(self,
            hidden_dim,
            action_count,
            input_ch=3,
            input_shape=(8, 8),
        ):
        conv_factor_per_axis = int(2**2)
        after_conv_dim = int(input_shape[0]*input_shape[1]/(conv_factor_per_axis**2))
        super(QNet, self).__init__(
                conv1=L.Convolution2D(input_ch, 16, 3, stride=1, pad=1),
                conv2=L.Convolution2D(16, 32, 3, stride=1, pad=1),
                l3=L.Linear(32*after_conv_dim, hidden_dim),
                l4=L.Linear(hidden_dim, action_count),
            )
    
    def __call__(self, x, train):
        y = x
        # 3x8x8
        y = F.relu(self.conv1(y))
        y = F.max_pooling_2d(y, 2)
        # 16x4x4
        y = F.relu(self.conv2(y))
        y = F.max_pooling_2d(y, 2)
        # 32x2x2
        y = F.relu(self.l3(y))
        y = F.dropout(y, 0.5, train=train)
        y = self.l4(y)
        return y

class DQNAgent(object):
    def __init__(self):
        self.action_count = 5
        self.net = QNet(
                hidden_dim=32,
                action_count=self.action_count,
            )
        self.optimizer = optimizers.Adam(alpha=0.001)
        self.optimizer.setup(self.net)
        self.optimizer.add_hook(chainer.optimizer.WeightDecay(rate=0.00001))
        self.replay_memory = deque([], 1000)
        self.rl_max_minibatch = 64
        self.rl_gamma = 0.9
        self.incident = None
        self.exploration_rate = 1.0
        self.exploration_rate_delta = -0.05
        self.exploration_rate_min = 0.0
        self.exploration_rate_max = 1.0
        self.action_to_char=['・', '↓', '↑', '→', '←']
        self.no_update = False
        self.train_mode = True
        self.total_call = 0

    def select_action(self, x):
        p = np.random.uniform(0.0, 1.0)
        if p < self.exploration_rate:
            a = np.random.randint(0, self.action_count)
        else:
            q = self.net(x, train=False)
            a = q.data.argmax()
        return a
    
    def update_net(self, replay_memory):
        xp = self.net.xp
        mb = replay_memory if len(replay_memory) <= self.rl_max_minibatch else \
               [replay_memory[i] for i in np.random.choice(
               range(len(replay_memory)), self.rl_max_minibatch)]
        mb_size = len(mb)
        output_dim = self.action_count 
        if mb_size > 0:
            s = xp.concatenate([m.s for m in mb], axis=0).astype('float32')
            a = xp.asarray([m.a for m in mb], dtype='int32')
            r = xp.asarray([m.r for m in mb], dtype='int32')
            ns = xp.concatenate([m.ns for m in mb], axis=0).astype('float32')
            
            future_r = self.net(ns, train=True).data.max(axis=1)
            t_signal = r + self.rl_gamma*future_r
            a_mask = xp.broadcast_to(xp.arange(0, output_dim)[None, :],
                (mb_size, output_dim))
            a_mask = (a_mask == a[:, None])
            q_selected = F.sum(self.net(s, train=True) * a_mask, axis=1)
            loss = F.sum((q_selected - t_signal)**2)/mb_size
            
            self.net.zerograds()
            loss.backward()
            self.optimizer.update()

    def action_to_pos(self, s, a):
        px, py = s['pos']
        dx = dy = 0
        if a == 1:
    	    dx = +1
        if a == 2:
    	    dx = -1
        if a == 3:
    	    dy = +1
        if a == 4:
    	    dy = -1
        return (px+dx, py+dy)
    
    def next_step(self, s):
        s_input = np.concatenate([
            s['input1'][None, None, :, :],
            s['input2'][None, None, :, :],
            s['input3'][None, None, :, :]], axis=1
        )
        a = self.select_action(s_input)
        
        if not self.incident is None:
            print('%8d\t%s\t%.3f\t%.3f'%(
                self.total_call,
                self.action_to_char[self.incident.a],
                self.incident.r, 
                self.exploration_rate))
            self.incident.ns = s_input
            self.replay_memory.append(self.incident)
            if not self.no_update:
                self.update_net(self.replay_memory)
        
        self.incident = Incident(s=s_input, a=a)

        self.total_call += 1
        
        return self.action_to_pos(s, a)

    def reward(self, r):
        if not self.incident is None:
            self.incident.r = r
    
    def reset(self):
        self.exploration_rate += self.exploration_rate_delta
        self.exploration_rate = max(self.exploration_rate_min, self.exploration_rate)
        self.exploration_rate = min(self.exploration_rate_max, self.exploration_rate)

    def save_net(self, path):
        serializers.load_npz(self.net, path)

    def load_net(self, path):
        serializers.save_npz(self.net, path)

if __name__ == '__main__':
    env = Environment()
    env.params['step_max']=1000
    env.agent = DQNAgent()
    env.iterate(10000)
    agent.save_net('learned.npz')
    env.demonstrate(10)
