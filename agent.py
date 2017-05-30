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

from env import Environment

class Incident(object):
    def __init__(self, s=None, a=None, r=None, ds_sum=None, ns=None):
        self.s = s
        self.a = a
        self.r = r
        self.ds_sum = ds_sum
        self.ns = ns

class QNet(chainer.Chain):
    def __init__(self,
            hidden_dim,
            action_count,
            action_dim=10,
            input_ch=3,
            input_shape=(8, 8),
        ):
        conv_factor_per_axis = int(2**2)
        after_conv_dim = int(input_shape[0]*input_shape[1]/(conv_factor_per_axis**2))
        super(QNet, self).__init__(
                action_embed=L.EmbedID(action_count, action_dim),
                com_conv1=L.Convolution2D(input_ch, 16, 3, stride=1, pad=1),
                com_conv2=L.Convolution2D(16, 32, 3, stride=1, pad=1),
                l1=L.Linear(32*after_conv_dim + action_dim, hidden_dim),
                q_out=L.Linear(hidden_dim, 1),
                p_out=L.Linear(hidden_dim, input_ch*input_shape[0]*input_shape[1]),
            )
        self.input_shape = input_shape
        self.input_ch = input_ch
    
    def image_feature(self, x):
        y = x
        # 3x8x8
        y = F.relu(self.com_conv1(y))
        y = F.max_pooling_2d(y, 2)
        # 16x4x4
        y = F.relu(self.com_conv2(y))
        y = F.max_pooling_2d(y, 2)
        y = F.reshape(y, (y.data.shape[0], -1))
        return y
    
    def __call__(self, x, a, train, prediction=False):
        h = F.concat([self.image_feature(x), self.action_embed(a)], axis=1)
        h = F.dropout(h, 0.5, train=train)
        h = F.reshape(F.relu(self.l1(h)), (a.shape[0], -1))
        q = self.q_out(h)
        p = None
        if prediction:
            p = F.sigmoid(self.p_out(h))
            p = F.reshape(p, (-1, self.input_ch, self.input_shape[0], self.input_shape[1]))
        return q, p

class DQNAgent(object):
    def __init__(self):
        self.action_count = 5
        self.gpuid = 0
        self.net = QNet(
                hidden_dim=32,
                action_count=self.action_count,
            )
        if self.gpuid >= 0:
            self.net.to_gpu(self.gpuid)
        self.optimizer = optimizers.SGD(lr=0.1)
        self.optimizer.setup(self.net)
        self.optimizer.add_hook(chainer.optimizer.WeightDecay(rate=0.00001))
        self.replay_memory = deque([], 1000)
        self.rl_max_minibatch = 64
        self.rl_gamma = 0.9
        self.incident = None
        self.exploration_rate = 0.0
        self.exploration_rate_delta = -0.1
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
            s = [self.action_count] + list(x.shape)[1:]
            x = self.net.xp.broadcast_to(self.net.xp.asarray(x), s)
            a = self.net.xp.arange(0, 5).astype('int32')
            q, _ = self.net(x, a, train=False, prediction=False)
            a = int(q.data.argmax())
        return a
    
    def update_net(self, replay_memory):
        xp = self.net.xp
        mb = replay_memory if len(replay_memory) <= self.rl_max_minibatch else \
               [replay_memory[i] for i in np.random.choice(
               range(len(replay_memory)), self.rl_max_minibatch)]
        mb_size = len(mb)
        output_dim = self.action_count 
        if mb_size > 0:
            s = np.concatenate([m.s for m in mb], axis=0).astype('float32')
            s = xp.asarray(s)
            a = xp.asarray([m.a for m in mb], dtype='int32')
            r = xp.asarray([m.r for m in mb], dtype='int32')
            ns = np.concatenate([m.ns for m in mb], axis=0).astype('float32')
            ns = xp.asarray(ns)
            
            nq, _ = self.net(ns, a, train=True, prediction=False)
            nq = nq.data[:, 0]
            t_signal = r + self.rl_gamma*nq
            q, p = self.net(s, a, train=True, prediction=True)
            pixel_count = ns.shape[0]*ns.shape[1]*ns.shape[2]*ns.shape[3]
            loss = F.sum(abs(q[:, 0] - t_signal))/mb_size + F.sum(abs(p - ns))/pixel_count
            
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
            self.incident.ns = s_input
            #s_bar = self.net.pred(self.net.xp.asarray(self.incident.s), self.net.xp.asarray([self.incident.a], dtype='int32'), train=False)
            #ds_sum = F.sum(abs(s_bar - self.net.xp.asarray(self.incident.ns)))/(s_input.shape[0]*s_input.shape[1]*s_input.shape[2]*s_input.shape[3])
            #self.incident.ds_sum = float(ds_sum.data)
            self.incident.ds_sum = 0.0
            #self.incident.r += float(ds_sum.data)
            self.replay_memory.append(self.incident)
            
            if not self.no_update:
                self.update_net(self.replay_memory)
            #    self.net.zerograds()
            #    ds_sum.backward()
            #    self.optimizer.update()
            
            log_string='%8d\t%s\t%.3f\t%.3f\t%.3f'%(
                self.total_call,
                self.action_to_char[self.incident.a],
                self.incident.r,
                self.incident.ds_sum,
                self.exploration_rate)
            print(log_string)
            with open('log.txt', 'a') as f:
                f.write(log_string + '\n')
        
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
        self.net.to_cpu()
        serializers.save_npz(path, self.net)
        if self.gpuid >= 0:
            self.net.to_gpu(self.gpuid)

    def load_net(self, path):
        serializers.load_npz(path, self.net)

if __name__ == '__main__':
    env = Environment()
    env.params['step_max']=1000
    env.agent = DQNAgent()
    env.iterate(10000)
    env.agent.save_net('learned.npz')
    if env.matplotlib_available:
        env.demonstrate(10)

