# coding: utf-8

import numpy as np

matplotlib_available=True
try:
    import matplotlib.pyplot as plt
    import matplotlib.patches as patches
    import matplotlib.animation as animation
except:
    matplotlib_available=False

class Environment(object):
    def __init__(self, map_size=(60, 90), input_size=(8, 8), call_reset=True):
        self.params = {
                 'space_size': map_size,
                 'agent_size': input_size,
                 'gamma1': 0.95,
                 'gamma2': 0.02,
                 'gamma3': 0.005,
                 'step_max': 10,
            }
        self.matplotlib_available = matplotlib_available
        self.agent = None
        self.call_reset = call_reset
        self.reset_state()
    
    def is_in(self, goal, p):
        if p[0] < goal[0]: return False
        if p[0] > goal[2]: return False
        if p[1] < goal[1]: return False
        if p[1] > goal[3]: return False
        return True
    
    def is_solved(self, s):
        px, py = s['pos']
        sx, sy = self.params['agent_size']
        return self.is_in(s['goal'], (px, py)) or \
            self.is_in(s['goal'], (px+sx, py)) or \
            self.is_in(s['goal'], (px, py+sy)) or \
            self.is_in(s['goal'], (px+sx, py+sy))
    
    def get_local_img(self, s, agent_size):
	    x, y = s['pos']
	    return s['map'][x:x+agent_size[0], y:y+agent_size[1]]

    def make_space(self, shape, seed=None):
        rand = np.random.RandomState(seed)
        m = np.zeros(shape).astype('float32')
        mx, my = m.shape
        ux = mx//4
        uy = my//3
        x = np.arange(0, uy)/uy
        r = rand.uniform(0.0, 1.0)
        x = (0.5*(np.sin(x*(0.25+0.75*r)*2*3.14*20)+1.0)).astype('float32')
        m[ux:3*ux,:uy] = x[None, :]
        m[:, uy:2*uy] = np.minimum(1, rand.uniform(0, 2, (mx, uy)))
        m[0:ux, 2*uy:] = 1.0
        m[3*ux:,2*uy:] = 1.0
        if r < 0.5:
            goal_rect = (0, 2*uy, ux, 3*uy)
        else:
        	goal_rect = (3*ux, 2*uy, 4*ux, 3*uy)
        return m, goal_rect
    
    def reset_state(self, with_screen_operation=False):
        new_space, goal_rect = self.make_space(self.params['space_size'])
        s = {
                'step': 0,
                'pos': ((self.params['space_size'][0] - self.params['agent_size'][0])//2, 0),
                'input1': np.zeros(self.params['agent_size']).astype('float32'),
                'input2': np.zeros(self.params['agent_size']).astype('float32'),
                'input3': np.zeros(self.params['agent_size']).astype('float32'),
                'map': new_space,
                'goal': goal_rect,
    	    }
        
        if with_screen_operation:
            self.im4.set_array(s['map'])
            self.answer_rect.set_y(s['goal'][0])

        if (not self.agent is None) and self.call_reset:
            self.agent.reset()
        
        self.state = s
    
    def update_state(self, data, with_screen_operation=False):
        if self.is_solved(self.state):
            self.agent.reward(+1.0)
            print('solved')
            self.reset_state(with_screen_operation)
        else:
            self.agent.reward(+0.0)
        
        if self.state['step'] >= self.params['step_max']:
            self.reset_state(with_screen_operation)
        # moving
        new_pos_candidate = self.agent.next_step(self.state)
        if new_pos_candidate[0] >= 0 and \
            new_pos_candidate[0] + self.params['agent_size'][0] < self.params['space_size'][0] and \
            new_pos_candidate[1] >= 0 and \
            new_pos_candidate[1] + self.params['agent_size'][1] < self.params['space_size'][1]:
            self.state['pos'] = new_pos_candidate
        
        # inputs change
        img = self.get_local_img(self.state, self.params['agent_size'])
        self.state['input1'] = self.params['gamma1']*img + (1.0-self.params['gamma1'])*self.state['input1']
        self.state['input2'] = self.params['gamma2']*img + (1.0-self.params['gamma2'])*self.state['input2']
        self.state['input3'] = self.params['gamma3']*img + (1.0-self.params['gamma3'])*self.state['input3']
        
        if with_screen_operation:
            self.im1.set_array(self.state['input1'])
            self.im2.set_array(self.state['input2'])
            self.im3.set_array(self.state['input3'])
            self.scope_rect.set_xy((self.state['pos'][1], self.state['pos'][0]))
        self.state['step'] += 1
    
    def iterate(self, iteration_count):
        for i in range(iteration_count):
            self.update_state(None, with_screen_operation=False)
    
    def demonstrate(self, interval=100):
        self.reset_state()
        self.state['input1'][0,0]=1.0
        self.state['input2'][0,0]=1.0
        self.state['input3'][0,0]=1.0
        map_00 = self.state['map'][0,0]
        self.state['map'][0,0]=1.0
        fig = plt.figure()
    
        # input1
        sp1 = plt.subplot2grid((3, 3), (0, 0))
        sp1.set_xticklabels([])
        sp1.set_yticklabels([])
        self.im1 = sp1.imshow(self.state['input1'], cmap='gray')
        
        # input2
        sp2 = plt.subplot2grid((3, 3), (0, 1))
        sp2.set_yticklabels([])
        sp2.set_xticklabels([])
        self.im2 = sp2.imshow(self.state['input2'], animated=True, cmap='gray')
        
        # input3
        sp3 = plt.subplot2grid((3, 3), (0, 2))
        sp3.set_xticklabels([])
        sp3.set_yticklabels([])
        self.im3 = sp3.imshow(self.state['input3'], animated=True, cmap='gray')
        
        # whole map
        sp4 = plt.subplot2grid((3, 3), (1, 0), rowspan=2, colspan=3)
        sp4.set_xticklabels([])
        sp4.set_yticklabels([])
        self.im4 = sp4.imshow(self.state['map'], animated=True, cmap='gray')
        
        # indicator
        self.scope_rect = patches.Rectangle(
                (self.state['pos'][1], self.state['pos'][0]),  
                self.params['agent_size'][0], self.params['agent_size'][1], fill=False, edgecolor="blue"
            )
        sp4.add_patch(self.scope_rect)
        self.answer_rect = patches.Rectangle(
                (self.state['goal'][1], self.state['goal'][0]),  
                self.params['space_size'][1]//3-1, self.params['space_size'][0]//4-1, fill=False, edgecolor="green"
            )
        sp4.add_patch(self.answer_rect)
        
        self.state['input1'][0, 0]=0.0
        self.state['input2'][0, 0]=0.0
        self.state['input3'][0, 0]=0.0
        self.state['map'][0,0]=map_00
        
        f = lambda x: self.update_state(x, with_screen_operation=True)
        ani = animation.FuncAnimation(fig, f, interval=interval)
        plt.show()

