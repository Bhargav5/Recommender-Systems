"""
Wavefollower environment includes to create a sine wave as an environment
and reward system so that agent learn to follow the sine wave.
"""

import logging
import math
import gym
from gym import spaces
from gym.utils import seeding
import numpy as np

logger = logging.getLogger(__name__)

class WavefollowerEnv(gym.Env):
    metadata = {
        'render.modes': ['human', 'rgb_array'],
        'video.frames_per_second' : 50
    }

    def __init__(self):

        self.freq = 1
        #high = np.array([[10000,1,1],[10000,1,1],[100000,1,1],[100000,1,1]])

        self.action_space = spaces.Discrete(9)
        self.actions = [-1,-0.5,-0.1,-0.01,0,0.01,0.1,0.5,1]
        self.observation_space = spaces.Box(np.array([0,-1,-1]),np.array([10000,1,1]))

        self._seed()
        self.viewer = None
        self.state = None

        self.steps_beyond_done = None

    def _seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def _step(self, action):
        assert self.action_space.contains(action), "%r (%s) invalid"%(action, type(action))
        # Logic to update the state
        state = self.state
        time, sin_value, agent_val = state
        time_n= time + 1
        sin_value_n = np.sin(time_n * self.freq * np.pi / 180)
        agent_val_n = agent_val + self.actions[action]
        #del(state[0])
        #state.append([time_n, sin_value_n, agent_val_n])
        self.state = np.array([time_n,sin_value_n,agent_val_n])
        #print ("Time Stamp = {}".format(self.state[0]))
        #Logic to define the reward, here less the distance between sin_value_n and agent_val_n higher the reward and
        #viceversa
        if (sin_value_n == agent_val_n):
            reward = 110
        else:
            reward = 1.0 / abs(agent_val_n - sin_value_n)

        if (self.state[0] % 3600 == 0):
            done = True
        else:
            done = False
        print ("State = {}".format(self.state))
        print("Reward = {}".format(reward))
        return self.state, reward, done, {}

    def _reset(self):
        self.state = np.array([0,0,np.random.random()])
        #self.state = self.np_random.uniform(low=-0.05, high=0.05, size=(4,))
        self.steps_beyond_done = None
        return self.state

    def _render(self, mode='human', close=False):
        if close:
            if self.viewer is not None:
                self.viewer.close()
                self.viewer = None
            return

        screen_width = 600
        screen_height = 400

        world_width = self.x_threshold*2
        scale = screen_width/world_width
        carty = 100 # TOP OF CART
        polewidth = 10.0
        polelen = scale * 1.0
        cartwidth = 50.0
        cartheight = 30.0

        if self.viewer is None:
            from gym.envs.classic_control import rendering
            self.viewer = rendering.Viewer(screen_width, screen_height)
            l,r,t,b = -cartwidth/2, cartwidth/2, cartheight/2, -cartheight/2
            axleoffset =cartheight/4.0
            cart = rendering.FilledPolygon([(l,b), (l,t), (r,t), (r,b)])
            self.carttrans = rendering.Transform()
            cart.add_attr(self.carttrans)
            self.viewer.add_geom(cart)
            l,r,t,b = -polewidth/2,polewidth/2,polelen-polewidth/2,-polewidth/2
            pole = rendering.FilledPolygon([(l,b), (l,t), (r,t), (r,b)])
            pole.set_color(.8,.6,.4)
            self.poletrans = rendering.Transform(translation=(0, axleoffset))
            pole.add_attr(self.poletrans)
            pole.add_attr(self.carttrans)
            self.viewer.add_geom(pole)
            self.axle = rendering.make_circle(polewidth/2)
            self.axle.add_attr(self.poletrans)
            self.axle.add_attr(self.carttrans)
            self.axle.set_color(.5,.5,.8)
            self.viewer.add_geom(self.axle)
            self.track = rendering.Line((0,carty), (screen_width,carty))
            self.track.set_color(0,0,0)
            self.viewer.add_geom(self.track)

        if self.state is None: return None

        x = self.state
        cartx = x[0]*scale+screen_width/2.0 # MIDDLE OF CART
        self.carttrans.set_translation(cartx, carty)
        self.poletrans.set_rotation(-x[2])

        return self.viewer.render(return_rgb_array = mode=='rgb_array')
