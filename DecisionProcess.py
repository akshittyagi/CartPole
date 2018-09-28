import os
import sys
import random
import argparse
import pickle as pkl
import csv
from enum import Enum

import numpy as np
from Environment import Environment, Actions

class MDP():

    def __init__(self, env, gamma, g=9.8, f=10, time_step=0.02, fail_angle=90, terminate_time=20.2):
        self.env = env
        self.g = g
        self.f = f
        self.time_step = time_step
        self.fail_angle = fail_angle
        self.terminate_time = terminate_time
        self.gamma = gamma
        self.track_limits = np.arange(-env.track_limits, env.track_limits)
        self.actions = Actions
        self.time = 0

    def get_init_state(self):
        x = self.env.start_position
        v = self.env.start_velocity
        theta = self.env.start_angle
        omega = self.env.start_angular_velocity
        return (x,v,theta,omega)

    def get_state_tuple(self, env):
        return (env.cart.position, env.cart.velocity, env.pole.theta, env.pole.omega)

    def is_terminal_state(self, state):
        if time == 20.2 or abs(state[2]) == 90 or state[0]>=max(self.track_limits) or state[0]<=min(self.track_limits):
            return True
        return False
    
    def is_valid_state(self, state):
        pass

    def policy(self, state):
        pass
    
    def transition_function(self, state, action):
        pass
    
    def reward_function(self, s_t, a_t, s_t_1, time_step):
        return 1*(self.gamma**time_step)
    
    def run_episode(self, policy):
        s_t = self.get_init_state()
        total_reward = 0
        time_counter = 0

        while(not self.is_terminal_state(s_t)):
            a_t = self.policy(s_t)
            s_t_1 = self.transition_function(s_t, a_t)
            r_t = self.reward_function(s_t, a_t, s_t_1, time_counter)
            total_reward += r_t
            time_counter += 1
            s_t = s_t_1

    def learn_policy(self, num_episodes, policy):
        pass

    

    