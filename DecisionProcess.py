import os
import sys
import random
import argparse
import pickle as pkl
import csv
from enum import Enum

from Environment import Environment, Actions

class MDP():

    def __init__(self, env, gamma, g=9.8, f=10, time_step=0.02, fail_angle=90, terminate_time=20.2):
        self.cart = env.cart 
        self.pole = env.pole
        self.g = g
        self.f = f
        self.time_step = time_step
        self.fail_angle = fail_angle
        self.terminate_time = terminate_time
        self.gamma = gamma
        self.start_position = env.start_position
        self.track_limits = [-env.track_limits, env.track_limits]
        self.actions = Actions

    def get_init_state(self):
        pass

    def get_state_tuple(self, env):
        pass

    def is_terminal_state(self, state):
        pass
    
    def is_valid_state(self, state):
        pass

    def policy(self, state):
        pass
    
    def transition_function(self, state, action):
        pass
    
    def reward_function(self, s_t, a_t, s_t_1):
        return 1
    
    def run_episode(self, policy):
        pass

    def learn_policy(self, num_episodes, policy):
        pass

    

    