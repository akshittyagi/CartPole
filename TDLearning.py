import os
import sys
import random
import argparse
import pickle as pkl
import csv
from enum import Enum
import multiprocessing
from multiprocessing import Pool
import time

import numpy as np
from matplotlib import pyplot as plt
from DecisionProcess import MDP
from Environment import Environment, Actions

class TD(object):
    '''TD Eval and Learning'''
    def __init__(self, mdp, num_training_episodes, num_eval_episodes, order=3, alpha=0.01, policy='uniform'):
        '''Initialising from another MDP'''
        self.mdp = mdp
        self.num_train = num_training_episodes
        self.num_eval = num_eval_episodes
        self.policy = policy
        self.alpha = alpha
        self.gamma = self.mdp.gamma
        self.order = order
        self.w = np.zeros((order + 1) ** mdp.states)
    
    def initialize_weights(self):
        '''Init W again'''
        self.w = np.zeros((self.order + 1) ** self.mdp.states)

    def estimate_value_function(self):
        '''Estimate the value for num_train episodes'''
        s_t = self.mdp.get_init_state()
        time_counter = 0
        while not self.mdp.is_terminal_state(s_t, time_counter):
            a_t = self.mdp.policy(state=s_t, policy=self.policy)
            s_t_1 = self.mdp.transition_function(s_t, a_t)
            r_t = self.mdp.reward_function(s_t, a_t, s_t_1)
            v_s_t_1, _ = self.mdp.get_value_function(self.order, self.w, s_t_1)
            v_s_t, dv_dw = self.mdp.get_value_function(self.order, self.w, s_t)
            self.w = self.w + (r_t + self.gamma*v_s_t_1 - v_s_t)*dv_dw
            s_t = s_t_1
            time_counter += 1
  
    def evaluate_error(self):
        '''Eval error for an episode'''
        s_t = self.mdp.get_init_state()
        error = 0
        time_step = 0
        while not self.mdp.is_terminal_state(s_t, time_step):
            a_t = self.mdp.policy(state=s_t, policy=self.policy)
            s_t_1 = self.mdp.transition_function(s_t, a_t)
            r_t = self.mdp.reward_function(s_t, a_t, s_t_1)
            v_s_t_1, _ = self.mdp.get_value_function(self.order, self.w, s_t_1)
            v_s_t, dv_dw = self.mdp.get_value_function(self.order, self.w, s_t)
            error = error + (r_t + self.gamma*v_s_t_1 - v_s_t)**2
            s_t = s_t_1
            time_step += 1
        return (1.0*error)/time_step

    def update_weights(self):
        '''Update weights for a series of episodes'''
        self.initialize_weights()
        for episode in range(self.num_train):
            # print "UPDATING WEIGHTS FOR EPISODE: ", episode + 1
            self.estimate_value_function()

    def evaluate_policy(self):
        '''Evaluate the policy for a series of episodes'''
        error = 0.0
        for episode in range(self.num_eval):
            # print "EVALUATING TD ERROR FOR EPISODE: ", episode + 1
            error = error + self.evaluate_error()
        error = error*1.0/self.num_eval
        return error

    def create_plots_for_alphas(self, alphas):
        '''Create plots for log scaled alphas'''
        X, y = [], []
        for alpha in alphas:
            print "At alpha: ", alpha
            self.alpha = alpha
            X.append(alpha)
            self.update_weights()
            y.append(self.evaluate_policy()) 
        X = np.log(np.array(X))/np.log(10)
        plt.plot(X, y)
        plt.show()

if __name__ == "__main__":
    env = Environment(cart_mass=1,pole_mass=0.1,pole_half_length=0.5,start_position=0,start_velocity=0,start_angle=0,start_angular_velocity=0)
    mdp = MDP(env,1,debug=False)
    td = TD(mdp, 100, 100, 3)
    td.create_plots_for_alphas([1e-18,1e-17,1e-16,1e-15,1e-14,1e-13,1e-12,1e-11])