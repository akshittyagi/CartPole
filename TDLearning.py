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
            self.w = self.w + self.alpha*(r_t + self.gamma*v_s_t_1 - v_s_t)*dv_dw
            s_t = s_t_1
            time_counter += 1
  
    def evaluate_error(self, plot=False):
        '''Eval error for an episode'''
        s_t = self.mdp.get_init_state()
        error = 0
        time_step = 0
        if plot:
            X = []
            y = []
        while not self.mdp.is_terminal_state(s_t, time_step):
            a_t = self.mdp.policy(state=s_t, policy=self.policy)
            s_t_1 = self.mdp.transition_function(s_t, a_t)
            r_t = self.mdp.reward_function(s_t, a_t, s_t_1)
            v_s_t_1, _ = self.mdp.get_value_function(self.order, self.w, s_t_1)
            v_s_t, dv_dw = self.mdp.get_value_function(self.order, self.w, s_t)
            curr_error_2 = (r_t + self.gamma*v_s_t_1 - v_s_t)**2
            error = error + curr_error_2
            s_t = s_t_1
            if plot:
                X.append(time_step)
                y.append(curr_error_2)
            time_step += 1
        if plot:
            plt.plot(X, y)
            plt.show()
        return (1.0*error)/time_step

    def update_weights(self):
        '''Update weights for a series of episodes'''
        self.initialize_weights()
        for episode in range(self.num_train):
            # print "UPDATING WEIGHTS FOR EPISODE: ", episode + 1
            self.estimate_value_function()

    def evaluate_policy(self, alpha):
        '''Evaluate the policy for a series of episodes'''
        error = 0.0
        X = []
        y = []
        for episode in range(self.num_eval):
            # print "EVALUATING TD ERROR FOR EPISODE: ", episode + 1
            if episode % 55 == 0:
                curr_error = self.evaluate_error(plot=False)
                error = error + curr_error
                X.append(episode + 1)
                y.append(curr_error)
            else:
                curr_error = self.evaluate_error(plot=False)
                error = error + curr_error
                X.append(episode + 1)
                y.append(curr_error)
        plt.plot(X, y)
        plt.savefig(str(alpha) + "_fig_CP_" + str(self.order) + ".png")
        plt.clf()
        plt.cla()
        plt.close()
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
            y.append(self.evaluate_policy(alpha))
        print(X)
        print(y) 
        X = np.log(np.array(X))/np.log(10)
        plt.plot(X, y)
        plt.show()

if __name__ == "__main__":
    env = Environment(cart_mass=1,pole_mass=0.1,pole_half_length=0.5,start_position=0,start_velocity=0,start_angle=0,start_angular_velocity=0)
    mdp = MDP(env,1,debug=False)
    td = TD(mdp, 100, 100, 3)
    td.create_plots_for_alphas([1e-8,1e-7,1e-6,1e-5,1e-4,1e-3,1e-2,1e-1])