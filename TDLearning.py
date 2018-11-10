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

class TD(object):
    '''TD Eval and Learning'''
    def __init__(self, mdp, num_training_episodes, num_eval_episodes, alpha=0.01, policy='uniform'):
        '''Initialising from another MDP'''
        self.mdp = mdp
        self.num_train = num_training_episodes
        self.num_eval = num_eval_episodes
        self.policy = policy
        self.alpha = alpha
        self.gamma = self.mdp.gamma
        self.value_function = self.mdp.initValueFunction()

    def estimate_value_function(self):
        '''Estimate the value for num_train episodes'''
        s_t = self.mdp.getInitialState()
        while not self.mdp.isTerminalState(s_t):
            a_t = self.mdp.getActionFromPolicy(s_t, policy=self.policy)
            s_t_1 = self.mdp.TransitionFunction(s_t, a_t)
            r_t = self.mdp.RewardFunction(s_t, a_t, s_t_1)
            s = self.mdp.getStateId(s_t)
            s_ = self.mdp.getStateId(s_t_1)
            self.value_function[s] += self.alpha*1.0*(r_t + self.gamma*1.0*(self.value_function[s_]) - self.value_function[s])
            s_t = s_t_1
  
    def evaluate_error(self):
        '''Eval error for an episode'''
        s_t = self.mdp.getInitialState()
        error = 0
        time_step = 0
        while(not self.mdp.isTerminalState(s_t)):
            a_t = self.mdp.getActionFromPolicy(s_t, policy=self.policy)
            s_t_1 = self.mdp.TransitionFunction(s_t, a_t)
            r_t = self.mdp.RewardFunction(s_t, a_t, s_t_1)
            s = self.mdp.getStateId(s_t)
            s_ = self.mdp.getStateId(s_t_1)
            error = error + (r_t + self.gamma*1.0*(self.value_function[s_]) - self.value_function[s])**2
            s_t = s_t_1
            time_step += 1
        return (1.0*error)/time_step

    def update_weights(self):
        '''Update weights for a series of episodes'''
        self.value_function = self.mdp.initValueFunction()
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
    td = TD(mdp, 100, 100)
    td.create_plots_for_alphas([1e-8,1e-7,1e-6,1e-5,1e-4,1e-3,1e-2,1e-1])