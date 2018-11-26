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
import math

import numpy as np
from matplotlib import pyplot as plt
from DecisionProcess import MDP
from Environment import Environment, Actions

def sigmoid(x):
    return 1.0/(1 + np.exp(-x))

class TD(object):
    '''TD Eval and Learning'''
    def __init__(self, mdp, num_training_episodes=100, num_eval_episodes=100, order=3, alpha=0.01, policy='uniform'):
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

class Sarsa(TD):
    '''Sarsa docstring'''
    def __init__(self, mdp, epsilon, alpha, train_episodes):
        super(Sarsa, self).__init__(mdp, alpha=alpha)
        self.episodes = train_episodes
        self.epsilon = epsilon
        self.gamma = self.mdp.gamma

    def epsilon_greedy_action_selection(self, state, temperature=1):
        random_number = 1.0*random.randint(0,99)/100
        epsilon = self.epsilon / temperature
        q_a_1, _ = self.mdp.get_q_value_function(self.order, self.w, state, 1)
        q_a_2, _ = self.mdp.get_q_value_function(self.order, self.w, state, -1)
        q_values = [q_a_1, q_a_2]
        if 1 - epsilon >= random_number:
            '''Select uniformly from the set of values argmax(q_values[s, ... ])'''
            arg_max = np.argwhere(q_values == np.amax(q_values))
            coin_toss = random.randint(0, len(arg_max) - 1)
            argmax = arg_max[coin_toss]
            action = (1 if argmax % 2 == 0 else -1)
            return action
        else:
            '''Select uniformly randomly from the set of actions'''
            coin_toss = random.randint(0, len(q_values) - 1)
            action = (1 if coin_toss % 2 == 0 else -1)
            return action

    def learn(self, reduction_factor=4, plot=False, debug=False):
        X, y = [], []
        X_ep, y_ep = [], []
        global_time_step, time_step = 0, 0
        alpha = self.alpha
        temperature = 1
        
        self.initialize_weights()

        for episode in range(self.episodes):
            if debug:
                print "------------------------------"
                print "AT EPISODE: ", episode + 1
            s_t = self.mdp.get_init_state()
            a_t = self.epsilon_greedy_action_selection(s_t)
            mse = 0
            time_step = 0
            temperature = 1
            g = 0
            while not self.mdp.is_terminal_state(s_t, time_step):
                alpha = alpha / temperature
                s_t_1 = self.mdp.transition_function(s_t, a_t)
                r_t = self.mdp.reward_function(s_t, a_t, s_t_1)
                a_t_1 = self.epsilon_greedy_action_selection(s_t_1, temperature=temperature)
                q_s_prime_a_prime, _ = self.mdp.get_q_value_function(self.order, self.w, s_t_1, a_t_1)
                q_s_a, dq_dw = self.mdp.get_q_value_function(self.order, self.w, s_t, a_t)
                q_td_error = ((r_t + self.gamma*(q_s_prime_a_prime) - q_s_a)*dq_dw)
                self.w += alpha*(q_td_error)
                s_t = s_t_1
                a_t = a_t_1
                g = g + r_t*(self.gamma**time_step)
                global_time_step += 1
                time_step += 1
                mse += q_td_error**2
                # temperature = global_time_step**(1.0/reduction_factor)
            mse = mse / time_step
            X_ep.append(episode)
            y_ep.append(g)
            if debug:
                print "Return:  ", g
                print "------------------------------"
        if plot:
            plt.plot(X_ep, y_ep)
            plt.show()
        return X_ep, y_ep, np.sum(np.array(y_ep))*1.0/len(y_ep)

class Sarsa_NN(TD):
    '''Sarsa docstring'''
    def __init__(self, mdp, epsilon, alpha, train_episodes):
        super(Sarsa_NN, self).__init__(mdp, alpha=alpha)
        self.episodes = train_episodes
        self.epsilon = epsilon
        self.gamma = self.mdp.gamma
        self.hidden = 10
        self.w1 = np.random.rand(self.mdp.states, self.hidden)
        self.w2 = np.random.rand(self.hidden, 1)
        self.b1 = np.ones(self.hidden)
        self.b2 = 1

    def init_weights(self):
        self.w1 = np.random.rand(self.mdp.states, self.hidden)
        self.w2 = np.random.rand(self.hidden, 1)
        self.b1 = np.ones(self.hidden).reshape(self.hidden, 1)
        self.b2 = 1

    def get_q_value_function(self, state, action):
        curr_s = list(state)
        curr_s.append(action)
        curr_s = np.array(curr_s).reshape(len(curr_s), 1)
        sig = sigmoid(self.w1.T.dot(curr_s) + self.b1)
        out = self.w2.T.dot(sig) + self.b2
        self.cache = (sig, out, curr_s)
        return out[0][0]
    
    def backprop_weights(self, error, alpha):
        sig, out, curr_s = self.cache
        dq_dw2 = sig
        dq_db2 = 1
        dq_dw1 = curr_s.dot((self.w2*(sig)*(1-sig)).T)
        dq_db1 = self.w2*(sig)*(1-sig)
        self.w1 += alpha*error*dq_dw1
        self.w2 += alpha*error*dq_dw2
        self.b1 += alpha*error*dq_db1
        self.b2 += alpha*error*dq_db2

    def epsilon_greedy_action_selection(self, state, temperature=1):
        random_number = 1.0*random.randint(0,99)/100
        epsilon = self.epsilon / temperature
        q_a_1 = self.get_q_value_function(state, 1)
        q_a_2 = self.get_q_value_function(state, -1)
        q_values = [q_a_1, q_a_2]
        if 1 - epsilon >= random_number:
            '''Select uniformly from the set of values argmax(q_values[s, ... ])'''
            arg_max = np.argwhere(q_values == np.amax(q_values))
            coin_toss = random.randint(0, len(arg_max) - 1)
            argmax = arg_max[coin_toss]
            action = (1 if argmax[0] % 2 == 0 else -1)
            return action
        else:
            '''Select uniformly randomly from the set of actions'''
            coin_toss = random.randint(0, len(q_values) - 1)
            action = (1 if coin_toss % 2 == 0 else -1)
            return action

    def learn(self, reduction_factor=4, plot=False, debug=False):
        X, y = [], []
        X_ep, y_ep = [], []
        global_time_step, time_step = 0, 0
        alpha = self.alpha
        temperature = 1

        self.init_weights()

        for episode in range(self.episodes):
            if debug:
                print "------------------------------"
                print "AT EPISODE: ", episode + 1
            s_t = self.mdp.get_init_state()
            a_t = self.epsilon_greedy_action_selection(s_t)
            mse = 0
            time_step = 0
            temperature = 1
            g = 0
            while not self.mdp.is_terminal_state(s_t, time_step):
                alpha = alpha / temperature
                s_t_1 = self.mdp.transition_function(s_t, a_t)
                r_t = self.mdp.reward_function(s_t, a_t, s_t_1)
                a_t_1 = self.epsilon_greedy_action_selection(s_t_1, temperature=temperature)
                q_s_prime_a_prime = self.get_q_value_function(s_t_1, a_t_1)
                q_s_a = self.get_q_value_function(s_t, a_t)
                q_td_error = ((r_t + self.gamma*(q_s_prime_a_prime) - q_s_a))
                self.backprop_weights(q_td_error, alpha)
                s_t = s_t_1
                a_t = a_t_1
                g = g + r_t*(self.gamma**time_step)
                global_time_step += 1
                time_step += 1
                mse += q_td_error**2
                temperature = global_time_step**(1.0/reduction_factor)
            mse = mse / time_step
            X_ep.append(episode)
            y_ep.append(g)
            if debug:
                print "Return:  ", g
                print "------------------------------"
        if plot:
            plt.plot(X_ep, y_ep)
            plt.show()
        return X_ep, y_ep, np.sum(np.array(y_ep))*1.0/len(y_ep)

class Qlearning(TD):
    '''Sarsa docstring'''
    def __init__(self, mdp, epsilon, alpha, train_episodes):
        super(Qlearning, self).__init__(mdp, alpha=alpha)
        self.episodes = train_episodes
        self.epsilon = epsilon
        self.gamma = self.mdp.gamma

    def epsilon_greedy_action_selection(self, state, temperature=1):
        random_number = 1.0*random.randint(0,99)/100
        epsilon = self.epsilon / temperature
        q_a_1, _ = self.mdp.get_q_value_function(self.order, self.w, state, 1)
        q_a_2, _ = self.mdp.get_q_value_function(self.order, self.w, state, -1)
        q_values = [q_a_1, q_a_2]
        if 1 - epsilon >= random_number:
            '''Select uniformly from the set of values argmax(q_values[s, ... ])'''
            arg_max = np.argwhere(q_values == np.amax(q_values))
            coin_toss = random.randint(0, len(arg_max) - 1)
            argmax = arg_max[coin_toss]
            action = (1 if argmax % 2 == 0 else -1)
            return action
        else:
            '''Select uniformly randomly from the set of actions'''
            coin_toss = random.randint(0, len(q_values) - 1)
            action = (1 if coin_toss % 2 == 0 else -1)
            return action

    def learn(self, reduction_factor=4, plot=False, debug=False):
        X, y = [], []
        X_ep, y_ep = [], []
        global_time_step, time_step = 0, 0
        alpha = self.alpha
        temperature = 1
        
        self.initialize_weights()

        for episode in range(self.episodes):
            if debug:
                print "------------------------------"
                print "AT EPISODE: ", episode + 1
            s_t = self.mdp.get_init_state()
            mse = 0
            time_step = 0
            temperature = 1
            g = 0
            while not self.mdp.is_terminal_state(s_t, time_step):
                alpha = alpha / temperature
                a_t = self.epsilon_greedy_action_selection(s_t)
                s_t_1 = self.mdp.transition_function(s_t, a_t)
                r_t = self.mdp.reward_function(s_t, a_t, s_t_1)
                q_s_prime_a1, _ = self.mdp.get_q_value_function(self.order, self.w, s_t_1, 1)
                q_s_prime_a2, _ = self.mdp.get_q_value_function(self.order, self.w, s_t_1, -1)
                q_s_a, dq_dw = self.mdp.get_q_value_function(self.order, self.w, s_t, a_t)
                q_td_error = ((r_t + self.gamma*(max(q_s_prime_a1, q_s_prime_a2)) - q_s_a)*dq_dw)
                self.w += alpha*(q_td_error)
                s_t = s_t_1
                g = g + r_t*(self.gamma**time_step)
                global_time_step += 1
                time_step += 1
                mse += q_td_error**2
                # temperature = global_time_step**(1.0/reduction_factor)
            mse = mse / time_step
            X_ep.append(episode)
            y_ep.append(g)
            if debug:
                print "Return:  ", g
                print "------------------------------"
        if plot:
            plt.plot(X_ep, y_ep)
            plt.show()
        return X_ep, y_ep, np.sum(np.array(y_ep))*1.0/len(y_ep)
    
class Qlearning_NN(TD):
    '''Sarsa docstring'''
    def __init__(self, mdp, epsilon, alpha, train_episodes):
        super(Qlearning_NN, self).__init__(mdp, alpha=alpha)
        self.episodes = train_episodes
        self.epsilon = epsilon
        self.gamma = self.mdp.gamma
        self.hidden = 10
        self.w1 = np.random.rand(self.mdp.states, self.hidden)
        self.w2 = np.random.rand(self.hidden, 1)
        self.b1 = np.ones(self.hidden)
        self.b2 = 1

    def init_weights(self):
        self.w1 = np.random.rand(self.mdp.states, self.hidden)
        self.w2 = np.random.rand(self.hidden, 1)
        self.b1 = np.ones(self.hidden).reshape(self.hidden, 1)
        self.b2 = 1

    def get_q_value_function(self, state, action):
        curr_s = list(state)
        curr_s.append(action)
        curr_s = np.array(curr_s).reshape(len(curr_s), 1)
        sig = sigmoid(self.w1.T.dot(curr_s) + self.b1)
        out = self.w2.T.dot(sig) + self.b2
        self.cache = (sig, out, curr_s)
        return out[0][0]
    
    def backprop_weights(self, error, alpha):
        sig, out, curr_s = self.cache
        dq_dw2 = sig
        dq_db2 = 1
        dq_dw1 = curr_s.dot((self.w2*(sig)*(1-sig)).T)
        dq_db1 = self.w2*(sig)*(1-sig)
        self.w1 += alpha*error*dq_dw1
        self.w2 += alpha*error*dq_dw2
        self.b1 += alpha*error*dq_db1
        self.b2 += alpha*error*dq_db2

    def epsilon_greedy_action_selection(self, state, temperature=1):
        random_number = 1.0*random.randint(0,99)/100
        epsilon = self.epsilon / temperature
        q_a_1 = self.get_q_value_function(state, 1)
        q_a_2 = self.get_q_value_function(state, -1)
        q_values = [q_a_1, q_a_2]
        if 1 - epsilon >= random_number:
            '''Select uniformly from the set of values argmax(q_values[s, ... ])'''
            arg_max = np.argwhere(q_values == np.amax(q_values))
            coin_toss = random.randint(0, len(arg_max) - 1)
            argmax = arg_max[coin_toss]
            action = (1 if argmax[0] % 2 == 0 else -1)
            return action
        else:
            '''Select uniformly randomly from the set of actions'''
            coin_toss = random.randint(0, len(q_values) - 1)
            action = (1 if coin_toss % 2 == 0 else -1)
            return action

    def learn(self, reduction_factor=4, plot=False, debug=False):
        X, y = [], []
        X_ep, y_ep = [], []
        global_time_step, time_step = 0, 0
        alpha = self.alpha
        temperature = 1

        self.init_weights()

        for episode in range(self.episodes):
            if debug:
                print "------------------------------"
                print "AT EPISODE: ", episode + 1
            s_t = self.mdp.get_init_state()
            mse = 0
            time_step = 0
            temperature = 1
            g = 0
            while not self.mdp.is_terminal_state(s_t, time_step):
                alpha = alpha / temperature
                a_t = self.epsilon_greedy_action_selection(s_t)
                s_t_1 = self.mdp.transition_function(s_t, a_t)
                r_t = self.mdp.reward_function(s_t, a_t, s_t_1)
                q_s_prime_a1 = self.get_q_value_function(s_t_1, 1)
                q_s_prime_a2 = self.get_q_value_function(s_t_1, -1)
                q_s_a = self.get_q_value_function(s_t, a_t)
                q_td_error = ((r_t + self.gamma*(max(q_s_prime_a1, q_s_prime_a2)) - q_s_a))
                self.backprop_weights(q_td_error, alpha)
                s_t = s_t_1
                g = g + r_t*(self.gamma**time_step)
                global_time_step += 1
                time_step += 1
                mse += q_td_error**2
                temperature = global_time_step**(1.0/reduction_factor)
            mse = mse / time_step
            X_ep.append(episode)
            y_ep.append(g)
            if debug:
                print "Return:  ", g
                print "------------------------------"
        if plot:
            plt.plot(X_ep, y_ep)
            plt.show()
        return X_ep, y_ep, np.sum(np.array(y_ep))*1.0/len(y_ep)

if __name__ == "__main__":
    fourier_order = 3
    env = Environment(cart_mass=1,pole_mass=0.1,pole_half_length=0.5,start_position=0,start_velocity=0,start_angle=0,start_angular_velocity=0)
    mdp = MDP(env,1,include_action=True, debug=False)
    td = TD(mdp, 100, 100, fourier_order)
    num_trials = 100
    num_training_episodes = 100
    
    hyperparam_search = False
    switch_sarsa = 3
    X = np.arange(num_training_episodes)
    Y = []

    if switch_sarsa == 0:
        print "------------" 
        print "SARSA" 
        print "------------"
    elif switch_sarsa == 1: 
        print "------------"
        print "Q-LEARNING"
        print "------------"
    elif switch_sarsa == 2:
        print "------------" 
        print "SARSA NN" 
        print "------------"
    elif switch_sarsa == 3:
        print "------------"
        print "Q-LEARNING NN"
        print "------------"

    if hyperparam_search:
        '''HyperParameter Search'''
        alphas = get_hyperparams(range_of_param=[1e-3, 1e-1], interval=10, multiplicative=True)
        epsilons = get_hyperparams(range_of_param=[1e-2, 1e-1], interval=0.02, multiplicative=False)
        reduction_factors = get_hyperparams(range_of_param=[3,10], interval=1, multiplicative=False)
        G = -2**31
        params = []
        for alpha in alphas:
            for epsilon in epsilons:
                for reduction_factor in reduction_factors:
                    print "RETURN for alpha", str(alpha), " epsilon ", str(epsilon), " reductionFactor ", str(reduction_factor), " : "
                    if switch_sarsa == 0:
                        sarsa = Sarsa(mdp, epsilon=epsilon, alpha=alpha, train_episodes=num_training_episodes)
                        _, y, g = sarsa.learn(reduction_factor=reduction_factor, plot=False, debug=False)
                    elif switch_sarsa == 1:
                        qlearn = Qlearning(mdp, epsilon=epsilon, alpha=alpha, train_episodes=num_training_episodes)
                        _, y, g = qlearn.learn(reduction_factor=reduction_factor, plot=False, debug=False)
                    elif switch_sarsa == 2:
                        sarsa = Sarsa_NN(mdp, epsilon=epsilon, alpha=alpha, train_episodes=num_training_episodes)
                        _, y, g = sarsa.learn(reduction_factor=reduction_factor, plot=False, debug=False)
                    elif switch_sarsa == 3:
                        qlearn = Qlearning_NN(mdp, epsilon=epsilon, alpha=alpha, train_episodes=num_training_episodes)
                        _, y, g = qlearn.learn(reduction_factor=reduction_factor, plot=False, debug=False)
                    print g
                    if G < g:
                        G = g
                        params = [alpha, epsilon, reduction_factor]
                        print "BEST PARAMS: "
                        print params
   
    if not hyperparam_search:
        #alpha, epsilon, reduction_factor: alpha = alpha/(temp**red_fac)
        params = [1e-1, 1e-1, 2]

    for trial in range(num_trials):
        print "AT TRIAL: ", trial + 1
        if switch_sarsa == 0:
            sarsa = Sarsa(mdp, epsilon=params[1], alpha=params[0], train_episodes=num_training_episodes)
            _, y, _ = sarsa.learn(reduction_factor=params[2], plot=False, debug=False)
        elif switch_sarsa == 1:
            qlearn = Qlearning(mdp, epsilon=params[1], alpha=params[0], train_episodes=num_training_episodes)
            _, y, _ = qlearn.learn(reduction_factor=params[2], plot=False, debug=False)
        elif switch_sarsa == 2:
            sarsa = Sarsa_NN(mdp, epsilon=params[1], alpha=params[0], train_episodes=num_training_episodes)
            _, y, _ = sarsa.learn(reduction_factor=params[2], plot=False, debug=False)
        elif switch_sarsa == 3:
            qlearn = Qlearning_NN(mdp, epsilon=params[1], alpha=params[0], train_episodes=num_training_episodes)
            _, y, _ = qlearn.learn(reduction_factor=params[2], plot=False, debug=False)
        Y.append(y)
    Y = np.array(Y)
    Y_mean = np.sum(Y, axis=0)
    Y_mean = Y_mean/num_trials
    Y_diff = np.repeat(Y_mean.reshape(1, num_training_episodes), num_trials, axis=0)    
    Y_diff = Y - Y_diff
    Y_diff = Y_diff ** 2
    Y_diff = np.sum(Y_diff, axis=0) / num_trials
    Y_diff = np.sqrt(Y_diff)
    plt.errorbar(X, Y_mean, yerr=Y_diff, fmt='o')
    
    if switch_sarsa == 0:
        print "------------" 
        print "SARSA" 
        print "------------"
    elif switch_sarsa == 1: 
        print "------------"
        print "Q-LEARNING"
        print "------------"
    elif switch_sarsa == 2:
        print "------------" 
        print "SARSA NN" 
        print "------------"
    elif switch_sarsa == 3:
        print "------------"
        print "Q-LEARNING NN"
        print "------------"

    plt.show()