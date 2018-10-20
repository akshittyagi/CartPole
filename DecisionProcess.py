import os
import sys
import random
import argparse
import pickle as pkl
import csv
from enum import Enum
import time
import multiprocessing
from multiprocessing import Pool

import numpy as np
import matplotlib.pyplot as plt
from Environment import Environment, Actions
import util

class MDP():

    def __init__(self, env=Environment(cart_mass=1,pole_mass=0.1,pole_half_length=0.5,start_position=0,start_velocity=0,start_angle=0,start_angular_velocity=0), gamma=1, g=9.8, f=10, time_step=0.02, fail_angle=np.deg2rad(90), terminate_time=20.2, debug=False):
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
        self.epsilon = 1e-4
        self.debug = debug
        self.max_velocity = 10
        self.max_omega = 180

    def get_init_state(self):
        x = self.env.start_position
        v = self.env.start_velocity
        theta = self.env.start_angle
        omega = self.env.start_angular_velocity
        return (x,v,theta,omega)

    def get_state_tuple(self, env):
        return (env.cart.position, env.cart.velocity, env.pole.theta, env.pole.omega)

    def is_terminal_state(self, state, time_counter):
        time = time_counter*self.time_step
        if time == 20.2 or abs(state[2])>self.fail_angle or state[0]>=max(self.track_limits) or state[0]<=min(self.track_limits):
            return True
        return False
    
    def is_valid_state(self, state):
        pass

    def policy(self, policy, state):
        x,v,theta,omega = state
        discrete_step = self.time_step*self.max_velocity
        multiplier = x*1.0//discrete_step
        s_t = 0
        if x < 0 :
            nxt = -1
        else:
            nxt = 1
        if abs(x - multiplier*discrete_step) < abs(x-(multiplier+nxt)*discrete_step):
            s_t = multiplier
        else:
            s_t = multiplier + nxt
        s_t = int(s_t)
        idx = 0
        if s_t < 0:
            idx = 15 + abs(s_t)
        else:
            idx = s_t
        currRow = policy[idx]
        random_number = 1.0*random.randint(0,99)/100
        action_array = sorted(zip(np.arange(len(currRow)), currRow), key=lambda x: x[1], reverse=True)
        prev_proba = 0
        for action, probability in action_array:
            prev_proba += probability
            if random_number <= prev_proba:
                if self.debug:
                    print "Action Array: ", action_array
                    print "Rand number: ",random_number
                    print "Action selected: ", (-1 if action==0 else 1)    
                return (-1 if action==0 else 1)
        

    def get_accelerations(self, f, state):
        # For the derivations of the dynamics, see: https://coneural.org/florian/papers/05_cart_pole.pdf
        x = state[0]
        v = state[1]
        theta = state[2]
        omega = state[3]
        sin = np.sin
        cos = np.cos
        alpha = self.g*sin(theta) + cos(theta)*( -f - self.env.pole.mass*self.env.pole.length*(omega**2)*sin(theta))/(self.env.cart.mass + self.env.pole.mass)
        alpha /= (self.env.pole.length*(4/3 - (self.env.pole.mass*(cos(theta)**2))/(self.env.pole.mass + self.env.cart.mass)))
        a =( f + (self.env.pole.mass*self.env.pole.length*((omega**2)*sin(theta) - alpha*cos(theta))) )/ (self.env.cart.mass + self.env.pole.mass) 
        return a, alpha

    def transition_function(self, state, action):
        '''Using the forward euler approximation: http://web.mit.edu/10.001/Web/Course_Notes/Differential_Equations_Notes/node3.html'''
        x, v, theta, omega = state
        f = self.f*action
        a, alpha = self.get_accelerations(f, (x,v,theta,omega))
        x = x + v*self.time_step
        theta = theta + omega*self.time_step
        v = v + a*self.time_step
        omega = omega + alpha*self.time_step
        if omega > 0:
            omega = min(np.deg2rad(self.max_omega), omega)
        else:
            omega = max(-np.deg2rad(self.max_omega), omega)
        if v > 0:
            v = min(self.max_velocity, v)
        else:
            v = max(-self.max_velocity, v)
        return (x,v,theta,omega)
    
    def reward_function(self, s_t, a_t, s_t_1, time_step):
        return 1*(self.gamma**time_step)
    
    def run_episode(self, policy):
        s_t = self.get_init_state()
        total_reward = 0
        time_counter = 0
        while(not self.is_terminal_state(s_t, time_counter)):
            if self.debug:
                self.print_state(s_t)
            a_t = self.policy(policy, s_t)
            if self.debug:
                print("Action at time: ", time_counter, " : ", a_t)
            s_t_1 = self.transition_function(s_t, a_t)
            r_t = self.reward_function(s_t, a_t, s_t_1, time_counter)
            total_reward += r_t
            time_counter += 1
            s_t = s_t_1
        if self.debug:
            self.print_state(s_t)
            print("Time Steps: ", time_counter, " Total Reward: ", total_reward)
        return total_reward

    def print_state(self, s_t):
        print("Pos: ", s_t[0], " Velocity: ", s_t[1], " Theta: ", np.rad2deg(s_t[2]), " Omega: ", np.rad2deg(s_t[3]))

    def evaluate(self, theta_k, num_episodes):
        reward = 0
        for episode in range(num_episodes):
            curr_reward  = self.run_episode(policy=theta_k)
            reward += curr_reward
            if episode % num_episodes/10 == 0 and self.debug:
                print "At episode: ", episode
                print "Reward: ", reward
        if self.debug:
            print "Av Reward: ", reward*1.0/num_episodes
        return reward*1.0/num_episodes

    def iterable(self, array):
        for elem in array:
            yield elem

    def learn_policy_fchc(self, num_iter, sigma, num_episodes):
        reshape_param = (31, 2)
        curr_iter = 0
        data = []
        theta_max = []
        global_max = -2**31
        theta = util.get_init(state_space=reshape_param[0], action_space=reshape_param[1], sigma=sigma, condition=True)
        softmax_theta = np.exp(theta)
        softmax_theta = softmax_theta/np.sum(softmax_theta, axis=1)[:,None]
        j = self.evaluate(softmax_theta, num_episodes)
                
        while curr_iter < num_iter:
            print "-----------------------------"
            print "At ITER: ", curr_iter
            theta_sampled = util.sample(distribution='gaussian', theta=theta, sigma=sigma, reshape_param=reshape_param)
            softmax_theta = np.exp(theta_sampled)
            softmax_theta = softmax_theta/np.sum(softmax_theta, axis=1)[:,None]
            j_n = self.evaluate(softmax_theta, num_episodes)
            data.append(j_n)
            if j_n > j:
                theta = theta_sampled
                j = j_n
                print "MAX REWARD: ", j, " AT iter: ", curr_iter
            if j_n > global_max:
                global_max = j_n
                theta_max = theta
                print "GLOBAL MAX UPDATED: ", global_max, " AT iter: ", curr_iter
            print "-----------------------------"
            curr_iter += 1
        print "Saving Data"
        pkl.dump(data, open("fchcFILE.pkl", 'w'))
        pkl.dump(theta_max, open("fchcTHETA.pkl", 'w'))

    def learn_policy_bbo_multiprocessing(self, init_population, best_ke, num_episodes, epsilon, num_iter, steps_per_trial=15, variance=10):
        assert init_population >= best_ke
        assert num_episodes > 1
        curr_iter = 0
        reshape_param = (31, 2)
        data = []
        theta_max = []
        max_av_reward = -2**31
        while (curr_iter < num_iter):
            import pdb; pdb.set_trace()
            theta, sigma = util.get_init(state_space=reshape_param[0],action_space=reshape_param[1], sigma=variance)
            for i in range(steps_per_trial):
                values = []
                print "-----------------------------"
                print "At ITER: ", curr_iter
                print "AT step: ", i
                theta_sampled= util.sample('gaussian', theta, sigma, reshape_param, init_population)
                theta_sampled = variance*theta_sampled
                softmax_theta = np.exp(theta_sampled)
                tic = time.time()
                pool = Pool(multiprocessing.cpu_count())
                mp_obj = multiprocessing_obj(num_episodes)
                values = pool.map(mp_obj, self.iterable(softmax_theta))
                data.append(np.array(values)[:,1].tolist())
                pool.close()
                pool.join()
                toc = time.time()
                values = sorted(values, key=lambda x: x[1], reverse=True)
                print "Max reward: ", values[0][1]
                if max_av_reward < values[0][1]:
                    max_av_reward = values[0][1]
                    print "MAX REWARD UPDATED"
                    theta_max = values[0][0]
                theta, sigma = util.generate_new_distribution('gaussian', theta, values, best_ke, epsilon)
                print "-----------------------------"
            curr_iter += 1
        print "Saving data"
        pkl.dump(data, open("FILE.pkl", 'w'))
        pkl.dump(theta_max, open("THETA.pkl", 'w'))
    

    def learn_policy_bbo(self, init_population, best_ke, num_episodes, epsilon, num_iter, steps_per_trial=15, sigma=100):
        assert init_population >= best_ke
        assert num_episodes > 1
        curr_iter = 0
        reshape_param = (31, 2)
        data = []
        theta_max = []
        max_av_reward = -2**31
        while (curr_iter < num_iter):
            theta, sigma = util.get_init(state_space=reshape_param[0],action_space=reshape_param[1], sigma=sigma)
            for i in range(steps_per_trial):
                values = []
                print "-----------------------------"
                print "At ITER: ", curr_iter
                print "AT step: ", i
                theta_sampled= util.sample('gaussian', theta, sigma, reshape_param, init_population)
                theta_sampled = np.exp(theta_sampled)
                tic = time.time()
                for k in range(init_population):
                    theta_k = theta_sampled[k]
                    theta_k = theta_k/np.sum(theta_k, axis=1)[:,None]
                    j_k = self.evaluate(theta_k, num_episodes)
                    data.append(j_k)
                    if j_k > max_av_reward:
                        max_av_reward = j_k
                        theta_max = theta_k
                        print "MAX REWARD: ", max_av_reward, " AT step, iter: ", i, curr_iter
                    values.append((theta_k.reshape(reshape_param[0]*reshape_param[1], 1), j_k))  
                toc = time.time()
                print(toc-tic)
                values = sorted(values, key=lambda x: x[1], reverse=True)
                theta, sigma = util.generate_new_distribution('gaussian', theta, values, best_ke, epsilon)
                print "-----------------------------"
            curr_iter += 1
        print "Saving Data"
        pkl.dump(data, open("FILE.pkl", 'w'))
        pkl.dump(theta_max, open("THETA.pkl", 'w'))

class multiprocessing_obj(MDP):
        def __init__(self, num_episodes):
            MDP.__init__(self)
            self.num_episodes = num_episodes
        def __call__(self, theta):
            theta = theta/np.sum(theta, axis=1)[:,None]
            j = self.evaluate(theta, self.num_episodes)
            return theta.reshape(theta.shape[0]*theta.shape[1], 1), j

def generate_graphs(cond=False):
    if not cond:
        data = pkl.load(open('FILE.pkl', 'r'))
        num_policies = 100
        num_steps = 15
        num_trials = 20
        counter = 0
        steps = [0]*num_policies*num_steps
        steps = np.array(steps, dtype='float64')
        errors = [0]*num_policies*num_steps
        errors = np.array(errors, dtype='float64')
        while( counter < num_trials ):
            print "STEPS", counter
            curr_trial = data[counter*num_steps:(counter+1)*num_steps]
            arr = np.array(curr_trial)
            arr = arr.reshape(num_policies*num_steps)
            steps += arr
            counter += 1
        steps /= num_trials
        counter = 0
        while( counter < num_trials ):
            print "ERRORS", counter
            curr_trial = data[counter*num_steps:(counter+1)*num_steps]
            arr = np.array(curr_trial)
            arr = arr.reshape(num_policies*num_steps)
            diff = abs(arr-steps)
            errors += diff**2
            counter += 1
        errors /= num_trials
        errors = np.sqrt(errors)
        x = np.arange(1500)
        y = steps
        yerr = errors
        plt.errorbar(x,y,yerr=yerr,fmt='o')
        plt.show()
    else:
        data = pkl.load(open('../GridWorld/fchcFILE.pkl', 'r'))
        x = np.arange(len(data)/1000)
        idx = np.array(x)*1000
        y = np.array(data)[idx]
        plt.plot(x,y)
        plt.show()

if __name__ == "__main__":
    env = Environment(cart_mass=1,pole_mass=0.1,pole_half_length=0.5,start_position=0,start_velocity=0,start_angle=0,start_angular_velocity=0)
    mdp = MDP(env,1,debug=False)
    # mdp.learn_policy_bbo_multiprocessing(init_population=100, best_ke=10, num_episodes=10, epsilon=1e-2, num_iter=500, sigma=10)
    mdp.learn_policy_bbo_multiprocessing(init_population=100, best_ke=10, num_episodes=10, epsilon=1e-2, num_iter=20, sigma=10)
    # mdp.learn_policy_fchc(num_iter=500*15*10, sigma=10, num_episodes=10)
    generate_graphs(cond=False)
    