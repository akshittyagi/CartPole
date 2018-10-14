import os
import sys
import random
import argparse
import pickle as pkl
import csv
from enum import Enum

import numpy as np
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
        if policy == 'random':
            choice = np.random.randint(0,2)
            if choice%2 == 0:
                return self.actions['left']
            else:
                return self.actions['right']
    
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
        x = x + v*self.time_step
        theta = theta + omega*self.time_step
        a, alpha = self.get_accelerations(f, (x,v,theta,omega))
        v = v + a*self.time_step
        omega = omega + alpha*self.time_step
        if omega > 0:
            omega = min(np.deg2rad(180), omega)
        else:
            omega = max(-np.deg2rad(180), omega)
        if v > 0:
            v = min(10, v)
        else:
            v = max(-10, v)
        return (x,v,theta,omega)
    
    def reward_function(self, s_t, a_t, s_t_1, time_step):
        return 1*(self.gamma**time_step)
    
    def run_episode(self, policy):
        s_t = self.get_init_state()
        total_reward = 0
        time_counter = 0
        while(not self.is_terminal_state(s_t, time_counter)):
            self.print_state(s_t)
            a_t = self.policy(policy, s_t)
            print("Action at time: ", time_counter, " : ", a_t)
            s_t_1 = self.transition_function(s_t, a_t)
            r_t = self.reward_function(s_t, a_t, s_t_1, time_counter)
            total_reward += r_t
            time_counter += 1
            s_t = s_t_1
        self.print_state(s_t)
        if self.debug:
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

    def learn_policy_bbo_multiprocessing(self, init_population, best_ke, num_episodes, epsilon, num_iter, steps_per_trial=15, sigma=10):
        assert init_population >= best_ke
        assert num_episodes > 1
        curr_iter = 0
        reshape_param = (100, 2)
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
    

class multiprocessing_obj(MDP):
        def __init__(self, num_episodes):
            MDP.__init__(self,)
            self.num_episodes = num_episodes
        def __call__(self, theta):
            theta = theta/np.sum(theta, axis=1)[:,None]
            j = self.evaluate(theta, self.num_episodes)
            return theta.reshape(theta.shape[0]*theta.shape[1], 1), j


if __name__ == "__main__":
    env = Environment(cart_mass=1,pole_mass=0.1,pole_half_length=0.5,start_position=0,start_velocity=0,start_angle=0,start_angular_velocity=0)
    mdp = MDP(env,1,debug=True)
    mdp.run_episode('random')

    

    