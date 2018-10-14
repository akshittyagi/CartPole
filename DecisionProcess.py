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

    def __init__(self, env, gamma, g=9.8, f=10, time_step=0.02, fail_angle=np.deg2rad(90), terminate_time=20.2):
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
        print("Time Steps: ", time_counter, " Total Reward: ", total_reward)

    def print_state(self, s_t):
        print("Pos: ", s_t[0], " Velocity: ", s_t[1], " Theta: ", np.rad2deg(s_t[2]), " Omega: ", np.rad2deg(s_t[3]))

    def learn_policy(self, num_episodes, policy):
        pass

if __name__ == "__main__":
    env = Environment(cart_mass=1,pole_mass=0.1,pole_half_length=0.5,start_position=0,start_velocity=0,start_angle=0,start_angular_velocity=0)
    mdp = MDP(env,1)
    mdp.run_episode('random')

    

    