import os

class Pole():

    def __init__(self, pole_half_length, pole_mass):
        self.mass = pole_mass
        self.length = pole_half_length*2
        self.angular_velocity = 0 
        self.theta = 0