import os

class Pole():

    def __init__(self, pole_half_length, pole_mass, start_angle, start_angular_velocity):
        self.mass = pole_mass
        self.length = pole_half_length
        self.omega = start_angular_velocity
        self.theta = start_angle
