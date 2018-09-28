import os

from Cart import Cart
from Pole import Pole

class Environment():

    def __init__(self, cart_mass, pole_mass, pole_half_length, start_position, start_velocity, start_angle, start_angular_velocity):
        self.track_limits = 3
        self.start_position = start_position
        self.start_velocity = start_velocity
        self.start_angle = start_angle
        self.start_angular_velocity = start_angular_velocity
        self.cart = Cart(cart_mass, start_position, start_velocity)
        self.pole = Pole(pole_half_length, pole_mass, start_angle, start_angular_velocity)

Actions = {"left":-1, "right":+1}