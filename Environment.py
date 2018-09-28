import os

from Cart import Cart
from Pole import Pole

class Environment():

    def __init__(self, Cart, Pole):
        self.cart = Cart
        self.pole = Pole
        self.limit = 3
        self.start_position = 0

Actions = {"left":-1, "right":+1}