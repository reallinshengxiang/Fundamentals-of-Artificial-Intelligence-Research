# here I will define a task wher the desired output is a constant value
# const_val = 5
import numpy as np
import utils
import pdb
import torch


class ConstOutput():
    const_val = 5  # global variable

    def GetInput(self, N=40, mean=1, variance=1):
        inp = utils.GetGaussianVector(mean, variance, N)  # changed from 0.5
        inp_param = 0.0
        return inp, inp_param

    # note that here the desired output is baked into the loss function
    # ideally here I can make the function work with either torch or numpy
    def Loss(self, y, input_params=0):
        return ((y - self.GetDesired()) ** 2).mean()

    def GetDesired(self):
        return 25
