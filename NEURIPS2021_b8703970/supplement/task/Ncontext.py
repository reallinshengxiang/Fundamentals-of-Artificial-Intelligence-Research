# -*- coding: utf-8 -*-
"""
Created on Mon Aug 24 14:05:52 2020

context dependent integration task
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
from random import randrange

class Ncontext():
    def __init__(self, N=750, mean=0.1857, var=1, device="cuda:0", dim=3):
        if torch.cuda.is_available() and device=="cuda:0":
            self._device = torch.device(device)
        else:
            self._device = torch.device("cpu")
            
        self._dim = dim            # number of input contexts
        self.N = N               # trial duration
        self._mean = mean
        self._var = var
        self._version = ""
        self._name = "Ncontext"

    def _random_generate_input(self, mean):
        '''randomly generates an input channel'''
        if torch.rand(1).item() < 0.5:
            return mean*torch.ones(self.N)
        return -mean*torch.ones(self.N)

    def get_inps_and_targs(self, mean_overide=-1, var_overide=1, dt=""):
        '''returns inputs, hints, and targets for FF training algorithm'''
        inpts, target = self.GetInput(mean_overide=mean_overide, var_overide=var_overide)
        inpts = inpts.cpu().detach().numpy()
        targets = np.ones((self.N, 1))*target.item()
        return inpts, targets, targets
        
    def GetInput(self, mean_overide=-1, var_overide=1):
        '''
        

        Parameters
        ----------
        mean_overide : TYPE, optional
            DESCRIPTION. The default is 1.

        Returns
        -------
        inpts : PyTorch CUDA Tensor
            DESCRIPTION.
        target : TYPE
            DESCRIPTION.

        '''
        inpts = torch.zeros((self.N, 2*self._dim)).to(self._device)
        
        #randomly draw mean from distribution
        mean = torch.rand(1).item() * self._mean
        if mean_overide != -1:
            mean = mean_overide
        
        # randomly generates N input channels
        go_channel = randrange(self._dim)
        for channel_num in range(self._dim):
            inpts[:,channel_num] = self._random_generate_input(mean)
            # generate the GO signals
            if go_channel == channel_num:
                inpts[:, self._dim + channel_num] = 1
                target = torch.sign(torch.mean(inpts[:, channel_num]))
            else:
                inpts[:, self._dim + channel_num] = 0

        
        # adds noise to inputs
        inpts[:,:self._dim] += self._var*torch.randn(self.N, self._dim).to(self._device)
        return inpts, target
    
    def PsychoTest(self, coherence, context=0):
        inpts = torch.zeros((self.N, 2*self._dim)).cuda()
        inpts[:,0] = coherence*torch.ones(self.N)                      # attended signal       changed 0->2
        for i in range(1, self._dim):
            inpts[:,i] = 2*(torch.rand(1)-0.5)*0.1857*torch.ones(self.N)
        #inpts[:,1] = 2*(torch.rand(1)-0.5)*0.1857*torch.ones(self.N)   # ignored signal  changed 1->3
        #inpts[:,2] = 2*(torch.rand(1)-0.5)*0.1857*torch.ones(self.N)

        try:
            inpts[:, self._dim+context] = 1    # attend to specified context
        except IndexError:
            raise ValueError("Inappropriate value for Ncontext PsychoTest")
        inpts[:,:self._dim] += self._var*torch.randn(750, self._dim).cuda()    # adds noise to inputs
        
        assert(inpts.shape[1] == 2*self._dim)
        return inpts
    
    def Loss(self, y, target, errorTrigger=-1):
        if (errorTrigger != -1):
            yt = y[errorTrigger:]
            print("y", torch.mean(yt[:]).item())
        else:
            yt = y[-1]
        ys = y[0]
        if type(y) is np.ndarray:
            assert False, "input to loss must be a PyTorch Tensor"
        # use loss from Mante 2013
        squareLoss = (yt-torch.sign(target.T))**2 + (ys - 0)**2
        meanSquareLoss = torch.sum( squareLoss, axis=0 )
        return meanSquareLoss


if __name__ == '__main__':
    import os,sys,inspect
    currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
    parentdir = os.path.dirname(currentdir)
    sys.path.insert(0,parentdir) 
    
    task = Ncontext(var=0.1, dim=1)
    inpts, target = task.GetInput()
    inpts = inpts.cpu().detach().numpy()
    print("target:", target.item())
    plt.figure(1)
    N = inpts.shape[1] / 2
    N = int(N)
    print("Number of dimensions: ", N)
    plt_legend = []
    for i in range(N):
        plt.plot(inpts[:,i])
        plt_legend.append("Context" + str(i+1))
    plt.legend(plt_legend)
    
    
    plt.figure(2)
    plt_legend = []
    for i in range(N):
        plt.plot(inpts[:,N+i])
        plt_legend.append("Go" + str(i+1))
    plt.legend(plt_legend)