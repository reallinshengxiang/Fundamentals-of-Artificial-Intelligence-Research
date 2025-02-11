# -*- coding: utf-8 -*-
"""
Created on Mon Aug 24 14:05:52 2020

context dependent integration task
"""

import torch
import numpy as np
import matplotlib.pyplot as plt

class context_task():
    def __init__(self, N=750, mean=0.5, var=1):
        self.N = N
        self._mean = mean
        self._var = var
        self._version = ""
        self._name = "context"
        
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
        inpts = torch.zeros((self.N, 4)).cuda()
        
        #randomly draw mean from distribution
        means = torch.rand(2) * self._mean
        if mean_overide != -1:
            means[0] = mean_overide
            means[1] = mean_overide
        
        # randomly generates context 1
        if torch.rand(1).item() < 0.5:
            inpts[:,0] = means[0]*torch.ones(self.N)
        else:
            inpts[:,0] = -means[0]*torch.ones(self.N)
            
        # randomly generates context 2
        if torch.rand(1).item() < 0.5:
            inpts[:,1] = means[1]*torch.ones(self.N)
        else:
            inpts[:,1] = -means[1]*torch.ones(self.N)
            
        # randomly sets GO cue
        if torch.rand(1).item() > 0.5:
            inpts[:, 2] = 1
            target = torch.sign(torch.mean(inpts[:,0]))
        else:
            inpts[:,3] = 1
            target = torch.sign(torch.mean(inpts[:,1]))
        
        # adds noise to inputs
        inpts[:,:2] += self._var*torch.randn(750, 2).cuda()
        
        return inpts, target
    
    def PsychoTest(self, coherence, context=0):
        inpts = torch.zeros((self.N, 4)).cuda()
        inpts[:,0] = coherence*torch.ones(self.N)                # attended signal       changed 0->2
        inpts[:,1] = 2*(torch.rand(1)-0.5)*0.1857*torch.ones(self.N)   # ignored signal  changed 1->3
        if context==0:  # signals attended signal
             inpts[:, 2] = 1    # changed 2 - >0
             
        elif context==1: # attends to the ignored signal
             inpts[:, 3] = 1    # changed 3 -> 1
        else:
            raise ValueError("Inappropriate value for context")
        inpts[:,:2] += self._var*torch.randn(750, 2).cuda()    # adds noise to inputs
        
        assert(inpts.shape[1] == 4)
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
        else:
            # use loss from Mante 2013
            squareLoss = (yt-torch.sign(target.T))**2 + (ys - 0)**2
            meanSquareLoss = torch.sum( squareLoss, axis=0 )
            return meanSquareLoss


if __name__ == '__main__':
    import os,sys,inspect
    currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
    parentdir = os.path.dirname(currentdir)
    sys.path.insert(0,parentdir) 
    from rnn import loadRNN, RNN
    
    model = loadRNN("Heb_1000")
    
    
    task = context_task()
    inpts, target = task.GetInput()
    model_output = model.feed(torch.unsqueeze(inpts.t(),0))
    inpts = inpts.cpu().detach().numpy()
    print("target:", target.item())
    plt.figure(1)
    plt.plot(inpts[:,0])
    plt.plot(inpts[:,1])
    plt.legend(["Context 1", "Context 2"])
    
    
    plt.figure(2)
    plt.plot(inpts[:,2])
    plt.plot(inpts[:,3])
    plt.legend(["Go 1", "Go 2"])
    
    plt.figure()
    plt.plot(model_output.detach().cpu().numpy())