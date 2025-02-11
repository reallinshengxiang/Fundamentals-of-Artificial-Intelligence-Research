 # here the task is: return a 1 if the average input is > 0, return a 0 if the average input is < 0

import numpy as np
import utils
import torch
import pdb

# note that whenever I call getInput, I'm drawing a sample from the distribution


class Williams():
    def __init__(self, N=750, mean=0.1857, variance=0.333, version="", device="cuda:0"):
        self.N = N
        self._mean = mean
        self.variance = variance
        self._version = version
        self._name = "rdm"
        self._device = device
        
        
    def get_inps_and_targs(self, mean_overide=-1, var_overide=0, dt=""):
        '''returns inputs, hints, and targets for FF training algorithm'''
        inpts, target = self.GetInput(mean_overide=mean_overide, var_overide=var_overide)
        inpts = inpts.cpu().detach().numpy()
        targets = np.ones((self.N, 1))*target.item()
        return inpts, targets, targets
        
    def GetInput(self, mean_overide=-1, var_overide=False):
        '''
        GetInput will randomly generate a positive or negative 
        data sequence of length N

        mean: the data sequence will be centered around plus or
        minus this value

        variance: variance of data sequence
        DEFAULTS
        N=40
        mean=1
        variance=1


        used to be mean +/-1 and variance 1
        testing mean and mean+1 with variance = 0.2
        '''
        inp = torch.zeros((self.N, 1))
        # create a random mean in [0, 1)
        mean = torch.rand(1)*self._mean #self.mean#
        if mean_overide != -1:
            mean = mean_overide
        if torch.rand(1) < 0.5:
            # create a negative input
            if var_overide:
                inp = -mean*torch.ones((self.N,1))
            else:
                inp = utils.GetGaussianVector(-mean, self.variance, self.N)  # changed from 0.5
            condition = torch.tensor([-1]).float()
        else:
            # create a positive input
            if var_overide:
                inp = mean*torch.ones((self.N, 1))
            else:
                inp = utils.GetGaussianVector(mean, self.variance, self.N)  # changed from 0.5
            condition = torch.tensor([1]).float()
            
        #######################################################################
        #if self._version == "Heb":   # reformats input for Hebbian trained RDM networks
        #    inpts = torch.zeros((self.N, 2))
        #    inpts[:,1:2] = inp
        #    inp = inpts
        #######################################################################    
            
        #ensures a PyTorch Tensor object is returned
        if not torch.is_tensor(inp):
            inp = torch.from_numpy(inp).float()
        return inp.to(self._device), condition


    def GetInputPulse(self, mean_overide=-1, var_overide=False):
        '''
        GetInputPulse will randomly generate a positive or negative 
        pulsed data sequence of length N. Data is zero except for the
        initial pulse

        mean: the data sequence will be centered around plus or
        minus this value

        variance: variance of data sequence
        DEFAULTS
        N=40
        mean=1
        variance=1


        used to be mean +/-1 and variance 1
        testing mean and mean+1 with variance = 0.2
        '''
        inp = torch.zeros((self.N, 1))
        # create a random mean in [0, 1)
        mean = torch.rand(1)*self._mean #self.mean#
        if mean_overide != -1:
            mean = mean_overide
        if torch.rand(1) < 0.5:
            # create a negative input
            if var_overide:
                inp[:-700] = -mean*torch.ones((50,1))
            else:
                inp[:-700] = utils.GetGaussianVector(-mean, self.variance, self.N-700)  # changed from 0.5
            condition = torch.tensor([-1]).float()
        else:
            # create a positive input
            if var_overide:
                inp[:-700] = mean*torch.ones((50, 1))
            else:
                inp[:-700] = utils.GetGaussianVector(mean, self.variance, self.N-700)  # changed from 0.5
            condition = torch.tensor([1]).float()
            
        #######################################################################
        #if self._version == "Heb":   # reformats input for Hebbian trained RDM networks
        #    inpts = torch.zeros((self.N, 2))
        #    inpts[:,1:2] = inp
        #    inp = inpts
        #######################################################################    
            
        #ensures a PyTorch Tensor object is returned
        if not torch.is_tensor(inp):
            inp = torch.from_numpy(inp).float()
        return inp.to(self._device), condition

    # TODO: delete GetDesired function
    def GetDesired(self):
        '''
        some classess attempt to make a call to GetDesired instead of GetInput. GetDesired 
        is only a wrapper fo the GetInput function
        '''
        inp, condition = self.GetInput()
        return inp, condition
    
    def PsychoTest(self, coherence, context=0):
        var = self.variance#1-np.abs(coherence)
        inp = utils.GetGaussianVector(coherence, var, self.N)  # changed from 0.5
        #inp = inp.to(torch.device('cuda'))
        return inp

    # note that here the desired output is baked into the loss function
    # TODO: see if there's a better way so I don't need to specify the numpy and torch version
    def Loss(self, y, mu, errorTrigger=-1):
        if errorTrigger != -1:
            yt = y[errorTrigger:]
            print("y", torch.mean(yt[:]).item())
        else:
            yt = y[-1]#errorTrigger:]
        ys = y[0]
        if type(y) is np.ndarray:
            #return np.log(1 + np.exp(-yt * mu))
            return (yt-np.sign(mu))**2
        else:
            # use loss from Mante 2013
            squareLoss = (yt-torch.sign(mu.T))**2 + (ys - 0)**2
            meanSquareLoss = torch.sum( squareLoss, axis=0 ) #/200#/ self.N
            return meanSquareLoss
            #return torch.log(1 + torch.exp(-yt * mu.T))

        # return log(1 + exp(-yt * mu))

if __name__ == '__main__':
    import matplotlib.pyplot as plt 

    task = Williams(variance=0.25)
    for _ in range(1):
        meanOverride = 1
        inp, condition = task.GetInput(mean_overide=meanOverride)
        if condition == 1:
            plt.plot(inp.detach().cpu().numpy()[:20], c='r')
        else:
            plt.plot(inp.detach().cpu().numpy(), c='b')
        plt.ylim([-2,4])
    plt.xlabel('Time')
    plt.ylabel('Input')
    plt.title('Perceptual Decision Making Task')
    print(inp.shape)
    plt.show()

