'''
delayed non-match to sample
----segment of RDM-----delay-----segment of rdm----
Network must output if the two RDM segments have the same net sign

lets start with an easier version of this task, say 0.5 input mean

standard trial:
1s rest
500ms fixation
650ms of sample
1s rest
250ms test
1s choice period

we will shorten this to
500ms sample, 500ms rest, 250ms test
'''

import torch
import numpy as np
import pdb
class DMC():
    def __init__(self, N=750, mean=0.5, var=0.05):
        self.N = N
        self._mean = mean
        self._var = var
        self._version = ""
        self._name = "dnms"
        
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
        inpts = 0.2*torch.ones((self.N, 2)).cuda()   # inputs are timesteps by 2 channels
        
        '''
        #randomly draw mean from distribution
        mean = torch.rand(1).item() * self._mean
        if mean_overide != -1:
            mean = mean_overide
        '''
        
        # randomly choose channel to be active
        if torch.rand(1).item() < 0.5:   # channel one is active
            inpts[:300,0] = torch.ones(300)
            sample_channel=0
        else:                            # signal is negative
            inpts[:300,1] = torch.ones(300)
            sample_channel=1

        # adds noise to sample stimulus
        #inpts[:200,0] += self._var*torch.randn(200).cuda()

        # randomly generates test stimulus
        if torch.rand(1).item() < 0.5:   # channel zero is active
            inpts[400:700,0] = torch.ones(300)
            test_channel=0
        else:                            # channel two is active
            inpts[400:700,1] = torch.ones(300)
            test_channel=1
        # adds noise to sample stimulus
        inpts += self._var*torch.randn(self.N, 2).cuda()
         
        # determine target
        if (sample_channel == test_channel):
        	target = torch.tensor(1.0)
        else:
        	target = torch.tensor(-1.0)
        
        return inpts, target

    def Loss(self, y, target, errorTrigger=-1):
        '''
        inputs: y
        '''
        if (errorTrigger != -1):
            yt = y[errorTrigger:]
            print("y", torch.mean(yt[:]).item())
        else:
            yt = y[-1]
        ys = y[0]
        if type(y) is np.ndarray:
            assert False, "input to loss must be a PyTorch Tensor"
        else:
            #pdb.set_trace()
            #pdb.set_trace()
            # use loss from Mante 2013
            squareLoss = (yt-torch.sign(target.T))**2 + (ys - 0)**2
            meanSquareLoss = torch.sum( squareLoss, axis=1 ) / 500

            #print("MSE:", meanSquareLoss.item())
            return meanSquareLoss

# construct

if __name__=="__main__":
    import matplotlib.pyplot as plt
    import pdb
    print("Delayed Categorical Match Task Called Directly! \n")
    dnms = DMC()

    for i in range(10):
        signal, target = dnms.GetInput()
        signal = signal.detach().cpu().numpy()
        plt.figure(i)
        plt.plot(signal[:,0])
        plt.plot(signal[:,1])
        plt.legend(["channel 1", "channel 2"])
        plt.ylim([-0.5 ,1.5])
        if target==-1:
            plt.title("Match")
        else:
            plt.title("Non-Match")
    plt.show()