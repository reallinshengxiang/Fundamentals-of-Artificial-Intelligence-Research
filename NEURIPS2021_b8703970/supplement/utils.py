# This is a script with useful utility functions
import numpy as np
import torch


def GetGaussianVector(mean, std, N):
    if std > 0:
        temp = std*torch.randn(N,1)+mean
        #temp = np.random.normal(mean, std, N).reshape((N, 1))
        return temp
    else:
        return mean * torch.ones(N,1)


# this is useful for the backpropagation code
def GetGaussianVectorTorch(mean, std, N):
    return torch.normal(mean=mean, std=std * torch.ones(N, 1))
