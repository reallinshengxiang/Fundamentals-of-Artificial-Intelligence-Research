# -*- coding: utf-8 -*-
"""
Created on Thu Jul 30 19:31:36 2020


This script trains RNN models using a specified learning rule
"""
from genetic import Genetic
from bptt import Bptt
import argparse
import pdb 
from task.Ncontext import Ncontext
from trainNcontext import train_ff, train_ff_rdm

hyperParams = {                  # dictionary of all RNN hyper-parameters
   "inputSize" : 1,
   "hiddenSize" : 50,
   "outputSize" : 1,
   "g" : 1 ,
   "inputVariance" : 0.5,
   "outputVariance" : 0.5,
   "biasScale" : 0,
   "initScale" : 0.3,
   "dt" : 0.1,
   "batchSize" : 500,
   "taskMean" : 0.1857,
   "taskVar" : 1,
   "ReLU" : 0
   }

def TrainRDM(name, hyperParams, use_ReLU=False):
    if use_ReLU:
        print("Using ReLU activations")
        hyperParams["ReLU"] = 1

    print(name.lower()[:2])
    if name.lower()[:2] == "bp":
        rnnModel = Bptt(hyperParams)
    elif name.lower()[:2] == "he":
        raise NotImplementedError()
    elif name.lower()[:2] == "ga" or name.lower()[:2] == "ge":      # use genetic learning rule
        rnnModel = Genetic(hyperParams)
    elif name.lower()[:2] == "ff":
        train_ff_rdm(args.model_name, args.variance)  # train
        return # this model has its own saver method
    else:
        print("unclear which learning rule should be used for training")
        raise NotImplementedError()
    rnnModel.setName(name)
    rnnModel.train()
    rnnModel.save()


def TrainMulti(name, hyperParams, use_ReLU=False):
    if use_ReLU:
        print("Using ReLU activations")
        hyperParams["ReLU"] = 1

    print(name.lower()[:2])
    if name.lower()[:2] == "bp":
        rnnModel = Bptt(hyperParams, task="multi", lr=1e-4)
    elif name.lower()[:2] == "he":
        raise NotImplementedError()
    elif name.lower()[:2] == "ga" or name.lower()[:2] == "ge":      # use genetic learning rule
        rnnModel = Genetic(hyperParams, task="multi")
    else:
        print("unclear which learning rule should be used for training")
        raise NotImplementedError()
    # trains the  network according to learning rule specified by name    
    rnnModel.setName(name)
    rnnModel.train()
    rnnModel.save()
    
def TrainContext(name, hyperParams, use_ReLU=False):
    if use_ReLU:
        print("Using ReLU activations")
        hyperParams["ReLU"] = 1
    if name.lower()[:2] == "bp":
        print("begining training BPTT network")
        rnnModel = Bptt(hyperParams, task="context")
    elif name.lower()[:2] == "he":
        raise NotImplementedError()
    elif name.lower()[:2] == "ga" or name.lower()[:2] == "ge":      # use genetic learning rule
        rnnModel = Genetic(hyperParams, task="context", mutation=0.005, numPop=20)
    else:
        print("unclear which learning rule should be used for training")
        raise NotImplementedError()
    rnnModel.setName(name)
    rnnModel.train()
    rnnModel.save()
    
def TrainN(name, hyperParams, use_ReLU=False, N=3):
    task = Ncontext(mean=hyperParams["taskMean"], var=hyperParams["taskVar"], dim=N)
    if use_ReLU:
        print("Using ReLU activations")
        hyperParams["ReLU"] = 1
    if name.lower()[:2] == "bp":
        print("begining training BPTT network")
        rnnModel = Bptt(hyperParams, task=task)
    elif name.lower()[:2] == "ff":
        train_ff(args.model_name, args.N, args.variance)  # train using full force
        return # full force takes care of saving model
    elif name.lower()[:2] == "he":
        raise NotImplementedError()
    elif name.lower()[:2] == "ga" or name.lower()[:2] == "ge":      # use genetic learning rule
        rnnModel = Genetic(hyperParams, task=task, mutation=0.005, numPop=20)
    else:
        print("unclear which learning rule should be used for training")
        raise NotImplementedError()
    rnnModel.setName(name)
    rnnModel.train()
    rnnModel.save()


parser = argparse.ArgumentParser(description="Trains RNN models")
parser.add_argument("model_name", help="name of model")
parser.add_argument("-m", "--mean", type=float, help="input mean", default=0.1857)
parser.add_argument("-v", "--variance", type=float, help="input variance", default=0.5)
parser.add_argument("--relu", action="store_true", default=False)
task_choice = parser.add_mutually_exclusive_group()
task_choice.add_argument("--rdm", action="store_true", default=False)
task_choice.add_argument("--context", action="store_true", default=False)
task_choice.add_argument("--multi", action="store_true", default=False)
task_choice.add_argument("--N", type=int, default=0)

args = parser.parse_args()
hyperParams["taskVar"] = args.variance
hyperParams["taskMean"] = args.mean
if args.rdm:
    TrainRDM(args.model_name, hyperParams, use_ReLU=args.relu)
elif args.context:
    hyperParams["inputSize"] = 4
    TrainContext(args.model_name, hyperParams, use_ReLU=args.relu)
elif args.multi:
    print("Training network on multisensory integration task!")
    hyperParams["inputSize"] = 2
    hyperParams["g"] = 1
    hyperParams["hiddenSize"] = 50
    TrainMulti(args.model_name, hyperParams)
elif args.N != 0:
    print("Training network on N dimensional context task")
    hyperParams["inputSize"] = args.N * 2
    hyperParams["g"] = 1
    hyperParams["hiddenSize"] = 50
    TrainN(args.model_name, hyperParams, N=args.N, use_ReLU=args.relu)
else:
    print("Please specify a training task")
    exit()  