# -*- coding: utf-8 -*-
"""
Created on Mon Oct  5 23:08:11 2020

"""

import time
import FF_Demo
from task.Ncontext import Ncontext
from task.williams import Williams
import pickle          # will be used to save model parameters
# import argparse

# parser = argparse.ArgumentParser(description="Trains RNN models")
# parser.add_argument("model_name", help="name of model")
# parser.add_argument("-v", "--variance", type=float, help="input variance", default=0.1)
# parser.add_argument("-N", type=int, help="Number of dimensions", default = 3)
# args = parser.parse_args()
def train_ff(model_name, N, var):
    task = Ncontext(mean=0.1857, var=var, device="cpu", dim=N)   # FULL FORCE doesn't use cuda
    inps_and_targs = task.get_inps_and_targs
    
    # create the network and set hyper-parameters
    p = FF_Demo.create_parameters(dt=0.003)
    p['g'] = 1.0
    p['network_size'] = 50
    p['tau'] = 0.03
    p['test_init_trials']=10
    p['test_trials'] = 2000
    p['ff_alpha'] = 10000
    p['ff_steps_per_update']=2
    rnn = FF_Demo.RNN(p,N*2,1)   # hyper-params, num_inputs, num_outputs
    
    # train the current model
    num_train_attempts = 0
    training_exit_status = False
    while (not training_exit_status):
        num_train_attempts += 1
        training_exit_status = rnn.train_rdm(inps_and_targs, errorTrigger=200)
    
    
    
    # create a dictionary containing the desired model data to be saved
    model_data = {
            'input_weights': rnn.rnn_par['inp_weights'],
            'rec_weights' : rnn.rnn_par['rec_weights'],
            'output_weights' : rnn.rnn_par['out_weights'],
            'bias' : rnn.rnn_par['bias'],
            'activity_tensor' : rnn.activity_tensor,
            'trgets' : rnn.activity_targets,
            'losses' : rnn.losses,
            'w_rec_hist' : rnn.Wrec_hist,
            'w_i_hist' : rnn.Win_hist,
            'valHist' : rnn.valHist
            }
       
    
    # converts to RNN object
    p["inputSize"] = N*2
    p["hiddenSize"] = p["network_size"]
    p["outputSize"] = 1
    p["inputVariance"] = 0.5
    p["outputVariance"] = 0.5
    p["biasScale"] = 0
    p["initScale"] = 0.3
    p["dt"] = 0.1
    p["batchSize"] = 1
    p["taskVar"] = var
    p["taskMean"] = 0.1857
    p["num_train_attempts"] = num_train_attempts
    
    import sys
    from rnn import RNN
    import torch
    
    
    my_rnn = RNN(p)    # constructs RNN object
    # convert the model
    
    	
    inp_weights = rnn.rnn_par['inp_weights'].T
    rec_weights = rnn.rnn_par['rec_weights'].T
    out_weights = rnn.rnn_par['out_weights'].T
    bias = rnn.rnn_par['bias']
    bias2 = torch.from_numpy(bias).float()
    my_rnn._activityTensor = rnn.activity_tensor
    my_rnn.AssignWeights(inp_weights, rec_weights, out_weights)
    my_rnn._J['bias'][:] = bias2.t()[:]
    my_rnn._targets = rnn.activity_targets
    my_rnn.losses = rnn.losses
    
    	
    my_rnn._MODEL_NAME = model_name
    my_rnn.save()
    
    import shutil
    
def train_ff_rdm(model_name, var):
    task = Williams(mean=0.1857, variance=var, device="cpu")  # task must b
    inps_and_targs = task.get_inps_and_targs
    
    # create the network and set hyper-parameters
    p = FF_Demo.create_parameters(dt=0.003)
    p['g'] = 1
    p['network_size'] = 50
    p['tau'] = 0.03
    p['test_init_trials']=10
    p['test_trials'] = 2_000
    p['ff_alpha'] = 50_000
    p['ff_steps_per_update']=2
    rnn = FF_Demo.RNN(p,1,1)   # hyper-params, num_inputs, num_outputs
    
    # train the current model
    num_train_attempts = 0
    training_exit_status = False
    while (not training_exit_status):
        num_train_attempts += 1
        training_exit_status = rnn.train_rdm(inps_and_targs, errorTrigger=200)
    
    
    
    # create a dictionary containing the desired model data to be saved
    model_data = {
            'input_weights': rnn.rnn_par['inp_weights'],
            'rec_weights' : rnn.rnn_par['rec_weights'],
            'output_weights' : rnn.rnn_par['out_weights'],
            'bias' : rnn.rnn_par['bias'],
            'activity_tensor' : rnn.activity_tensor,
            'trgets' : rnn.activity_targets,
            'losses' : rnn.losses,
            'w_rec_hist' : rnn.Wrec_hist,
            'w_i_hist' : rnn.Win_hist,
            'valHist' : rnn.valHist
            }
       
    
    # converts to RNN object
    p["inputSize"] = 1
    p["hiddenSize"] = p["network_size"]
    p["outputSize"] = 1
    p["inputVariance"] = 0.5
    p["outputVariance"] = 0.5
    p["biasScale"] = 0
    p["initScale"] = 0.3
    p["dt"] = 0.1
    p["batchSize"] = 1
    p["taskVar"] = var
    p["taskMean"] = 0.1857
    p["num_train_attempts"] = num_train_attempts
    
    import sys
    from rnn import RNN
    import torch
    
    
    my_rnn = RNN(p)    # constructs RNN object
    # convert the model
    
    	
    inp_weights = rnn.rnn_par['inp_weights'].T
    rec_weights = rnn.rnn_par['rec_weights'].T
    out_weights = rnn.rnn_par['out_weights'].T
    bias = rnn.rnn_par['bias']
    bias2 = torch.from_numpy(bias).float()
    my_rnn._activityTensor = rnn.activity_tensor
    my_rnn.AssignWeights(inp_weights, rec_weights, out_weights)
    my_rnn._J['bias'][:] = bias2.t()[:]
    my_rnn._targets = rnn.activity_targets
    my_rnn.losses = rnn.losses
    
    	
    my_rnn._MODEL_NAME = model_name
    my_rnn.save()