import numpy as np 
import numpy.linalg as LA
import torch
from rnn import RNN, loadRNN, create_activity_tensor 
 
import rnntools as r
from FP_Analysis import Roots
import time
from task.williams import Williams
from task.context import context_task
from task.multi_sensory import multi_sensory
from task.dnms import DMC
from sklearn.decomposition import PCA
import json
import sys
import pdb
import matplotlib.pyplot as plt
from task.Ncontext import Ncontext

def weights_and_outputs(modelPath):
    model = loadRNN(modelPath)
    '''Plots sample outputs and weights sorted by neuron factor'''
    cs = ['r', 'r', 'r', 'r', 'r', 'b', 'b', 'b', 'b', 'b']
    trial_data, trial_labels = r.record(model, \
        title='fixed points', print_out=True, plot_recurrent=False, cs=cs, pulse=False, mean_overide=0.1857)

    ###########################################################################
    #TENSOR COMPONENT ANALYSIS
    ###########################################################################
    if np.all(model._activityTensor == 0):  # get the activity tensor if it is empty
        print("generating activity tensor...")
        create_activity_tensor(model)
    activity_tensor = model._activityTensor
    neuron_factor = r.plotTCs(activity_tensor, model._targets, 1)
    neuron_idx = np.argsort(neuron_factor)     # sorted indices of artificial neurons
    
    # find the index where neuron_factors changes sign
    # p is the index that partitions neuron_idx into two clusters
    sign_change_idx = np.diff(np.sign(neuron_factor[neuron_idx]))
    if not np.all(sign_change_idx==0):
        # executes when neuron factors are not all the same sign
        last_pos_neuron = np.nonzero(sign_change_idx)[0][0]
        print('number of positive neurons:', last_pos_neuron)
        print('number of negative neurons:', len(neuron_factor)-last_pos_neuron)
        p = np.nonzero(np.diff(np.sign(neuron_factor[neuron_idx])))[0][0] + 1
    else:
        # executes when neuron factors all have the same sign
        print('artifical neurons all have sign', np.sign(neuron_factor[0]))
        p = 25
    
    
    # ###########################################################################
    # #VISUALIZE RECURRENT WEIGHTS
    # ###########################################################################
    model._neuronIX = neuron_idx
    model.VisualizeWeightMatrix()
    model.VisualizeWeightClusters(neuron_idx, p)

def rdm_fixed_points(modelPath, inputs, save_fp=False):
    assert (inputs == 'large' or inputs == 'small' or inputs == 'sparse'), "Must use either small, large, or sparse inputs"
    model = loadRNN(modelPath)
    #AnalyzeLesioned(model, modelPath, xmin, xmax, ymin, ymax)
    
    inpts = {}  # two sets of inputs for solving fixed points
    inpts['large'] = np.array ( [[0.5], [0.2], [0], [-0.2], [-0.5]] )
    inpts['small'] = 0.03 * np.array( [[0], [0.1], [-0.1], [0.2], [-0.2], [0.3], [-0.3], [0.4], [-0.4], [0.5], 
                                [-0.5], [0.6], [-0.6], [0.7], [-0.7], [0.8], 
                                [-0.8], [0.9], [-0.9], [1.0], [-1.0]] )
    inpts['sparse'] = 0.03 * np.array( [[0], [0.2], [-0.2], [0.4], [-0.4], [0.6], [-0.6], [0.8], 
                                [-0.8], [1.0], [-1.0]] )
    
    input_values = inpts[inputs]   # user specified input set

    model_roots = Roots(model)
    model_roots.FindFixedPoints(input_values)  # compute RNNs fixed points
   
    if inputs=="large":
        model_roots.plot(fixed_pts=True, slow_pts=True, end_time = 50)
        plt.title("Early")
        model_roots.plot(fixed_pts=True, slow_pts=True, start_time=50, end_time=200)
        plt.title("Mid")
        model_roots.plot(fixed_pts=True, slow_pts=True, start_time=200)
        plt.title("Late")
    
    plt.figure(2)
    model_roots.plot(fixed_pts=True, slow_pts=False, plot_traj=False)
    plt.title("Model Attractors") 
    
    if inputs=="large":
        plt.figure()
        model_roots.plot(fixed_pts=False, slow_pts=False, plot_traj=False, plot_PC1=True)
        plt.title("PC1")
        plt.xlabel("Time")
        plt.ylabel("PC1")
            
    
    if save_fp:
        model_roots.save(modelPath)   


def context_fixed_points(modelPath, inputs, save_fp=False):
    model = loadRNN(modelPath)
    model._task = Ncontext(device="cpu", dim=2)

    # construct inputs
    if inputs=='small':
        fixed_point_resolution = 21
        tmp = np.linspace(-0.02, 0.02, fixed_point_resolution)    
        tmp = tmp[np.argsort(np.abs(tmp))]
        static_inpts = np.zeros((2*fixed_point_resolution, 4))
        static_inpts[0::2, 2] = 1
        static_inpts[0::2, 0] = tmp
        static_inpts[1::2, 3] = 1
        static_inpts[1::2, 1] = tmp
                                 # go signal for color context
    elif inputs == 'large':
        fixed_point_resolution = 5
        tmp = np.linspace(-0.4, 0.4, fixed_point_resolution)    
        tmp = tmp[np.argsort(np.abs(tmp))]
        static_inpts = np.zeros((2*fixed_point_resolution, 4))
        static_inpts[0::2, 2] = 1
        static_inpts[0::2, 0] = tmp
        static_inpts[1::2, 3] = 1
        static_inpts[1::2, 1] = tmp
       

    model_roots = Roots(model)
    model_roots.FindFixedPoints(static_inpts)


    plt.figure(100)
    plt.title("PCA of Fixed Points For Contextual Integration Task")
    
    
    model_roots.plot(fixed_pts=True, slow_pts=True, end_time = 100)
    plt.title("Early")
    model_roots.plot(fixed_pts=True, slow_pts=True, start_time = 100, end_time = 300)
    plt.title("Mid")
    model_roots.plot(fixed_pts=True, slow_pts=True, start_time = 400)
    plt.title("Late")
    
    model_roots.plot(fixed_pts=True, slow_pts=True)

    plt.figure()
    model_roots.plot(fixed_pts=True, slow_pts=False, plot_traj=False)
    plt.title("Model Attractors")

    plt.figure(123)
    plt.title('Evaluation of Model on Multisensory Task')   

    if save_fp:
        model_roots.save(modelPath)   

def N_fixed_points(modelPath, inputs, save_fp=False, verbose=False):
    
    model = loadRNN(modelPath)
    n_contexts = int(model._inputSize / 2)
    model._task = Ncontext(device="cpu", dim=n_contexts)
    # construct inputs
    if inputs=='small':
        fixed_point_resolution = 21
        static_inpts = np.zeros((n_contexts*fixed_point_resolution, n_contexts*2))
        for context_count in range(n_contexts):
            tmp = np.linspace(-0.02, 0.02, fixed_point_resolution)    
            tmp = tmp[np.argsort(np.abs(tmp))]
            static_inpts[context_count::n_contexts, context_count + n_contexts] = 1  # skip n_context rows at a time
            static_inpts[context_count::n_contexts, context_count] = tmp
                                 # go signal for color context
    elif inputs == 'sparse':
        fixed_point_resolution = 11
        tmp = np.linspace(-0.02, 0.02, fixed_point_resolution)     
        tmp = tmp[np.argsort(np.abs(tmp))]
        static_inpts = np.zeros((2*fixed_point_resolution, 4))
        static_inpts[0::2, 2] = 1
        static_inpts[0::2, 0] = tmp
        static_inpts[1::2, 3] = 1
        static_inpts[1::2, 1] = tmp
        
    elif inputs == 'large':
        fixed_point_resolution = 5
        tmp = np.linspace(-0.4, 0.4, fixed_point_resolution)    
        tmp = tmp[np.argsort(np.abs(tmp))]
        static_inpts = np.zeros((2*fixed_point_resolution, 4))
        static_inpts[0::2, 2] = 1
        static_inpts[0::2, 0] = tmp
        static_inpts[1::2, 3] = 1
        static_inpts[1::2, 1] = tmp
       

    model_roots = Roots(model)
    model_roots.FindFixedPoints(static_inpts)
    if verbose:
        plt.figure(100)
        plt.title("PCA of Fixed Points For Contextual Integration Task")
        
        
        model_roots.plot(fixed_pts=True, slow_pts=True, end_time = 100)
        plt.title("Early")
        model_roots.plot(fixed_pts=True, slow_pts=True, start_time = 100, end_time = 300)
        plt.title("Mid")
        model_roots.plot(fixed_pts=True, slow_pts=True, start_time = 400)
        plt.title("Late")
        
        model_roots.plot(fixed_pts=True, slow_pts=True)
    
        plt.figure()
        model_roots.plot(fixed_pts=True, slow_pts=False, plot_traj=False)
        plt.title("Model Attractors")
    
        plt.figure(123)
        plt.title('Evaluation of Model on Multisensory Task')   

    if save_fp:
        model_roots.save(modelPath)   

    if verbose:
        plt.show()