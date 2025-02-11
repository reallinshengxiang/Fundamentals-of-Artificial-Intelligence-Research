'''
implementation of fixed point analysis

Date: May 30, 2019
'''

import numpy as np
#import tensorflow as tf
from scipy.optimize import fsolve
from scipy.optimize import root
from sklearn.manifold import TSNE
from sklearn.manifold import LocallyLinearEmbedding as LLE
import matplotlib.pyplot as plt 
import numpy.linalg as LA
import time
from sklearn.decomposition import PCA
import sys
from AnhilationPotential import *           
import torch
import pdb
import numpy as np 
import rnntools as r
import pickle

NUM_ITERS = 100

class Roots(object):
    """The fixed points for a trained RNN model.

    Attributes:
        flight_speed     The maximum speed that such a bird can attain.
        nesting_grounds  The locale where these birds congregate to reproduce.

    Methods:
        FindFixedPoints   Solves for the roots
        FindSlowPoints    Solves for regions of slow dynamics
        getNumRoots       Returns number of roots (for a specific input)
        stable            Returns True if root is stable, otherwise False
        plot              Plots the roots in PC space
        save              Saves the roots
        load              Loads roots from file
    """
    def __init__(self, rnn_model=None):

        self._stability = []          # stability of each root found
        self._static_inputs = []      # static input used for each root
        self._values = []             # each element of value will be a numpy array of roots corresponding to a static input #self._values[static_input][root_num] = np.array(that root)
        self._embedded = []           # list of embedded roots
        self._model = rnn_model       # rnn model for which roots will be found
        self._slow_points = []

        # private
        self._progress_fraction = 0

    def FindFixedPoints(self, static_inpts):
        '''Solves for the models fixed points under static input conditions
        
        functions is a list of functions for which we desire to find the roots
        most likley, each function in the list corresponds to a recurrent neural
        network update function, (dx/dt) = F(x), under a different input condition

        Parameters
        ----------
        model : RNN object
            trained model for which we want to find the fixed points.

        Returns
        -------
        TYPE
            DESCRIPTION.

        '''
        rnn_update_eq = self._model.GetF()
        num_static_inpts = len(static_inpts)
        static_inpts = np.array(static_inpts)                                   # inpts is array with shape (num_static_inputs, input_dim)
        num_roots = np.zeros((len(static_inpts), 1+len(static_inpts[0])))              # use this line for non-context tasks
        
        F = []         # holds RNN update functions under different static inputs
        for static_input in static_inpts:
            F.append(rnn_update_eq(static_input))

        
        #find the roots of each desired function
        print('\nSEARCHING FOR ZEROS ... ')
        labels = []
        stability_flag = []                 # denotes stability of fixed point
        for IX, static_input in enumerate(static_inpts):                     # loop over static inputs
            num_roots[IX,:len(static_inpts[0])] = static_input               # update summary table
            self._updateStatusBar()    # reports progress to console
            roots_found = []
            if (not FindZeros(F[IX], roots_found, num_hidden=self._model._hiddenSize)):  # no root found
                print("No root was found !")
                num_roots[IX, -1] = 0
                continue

            unique_roots = GetUnique(roots_found)
            for root in unique_roots:
                self._values.append(root)
                self._static_inputs.append(static_input)
                self._stability.append(IsAttractor(root, F[IX]))

            #curr_roots = self._values[-1]
            num_roots[IX,-1] = len(unique_roots)
            # end loop over roots associated with current static input

    def FindSlowPoints(self):
        '''Solves for the models fixed points under static input conditions
        
        functions is a list of functions for which we desire to find the roots
        most likley, each function in the list corresponds to a recurrent neural
        network update function, (dx/dt) = F(x), under a different input condition

        Parameters
        ----------
        model : RNN object
            trained model for which we want to find the fixed points.

        Returns
        -------
        TYPE
            DESCRIPTION.

        '''
        rnn_update_eq = self._model.GetF()
        num_inputs = self._model._inputSize
        if num_inputs == 4:  # context task
            zero_input = np.zeros((num_inputs, 2))
            zero_input[2, 0] = 1
            zero_input[3, 1] = 1
        elif num_inputs == 6:  # Ncontext task
            zero_input = np.zeros((num_inputs, 3)) #6,3
            zero_input[3, 0] = 1    # GO 1 on
            zero_input[4, 1] = 1    # GO 2 on
            zero_input[5, 2] = 1    # GO 3 on
        else:
            zero_input = np.zeros((num_inputs, 1))
            
        num_slow_regions = zero_input.shape[1]
        for ix in range(num_slow_regions):        
            F = rnn_update_eq(zero_input[:,ix])         
            
            #find all unique roots under zero input condition
            print('\nSEARCHING FOR SLOW POINTS ... ')
            slow_pts = []
            if (not FindZeros(F, slow_pts, num_hidden=self._model._hiddenSize, tol=1)):
                print("Failed to find any slow points")
                return False
            tmp = GetUnique(slow_pts)
            if ix == 0:
                self._slow_points = np.array(tmp)
            else:
                self._slow_points = np.vstack((self._slow_points, np.array(tmp)))
        return True
        
        #self._slow_points = np.squeeze(np.array(self._slow_points))

    def getNumRoots(self, static_input=None):
        '''returns the number of roots corresponding to static_input
        or returns the total number of roots if static_input is none'''
        totalNumRoots = 0
        for static_input_ix in range(len(self._values)):
            currNumRoots = len(self._values[static_input_ix])
            if (self._static_inputs[static_input_ix] == static_input).all():
                return currNumRoots
            else:
                totalNumRoots += currNumRoots
        return totalNumRoots

    def _embed(self, save_fixed_points=False, mean_overide=1, pulse=True):
        # perform PCA on trajectories to get embedding
        cs = ['r', 'r', 'r', 'r', 'r', 'b', 'b', 'b', 'b', 'b']
        # TODO: check that record is working properly inside this function call
        trial_data, self._labels = r.record(self._model, \
            title='fixed points', print_out=False, plot_recurrent=False, cs=cs,
            mean_overide=mean_overide, pulse=pulse)
        self._model._pca = PCA()
        self._trajectories = self._model._pca.fit_transform(trial_data.reshape(-1, self._model._hiddenSize)).reshape(10,-1,self._model._hiddenSize)
        # model_trajectories is (t_steps, hiddenSize)
        assert(self._trajectories.shape[1]>=self._model._task.N)
        assert(self._trajectories.shape[2]==self._model._hiddenSize)
            
        num_fixed_pts = len(self._values)
        fixed_pts = np.squeeze(np.array(self._values))    # cast fixed points as NumPy array
        if num_fixed_pts == 1:   # must reshape to a single row matrix 
            fixed_pts = fixed_pts.reshape(1,-1)   # data contains a single sample

        if fixed_pts != []:
            roots_embedded = self._model._pca.transform(fixed_pts)
            self._embedded = roots_embedded

    def plot(self, fixed_pts=False, slow_pts=True, plot_traj=True, plot_PC1=False, start_time = 0, end_time=-1):
        '''Plots the embedded fixed points in two dimensions

        Parameters
        ----------
        roots_embedded : NumPy array
            contains the embedded roots along with stability flag (first column) and 
            static_input (second column) has shape (num_roots, 2+hidden_size).

        Returns
        -------
        None.
        '''
        print("0000")
        if self._embedded == []:
            self._embed()
        cs = ['g', 'b', 'r']
        plt.figure()
        if fixed_pts and self._embedded != []:
            n_roots = self._embedded.shape[0]
            if len(self._static_inputs) > 10:  # use high resolution map
                colors = high_res_cmap 
            else:  # use standard color map when there are less than 44 static inputs
                colors = cmap 
            for root_ix in range(n_roots):  # loop over roots
                if self._stability[root_ix]:  # root is stable
                    plt.scatter(self._embedded[root_ix, 0], self._embedded[root_ix, 1], edgecolors=colors(self._static_inputs[root_ix]), alpha=0.5, s=500, facecolors='none', linewidth=3)
                else:    # root is unstable
                    pass # don't plot unstable roots!               
                    #plt.scatter(self._embedded[root_ix, 0], self._embedded[root_ix, 1], marker='x', c=cmap(self._static_inputs[root_ix]), alpha=0.5)
     

        if plot_traj:
            for i in range(10): # plot the trajectories
                if start_time != 0:
                    pass
                    plt.plot(self._trajectories[i,:start_time,0], self._trajectories[i,:start_time,1], c = 'k', alpha=0.1)
                plt.plot(self._trajectories[i,start_time:end_time,0], self._trajectories[i,start_time:end_time,1], c = cs[int(self._labels[i])], alpha=0.25)
            
        if slow_pts:    # plot the slow points
            if self._slow_points == []:    # slow points have not been found yet
                self.FindSlowPoints()
        
            num_slow_pts = len(self._slow_points)
            slow_pts = np.squeeze(np.array(self._slow_points))    # cast slow points as NumPy array
            if num_slow_pts == 1:   # must reshape to a single row matrix 
                slow_pts = slow_pts.reshape(1,-1)
            if slow_pts != []: # if we were able to find any slow points
                slow_embedded = self._model._pca.transform(slow_pts)
                for ix in range(num_slow_pts):
                    plt.scatter(slow_embedded[ix, 0], slow_embedded[ix, 1], c='k', marker='x', alpha=0.1)
        
        if plot_PC1:  # plot PC1 against time
            inpt_colors = ['r', 'g', 'b', 'y', 'm', 'c']
            for ii, mean in enumerate([0, 0.2, 0.6]):
                self._embed(mean_overide=mean, pulse=False)
                for i in range(10):
                    plt.plot(self._trajectories[i,start_time:end_time,0], c = inpt_colors[ii], alpha=0.25)
            

    def save(self, fname):
        fname+= ".pkl"
        with open(fname, 'wb') as output:
            pickle.dump(self, output)

    def load(self, fname):
        fname += ".pkl"
        with open(fname, 'rb') as inpt:
            tmp = pickle.load(inpt)

            self._stability = tmp._stability          # stability of each root found
            self._static_inputs = tmp._static_inputs      # static input used for each root
            self._values = tmp._values             # each element of value will be a numpy array of roots corresponding to a static input #self._values[static_input][root_num] = np.array(that root)
            self._embedded = tmp._embedded           # list of embedded roots
            self._model = tmp._model 
         
    def cluster(self):
        pass

    def _updateStatusBar(self):
        '''updates the status of a task and prints it to console'''
        self._progress_fraction += 1
        sys.stdout.write('\r')
        sys.stdout.write('[%-19s] %.2f%% ' %('='*self._progress_fraction, 5.26*self._progress_fraction))
        sys.stdout.flush()

######################################################################
# Auxillary Functions
######################################################################
def FindZeros(F, result, num_hidden=50, tol=1e-8, norm=False):
    '''
    FindZeros takes a function F and will search for zeros
    using randomly generated initial conditions
    '''
    roots_found = []

    for _ in range(NUM_ITERS):
        #random activations on U[-1,1]
        x0 = 10*(np.random.rand(num_hidden,1)-0.5)
        # tolerance changed from 1e-8
        sol = root(F, x0, tol=tol, method='lm')   # lm
        if sol.success == True:
            if norm:
                #if not a zero vector
                if LA.norm(sol.x) != 0:
                    curr_root = np.round( sol.x/LA.norm(sol.x), decimals=3 )
                #else don't take the norm (because it is a zero vector)
                else:
                    curr_root = np.round( sol.x, decimals=3 )
            else:
                curr_root = sol.x
            roots_found.append(curr_root)

    #print('roots found:', roots_found.shape)
    if len(roots_found) == 0: # failed to find any roots
        return False   
    
    for point in roots_found: # add each root found to the results list
        result.append(point)
    return True

def ComputeDistance(point1, point2):
    '''Computes euclidean distance between two points'''
    point_dimension = len(point1)
    squared_distances = 0
    for idx in range(point_dimension):
        squared_distances += (point1[idx] - point2[idx])**2
    distance = np.sqrt(squared_distances)
    return distance

def IsAttractor(fixed_point, F, NumSimulations=25):   #NumSimulations=2500
    '''
    IsAttractor will determine if a fixed point is stable or unstable

    returns True if point is stable and False if point is unstable
    '''
    num_stable_iters = 0
    num_unstable_iters = 0

    #reformat fixed point
    original_shape = fixed_point.shape
    fixed_point = fixed_point.reshape(-1,1)

    for simulation in range(NumSimulations):
        epsilon = 10e-5 * np.random.randn(len(fixed_point), 1)                 #10e-3
        epsilon[1] = 1
        epsilon[10] = 1
        epsilon[11] = -1
        nearby_point = fixed_point+epsilon
        initial_distance = ComputeDistance(nearby_point, fixed_point)
        for iterator in range(100):
            nearby_point += F(nearby_point).reshape(-1,1)
        #end of iterations
        final_distance = ComputeDistance(nearby_point, fixed_point)
        if final_distance > initial_distance:
            num_unstable_iters += 1
            #print('iteration diverged...')
        else:
            num_stable_iters += 1
            #print('iteration converged...')
    #end of simulations
    if num_unstable_iters >= 1:
        return False
    else:
        return True

def cmap(static_inpt, max_inpt=3):
    '''
    generates a color for plotting fixed point found under static_input. Colors 
    are generated from the following look-up table:
        |INPUT (%of max)     COLOR
        |>80                 blue
        |>60                 orange
        |>40                 green
        |>20                 red
        |>0                  purple
        |>-20                brown
        |>-40                pink
        |>-60                grey
        |>-inf               olive
    This color scheme was chosen to maximize contrast over a continuous color 
    gradient. With a continuous color gradient nearby points were difficulty to
    discern. The colors in the lookup table may be changed to further enhance 
    contrast.

    Parameters
    ----------
    static_inpt : float
        input value to network for which current fixed point was found.

    Returns
    -------
    list
        r,g,b color that should be used to plot current fixed points.

    '''
    # create the lookup table 
    lookup = {}
    lookup[-4] = 'tab:blue'
    lookup[-3] = 'tab:orange'
    lookup[-2] = 'tab:green'
    lookup[-1] = 'tab:red'
    lookup[0] = 'tab:purple'
    lookup[1] = 'tab:brown'
    lookup[2] = 'tab:pink'
    lookup[3] = 'tab:gray'
    lookup[4] = 'tab:olive'
    lookup['!'] = 'k'     # special case of exactly zero

    # DETERMINE THE MAXIMUM INPUT VALUE
    # case 1: the static_input is two dimensional as in the multisensory task
    # we just use the maximum of the two channels
    if static_inpt.shape[0] == 4:     # context task
        ix_of_nonzero_inpt = np.nonzero(static_inpt[:2])
        if len(ix_of_nonzero_inpt[0]) == 0:    # both inputs are zero
            static_inpt = 0
        else:
            static_inpt = static_inpt[ix_of_nonzero_inpt][0]
        max_inpt = 0.4
    elif static_inpt.shape[0] == 6:     # Ncontext task
        ix_of_nonzero_inpt = np.nonzero(static_inpt[:3])
        if len(ix_of_nonzero_inpt[0]) == 0:    # both inputs are zero
            static_inpt = 0
        else:
            static_inpt = static_inpt[ix_of_nonzero_inpt][0]
        max_inpt = 0.1857
    elif static_inpt.shape[0] == 1:    # RDM task
        static_inpt = static_inpt[0]
        max_inpt = 0.6
    else:
        # TODO: handle Ncontext tasks
        static_inpt = 0
    # END DETERMINE MAXIMUM INPUT
    if static_inpt == 0:
        return 'k'
    else:
        input_level = int(5*static_inpt / max_inpt)
    if input_level > 4:
        input_level = 4
    elif input_level < -4:
        input_level = -4
    return lookup[input_level]


def high_res_cmap(static_inpt, max_inpt=0.02):
    '''
    generates a color for plotting fixed point found under static_input. Colors 
    go from red (positive inputs) to blue (negative inputs)
    Parameters. This function is only appropriate for plotting the attractors 
    near zero.
    ----------
    static_inpt : float
        input value to network for which current fixed point was found.
    Returns
    -------
    list
        r,g,b color that should be used to plot current fixed points.
    '''
    if static_inpt.shape[0] == 4:     # context task
        ix_of_nonzero_inpt = np.nonzero(static_inpt[:2])
        if len(ix_of_nonzero_inpt[0]) == 0:    # both inputs are zero
            static_inpt = 0
        else:
            static_inpt = static_inpt[ix_of_nonzero_inpt][0]
        max_inpt = 0.02

    elif static_inpt.shape[0] == 1:    # RDM task
        static_inpt = static_inpt[0]
        max_inpt = 0.03
        
    else:   # we don't care about the colors for other tasks
        return [[0, 0, 0]]
    
    # clamps input so RGGB values remain in range
    if static_inpt > max_inpt:
        static_inpt = max_inpt
    elif static_inpt < -max_inpt:
        static_inpt = -max_inpt
        
    if static_inpt == 0:
        return [[0,0,0]]  # return black for THE zero input
    
    m_r = -0.5 / max_inpt
    m_b = 0.5 / max_inpt
    
    r = m_r * static_inpt + 0.5
    b = m_b * static_inpt + 0.5
    g = m_r*0
    return [[r, g, b]]
