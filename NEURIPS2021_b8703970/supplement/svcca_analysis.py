# -*- coding: utf-8 -*-
"""
Created on Mon Aug 17 19:21:08 2020

"""

import rnn
import numpy as np
import torch
import matplotlib.pyplot as plt
import svcca.cca_core as cca_core
from sklearn.manifold import MDS
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
from sklearn.metrics import silhouette_score
import pdb
import argparse
from FP_Analysis import ComputeDistance

# TODO: refactor code as class
# class NetworkDistances:
#     # public:
#     def getDistances():
#         self._getActivations()
#     # private: 
#     def loadRNNS():  # loads RNNs to be analyzed
#         pass
#     def _getActivations():  # gets the activations of all RNNs
#         pass
    


def getNumSVs(singularValues):
    sv_squares = singularValues**2
    svs_needed = 0
    explained_var = 0
    sv_square_totals = np.sum(sv_squares)
    while (explained_var < 0.95):
        explained_var += (singularValues[svs_needed]**2 / sv_square_totals)
        svs_needed+=1
    return svs_needed

def get_rdm_inputs(inputSize):
    # returns inputs for the RDM task
    #assert False, "you are using RDM, use N=1 network instead"
    NUM_INPUT_CONDITIONS = 10
    static_inputs = np.linspace(-0.1857, 0.1857, NUM_INPUT_CONDITIONS).reshape(1, NUM_INPUT_CONDITIONS)
    static_inputs = np.matmul(np.ones((750, 1)), static_inputs)
    static_inputs = torch.tensor(static_inputs).float()
    static_inputs = torch.unsqueeze(static_inputs.t(), 1)
    return static_inputs
    
def get_static_inputs(inputSize):
    # will get inputs for the N = 1 context task
    # N_INPUTS = 120   # needs to be a multiple of all possible N
    N_INPUTS_PER_CONTEXT = 10
    N_CONTEXTS = int(inputSize / 2)
    #N_INPUTS_PER_CONTEXT = int(N_INPUTS / N_CONTEXTS) 
    MIN_INPUT = -0.1857
    MAX_INPUT = 0.1857
    static_inpt_arr = np.zeros((N_INPUTS_PER_CONTEXT*N_CONTEXTS, inputSize))
    #static_inpt_arr = np.zeros((N_INPUTS_PER_CONTEXT*N_CONTEXTS, inputSize))
    # loop over contexts
    for context in range(N_CONTEXTS):
       strt_ix = context*N_INPUTS_PER_CONTEXT
       end_ix = (context+1)*N_INPUTS_PER_CONTEXT
       static_inpt_arr[strt_ix:end_ix, context] = np.linspace(MIN_INPUT, MAX_INPUT, N_INPUTS_PER_CONTEXT)  # set inputs
       static_inpt_arr[strt_ix:end_ix, context+N_CONTEXTS] = 1  # set GO signal
    # static_inpt_arr now filled in with shape (20, 8)
    static_inpt_arr = np.expand_dims(static_inpt_arr, -1)  # (20, 8, 1)
    assert static_inpt_arr.shape == (N_INPUTS_PER_CONTEXT*N_CONTEXTS, inputSize, 1)
    inpts = np.matmul(static_inpt_arr, np.ones((1, 750))) # (20, 8, 750)
    # feed batch of 20 inpts into network to get ativations 
    # cast static inputs to PyTorch Tensor
    static_inputs = torch.tensor(inpts).float() # 20, 8, 750
    return static_inputs
        
        
def getActivations(rnn_model):
    '''generates inputs for SVCCA'''
    switcher = {
            1:get_rdm_inputs,
            2:get_static_inputs,
            4:get_static_inputs,
            6:get_static_inputs,
            8:get_static_inputs,
            10:get_static_inputs,
            12:get_static_inputs
        }
    static_inputs = switcher[rnn_model._inputSize](rnn_model._inputSize)
    _, activations = rnn_model.feed(static_inputs, return_hidden=True)
    if rnn_model._useHeb:
        heb_mask = np.ones(50).astype(int)
        heb_mask[1] = 0; heb_mask[10] = 0; heb_mask[11] = 0
        activations = activations[:, np.where(heb_mask==1)[0], :]
        activations = np.swapaxes(activations, 0, 1).reshape(rnn_model._hiddenSize-3, -1)
        return activations #finalActivations    # 50 x 500  ----> numHidden x numInputs
    #finalActivations = activations[-1,:,:]       # keeps activation at end of trial only
    activations = np.swapaxes(activations, 0, 1).reshape(rnn_model._hiddenSize, -1)
    return activations #finalActivations    # 50 x 500  ----> numHidden x numInputs
        
# get a file of models to analyze
parser = argparse.ArgumentParser(description="Clusters RNNs by SVCCA")
parser.add_argument("fname", help="name of file containing RNNs to analyze")
parser.add_argument("-reload", action="store_true", default=False)
args = parser.parse_args()
file_of_rnns = open("models/"+args.fname, 'r')
models = [(line.strip()).split() for line in file_of_rnns]
file_of_rnns.close()


modelActivations = []
NUM_DIMENSIONS = 2

#models = [bptt, ga]
numModelsOfType = {}
count = 0
for i in range(len(models)):
    numModelsOfType[models[i][0]] = len(models[i])-1
    count += len(models[i]) - 1
numModelsOfType["total"] = count


if args.reload:    
    distances = np.zeros((numModelsOfType["total"], numModelsOfType["total"]))
    for modelType in models:
        for count, model in enumerate(modelType):
            if (count == 0):
                print("model name:", model)
            else:
                print("name:", model)
                rnn_inst = rnn.loadRNN("models/"+model)
                activations = getActivations(rnn_inst)
                modelActivations.append(activations)
     
    # compute the distance matrix
    print("computing distance matrix ...\n")
    for i in range(len(modelActivations)):
        print("    Computing row", int(i))
        for j in range(i, len(modelActivations)):   # ~2X speedup using symmetry
            activationsI = modelActivations[i]
            activationsJ = modelActivations[j]
            activationsI -= np.mean(activationsI, axis=1, keepdims=True)
            activationsJ -= np.mean(activationsJ, axis=1, keepdims=True)
            
            # take top 20 singular values
            U1, s1, V1 = np.linalg.svd(activationsI, full_matrices=False)
            U2, s2, V2 = np.linalg.svd(activationsJ, full_matrices=False)
            
            #print("Number of singular values needed to explain 95% of variance:", getNumSVs(s1))
            NUM_DIMENSIONS = getNumSVs(s1)
            svacts1 = np.dot(s1[:NUM_DIMENSIONS]*np.eye(NUM_DIMENSIONS), V1[:NUM_DIMENSIONS])
            NUM_DIMENSIONS = getNumSVs(s2)
            svacts2 = np.dot(s2[:NUM_DIMENSIONS]*np.eye(NUM_DIMENSIONS), V2[:NUM_DIMENSIONS])
            
            if svacts1.shape != svacts2.shape:
                print("SVACTS1:", svacts1.shape)
                print("SVACTS2:", svacts2.shape)
            svcca_results = cca_core.get_cca_similarity(svacts1, svacts2, epsilon=1e-10, verbose=False)
            distances[i, j] = 1 - np.mean(svcca_results["cca_coef1"])
            distances[j, i] = distances[i, j]
            # distances are shape (numRNNs, numRNNs)
    
    # save the distances
    np.save("models/" + args.fname[:-4] + "_distances", distances)
else:
    distances = np.load("models/" + args.fname[:-4] + "_distances.npy")
plt.imshow(distances)
plt.clim(0, 1.0/3)
plt.colorbar()
plt.title("SVCCA Distances Between Networks")

clustering_algorithm = MDS()
clustered_data = clustering_algorithm.fit_transform(distances)



plt.show()


def Method1(rnn_distances, labels, class_counts):
    ''' 
    Model Agnostic Method for measuring how clustered Learning Rules are
    1.) Find the mean for each learning rule
    2.) Compute distance of all points for a given learning rule to the mean 
        for that learning rule (d_lr)
    3.) Compute the distance of all points from other learning rules to the 
        mean for the current learning rule (d_other)
    4.) Compute the ratio of d_lr / d_other
    '''
    N_RNNS = len(rnn_distances)
    # compute the mean/centroid for each learning rule
    centroid_positions = {0: np.zeros((N_RNNS)), 1: np.zeros((N_RNNS)), 2: np.zeros((N_RNNS)), 3: np.zeros((N_RNNS))}
    for rnn_ix in range(N_RNNS):  # loop over all RNNs
        centroid_positions[labels[rnn_ix]] += rnn_distances[rnn_ix]
    # now we must normalize the means
    for class_key in centroid_positions:
        centroid_positions[class_key] /= class_counts[class_key]
    # centroid_positions now contains the centroid for each learning rule

    # loop through all RNNs and get distance from centroids
    ooc_distances = {0:[], 1:[], 2:[], 3:[]}
    within_class_distances = {0:[], 1:[], 2:[], 3:[]}
    for rnn_ix in range(N_RNNS):
        # compute within class distance
        for class_key in centroid_positions:
            if class_key == labels[rnn_ix]: 
                # compute within class distance\
                within_class_distances[class_key].append(ComputeDistance(centroid_positions[class_key], rnn_distances[rnn_ix]))
            else: 
                # compute out of class distance
                ooc_distances[class_key].append(ComputeDistance(centroid_positions[class_key], rnn_distances[rnn_ix]))

    # get the mean of each list in both dictionaries and get the ratio
    ratios = []
    for class_key in within_class_distances:
        ratios.append(np.mean(within_class_distances[class_key])/np.mean(ooc_distances[class_key]))
    return ratios


def Method2(rnn_distances, labels, class_counts):
    ''' 
    Model Agnostic Method for measuring how clustered Learning Rules are
    1.) Compute average distance for any two points in the same learning rule 
        (d_same)
    2.) Compute average distance between any two points (d_all)
    3.) Ratio of d_same / d_all
    '''
    N_RNNS = len(rnn_distances)

    # loop through all RNNs and get distance from all other RNNs
    ooc_distances = {0:[], 1:[], 2:[], 3:[]}
    within_class_distances = {0:[], 1:[], 2:[], 3:[]}
    for rnn_i in range(N_RNNS):
        for rnn_j in range(rnn_i+1, N_RNNS):
            # compute distance between RNN_i and RNN_j
            d = ComputeDistance(rnn_distances[rnn_i], rnn_distances[rnn_j])
            if labels[rnn_i] == labels[rnn_j]:  # update within class
                within_class_distances[labels[rnn_i]].append(d)
            else:  # update out-of-class
                ooc_distances[labels[rnn_i]].append(d)
                ooc_distances[labels[rnn_j]].append(d)
    # get the mean of each list in both dictionaries and get the ratio
    ratios = []
    for class_key in within_class_distances:
        ratios.append(np.mean(within_class_distances[class_key])/np.mean(ooc_distances[class_key]))
    return ratios


def Purity(rnn_distances, predicted, labels):
    '''
    Computes purity of clusters found with SVCCA

    Parameters
    ----------
    rnn_distances : obj
        model will be saved as a .pt file with this name in the /models/ directory.
        
    predicted : list<int>
        list of predicted class for each rnn
        
    labels : list<int>
        list of true class for each rnn

    Returns
    -------
    None.

    for each cluster,
        count the number of data points from the most common learning rule 
        sum this accross all clusters
    divide the running sum by the total number of data points
        
    '''
    N_CLASSES = 4  # total number of rnn classes
    N_CLUSTERS = 4 # total number of rnn clusters (should be same as classes)
    N_RNNS = len(distances)  # total number of rnns clustered
    # dictionary counts the number of points belonging to each class 
    # for each cluster. For example, class_counts[0][1] = 3 would 
    # indicate that there were three points belonging to class 1 in 
    # cluster 3. More generally,
    #    class_counts[cluster_id][class_id] = num_points_in_class_id_and_cluster_id
    #
    class_counts = {0:{0:0, 1:0, 2:0, 3:0}, 
                    1:{0:0, 1:0, 2:0, 3:0}, 
                    2:{0:0, 1:0, 2:0, 3:0}, 
                    3:{0:0, 1:0, 2:0, 3:0}}  
    for ix in range(N_RNNS):  # loop through each RNN
        cluster_id = predicted[ix]  # cluster predicted by model
        class_id = labels[ix]       # true class of rnn
        # increment number of this class contained in this cluster
        class_counts[cluster_id][class_id] += 1  

    # now we must sum the total number of points for each cluster belonging to the maximum 
    # represented class
    max_per_class = np.zeros((N_CLUSTERS, 1))
    for cluster_id in range(N_CLASSES):
        # get the max in this dictionary
        max_pts_found = -1
        for key in class_counts[cluster_id]:
            if class_counts[cluster_id][key] > max_pts_found:  # current class contains the most rnns
                max_pts_found = class_counts[cluster_id][key]
        max_per_class[cluster_id] = max_pts_found
    # max_per_class now contains the number of points for the most represented class in 
    # each cluster
    purity = np.sum(max_per_class, axis=0) / N_RNNS
    return purity

def getTrueLabelsHelper(numModelsOfType):
    '''
    numModelsOfType is a dictionary where 
    each key is a model type and each value
    is the number of models of that type
    '''
    labels = []  # holds the ground truth labels
    class_counts = [0 for i in range(len(numModelsOfType)-1)]
    for class_id, key in enumerate(numModelsOfType):
        if key=="total":
            continue   # don't use totals
        curr_class_count = 0
        for j in range(numModelsOfType[key]):
            labels.append(class_id)
            curr_class_count += 1  # number of rnns in this class
        class_counts[class_id] = curr_class_count
    # note the labels in class_id don't have to 
    # correspond with the labels used by the clustering
    # algorithm since we only consider purity and not 
    # that the labels match
    return labels, class_counts
    
labels, class_counts = getTrueLabelsHelper(numModelsOfType)

if len(numModelsOfType) != 5 and len(numModelsOfType) != 3:
    N_colors = ['r', 'orange', 'g', 'b', 'indigo', 'violet']  # use when plotting against noise levels
else:
    N_colors = ['blue', 'orange', 'green', 'red']              # use when plotting the 4 learning rules
plt.figure(11)  # Scatter plot
count = 0     # keeps track of how many models plotted so far
ci = 0
legends = []  # holds model names to be used in legend
for modelType in numModelsOfType:
    if modelType == "total": # no plotting for total since it isn't a valid model type
        continue
    value = numModelsOfType[modelType]
    if modelType[0] == 'B':
        
        plt.scatter(clustered_data[count:count+value,0], clustered_data[count:count+value,1], c=N_colors[ci])
    else:
        
        plt.scatter(clustered_data[count:count+value,0], clustered_data[count:count+value,1], c=N_colors[ci])
    i+=1
    ci+=1
    count += value  # increment the number of models plotted
    legends.append(modelType) # add this model type name to the legend
    
plt.legend(legends)
plt.title("Sillhouette Score:" + str(silhouette_score(distances, labels)))


plt.show()