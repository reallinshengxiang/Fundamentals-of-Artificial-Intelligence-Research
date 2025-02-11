# -*- coding: utf-8 -*-
"""
Created on Mon Aug 24 13:07:16 2020

Description: Clusters model fixed points based on input topology

Must pass this script a text file formatted as follows:
---------------------------------------
bptt 4000 4001 4002
ga 4000 4003
...
---------------------------------------
where each line starts with the name of the learing rule and each subsequent entry
on that line is the model number. Dashes are used in this comment only to indicate
start and end of text file and should not be included in the actual text file. in 
the case of the file above, the first line in the text file should read: "bptt 4000 4001 4002"

The file can not contain any blank lines
"""
import numpy as np
from rnn import loadRNN, RNN
from sklearn.manifold import MDS
from sklearn.neighbors import NearestNeighbors
import matplotlib.pyplot as plt
import os
import sys
import pdb
import argparse
from FP_Analysis import Roots
import rnnanalysis as analyze
from sklearn.metrics import silhouette_score

N_level = 1                  # set to N value of CDI task or 1 for RDM task
N_NEIGHBORS = 3*N_level      # How many neighbors to consider in clustering
N_FIXED_POINTS = 10*N_level  # how many fixed points to consider in clustering

def find_fixed_points(modelPath):
    '''finds fixed points for a model'''
    # look at text file to see how many inputs rnn has and then decide on task based on the input size
    # call the appropriate function from perturbation experiments
    f = open(modelPath+".txt", 'r')
    hyperParams = {}
    for line in f:
        key, value = line.strip().split(':')
        hyperParams[key.strip()] = float(value.strip())
    f.close()
    if hyperParams['inputSize'] == 4:
        analyze.context_fixed_points(modelPath, 'small', save_fp=True)
    elif hyperParams['inputSize'] == 1:
        analyze.rdm_fixed_points(modelPath, 'small', save_fp=True)
    else:
        analyze.N_fixed_points(modelPath, 'small', save_fp=True)

def key2array(key):
    '''
    converts fixed point input key to an array with type float

    Parameters
    ----------
    key : string
        input key value in fixed points dictionary.

    Returns
    -------
    array (float)
        input value for fixed point associated with key.

    '''
    return np.array(float(key[1:-1]))

def parse_learning_rule(str):
    '''parses learning rule from name string'''
    if str[0].lower() == 'b':
        return 'bptt'
    elif str[0].lower() == 'g':
        return 'ga'
    elif str[0].lower() == 'h':
        return 'heb'
    elif str[0].lower() == 'f':
        return 'ff'
    else:
        raise NameError("Ensure the first word on each line of text " \
            "file starts with a letter designating the learning rule (b/g/h/f), got", str[0])

def getMDS(modelNum, learningRule="bptt"):
    '''
    Creates an embedding matrix from MDS based on input topology of fixed points
    for a single RNN. 

    Parameters
    ----------
    modelNum : int
        specifies the RNN model to load.
    learningRule : string, optional
        specifies which model to load based on the learning rule. The default is "bptt".

    Returns
    -------
    MDS_embedding : NumPy array
        embedding of networks fixed points based on input topology two nearest 
        neighbors. Has shape (num_fixed_points, 3).

    '''
    learningRule = parse_learning_rule(learningRule)
    #modelPath = 'models\\' + learningRule + '_' + modelNum
    modelPath = 'models\\' + modelNum
    
    roots = Roots()
    try:  # load roots
        roots.load(modelPath)
    except FileNotFoundError as e:
        find_fixed_points(modelPath)    # solves for and saves model fixed points
        roots.load(modelPath)            # load newly found fixed points

    print(modelPath)
    inpt_values = np.array(roots._static_inputs)[:,0]  #model._fixedPoints[:,1]
    fixed_points = np.squeeze(np.array(roots._values))
    
    nbrs = NearestNeighbors(n_neighbors=N_NEIGHBORS, algorithm='ball_tree').fit(fixed_points)
    distances, indices = nbrs.kneighbors(fixed_points)
    # indices are (num_fixed_points, 3) and represent the two nearest neighbors to the fixed point
    # indexed by the first column
    
    MDS_embedding = np.array(inpt_values)[indices]
    num_fixed_points_found = 5
    start_idx = 0
    end_idx = N_FIXED_POINTS
    
    return MDS_embedding[start_idx:end_idx].reshape(N_FIXED_POINTS,N_NEIGHBORS)

# determines what analysis to run
parser = argparse.ArgumentParser(description="Clusters RNNs by Topology of Fixed Points")
parser.add_argument("fname", help="name of file containing RNNs to analyze")
args = parser.parse_args()

a_file = open("models/"+args.fname, 'r')
list_of_lists = [(line.strip()).split() for line in a_file]
a_file.close()

embeddings = []
names = []
max_fixed_points = 0
start_ix = []
end_ix = []
counter = 0
new_line = True
num_lists = len(list_of_lists)

numModelsOfType = {}   # indicates how many models of each type we have
count = 0
for list_ix in range(num_lists):
    start_ix.append(counter)
    for model_num in list_of_lists[list_ix]:
        if new_line:
            names.append(model_num)
            new_line = False
            continue
        #num = int(model_num)
        embeddings.append(getMDS(model_num, learningRule=names[-1]).reshape(1,-1))
        counter += 1
    end_ix.append(counter)
    new_line = True
start_ix.append(counter)  # last element of start ix is the total
embeddings = np.squeeze(np.array(embeddings))

# pad embeddings to account for differences in number of fixed points between RNNs
# padded_embeddings = -np.ones((len(embeddings), max_fixed_points))
# for rnn_num, embedding in enumerate(embeddings):
#     padded_embeddings[rnn_num:rnn_num+1, :embedding.shape[1]] = embedding


# plot MDS clusters
true_labels = np.zeros((start_ix[-1]))
true_labels -= 1
for ix in start_ix[:-1]:
    true_labels[ix:] += 1
print("score:", silhouette_score(embeddings, true_labels))
clustering_algorithm = MDS()
clustered_data = clustering_algorithm.fit_transform(embeddings)
plt.figure()


if num_lists != 4 and num_lists != 2:
    N_colors = ['r', 'orange', 'g', 'b', 'indigo', 'violet']  # use when plotting against noise levels
else:
    N_colors = ['blue', 'orange', 'green', 'red']
c_ix = 0
for ix in range(num_lists):
    plt.scatter(clustered_data[start_ix[ix]:end_ix[ix],0], clustered_data[start_ix[ix]:end_ix[ix],1], c=N_colors[c_ix])
    c_ix += 1
plt.legend(names) 
plt.title("Silhouette Score " + str(silhouette_score(embeddings, true_labels)))


plt.show()