'''
Generates dynamical trajectories and attractor topologies for trained 
RNNs

'''
from rnnanalysis import *
from task.williams import Williams
import numpy.linalg as LA
import numpy as np
import matplotlib.pyplot as plt
import argparse
import FP_Analysis as fp

##############################################################
# determines what analysis to run
##############################################################
parser = argparse.ArgumentParser(description="Analyzes trained RNNs")
parser.add_argument("model_name", help="filename of model to analyze")
args = parser.parse_args()
##############################################################

weights_and_outputs('models/'+args.model_name)
plt.show()