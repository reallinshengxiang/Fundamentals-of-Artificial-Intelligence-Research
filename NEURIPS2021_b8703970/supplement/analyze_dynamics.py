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
task_type = parser.add_mutually_exclusive_group()
task_type.add_argument("--N", type=int, default=0)

parser.add_argument("model_name", help="filename of model to analyze")
parser.add_argument("--save_fp", action="store_true", default=False)
parser.add_argument("input_choice", 
                    help="choice of inputs ('large' or 'small') to analyze")

args = parser.parse_args()
##############################################################

if args.N == 0:  # RDM Task
    rdm_fixed_points('models/'+args.model_name, args.input_choice)
else:            # N-dimensional context task
    plt.figure()
    print("abstract N dimensional context")
    N_fixed_points('models/'+args.model_name, args.input_choice, verbose=False)

plt.show()