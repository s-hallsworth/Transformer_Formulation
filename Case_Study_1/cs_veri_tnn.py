"""
This code is created based on code from: https://github.com/cog-imperial/PartitionedFormulations_NN/blob/main/src/optimalAdversary.py

This file contains implementations of the Optimal Adversary problem described in
Section 4.1 of the manuscript. The file can be run directly, with the following parameters.
Parameters can be changed on lines 16-18.
    
    Parameters:
        problemNo (int): index of problem to be solved, can be in range(100)
        N (int): number of partitions to use
        epsilon (real): maximum l1-norm defining perturbations
"""

import numpy as np
import gurobipy as gb

problemNo = 0 # image to select from MNIST dataset
N = 1 # number of partitions = 1 --> big M formulation
epsilon = 5

nLayers = 3
instances = np.load(r'.\data\mnist2x50instances.npz')
inputimage = instances['images'][problemNo]
labels = instances['labels'][problemNo] # [true label, adversary label]

print(instances['images'])

#BEGIN OPTIMIZATION MODEL
# model = gb.Model()
# x = {}; y = {}; z2 = {}; sig = {}

# # Create input nodes; layer '0' is the input
# x[0] = {}; z2[0] = {}; sig[0] = {}
# for i in range(instances['w1'].shape[1]):
#     x[0][i] = model.addVar(max(inputimage[i] - epsilon, 0), min(1, inputimage[i] + epsilon), name='x_' + str(0) + '_' + str(i)) #purturbed image
#     y[i] = model.addVar(0, epsilon) # purturbation at location i of image (lb:0, ub:epsilon) 
    
#     # constrain purturbation to be greater than or equal to diff between target and adversary image
#     model.addConstr(y[i] >= x[0][i] - inputimage[i]) 
#     model.addConstr(y[i] >= inputimage[i] - x[0][i])

# # l1-norm constraint
# model.addConstr(sum(y[i] for i in range(instances['w1'].shape[1])) <= epsilon)
# model.update()

# Set objectivve
model.setObjective(-(x[ind+1][labels[1]] - x[ind+1][labels[0]])) #default is min. Thus equiv to max: x[ind+1][labels[1]] - x[ind+1][labels[0]]

model.setParam('MIPFocus',3) # 3: focus on improving  the dual bound. https://www.gurobi.com/documentation/current/refman/mipfocus.html
model.setParam('Cuts',1)     # 1: Moderate cut generation. https://www.gurobi.com/documentation/current/refman/cuts.html 
model.setParam('Method', 1)  # 1: dual simplex used to solve continuous model or root node relaxation
model.setParam('TimeLimit',3600)
#model.setParam('DisplayInterval', 50)
           
# model.optimize()
    



        
        
    
