import numpy as np
import os
import torch
from gurobipy import GRB
from gurobi_ml import add_predictor_constr
import torchvision

## Import from repo files
from veri_tnn_exp import *   #import problem definition and TNN architecture
from MINLP_tnn.helpers.print_stats import save_gurobi_results
import MINLP_tnn.helpers.convert_pyomo as convert_pyomo
from MINLP_tnn.helpers.GUROBI_ML_helper import get_inputs_gurobipy_FFN

# turn off floating-point round-off
os.environ["TF_ENABLE_ONEDNN_OPTS"] = '0' 

"""
Module for running image classification verification experiments with the MNIST dataset.

This script defines and executes experiments to solve verification problems for 
Transformer-based Neural Networks (TNNs) using the MNIST dataset. The experiments 
vary in setup, including active constraints, TNN files, and solver frameworks.

The key functionalities of the script include:
1. Configuring the experiment with different image sizes, TNN models, and constraint setups.
2. Defining and activating constraints for the TNN formulation.
3. Creating optimization models with Transformer constraints.
4. Converting the optimization models to the Gurobi framework for solving.
5. Logging experiment configurations and results for further analysis.

Main Steps:
1. **Setup**: Define experimental parameters such as image size, solver framework, 
   and repetition count.
2. **Constraint Configuration**: Configure active constraints and cuts for the 
   Transformer Neural Network (TNN) formulation.
3. **Data Preparation**: Load the MNIST dataset and preprocess images for the defined image size.
4. **Model Definition**: Define and load the TNN model and its constraints into the 
   optimization problem.
5. **Optimization and Results Logging**: Solve the optimization problem using Gurobi, 
   and log results including bounds, cuts, and decision variables.

Key Variables:
- `TESTING`: Enables faster testing by reducing the number of repetitions.
- `ACTI`: Defines groups of constraints for Layer Normalization (LN) and Multi-Head Attention (MHA).
- `combinations`: Specifies combinations of active constraints.
- `tnn_config`: Stores experiment results and hyperparameters.

Usage:
Run the script to perform the experiments as defined in the configurations. Adjust 
the parameters in the "TO DO" sections to customize the experiment setup.
"""


## Set Up
### ------------------ TO DO: CONFIG EXPERIMENTS ----------------------###
TESTING = False # fix TNN input for testing (faster solve)
if TESTING: 
    REP = 1
else:
    REP = 1             # number of repetitions of each scenario
r_offset = 0            # number to start repetitions count at
SOLVER = "gurobi"
FRAMEWORK = "gurobipy"
EXPERIMENT = "test"     # experiment name used to name output files
im_sz=[4]               # Define image pixels (and folder to select tnn models from)
file_names = ["vit_12_1_6_12", "vit_12_2_6_12", "vit_12_4_6_12"] # changing depth
### ---------------------------------------------------------------------###


## Define Transformer Constraint config (# which bounds and cuts to apply to the MINLP TNN formulation)
ACTI_LIST_FULL = [ 
            "LN_var", "LN_mean", "LN_num", "LN_num_squ", "LN_denom", "LN_num_squ_sum",
                "MHA_Q", "MHA_K", "MHA_V", "MHA_attn_weight_sum", "MHA_attn_weight",
            "MHA_compat", "MHA_compat_exp", "MHA_compat_exp_sum", "MHA_QK_MC", "MHA_WK_MC", "MHA_attn_score", "MHA_output", 
            "RES_var", "MHA_softmax_env", "AVG_POOL_var", "embed_var"]
activation_dict = {}
for key in ACTI_LIST_FULL:
    activation_dict[key] = False
### ------------------ TO DO: SET BOUND+CUT CONFIGS ----------------------### 
# define configuartions  
combinations = [ 
    [1 , 0, 1, 1, 0], #2 -- fastest feasibile solution on trajectory problem _/  No Mc
    [1 , 0, 1, 0, 0], #3 -- I only
    [1 , 0, 0, 1, 0], #6 -- fastest optimal solution on trajectory problem _/ LNprop
    [1 , 0, 1, 1, 1], #1 -- all
]
### ----------------------------------------------------------------------###

combinations = [[bool(val) for val in sublist] for sublist in combinations]
ACTI = {}  
ACTI["LN_I"] = {"list": ["LN_var"]}
ACTI["LN_D"] = {"list": ["LN_num", "LN_num_squ", "LN_denom"]}
ACTI["MHA_I"] = {"list": ["MHA_attn_weight_sum", "MHA_attn_weight"]}
ACTI["MHA_D"] = {"list": ["MHA_Q", "MHA_K", "MHA_V", "MHA_compat", "MHA_compat_exp", "MHA_compat_exp_sum", "MHA_attn_score", "MHA_output" , "RES_var"]}
ACTI["MHA_MC"] = {"list":[ "MHA_QK_MC", "MHA_WK_MC"]}



## Define Optimisation Problem:
### ------------------ TO DO: SET SPROBLEM DEFINITION PARAMS ----------------------###
problemNo = 0 # image to select from MNIST dataset ( test: 0 --> the number 7)
epsilon = 0.0001 ##
channels = 1
patch_size=2 
classification_labels = 10
adv_label = 1
### -------------------------------------------------------------------------------###



## RUN EXPERIMENTS:
tnn_config={} # dict storing additional log information which is later converted to csv

# For varied pixel sized images
for image_size in im_sz: 
    
    # Set output directory
    PATH =  f".\\Experiments\\Verification_{image_size*image_size}_{EXPERIMENT}"
    if not os.path.exists(PATH): # Create directory if does not exist
        os.makedirs(PATH)
        os.makedirs(PATH+"\\Logs")
    PATH += "\\"
    
    
    # Load Data Set with pixels = image_size
    torch.manual_seed(42)
    DOWNLOAD_PATH = '/data/mnist'
    transform_mnist = torchvision.transforms.Compose([ torchvision.transforms.Resize((image_size, image_size)), ##
                                                    torchvision.transforms.ToTensor(),
                                    torchvision.transforms.Normalize((0,), (1,))])
    mnist_testset = torchvision.datasets.MNIST(DOWNLOAD_PATH, train=False, download=True, transform=transform_mnist)
    test_loader = torch.utils.data.DataLoader(mnist_testset, batch_size=1, shuffle=False)
    images, labels = next(iter(test_loader))
    inputimage = images[problemNo] # flattened image
    labels = [labels[problemNo].item(), adv_label] # [true label, adversary label]
    
    
    # Load problem definition
    model, input = verification_problem(inputimage, epsilon, channels, image_size, labels, classification_labels)


    # For each trained TNN
    for file_name in file_names:
        
        # For each experiment repetition
        for r in range(REP):
            
            # For each constraint combination
            for c, combi in enumerate(combinations):
                print("Configuration, C = ", c+1)    
                experiment_name = f"{file_name}_i{image_size}_r{r+1+r_offset}_c{c+1}"
                
                # Activate constraints
                ACTI["LN_I"]["act_val"], ACTI["LN_D"]["act_val"], ACTI["MHA_I"]["act_val"] , ACTI["MHA_D"]["act_val"], ACTI["MHA_MC"]["act_val"] = combi
                for k, val in ACTI.items():
                    for elem in val["list"]:
                        activation_dict[elem] = val["act_val"]          # set activation dict to new combi
                tnn_config["Activated Bounds/Cuts"] = activation_dict   # save act config


                # Create a clone of the optimization model
                m = model.clone()
                
                
                # Define Transformer Model:
                script_dir = os.path.dirname(os.path.abspath(__file__))
                tnn_path = f".\\training\\models\\verification_{image_size}x{image_size}\\{file_name}.pt" 
                tnn_path= os.path.join(script_dir, tnn_path)
                device = 'cpu'
                print("TNN file: ", file_name, tnn_path)
                
                m, ffn_parameter_dict, layer_outputs_dict, transformer = verification_tnn(m, inputimage, image_size, patch_size, channels, file_name, tnn_path, activation_dict, device) # create TNN architecture
                tnn_config["TNN Out Expected"] = np.array(list(layer_outputs_dict['mlp_head'])[0].tolist()).flatten() # store trained TNN output


                # Convert optimization model to gurobipy
                gurobi_model, map_var, _ = convert_pyomo.to_gurobi(m)
                
                
                # Add FNNs to gurobi model using GurobiML
                for key, value in ffn_parameter_dict.items():
                    nn, input_nn, output_nn = value
                    input, output = get_inputs_gurobipy_FFN(input_nn, output_nn, map_var)
                    pred_constr = add_predictor_constr(gurobi_model, nn, input, output)
                gurobi_model.update() # update gurobi model with FFN constraints



                # Define Solve settings
                ### ------------------ TO DO: SET SOLVE SETTINGS ----------------------###
                gurobi_model.setParam('MIPFocus',1) 
                gurobi_model.setParam('LogToConsole', 0)
                gurobi_model.setParam('OutputFlag', 1)
                gurobi_model.setParam('LogFile', PATH+f'Logs\\{experiment_name}.log')
                gurobi_model.setParam('TimeLimit', 16200) #4.5h
                ### -------------------------------------------------------------------###
                
                ## Optimize
                gurobi_model.optimize()
                
                ## Get decision variables
                if gurobi_model.status == GRB.OPTIMAL:
                    optimal_parameters = {}
                    for v in gurobi_model.getVars():
                        #print(f'var name: {v.varName}, var type {type(v)}') #print variable name and type
                        if "[" in v.varName:
                            name = v.varname.split("[")[0]
                            if name in optimal_parameters.keys():
                                optimal_parameters[name] += [v.x]
                            else:
                                optimal_parameters[name] = [v.x]
                        else:    
                            optimal_parameters[v.varName] = v.x
                    
                    # Save values of purturb and output label
                    tnn_config["max purturb"] = np.max(np.array(optimal_parameters["purturb"])) 
                    tnn_config["min purturb"] = np.min(np.array(optimal_parameters["purturb"]))
                    tnn_config["TNN Out"] = np.array(optimal_parameters["output"])  
                else:
                    tnn_config["TNN Out"] = None
                    
                    
                # Store transformer hyper-parameters
                tnn_config["Enc Seq Len"] = transformer.N
                tnn_config["TNN Model Dims"] = transformer.d_model
                tnn_config["TNN Head Dims"] = transformer.d_k
                tnn_config["TNN Head Size"] = transformer.d_H
                tnn_config["TNN Input Dim"] = transformer.input_dim


                if not TESTING:
                    # Save tnn_config dict as csv file
                    save_gurobi_results(gurobi_model, PATH+experiment_name, experiment_name, r+1+r_offset, tnn_config)