import pyomo.environ as pyo
import numpy as np
import torch
import math
import json
import os
from helpers.extract_from_pretrained import get_pytorch_learned_parameters, get_hugging_learned_parameters
from omlt import OmltBlock
from omlt.neuralnet import *
from omlt.io.keras import keras_reader
import omlt
import helpers.OMLT_helper 
import helpers.GUROBI_ML_helper as GUROBI_ML_helper
from typing import Union

os.environ["TF_ENABLE_ONEDNN_OPTS"] = '0' # turn off floating-point round-off

def activate_envelope_att(model):
    model.constr_convex.deactivate()
    model.constr_concave.deactivate() 
    model.constr_convex_tp.deactivate()
    model.constr_convex_tp_sct.deactivate()
    model.constr_concave_tp.deactivate()
    model.constr_concave_tp_sct.deactivate()

    if model.s_cv == 0: # --> convex region only
        model.constr_convex.activate()
    elif model.s_cc == 0: # --> concave region only
        model.constr_concave.activate() 
    else: # in both regions
        if model.t_cv == 0: # --> if x <= x_cv_tiepoint -->convex region
            model.constr_convex_tp.activate()
        elif model.t_cv == 1: # -->concave region
            model.constr_convex_tp_sct.activate()
            
        if model.t_cc == 0: # --> if x >= x_cc_tiepoint -->concave region
            model.constr_concave_tp.activate()
        elif model.t_cc == 1:# --> convex region
            model.constr_concave_tp_sct.activate()

class Transformer:
    """ 
    A MINLP formulation of a transformer neural network based on Vaswani et al's "Attention is All You Need" paper.
    
    Attributes:
        hyper_params (Union[list,str]): Hyper-parameters of the TNN model including headsize, number of heads and embedding dimension
        opt_model (pyo.ConcreteModel()): A pyomo optimisation model
        set_bound_cut (dict, optional): A dictionary with boolean values to indicate which bounds and cuts to activate/deactivate.
                              If this value is not specified (None) a default set of bounds and cuts will be used. 
                              This configuration corresponds to "All in the associated thesis document.
    """
    
    def __init__(self, hyper_params:Union[list,str], opt_model, set_bound_cut=None):
        """
        Initializes the Transformer optimization model with specified hyperparameters and bound/cut configurations.

        Args:
            hyper_params (Union[list, str]): 
                Hyperparameters of the Transformer model. Can be provided either as:
                - A list: [sequence_length (N), embedding_dimensions (d_model), head_size (d_k), number_of_heads (d_H), input_dimensions (input_dim), epsilon (layer_norm_epsilon)].
                - A string: Path to a JSON file containing the hyperparameters in the following format:
                ```json
                {
                    "hyper_params": {
                        "N": <sequence_length>,
                        "d_model": <embedding_dimensions>,
                        "d_k": <head_size>,
                        "d_H": <number_of_heads>,
                        "input_dim": <input_dimensions>,
                        "epsilon": <layer_norm_epsilon>
                    }
                }
                ```
            opt_model (pyomo.ConcreteModel): 
                The Pyomo optimization model instance where the Transformer formulation will be added.
            set_bound_cut (dict, optional): 
                A dictionary specifying which bounds and cuts to activate. If None, default values are applied. Keys include:
                - "embed_var"
                - "LN_var"
                - "MHA_softmax_env", "MHA_Q", "MHA_K", "MHA_V", "MHA_attn_weight_sum", "MHA_attn_weight"
                - "MHA_compat", "MHA_compat_exp", "MHA_compat_exp_sum", "MHA_QK_MC", "MHA_WK_MC"
                - "MHA_attn_score", "MHA_output", "RES_var", "AVG_POOL_var".
                Defaults to None, which activates a predefined set of bounds and cuts optimized for typical use cases.

        Attributes:
            M (pyomo.ConcreteModel): 
                The optimization model instance to which the Transformer components will be added.
            N (int): 
                The sequence length of the encoder.
            d_model (int): 
                The embedding dimensions of the Transformer model.
            d_k (int): 
                The size of each attention head.
            d_H (int): 
                The number of attention heads in the model.
            input_dim (int): 
                The input dimension for the model.
            epsilon (float): 
                A small value added for numerical stability in layer normalization.
            bound_cut_activation (dict): 
                A dictionary indicating which bounds and cuts are active.
            model_dims (pyomo.Set): 
                A Pyomo set initialized with the model dimensions (0 to d_model-1).

        Notes:
            - Prints the percentage of activated bounds and cuts during initialization.
            - The `set_bound_cut` argument allows flexibility in configuring specific bounds and cuts for model optimization.
        """
        # Define optimisation model
        self.M = opt_model

        # Define hyper parameters either from json file or list
        if isinstance(hyper_params, str):
            with open(hyper_params, "r") as file:
                config = json.load(file)
            self.N = config['hyper_params']['N']                # enc sequence length
            self.d_model = config['hyper_params']['d_model']    # embedding dimensions of model
            self.d_k = config['hyper_params']['d_k']            # head size
            self.d_H = config['hyper_params']['d_H']            # number of heads
            self.input_dim = config['hyper_params']['input_dim']# dimension of input
            self.epsilon = config['hyper_params']['epsilon']    # epsilon used for layer norm
            file.close()
        else:
            self.N = hyper_params[0]            # enc sequence length
            self.d_model = hyper_params[1]      # embedding dimensions of model
            self.d_k = hyper_params[2]          # head size
            self.d_H = hyper_params[3]          # number of heads
            self.input_dim = hyper_params[4]    # dimension of input
            self.epsilon = hyper_params[5]      # epsilon used for layer norm

        # Define active bounds and cuts
        list_act= [ "embed_var",
        "LN_var", "LN_mean", "LN_num", "LN_num_squ", "LN_denom", "LN_num_squ_sum",
        "MHA_softmax_env", "MHA_Q", "MHA_K", "MHA_V", "MHA_attn_weight_sum", "MHA_attn_weight",
        "MHA_compat", "MHA_compat_exp", "MHA_compat_exp_sum", "MHA_QK_MC", "MHA_WK_MC", "MHA_attn_score","MHA_output", 
        "RES_var", "AVG_POOL_var"] #Names of all potential bounds and cuts
        
        self.bound_cut_activation = {}
        if set_bound_cut is None: # If no bounds and cuts defined set the following defaults
            # set all false
            for item in list_act: 
                self.bound_cut_activation[item] = False
              
            # set defaults to true  
            default_act = ["embed_var", "LN_var", "MHA_attn_weight_sum", "MHA_attn_weight", "MHA_Q", "MHA_K", "MHA_V", 
                        "MHA_compat", "MHA_compat_exp", "MHA_compat_exp_sum", "MHA_attn_score", "MHA_output" , "RES_var", 
                        "MHA_QK_MC", "MHA_WK_MC"]  # All configuration from report
            for item in default_act: 
                self.bound_cut_activation[item] = True
                
            #self.bound_cut_activation["MHA_softmax_env"] = False ## TO DO: add this as dynamic cut in callback
        else:
            self.bound_cut_activation = set_bound_cut 
        print('Percentage activated bounds + cuts: ', 100 * sum(self.bound_cut_activation.values())/ len(list_act))

        # initialise set of model dims
        if not hasattr( self.M, "model_dims"):
            self.M.model_dims = pyo.Set(initialize= list(range(self.d_model)))

    def build_from_hug_torch(self, tnn_model, sample_enc_input, sample_dec_input, enc_bounds = None , dec_bounds = None, Transformer='pytorch', default=True, hugging_face_dict=None):
        """ 
        Constructs a mathematical formulation of a trained PyTorch or HuggingFace Transformer model with both encoder and decoder components.

        Args:
            tnn_model (torch.nn.Module, TimeSeriesTransformerForPrediction): 
                The trained Transformer model to be formulated.
            sample_enc_input (torch.Tensor): 
                Sample input tensor for the encoder to extract model dimensions and structure.
            sample_dec_input (torch.Tensor): 
                Sample input tensor for the decoder to extract model dimensions and structure.
            enc_bounds (tuple, optional): 
                A tuple specifying the upper and lower bounds on encoder input values. Defaults to None.
            dec_bounds (tuple, optional): 
                A tuple specifying the upper and lower bounds on decoder input values. Defaults to None.
            Transformer (str, optional): 
                Specifies the type of transformer model ('pytorch' or 'huggingface'). Defaults to 'pytorch'.
            default (bool, optional): 
                Indicates whether to build the default architecture or parse the model's layer structure. Defaults to True.
                **Note**: Parsing issues may occur for layers not implemented as modules in the model architecture. For example, 
                residual layers are often excluded from the module list and are inferred based on their typical placement 
                (e.g., alongside normalization layers in encoder and decoder blocks).
            hugging_face_dict (dict, optional): 
                Additional parameters required for HuggingFace layer components if `Transformer='huggingface'`. Example structure:
                ```
                hugging_face_dict = {
                    "past_values": past_values,
                    "past_time_features": past_time_features,
                    "past_observed_mask": past_observed_mask,
                    "future_time_features": future_time_features
                }
                ```

        Returns:
            list: 
                A list containing:
                - The input variable name (str).
                - The output variable name (str).
                - A dictionary of feed-forward network parameters (dict).
        """
        
        # Parse to extract model archicture information and learnt parameters
        if Transformer == 'pytorch':
            layer_names, parameters, _, enc_dec_count, _ = get_pytorch_learned_parameters(tnn_model, sample_enc_input, sample_dec_input ,self.d_H, self.N)
        elif Transformer == 'huggingface':
            layer_names, parameters, _, enc_dec_count, _ = get_hugging_learned_parameters(tnn_model, sample_enc_input, sample_dec_input ,self.d_H, hugging_face_dict)
    
        # Build transformer
        if default: # build default pytorch architecture
            input_var_name, output_var_name, ffn_parameter_dict = self.__build_layers_default( layer_names, parameters, enc_dec_count , enc_bounds, dec_bounds)
        else: # build architecture from layer parsing
            input_var_name, output_var_name, ffn_parameter_dict = self.__build_layers_parse( layer_names, parameters, enc_dec_count , enc_bounds, dec_bounds)
        
        return [input_var_name, output_var_name, ffn_parameter_dict]

    def __add_pos_encoding(self, parameters, layer, input_name:Union[pyo.Var,str]):
        """
        Adds positional encoding to the input of a specific layer.

        Args:
            parameters (dict): Dictionary containing learned parameters of the Transformer model.
            layer (str): Layer identifier for the positional encoding.
            input_name (Union[pyo.Var,str]): Input variable or variable name to the current layer

        Returns:
            pyo.Var: The output variable from the positional encoding layer.
        """
        b_pe = parameters[layer,'b']
            
        output_name = layer          
        output_var = self.add_pos_encoding(input_name, output_name, b_pe)
            
        return output_name
    
    def __add_linear(self, parameters, layer, input_name, embed_dim):
        """
        Adds a linear transformation layer with optional bias.

        Args:
            parameters (dict): Dictionary containing learned parameters of the Transformer model.
            layer (str): Layer identifier for the positional encoding.
            input_name (Union[pyo.Var,str]): Input variable or variable name to the current layer
            embed_dim (int): Size of embedding dimension of the linear layer.
        Returns:
            pyo.Var: The output variable from the positional encoding layer.
        """
        # Get learnt parameter
        W_linear = parameters[layer,'W']
        try:
            b_linear = parameters[layer,'b']
        except:
            b_linear = None

        # Add linear layer                
        if not b_linear is None:    
            output_var = self.embed_input( input_name, layer, embed_dim, W_linear, b_linear) 
        else:
            output_var = self.embed_input( input_name, layer, embed_dim, W_linear)
            
        # return name of input to next layer
        return output_var
    
    def __add_ffn(self, parameters,ffn_parameter_dict, layer, input_name):
        """
        Adds a feed-forward neural network (FFN) layer with ReLU activation.

        Args:
            parameters (dict): Dictionary containing learned parameters of the Transformer model.
            ffn_parameter_dict (dict): Dictionary to store FFN layer parameters.
            layer (str): Layer identifier for the FFN.
            input_name (Union[pyo.Var,str]): Input variable or variable name to the current layer.

        Returns:
            tuple: A tuple containing the name of the output variable from the FFN layer and the updated FFN parameter dictionary.
        """
        # Create GurobiML Network Definition of FFN layers
        input_shape = np.array(parameters[layer]['input_shape'])
        ffn_params = self.get_fnn( input_name, layer, layer, input_shape, parameters)

        # Store network definition, NN input var and NN output var
        ffn_parameter_dict[layer] = ffn_params
        
        return layer, ffn_parameter_dict # return name of input to next layer, FFN parameters
    
    def __add_layer_norm(self, parameters, layer, input_name):
        """
        Adds a layer normalization layer to the model.

        Args:
            parameters (dict): Dictionary containing learned parameters of the Transformer model.
            layer (str): Layer identifier for layer normalization.
            input_name (Union[pyo.Var,str]): Input variable or variable name to the current layer
            
        Returns:
            pyo.Var: The output variable from the layer normalization layer.
        """
        # Get learnt parameter
        gamma = parameters[layer, 'gamma']
        beta  = parameters[layer, 'beta']
        
        # add layer normalization layer
        output_var = self.add_layer_norm( input_name, layer, gamma, beta)
        
        # return name of input to next layer
        return output_var
    
    def __add_cross_attn(self, parameters, layer, input_name, enc_output_name):
        """
        Adds a cross-attention layer that integrates information from the encoder into the decoder.

        Args:
            parameters (dict): Dictionary containing learned parameters of the Transformer model.
            layer (str): Layer identifier for layer normalization.
            input_name (Union[pyo.Var,str]): Input variable or variable name to the current layer
            enc_output_name (Union[pyo.Var,str]): Encoder's output variable.
            
        Returns:
            pyo.Var: The output variable from the cross-attention layer.
        """
        # Get learnt parameter
        W_q = parameters[layer,'W_q']
        W_k = parameters[layer,'W_k']
        W_v = parameters[layer,'W_v']
        W_o = parameters[layer,'W_o']
        
        try:
            b_q = parameters[layer,'b_q']
            b_k = parameters[layer,'b_k']
            b_v = parameters[layer,'b_v']
            b_o = parameters[layer,'b_o']
            
            # add attention layer with biases
            output_var = self.add_attention( input_name, layer, W_q, W_k, W_v, W_o, b_q, b_k, b_v, b_o, cross_attn=True, encoder_output=enc_output_name)
        except: # no bias values found
            # add attention layer without biases
            output_var = self.add_attention( input_name, layer, W_q, W_k, W_k, W_o, cross_attn=True, encoder_output=enc_output_name)
        
        return output_var
    
    def __add_self_attn(self, parameters, layer, input_name, mask=False):
        """
        Adds a self-attention layer for processing intra-sequence dependencies.

        Args:
            parameters (dict): Dictionary containing learned parameters of the Transformer model.
            layer (str): Layer identifier for layer normalization.
            input_name (Union[pyo.Var,str]): Input variable or variable name to the current layer
            mask (boolean): Indicator of whether to use casual mask (True) or not (False). Default: False
            
        Returns:
            pyo.Var: The output variable from the self-attention layer.
        """
        # Get learnt parameter
        W_q = parameters[layer,'W_q']
        W_k = parameters[layer,'W_k']
        W_v = parameters[layer,'W_v']
        W_o = parameters[layer,'W_o']

        try:
            b_q = parameters[layer,'b_q']
            b_k = parameters[layer,'b_k']
            b_v = parameters[layer,'b_v']
            b_o = parameters[layer,'b_o']
        except: # no bias values found
                b_q = None
        
        if not b_q is None:    
            # add attention layer with biases 
            output_var = self.add_attention( input_name, layer, W_q, W_k, W_v, W_o, b_q, b_k, b_v, b_o, mask)
        else:
            # add attention layer without biases
            output_var = self.add_attention( input_name, layer, W_q, W_k, W_k, W_o, mask)
            
        return output_var
    
    def __add_encoder_layer(self, parameters, layer, input_name, enc_layer, ffn_parameter_dict):
        """
        Adds default Pytorch encoder layer, including self-attention, layer normalization, and FFN components.

        Args:
            parameters (dict): Dictionary containing learned parameters of the Transformer model.
            layer (Union[pyo.Var, str]): Layer identifier for the encoder layer.
            input_name (str): Input variable or variable name to the current layer
            enc_layer (int): Index of the current encoder layer.
            ffn_parameter_dict (dict): Dictionary to store FFN layer parameters.

        Returns:
            tuple: A tuple containing the output variable and the updated FFN parameter dictionary.
        """
        
        input_name_1 = self.__add_self_attn(parameters, f"enc__self_attention_1", input_name)
        self.add_residual_connection(input_name, input_name_1, f"{layer}__{enc_layer}_residual_1")
        input_name_2 = self.__add_layer_norm(parameters, "enc__layer_normalization_1", f"{layer}__{enc_layer}_residual_1", enc_layer)
        
        input_name, ffn_parameter_dict = self.__add_ffn(parameters, ffn_parameter_dict, "enc__ffn_1", input_name_2) # add ReLU ANN
        
        self.add_residual_connection(input_name, input_name_2, f"{layer}__{enc_layer}_residual_2")
        input_name = self.__add_layer_norm(parameters, "enc__layer_normalization_2", f"{layer}__{enc_layer}_residual_2", enc_layer)
        
        return input_name, ffn_parameter_dict
    
    def __add_decoder_layer(self, parameters, layer, input_name, dec_layer, ffn_parameter_dict, enc_output_name):
        """
        Adds default Pytorch decoder layer, including self-attention, layer normalization, and FFN components.

        Args:
            parameters (dict): Dictionary containing learned parameters of the Transformer model.
            layer (Union[pyo.Var, str]): Layer identifier for the encoder layer.
            input_name (str): Input variable or variable name to the current layer
            dec_layer (int): Index of the current decoder layer.
            ffn_parameter_dict (dict): Dictionary to store FFN layer parameters.
            enc_output_name (Union[pyo.Var,str]): Encoder's output variable.

        Returns:
            tuple: A tuple containing the output variable and the updated FFN parameter dictionary.
        """
        
        embed_dim = self.M.model_dims 
        
        input_name_1 = self.__add_self_attn(parameters, f"dec__self_attention_1", input_name, mask=True)
        self.add_residual_connection(input_name, input_name_1, f"{layer}__{dec_layer}_residual_1")
        input_name_2 = self.__add_layer_norm(parameters, "dec__layer_normalization_1", f"{layer}__{dec_layer}_residual_1", dec_layer)
        
        input_name = self.__add_cross_attn(parameters, "dec__multi_head_attention_1", input_name_2, enc_output_name)
        self.add_residual_connection(input_name, input_name_2, f"{layer}__{dec_layer}_residual_2")
        input_name_3 = self.__add_layer_norm(parameters, "dec__layer_normalization_2", f"{layer}__{dec_layer}_residual_2", dec_layer)
        
        input_name, ffn_parameter_dict = self.__add_ffn(parameters, ffn_parameter_dict, "dec__ffn_1", input_name_3) # add ReLU ANN
        
        self.add_residual_connection(input_name, input_name_3, f"{layer}__{dec_layer}_residual_3")
        input_name = self.__add_layer_norm(parameters, "dec__layer_normalization_3", f"{layer}__{dec_layer}_residual_3", dec_layer)
        
        # return name of input to next layer
        return input_name, ffn_parameter_dict
    
    def __build_layers_default(self, layer_names, parameters, enc_dec_count, enc_bounds, dec_bounds):
        """
        Creates model with default transformer architecture for pytorch
        See https://pytorch.org/docs/stable/_modules/torch/nn/modules/transformer.html#Transformer
        
        **NB**: 
        - extracting transformer layers from "summary" function does not include residual layers
        
        Default architecture:
        - transfromer:
            - encoder: # for norm_first=False (default)
                x  = src
                x  = self_attn(x)
                x  = dropout(x)
                x1 = norm_1( src + x)
                
                x  = linear_1(x1)
                x  = activation(x)
                x  = dropout_1(x)
                x  = linear_2(x)
                x  = dropout_2(x)
                x  = norm_2( x1 + x)
                
            - decoder: # for norm_first=False (default)
                x  = src
                x  = self_attn(x)
                x  = dropout(x)
                x1 = norm_1( src + x)
                
                x  = multihead_attn(x1)
                x  = dropout_2(x)
                x2 = norm_2( x1 + x)
                
                x  = linear_1(x)
                x  = activation(x)
                x  = dropout(x)
                x  = linear_2(x)
                x  = dropout_3(x)
                x  = norm_3( x2 + x)
        
        

        Args:
            layer_names (list): names of layers in transformer model
            parameters (dict): transformer model's learned parameters
            enc_bounds ((float,float), optional): upper, lower bound on encoder input (choose a logical value based on training data)
            dec_bounds ((float,float), optional): upper, lower bound on decoder input (choose a logical value based on training data)
            
        Returns:
            tuple: Encoder input name, decoder input name, transformer output name, ffn parameters dictionary
        """
        ffn_parameter_dict = {}
        
        # define sets for input params
        enc_dim_1 = self.N 
        dec_dim_1 = self.N 
        self.M.enc_time_dims  = pyo.Set(initialize= list(range(enc_dim_1)))
        self.M.dec_time_dims  = pyo.Set(initialize= list(range(dec_dim_1)))
        self.M.dec_time_dims_param =  pyo.Set(initialize= list(range(dec_dim_1))) # - 2
        self.M.input_dims = pyo.Set(initialize= list(range(self.input_dim)))
        enc_layer = 0
        dec_layer = 0 
        
        for l, layer in enumerate(layer_names):
            print("layer iteration", layer)
            
            
            if l == 0: #input layer
                self.M.enc_input= pyo.Var(self.M.enc_time_dims,  self.M.input_dims, bounds=enc_bounds)
                enc_input_name = "enc_input"
                
                self.M.dec_input = pyo.Var(self.M.enc_time_dims,  self.M.input_dims, bounds=dec_bounds)
                dec_input_name = "dec_input"
                
            if "enc" in layer:
                residual = None
                if "norm" in layer and "enc" in layer_names[l - 2]:
                    residual = layer_names[l - 2]
                    
                enc_input_name, ffn_parameter_dict  = self.__add_ED_layer(parameters, layer, enc_input_name, enc_layer, ffn_parameter_dict, enc_output_name=None, residual=residual)              
                if "self_attention" in layer:
                    enc_layer +=1    
    
            elif "dec" in layer:
                residual = None
                if "norm" in layer and "dec" in layer_names[l - 2]:
                    residual = layer_names[l - 2]
                dec_input_name, ffn_parameter_dict  = self.__add_ED_layer(parameters, layer, dec_input_name, dec_layer, ffn_parameter_dict, enc_output_name=enc_input_name, residual=residual)
                
                if "self_attention" in layer:
                    dec_layer +=1
                     
            elif "layer_norm" in layer:
                if dec_layer > 1: #if final layer, only apply on decoder
                    dec_input_name = self.__add_layer_norm(parameters, layer, dec_input_name)
                else: 
                    enc_input_name = self.__add_layer_norm(parameters, layer, enc_input_name)
                    dec_input_name = self.__add_layer_norm(parameters, layer, dec_input_name)
                
            elif "linear" in layer:
                if dec_layer > 1: 
                    embed_dim = self.M.input_dims
                    dec_input_name = self.__add_linear( parameters, layer, dec_input_name, embed_dim)
                else:
                    embed_dim = self.M.model_dims # embed from current dim to self.M.model_dims
                    enc_input_name = self.__add_linear( parameters, layer, enc_input_name, embed_dim)
                    dec_input_name = self.__add_linear( parameters, layer, dec_input_name, embed_dim)
            
            elif "ffn" in layer:
                if dec_layer > 1: 
                    dec_input_name,ffn_parameter_dict = self.__add_ffn(parameters,ffn_parameter_dict, layer, dec_input_name)
                else:
                    enc_input_name,ffn_parameter_dict = self.__add_ffn(parameters,ffn_parameter_dict, layer, enc_input_name)
                    dec_input_name,ffn_parameter_dict = self.__add_ffn(parameters,ffn_parameter_dict, layer, dec_input_name)
        
        #return: encoder input name, decoder input name, transformer output name, ffn parameters dictionary
        return [["enc_input","dec_input"], dec_input_name , ffn_parameter_dict]
    
    def __build_layers_parse(self, layer_names, parameters, enc_dec_count, enc_bounds, dec_bounds):
        """
        Create encoder-decoder TNN model based on parsed layers list
        
        Args:
            layer_names (list): names of layers in transformer model
            parameters (dict): transformer model's learned parameters
            enc_bounds ((float,float), optional): upper, lower bound on encoder input (choose a logical value based on training data)
            dec_bounds ((float,float), optional): upper, lower bound on decoder input (choose a logical value based on training data)
            
        Returns:
            tuple: Encoder input name, decoder input name, transformer output name, ffn parameters dictionary
        """
        ffn_parameter_dict = {}
        
        # define sets for input params
        enc_dim_1 = self.N 
        dec_dim_1 = self.N 
        self.M.enc_time_dims  = pyo.Set(initialize= list(range(enc_dim_1)))
        self.M.dec_time_dims  = pyo.Set(initialize= list(range(dec_dim_1)))
        self.M.dec_time_dims_param =  pyo.Set(initialize= list(range(dec_dim_1))) # - 2
        self.M.input_dims = pyo.Set(initialize= list(range(self.input_dim)))
        enc_layer = 0
        dec_layer = 0 
        enc_post_attn_flag = False
        
        # for each layer in TNN
        for l, layer in enumerate(layer_names):
            #print("layer iteration", layer)
            
            if l == 0: #input layer
                self.M.enc_input= pyo.Var(self.M.enc_time_dims,  self.M.input_dims, bounds=enc_bounds)
                enc_input_name = "enc_input"
                
                self.M.dec_input = pyo.Var(self.M.enc_time_dims,  self.M.input_dims, bounds=dec_bounds)
                dec_input_name = "dec_input"
                
            if "enc" in layer:
                residual = None
                if "norm" in layer and not layer.endswith("_1"):
                    if enc_post_attn_flag:
                        residual = enc_norm_1 #in each layer the norm after self attention has residual to first norm.
                        enc_post_attn_flag = False
                    residual = layer_names[l - 2]
                else:
                    enc_norm_1 = layer
                    enc_post_attn_flag = True
                    
                enc_input_name, ffn_parameter_dict  = self.__add_ED_layer(parameters, layer, enc_input_name, enc_layer, ffn_parameter_dict, enc_output_name=None, residual=residual)              
                if "self_attention" in layer:
                    enc_layer +=1    
    
            elif "dec" in layer:
                residual = None
                if "norm" in layer and not layer.endswith("_1"):
                    residual = layer_names[l - 2]
                dec_input_name, ffn_parameter_dict  = self.__add_ED_layer(parameters, layer, dec_input_name, dec_layer, ffn_parameter_dict, enc_output_name=enc_input_name, residual=residual, mask=True)
                
                if "self_attention" in layer:
                    dec_layer +=1
                    
                     
            elif "layer_norm" in layer:
                if dec_layer > 1: #if final layer, only apply on decoder
                    dec_input_name = self.__add_layer_norm(parameters, layer, dec_input_name)
                else: 
                    enc_input_name = self.__add_layer_norm(parameters, layer, enc_input_name)
                    dec_input_name = self.__add_layer_norm(parameters, layer, dec_input_name)
                
            elif "linear" in layer:
                if dec_layer > 1: 
                    embed_dim = self.M.input_dims
                    dec_input_name = self.__add_linear( parameters, layer, dec_input_name, embed_dim)
                else:
                    embed_dim = self.M.model_dims # embed from current dim to self.M.model_dims
                    enc_input_name = self.__add_linear( parameters, layer, enc_input_name, embed_dim)
                    dec_input_name = self.__add_linear( parameters, layer, dec_input_name, embed_dim)
            
            elif "ffn" in layer:
                if dec_layer > 1: 
                    dec_input_name,ffn_parameter_dict = self.__add_ffn(parameters,ffn_parameter_dict, layer, dec_input_name)
                else:
                    enc_input_name,ffn_parameter_dict = self.__add_ffn(parameters,ffn_parameter_dict, layer, enc_input_name)
                    dec_input_name,ffn_parameter_dict = self.__add_ffn(parameters,ffn_parameter_dict, layer, dec_input_name)
        
        #return: encoder input name, decoder input name, transformer output name, ffn parameters dictionary
        return [["enc_input","dec_input"], dec_input_name , ffn_parameter_dict]
    
    def __add_ED_layer(self, parameters, layer, input_name, layer_num, ffn_parameter_dict, enc_output_name=None, residual=None, mask=False):
        """
        Add components of encoder or decoder. Depending on the type of layer specified, this method constructs the appropriate subcomponent 
        (e.g., self-attention, cross-attention, layer normalization, linear transformations, feed-forward neural networks, or positional encoding).

        Args:
            parameters (dict): Dictionary containing learned parameters of the Transformer model.
            layer (str): Layer identifier specifying the type of layer to add.
            input_name (Union[pyo.Var,str]): Input variable or variable name to the current layer
            layer_num (int): Index of the current layer in the encoder or decoder stack.
            ffn_parameter_dict (dict): Dictionary to store parameters for feed-forward neural network layers.
            enc_output_name (Union[pyo.Var,str], optional): Name of the encoder's output variable, used in cross-attention layers. Defaults to None.
            residual (str, optional): Name of the residual connection variable. If provided, a residual connection will be added. Defaults to None.
            mask (boolean, optional): Indicates whether a mask should be applied for self-attention layers. Defaults to False.

        Returns:
            tuple:
                - str: The name of the output variable from the last component added in the layer.
                - dict: Updated dictionary of feed-forward neural network parameters.

        Notes:
            - The method checks the type of layer specified by the `layer` argument and dynamically builds the corresponding component.
            - Residual connections are added for specific layers if a `residual` is provided.
        """
        
        if "self_attention" in layer:
            output_name = self.__add_self_attn(parameters, layer, input_name, mask)
            
        if "multi_head_attention" in layer:
            output_name = self.__add_cross_attn(parameters, layer, input_name, enc_output_name)
                
        if "layer_norm" in layer:
            # add residual connection if supplied
            if not residual is None:
                self.add_residual_connection(input_name, residual, f"{layer}__{layer_num}_residual")
                input_name = f"{layer}__{layer_num}_residual"
                
            # add layer normalisation
            output_name = self.__add_layer_norm(parameters, layer, input_name, layer_num)
            
        if "linear" in layer:
            embed_dim = self.M.model_dims
            output_name = self.__add_linear(parameters, layer, input_name, embed_dim, layer_num)
        
        if "ffn" in layer:
            output_name, ffn_parameter_dict = self.__add_ffn(parameters, ffn_parameter_dict, layer, input_name)
            
        if "pos_encoding" in layer:
            output_name = self.__add_pos_encoding(parameters, layer, input_name, layer_index="")
            
        return output_name, ffn_parameter_dict 
    
    
    def add_input_var(self, input_var_name, dims, bounds):
        """
        Adds a TNN input variable to the Pyomo model with specified dimensions and bounds.

        Args:
            input_var_name (str): Name of the input variable to be added to the Pyomo model.
            dims (tuple): Dimensions of the input variable in the form `(dim_0, dim_1)`. 
            bounds (tuple): Bounds for the input variable in the form `(lower_bound, upper_bound)`.

        Returns:
            pyo.Var: The created or existing Pyomo variable representing the input to TNN.
        """
        if not hasattr(self.M, input_var_name+"dim_0"):
            setattr(self.M, input_var_name+"dim_0", pyo.Set(initialize= list(range(dims[0]))))
            dim_0 = getattr(self.M, input_var_name+"dim_0")  
            
            setattr(self.M, input_var_name+"dim_1", pyo.Set(initialize= list(range(dims[1]))))
            dim_1 = getattr(self.M, input_var_name+"dim_1") 
        
            setattr(self.M, input_var_name, pyo.Var(dim_0, dim_1, bounds=bounds))
            input_var = getattr(self.M, input_var_name)
        else:
            raise ValueError('Attempting to overwrite variable')
        
        return input_var
    
    def add_pos_encoding(self, input_var_name:Union[pyo.Var,str], embed_var_name, b_emb):
        """
        Add positional encoding layer
        
        Args:
            input_var_name (Union[pyo.Var,str]): Input variable or variable name to the current layer
            embed_var_name (str): Identifier of variables added by this function. 
            b_emb (list): List containing positional encoding parameters extracted from trained TNN.

        Returns:
            pyo.Var: Output of positional encoding layer.
        
        """
        
        # Create constraint list for layer
        if not hasattr(self.M, "pe_constraints"):
            self.M.pe_constraints = pyo.ConstraintList()
        
        # Get input var
        if not isinstance(input_var_name, pyo.Var):
            input_var = getattr(self.M, input_var_name)
        else:
            input_var = input_var_name
        
        if input_var.is_indexed(): 
            # Create layer output variable
            if not hasattr(self.M, embed_var_name):
                setattr(self.M, embed_var_name, pyo.Var(input_var.index_set(), within=pyo.Reals, initialize= 0))
                embed_var = getattr(self.M, embed_var_name)   
            else:
                raise ValueError('Attempting to overwrite variable')
            
            # Get dimensions of input
            dims = []
            for set in str(input_var.index_set()).split("*"):
                dims.append( getattr( self.M, set) )

            # Create dict with positional encoding param indexed by input dim
            count_i = 0
            count_j = 0
            b_emb_dict = {}
            for i in dims[0]:
                for j in dims[1]:
                    b_emb_dict[(i,j)] = b_emb[count_i][count_j]
                    count_j += 1
                count_i += 1
                count_j = 0
            
            # Create Pyomo parameter to store positional encoding values
            setattr(self.M, embed_var_name+"_b_pe", pyo.Param(input_var.index_set() , initialize=b_emb_dict))
            b_emb= getattr(self.M, embed_var_name+"_b_pe")  
        
            # Add positional encoding values to layer input
            for index in input_var.index_set() :
                self.M.pe_constraints.add(embed_var[index] == input_var[index] +  b_emb[index])
                if self.bound_cut_activation["embed_var"]:
                    if isinstance(input_var, pyo.Var):
                        if not input_var[index].ub is None and not input_var[index].lb is None:
                            embed_var[index].ub = input_var[index].ub  +  b_emb[index]
                            embed_var[index].lb = input_var[index].lb  +  b_emb[index]
                            
                    elif isinstance(input_var, pyo.Param):
                        embed_var[index].ub = input_var[index] +  b_emb[index]
                        embed_var[index].lb = input_var[index] +  b_emb[index]
        else:
            raise ValueError('Non-indexed input provided. Input must be an indexed variable representing the sequence')
        
        return embed_var # return positionallu encoded variable
    
    def embed_input(self, input_var_name:Union[pyo.Var,str], embed_var_name, embed_dim_2, W_emb=None, b_emb = None):
        """
        Embeds the feature dimensions of an input variable by applying linear transformations 
        
        Args:
            input_var_name (Union[pyo.Var, str]): The name or instance of the input variable to be embedded.
            embed_var_name (str): The name for the resulting embedded variable in the Pyomo model.
            embed_dim_2 (pyo.Set): Dimensions of the embedding output.
            W_emb (list, optional): Weight matrix for the embedding transformation. Defaults to None.
            b_emb (list, optional): Bias vector for the embedding transformation. Defaults to None.

        Returns:
            pyo.Var: The Pyomo variable representing the embedded input.
        """
        # Create constraint list for layer
        if not hasattr(self.M, "embed_constraints"):
            self.M.embed_constraints = pyo.ConstraintList()
           
        # Get input var
        if not isinstance(input_var_name, pyo.Var):
            input_var = getattr(self.M, input_var_name)
        else:
            input_var = input_var_name
        
        if input_var.is_indexed():
            # Get dimensions of input variable
            set_var = input_var.index_set()
            indices = []
            for set in str(set_var).split("*"):
                indices.append( getattr( self.M, set) )

            # Create layer output variable
            if not hasattr(self.M, embed_var_name):
                setattr(self.M, embed_var_name, pyo.Var(indices[0], embed_dim_2 , within=pyo.Reals, initialize= 0))
                embed_var = getattr(self.M, embed_var_name)   
            else:
                raise ValueError('Attempting to overwrite variable')
            
            if W_emb is None:
                if b_emb is None:
                    raise ValueError('Layer has no weight or bias parameters. Weight is None and bias is None')  
                                
                elif len(embed_dim_2) == len(indices[1]):
                    # Create bias variable
                    b_emb_dict = {
                        (embed_dim_2.at(d+1)): b_emb[d]
                        for d in range(len(embed_dim_2))
                    }
                    
                    setattr(self.M, embed_var_name+"_b_emb", pyo.Param(embed_dim_2 , initialize=b_emb_dict))
                    b_emb= getattr(self.M, embed_var_name+"_b_emb")  
                
                    # Add bias
                    for d,s in zip(embed_dim_2, indices[1]) :
                        for t in indices[0]:
                            self.M.embed_constraints.add(embed_var[t, d] 
                                                    == input_var[t,s] +  b_emb[d]
                                                    )
                            if self.bound_cut_activation["embed_var"]:
                                if isinstance(input_var, pyo.Var):
                                    if not input_var[t,indices[1].first()].ub is None and not input_var[t,indices[1].first()].lb is None:
                                        embed_var[t, d].ub = input_var[t,s].ub  +  b_emb[d]
                                        embed_var[t, d].lb = input_var[t,s].lb  +  b_emb[d]
                                        
                                elif isinstance(input_var, pyo.Param):
                                    embed_var[t, d].ub = input_var[t,s] +  b_emb[d]
                                    embed_var[t, d].lb = input_var[t,s] +  b_emb[d]
                else:
                    raise ValueError(f'Mismatching dimensions: {len(embed_dim_2)} and {len(indices[1])}')  
                
                          
            else: # W_emb has a value
                # Create parameter to store learnt weight matrix
                W_emb_dict = {
                    (indices[1].at(s+1),embed_dim_2.at(d+1)): W_emb[d][s]
                    for s in range(len(indices[1]))
                    for d in range(len(embed_dim_2))
                }
                setattr(self.M, embed_var_name+"_W_emb", pyo.Param(indices[1], embed_dim_2 , initialize=W_emb_dict))
                W_emb= getattr(self.M, embed_var_name+"_W_emb")   
                
                if not b_emb is None: # with bias
                    # Create bias parameter
                    b_emb_dict = {
                        (embed_dim_2.at(d+1)): b_emb[d]
                        for d in range(len(embed_dim_2))
                    }
                    setattr(self.M, embed_var_name+"_b_emb", pyo.Param(embed_dim_2 , initialize=b_emb_dict))
                    b_emb= getattr(self.M, embed_var_name+"_b_emb")  
                
                    # Perform matrix multiplication
                    for d in embed_dim_2 :
                        for t in indices[0]:
                            self.M.embed_constraints.add(embed_var[t, d] 
                                                    == sum(input_var[t,s] * W_emb[s,d] for s in indices[1]) +  b_emb[d]
                                                    )
                            if self.bound_cut_activation["embed_var"]:
                                if isinstance(input_var, pyo.Var):
                                    if not input_var[t,indices[1].first()].ub is None and not input_var[t,indices[1].first()].lb is None:
                                        embed_var[t, d].ub = sum(max(input_var[t,s].ub * W_emb[s,d], input_var[t,s].lb * W_emb[s,d]) for s in indices[1]) +  b_emb[d]
                                        embed_var[t, d].lb = sum(min(input_var[t,s].ub * W_emb[s,d], input_var[t,s].lb * W_emb[s,d]) for s in indices[1]) +  b_emb[d]
                                        
                                elif isinstance(input_var, pyo.Param):
                                    embed_var[t, d].ub = sum(input_var[t,s] * W_emb[s,d] for s in indices[1]) +  b_emb[d]
                                    embed_var[t, d].lb = sum(input_var[t,s] * W_emb[s,d] for s in indices[1]) +  b_emb[d]
                else: # no bias
                    # Perform matrix multiplication
                    for d in embed_dim_2 :
                        for t in indices[0]:
                            self.M.embed_constraints.add(embed_var[t, d] 
                                                    == sum(input_var[t,s] * W_emb[s,d] for s in indices[1])
                                                    )
                            if self.bound_cut_activation["embed_var"]:
                                if isinstance(input_var, pyo.Var):
                                    if not input_var[t,indices[1].first()].ub is None and not input_var[t,indices[1].first()].lb is None:
                                        embed_var[t, d].ub = sum(max(input_var[t,s].ub * W_emb[s,d], input_var[t,s].lb * W_emb[s,d]) for s in indices[1])
                                        embed_var[t, d].lb = sum(min(input_var[t,s].ub * W_emb[s,d], input_var[t,s].lb * W_emb[s,d]) for s in indices[1])
                                        
                                elif isinstance(input_var, pyo.Param):
                                    embed_var[t, d].ub = sum(input_var[t,s] * W_emb[s,d] for s in indices[1])
                                    embed_var[t, d].lb = sum(input_var[t,s] * W_emb[s,d] for s in indices[1])
        else:
            raise ValueError('Input value must be indexed')
        return embed_var
        
    def add_layer_norm(self, input_var_name:Union[pyo.Var,str], LN_var_name, gamma= None, beta = None, eps=None):  # non-linear
        """
        Adds layer normalisation layer.
        
        This method normalizes the input variable over the sequence dimensions by calculating the mean 
        and variance, then applying scaling (`gamma`) and shifting (`beta`) parameters.
        
        Args:
            input_var_name (Union[pyo.Var, str]): Name or instance of the input variable to be normalized.
            LN_var_name (str): Name for the output variable of the layer normalization operation.
            gamma (list, optional): Learnt scaling factors for each model dimension. Defaults to None (set to 1).
            beta (list, optional):  Learnt shifting factors for each model dimension. Defaults to None (set to 0).
            eps (float, optional): Small positive value to avoid division by zero during variance normalization. Defaults to the `self.epsilon` value.
            
        Returns:
            pyo.Var: The Pyomo variable representing the output of the layer normalization operation.

        """
        # Create list for layers constraints
        if not hasattr( self.M, "layer_norm_constraints"):
            self.M.layer_norm_constraints = pyo.ConstraintList()
            
        # Update epsilon value if needed
        if not eps is None:
            self.epsilon = eps
        
        # Get input
        if not isinstance(input_var_name, pyo.Var):
            input_var = getattr(self.M, input_var_name)
        else:
            input_var = input_var_name
        
        # Determine indices of input
        if input_var.is_indexed():
            set_var = input_var.index_set()
            indices = []
            for set in str(set_var).split("*"):
                indices.append( getattr( self.M, set) )
                
            time_dim = indices[0]
            model_dims = indices[1]
        else:
            raise ValueError('Input value must be indexed (time, model_dim)')
                
        
        # Initialize variables
        if not hasattr( self.M, LN_var_name):
            # define layer norm output var
            setattr( self.M, LN_var_name, pyo.Var(time_dim, model_dims, within=pyo.Reals))
            layer_norm_var = getattr( self.M, LN_var_name)
            
            # define gamma, beta params
            if not gamma is None and not beta is None:
            
                dict_gamma = {(v): val for v,val in zip( self.M.model_dims, gamma)}
                dict_beta  = {(v): val for v,val in zip( self.M.model_dims, beta)}
            else:
                dict_gamma = {(v): 1 for v in self.M.model_dims}
                dict_beta  = {(v): 0 for v in self.M.model_dims}
                
            # define new gamma and beta params
            setattr( self.M, f"gamma_{LN_var_name}", pyo.Param( self.M.model_dims, initialize = dict_gamma))
            setattr( self.M, f"beta_{LN_var_name}", pyo.Param( self.M.model_dims, initialize = dict_beta))
            gamma = getattr( self.M, f"gamma_{LN_var_name}")
            beta  = getattr( self.M, f"beta_{LN_var_name}")
  
            # define calculation variables
            sum_name = 'sum_'+ LN_var_name
            setattr( self.M, sum_name, pyo.Var(time_dim, within=pyo.Reals))
            sum_t = getattr( self.M, sum_name)
            
            variance_name = 'variance_'+ LN_var_name
            setattr( self.M, variance_name, pyo.Var(time_dim, within=pyo.Reals))
            variance = getattr( self.M, variance_name)
            
            div_name = 'div_'+ LN_var_name
            setattr( self.M, div_name, pyo.Var(time_dim, model_dims, within=pyo.Reals))
            div = getattr( self.M, div_name)
            
            denominator_abs_name = 'denominator_abs_'+ LN_var_name
            setattr( self.M, denominator_abs_name, pyo.Var(time_dim, within=pyo.NonNegativeReals, bounds=(0,None)))
            denominator_abs = getattr( self.M, denominator_abs_name)
            
            numerator_name = 'numerator_'+ LN_var_name
            setattr( self.M, numerator_name, pyo.Var(time_dim, model_dims, within=pyo.Reals))
            numerator = getattr( self.M, numerator_name)

            numerator_scaled_name = 'numerator_scaled_'+ LN_var_name
            setattr( self.M, numerator_scaled_name, pyo.Var(time_dim, model_dims, within=pyo.Reals))
            numerator_scaled = getattr( self.M, numerator_scaled_name)
            
            numerator_squared_name = 'numerator_squared_'+ LN_var_name
            setattr( self.M, numerator_squared_name, pyo.Var(time_dim, model_dims, within=pyo.Reals, bounds=(0,None)))
            numerator_squared = getattr( self.M, numerator_squared_name)
              
            numerator_squared_sum_name = 'numerator_squared_sum_'+ LN_var_name
            setattr( self.M, numerator_squared_sum_name, pyo.Var(time_dim, within=pyo.Reals, bounds=(0,None)))
            numerator_squared_sum = getattr( self.M, numerator_squared_sum_name)
              
        else:
            raise ValueError('Attempting to overwrite variable')

        for t in time_dim: 
            self.M.layer_norm_constraints.add(expr= sum_t[t] == sum(input_var[t,  d_prime] for d_prime in model_dims) )
            
            self.M.layer_norm_constraints.add(expr= numerator_squared_sum[t] == sum(numerator_squared[t,d_prime] for d_prime in model_dims))
            self.M.layer_norm_constraints.add(expr= variance[t] * (len(model_dims)) == numerator_squared_sum[t])

            self.M.layer_norm_constraints.add(expr= variance[t] + self.epsilon == (denominator_abs[t]*denominator_abs[t]) )
            
            # Constraints for each element in sequence
            for d in model_dims:  
                self.M.layer_norm_constraints.add(expr= numerator[t,d] == input_var[t, d] - ((1/ len(model_dims)) *sum_t[t]))
                self.M.layer_norm_constraints.add(expr= numerator_squared[t,d] == numerator[t,d]**2)
                self.M.layer_norm_constraints.add(expr= div[t,d] * denominator_abs[t] == numerator[t,d] )
                
                self.M.layer_norm_constraints.add(expr= numerator_scaled[t,d] == gamma[d] * div[t,d])
                self.M.layer_norm_constraints.add(expr=layer_norm_var[t, d] == numerator_scaled[t,d] + beta[d])

                if self.bound_cut_activation["LN_var"]:
                    layer_norm_var[t, d].ub = max( beta[d] + 5*gamma[d], beta[d] - 5*gamma[d])
                    layer_norm_var[t, d].lb = min( beta[d] + 5*gamma[d], beta[d] - 5*gamma[d])
                    
                #Add bounds
                if input_var[t, d].ub and input_var[t, d].lb:
                    if self.bound_cut_activation["LN_num"]:
                        mean_u = (sum(input_var[t, d_prime].ub for d_prime in model_dims)/ len(model_dims) )
                        mean_l = (sum(input_var[t, d_prime].lb for d_prime in model_dims)/ len(model_dims))
                        numerator[t,d].ub = input_var[t, d].ub - mean_l
                        numerator[t,d].lb = input_var[t, d].lb - mean_u

                    if self.bound_cut_activation["LN_num_squ"]:
                        numerator_squared[t,d].ub = max(numerator[t,d].ub**2, numerator[t,d].lb**2) 
   
                    if self.bound_cut_activation["LN_denom"]:
                        denominator_abs[t].ub = (max(input_var[t,:].ub) - min(input_var[t,:].lb))/2 #standard deviation
                        denominator_abs[t].lb = 0
                if self.bound_cut_activation["LN_num_squ"]:
                    numerator_squared[t,d].lb = 0
                    
            if self.bound_cut_activation["LN_num_squ"]:       
                try:
                    numerator_squared_sum[t].ub = sum( (numerator_squared[t,d_prime].ub) for d_prime in model_dims) 
                except: 
                    pass
                numerator_squared_sum[t].lb = 0
        return layer_norm_var
        
    def add_attention(self, input_var_name:Union[pyo.Var,str], output_var_name, W_q, W_k, W_v, W_o, b_q = None, b_k = None, b_v = None, b_o = None, mask=False, cross_attn=False, encoder_output:Union[pyo.Var,str]=None, exp_approx=False, norm_softmax=False):
        """
        Adds multi-head attention layer
        
        This method computes the attention weights and output for an MHA layer based on the provided query (Q),
        key (K), and value (V) weight matrices and optional biases. It supports both self-attention and 
        cross-attention mechanisms and can optionally approximate the softmax operation using a power series 
        expansion. It also includes the optimal max-normalised softmax calculation which does not alter the computed 
        softmax values but can be used to reduce the domain of the exp(compatibility var) to [0,1] and the 
        sum of exp(compatibility var) to [1, N] where N is the number of elements in the sequence.

        Args:
            input_var_name (Union[pyo.Var, str]): Name or instance of the input variable for attention computation.
            output_var_name (str): Name of the output variable for the attention layer.
            W_q (np.ndarray): Query weight matrix of shape `(model_dims, heads, head_dims)`.
            W_k (np.ndarray): Key weight matrix of shape `(model_dims, heads, head_dims)`.
            W_v (np.ndarray): Value weight matrix of shape `(model_dims, heads, head_dims)`.
            W_o (np.ndarray): Output weight matrix of shape `(model_dims, heads, head_dims)`.
            b_q (np.ndarray, optional): Query bias vector of shape `(heads, head_dims)`. Defaults to None.
            b_k (np.ndarray, optional): Key bias vector of shape `(heads, head_dims)`. Defaults to None.
            b_v (np.ndarray, optional): Value bias vector of shape `(heads, head_dims)`. Defaults to None.
            b_o (np.ndarray, optional): Output bias vector of shape `(model_dims,)`. Defaults to None.
            mask (bool, optional): Whether to apply a mask to the attention weights. Defaults to False.
            cross_attn (bool, optional): Whether this is a cross-attention layer (e.g., decoder attending to encoder output). Defaults to False.
            encoder_output (Union[pyo.Var, str], optional): Name or instance of the encoder output variable 
                for cross-attention. Required if `cross_attn=True`. Defaults to None.
            exp_approx (bool, optional): Whether to approximate the softmax operation using a power series expansion. Defaults to False.
            norm_softmax (bool, optional): Whether to normalize the compatibility scores using the softmax operation. Defaults to False.

        Returns:
            pyo.Var: The Pyomo variable representing the output of the MHA layer.

        """
        
        # get input
        if not isinstance(input_var_name, pyo.Var):
            input_var = getattr(self.M, input_var_name)
        else:
            input_var = input_var_name
        
        # determine indices of input
        if input_var.is_indexed():
            set_var = input_var.index_set()
            indices = []
            for set in str(set_var).split("*"):
                indices.append( getattr( self.M, set) )
                
            time_dim = indices[0]
            model_dims = indices[1]
            model_dims_enc = indices[1] # kv dim same as q
            time_dim_enc = indices[0]
            d_heads = self.d_k
            d_heads_kv = self.d_k
        else:
            raise ValueError('Input value must be indexed (time, model_dim)')
        
        # Check for cross attention between encoder and decoder
        if cross_attn and not encoder_output is None:
    
            # get var if string
            if not isinstance(encoder_output, pyo.Var):
                encoder_output_var = getattr(self.M, encoder_output)
            else:
                encoder_output_var = encoder_output
            
            if encoder_output_var.is_indexed():
                set_var = encoder_output_var.index_set()
                indices = []
                for set in str(set_var).split("*"):
                    indices.append( getattr( self.M, set) )
                model_dims_enc  = indices[1] # Weights k,v dim based on enc dim but Weight q dim based on decoder
                time_dim_enc = indices[0]    # K and V first dim
            else:
                raise ValueError(f'{encoder_output} must be indexed (time, model_dim)')
            
            d_heads = len(model_dims)/self.d_H        # decoder head size
            d_heads_kv = len(model_dims_enc)/self.d_H # encoder head size
            
            assert( d_heads == d_heads_kv and d_heads==self.d_k) #check head size is as expected and head size of enc == head size dec
            
            
        # define variables and parameters of this layer
        if not hasattr( self.M, output_var_name):
            # Create output variable
            setattr( self.M, output_var_name, pyo.Var(time_dim, model_dims , within=pyo.Reals))
            attention_output = getattr( self.M, output_var_name)
            
            # Create multi-head attention layer as a pyo.Block
            setattr( self.M, "Block_"+output_var_name, pyo.Block())
            MHA_Block  = getattr( self.M, "Block_"+output_var_name)
            
            # Create constraint list for attention block
            MHA_Block.attention_constraints = pyo.ConstraintList()

            # If needed create constraint lists for softmax outer approximators
            if self.bound_cut_activation["MHA_softmax_env"]: 
                MHA_Block.constr_convex = pyo.ConstraintList()
                MHA_Block.constr_concave = pyo.ConstraintList()
                MHA_Block.constr_convex_tp = pyo.ConstraintList()
                MHA_Block.constr_convex_tp_sct = pyo.ConstraintList()
                MHA_Block.constr_concave_tp = pyo.ConstraintList()
                MHA_Block.constr_concave_tp_sct = pyo.ConstraintList()
        else:
            raise ValueError('Attempting to overwrite variable')

        # Define sets
        MHA_Block.heads = pyo.RangeSet(1, self.d_H)          # number of heads
        MHA_Block.head_dims = pyo.RangeSet(1, d_heads)       # head size Q
        
        # Define parameters
        W_q_dict = {
            (D, H, K): W_q[d][h][k]
            for d,D in enumerate(model_dims )
            for h,H in enumerate(MHA_Block.heads)
            for k,K in enumerate(MHA_Block.head_dims)
        }
        W_k_dict = {
            (D, H, K): W_k[d][h][k]
            for d,D in enumerate(model_dims_enc)
            for h,H in enumerate(MHA_Block.heads)
            for k,K in enumerate(MHA_Block.head_dims)
        }
        W_v_dict = {
            (D, H, K): W_v[d][h][k]
            for d,D in enumerate(model_dims_enc )
            for h,H in enumerate(MHA_Block.heads)
            for k,K in enumerate(MHA_Block.head_dims)
        }
        
        if not W_o is None:
            W_o_dict = {
                (D, H, K): W_o[h][k][d]
                for d,D in enumerate(model_dims )
                for h,H in enumerate(MHA_Block.heads)
                for k,K in enumerate(MHA_Block.head_dims)
            }
        else:
            W_o_dict = {
                (D, H, K): 1
                for d,D in enumerate(model_dims )
                for h,H in enumerate(MHA_Block.heads)
                for k,K in enumerate(MHA_Block.head_dims)
            }
            
 
        MHA_Block.W_q = pyo.Param(model_dims ,MHA_Block.heads,MHA_Block.head_dims, initialize=W_q_dict, mutable=False)
        MHA_Block.W_k = pyo.Param(model_dims_enc ,MHA_Block.heads,MHA_Block.head_dims, initialize=W_k_dict, mutable=False)
        MHA_Block.W_v = pyo.Param(model_dims_enc ,MHA_Block.heads,MHA_Block.head_dims, initialize=W_v_dict, mutable=False)
        MHA_Block.W_o = pyo.Param(model_dims ,MHA_Block.heads,MHA_Block.head_dims, initialize=W_o_dict, mutable=False)
        
        if not b_q is None:
            b_q_dict = {
                        (h, k): b_q[h-1][k-1]
                        for h in MHA_Block.heads
                        for k in MHA_Block.head_dims
                       }
        else:
            b_q_dict = {
                        (h, k): 0
                        for h in MHA_Block.heads
                        for k in MHA_Block.head_dims
                       }
        MHA_Block.b_q = pyo.Param(MHA_Block.heads, MHA_Block.head_dims, initialize=b_q_dict, mutable=False)
            
        if not b_k is None:
            b_k_dict = {
                        (h, k): b_k[h-1][k-1]
                        for h in MHA_Block.heads
                        for k in MHA_Block.head_dims
                       }
        else:
            b_k_dict = {
                        (h, k): 0
                        for h in MHA_Block.heads
                        for k in MHA_Block.head_dims
                       }
        MHA_Block.b_k = pyo.Param(MHA_Block.heads, MHA_Block.head_dims, initialize=b_k_dict, mutable=False)
            
        if not b_v is None: 
            b_v_dict = {
                        (h, k): b_v[h-1][k-1]
                        for h in MHA_Block.heads
                        for k in MHA_Block.head_dims
                       }
        else:
            b_v_dict = {
                        (h, k): 0
                        for h in MHA_Block.heads
                        for k in MHA_Block.head_dims
                       }
        MHA_Block.b_v = pyo.Param(MHA_Block.heads, MHA_Block.head_dims, initialize=b_v_dict, mutable=False)
            
        if not b_o is None:
            b_o_dict = {(d): val for d, val in zip(model_dims , b_o) }
        else:
            b_o_dict = {(d): 0 for d in model_dims }
            
        MHA_Block.b_o = pyo.Param(model_dims , initialize=b_o_dict, mutable=False)
            
        # Define variables
        MHA_Block.Q = pyo.Var(MHA_Block.heads, time_dim, MHA_Block.head_dims, within=pyo.Reals) 
        MHA_Block.K = pyo.Var(MHA_Block.heads, time_dim_enc, MHA_Block.head_dims, within=pyo.Reals)
        MHA_Block.V = pyo.Var(MHA_Block.heads, time_dim_enc, MHA_Block.head_dims, within=pyo.Reals) 

        MHA_Block.QK = pyo.Var(MHA_Block.heads, time_dim, time_dim_enc, MHA_Block.head_dims, within=pyo.Reals) 
        MHA_Block.compatibility = pyo.Var(MHA_Block.heads, time_dim, time_dim_enc, within=pyo.Reals) 
        
        scale = 1.0/(np.sqrt(d_heads))
        if norm_softmax:
            MHA_Block.compatibility_max = pyo.Var(MHA_Block.heads, time_dim, within=pyo.Reals) 
            MHA_Block.compatibility_max_s = pyo.Var(MHA_Block.heads, time_dim, time_dim_enc, within=pyo.Binary) 
            MHA_Block.compatibility_scaled = pyo.Var(MHA_Block.heads, time_dim, time_dim_enc, within=pyo.Reals) 
        
        MHA_Block.compatibility_exp = pyo.Var(MHA_Block.heads, time_dim, time_dim_enc, within=pyo.NonNegativeReals, bounds=(0,None)) 
        MHA_Block.compatibility_exp_sum = pyo.Var(MHA_Block.heads, time_dim, within=pyo.NonNegativeReals, bounds=(0,None))
        
        if exp_approx: # usepower series approx exp()
            MHA_Block.compatibility_2 = pyo.Var(MHA_Block.heads, time_dim, time_dim_enc, within=pyo.Reals)
            MHA_Block.compatibility_3 = pyo.Var(MHA_Block.heads, time_dim, time_dim_enc, within=pyo.Reals)
            MHA_Block.compatibility_4 = pyo.Var(MHA_Block.heads, time_dim, time_dim_enc, within=pyo.Reals)
            MHA_Block.compatibility_5 = pyo.Var(MHA_Block.heads, time_dim, time_dim_enc, within=pyo.Reals)
            MHA_Block.compatibility_6 = pyo.Var(MHA_Block.heads, time_dim, time_dim_enc, within=pyo.Reals)
            # MHA_Block.compatibility_7 = pyo.Var(MHA_Block.heads, time_dim, time_dim_enc, within=pyo.Reals)
            # MHA_Block.compatibility_8 = pyo.Var(MHA_Block.heads, time_dim, time_dim_enc, within=pyo.Reals)
            # MHA_Block.compatibility_9 = pyo.Var(MHA_Block.heads, time_dim, time_dim_enc, within=pyo.Reals)
            # MHA_Block.compatibility_10 = pyo.Var(MHA_Block.heads, time_dim, time_dim_enc, within=pyo.Reals)
            # MHA_Block.compatibility_11 = pyo.Var(MHA_Block.heads, time_dim, time_dim_enc, within=pyo.Reals)
    
        if self.bound_cut_activation["MHA_softmax_env"]: 
            MHA_Block.tie_point_cc = pyo.Var(MHA_Block.heads, time_dim, time_dim_enc, within=pyo.Reals)
            MHA_Block.tie_point_cv = pyo.Var(MHA_Block.heads, time_dim, time_dim_enc, within=pyo.Reals)
            MHA_Block.tie_point_cc_prime = pyo.Var(MHA_Block.heads, time_dim, time_dim_enc, within=pyo.Reals)
            MHA_Block.tie_point_cv_prime = pyo.Var(MHA_Block.heads, time_dim, time_dim_enc, within=pyo.Reals)
            MHA_Block.tp_cv_mult_1 = pyo.Var(MHA_Block.heads, time_dim, time_dim_enc, within=pyo.Reals)
            MHA_Block.tp_cv_mult_2 = pyo.Var(MHA_Block.heads, time_dim, time_dim_enc, within=pyo.Reals)
            MHA_Block.tp_cc_mult_1 = pyo.Var(MHA_Block.heads, time_dim, time_dim_enc, within=pyo.Reals)
            MHA_Block.tp_cc_mult_2 = pyo.Var(MHA_Block.heads, time_dim, time_dim_enc, within=pyo.Reals)
            
            BigM_s = 0.5
            BigM_t = 1
            MHA_Block.sct = pyo.Var(MHA_Block.heads, time_dim, time_dim_enc, within=pyo.Reals, bounds=(0,1))
            
            MHA_Block.s_cv= pyo.Var(MHA_Block.heads, time_dim, time_dim_enc, within=pyo.Binary)
            MHA_Block.t_cv= pyo.Var(MHA_Block.heads, time_dim, time_dim_enc, within=pyo.Binary)
            
            MHA_Block.s_cc= pyo.Var(MHA_Block.heads, time_dim, time_dim_enc, within=pyo.Binary)
            MHA_Block.t_cc= pyo.Var(MHA_Block.heads, time_dim, time_dim_enc, within=pyo.Binary)
            
            MHA_Block.tp_cv =pyo.Var(MHA_Block.heads, time_dim, time_dim_enc, within=pyo.Binary)
            MHA_Block.tp_cc =pyo.Var(MHA_Block.heads, time_dim, time_dim_enc, within=pyo.Binary)

            MHA_Block.attention_weight_cc = pyo.Var(MHA_Block.heads, time_dim, time_dim_enc, within=pyo.Reals, bounds=(0,1))
            MHA_Block.attention_weight_x_cc_prime = pyo.Var(MHA_Block.heads, time_dim, time_dim_enc, within=pyo.Reals, bounds=(0,1))
            MHA_Block.attention_weight_x_cc= pyo.Var(MHA_Block.heads, time_dim, time_dim_enc, within=pyo.Reals, bounds=(0,1))
            
            MHA_Block.attention_weight_cv = pyo.Var(MHA_Block.heads, time_dim, time_dim_enc, within=pyo.Reals, bounds=(0,1))
            MHA_Block.attention_weight_x_cv_prime = pyo.Var(MHA_Block.heads, time_dim, time_dim_enc, within=pyo.Reals, bounds=(0,1))
            MHA_Block.attention_weight_x_cv = pyo.Var(MHA_Block.heads, time_dim, time_dim_enc, within=pyo.Reals, bounds=(0,1))

            MHA_Block.tp_cv_sct = pyo.Var(MHA_Block.heads, time_dim, time_dim_enc, within=pyo.Reals, bounds=(0,1))
            MHA_Block.tp_cv_sct_mult_1 = pyo.Var(MHA_Block.heads, time_dim, time_dim_enc, within=pyo.Reals)
            MHA_Block.tp_cv_sct_mult_2 = pyo.Var(MHA_Block.heads, time_dim, time_dim_enc, within=pyo.Reals)
            MHA_Block.tp_cv_sct_mult_1_2 = pyo.Var(MHA_Block.heads, time_dim, time_dim_enc, within=pyo.Reals)
            MHA_Block.tp_cv_sct_mult_3 = pyo.Var(MHA_Block.heads, time_dim, time_dim_enc, within=pyo.Reals)
            
            MHA_Block.tp_cc_sct = pyo.Var(MHA_Block.heads, time_dim, time_dim_enc, within=pyo.Reals, bounds=(0,1))
            MHA_Block.tp_cc_sct_mult_1 = pyo.Var(MHA_Block.heads, time_dim, time_dim_enc, within=pyo.Reals)
            MHA_Block.tp_cc_sct_mult_2 = pyo.Var(MHA_Block.heads, time_dim, time_dim_enc, within=pyo.Reals)
            MHA_Block.tp_cc_sct_mult_1_2 = pyo.Var(MHA_Block.heads, time_dim, time_dim_enc, within=pyo.Reals)
            MHA_Block.tp_cc_sct_mult_3 = pyo.Var(MHA_Block.heads, time_dim, time_dim_enc, within=pyo.Reals)
        
        MHA_Block.attention_weight = pyo.Var(MHA_Block.heads, time_dim, time_dim_enc, within=pyo.Reals)#, bounds=(0,1))  # softmax ( (Q * K)/sqrt(d_k) )
        MHA_Block.attention_score = pyo.Var(
            MHA_Block.heads, time_dim, MHA_Block.head_dims, within=pyo.Reals
        )  # softmax ( (Q * K)/sqrt(d_k) ) * V
        MHA_Block.attWK = pyo.Var(MHA_Block.heads, time_dim, MHA_Block.head_dims, time_dim_enc , within=pyo.Reals)
        
        # Add multihead attention constaints
        for h in MHA_Block.heads: # for each head
            # Check if multihead attention or self attention
            if cross_attn and not encoder_output is None:
                input = encoder_output_var # calculate K and V from output of encoder
            else:
                input = input_var # calculate K and V from input variable

            # Define K and V
            for n in time_dim_enc: # for each sequence element
                    for k in MHA_Block.head_dims: # for each dimension of a given head

                        # constraints for Key
                        MHA_Block.attention_constraints.add(
                            expr=MHA_Block.K[h, n, k]
                            == sum(input[n, d] * MHA_Block.W_k[d, h, k] for d in model_dims_enc ) + MHA_Block.b_k[h,k]
                            )  
                        #Add bounds
                        if self.bound_cut_activation["MHA_K"]: 
                            try:
                                MHA_Block.K[h, n, k].ub = sum( max(input[n,d].ub * MHA_Block.W_k[d, h, k], input[n,d].lb * MHA_Block.W_k[d, h, k])  for d in model_dims_enc ) + MHA_Block.b_k[h,k]
                                MHA_Block.K[h, n, k].lb = sum( min(input[n,d].ub * MHA_Block.W_k[d, h, k], input[n,d].lb * MHA_Block.W_k[d, h, k])  for d in model_dims_enc ) + MHA_Block.b_k[h,k]
                            except:
                                pass
                            
                        # constraints for Value    
                        MHA_Block.attention_constraints.add(
                        expr=MHA_Block.V[h, n, k]
                        == sum(input[n, d] * MHA_Block.W_v[d, h, k] for d in model_dims_enc) + MHA_Block.b_v[h,k]
                        )  
                        #Add bounds
                        if self.bound_cut_activation["MHA_V"]: 
                            try:
                                MHA_Block.V[h, n, k].ub = sum( max(input[n,d].ub * MHA_Block.W_v[d, h, k], input[n,d].lb * MHA_Block.W_v[d, h, k])  for d in model_dims_enc ) + MHA_Block.b_v[h,k]
                                MHA_Block.V[h, n, k].lb = sum( min(input[n,d].ub * MHA_Block.W_v[d, h, k], input[n,d].lb * MHA_Block.W_v[d, h, k])  for d in model_dims_enc ) + MHA_Block.b_v[h,k]
                            except:
                                pass
            for n in time_dim:
                    for k in MHA_Block.head_dims:
                        
                        # constraints for Query
                        MHA_Block.attention_constraints.add(
                            expr=MHA_Block.Q[h, n, k]
                            == sum(input_var[n,d] * MHA_Block.W_q[d, h, k] for d in model_dims ) + MHA_Block.b_q[h,k] 
                            )  
                            
                        #Add bounds
                        if self.bound_cut_activation["MHA_Q"]: 
                            try:
                                MHA_Block.Q[h, n, k].ub = sum( max(input_var[n,d].ub * MHA_Block.W_q[d, h, k], input_var[n,d].lb * MHA_Block.W_q[d, h, k])  for d in model_dims ) + MHA_Block.b_q[h,k]
                                MHA_Block.Q[h, n, k].lb = sum( min(input_var[n,d].ub * MHA_Block.W_q[d, h, k], input_var[n,d].lb * MHA_Block.W_q[d, h, k])  for d in model_dims ) + MHA_Block.b_q[h,k]
                            except:
                                pass
                                
                        # attention score = sum(attention_weight * V)
                        for p in time_dim_enc:
                            MHA_Block.attention_constraints.add(expr= MHA_Block.attWK[h, n, k, p] == MHA_Block.attention_weight[h, n, p] * MHA_Block.V[h, p, k])
                        MHA_Block.attention_constraints.add(
                            expr=MHA_Block.attention_score[h, n, k]
                            == sum( MHA_Block.attWK[h, n, k, n2] for n2 in time_dim_enc)
                        )   
                        
                        
                    for p in time_dim_enc:
                        #compatibility sqrt(Q * K) across all pairs of elements
                        for k in MHA_Block.head_dims:
                            if mask and p > n:
                                continue
                            else:
                                MHA_Block.attention_constraints.add(expr=MHA_Block.QK[h, n, p, k] == ( MHA_Block.Q[h, n, k]) * MHA_Block.K[ h, p, k])
                            

                        # Max normalised compatibility
                        if norm_softmax:
                            """ exp(compatibility)
                            from Keras Softmax: 
                                exp_x = exp(x - max(x))
                                f(x) = exp_x / sum(exp_x)
                            """
                            if mask and p > n:
                                continue
                            else:
                                MHA_Block.attention_constraints.add(
                                    expr=MHA_Block.compatibility_scaled[h, n, p] 
                                    ==  scale * sum(MHA_Block.QK[h, n, p, k] for k in MHA_Block.head_dims)
                                ) 
                                MHA_Block.attention_constraints.add(expr= MHA_Block.compatibility_scaled[h,n,p] <= MHA_Block.compatibility_max[h,n])
                                try:
                                    
                                    max_compat = 0
                                    for k in MHA_Block.k_dims:
                                        compat_uu = MHA_Block.Q[h, n, k].ub * MHA_Block.K[h, p, k].ub
                                        compat_ul = MHA_Block.Q[h, n, k].ub * MHA_Block.K[h, p, k].lb
                                        compat_lu = MHA_Block.Q[h, n, k].lb * MHA_Block.K[h, p, k].ub
                                        compat_ll = MHA_Block.Q[h, n,k].lb  * MHA_Block.K[h, p, k].lb
                                        max_compat += max([compat_uu, compat_uu, compat_lu, compat_ll]) # sum max for each k       
                                    M_max_compat = scale * max_compat 
                                except:
                                    M_max_compat = 100 * self.d_k # Used when no bounds for Q or K. Expect that 100 >> values calculated in TNN (NN values usually in range -1 to 1)
                                    
                                MHA_Block.attention_constraints.add(expr=  MHA_Block.compatibility_scaled[h,n,p]  >= MHA_Block.compatibility_max[h,n] - (M_max_compat * (1 - MHA_Block.compatibility_max_s[h,n,p])))
                                MHA_Block.attention_constraints.add(expr= MHA_Block.compatibility[h,n,p] == MHA_Block.compatibility_scaled[h,n,p] - MHA_Block.compatibility_max[h,n])
                        
                        # Compatibility
                        else:
                            if mask and p > n:
                                continue
                            else:
                                MHA_Block.attention_constraints.add(
                                    expr=MHA_Block.compatibility[h, n, p] 
                                    ==  scale * sum(MHA_Block.QK[h, n, p, k] for k in MHA_Block.head_dims)
                                ) 

                        if exp_approx: # usepower series approx exp()
                            if mask and p > n:
                                MHA_Block.attention_constraints.add(expr= MHA_Block.compatibility_exp[h, n, p] == 0)
                            else:
                                # power series approx for EXP
                                MHA_Block.attention_constraints.add(expr= MHA_Block.compatibility[h, n, p]**2 == MHA_Block.compatibility_2[h, n, p] )#problem for gurobi
                                MHA_Block.attention_constraints.add(expr= MHA_Block.compatibility[h, n, p]*MHA_Block.compatibility_2[h, n, p] == MHA_Block.compatibility_3[h, n, p] )
                                MHA_Block.attention_constraints.add(expr= MHA_Block.compatibility[h, n, p]*MHA_Block.compatibility_3[h, n, p] == MHA_Block.compatibility_4[h, n, p] )
                                MHA_Block.attention_constraints.add(expr= MHA_Block.compatibility[h, n, p]*MHA_Block.compatibility_4[h, n, p] == MHA_Block.compatibility_5[h, n, p] )
                                MHA_Block.attention_constraints.add(expr= MHA_Block.compatibility[h, n, p]*MHA_Block.compatibility_5[h, n, p] == MHA_Block.compatibility_6[h, n, p] )
                                # MHA_Block.attention_constraints.add(expr= MHA_Block.compatibility[h, n, p]*MHA_Block.compatibility_6[h, n, p] == MHA_Block.compatibility_7[h, n, p] )
                                # MHA_Block.attention_constraints.add(expr= MHA_Block.compatibility[h, n, p]*MHA_Block.compatibility_7[h, n, p] == MHA_Block.compatibility_8[h, n, p] )
                                # MHA_Block.attention_constraints.add(expr= MHA_Block.compatibility[h, n, p]*MHA_Block.compatibility_8[h, n, p] == MHA_Block.compatibility_9[h, n, p] )
                                # MHA_Block.attention_constraints.add(expr= MHA_Block.compatibility[h, n, p]*MHA_Block.compatibility_9[h, n, p] == MHA_Block.compatibility_10[h, n, p] )
                                # MHA_Block.attention_constraints.add(expr= MHA_Block.compatibility[h, n, p]*MHA_Block.compatibility_10[h, n, p] == MHA_Block.compatibility_11[h, n, p] )
                                
                                MHA_Block.attention_constraints.add(expr= MHA_Block.compatibility_exp[h, n, p] == 1
                                                            + MHA_Block.compatibility[h, n, p]
                                                            + (0.5*MHA_Block.compatibility_2[h, n, p] ) 
                                                            + (0.166666667*MHA_Block.compatibility_3[h, n, p]) 
                                                            + (0.0416666667*MHA_Block.compatibility_4[h, n, p]) 
                                                            + (0.00833333333*MHA_Block.compatibility_5[h, n, p]) 
                                                            + (0.00138888889*MHA_Block.compatibility_6[h, n, p]) 
                                                            # + (0.000198412698*MHA_Block.compatibility_7[h, n, p]) 
                                                            # + (0.0000248015873*MHA_Block.compatibility_8[h, n, p]) 
                                                            # + (0.00000275573192*MHA_Block.compatibility_9[h, n, p]) 
                                                            # + (0.000000275573192*MHA_Block.compatibility_10[h, n, p])
                                                            # + (0.0000000250521084*MHA_Block.compatibility_11[h, n, p])
                                                            )# pyo.exp() only seems to work for constant args and pow operator must be <= 2
                        else:  # use exponential function  
                            if mask and p > n:
                                MHA_Block.attention_constraints.add(expr= MHA_Block.compatibility_exp[h, n, p] == 0)
                            else:
                                MHA_Block.attention_constraints.add(expr= pyo.exp(MHA_Block.compatibility[h,n,p]) == MHA_Block.compatibility_exp[h, n, p] )

                    # max compatibility: slack sum to 1
                    if norm_softmax:
                        MHA_Block.attention_constraints.add(expr=  sum(MHA_Block.compatibility_max_s[h,n,p] for p in time_dim_enc) == 1)    
                    
                    # sum over exp(compatbility)
                    MHA_Block.attention_constraints.add(expr= MHA_Block.compatibility_exp_sum[h, n] == sum(MHA_Block.compatibility_exp[h, n, p] for p in time_dim_enc))
                    
                    # sum over softmax = 1  
                    if self.bound_cut_activation["MHA_attn_weight_sum"]:   
                        MHA_Block.attention_constraints.add(
                            expr=sum(MHA_Block.attention_weight[h, n, n_prime] for n_prime in time_dim_enc) == 1
                        )
                    
                    for n2 in time_dim_enc:
                        if self.bound_cut_activation["MHA_attn_weight"]: 
                            MHA_Block.attention_weight[h, n, n2].ub = 1
                            MHA_Block.attention_weight[h, n, n2].lb = 0
                
                        #attention weights softmax(compatibility)  
                        MHA_Block.attention_constraints.add(
                            expr=MHA_Block.attention_weight[h, n, n2] * MHA_Block.compatibility_exp_sum[h, n]
                            == MHA_Block.compatibility_exp[h, n, n2]) 
                           
            #Add bounds            
            for n in time_dim:
                for p in time_dim_enc:
                    if self.bound_cut_activation["MHA_compat"]:
                        if norm_softmax:   
                                
                            MHA_Block.compatibility[h,n,p].ub = 0 
                            for k in MHA_Block.head_dims: 
                                
                                if MHA_Block.Q[h, n, k].lb is not None and MHA_Block.Q[h, n, k].ub is not None and MHA_Block.K[h, p, k].lb is not None and MHA_Block.K[h, p, k].ub is not None:
                                    MHA_Block.QK[h,n,p,k].ub = (max(MHA_Block.Q[h, n, k].lb * MHA_Block.K[ h, p, k].lb,
                                                                                            MHA_Block.Q[h, n, k].lb * MHA_Block.K[ h, p, k].ub, 
                                                                                            MHA_Block.Q[h, n, k].ub * MHA_Block.K[ h, p, k].lb, 
                                                                                            MHA_Block.Q[h, n, k].ub * MHA_Block.K[ h, p, k].ub) )
                                    MHA_Block.QK[h,n,p,k].lb =  (min(MHA_Block.Q[h, n, k].lb * MHA_Block.K[ h, p, k].lb, 
                                                                                            MHA_Block.Q[h, n, k].lb * MHA_Block.K[ h, p, k].ub, 
                                                                                            MHA_Block.Q[h, n, k].ub * MHA_Block.K[ h, p, k].lb, 
                                                                                            MHA_Block.Q[h, n, k].ub * MHA_Block.K[ h, p, k].ub) )                                                                 
                        else:
                            for k in MHA_Block.head_dims: 
                                if MHA_Block.Q[h, n, k].lb is not None and MHA_Block.Q[h, n, k].ub is not None and MHA_Block.K[h, p, k].lb is not None and MHA_Block.K[h, p, k].ub is not None:
                                    MHA_Block.QK[h,n,p,k].ub = (max(MHA_Block.Q[h, n, k].lb * MHA_Block.K[ h, p, k].lb,
                                                                                                MHA_Block.Q[h, n, k].lb * MHA_Block.K[ h, p, k].ub, 
                                                                                                MHA_Block.Q[h, n, k].ub * MHA_Block.K[ h, p, k].lb, 
                                                                                                MHA_Block.Q[h, n, k].ub * MHA_Block.K[ h, p, k].ub) )
                                    MHA_Block.QK[h,n,p,k].lb =  (min(MHA_Block.Q[h, n, k].lb * MHA_Block.K[ h, p, k].lb, 
                                                                                                MHA_Block.Q[h, n, k].lb * MHA_Block.K[ h, p, k].ub, 
                                                                                                MHA_Block.Q[h, n, k].ub * MHA_Block.K[ h, p, k].lb, 
                                                                                                MHA_Block.Q[h, n, k].ub * MHA_Block.K[ h, p, k].ub) )  
                            if MHA_Block.Q[h, n, MHA_Block.head_dims.first()].lb is not None:
                                MHA_Block.compatibility[h,n,p].ub = sum(MHA_Block.QK[h,n,p,k].ub for k in MHA_Block.head_dims)  
                                MHA_Block.compatibility[h,n,p].lb = sum(MHA_Block.QK[h,n,p,k].lb for k in MHA_Block.head_dims)       
                            else:
                                max_compat = 100 * self.d_k
                                min_compat = -100 * self.d_k
                                MHA_Block.compatibility[h,n,p].ub = max_compat
                                MHA_Block.compatibility[h,n,p].lb = min_compat       

                    if self.bound_cut_activation["MHA_compat_exp"]: 
                        try: 
                            if norm_softmax:
                                MHA_Block.compatibility_exp[h,n,p].ub = min(1, math.exp(MHA_Block.compatibility[h,n,p].ub))
                                MHA_Block.compatibility_exp[h,n,p].lb = math.exp(MHA_Block.compatibility[h,n,p].lb)
                            else:
                                MHA_Block.compatibility_exp[h,n,p].ub = math.exp(MHA_Block.compatibility[h,n,p].ub)
                                MHA_Block.compatibility_exp[h,n,p].lb = math.exp(MHA_Block.compatibility[h,n,p].lb)
                        except:
                            pass    
                    for k in MHA_Block.head_dims:  
                        if self.bound_cut_activation["MHA_QK_MC"]: 
                            if not mask and p < n :
                                if MHA_Block.Q[h, n, k].lb is not None and MHA_Block.Q[h, n, k].ub is not None and MHA_Block.K[h, p, k].lb is not None and MHA_Block.K[h, p, k].ub is not None:
                                    self.__McCormick_bb(MHA_Block.QK[h, n, p, k], MHA_Block.Q[h, n, k], MHA_Block.K[ h, p, k]) # add McCromick Envelope
   
                        if self.bound_cut_activation["MHA_WK_MC"]: 
                            if MHA_Block.attention_weight[h, n, p].lb is not None and MHA_Block.attention_weight[h, n, p].ub is not None and MHA_Block.V[h, p, k].lb is not None and MHA_Block.V[h, p, k].ub is not None:
                                self.__McCormick_bb(MHA_Block.attWK[h, n, k, p], MHA_Block.attention_weight[h, n, p], MHA_Block.V[h, p, k]) # add McCromick Envelope

                            
                    if self.bound_cut_activation["MHA_attn_score"]: 
                        for k in MHA_Block.head_dims: 
                            if MHA_Block.attention_weight[h, n, p].lb is not None and MHA_Block.attention_weight[h, n, p].ub is not None and MHA_Block.V[h, p, k].lb is not None and MHA_Block.V[h, p, k].ub is not None:
                                MHA_Block.attWK[h, n, k, p] .ub = (max(MHA_Block.attention_weight[h, n, p].lb * MHA_Block.V[h, p, k].lb,
                                                                            MHA_Block.attention_weight[h, n, p].lb * MHA_Block.V[h, p, k].ub, 
                                                                            MHA_Block.attention_weight[h, n, p].ub * MHA_Block.V[h, p, k].lb, 
                                                                            MHA_Block.attention_weight[h, n, p].ub * MHA_Block.V[h, p, k].ub) 
                                                                    )
                                MHA_Block.attWK[h, n, k, p] .lb = (min(MHA_Block.attention_weight[h, n,p].lb * MHA_Block.V[h, p, k].lb, 
                                                                                MHA_Block.attention_weight[h, n, p].lb * MHA_Block.V[h, p, k].ub, 
                                                                                MHA_Block.attention_weight[h, n, p].ub * MHA_Block.V[h, p, k].lb, 
                                                                                MHA_Block.attention_weight[h, n, p].ub * MHA_Block.V[h, p, k].ub) 
                                                                        )  

                        if MHA_Block.attWK[h, n, MHA_Block.head_dims.first(), p].lb is not None:
                            MHA_Block.attention_score[h, n, k].ub = sum(MHA_Block.attWK[h,n,k,p].ub for k in MHA_Block.head_dims)  
                            MHA_Block.attention_score[h, n, k].lb = sum(MHA_Block.attWK[h,n,k,p].lb for k in MHA_Block.head_dims)       
                
                if norm_softmax:
                    if MHA_Block.QK[h, n, p,MHA_Block.head_dims.first()].lb is not None:
                        max_compat = max( sum(MHA_Block.QK[h,n,n2,k].ub for k in MHA_Block.head_dims) for n2 in time_dim_enc)
                        MHA_Block.compatibility[h,n,p].lb = sum(MHA_Block.QK[h,n,p,k].lb for k in MHA_Block.head_dims) - max_compat    
                    else:
                        min_compat = -100 * self.d_k
                        MHA_Block.compatibility[h,n,p].lb = min_compat            
                 
                if self.bound_cut_activation["MHA_compat_exp_sum"]:   
                    try: 
                        if norm_softmax:
                            MHA_Block.compatibility_exp_sum[h, n].lb = max(1,sum( MHA_Block.compatibility_exp[h,n,n2].lb for n2 in time_dim))   # exp( x-max(x)) --> at least one value in sum = 1
                        else:
                            MHA_Block.compatibility_exp_sum[h, n].lb = sum( MHA_Block.compatibility_exp[h,n,n2].lb for n2 in time_dim)   
                            
                        MHA_Block.compatibility_exp_sum[h, n].ub = sum( MHA_Block.compatibility_exp[h,n,n2].ub for n2 in time_dim) 
                    except:
                        if norm_softmax:
                            MHA_Block.compatibility_exp_sum[h, n].lb = 1
                    
                if self.bound_cut_activation["MHA_softmax_env"]: 
                #-- begin add softmax outer approximators --# 
                    try: 
                        for p in time_dim_enc:    
                            # f(x) >= f_cv(x): attention weight >= convex envelope
                            MHA_Block.attention_constraints.add(
                                MHA_Block.attention_weight[h, n, p]  >= MHA_Block.attention_weight_cv[h, n, p]
                            )
                            # f(x) <= f_cc(x): attention weight <= concave envelope
                            MHA_Block.attention_constraints.add(
                                MHA_Block.attention_weight[h, n, p]  <= MHA_Block.attention_weight_cc[h, n, p]
                            )
                
                            # Constraints for Concave/convex envelope
                            # set convex aux var -- s=0: f(x_UB) <= 0.5 --> convex zone, s=1: f(x_UB) >= 0.5 --> concave zone
                            MHA_Block.attention_constraints.add(
                                expr= MHA_Block.attention_weight[h, n, p].ub <= 0.5  + (BigM_s * MHA_Block.s_cv[h,n,p])
                            )
                            
                            # MHA_Block.attention_constraints.add(
                            #     expr= MHA_Block.attention_weight[h, n, p].ub >= 0.5  - (BigM_s * (1 - MHA_Block.s_cv[h,n,p]))
                            # )
                            MHA_Block.attention_constraints.add(
                                expr= MHA_Block.attention_weight[h, n, p].ub - 0.5 + BigM_s >= BigM_s *  MHA_Block.s_cv[h,n,p]
                            )

                            # set convex aux var -- f(x_LB) <= 0.5 --> convex zone else f(x_LB) >= 0.5 --> concave zone
                            MHA_Block.attention_constraints.add(
                                expr= MHA_Block.attention_weight[h, n, p].lb >= 0.5 - (BigM_s *  (MHA_Block.s_cc[h,n,p]))
                            )
                            MHA_Block.attention_constraints.add(
                                expr= (BigM_s * MHA_Block.s_cc[h,n,p]) <= 0.5 + BigM_s - MHA_Block.attention_weight[h, n, p].lb
                            )
                            
                            # # sct(x)
                            A = ((MHA_Block.attention_weight[h, n, p].ub - MHA_Block.attention_weight[h, n, p].lb) / (MHA_Block.compatibility[h,n,p].ub - MHA_Block.compatibility[h,n,p].lb )) 
                            b = ( (MHA_Block.compatibility[h,n,p].ub * MHA_Block.attention_weight[h, n, p].lb) - (MHA_Block.compatibility[h,n,p].lb * MHA_Block.attention_weight[h, n, p].ub)) /(MHA_Block.compatibility[h,n,p].ub - MHA_Block.compatibility[h,n,p].lb )
                            MHA_Block.attention_constraints.add(
                                MHA_Block.sct[h, n, p]   == (A *  MHA_Block.compatibility[h,n,p]) + b
                            )

                            # # # # Add concave/convex evelope function constraints
                            # # when f(UB) <= 0.5: convex
                            MHA_Block.constr_convex.add( 
                                MHA_Block.attention_weight_cv[h, n, p] == MHA_Block.attention_weight[h, n, p]
                            )
                            MHA_Block.constr_convex.add( 
                                MHA_Block.attention_weight_cc[h, n, p] == MHA_Block.sct[h, n, p] 
                            )
                            # when f(LB) >= 0.5: concave 
                            MHA_Block.constr_concave.add( 
                                MHA_Block.attention_weight_cv[h, n, p] == MHA_Block.sct[h, n, p] 
                            )
                            MHA_Block.constr_concave.add( 
                                MHA_Block.attention_weight_cc[h, n, p] == MHA_Block.attention_weight[h, n, p] 
                            )
                            # otherwise: use concave and convex tie points
                            MHA_Block.constr_concave_tp.add( # when x >= x_cc
                                MHA_Block.attention_weight_cc[h, n, p] == MHA_Block.attention_weight[h, n, p] 
                            )
                            MHA_Block.constr_concave_tp_sct.add( # when x <= x_cc --> cc_sct()
                                MHA_Block.attention_weight_cc[h, n, p] == MHA_Block.tp_cc_sct[h, n, p]
                            ) 
                            MHA_Block.constr_convex_tp_sct.add( # when x >= x_cv --> cv_sct()
                                MHA_Block.attention_weight_cv[h, n, p] == MHA_Block.tp_cv_sct[h, n, p]
                            ) 
                            MHA_Block.constr_convex_tp.add( # when x <= x_cv
                                MHA_Block.attention_weight_cv[h, n, p] == MHA_Block.attention_weight[h, n, p]
                            )
                            
                            ## Add tp_cv_sct constraints
                            #bounds
                            MHA_Block.attention_constraints.add(# att(cv_prime)
                                expr=  MHA_Block.attention_weight_x_cv_prime[h, n, n2] <= 1 
                            )
                            MHA_Block.attention_constraints.add( # att(x_cv)
                                expr=  MHA_Block.attention_weight_x_cv[h, n, n2] <= 1
                            )
                            MHA_Block.attention_constraints.add( # att(x_cv)
                                expr=   MHA_Block.tp_cv_sct[h, n, p] <= 1
                            )
                            # tie_point_cv[h, n, p] = max(tie_point_cv_prime, compatibility.lb  )
                            BigM_prime = max( MHA_Block.compatibility[h,n,p_prime].ub for p_prime in time_dim)
                            MHA_Block.attention_constraints.add(
                                MHA_Block.tie_point_cv_prime[h, n, p] - MHA_Block.compatibility[h,n,p].lb <= BigM_prime * (1 - MHA_Block.tp_cv[h,n,p])
                            )
                            MHA_Block.attention_constraints.add(
                                MHA_Block.tie_point_cv_prime[h, n, p] - MHA_Block.compatibility[h,n,p].lb >= -BigM_prime * ( MHA_Block.tp_cv[h,n,p])
                            )
                            MHA_Block.attention_constraints.add( # define tie_point_cv
                                MHA_Block.tie_point_cv[h, n, p]  == MHA_Block.tie_point_cv_prime[h, n, p]*(1 - MHA_Block.tp_cv[h,n,p])  + (MHA_Block.compatibility[h,n,p].lb * MHA_Block.tp_cv[h,n,p])
                            )
                            MHA_Block.attention_constraints.add( # softmax(tie_point_cv)
                                MHA_Block.attention_weight_x_cv[h, n, p] == MHA_Block.attention_weight_x_cv_prime[h, n, p]*(1 - MHA_Block.tp_cv[h,n,p])  + MHA_Block.attention_weight[h,n,p].lb * MHA_Block.tp_cv[h,n,p]
                            )
                            # Is x <= x_cv? --> convex zone
                            MHA_Block.attention_constraints.add(
                                expr=  MHA_Block.tie_point_cv[h, n, p] - MHA_Block.compatibility[h,n,p] <= BigM_prime * (1-MHA_Block.t_cv[h,n,p])
                            )
                            MHA_Block.attention_constraints.add(
                                expr=  MHA_Block.tie_point_cv[h, n, p] - MHA_Block.compatibility[h,n,p] >= - BigM_prime * (MHA_Block.t_cv[h,n,p])
                            )
                            # define tie_point_cv_prime[h, n, p]
                            MHA_Block.attention_constraints.add( # 
                                expr=  MHA_Block.tp_cv_mult_1[h, n, p]  == MHA_Block.attention_weight[h,n,p].ub  - MHA_Block.attention_weight_x_cv_prime[h, n, p]
                            )
                            MHA_Block.attention_constraints.add( # 
                                expr=  MHA_Block.tp_cv_mult_2[h, n, p]  == MHA_Block.attention_weight_x_cv_prime[h, n, p] * ( 1 -  MHA_Block.attention_weight_x_cv_prime[h, n, p])
                            )
                            MHA_Block.attention_constraints.add( 
                                expr=  (MHA_Block.compatibility[h,n,p].ub - MHA_Block.tie_point_cv_prime[h, n, p]) * MHA_Block.tp_cv_mult_2[h, n, p]  == MHA_Block.tp_cv_mult_1[h, n, p]
                            )
                            # define tie point cv  secant
                            MHA_Block.constr_convex_tp_sct.add( 
                                expr=  MHA_Block.tp_cv_sct[h, n, p] - MHA_Block.attention_weight[h,n,p].ub == 
                                                                    + (MHA_Block.tp_cv_sct_mult_1_2[h, n, p] 
                                                                    * (MHA_Block.compatibility[h,n,p]
                                                                        - MHA_Block.compatibility[h,n,p].ub))
                            )
                            MHA_Block.constr_convex_tp_sct.add( 
                                expr=  MHA_Block.tp_cv_sct_mult_1_2[h, n, p] * MHA_Block.tp_cv_sct_mult_2[h, n, p] == MHA_Block.tp_cv_sct_mult_1[h, n, p] 
                            )
                            MHA_Block.constr_convex_tp_sct.add( 
                                expr=  MHA_Block.tp_cv_sct_mult_1[h, n, p] == MHA_Block.attention_weight[h,n,p].ub -  MHA_Block.attention_weight_x_cv[h, n, p]
                            )
                            MHA_Block.constr_convex_tp_sct.add( 
                                expr=  MHA_Block.tp_cv_sct_mult_2[h, n, p] == MHA_Block.compatibility[h,n,p].ub - MHA_Block.tie_point_cv[h, n, p]
                            )
                            
                            ## Add tp_cc_sct constraints
                            #bounds
                            MHA_Block.attention_constraints.add(# att(cc_prime)
                                expr=  MHA_Block.attention_weight_x_cc_prime[h, n, n2] <= 1 
                            )
                            MHA_Block.attention_constraints.add( # att(x_cc)
                                expr=  MHA_Block.attention_weight_x_cc[h, n, n2] <= 1
                            )
                            MHA_Block.attention_constraints.add( # att(x_cc)
                                expr=   MHA_Block.tp_cc_sct[h, n, p] <= 1
                            )
                            # tie_point_cc[h, n, p] = min(tie_point_cc_prime, compatibility.ub  )
                            MHA_Block.attention_constraints.add(
                                MHA_Block.tie_point_cc_prime[h, n, p] - MHA_Block.compatibility[h,n,p].ub <= BigM_prime * (1 - MHA_Block.tp_cc[h,n,p])
                            )
                            MHA_Block.attention_constraints.add(
                                MHA_Block.tie_point_cc_prime[h, n, p] - MHA_Block.compatibility[h,n,p].ub >= -BigM_prime * ( MHA_Block.tp_cc[h,n,p])
                            )
                            MHA_Block.attention_constraints.add( # define tie_point_cc
                                MHA_Block.tie_point_cc[h, n, p]  == MHA_Block.tie_point_cc_prime[h, n, p]*(MHA_Block.tp_cc[h,n,p])  + (MHA_Block.compatibility[h,n,p].ub * (1 - MHA_Block.tp_cc[h,n,p]))
                            )
                            MHA_Block.attention_constraints.add( # softmax(tie_point_cc)
                                MHA_Block.attention_weight_x_cc[h, n, p] == MHA_Block.attention_weight_x_cc_prime[h, n, p]*(MHA_Block.tp_cc[h,n,p])  + (MHA_Block.attention_weight[h,n,p].ub * (1 - MHA_Block.tp_cc[h,n,p]))
                            )
                            # Is x <= x_cc? --> convex zone
                            MHA_Block.attention_constraints.add(
                                expr=  MHA_Block.compatibility[h,n,p] - MHA_Block.tie_point_cc[h, n, p] <= BigM_prime * (1 - MHA_Block.t_cc[h,n,p])
                            )
                            MHA_Block.attention_constraints.add(
                                expr=  MHA_Block.compatibility[h,n,p] - MHA_Block.tie_point_cc[h, n, p]>= - BigM_prime * (MHA_Block.t_cc[h,n,p])
                            )
                            # define tie_point_cc_prime[h, n, p]
                            MHA_Block.attention_constraints.add( # 
                                expr=  MHA_Block.tp_cc_mult_1[h, n, p]  == MHA_Block.attention_weight_x_cc_prime[h, n, p] - MHA_Block.attention_weight[h,n,p].lb
                            )
                            MHA_Block.attention_constraints.add( # 
                                expr=  MHA_Block.tp_cc_mult_2[h, n, p]  == MHA_Block.attention_weight_x_cc_prime[h, n, p] * ( 1 -  MHA_Block.attention_weight_x_cc_prime[h, n, p])
                            )
                            MHA_Block.attention_constraints.add( 
                                expr=  (MHA_Block.tie_point_cc_prime[h, n, p] - MHA_Block.compatibility[h,n,p].lb ) * MHA_Block.tp_cc_mult_2[h, n, p]  == MHA_Block.tp_cc_mult_1[h, n, p]
                            )
                            # define tie point cc  secant
                            MHA_Block.constr_concave_tp_sct.add( 
                                expr=  MHA_Block.tp_cc_sct[h, n, p] - MHA_Block.attention_weight[h,n,p].lb == 
                                                                    + (MHA_Block.tp_cc_sct_mult_1_2[h, n, p] 
                                                                    * (MHA_Block.compatibility[h,n,p]
                                                                        - MHA_Block.compatibility[h,n,p].lb))
                            )
                            MHA_Block.constr_concave_tp_sct.add( 
                                expr=  MHA_Block.tp_cc_sct_mult_1_2[h, n, p] * MHA_Block.tp_cc_sct_mult_2[h, n, p] == MHA_Block.tp_cc_sct_mult_1[h, n, p] 
                            )
                            MHA_Block.constr_concave_tp_sct.add( 
                                expr=  MHA_Block.tp_cc_sct_mult_1[h, n, p] == MHA_Block.attention_weight[h,n,p].lb -  MHA_Block.attention_weight_x_cc[h, n, p]
                            )
                            MHA_Block.constr_concave_tp_sct.add( 
                                expr=  MHA_Block.tp_cc_sct_mult_2[h, n, p] == MHA_Block.compatibility[h,n,p].lb - MHA_Block.tie_point_cc[h, n, p]
                            )
                    except:
                        pass 
                    
   
        # multihead attention output constraint
        for n in time_dim:
            for d in model_dims :
                MHA_Block.attention_constraints.add(
                        expr= attention_output[n, d]
                        == sum(
                            (sum(
                            MHA_Block.attention_score[h, n, k] * MHA_Block.W_o[d,h, k]
                            for k in MHA_Block.head_dims
                             ) )
                        for h in MHA_Block.heads
                        
                        ) + MHA_Block.b_o[d]
                    )
                if self.bound_cut_activation["MHA_output"]: 
                    try:
                        attention_output[n, d].ub  = sum(sum( max(MHA_Block.attention_score[h, n, k].ub * MHA_Block.W_o[d,h, k], MHA_Block.attention_score[h, n, k].lb * MHA_Block.W_o[d,h, k]) for k in MHA_Block.head_dims) for h in MHA_Block.heads) + MHA_Block.b_o[d]
                        attention_output[n, d].lb  = sum(sum( min(MHA_Block.attention_score[h, n, k].ub * MHA_Block.W_o[d,h, k], MHA_Block.attention_score[h, n, k].lb * MHA_Block.W_o[d,h, k]) for k in MHA_Block.head_dims) for h in MHA_Block.heads) + MHA_Block.b_o[d]
                    except:
                        pass
        # # activate softmax envelope constraints
        if self.bound_cut_activation["MHA_softmax_env"]:               
            MHA_Block.activate_constraints = pyo.BuildAction(rule=activate_envelope_att)               
        return attention_output
    
    def add_residual_connection(self, input_1_name:Union[pyo.Var,str], input_2_name:Union[pyo.Var,str], output_var_name):
        """
        Adds a residual connection between two input variables and creates a new variable representing their sum.

        This method adds constraints to ensure the residual variable is the element-wise sum of two input variables
        across the sequence and feature dimensions. Additionally, it propagates upper and lower bounds for the residual variable
        if bounds are defined for the inputs.

        Args:
            input_1_name (Union[pyo.Var, str]): The first input variable or its name as a string.
            input_2_name (Union[pyo.Var, str]): The second input variable or its name as a string.
            output_var_name (str): The name of the output variable representing the residual connection.

        Returns:
            pyo.Var: The Pyomo variable representing the residual connection.
        """
        # Get input 1
        if not isinstance(input_1_name, pyo.Var):
            input_1 = getattr(self.M, input_1_name)
        else:
            input_1 = input_1_name
            
        # Get input 2
        if not isinstance(input_2_name, pyo.Var):
            input_2 = getattr(self.M, input_2_name)
        else:
            input_2 = input_2_name
        
        # Get indices
        if input_1.is_indexed():
            set_var = input_1.index_set()
            indices = []
            for set in str(set_var).split("*"):
                indices.append( getattr( self.M, set) )
                
            time_dim = indices[0]
            model_dims = indices[1]
        else:
            raise ValueError('Input value must be indexed (time, model_dim)')
        
        # Create constraint list
        if not hasattr( self.M, "residual_constraints"):
            self.M.residual_constraints = pyo.ConstraintList()
        
        # Add new variable
        if not hasattr( self.M, output_var_name):
            setattr( self.M, output_var_name, pyo.Var(time_dim, model_dims , within=pyo.Reals))
            residual_var = getattr( self.M, output_var_name)
        else:
            raise ValueError('Attempting to overwrite variable')
        
        # Add constraints and bounds
        for n in time_dim:
            for d in model_dims :
                self.M.residual_constraints.add(expr= residual_var[n,d] == input_1[n,d] + input_2[n,d])
                try:
                    if self.bound_cut_activation["RES_var"]: 
                        residual_var[n,d].ub == input_1[n,d].ub + input_2[n,d].ub
                        residual_var[n,d].lb == input_1[n,d].lb + input_2[n,d].lb
                except:
                    continue
                
        return residual_var
    
    
    def add_FFN_2D(self, input_var_name:Union[pyo.Var,str], output_var_name, nn_name, input_shape, model_parameters, bounds = (-2, 2), formulation=ReluBigMFormulation):
        """
        Adds a 2D feed-forward neural network (FFN) using OMLT which works with Pyomo models.

        This method integrates a trained FFN into the Pyomo model, enabling it to
        interact with optimization problems. The FFN's parameters are defined using
        OMLT formulations.

        Args:
            input_var_name (Union[pyo.Var, str]): The name of or reference to the input Pyomo variable.
            output_var_name (str): The name of the output Pyomo variable.
            nn_name (str): Name of the neural network.
            input_shape (tuple): Shape of the input layer for the FFN.
            model_parameters (dict): Weights and biases of the FFN.
            bounds (tuple, optional): Bounds for the input variables. Defaults to (-2, 2).
            formulation: OMLT formulation class for the FFN. Defaults to ReluBigMFormulation.

        Returns:
            pyo.Var: The Pyomo variable for the FFN output.
        """
        
        # Get input var
        if not isinstance(input_var_name, pyo.Var):
            input_var = getattr(self.M, input_var_name)
        else:
            input_var = input_var_name

        # Add new variable
        if not hasattr( self.M, output_var_name + "_NN_Block"):
            NN_name = output_var_name + "_NN_Block"
            setattr( self.M, NN_name, OmltBlock())
            NN_block = getattr( self.M, NN_name)
            
            setattr( self.M, output_var_name, pyo.Var(input_var.index_set(), within=pyo.Reals))
            output_var = getattr( self.M, output_var_name)
            
            setattr( self.M, output_var_name+"_constraints", pyo.ConstraintList())
            ffn_constraints = getattr( self.M, output_var_name+"_constraints")
        else:
            raise ValueError('Attempting to overwrite variable')

        # Get input indices
        input_indices_len, input_indices_attr = self.__get_indices( input_var)
        
        # Create dict of input bounds
        input_bounds={}
        for dim in input_indices_attr[1]:
            input_bounds[dim] = bounds
        
        # Get Network Definition of FFN
        net_relu = helpers.OMLT_helper.weights_to_NetDef(output_var_name, nn_name, input_shape, model_parameters, input_bounds)
        
        # Build FFN of type "formlation"
        NN_block.build_formulation(formulation(net_relu))
        
        # Set input constraints
        if input_indices_len == 1:
            for i, index in  enumerate(input_indices_attr[0]):
                ffn_constraints.add(expr= input_var[index] == NN_block.inputs[i])
        elif input_indices_len == 2:
            for i, i_index in  enumerate(input_indices_attr[0]):
                for j, j_index in  enumerate(input_indices_attr[1]):
                    ffn_constraints.add(expr= input_var[i_index, j_index] == NN_block.inputs[j])
                    
                    
        # Set output constraints
        output_indices_len, output_indices_attr = self.__get_indices( output_var)
        if output_indices_len == 1:
            for i, index in  enumerate(output_indices_attr[0]):
                ffn_constraints.add(expr= output_var[index] == NN_block.outputs[i])
        elif output_indices_len == 2:
            for i, i_index in  enumerate(output_indices_attr[0]):
                for j, j_index in  enumerate(output_indices_attr[1]):
                    ffn_constraints.add(expr= output_var[i_index, j_index] == NN_block.outputs[j])
                      
                
        return output_var
    
    def get_fnn(self, input_var_name:Union[pyo.Var,str], output_var_name, nn_name, input_shape, model_parameters):
        """ 
        Helper function to define a GurobiML feed-forward neural network (FFN).

        This method prepares a trained FFN to be create a GurobiML FFN. The outputs of this function are used to create
        a GurobiML FFN after the Pyomo model has been converted to a Gurobipy model. 

        Args:
            input_var_name (Union[pyo.Var, str]): The name of or reference to the input Pyomo variable.
            output_var_name (str): The name of the output Pyomo variable.
            nn_name (str): Name of the neural network.
            input_shape (tuple): Shape of the input layer for the FFN.
            model_parameters (dict): Weights and biases of the FFN.

        Returns:
            tuple:
                - NetDef: The FFN's network definition.
                - pyo.Var: The input Pyomo variable.
                - pyo.Var: The output Pyomo variable.
        """
        # Get input var
        if not isinstance(input_var_name, pyo.Var):
            input_var = getattr(self.M, input_var_name)
        else:
            input_var = input_var_name
            
        # Determine indices of input
        if input_var.is_indexed():
            set_var = input_var.index_set()
        else:
            raise ValueError('Input value must be indexed (time, model_dim)')
        
        # Add new variable
        if not hasattr( self.M, output_var_name + "_NN_Block"):
            
            setattr( self.M, output_var_name, pyo.Var(set_var, within=pyo.Reals))
            output_var = getattr( self.M, output_var_name)
            
            setattr( self.M, output_var_name+"_constraints", pyo.ConstraintList())
            ffn_constraints = getattr( self.M, output_var_name+"_constraints")
        else:
            raise ValueError('Attempting to overwrite variable')
        
        # Get Netowrk Definition of FFN (needed to create GurobiML FFN)
        nn= GUROBI_ML_helper.weights_to_NetDef(output_var_name, nn_name, input_shape, model_parameters)
       
        return nn, input_var, output_var
            
        
    def add_avg_pool(self, input_var_name:Union[pyo.Var,str], output_var_name):
        """
        Adds an average pooling layer over the time dimension.

        This method computes the average of a Pyomo variable over the sequence dimension
        and creates a new variable for the pooled output.

        Args:
            input_var_name (Union[pyo.Var, str]): The name of or reference to the input Pyomo variable.
            output_var_name (str): The name of the output Pyomo variable.

        Returns:
            pyo.Var: The Pyomo variable for the average pooled output.
        """
        # Get input
        if not isinstance(input_var_name, pyo.Var):
            input_var = getattr(self.M, input_var_name)
        else:
            input_var = input_var_name
        
        # Determine indices of input
        if input_var.is_indexed():
            set_var = input_var.index_set()
            indices = []
            for set in str(set_var).split("*"):
                indices.append( getattr( self.M, set) )
                
            time_dim = indices[0]
            model_dims = indices[1]
        else:
            raise ValueError('Input value must be indexed (time, model_dim)')

        # Add new variable
        if not hasattr( self.M, output_var_name):
            setattr( self.M, "avg_pool_constr_"+output_var_name, pyo.ConstraintList())
            constraints = getattr( self.M, "avg_pool_constr_"+output_var_name) 
            
            setattr( self.M, output_var_name, pyo.Var(model_dims , within=pyo.Reals))
            output_var = getattr( self.M, output_var_name)
        else:
            raise ValueError('Attempting to overwrite variable')

        # Compute average over sequence dimension
        for d in model_dims : 
            constraints.add(expr= output_var[d] * self.N == sum(input_var[t,d] for t in time_dim))
            
            try:
                if self.bound_cut_activation["AVG_POOL_var"]:
                    output_var[d].ub  == sum(input_var[t,d].ub for t in time_dim) / self.N
                    output_var[d].lb  == sum(input_var[t,d].lb for t in time_dim) / self.N
            except:
                continue
            
        return output_var
    
    def __McCormick_bb(self, w, x, y):
        """ 
        Adds McCormick envelope constraints for a bilinear variable.

        This method approximates the bilinear product `w = x * y` using McCormick
        envelopes.

        Args:
            w (pyo.Var): The Pyomo variable representing the bilinear product.
            x (pyo.Var): The first input Pyomo variable.
            y (pyo.Var): The second input Pyomo variable.

        Returns:
            None
        """
        
        # Create constraint list for layer
        if not hasattr( self.M, "mccormick_bb_constr_list"):
            setattr( self.M, "mccormick_bb_constr_list", pyo.ConstraintList())
        constraints = getattr( self.M, "mccormick_bb_constr_list")   
        
        # Add cut constraints
        constraints.add( expr= w >= (x.lb * y) + (x * y.lb) - (x.lb * y.lb))
        constraints.add( expr= w <= (x.ub * y) + (x * y.lb) - (x.ub * y.lb))
        constraints.add( expr= w <= (x * y.ub) + (x.lb * y) - (x.lb * y.ub))
        constraints.add( expr= w >= (x.ub * y) + (x * y.ub) - (x.ub * y.ub))

    def __get_indices(self, input_var):
        """
        Extracts the indices of a Pyomo variable indexed by a set.

        This method retrieves the length and attributes of the indices used
        for the provided Pyomo variable.

        Args:
            input_var (pyo.Var): The input Pyomo variable to extract indices from.

        Returns:
            tuple:
                - int: Length of the indices.
                - list: List of attributes corresponding to the Pyomo sets.
        """
        indices = str(input_var.index_set()).split('*')
        indices_len = len(indices)
        indices_attr = []
        for i in indices:
            try: 
                indices_attr += [getattr(self.M, i)]
            except:
                raise ValueError('Input variable not indexed by a pyomo Set')
        
        return indices_len, indices_attr