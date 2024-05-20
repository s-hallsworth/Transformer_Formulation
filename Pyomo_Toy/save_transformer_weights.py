import keras
import os
import json
import extract_from_pretrained


# variable declaration
os.environ["TF_ENABLE_ONEDNN_OPTS"] = '0' # turn off floating-point round-off
model_path = "..\\Transformer_Toy\\transformer_small_copy_TOY.keras"

# save moel weigths
extract_from_pretrained.save_weights_json(model_path, '.\\data\\weights_save.json')


# # read H5
# import h5py
 
# #Open the H5 file in read mode
# with h5py.File('W_model.weights.h5', 'r') as file:
#     print("Keys: %s" % file.keys())
#     layers = list(file.keys())[0]
     
#     # Getting the data
#     data = list(file[layers])
#     print("Layers: ",data)