import keras
import os
import json


# variable declaration
os.environ["TF_ENABLE_ONEDNN_OPTS"] = '0' # turn off floating-point round-off
model_path = "..\\Transformer_Toy\\transformer_TOY.keras"

# load pre-trained model
model = keras.models.load_model(model_path)

# print model summary
# print("--- Model Summary ---")
# model.summary()

# extract weights
model_weights = {}

for layer in model.layers:
    weights = layer.get_weights()
    if weights:  
        model_weights[layer.name] = [w.tolist() for w in weights]  

# save weights    
# file_path = 'model_weights.json'
# with open(file_path, 'w') as json_file:
#     json.dump(model_weights, json_file)
    
# print(f"Weights of the model have been saved to {file_path}")

# model.save_weights('W_model.weights.h5', overwrite=True)


# # read H5
# import h5py
 
# #Open the H5 file in read mode
# with h5py.File('W_model.weights.h5', 'r') as file:
#     print("Keys: %s" % file.keys())
#     layers = list(file.keys())[0]
     
#     # Getting the data
#     data = list(file[layers])
#     print("Layers: ",data)