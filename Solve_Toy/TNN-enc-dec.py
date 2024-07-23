# %% [markdown]
# <a href="https://colab.research.google.com/github/jeffheaton/t81_558_deep_learning/blob/master/t81_558_class_10_5_keras_transformers.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>
# 

# %% [markdown]
# Adapted from: https://colab.research.google.com/github/jeffheaton/t81_558_deep_learning/blob/master/t81_558_class_10_5_keras_transformers.ipynb

# %%
import pandas as pd
import numpy as np
import os

def to_sequences(SEQUENCE_SIZE, obs_x, obs_u, obs_t):

    x = []
    y = []
    for i in range((len(obs_x))-SEQUENCE_SIZE):

        window1 = obs_x[i:(i+SEQUENCE_SIZE)]  #[i:(i+SEQUENCE_SIZE)] 
        window2 = obs_u[i:(i+SEQUENCE_SIZE)] 
        after_window = [obs_x[i+SEQUENCE_SIZE]] 
        window = [[x, t] for x,t in zip(window1, window2)]
        #print("{} - {}".format(window,after_window))
        x.append(window)
        y.append(after_window)

    return np.array(x),np.array(y)

# Prepare Data
names = ['time', 'u' , 'x', 'obs_num']

path = r"C:\Users\sian_\OneDrive\Documents\Thesis\MILP_Formulation\Optimal_Control_Toy"
df = pd.read_csv("data_N200.csv",sep=',', header=0, names=names,index_col=False)
df.replace([np.inf, -np.inf], np.nan, inplace=True)
df.bfill(inplace=True)
df.ffill(inplace=True) #for obs num


print("Starting file:")
print(df[0:10])

print("Ending file:")
print(df[-10:])

print(df.shape)

# Prepare Data
df['u'] = df['u'].astype(float)
df12 = df[df['obs_num']<3.0]
df123 = df[df['obs_num']<4.0]

df_train = df123[df123['time']<0.7]
df_test = df123[df123['time']>=0.7]

spots_train_u = df_train['u'].tolist()
spots_train_x = df_train['x'].tolist()
spots_train_t = df_train['time'].tolist()

spots_test_u = df_test['u'].tolist()
spots_test_x = df_test['x'].tolist()
spots_test_t = df_test['time'].tolist()

print("Training set has {} observations.".format(len(spots_train_u)))
print("Test set has {} observations.".format(len(spots_test_u)))
# print(spots_train)

# %%

df1 = df[df['obs_num']==1.0]
df2 = df[df['obs_num']==2.0]
df3 = df[df['obs_num']==3.0]

df_test1 = df1[df1['time']>=0.7]
df_test2 = df2[df2['time']>=0.7]
df_test3 = df3[df3['time']>=0.7]

spots_test1_u = df_test1['u'].tolist()
spots_test1_x = df_test1['x'].tolist()
spots_test1_t = df_test1['time'].tolist()

spots_test2_x = df_test2['x'].tolist()
spots_test2_u = df_test2['u'].tolist()
spots_test2_t = df_test2['time'].tolist()

spots_test3_x = df_test3['x'].tolist()
spots_test3_u = df_test3['u'].tolist()
spots_test3_t = df_test3['time'].tolist()

print("Test set 1 has {} observations.".format(len(spots_test1_u)))
print("Test set 2 has {} observations.".format(len(spots_test2_u)))
print("Test set 3 has {} observations.".format(len(spots_test3_u)))



# creating training sequences
SEQUENCE_SIZE = 4
x_train,y_train = to_sequences(SEQUENCE_SIZE,spots_train_x, spots_train_u, spots_train_t)
x_test,y_test = to_sequences(SEQUENCE_SIZE,spots_test_x, spots_test_u, spots_test_t)

# print("Shape of x train set: {}".format(x_train.shape))
# print("Shape of x test set: {}".format(x_test.shape))

# print("Shape of y train set: {}".format(y_train.shape))
# print("Shape of y test set: {}".format(y_test.shape))

# %%

x_test1, y_test1 = to_sequences(SEQUENCE_SIZE,spots_test1_x, spots_test1_u ,  spots_test1_t)
# print("Shape of y1 test set: {}".format(y_test1.shape))

x_test2, y_test2 = to_sequences(SEQUENCE_SIZE,spots_test2_x, spots_test2_u,  spots_test1_t)
# print("Shape of y2 test set: {}".format(y_test2.shape))

x_test3, y_test3 = to_sequences(SEQUENCE_SIZE,spots_test3_x, spots_test3_u,  spots_test1_t)
# print("Shape of y3 test set: {}".format(y_test3.shape))


# %%
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

def transformer_encoder(inputs, head_size, num_heads, ff_dim, dropout=0):
    # Normalization and Attention
    x = layers.LayerNormalization(epsilon=1e-6)(inputs)
    x = layers.MultiHeadAttention(
        key_dim=head_size, num_heads=num_heads, dropout=dropout
    )(x, x)
    x = layers.Dropout(dropout)(x)
    res = x + inputs

    # Feed Forward Part
    ## NB dropout layer is only applied during training not inference (https://keras.io/api/layers/regularization_layers/dropout/)
    
    x = layers.LayerNormalization(epsilon=1e-6)(res)
    # x = layers.Conv1D(filters=ff_dim, kernel_size=1, activation="relu")(x)
    # x = layers.Dropout(dropout)(x)
    # x = layers.Conv1D(filters=inputs.shape[-1], kernel_size=1)(x)

    x = layers.Dense(ff_dim, activation="relu")(x)
    x = layers.Dropout(dropout)(x)
    x = layers.Dense(inputs.shape[-1], activation="relu")(x)
    
    #x = layers.Conv2D(filters=ff_dim, kernel_size=1, activation="relu")(x)
    #x = layers.Dropout(dropout)(x)
    #x = layers.Conv2D(filters=inputs.shape[-1], kernel_size=1)(x)

    return x + res

# %%
def transformer_decoder(inputs, encoder_outputs, head_size, num_heads, ff_dim, dropout=0):
    # Masked multi-head self-attention
    x = layers.LayerNormalization(epsilon=1e-6)(inputs)
    attention_output = layers.MultiHeadAttention(
        key_dim=head_size, num_heads=num_heads, dropout=dropout
    )(x, x, attention_mask=create_look_ahead_mask(x.shape[1]))
    x = layers.Dropout(dropout)(attention_output)
    res = x + inputs

    # Cross-attention with encoder outputs
    x = layers.LayerNormalization(epsilon=1e-6)(res)
    attention_output = layers.MultiHeadAttention(
        key_dim=head_size, num_heads=num_heads, dropout=dropout
    )(x, encoder_outputs)
    x = layers.Dropout(dropout)(attention_output)
    res = x + res

    # Feed Forward Part
    x = layers.LayerNormalization(epsilon=1e-6)(res)
    x = layers.Dense(ff_dim, activation="relu")(x)
    x = layers.Dropout(dropout)(x)
    x = layers.Dense(inputs.shape[-1])(x)
    
    return x + res

def create_look_ahead_mask(size):
    mask = tf.linalg.band_part(tf.ones((size, size)), -1, 0) #1 - tf.linalg.band_part(tf.ones((size, size)), -1, 0)
    print(mask)
    return mask

# %% [markdown]
# The following function is provided to build the model, including the attention layer.

# %%


def build_model(
    input_shape,
    head_size,
    num_heads,
    ff_dim,
    num_transformer_blocks,
    mlp_units,
    dropout=0.25,
    mlp_dropout=0.25,
):
   

    inputs = keras.Input(shape=input_shape)
    x = inputs
    x = layers.Dense(10)(inputs)
    for _ in range(num_transformer_blocks):
        x = transformer_encoder(x, head_size, num_heads, ff_dim, dropout)
    encoder_outputs = x
    
    # Decoder
    decoder_inputs = keras.Input(shape=input_shape)
    x = decoder_inputs
    x = layers.Dense(10)(decoder_inputs)
    for _ in range(num_transformer_blocks):
        x = transformer_decoder(x, encoder_outputs, head_size, num_heads, ff_dim, dropout)
    decoder_outputs = x
    
    x = layers.GlobalAveragePooling1D(data_format="channels_last")(x)
    #x = layers.GlobalAveragePooling2D(data_format="channels_first")(x)
    print(x.shape, x)
    for dim in mlp_units:
        x = layers.Dense(dim, activation="relu")(x)
        x = layers.Dropout(mlp_dropout)(x)
    outputs = layers.Dense(1, activation="relu")(x)
    return keras.Model(inputs= [inputs, decoder_inputs], outputs = outputs)

# %%
# model parameters:
Set_model_params = {
    'input_shape': [(SEQUENCE_SIZE,2)],
    'head_size':  [1, 2, 4, 8],
    'num_heads': [5, 10, 15, 20],
    'ff_dim': [8, 16, 24, 32], #32
    'num_transformer_blocks': [1],
    'mlp_units': [[12], [24], [48], [64], [128]], #24
    'mlp_dropout': [0, 0.05, 0.1, 0.25],
    'dropout': [0, 0.05, 0.1, 0.25],
    'validation_split': [0.2],
    'epochs': [1000],
    'batch_size': [16,32,48,64],
    'learning_rate': [1e-7]
}

model_params = {
    'input_shape': (SEQUENCE_SIZE,2),
    'head_size': 2,
    'num_heads': 5,
    'ff_dim': 8, #32
    'num_transformer_blocks': 1,
    'mlp_units': [12], #24
    'mlp_dropout': 0,
    'dropout': 0.25,
    'validation_split': 0.2,
    'epochs':1000,
    'batch_size': 32,
    'learning_rate': 1e-7
}

# Build and train the model.

def run_model(x_train, y_train, x_test, y_test, x_test1, y_test1, x_test2, y_test2,x_test3, y_test3, save = False):
    model = build_model(
        input_shape = model_params["input_shape"],
        head_size = model_params["head_size"],
        num_heads = model_params["num_heads"],
        ff_dim = model_params["ff_dim"],
        num_transformer_blocks = model_params['num_transformer_blocks'],
        mlp_units = model_params["mlp_units"],
        mlp_dropout = model_params["mlp_dropout"],
        dropout = model_params["dropout"],
    )

    model.compile(
        loss="mean_squared_error",
        optimizer=keras.optimizers.Adam(learning_rate=model_params['learning_rate'])
    )
    model.summary()

    callbacks = [keras.callbacks.EarlyStopping(patience=2, \
        restore_best_weights=True)]

    model.fit(
        [x_train, x_train],
        y_train,
        validation_split = model_params["validation_split"],
        epochs = model_params["epochs"],
        batch_size = model_params["batch_size"],
        callbacks=callbacks,
        shuffle=True,
    )

    model.evaluate([x_test, x_test], y_test, verbose=0)

    # # save model
    unq = "1"
    name = 'TNN_enc_dec_'+unq+'.keras'
    if save:
        model.save(name , overwrite=True)


    from sklearn import metrics

    preds_x = []
    rmse_x = []
    preds_u = []
    rmse_u = []

    for x,y in [[x_test1, y_test1],[x_test2, y_test2],[x_test3, y_test3]]:
        pred = model.predict([x,x])

        score_x = np.sqrt(metrics.mean_squared_error(pred[:,0],y[:,0]))
        print("X Score (RMSE): {}".format(score_x))
        preds_x.append(pred[:,0])
        rmse_x.append(score_x)
        try:
            score_u = np.sqrt(metrics.mean_squared_error(pred[:,1],y[:,1]))
            print("U Score (RMSE): {}".format(score_u))
            preds_u.append(pred[:,1])
            rmse_u.append(score_u)
        except:
            continue

    print('Prediction shape: ', len(preds_x))

    # Save results
    import csv

    new_filename = "TNN_enc-dec.csv"

    # Create and write to the new CSV file with model parameters and results
    with open(new_filename, 'a', newline='') as file:
        writer = csv.writer(file)
        # Writing headers for model parameters
        headers = list(model_params.keys())
        headers.append('rmse_x_1')
        headers.append('rmse_x_2')
        headers.append('rmse_x_3')

        headers.append(name)
        writer.writerow(headers)
        
        # Writing values
        values = list(model_params.values())
        values.append(rmse_x[0])  # Adding first RMSE
        values.append(rmse_x[1])  # Adding second RMSE
        values.append(rmse_x[2])  
        # values.append(rmse_u[0])  # Adding first RMSE
        # values.append(rmse_u[1])  # Adding second RMSE
        # values.append(rmse_u[2])
        values.append('')
        writer.writerow(values)


## RUN WITH COMBINATION OF PARAMS
# import itertools
# import pprint

# keys, values = zip(*Set_model_params.items())
# combinations = [dict(zip(keys, v)) for v in itertools.product(*values)]
# for comb in combinations:
#     model_params = comb
#     run_model(x_train, y_train, x_test, y_test, x_test1, y_test1, x_test2, y_test2,x_test3, y_test3)
#     print("-----------------------------")

# RUN WITH SET PARAMS
run_model(x_train, y_train, x_test, y_test, x_test1, y_test1, x_test2, y_test2,x_test3, y_test3, save = True)
print("-----------------------------")
    
# %%
# import matplotlib.pyplot as plt
# func = 0
# plt.figure(figsize=(6, 4))
# plt.plot(y_test1[:,0], 's-',color='C1', label=f'x actual {func+1}')
# plt.plot(preds_x[func], '-',color='C2', label=f'x pred {func+1}')

# try:
#     plt.plot(y_test1[:,1], 's-',color='C0', label=f'u actual {func+1}')
#     plt.plot(preds_u[func], '-',color='C3', label=f'u pred {func+1}')
# except:
#     pass
# plt.legend()

# # %%
# import matplotlib.pyplot as plt
# func = 1
# plt.figure(figsize=(6, 4))
# plt.plot(y_test2[:,0], 's-',color='C1', label=f'x actual {func+1}')
# plt.plot(preds_x[func], '-',color='C2', label=f'x pred {func+1}')

# try:
#     plt.plot(y_test2[:,1], 's-',color='C0', label=f'u actual {func+1}')
#     plt.plot(preds_u[func], '-',color='C3', label=f'u pred {func+1}')
# except:
#     pass
# plt.legend()

# # %%
# import matplotlib.pyplot as plt
# func = 2
# plt.figure(figsize=(6, 4))
# plt.plot(y_test3[:,0], 's-',color='C1', label=f'x actual {func+1}')
# plt.plot(preds_x[func], '-',color='C2', label=f'x pred {func+1}')

# try:
#     plt.plot(y_test3[:,1], 's-',color='C0', label=f'u actual {func+1}')
#     plt.plot(preds_u[func], '-',color='C3', label=f'u pred {func+1}')
# except:
#     pass
# plt.legend()


