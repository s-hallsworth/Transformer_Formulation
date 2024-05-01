import numpy as np
import torch


def embed(x, W):
    """
    Embed an input vector x in a feature space with dimensions=d_model.

    x: input vector with size (d_x, 1)
    W: learned weight matrix with size (1, d_model)
    embedding: x * (W/sqrt(d_model))
    """
    return torch.matmul(x, W / np.sqrt(W.shape[-1]))


def encode_pos(x):
    """
    Encode position of object in input sequence using sin and cos

    x: input sequence with size (d_x, d_model)

    ------- Additional variables ----------
    d_x: number of objects in input sequence
    d_model: embedding dimension
    """
    PE = torch.zeros(x.shape)

    # for each position in input sequence
    for pos in range(1, PE.shape[0]):
        # for each embedding dimension
        for i in range(1, PE.shape[1]):
            PE[pos, 2 * i] = np.sin(pos / (10000 ^ (2 * i / PE.shape[1])))
            PE[pos, (2 * i) + 1] = np.cos(pos / (10000 ^ (2 * i / PE.shape[1])))

    return PE


def multi_head_attention(x, H, W_QKV, mask=False):
    """
    Multi-head attention is used to represent the input sequence in a contextually rich way.
    It focuses on different parts of the input sequence and finds the compatibility between each pair of objects in the sequence.

    x: input sequence with size (d_x, d_model)
    W_QKV: learned weights for query, key, value at head, h (h, 3, d_model, d_k)

    ------- Additional variables ----------
    Q: query - rep. of an object that is current object of focus during attention
    K: key - rep. of all objects in the sequence against which the query is scored
    V: value - rep. of actual content of all objects in the sequence
    H: number of attention heads
    d_k: d_model/H
    """
    X = x.expand(H, 3, x.shape[0], x.shape[1])
    # QKV = torch.matmul(torch.transpose(W_QKV, 2, 3) , X)
    return X
