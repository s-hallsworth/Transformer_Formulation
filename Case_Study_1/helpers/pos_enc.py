import torch
import math 

def pos_enc(d_model, max_len):
    #max len = enc seq len + dec seq_len - overlap enc dec (total time steps)
    
    position = torch.arange(max_len).unsqueeze(1)
    div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
    pe = torch.zeros(max_len, 1, d_model)
    e_count = 0
    o_count = 0
    for d in range(d_model):
        if d%2 == 0:
            pe[:, 0, d] = torch.sin(position * div_term)[:, e_count]
            e_count += 1
        
        else:
            pe[:, 0, d] = torch.cos(position * div_term)[:, o_count]
            o_count += 1

    return pe.permute(1,0,2) #shape: batch, length of time, model dims

# x = torch.ones(1,1,9)
# pe = pos_enc(x.size(-1), 8)
# y = 3 * torch.ones(1,8,9)
# print(pe.shape)
# print(x)
# print(pe)
# x = x + pe[:, :x.size(1), :]
# print(x)

# print( y + pe[:, :y.size(1), :])