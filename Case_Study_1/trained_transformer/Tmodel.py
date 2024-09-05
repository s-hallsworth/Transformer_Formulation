import torch

class TransformerModel(torch.nn.Module):
    def __init__(self, input_dim, output_dim, d_model=64, nhead=4, num_encoder_layers=3, num_decoder_layers=3):
        super(TransformerModel, self).__init__()
        self.model_type = 'Transformer'
        self.linear1 = torch.nn.Linear(input_dim, d_model)
        self.transformer = torch.nn.Transformer(d_model=d_model, nhead=nhead,
                                          num_encoder_layers=num_encoder_layers,
                                          num_decoder_layers=num_decoder_layers, batch_first=True)
        self.linear2 = torch.nn.Linear(d_model, output_dim)

    def forward(self, src, tgt, sequence_size):
        src = self.linear1(src)
        tgt = self.linear1(tgt)
        
        src_mask = self.transformer.generate_square_subsequent_mask(sequence_size)
        output = self.transformer(src, tgt, src_mask)
        output = self.linear2(output)
        return output