import torch
import torch.nn as nn

class ContentAE(nn.Module):
    def __init__(self, input_dim=8, hidden_dims=[32,16,8], bottleneck=4):
        super().__init__()
        enc_layers = []
        last = input_dim
        for h in hidden_dims:
            enc_layers += [nn.Linear(last, h), nn.ReLU()]
            last = h
        enc_layers += [nn.Linear(last, bottleneck), nn.ReLU()]
        self.encoder = nn.Sequential(*enc_layers)

        dec_layers = []
        last = bottleneck
        for h in reversed(hidden_dims):
            dec_layers += [nn.Linear(last, h), nn.ReLU()]
            last = h
        dec_layers += [nn.Linear(last, input_dim)]
        self.decoder = nn.Sequential(*dec_layers)

    def forward(self, x):
        z = self.encoder(x)
        out = self.decoder(z)
        return out
