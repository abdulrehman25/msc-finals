import torch
import torch.nn as nn

class SessionLSTMVAE(nn.Module):
    def __init__(self, input_dim=8, lstm_hidden=32, latent_dim=8, num_layers=1):
        super().__init__()
        self.encoder = nn.LSTM(input_dim, lstm_hidden, num_layers=num_layers, batch_first=True)
        self.mu = nn.Linear(lstm_hidden, latent_dim)
        self.logvar = nn.Linear(lstm_hidden, latent_dim)
        self.decoder = nn.LSTM(latent_dim, lstm_hidden, num_layers=num_layers, batch_first=True)
        self.out = nn.Linear(lstm_hidden, input_dim)

    def reparam(self, mu, logvar):
        std = torch.exp(0.5*logvar)
        eps = torch.randn_like(std)
        return mu + eps*std

    def forward(self, x):
        enc_out, _ = self.encoder(x)
        h_last = enc_out[:,-1,:]
        mu = self.mu(h_last)
        logvar = self.logvar(h_last)
        z = self.reparam(mu, logvar).unsqueeze(1).repeat(1, x.size(1), 1)
        dec_out, _ = self.decoder(z)
        recon = self.out(dec_out)
        return recon, mu, logvar

def vae_loss(recon, x, mu, logvar):
    recon_loss = nn.functional.mse_loss(recon, x, reduction="mean")
    kld = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
    return recon_loss + 0.001*kld, recon_loss, kld
