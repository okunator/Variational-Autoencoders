import torch
import torch.nn as nn
import torch.nn.functional as F

class Autoencoder(nn.Module):
    """
    Straight forward implementation of a deep Autoencoder first introduced by Hopfield (1982)
    """
    def __init__(self, latent_size):
        super(Autoencoder, self).__init__()

        #Encoder
        self.fc1 = nn.Linear(13714, 128)
        self.fc2 = nn.Linear(128, 32)
        self.fc3 = nn.Linear(32, 16)
        self.fc4 = nn.Linear(16, latent_size)

        #Decoder
        self.fc5 = nn.Linear(latent_size, 16)
        self.fc6 = nn.Linear(16, 32)
        self.fc7 = nn.Linear(32, 128)
        self.fc8 = nn.Linear(128, 13714)

    def encode(self, x):
        h1 = self.fc1(x)
        h2 = F.relu(self.fc2(h1))
        h3 = F.relu(self.fc3(h2))
        z = self.fc4(h3)
        return z

    def decode(self, z):
        h5 = F.relu(self.fc5(z))
        h6 = F.relu(self.fc6(h5))
        h7 = F.relu(self.fc7(h6))
        x_hat = self.fc8(h7)

        return x_hat

    def forward(self, x):
        z = self.encode(x)
        x_hat = self.decode(z)
        return x_hat

def MSE_loss(x_hat, x):
    MSE = nn.MSELoss(reduction='sum')
    MSE_loss = MSE(x_hat, x.view(-1, 13714))
    return MSE_loss
