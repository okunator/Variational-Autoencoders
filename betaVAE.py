import torch
import torch.nn as nn
import torch.nn.functional as F

class betaVAE(nn.Module) :
    """
    [9] Irina Higgins, Loic Matthey, Arka Pal, Christopher Burgess, Xavier Glorot, Matthew Botvinick, Shakir Mohamed, Alexander Lerchner  (2016)
    **beta-VAE: Learning Basic Visual Concepts with a Constrained Variational Framework**
    ICLR 2017 conference submission
    """
    def __init__(self, latent_size) :
        super(betaVAE, self).__init__()

        #Encoder
        self.fc1 = nn.Linear(13714, 800)
        self.fc2 = nn.Linear(800, 400)
        self.fc31 = nn.Linear(400, latent_size)
        self.fc32 = nn.Linear(400, latent_size)

        #Decoder
        self.fc4m = nn.Linear(latent_size, 80)
        self.fc3m = nn.Linear(80, 400)
        self.fc2m = nn.Linear(400, 800)
        self.fc1m = nn.Linear(800, 13714)

    def encode(self, x):
        h1 = self.fc1(x)
        h2 = F.relu(self.fc2(h1))
        mu = self.fc31(h2)
        log_variance = self.fc32(h2)
        return mu, log_variance

    # Reparametrization trick
    def reparameterize(self, mu, log_variance):
        std = torch.exp(0.5*log_variance)
        epsilon = torch.randn_like(std)
        return mu + epsilon*std #sample z from mu & log(sigma^2)

    def decode(self, z):
        h5 = F.relu(self.fc4m(z))
        h6 = F.relu(self.fc3m(h5))
        h7 = F.relu(self.fc2m(h6))
        x_hat = self.fc1m(h7)

        return x_hat

    def forward(self, x) :
        mu, log_var = self.encode(x)
        z = self.reparameterize(mu, log_var)
        x_hat = self.decode(z)
        return x_hat, mu, log_var

#ELBO loss
def ELBO_loss(x_hat, x, mu, log_variance, beta):
    MSE = nn.MSELoss(reduction='sum')
    BCE = nn.BCELoss(reduction='sum')
    loss = MSE(x_hat, x.view(-1, 13714))
    KLD = -0.5*torch.sum(1 + log_variance - mu.pow(2) - log_variance.exp())
    ELBO = loss + beta*KLD
    return ELBO

# HELPER FUNCTION TO TRAIN DIFFERENT beta-VAE models
def train_betaVAE(beta, betaVae, train_loader, device, optimizer):
    print_every=75
    print('Beta value: ', beta)
    for epoch in range(200):
        running_loss = 0.0
        for i, (inputs, labels) in enumerate(train_loader, 0):
            inputs = inputs.to(device)
            optimizer.zero_grad()
            x_hat, mu, logvar = betaVae(inputs.float())
            loss = ELBO_loss(x_hat, inputs.float(), mu, logvar, beta=beta)
            loss.backward()
            optimizer.step()

            # print statistics
            running_loss += loss.item()
            if i % print_every == 0:    # print every 15 mini-batches
                print('loss: ', loss.item(), 'epoch: ', epoch)

    print('Finished Training')
