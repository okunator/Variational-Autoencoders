import torch
import torch.nn as nn
import torch.nn.functional as F

class betaVSC(nn.Module):
    """
    [10] Francesco Tonolini, Bjorn Sand Jensen, Roderick Murray-Smith (2019)
    Variational sparse coding
    ICLR 2019 Conference Blind Submission
    """

    def __init__(self, latent_size):
        super(betaVSC, self).__init__()
        self.c = 50.0

        #Encoder
        self.fc1 = nn.Linear(13714, 800)
        self.fc2 = nn.Linear(800, 400)
        self.fc31 = nn.Linear(400, latent_size)
        self.fc32 = nn.Linear(400, latent_size)
        self.fc33 = nn.Linear(400, latent_size)

        #Decoder
        self.fc4m = nn.Linear(latent_size, 80)
        self.fc3m = nn.Linear(80, 400)
        self.fc2m = nn.Linear(400, 800)
        self.fc1m = nn.Linear(800, 13714)

    def encode(self, x):
        h0 = F.relu(self.fc1(x))
        h1 = F.relu(self.fc2(h0))
        mu = self.fc31(h1)
        log_variance = self.fc32(h1)
        log_spike = -F.relu(-self.fc33(h1))

        return mu, log_variance, log_spike

    # Reparametrization trick
    def reparameterize(self, mu, log_variance, log_spike):
        std = torch.exp(0.5*log_variance)
        epsilon = torch.randn_like(std)
        gaussian = epsilon.mul(std).add_(mu)
        eta = torch.rand_like(std)
        selection = torch.sigmoid(self.c*(eta + log_spike.exp() - 1))
        return selection.mul(gaussian)

    def decode(self, z):
        h2 = F.relu(self.fc4m(z))
        h3 = F.relu(self.fc3m(h2))
        h4 = F.relu(self.fc2m(h3))
        x_hat = self.fc1m(h4)

        return x_hat

    def forward(self, x) :
        mu, log_variance, log_spike = self.encode(x)
        z = self.reparameterize(mu, log_variance, log_spike)
        x_hat = self.decode(z)
        return x_hat, mu, log_variance, log_spike

# VSC-ELBO
def ELBO_loss(x_hat, x, mu, log_variance, beta, log_spike):
    alpha=0.5
    MSE = nn.MSELoss(reduction='sum')
    MSE_loss = MSE(x_hat, x.view(-1, 13714))
    spike = torch.clamp(log_spike.exp(), 1e-6, 1.0 - 1e-6)
    KLD = -0.5*torch.sum(spike.mul(1 + log_variance - mu.pow(2) - log_variance.exp())) + \
                         torch.sum((1 - spike).mul(torch.log((1 - spike)/(1 - alpha))) + \
                         spike.mul(torch.log(spike/alpha)))
    ELBO = MSE_loss + beta*KLD
    return ELBO

#TRAIN FUNCTION
def train_VSC(beta, VSC, train_loader, device, optimizer):
    print_every=75
    for epoch in range(200):
        running_loss = 0.0
        for i, (inputs, labels) in enumerate(train_loader, 0):
            inputs = inputs.to(device)
            optimizer.zero_grad()
            x_hat, mu, logvar, log_spike = VSC(inputs.float())
            loss = ELBO_loss(x_hat, inputs.float(), mu, logvar, beta=1, log_spike=log_spike)
            loss.backward()
            optimizer.step()

            # print statistics
            running_loss += loss.item()
            if i % print_every == 0:    # print every 15 mini-batches
                print('loss: ', loss.item(), 'epoch: ', epoch)

    print('Finished Training')
