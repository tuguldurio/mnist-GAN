import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
from torchvision import datasets, transforms

import models

def train(G, D, trainloader, criterion, G_optim, D_optim, epoch, device):
    for i, (x, _) in enumerate(trainloader):
        x = x.to(device)

        # Train D
        D_optim.zero_grad()
        y_real = torch.ones(x.size()[0]).to(device)
        y_fake = torch.zeros(x.size()[0]).to(device)

        print(x.shape)
        D_pred = D(x).squeeze()
        D_real_loss = criterion(D_pred, y_real)
        print(D_real_loss)

        # Train G
        G_optim.zero_grad()
        print('a')


def main():
    # Load data
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5), (0.5))
    ])

    trainset = datasets.MNIST('./data', train=True, 
                        download=True, transform=transform)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=64, 
                        shuffle=True, num_workers=2)

    testtest = datasets.MNIST('./data', train=False, 
                        download=True, transform=transform)
    testloader = torch.utils.data.DataLoader(trainset, batch_size=100, 
                        shuffle=False, num_workers=2)

    # cuda or cpu
    if torch.cuda.is_available():
        device = torch.device('cuda')
        print('backend: GPU')
    else:   
        device = torch.device('cpu')
        print('backend: CPU')

    # define models
    G = models.Generator()
    D = models.Discriminator()
    G.weight_init(0, 0.2)
    D.weight_init(0, 0.2)
    G.to(device)
    D.to(device)

    # loss
    criterion = nn.BCELoss()

    # optimizer
    G_optim = optim.Adam(G.parameters(), 0.001, (0.5, 0.999))
    D_optim = optim.Adam(D.parameters(), 0.001, (0.5, 0.999))

    # train
    for epoch in range(1, 2):
        train(G, D, trainloader, criterion, G_optim, D_optim, epoch, device)
    

if __name__ == '__main__':
    main()