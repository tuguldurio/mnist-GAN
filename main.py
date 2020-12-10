import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
from torchvision import datasets, transforms
from torchvision.utils import save_image
from torch.autograd import Variable
import matplotlib  
import matplotlib.pyplot as plt
import argparse
import models

def main():
    global parser
    parser = argparse.ArgumentParser(description='MNIST DCGAN')
    parser.add_argument('-z','--z-dim', type=int, default=100, help='dimension of noise')
    parser.add_argument('--log-interval', type=int, default=100, help='batches to wait before logging')
    args = parser.parse_args()

    # Load data
    transform = transforms.Compose([
        transforms.Resize(32),
        transforms.ToTensor(),
        transforms.Normalize((0.5), (0.5))
    ])

    trainset = datasets.MNIST('./data', train=True, 
                        download=True, transform=transform)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=128, 
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
    G = models.Generator(args.z_dim, 128)
    D = models.Discriminator(64)
    G.weight_init(0.0, 0.02)
    D.weight_init(0.0, 0.02)
    G.to(device)
    D.to(device)

    # loss
    criterion = nn.BCELoss()

    # optimizer
    G_optimizer = optim.Adam(G.parameters(), 0.0002, betas=(0.5, 0.999))
    D_optimizer = optim.Adam(D.parameters(), 0.0002, betas=(0.5, 0.999))

    z_test = Variable(torch.randn(1, args.z_dim, 1, 1).to(device))

    # train
    for epoch in range(1, 21):
        for i, (x, _) in enumerate(trainloader, 1):
            # Train D 
            D.zero_grad()
            y_real = torch.ones(x.size(0))
            y_fake = torch.zeros(x.size(0))

            x, y_real, y_fake = Variable(x.to(device)), Variable(y_real.to(device)), Variable(y_fake.to(device))

            D_pred = D(x).squeeze()
            D_real_loss = criterion(D_pred, y_real)

            z = torch.randn(x.size(0), args.z_dim, 1, 1)
            z = Variable(z.to(device))
            G_pred = G(z)

            D_pred = D(G_pred).squeeze()
            D_fake_loss = criterion(D_pred, y_fake)
            # D_fake_score = D_pred.data.mean()

            D_train_loss = D_real_loss + D_fake_loss

            D_train_loss.backward()
            D_optimizer.step()

            # Train G
            G.zero_grad()

            z = torch.randn(x.size(0), args.z_dim, 1, 1)
            z = Variable(z.to(device))

            G_pred = G(z)
            D_pred = D(G_pred).squeeze()
            G_train_loss = criterion(D_pred, y_real)
            G_train_loss.backward()
            G_optimizer.step()

            if i % args.log_interval == 0:
                print('[epoch {}, {}/{}] D_loss {:.3f}, G_loss {:.3f}'.format(
                    epoch, i, len(trainloader), 
                    D_train_loss.item(), G_train_loss.item())
                    )
                save_image(G(z_test), f'results/{epoch}_{i}.png')
    

if __name__ == '__main__':
    main()