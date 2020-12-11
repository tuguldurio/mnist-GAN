import time
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
from torchvision import datasets, transforms
from torchvision.utils import save_image
from torch.autograd import Variable
import matplotlib  
import matplotlib.pyplot as plt

import models

def main():
    global parser
    parser = argparse.ArgumentParser(description='MNIST DCGAN')
    parser.add_argument('-z','--z-dim', type=int, default=100, help='dimension of noise')
    parser.add_argument('--lr', type=float, default=0.0002, help='learning rate')
    parser.add_argument('-bs', '--batch-size', type=int, default=128, help='batch size')
    parser.add_argument('--epochs', type=int, default=20, help='number of epochs to train')
    parser.add_argument('--test-imgs', type=int, default=5, help='number of test images to save')
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
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=args.batch_size, 
                        shuffle=True, num_workers=2)

    # cuda or cpu
    if torch.cuda.is_available():
        device = torch.device('cuda')
        print('backend: GPU')
    else:   
        device = torch.device('cpu')
        print('backend: CPU')

    # define models
    G = models.Generator(args.z_dim, 128)
    D = models.Discriminator(128)
    G.weight_init(0.0, 0.02)
    D.weight_init(0.0, 0.02)
    G.to(device)
    D.to(device)

    # loss
    criterion = nn.BCELoss()

    # optimizer
    G_optimizer = optim.Adam(G.parameters(), args.lr, betas=(0.5, 0.999))
    D_optimizer = optim.Adam(D.parameters(), args.lr, betas=(0.5, 0.999))

    # test noise
    z_fixed = torch.randn(args.test_imgs, args.z_dim, 1, 1)
    z_fixed = Variable(z_fixed.to(device))

    # train
    
    for epoch in range(1, args.epochs+1):
        epoch_time_start = time.time()
        for i, (x, _) in enumerate(trainloader, 1):
            #########################
            # Train discriminator D #
            #########################
            D.zero_grad()
            y_real = torch.ones(x.size(0))
            y_fake = torch.zeros(x.size(0))

            x, y_real, y_fake = Variable(x.to(device)), Variable(y_real.to(device)), Variable(y_fake.to(device))

            # Train D with real data
            D_pred = D(x).squeeze()
            D_real_loss = criterion(D_pred, y_real)

            # Train D with fake fata
            z = torch.randn(x.size(0), args.z_dim, 1, 1)
            z = Variable(z.to(device))

            G_pred = G(z)

            D_pred = D(G_pred).squeeze()
            D_fake_loss = criterion(D_pred, y_fake)
            # D_fake_score = D_pred.data.mean()

            D_train_loss = D_real_loss + D_fake_loss

            D_train_loss.backward()
            D_optimizer.step()

            #####################
            # Train generator G #
            #####################
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
                G.eval()
                test_pred = G(z_fixed)
                G.train()

                for img_i in range(z_fixed.size(0)):
                    save_image(test_pred[img_i], f'results/{epoch}_{i}_{img_i}.png')

        print('epoch {} took {:.2f}s to train'.format(epoch, time.time() - epoch_time_start))
    

if __name__ == '__main__':
    main()