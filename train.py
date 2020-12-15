import time
import math
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
import matplotlib  
import matplotlib.pyplot as plt
import utils
import model

def main():
    parser = argparse.ArgumentParser(description='MNIST DCGAN')
    parser.add_argument('-z','--z-dim', type=int, default=100, help='dimension of noise')
    parser.add_argument('--lr', type=float, default=0.0002, help='learning rate')
    parser.add_argument('-bs', '--batch-size', type=int, default=128, help='batch size')
    parser.add_argument('--epochs', type=int, default=20, help='number of epochs to train')
    parser.add_argument('--test-imgs', type=int, default=25, help='number of test images to save')
    parser.add_argument('--log-interval', type=int, default=100, help='batches to wait before logging')
    args = parser.parse_args()

    # Load data
    img_size = 64
    trainloader = utils.load_data(args.batch_size, img_size)

    # cuda or cpu
    if torch.cuda.is_available():
        device = torch.device('cuda')
        print('backend: GPU')
    else:   
        device = torch.device('cpu')
        print('backend: CPU')

    # define models
    G = model.Generator(args.z_dim, 64)
    D = model.Discriminator(64)
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

        utils.plot_result(G, z_fixed, epoch, img_size, 'result',  (math.sqrt(args.test_imgs), math.sqrt(args.test_imgs)), device)

        print('epoch {} took {:.2f}s to train'.format(epoch, time.time() - epoch_time_start))
    
    torch.save(G.state_dict(), 'model/Generator.pth')

if __name__ == '__main__':
    main()