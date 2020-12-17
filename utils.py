import os
from os import walk
from PIL import Image
import glob
import natsort
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torch.autograd import Variable

fp_in = './results/*.png'
fp_out = 'result.gif'

def load_data(batch_size, img_size):
    transform = transforms.Compose([
        transforms.Resize(img_size),
        transforms.ToTensor(),
        transforms.Normalize((0.5), (0.5))
    ])

    trainset = datasets.MNIST('./data', train=True, 
                        download=True, transform=transform)
    trainloader = DataLoader(trainset, batch_size=batch_size, 
                        shuffle=True)
    return trainloader

def plot_result(G, z, epoch, image_size, save_dir, fig_size, device):
    G.eval()
    noise = Variable(z.to(device))
    test_pred = G(z)
    G.train()

    # save figure
    # if save:
    # if not os.path.exists(save_dir):
    #     os.mkdir(save_dir)
    # save_fn = save_dir + 'MNIST_DCGAN_epoch_{:d}'.format(epoch+1) + '.png'
    # plt.savefig(save_fn)

def save_gif():
    images = os.listdir('results')
    images = natsort.natsorted(images)
    rng = images[-1].split('_')
    for i in range(int(rng[0])):
        for j in range(int(rng[1])):
            pass 

if __name__ == '__main__':
    save_gif()