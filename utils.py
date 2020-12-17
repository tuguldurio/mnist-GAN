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