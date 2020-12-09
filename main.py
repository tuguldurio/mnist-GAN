import torch
import torch.nn as nn
import torchvision
from torchvision import datasets, transforms

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
                        shuffle=True, num_workers=2)

    train_iter = iter(trainloader)
    x, y = next(train_iter)

if __name__ == '__main__':
    main()