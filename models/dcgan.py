import torch
import torch.nn as nn
import torch.nn.functional as F

class Generator(nn.Module):
    def __init__(self, z, d=128):
        super().__init__()
        self.deconv1 = nn.ConvTranspose2d(z, d*4, 4, 1, 0)
        self.bn1     = nn.BatchNorm2d(d*4)
        self.deconv2 = nn.ConvTranspose2d(d*4, d*2, 4, 2, 1)
        self.bn2     = nn.BatchNorm2d(d*2)
        self.deconv3 = nn.ConvTranspose2d(d*2, d, 4, 2, 1)
        self.bn3     = nn.BatchNorm2d(d)
        # self.deconv4 = nn.ConvTranspose2d(d*2, d, 4, 2, 1)
        # self.bn4     = nn.BatchNorm2d(d)
        self.deconv5 = nn.ConvTranspose2d(d, 1, 4, 2, 1)

    def forward(self, x):
        x = F.relu(self.bn1(self.deconv1(x)))
        x = F.relu(self.bn2(self.deconv2(x)))
        x = F.relu(self.bn3(self.deconv3(x)))
        # x = F.relu(self.bn4(self.deconv4(x)))
        x = torch.tanh(self.deconv5(x))
        return x

    def weight_init(self, mean, std):
        for m in self._modules:
            if isinstance(m, nn.ConvTranspose2d):
                m.weight.data.normal_(mean, std)
                m.bias.data.zero_()

class Discriminator(nn.Module):
    def __init__(self, d=128):
        super().__init__()
        self.conv1 = nn.Conv2d(1, d, 4, 2, 1)
        self.conv2 = nn.Conv2d(d, d*2, 4, 2, 1)
        self.bn2   = nn.BatchNorm2d(d*2)
        self.conv3 = nn.Conv2d(d*2, d*4, 4, 2, 1)
        self.bn3   = nn.BatchNorm2d(d*4)
        # self.conv4 = nn.Conv2d(d*4, d*8, 4, 2, 1)
        # self.bn4   = nn.BatchNorm2d(d*8)
        self.conv5 = nn.Conv2d(d*4, 1, 4, 1, 0)

    def forward(self, x):
        x = F.leaky_relu(self.conv1(x), 0.2)
        x = F.leaky_relu(self.bn2(self.conv2(x)), 0.2)
        x = F.leaky_relu(self.bn3(self.conv3(x)), 0.2)
        # x = F.leaky_relu(self.bn4(self.conv4(x)), 0.2)
        x = torch.sigmoid(self.conv5(x))
        return x

    def weight_init(self, mean, std):
        for m in self._modules:
            if isinstance(m, nn.Conv2d):
                m.weight.data.normal_(mean, std)
                m.bias.data.zero_()
        

if __name__ == '__main__':
    D = Discriminator()
    D.weight_init(0, 0.02)
    print(D)
    x = torch.rand(4, 1, 32, 32)
    y = D(x)
    print(y.shape)

    G = Generator(100)
    G.weight_init(0, 0.02)
    print(G)
    x = torch.rand(4, 100, 1, 1)
    y = G(x)
    print(y.shape)