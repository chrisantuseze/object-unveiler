import torch
from torch import nn
import torchvision
from torchvision.utils import save_image
import torch.optim as optim
from PIL import Image

class Generator(nn.Module):
    def __init__(self, n_classes, ):
        super(Generator, self).__init__()

        self.yz = nn.Sequential(
            nn.Linear(100, 200),
            nn.ReLU(True)
        )

        self.label_emb = nn.Embedding(n_classes, n_classes)
        self.fc = nn.Linear(n_classes, n_classes * n_classes)
        
        self.main = nn.Sequential(
            nn.ConvTranspose2d(1200, ngf * 8, 4, 1, 0, bias=False),
            nn.BatchNorm2d(ngf * 8),
            nn.ReLU(True),
            # state size. (ngf*8) x 4 x 4
            nn.ConvTranspose2d(ngf * 8, ngf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 4),
            nn.ReLU(True),
            # state size. (ngf*4) x 8 x 8
            nn.ConvTranspose2d(ngf * 4, ngf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 2),
            nn.ReLU(True),
            # state size. (ngf*2) x 16 x 16
            nn.ConvTranspose2d(ngf * 2,ngf, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf),
            nn.ReLU(True),
            # state size. (ngf) x 32 x 32
            nn.ConvTranspose2d(ngf,nc, 4, 2, 1, bias=False),
            nn.Sigmoid()
            # state size. (nc) x 64 x 64
        )

    def forward(self, z, y):
        
        #mapping noise and label
        z = self.yz(z)

        y = self.label_emb(y)
        y = self.fc(y)
        
        #mapping concatenated input to the main generator network
        inp = torch.cat((z,y), dim=1)
        inp = inp.view(-1, 1200, 1, 1)
        output = self.main(inp)

        return output
    

class Discriminator(nn.Module):
    def __init__(self, ngpu):
        super(Discriminator, self).__init__()
            
        self.ylabel=nn.Sequential(
            nn.Linear(120,64*64*1),
            nn.ReLU(True)
        )
        
        self.main = nn.Sequential(
            # input is (nc) x 64 x 64
            nn.Conv2d(nc+1, ndf, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf) x 32 x 32
            nn.Conv2d(ndf, ndf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 2),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*2) x 16 x 16
            nn.Conv2d(ndf * 2, ndf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 4),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*4) x 8 x 8
            nn.Conv2d(ndf * 4, ndf * 8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 8),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*8) x 4 x 4
            nn.Conv2d(ndf * 8, 1, 4, 1, 0, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x,y):
        y=self.ylabel(y)
        y=y.view(-1,1,64,64)
        inp=torch.cat([x,y],1)
        output = self.main(inp)
        
        return output.view(-1, 1).squeeze(1)