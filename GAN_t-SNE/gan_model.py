import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F

class Generator(nn.Module):
    def __init__(self):
        super(Generator,self).__init__()
        self.layer1=nn.Sequential(
            nn.ConvTranspose2d(100,512,5,1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(512,256,4,2),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(256,64,4,2,padding=1,output_padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(64,1,4,1),
            nn.Tanh()
        )
        init.xavier_uniform(self.layer1[0].weight)
        init.xavier_uniform(self.layer1[3].weight)
        init.xavier_uniform(self.layer1[6].weight)

    def forward(self,x):
        x=self.layer1(x)
        return x

class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator,self).__init__()
        self.layer1=nn.Sequential(
            nn.Conv2d(1,32,5,1,2),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(32, 16, 5, 1),
            nn.BatchNorm2d(16),
            nn.LeakyReLU(),

            nn.Conv2d(16, 8, 5, 1),
            nn.BatchNorm2d(8),
            nn.LeakyReLU(),
        )

        self.fc=nn.Linear(6*6*8,1)
    def forward(self,x):
        x = self.layer1(x)
        x=x.view(x.shape[0],-1)
        x = F.sigmoid(self.fc(x))
        return x
