import torch.nn as nn

class MNIST_CNN(nn.Module):
    def __init__(self):
        super(MNIST_CNN,self).__init__()
        self.conv = \
            nn.Sequential(
                nn.Conv2d(1,20,5),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(2),
                nn.Conv2d(20,60,5),
                nn.ReLU(inplace=True),
                nn.Conv2d(60, 10, 5),
                nn.ReLU(inplace=True),
            )
        self.fc = \
            nn.Sequential(
                nn.Linear(4*4*10,32),
                nn.ReLU(inplace=True),
                nn.Linear(32,10)
        )

    def forward(self, x):
        x = self.conv(x)
        x=x.view(x.shape[0],-1)
        t_sne_out = x
        x = self.fc(x)
        return x, t_sne_out
