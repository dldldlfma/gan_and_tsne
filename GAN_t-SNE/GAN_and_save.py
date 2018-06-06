import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
from torch.optim import lr_scheduler
from torch.autograd import Variable
import torchvision
import torchvision.transforms as transforms
import torchvision.datasets as data_sets
from torch.utils.data import dataset, DataLoader
import torch.optim as optim
import visdom
from collections import OrderedDict
import numpy as np

vis = visdom.Visdom()
batch_size=32
T=transforms.Compose([transforms.ToTensor(),transforms.Normalize(mean=(0.5,0.5,0.5), std=(0.5,0.5,0.5))])

train_set = data_sets.MNIST(root='./data',download=True,transform=T,train=True)
test_set = data_sets.MNIST(root='./data',download=False,transform=T,train=False)

zero_set = torch.load('./split_img/two_img.pkl')

train_loader =DataLoader(zero_set,batch_size=batch_size,shuffle=True,num_workers=4)
test_loader =DataLoader(test_set,batch_size=batch_size,shuffle=False,num_workers=4)

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

D=Discriminator().cuda()
G=Generator().cuda()

D_optim = optim.Adam(D.parameters(),lr=0.002)
G_optim = optim.Adam(G.parameters(),lr=0.002)

loss = nn.BCELoss()

lr_sche = lr_scheduler.StepLR(G_optim,step_size=2,gamma=0.9)


def loss_tracker(loss_plot, loss_value, num):
    vis.line(X=np.stack(np.arange(num, num + 1)),
             Y=np.stack([loss_value]),
             win=loss_plot,
             update='append'
             )

len(train_loader)

d_loss_plot = vis.line(Y=np.random.rand(1))
g_loss_plot = vis.line(Y=np.random.rand(1))

epoch = 50

for ep in range(epoch):
    lr_sche.step()
    print(ep)
    for i, data in enumerate(train_loader):
        # real_imgs, _= data
        real_imgs = data
        real_imgs = real_imgs.cuda()
        real_imgs = Variable(real_imgs)

        z_inputs = torch.randn(real_imgs.shape[0], 100, 1, 1)
        z_inputs = Variable(z_inputs.cuda())

        true_labels = Variable((torch.ones(z_inputs.shape[0], 1)).cuda())
        false_labels = Variable((torch.zeros(z_inputs.shape[0], 1)).cuda())

        D_optim.zero_grad()

        fake_imgs = G(z_inputs)
        fake_out = D(fake_imgs)

        real_out = D(real_imgs)
        d_loss = loss(real_out, true_labels) + loss(fake_out, false_labels)
        d_loss.backward()
        D_optim.step()

        G_optim.zero_grad()
        z_inputs = torch.randn(real_imgs.shape[0], 100, 1, 1)
        z_inputs = Variable(z_inputs.cuda())

        fake_imgs = G(z_inputs)
        fake_out = D(fake_imgs)

        g_loss = loss(fake_out, true_labels)
        g_loss.backward(retain_graph=True)
        G_optim.step()

        if i % 100 == 1:
            loss_tracker(d_loss_plot, (float(d_loss.data.cpu())), ((ep * 1875) + i))
            loss_tracker(g_loss_plot, (float(g_loss.data.cpu())), ((ep * 1875) + i))
            z_inputs = torch.randn(real_imgs.shape[0], 100, 1, 1)
            z_inputs = Variable(z_inputs.cuda())
            fake_imgs = G(z_inputs)
            vis.images(fake_imgs.data.cpu())


torch.save(G, './Generator/Generator_two.pth')


G2=torch.load('./Generator/Generator_two.pth')
CNN = torch.load('./MNIST_CNN.pth')



input_test = Variable( (torch.randn(100,100,1,1)).cuda())
fake_imgs = G2(input_test)
vis.images(fake_imgs.data.cpu())
out, _= CNN(fake_imgs)
_, predicted = torch.max(out.data, 1)

print(predicted)