import torch
from torchvision import datasets, transforms
import torch.optim as optim
import torch.nn as nn
from torch.autograd import Variable
import torch.utils.data as data
import mnist_cnn

import visdom
import numpy as np


vis = visdom.Visdom()

def plt_tracker(loss_plot, loss_value, num):
    vis.line(X=np.stack(np.arange(num, num + 1)),
             Y=np.stack([loss_value]),
             win=loss_plot,
             update='append'
             )


batch_size=32
epoch = 10
gpu = 1


T=transforms.Compose([transforms.ToTensor(),transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5))])

train_data = datasets.MNIST('./data',train=True,transform=T,download=True)
test_data = datasets.MNIST('./data',train=False,transform=T,download=True)

train_loader = data.DataLoader(train_data, batch_size=batch_size,shuffle=True,num_workers=4)
test_loader = data.DataLoader(test_data, batch_size=batch_size,shuffle=True,num_workers=4)

CNN = mnist_cnn.MNIST_CNN()
if gpu == 1:
    CNN = CNN.cuda()

optimizer = optim.SGD(CNN.parameters(),lr = 0.002, momentum=0.5)

loss_fn = nn.CrossEntropyLoss()

lr_scheduler = optim.lr_scheduler.StepLR(optimizer, step_size = 3, gamma = 0.9)

loss_plot = vis.line(Y=np.random.rand(1))


train_len = len(train_loader)
print(train_len)

for ep in range(epoch):
    lr_scheduler.step()
    for i, data in enumerate(train_loader):
        imgs, labels = data
        imgs, labels = Variable(imgs.cuda()), Variable(labels.cuda())

        optimizer.zero_grad()
        out, _ = CNN(imgs)

        loss = loss_fn(out,labels)
        loss.backward()
        optimizer.step()
        print(i, (ep*1875) + i )

        plt_tracker(loss_plot, (float(loss.data.cpu())), ((ep *1875) + i))


