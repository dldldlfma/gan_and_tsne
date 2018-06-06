import torch
import torch.nn as nn
from torch.autograd import Variable
import visdom

vis=visdom.Visdom()

import gan_model

G=gan_model.Generator()
G=G.cuda()

G.load_state_dict(torch.load('./Generator/Generator_two_state_dict.pth'))

a=torch.randn(10,100,1,1)

a=Variable(a.cuda())

vis.images( (G(a)).data.cpu() )

