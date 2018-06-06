import torch
import torchvision.transforms as transforms
import torchvision.datasets as data_sets
from torch.utils.data import dataset, DataLoader
import visdom

vis = visdom.Visdom()
batch_size=1
T=transforms.Compose([transforms.ToTensor(),transforms.Normalize(mean=(0.5,0.5,0.5), std=(0.5,0.5,0.5))])

train_set = data_sets.MNIST(root='./data',download=True,transform=T,train=True)
test_set = data_sets.MNIST(root='./data',download=True,transform=T,train=False)

train_loader =DataLoader(train_set,batch_size=batch_size,shuffle=False)
test_loader =DataLoader(test_set,batch_size=batch_size,shuffle=False)


zero_images = torch.FloatTensor(0)
one_images = torch.FloatTensor(0)
two_images = torch.FloatTensor(0)
three_images = torch.FloatTensor(0)
four_images = torch.FloatTensor(0)
five_images = torch.FloatTensor(0)
six_images = torch.FloatTensor(0)
seven_images = torch.FloatTensor(0)
eight_images = torch.FloatTensor(0)
nine_images = torch.FloatTensor(0)

print(train_loader.__len__())
for i, data in enumerate(train_loader):
    images, labels = data

    labels = int(labels)
    if labels == 0:
        zero_images = torch.cat((zero_images,images),0)
    elif labels == 1:
        one_images = torch.cat((one_images, images), 0)
    elif labels == 2:
        two_images = torch.cat((two_images, images), 0)
    elif labels == 3:
        three_images = torch.cat((three_images, images), 0)
    elif labels == 4:
        four_images = torch.cat((four_images, images), 0)
    elif labels == 5:
        five_images = torch.cat((five_images, images), 0)
    elif labels == 6:
        six_images = torch.cat((six_images, images), 0)
    elif labels == 7:
        seven_images = torch.cat((seven_images, images), 0)
    elif labels == 8:
        eight_images = torch.cat((eight_images, images), 0)
    elif labels == 9:
        nine_images = torch.cat((nine_images, images), 0)
    print(i)

print("0")
print(zero_images.shape)
print("1")
print(one_images.shape)
print("2")
print(two_images.shape)
print("3")
print(three_images.shape)
print("4")
print(four_images.shape)
print("5")
print(five_images.shape)
print("6")
print(six_images.shape)
print("7")
print(seven_images.shape)
print("8")
print(eight_images.shape)
print("9")
print(nine_images.shape)


vis.images(zero_images)
vis.images(seven_images)
vis.images(nine_images)


torch.save(zero_images,'./split_img/zero_img.pkl')
torch.save(one_images,'./split_img/one_img.pkl')
torch.save(two_images,'./split_img/two_img.pkl')
torch.save(three_images,'./split_img/three_img.pkl')
torch.save(four_images,'./split_img/four_img.pkl')
torch.save(five_images,'./split_img/five_img.pkl')
torch.save(six_images,'./split_img/six_img.pkl')
torch.save(seven_images,'./split_img/seven_img.pkl')
torch.save(eight_images,'./split_img/eight_img.pkl')
torch.save(nine_images,'./split_img/nine_img.pkl')


