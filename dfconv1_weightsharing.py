from torch_deform_conv.layers import ConvOffset2D as dconv
import torch
import torchvision.models as models
from torch import nn
from torch.optim import lr_scheduler
from torch.autograd import Variable
import numpy as np
import torchvision
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

vgg16 = models.vgg16(pretrained=True)


for n,p in vgg16.features._modules.items():
   if isinstance(p,torch.nn.ReLU):
        p.inplace = False
        print p.inplace
#for param in vgg16.parameters():
   # print param
#    print param.requires_grad

#class

class Network(nn.Module):
    def __init__(self, models, number_sharing):
        super(Network,self).__init__()
        self.layer1 = nn.Sequential(*list(models.features.children())[0:19])
      
        self.dflayer1 = dconv(512)
       # self.dflayer2 = dconv(256)
       # self.dflayer3 = dconv(256)
        self.layer2 = nn.Sequential(*list(models.features.children())[19:21])
        
        self.layer3 = nn.Sequential(*list(models.features.children())[21:23])
       # self.maxpooling = nn.MaxPool2d(2, stride =2) 
        self.pooling = nn.AvgPool2d(4,stride = 2)
        self.fc1 =  nn.Linear(512,512)
        self.fc2 = nn.Linear(512,100)
        self.count = number_sharing
    def forward(self, x):
        output = self.layer1(x)
       # output= self.layer2(output)
        #output = output.unsqueeze(0)
       # output = output.unsqueeze(0)
       # output = torch.cat((output,output,output,output),0)
         
        output1 = self.layer2(output)           
               # output1 = self.layer3(output1)
               # output1 = output1.unsqueeze(0)
        
        #output3 = self.dflayer1(output1)
        output3 = self.layer2(output1)
      #  output4 = self.dflayer2(output3)
        output4 = self.layer2(output3)
        output5 = self.layer2(output4)
        output1 = self.dflayer1(output1)
        output3 = self.dflayer1(output3)
        output4 = self.dflayer1(output4)
        output5 = self.dflayer1(output5)
       # output5 = self.dflayer1(output4)
       # output5 = self.layer2(output5) 
        output1 = output1.unsqueeze(0)
        output3 = output3.unsqueeze(0)
        output4 = output4.unsqueeze(0)
        output5 = output5.unsqueeze(0)
        output1 = torch.cat((output1,output3,output4,output5),0)
         
        for i in range(self.count):
            if i == 0:  
                output2 = self.pooling(output1[0])
               # output2 = self.layer3(output1)
      #  output2 = self.pooling(output2)
                output2 = output2.view(-1,512)
                output2 = nn.Dropout(p=0.5)(output2)
                output2 = self.fc1(output2)
                output2 = nn.ReLU()(output2)
                output2 = self.fc2(output2)
                output2 = output2.unsqueeze(0)
            else:
                temp = self.pooling(output1[i])
                temp = temp.view(-1,512)
                temp = nn.Dropout(p=0.5)(temp) 
                temp = self.fc1(temp)
                temp = nn.ReLU()(temp)
                temp = self.fc2(temp)
            
                output2 = torch.cat((output2,temp.unsqueeze(0)),0) 
        return output2

batch_size = 500
learning_rate = 0.01
num_epochs = 50
scale = 40
image_size = 32
# bilinear_interpolation -> size
transform =  transforms.ToTensor()
train_dataset = datasets.CIFAR100(
    root= '../data',
    train = True,
    download = True,
    transform= transform
)
train_loader = DataLoader(train_dataset,
                          batch_size = batch_size,
                          shuffle= True,
                          num_workers=4)
test_dataset = datasets.CIFAR100(
    root = '../data',
    train = False,
    download= True,
    transform =transforms.ToTensor())
test_loader = DataLoader(test_dataset,
                         batch_size = batch_size,
                         shuffle= False,
                         num_workers= 4)
number_sharing = 4
net = Network(vgg16,number_sharing)
net.cuda()
#net = torch.nn.DataParallel(net, device_ids=range(torch.cuda.device_count()))
#cudnn.benchmark = True 

#for params in net.parameters():
   # print params.requires_grad
print net._modules.items
for n,p in net._modules.items():
  # if isinstance(p,torch.nn.ReLU):
        p.inplace = False
        print p.inplace
#for params in net.parameters():
  #  params.cuda()
#print net.layer1.parameters
#print net.layer2.parameters
#print net.layer3.parameters()
#print net.fc1.parameters
#print net.fc2.parameters
criterion = nn.CrossEntropyLoss()

optimizer = torch.optim.SGD(net.parameters() ,lr = learning_rate,momentum= 0.9)
first_decay = num_epochs //2
second_decay = num_epochs * 3 //4
scheduler = lr_scheduler.MultiStepLR(optimizer,
                                     [first_decay,second_decay],
                                     gamma = 0.1)

for epoch in range(num_epochs):
    scheduler.step()
    num = 0
    total_valid = 0
    correct_valid = 0
    for i, (images, labels) in enumerate(train_loader):
        net.train()
        images = Variable(images).cuda()

        #   label = labels
        labels = Variable(labels).cuda()

        # Forward + Backward + Optimize
        optimizer.zero_grad()
        outputs = net.forward(images) 
       # output = outputs[3]
        loss3 = criterion(outputs[3], labels)
        loss2 = criterion(outputs[2], labels)
        loss1 = criterion(outputs[1], labels)
        loss0 = criterion(outputs[0], labels)
        loss = loss0 + loss1 + loss2 + loss3
        loss.backward()
        optimizer.step()


        if (i + 1) % 100 == 0:
            print ('Epoch [%d/%d], Iter [%d/%d] Loss: %.4f'
                   % (epoch + 1, num_epochs, i + 1, len(train_dataset) // batch_size, loss.data[0]))


    if epoch == 19 or epoch == 29 or epoch == 39:
        net.eval()
        correct0 = 0
        correct1 = 0
        correct2 = 0
        correct3 = 0
        total = 0

        for images, labels in test_loader:
            images = Variable(images).cuda()

            outputs = net(images)
            _, predicted0 = torch.max(outputs[0].data, 1)
            _, predicted1 = torch.max(outputs[1].data, 1)
            _, predicted2 = torch.max(outputs[2].data, 1)
            _, predicted3 = torch.max(outputs[3].data, 1)
            total += labels.size(0)
            correct0 += (predicted0.cpu() == labels).sum()
            correct1 += (predicted1.cpu() == labels).sum()
            correct2 += (predicted2.cpu() == labels).sum()
            correct3 += (predicted3.cpu() == labels).sum()
        print('Test Accuracy of the model on the 10000 test images: %.2f %%' % (100.0 * correct0 / total))
        print('Test Accuracy of the model on the 10000 test images: %.2f %%' % (100.0 * correct1 / total))
        print('Test Accuracy of the model on the 10000 test images: %.2f %%' % (100.0 * correct2 / total))
        print('Test Accuracy of the model on the 10000 test images: %.2f %%' % (100.0 * correct3 / total))

torch.save(net, 'net.pkl')

net.eval()
correct0 = 0
correct1 = 0
correct2 = 0
correct3 = 0
total = 0

for images, labels in test_loader:
    images = Variable(images).cuda()

    outputs = net(images)
    _, predicted0 = torch.max(outputs[0].data, 1)
    _, predicted1 = torch.max(outputs[1].data, 1)
    _, predicted2 = torch.max(outputs[2].data, 1)
    _, predicted3 = torch.max(outputs[3].data, 1)
    total += labels.size(0)
    correct0 += (predicted0.cpu() == labels).sum()
    correct1 += (predicted1.cpu() == labels).sum()
    correct2 += (predicted2.cpu() == labels).sum()
    correct3 += (predicted3.cpu() == labels).sum()
print('Test Accuracy of the model on the 10000 test images: %.2f %%' % (100.0 * correct0 / total))
print('Test Accuracy of the model on the 10000 test images: %.2f %%' % (100.0 * correct1 / total))
print('Test Accuracy of the model on the 10000 test images: %.2f %%' % (100.0 * correct2 / total))
print('Test Accuracy of the model on the 10000 test images: %.2f %%' % (100.0 * correct3 / total))

