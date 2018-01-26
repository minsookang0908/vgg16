
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

#for param in VGG.parameters():
   # print param
    #print param.requires_grad

#class

class Network(nn.Module):
    def __init__(self, models, number_sharing):
        super(Network,self).__init__()
        self.layer1 = nn.Sequential(*list(models.features.children())[0:5])
        self.layer2 = nn.Sequential(*list(models.features.children())[5:10])
        self.layer3 = nn.Sequential(*list(models.features.children())[10:17])
        self.layer4 = nn.Sequential(*list(models.features.children())[17:24])
        self.layer5 = nn.Sequential(*list(models.features.children())[24:26])   
        self.strideconvolution11 = nn.Conv2d(64,128,3,stride =2, padding =2) 
        self.strideconvolution12 = nn.Conv2d(128,256,3, stride =2, padding =2) 
        self.pooling11 = nn.MaxPool2d(2, stride = 2)
        self.pooling12 = nn.MaxPool2d(2, stride = 2)
        self.strideconvolution21 = nn.Conv2d(128,256,3, stride =2, padding =1)   
        self.strideconvolution22 = nn.Conv2d(256,512,3, stride =2, padding =1)  
        self.pooling21 = nn.MaxPool2d(2, stride =2) 
        self.strideconvolution31 = nn.Conv2d(256,512,3, stride=2, padding =2)
        self.pooling31 = nn.MaxPool2d(2, stride =2)
        self.pooling41 = nn.MaxPool2d(2,stride = 2)
        self.fc11 = nn.Linear(256,256)
        self.fc12 = nn.Linear(256,100) 
        self.fc21 = nn.Linear(512,512)
        self.fc22 = nn.Linear(512,100)
        self.fc31 = nn.Linear(512,512)
        self.fc32 = nn.Linear(512,100) 
        self.fc41 =  nn.Linear(512,512)
        self.fc42 = nn.Linear(512,100)
        self.count = number_sharing
    def forward(self, x):
        conv1 = self.layer1(x)
        output1 = self.strideconvolution11(conv1)
        output1 = nn.ReLU()(output1)
        output1 = self.pooling11(output1)
        output1 = self.strideconvolution12(output1)
        output1 = nn.ReLU()(output1)
        output1 = self.pooling12(output1)
        output1 = output1.view(-1,256) 
        output1 = nn.Dropout(p=0.5)(output1) 
        output1 = self.fc11(output1)
        output1 = nn.ReLU()(output1)
        output1 = self.fc12(output1) 

        conv2 = self.layer2(conv1)
        output2 = self.strideconvolution21(conv2) 
        output2 = nn.ReLU()(output2)
        output2 = self.pooling21(output2)
        output2 = self.strideconvolution22(output2)
        output2 = nn.ReLU()(output2)
        output2 = output2.view(-1,512) 
        output2 = nn.Dropout(p=0.5)(output2)
        output2 = self.fc21(output2)
        output2 = nn.ReLU()(output2)
        output2 = self.fc22(output2)
    
        conv3 = self.layer3(conv2)
        output3 = self.strideconvolution31(conv3)
        output3 = nn.ReLU()(output3)
        output3 = self.pooling31(output3)
        output3 = output3.view(-1,512)
        output3 = nn.Dropout(p=0.5)(output3)
        output3 = self.fc31(output3) 
        output3 = nn.ReLU()(output3)
        output3 = self.fc32(output3) 
        
        conv4 = self.layer4(conv3)
        for _ in range(self.count):
            conv4 = self.layer5(conv4)
        output4 = self.pooling41(conv4)
        output4 = output4.view(-1, 512)
        output4 = nn.Dropout(p=0.5)(output4)
        output4 = self.fc41(output4)
        output4 = nn.ReLU()(output4)
        output4 = nn.Dropout(p=0.5)(output4)
        output4 = self.fc42(output4)
        return output1, output2, output3, output4

batch_size = 100
learning_rate = 0.01
num_epochs = 50
scale = 40
image_size = 32
# bilinear_interpolation -> size
transform = transforms.Compose([transforms.Resize(scale,interpolation=2),
                                transforms.RandomHorizontalFlip(),
                                transforms.CenterCrop(image_size),
                                transforms.ToTensor(),
transforms.Normalize([0.5,0.5,0.5],[0.5,0.5,0.5])
])
train_dataset = datasets.CIFAR100(
    root= './data',
    train = True,
    download = True,
    transform= transform
)
train_loader = DataLoader(train_dataset,
                          batch_size = batch_size,
                          shuffle= True,
                          num_workers=4)
test_dataset = datasets.CIFAR100(
    root = './data',
    train = False,
    download= True,
    transform =transforms.Compose([transforms.ToTensor(),
transforms.Normalize([0.5,0.5,0.5],[0.5,0.5,0.5])]
))
test_loader = DataLoader(test_dataset,
                         batch_size = batch_size,
                         shuffle= False,
                         num_workers= 4)
number_sharing = 3
net = Network(vgg16,number_sharing)
net.cuda()
#print net.layer1.parameters
#print net.layer2.parameters
#print net.layer3.parameters()
#print net.fc1.parameters
#print net.fc2.parameters
criterion = nn.CrossEntropyLoss()

optimizer = torch.optim.SGD(net.parameters() ,lr = learning_rate,
                            momentum= 0.9)
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
        output1,output2,output3,output4 = net.forward(images)
        loss1 = criterion(output1, labels)
        loss2 = criterion(output2, labels)
        loss3 = criterion(output3, labels)
        loss4 = criterion(output4, labels)
        loss = loss1 + loss2 + loss3 + loss4
        loss.backward()
        optimizer.step()


        if (i + 1) % 100 == 0:
            print ('Epoch [%d/%d], Iter [%d/%d] Loss: %.4f'
                   % (epoch + 1, num_epochs, i + 1, len(train_dataset) // batch_size, loss.data[0]))


    if epoch == 19 or epoch == 29 or epoch == 39:
        net.eval()
        correct1 = 0
        correct2 = 0
        correct3 = 0
        correct4 = 0
        total = 0

        for images, labels in test_loader:
            images = Variable(images).cuda()

            output1,output2,output3, output4  = net(images)
            _, predicted1 = torch.max(output1.data, 1)
            _, predicted2 = torch.max(output2.data, 1) 
            _, predicted3 = torch.max(output3.data, 1)
            _, predicted4 = torch.max(output4.data, 1) 
            total += labels.size(0)
            correct1 += (predicted1.cpu() == labels).sum()
            correct2 += (predicted2.cpu() == labels).sum()
            correct3 += (predicted3.cpu() == labels).sum()
            correct4 += (predicted4.cpu() == labels).sum()
        print('1Test Accuracy of the model on the 10000 test images: %.2f %%' % (100.0 * correct1 / total))
	print('2Test Accuracy of the model on the 10000 test images: %.2f %%' % (100.0 * correct2 / total))
	print('3Test Accuracy of the model on the 10000 test images: %.2f %%' % (100.0 * correct3 / total))
	print('4Test Accuracy of the model on the 10000 test images: %.2f %%' % (100.0 * correct4 / total))

torch.save(net, 'net.pkl')
net.eval()
correct1 = 0
correct2 = 0
correct3 = 0
correct4 = 0
total = 0

for images, labels in test_loader:
    images = Variable(images).cuda()

    output1, output2,output3, output4 = net(images)
    _, predicted1 = torch.max(output1.data, 1)
    _, predicted2 = torch.max(output2.data, 1) 
    _, predicted3 = torch.max(output3.data, 1)
    _, predicted4 = torch.max(output4.data, 1)
   
    total += labels.size(0)
    correct1 += (predicted1.cpu() == labels).sum()
    correct2 += (predicted2.cpu() == labels).sum()
    correct3 += (predicted3.cpu() == labels).sum()
    correct4 += (predicted4.cpu() == labels).sum() 
    
print('1Test Accuracy of the model on the 10000 test images: %.2f %%' %(100.0* correct1 / total))
print('2Test Accuracy of the model on the 10000 test images: %d %%' % (100 * correct2 / total))
print('3Test Accuracy of the model on the 10000 test images: %d %%' % (100 * correct3 / total))
print('4Test Accuracy of the model on the 10000 test images: %d %%' % (100 * correct4 / total))
