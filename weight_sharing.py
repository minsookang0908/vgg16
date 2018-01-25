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


for n,p in vgg16.features.items():
   if isinstance(p,torch.nn.ReLU):
        p.inplace = False

#for param in VGG.parameters():
   # print param
    #print param.requires_grad

#class

class Network(nn.Module):
    def __init__(self, models, number_sharing):
        super(Network,self).__init__()
        self.layer1 = nn.Sequential(*list(models.features.children())[0:-3])
        self.layer2 = nn.Sequential(*list(models.features.children())[-3:-1])
        self.pooling = nn.MaxPool2d(2,stride = 1)
        self.fc1 =  nn.Linear(512,512)
        self.fc2 = nn.Linear(512,100)
        self.count = number_sharing
    def forward(self, x):
        output = self.layer1(x)
        for _ in range(self.count):
            output = self.layer2(output)
        output = self.pooling(output)
        output = output.view(-1, 512)
        output = nn.Dropout(p=0.5)(output)
        output = self.fc1(output)
        output = nn.ReLU()(output)
        output = nn.Dropout(p=0.5)(output)
        output = self.fc2(output)
        return output

batch_size = 100
learning_rate = 0.01
num_epochs = 100
scale = 40
image_size = 32
# bilinear_interpolation -> size
transform = transforms.Compose([transforms.Resize(scale,interpolation=2),
                                transforms.RandomHorizontalFlip(),
                                transforms.CenterCrop(image_size),
                                transforms.ToTensor(),
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
                          num_workers=2)
test_dataset = datasets.CIFAR10(
    root = './data',
    train = False,
    download= True,
    transform = transforms.ToTensor()
)
test_loader = DataLoader(test_dataset,
                         batch_size = batch_size,
                         shuffle= False,
                         num_workers= 2)
number_sharing = 3
net = Network(vgg16,number_sharing)
net.cuda()
criterion = nn.CrossEntropyLoss()

optimizer = torch.optim.SGD(net.parameters(), lr = learning_rate,
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
        outputs = net.forward(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()


        if (i + 1) % 100 == 0:
            print ('Epoch [%d/%d], Iter [%d/%d] Loss: %.4f'
                   % (epoch + 1, num_epochs, i + 1, len(train_dataset) // batch_size, loss.data[0]))


    if epoch == 39 or epoch == 59 or epoch == 74:
        net.eval()
        correct = 0
        total = 0

        for images, labels in test_loader:
            images = Variable(images).cuda()

            outputs = net(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted.cpu() == labels).sum()

        print('Test Accuracy of the model on the 10000 test images: %d %%' % (100 * correct / total))


torch.save(net, 'net.pkl')
net.eval()
correct = 0
total = 0

for images, labels in test_loader:
    images = Variable(images).cuda()

    outputs = net(images)
    _, predicted = torch.max(outputs.data, 1)
    total += labels.size(0)
    correct += (predicted.cpu() == labels).sum()

