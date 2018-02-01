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
from logger import Logger
import argparse
import datetime,time,os

class vgg16_withdfconvolution(nn.Module):
    def __init__(self,models,number_sharing,output_number):
        super(vgg16_withdfconvolution,self).__init__()
        #vgg = models.vgg16(pretrained = True)
    
        for n,p in models.features._modules.items():
            p.inplace = False
        self.layer1 = nn.Sequential(*list(models.features.children())[0:19])
        self.dflayer1 = dconv(512)
        self.layer2 = nn.Sequential(*list(models.features.children())[19:21])
        self.layer3 = nn.Sequential(*list(models.features.children())[21:23])
        self.pooling = nn.MaxPool2d(4,stride = 2)
        self.fc1 =  nn.Linear(512,512)
        self.fc2 = nn.Linear(512,100)
        self.tr = number_sharing
        self.te = output_number
    def forward(self, x):
        output = self.layer1(x)
        if self.training:
            count = self.tr
        else:
            count = self.te

        for i in range(count):
            if i == 0:
                output1 = self.layer2(output)
                output1 =output1.unsqueeze(0)
            else:
                output1 = torch.cat((output1,self.layer2(output1[i-1]).unsqueeze(0)),0)

        
        output_last = self.dflayer1(output1[count-1])
         
        for i in range(count):
            if i == 0:  
                output2 = self.pooling(output1[0])
                output2 = output2.view(-1,512)
                output2 = nn.Dropout(p=0.5)(output2)
                output2 = self.fc1(output2)
                output2 = nn.ReLU()(output2)
                output2 = self.fc2(output2)
                output2 = output2.unsqueeze(0)
            elif i == count-1:
                temp = self.pooling(output_last)
                temp = temp.view(-1,512)
                temp = nn.Dropout(p=0.5)(temp)
                temp = self.fc1(temp)
                temp = nn.ReLU()(temp)
                temp = self.fc2(temp)
                output2 = torch.cat((output2, temp.unsqueeze(0)),0)
            else:
                temp = self.pooling(output1[i])
                temp = temp.view(-1,512)
                temp = nn.Dropout(p=0.5)(temp) 
                temp = self.fc1(temp)
                temp = nn.ReLU()(temp)
                temp = self.fc2(temp)
                output2 = torch.cat((output2,temp.unsqueeze(0)),0) 
        return output2

def evaluation(net,test_loader,output_number,logger):
    net.eval()
    start = time.time()
    total = 0
    correct = [0] * output_number
    for images, labels in test_loader:
        images = Variable(images).cuda()

        outputs = net(images)
    #    correct = [0.0] * output_number
        total += labels.size(0)
        for i in range(output_number):
            _, predicted = torch.max(outputs[i].data, 1)
            correct[i] += (predicted.cpu() == labels).sum()
    end  =time.time()
    for i in range(output_number):
        accuracy = 100.0 * correct[i] / total
        print('Test Accuracy of the model on the 10000 test images: %.2f %%, Time : %.3f' %(accuracy, end - start))
        logger.scalar_summary('test_accuracy' + str(i), accuracy, epoch)

def training(net,save_path, num_epochs=50, learning_rate=0.01, batch_size=500, numner_sharing=4, output_number=4):
    date = datetime.datetime.now().strftime('%Y%m%d_%H%M')
    log_dir = date
    logger = Logger(os.path.join('./logger', log_dir))
    batch_size = batch_size
    learning_rate = learning_rate
    num_epochs = num_epochs
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
    output_number = output_number
    net = net
    #net = vgg16_withdfconvolution(number_sharing,output_number)
    net.cuda()
    for n,p in net._modules.items():
        p.inplace = False
    criterion = nn.CrossEntropyLoss()

    optimizer = torch.optim.SGD(net.parameters() ,lr = learning_rate,momentum= 0.9)
    first_decay = num_epochs //2
    second_decay = num_epochs * 3 //4
    scheduler = lr_scheduler.MultiStepLR(optimizer,
                                     [first_decay,second_decay],
                                     gamma = 0.1)
    global epoch
    print num_epochs
    for epoch in range(num_epochs):
        net.train() 
        scheduler.step()
        num = 0
        total_valid = 0
        correct_valid = 0
        for i, (images, labels) in enumerate(train_loader):
            net.train()
            images = Variable(images).cuda()

            labels = Variable(labels).cuda()
            optimizer.zero_grad()
            outputs = net.forward(images)
            for j in range(number_sharing): 
                if j ==0:    
                    loss = [criterion(outputs[0], labels)]
                else:
                    loss.append(criterion(outputs[j],labels))
            
            final_loss = sum(loss)
            final_loss.backward()
            optimizer.step()
            if (i + 1) % 20 == 0:
                print ('Epoch [%d/%d], Iter [%d/%d] Loss: %.4f'
                   % (epoch + 1, num_epochs, i + 1, len(train_dataset) // batch_size, final_loss.data[0]))


        if (epoch+1) % 2 ==0:
            torch.save(net.state_dict(), os.path.join('./models', save_path))
        evaluation(net,test_loader,output_number,logger)
    torch.save(model.state_dict(), os.path.join('./models', 'final_models'))
parser = argparse.ArgumentParser(description = 'using_deformableconvolution')
parser.add_argument('--num_epochs', default = 50, type = int, metavar = 'N', help = 'number of total epochs')
parser.add_argument('--batch_size', default = 500, type = int, metavar = 'N', help = 'batch_size')
parser.add_argument('--lr', '--learning_rate', default = 0.01, type = float, metavar = 'LR', help = 'initial learning rate')
parser.add_argument('--save_path', nargs = '?', type = str, default = 'result_model_i_rand.pth', help = 'save_path')
parser.add_argument('--number_sharing', type = int, default = 4, help =' how many training sharing')
parser.add_argument('--output_number', type = int, default = 4, help = 'how many output number')
args = parser.parse_args()
vgg16 = models.vgg16(pretrained = True)
net = vgg16_withdfconvolution(vgg16,args.number_sharing, args.output_number) 
training(net,args.save_path, args.num_epochs, args.lr, args.batch_size, args.number_sharing, args.output_number)
