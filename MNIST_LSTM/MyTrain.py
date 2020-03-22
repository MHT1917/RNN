import torch
import torch.nn as nn
import torch.optim as opt
from torchvision import datasets,transforms
import matplotlib.pyplot as plt
from MNIST_LSTM.MyNet import Mynet
from torch.utils.data import DataLoader

class Train():
    def __init__(self):
        self.net = Mynet().cuda()
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = opt.Adam(self.net.parameters())
        self.data()
        self.epochs = 10

    def data(self):
        transform = transforms.Compose([
            transforms.Resize(28),
            transforms.ToTensor(),
            transforms.Normalize((0.5,),(0.5,))
        ])
        self.dataset = datasets.MNIST(root="datasets",train=True,download=False,transform=transform)
        self.testdataset = datasets.MNIST(root="datasets",train=False,download=False,transform=transform)
    def load_data(self,batch_size):
        self.trainlist = DataLoader(self.dataset,batch_size=batch_size,shuffle=True)
        self.testlist = DataLoader(self.testdataset,batch_size=batch_size,shuffle=False)
    def train(self):
        losses = []
        for i in range(self.epochs):
            correct = 0
            total = 0
            print("epochs :{}".format(i))
            self.load_data(512)
            for j,(input,target) in enumerate(self.trainlist):
                input,target = input.cuda(),target.cuda()
                output = self.net(input)
                # target = torch.zeros(target.size(0),10).cuda().scatter(1,target.view(-1,1),1)
                loss = self.criterion(output,target)
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                if j % 10 == 0:
                    losses.append(loss.float())
                    print("[epochs - {0} - {1}/{2}]loss :{3}".format(i,j,len(self.trainlist),loss.float()))
                    plt.clf()
                    plt.plot(losses)
                    plt.savefig('loss.jpg')
                    plt.pause(0.01)
            for input,target in self.testlist:
                input, target = input.cuda(), target.cuda()
                output = self.net(input)
                predicted = torch.argmax(output,1)
                total += target.size(0)
                correct += (predicted.eq(target)).sum()
                accuracy = correct.float()/total
            print("[epochs - {0}]Accuracy :{1}%".format(i,accuracy*100))
        torch.save(self.net,"models/net.pth")

if __name__ == '__main__':
    obj = Train()
    obj.train()