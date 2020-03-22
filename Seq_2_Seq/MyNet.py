import os
import numpy as np
import torch
from torch import nn
import  torch.utils.data as data
from Seq_2_Seq.MyData import MyDataset

img_path = "data"
BATCH_SIZE = 64
NUM_WORKERS = 4
EPOCH = 100
save_path = r"params/seq2seq.pkl"

class Encoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(3,64,3,1,1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(),
            nn.Conv2d(64, 128, 3, 2, 1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(),
            nn.Conv2d(128, 128, 3, 1,1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(),
            nn.Conv2d(128, 64, 3, 2, 1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(),
            nn.Conv2d(64, 32, 3, 1,1),
            nn.BatchNorm2d(32),
            nn.LeakyReLU()
        )
        self.fc1 = nn.Linear(32*15*30,128)

    def forward(self, x):
        conv_out = self.conv(x)
        conv_out = conv_out.reshape(-1,32*15*30)
        out = self.fc1(conv_out)
        return out

class Decoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.lstm = nn.LSTM(128,128,2,batch_first=True)
        self.out = nn.Linear(128,10)
    def forward(self, x):
        x = x.reshape(-1,1,128)
        x = x.expand(BATCH_SIZE,4,128)#N,4,128
        lstm_out,(h_n,h_c) = self.lstm(x)#N,4,128
        lstm_out = lstm_out.reshape(-1,128)#N*4,128
        out = self.out(lstm_out)#N*4,10
        out = out.reshape(-1,4,10)#N,4,10
        return out

class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = Encoder()
        self.decoder = Decoder()
    def forward(self, x):
        encoder = self.encoder(x)
        out = self.decoder(encoder)
        return out
if __name__ == '__main__':
    net = Net().cuda()
    opt = torch.optim.Adam(net.parameters())
    loss_func = nn.MSELoss()

    if os.path.exists(save_path):
        net.load_state_dict(torch.load(save_path))
    train_data = MyDataset(root="data")
    train_loader = data.DataLoader(train_data,BATCH_SIZE,shuffle=True,drop_last=True,num_workers=NUM_WORKERS)

    for epoch in range(EPOCH):
        for i,(x,y) in enumerate(train_loader):
            batch_x = x.cuda()
            batch_y = y.float().cuda()

            out = net(batch_x)

            loss = loss_func(out,batch_y)

            opt.zero_grad()
            loss.backward()
            opt.step()

            if i % 5 == 0:
                test_y = torch.argmax(y, 2).detach().cpu().numpy()
                pred_y = torch.argmax(out, 2).cpu().detach().numpy()
                acc = np.mean(np.all(pred_y == test_y, axis=1))
                print("epoch:", epoch, "Loss:", loss.item(), "acc:", acc)
                print("test_y:", test_y[0])
                print("pred_y", pred_y[0])
        torch.save(net.state_dict(), save_path)