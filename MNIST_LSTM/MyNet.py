import torch
import torch.nn as nn

class Mynet(nn.Module):
    def __init__(self):
        super(Mynet, self).__init__()
        self.rnn_layer = nn.LSTM(28,64,1,batch_first=True)
        self.out_layer = nn.Linear(64,10)
    def forward(self, x):
        input = x.reshape(-1,28,28)
        h0 = torch.zeros(1,x.shape[0],64).cuda()
        c0 = torch.zeros(1,x.shape[0],64).cuda()
        outputs,(hn,cn) = self.rnn_layer(input,(h0,c0))
        output = outputs[:,-1]
        output = self.out_layer(output)
        return output