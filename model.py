import torch
from torch import nn
import torch.nn.functional as F
from module import EleAttG,VA_subnetwork

class stacked_rnn(nn.Module):
    def __init__(self,input_size,hidden_size,num_class):
        super(type(self),self).__init__()
        self.input_size=input_size
        self.num_class=num_class
        self.hidden_size=hidden_size
        self.rnn_1=nn.LSTMCell(self.input_size,self.hidden_size).float()
        self.rnn_2=nn.LSTMCell(self.hidden_size,self.hidden_size).float()
        self.fc=nn.Linear(self.hidden_size,self.num_class)
    def forward(self,x,h1,c1,h2,c2,mode):
        batch,length,_,_=x.shape
        x=x.view(batch,length,-1)
        assert mode in ['train','val'],'mode undefined'
        assert x.shape[2]==self.input_size,'input size not match'
        for i in range(length):
            data=x[:,i]
            h1,c1=self.rnn_1(data,(h1,c1))
            if mode=='train':
                h2,c2=self.rnn_2(F.dropout(h1),(h2,c2))
            elif mode=='val':
                h2,c2=self.rnn_2(h1,(h2,c2))
        return self.fc(h2)

class stacked_EleAttg(nn.Module):
    def __init__(self,input_size,hidden_size,num_class):
        super(type(self),self).__init__()
        self.input_size=input_size
        self.num_class=num_class
        self.hidden_size=hidden_size
        self.va_sub=VA_subnetwork(self.input_size,self.input_size)
        self.rnn_1=EleAttG(self.input_size,self.hidden_size)
        self.rnn_2=EleAttG(self.hidden_size,self.hidden_size)
        self.rnn_3=EleAttG(self.hidden_size,self.hidden_size)
        self.fc=nn.Linear(self.hidden_size,self.num_class)
    def forward(self,x,rt_h,dt_h,h1,h2,h3,mode):
        batch,length,_,_=x.shape
        x=self.va_sub(x,rt_h,dt_h)
        x=x.view(batch,length,-1)
        assert mode in ['train','val'],'mode undefined'
        assert x.shape[2]==self.input_size,'input size not match'
        for i in range(length):
            data=x[:,i]
            h1=self.rnn_1(data,h1)
            if mode=='train':
                h2=self.rnn_2(F.dropout(h1),h2)
                h3=self.rnn_3(F.dropout(h2),h3)
            elif mode=='val':
                h2=self.rnn_2(F.dropout2d(h1),h2)
                h3=self.rnn_3(F.dropout2d(h2),h3)
        return self.fc(h3)

