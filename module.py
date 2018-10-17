import torch
from torch import nn
from torch.nn import GRUCell
import torch.nn.functional as F

class EleAttG(nn.Module):
    def __init__(self,input_size,hidden_size,bias=True):
        super(type(self),self).__init__()
        self.input_size=input_size
        self.hidden_size=hidden_size


        self.rnn=GRUCell(self.input_size,self.hidden_size,bias=True)
        self.fc_xa=nn.Linear(self.input_size,self.input_size,bias=bias)
        self.fc_ha=nn.Linear(self.hidden_size,self.input_size,bias=bias)
    def forward(self,x,hx):
        batch,_=x.shape
        attention=F.sigmoid(self.fc_xa(x)+self.fc_ha(hx))
        x=attention*x
        hx=self.rnn(x,hx)
        return hx
class VA_subnetwork(nn.Module):
    def __init__(self, input_size, hidden_size, bias=True):
        super(type(self), self).__init__()
        self.input_size=input_size
        self.hidden_size=hidden_size
        self.rt_gru=GRUCell(self.input_size, self.hidden_size, bias=True)
        self.dt_gru=GRUCell(self.input_size, self.hidden_size, bias=True)
        self.rt_fc=nn.Linear(self.hidden_size, 3, bias=False)
        self.dt_fc=nn.Linear(self.hidden_size, 3, bias=False)

    def forward(self, x, rt_h, dt_h):
        batch_size, time, joint, _= x.shape
        x_processed=list()
        for i in range(time):
            input_x=x[:, i, :, :]
            rt_h=self.rt_gru(input_x.view(batch_size,-1), rt_h)
            dt_h=self.dt_gru(input_x.view(batch_size,-1), dt_h)
            rt=self.rt_fc(rt_h)
            dt=self.dt_fc(dt_h)
            rotate_matrix=self.rotate(rt)

            input_x=input_x.view(batch_size,3,joint)
            x_processed.append((rotate_matrix@input_x).transpose_(1,2)+dt.unsqueeze(1))
        return torch.stack(x_processed,dim=1)

    def rotate(self,rt):
        alpha, belta, gamma= rt.split(dim=1,split_size=1)
        batch_size,_=alpha.shape
        rotate_alpha=torch.zeros(batch_size,3,3).cuda()
        rotate_belta=torch.zeros(batch_size,3,3).cuda()
        rotate_gamma=torch.zeros(batch_size,3,3).cuda()
        rotate_alpha[:,0,0]=1

        rotate_alpha[:, 1, 1]=torch.cos(alpha).squeeze(1)
        rotate_alpha[:,1,2]=torch.sin(alpha).squeeze(1)
        rotate_alpha[:,2,1]=-torch.sin(alpha).squeeze(1)
        rotate_alpha[:, 2, 2]=torch.cos(alpha).squeeze(1)
        rotate_belta[:,0,0]=torch.cos(belta).squeeze(1)
        rotate_belta[:,0,2]=-torch.sin(belta).squeeze(1)
        rotate_belta[:,1,1]=1
        rotate_belta[:,2,0]=torch.sin(belta).squeeze(1)
        rotate_belta[:,2,2]=torch.cos(belta).squeeze(1)

        rotate_gamma[:,0,0]=torch.cos(gamma).squeeze(1)
        rotate_gamma[:,2,2]=1
        rotate_gamma[:,1,0]=torch.sin(gamma).squeeze(1)
        rotate_gamma[:,0,1]=-torch.sin(gamma).squeeze(1)
        rotate_gamma[:,1,1]=torch.cos(gamma).squeeze(1)
        return rotate_alpha@rotate_belta@rotate_gamma




        

