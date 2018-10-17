import random 
import torch
import torchvision
import numpy as np
import math

class rotate(object):
    def __init__(self):
        self.rand=0
        self.rotate_x=torch.zeros((3,3))
        self.rotate_y=torch.zeros((3,3))
        self.rotate_z=torch.zeros((3,3))
        self.rotate=torch.zeros((3,3))
    def __call__(self,tensor):
        self.rand_x=(np.random.rand()-0.5)*70*math.pi/180
        self.rand_y=(np.random.rand()-0.5)*70*math.pi/180
        self.rand_z=(np.random.rand()-0.5)*70*math.pi/180
        self.rotate_y[0,0]=math.cos(self.rand)
        self.rotate_y[1,1]=1
        self.rotate_y[0,2]=math.sin(self.rand)
        self.rotate_y[2,0]=-math.sin(self.rand)
        self.rotate_y[2,2]=math.cos(self.rand)

        self.rotate_x[1,1]=math.cos(self.rand)
        self.rotate_x[0,0]=1
        self.rotate_x[2,1]=math.sin(self.rand)
        self.rotate_x[1,2]=-math.sin(self.rand)
        self.rotate_x[2,2]=math.cos(self.rand)

        self.rotate_z[0,0]=math.cos(self.rand)
        self.rotate_z[2,2]=1
        self.rotate_z[1,0]=math.sin(self.rand)
        self.rotate_z[0,1]=-math.sin(self.rand)
        self.rotate_z[1,1]=math.cos(self.rand)
        tensor=torch.Tensor(tensor)
        self.rotate=torch.matmul(self.rotate_z,torch.matmul(self.rotate_y,self.rotate_x))
        _,n,t=tensor.shape
        return torch.matmul(self.rotate,tensor.contiguous().view(3,-1)).view(3,n,t)



