import torch
import numpy as np
import torch.utils.data as data
import math
import h5py
import os
import random

class ntu(data.Dataset):
    def __init__(self,mode,seq_length,rotate,version):
        super(type(self),self).__init__()
        self.mode=mode
        assert self.mode in ['train','test'],'mode undefined'
        self.seq_length=seq_length
        self.rotate=rotate
        self.version=version
        assert self.version in ['subj_seq','view_seq'],'version undefined'
        self.h5='/home/zf/datasets/ntu/data/{}/array_list_{}.h5'.format(self.version,self.mode)
        self.list='/home/zf/datasets/ntu/data/{}/file_list_{}.txt'.format(self.version,self.mode)
        with open(self.list,'r') as f:
            keys=f.readlines()
        self.keys=[i.strip('\n') for i in keys]
        self.data=h5py.File(self.h5,'r')
    def __getitem__(self,index):
        data=self.data[self.keys[index]][:]
        data=self.sample(data)
        '''
        data=self.subtract_mean(data,smooth=True,scale=True)
        '''
        data=data.astype(np.float32)
        label=int(self.keys[index][17:20])-1
        return data,label
    def __len__(self):
        return len(self.keys)

    def sample(self,data):
        length,joints,_=data.shape
        if length>=self.seq_length:
            index=np.random.choice(length,self.seq_length,replace=False)
            index.sort()
            data=data[index]
        else:
            data=np.concatenate((data,np.zeros((self.seq_length-length,joints,3))),axis=0)
        return data

    def calculate_height(self, data):
        center1 = (data[:,2,:] + data[:,8,:] + data[:,4,:] + data[:,20,:])/4
        w1 = data[:,23,:] - center1
        w2 = data[:,22,:] - center1
        center2 = (data[:,1,:] + data[:,0,:] + data[:,16,:] + data[:,12,:])/4
        h0 = data[:,3,:] - center2
        h1 = data[:,19,:] - center2
        h2 = data[:,15,:] - center2
        width = np.max([np.max(np.abs(w1[:,0])), np.max(np.abs(w2[:,0]))])
        heigh1 = np.max(h0[:,1])
        heigh2 = np.max([np.max(np.abs(h1[:,1])), np.max(np.abs(h2[:,1]))])
        return width,max(heigh1,heigh2)
        
    def smooth_data(self, data):
        assert(data.shape[2] == 3), ' input must be data array'
        filt = np.array([-3,12,17,12,-3])/35.0
        skt = np.concatenate((data[0:2], data, data[-2:]), axis=0)
        for idx in range(2, skt.shape[0]-2):
            data[idx-2] = np.swapaxes(np.dot(np.swapaxes(skt[idx-2:idx+3], 0, -1), filt), 0, -1)
        return data

    def subtract_mean(self, data, smooth=False, scale=True):
        if smooth:
            data = self.smooth_data(data)
        # substract mean values
        # notice: use two different mean values to normalize data data
        center1 = (data[:,2,:] + data[:,8,:] + data[:,4,:] + data[:,20,:])/4
        center2 = (data[:,1,:] + data[:,0,:] + data[:,16,:] + data[:,12,:])/4

        for idx in [24,25,12,11,10,9, 5,6,7,8,23,22]:
            data[:, idx-1] = data[:, idx-1] - center1
        for idx in (set(range(1, 1+data.shape[1]))-set([24,25,12,11,10,9,  5,6,7,8,23,22])):
            data[:, idx-1] = data[:, idx-1] - center2

        if scale:
            width, heigh = self.calculate_height(data)
            scale_factor1, scale_factor2 = 0.36026082, 0.61363413
            data[:,:,0] = scale_factor1*data[:,:,0]/width
            data[:,:,1] = scale_factor2*data[:,:,1]/heigh
        return data

    def rand_view_transform(self, X, angle1=-10, angle2=10, s1=0.9, s2=1.1):
        # s00keleton data X, tensor3
        # genearte rand matrix
        random.random()
        agx = random.randint(angle1, angle2)
        agy = random.randint(angle1, angle2)
        s = random.uniform(s1, s2)
        agx = math.radians(agx)
        agy = math.radians(agy)
        Rx = np.asarray([[1,0,0], [0,math.cos(agx),math.sin(agx)], [0, -math.sin(agx),math.cos(agx)]])
        Ry = np.asarray([[math.cos(agy), 0, -math.sin(agy)], [0,1,0], [math.sin(agy), 0, math.cos(agy)]])
        Ss = np.asarray([[s,0,0],[0,s,0],[0,0,s]])
        # X0 = np.reshape(X,(-1,3))*Ry*Rx*Ss
        X0 = np.dot(np.reshape(X,(-1,3)), np.dot(Ry,np.dot(Rx,Ss)))
        X = np.reshape(X0, X.shape)
        X = X.astype(np.float32)
        return X


