import torch 
from torch.utils.data import DataLoader
import numpy as np
import os
import argparse

from transform import rotate
from model import stacked_rnn,stacked_EleAttg
from utils.logger import Logger
from utils.utils import *
from dataset import ntu

def main(args):
    if args.mode=='xsub':
        train_data=DataLoader(ntu(mode='train',seq_length=100,rotate=True,version='subj_seq'),batch_size=256,shuffle=True)
        val_data=DataLoader(ntu(mode='test',seq_length=100,rotate=True,version='subj_seq'),batch_size=256,shuffle=True)
    else:
        train_data=DataLoader(ntu(mode='train',seq_length=100,rotate=True,version='view_seq'),batch_size=256,shuffle=True)
        val_data=DataLoader(ntu(mode='test',seq_length=100,rotate=True,version='view_seq'),batch_size=256,shuffle=True)
    rnn=stacked_EleAttg(input_size=75,hidden_size=100,num_class=60).cuda()
    save_path='./save/'+str(args.mode)+'/'+str(args.model)
    train_log_path='./log/'+str(args.mode)+'/'+str(args.model)+'/train'
    val_log_path='./log/'+str(args.mode)+'/'+str(args.model)+'/val'
    if not os.path.exists(train_log_path):
        os.makedirs(train_log_path)
    if not os.path.exists(val_log_path):
        os.makedirs(val_log_path)
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    start=0
    cross_entropy=torch.nn.CrossEntropyLoss()
    optimizer=torch.optim.Adam(rnn.parameters(),lr=args.lr,weight_decay=5e-4)

    if args.restore is not None:
        ckpt=torch.load(args.restore)
        rnn.load_state_dict(ckpt['state_dict'])
        start=ckpt['epoch']
        optimizer.load_state_dict(ckpt['optim_state'])
    train_logger=Logger(train_log_path)
    val_logger=Logger(val_log_path)

    for epoch in range(start,args.epoch):
        lr=args.lr*0.1**(np.sum(epoch>np.array([40,60])))
        for i,(data,label) in enumerate(train_data):
            data=data.cuda()
            label=label.cuda()
            batch_size=data.shape[0]
            rt_h=torch.zeros(batch_size,75).cuda()
            dt_h=torch.zeros(batch_size,75).cuda()
            h1=torch.zeros(batch_size,100).cuda()
            h2=torch.zeros(batch_size,100).cuda()
            h3=torch.zeros(batch_size,100).cuda()
            logits=rnn(data,rt_h,dt_h,h1,h2,h3,mode='train')
            optimizer.zero_grad()
            loss=cross_entropy(logits,label)
            loss.backward()
            optimizer.step()
            train_logger.scalar_summary('train_loss',loss.data.cpu().numpy(),epoch*len(train_data)+i)
            if i%30==0:
                for name , value  in rnn.named_parameters():
                    train_logger.histo_summary(name+'/weight',value.data.cpu().numpy(),step=epoch*len(train_data)+i)
                    train_logger.histo_summary(name+'/grad',value.grad.cpu().numpy(),step=epoch*len(train_data)+i)

            if i%30==0:
                with torch.no_grad():
                    val_loss=0
                    acc=0
                    for val_step,(data,label) in enumerate(val_data):
                        if val_step>10:
                            break
                        data=data.cuda()
                        label=label.cuda()
                        batch_size=data.shape[0]
                        rt_h=torch.zeros(batch_size,75).cuda()
                        dt_h=torch.zeros(batch_size,75).cuda()
                        h1=torch.zeros(batch_size,100).cuda()
                        h2=torch.zeros(batch_size,100).cuda()
                        h3=torch.zeros(batch_size,100).cuda()
                        logits=rnn(data,rt_h,dt_h,h1,h2,h3,mode='val')
                        val_loss+=cross_entropy(logits,label).data.cpu().numpy()
                        acc+=np.sum(logits.topk(1)[1].cpu().numpy().reshape(args.batch_size)==label.cpu().numpy())
                    val_loss/=val_step
                    acc/=(val_step*args.batch_size)
                    acc*=100
                msg='{} epoch:{} iter train_loss={:.5f} val_loss={:.5f} val_acc={:.3f}%'
                print(msg.format(epoch,i,loss,val_loss,acc))

            val_logger.scalar_summary('val_loss',val_loss,epoch*len(train_data)+i)
            val_logger.scalar_summary('acc',acc,epoch*len(train_data)+i)

        torch.save({
            'state_dict':rnn.state_dict(),
            'epoch':epoch,
            'optim_state':optimizer.state_dict()},save_path+'/{}.pth'.format(epoch))
    print('done!')
        

if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--model',type=str,default='stacked_EleAttg_3_softmax')
    parser.add_argument('--seq_length',type=int,default=100)
    parser.add_argument('--mode',type=str,default='xsub')
    parser.add_argument('--batch_size',type=int,default=256)
    parser.add_argument('--lr',type=float,default=0.005)
    parser.add_argument('--restore',type=str,default=None)
    parser.add_argument('--epoch',type=int,default=200)


    args=parser.parse_args()
    main(args)
