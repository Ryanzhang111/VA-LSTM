from dataset import ntu

data=ntu('train',100,rotate=True,version='subj_seq')
print(data.__getitem__(0)[0].shape)
