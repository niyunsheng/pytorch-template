from torch.utils import data
import torch
from torchvision import transforms
from PIL import Image
import pandas as pd
import os

import Augmentor
from torch.nn import functional as F

# 最为一般的dataset，读取图片的根目录和index文件，其中index文件包含了相对路径和标签

class CommomDataset(data.Dataset):
    def __init__(self,root,indexfile,transform=None):
        super(CommomDataset,self).__init__()
        self.root = root
        self.indexfile = indexfile
        self.transform = transform
        self.imgs = pd.read_csv(os.path.join(root,indexfile),header = None).values # type:ndarray
        self.class_num = 10

    def __getitem__(self,idx):
        img_path = os.path.join(self.root,self.imgs[idx][0])
        data = Image.open(img_path)
        if self.transform is not None:
            data = self.transform(data)
        label = self.imgs[idx][1] # int
        return data,label
    def __len__(self):
        return len(self.imgs)

if __name__ == '__main__':
    root = '/media/raid0/ImageNet'
    indexfile = 'ilsvrc12_val.txt'
    p = Augmentor.Pipeline()
    p.random_contrast(probability=0.5,min_factor=0.6,max_factor=1.4)
    train_dataset = CommomDataset(root,indexfile,transforms.Compose([
        p.torch_transform(),
        transforms.Resize((28,28)),
        transforms.ToTensor()
    ]))
    batch_size = 64
    trainloader = data.DataLoader(train_dataset,batch_size = batch_size,
        shuffle = True,
        num_workers = 8
    )
    class_num = 10
    for i,(data,label) in enumerate(trainloader):
        data = data.cuda()
        # label = F.one_hot(label,num_classes=class_num)
        label = label.cuda()
        print('data',data.shape)
        print('label',label.shape)
        print('label',label)
        break

