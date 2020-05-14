# 生成imagenet的数据和标签列表
import pandas as pd
import os
from tqdm import trange

image_root = '/media/raid0/ImageNet'
train_dir = 'ILSVRC2012_img_train'
val_dir = 'ILSVRC2012_img_val'
ilsvrc12_train_file = 'ilsvrc12_train.txt'
ilsvrc12_val_file = 'ilsvrc12_val.txt'
# 写入train文件
train_map = pd.read_csv('./ILSVRC2012_mapping.txt',header=None,sep=' ')
train = []
for i in trange(1000):
    label,fold = train_map.loc[i][0],train_map.loc[i][1]
    subdir = os.path.join(image_root,train_dir,fold)
    for img in os.listdir(subdir):
        train.append([os.path.join(train_dir,fold,img),label])
csv_path = os.path.join(image_root,ilsvrc12_train_file)
pd.DataFrame(columns=None,data=train).to_csv(csv_path,mode='w', header=False,index=False)

# 写入val文件
val_map = pd.read_csv('ILSVRC2012_validation_ground_truth.txt',header=None)
val = []
subdir = os.path.join(image_root,val_dir)
for i in trange(50000):
    img = 'ILSVRC2012_val_000{:0>5}.JPEG'.format(i+1)
    label = val_map[0][i]
    val.append([os.path.join(val_dir,img),label])
csv_path = os.path.join(image_root,ilsvrc12_val_file)
pd.DataFrame(columns=None,data=val).to_csv(csv_path,mode='w', header=False,index=False)