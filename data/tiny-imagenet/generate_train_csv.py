# 生成imagenet的数据和标签列表
import pandas as pd
import os
from tqdm import trange

image_root = '/home/nys/tiny-imagenet-200'
train_dir = 'train'
val_dir = 'val'
# 要写的文件
train_file = 'tinyimagenet_train.txt'
val_file = 'tinyimagenet_val.txt'
# 写入train文件
name_list = list(pd.read_csv(os.path.join(image_root,'wnids.txt'),header=None)[0])
name_dict = {k:v for k,v in zip(name_list,range(200))}
# print(name_dict)
# print(name_map)
train = []
for k,v in name_dict.items():
    # print(k,v,type(k))
    subdir = os.path.join(image_root,train_dir,k,'images')
    for img in os.listdir(subdir):
        train.append([os.path.join(train_dir,k,'images',img),v])
        # print([os.path.join(train_dir,k,'images',img),v])
csv_path = os.path.join(image_root,train_file)
pd.DataFrame(columns=None,data=train).to_csv(csv_path,mode='w', header=False,index=False)

# 写入val文件
val_map = pd.read_csv(os.path.join(image_root,'val/val_annotations.txt'),header=None,sep='\t')

val = []
for i in range(len(val_map)):
    # print(val_map[0][i],val_map[1][i],name_dict[val_map[1][i]])
    img = val_map[0][i]
    label = name_dict[val_map[1][i]]
    # print(img,label,type(label))
    val.append([os.path.join(val_dir,'images',img),label])
    # print([os.path.join(val_dir,'images',img),label])
csv_path = os.path.join(image_root,val_file)
pd.DataFrame(columns=None,data=val).to_csv(csv_path,mode='w', header=False,index=False)