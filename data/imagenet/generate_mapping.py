# 读取解析'devkit/data/meta.mat'文件
import scipy.io as sio
import os

rootdir = '/media/raid0/ImageNet/'
metafile = 'devkit/data/meta.mat'

data = sio.loadmat(os.path.join(rootdir,metafile))

print(type(data)) # dict

print(data.keys()) # ['__header__', '__version__', '__globals__', 'synsets']

print(type(data['synsets'])) # ndarray

print(len(data['synsets'])) # 1086

for d in data['synsets']:
    print(d)
    print(type(d)) # 'numpy.ndarray'
    print(d[0][0],d[0][1]) # [[1]] ['n02119789'] 即这里是label和文件夹名称的对应
    break