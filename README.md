# 多分类模版

目录结构如下:
```
project
|   README.md
|   requirements.txt
|   main.py     程序主流程
|   trainer.py  程序训练框架
|
|---data
|   |   保存各个数据集及其接口api文件
|   |   get_data.sh 获取数据
|
|---datasets
|   |   dataset.py  数据集处理
|
|---checkpoint  保存中间结果
|
|---models
|   |   NYS.py  自己写的简单CNN模型
|   |   torchModel_modify.py  修改官方torchvision中的模型
|
|---runs
|   | tensorboad的默认存储文件夹
|
|---log
|   | 记录程序训练的log
```

## get start with tiny-imagenet-200

```bash
cd data/tiny-imagenet/
bash get_data.sh
unzip tiny-imagenet-200.zip
rm tiny-imagenet-200.zip
cd ../../
python main.py
```