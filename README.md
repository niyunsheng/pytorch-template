# 多分类模版

目录结构如下:
```
project
|   README.md
|   requirements.txt
|   config.py   程序配置均在这里
|   main.py     程序主流程
|   trainer.py  程序训练框架
|
|---data
|   |   保存各个数据集及其接口api文件
|
|---datasets
|   |   dataset.py  数据集处理
|   |   get_data.sh 获取数据
|   
|---checkpoint  保存中间结果
|
|---models
|   |   NYS.py  自己写的简单CNN模型
|   |   VGG16.py    修改官方VGG16适用于MNIST数据集
|
|---runs
|   | tensorboad的默认存储文件夹
|
|---log
|   | 记录程序训练的log
```