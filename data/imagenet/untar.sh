#!/bin/bash
# 将该文件放在imagenet文件夹下执行即可解压所有压缩包
for element in `ls *tar`
do  
    dir_or_file=$element
    if [ -d $dir_or_file ]
    then 
        echo "pass"
    else
        mkdir ${dir_or_file%.*}
        tar -xvf $dir_or_file -C ${dir_or_file%.*}
    fi  
done

for element in `ls ILSVRC2012_img_train/`
do  
    dir_or_file="ILSVRC2012_img_train/"$element
    if [ -d $dir_or_file ]
    then 
        echo "pass"
    else
        mkdir ${dir_or_file%.*}
        tar -xvf $dir_or_file -C ${dir_or_file%.*}
    fi  
done