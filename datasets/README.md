# imagenet

从2010年开始,每年举办的ILSVRC图像分类和目标检测大赛。Imagenet数据集是目前深度学习图像领域应用得非常多的一个领域，关于图像分类、定位、检测等研究工作大多基于此数据集展开。Imagenet数据集文档详细，有专门的团队维护，使用非常方便，在计算机视觉领域研究论文中应用非常广，几乎成为了目前深度学习图像领域算法性能检验的“标准”数据集。Imagenet数据集有1400多万幅图片，涵盖2万多个类别；其中有超过百万的图片有明确的类别标注和图像中物体位置的标注。

* Task 1: Classification

* Task 2: Classification with localization

* Task 3: Fine-grained classification

[ImageNet官网地址](http://www.image-net.org/signup.php?next=download-images)

训练数据集和验证数据集分别是两个tar文件ILSVRC2012_img_train.tar和ILSVRC2012_img_val.tar。将这两个文件拷贝至服务器合适的地址中（如/dataset/imagenet），对着两个文件分别解压到当前目录下。
ILSVRC2012_img_train.tar解压后是1000个tar文件，每个tar文件表示1000个分类中的一个类。需要对这1000个tar文件再次解压，最后得到1000个文件夹。每个文件夹中是该类的图片。ILSVRC2012_img_val.tar解压后的文件夹包含了所有的验证集图片。

下载的文件如下：
* `ILSVRC2012_img_train`包含1000个文件夹，每个文件夹下面有1000张图片
* `ILSVRC2012_img_val.tar`解压后只有一个文件夹，里面有50000张图片
* `ILSVRC2012_img_test.tar`图片没有标签
* `ILSVRC2012_devkit_t12.tar`任务1和任务2的开发包，开发包提供的是matlib程序
  * data/ILSVRC2012_validation_ground_truth.txt 验证集的类别信息，共50000个类别信息
  * data/meta.mat matlib数据，包含各个类别信息到数字的映射，做分类时可以将此数据提取出来做为mapping信息，这个提取出来的文件`ILSVRC2012_mapping.txt`放在data文件夹下，分类任务时可用。
* `ILSVRC2012_devkit_t3.tar`任务3的开发包
* `ILSVRC2012_bbox_train_v2.tar`

在进行imagenet的分类任务训练时，还是应该首先用`ILSVRC2012_mapping.txt`文件和文件夹的图片信息生成标准的csv然后再用CommonDataset进行读取。
* `data/imagenet/generate_mapping.py`生成mapping文件`ILSVRC2012_mapping.txt`
* `data/imagenet/generate_train_csv.py`生成imagenet的训练和测试文件


# tiny-imagenet-200

预处理得到训练和测试的csv，然后用CommonDataset进行读取。

# MNIST 和 CIFAR10

采用torchvision官方接口

`torchvision.datasets.MNIST(root, train=True, transform=None, target_transform=None, download=False)`

`torchvision.datasets.CIFAR10(root, train=True, transform=None, target_transform=None, download=False)`