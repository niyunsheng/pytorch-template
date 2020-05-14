# torchvision自带数据集
from torchvision import datasets
from torch.utils import data
from torchvision import transforms

if __name__=="__main__":
    root = '../data'
    # MNIST
    # dataset = datasets.MNIST(root, train=True, 
    #                 transform=transforms.Compose([
    #                     transforms.Resize((28,28)),
    #                     transforms.ToTensor()
    #                 ]))

    # CIFAR10
    dataset = datasets.CIFAR10(root, train=True, 
                    transform=transforms.Compose([
                        transforms.Resize((28,28)),
                        transforms.ToTensor()
                    ]),download=True)

    dataloader = data.DataLoader(dataset,batch_size = 8,
        shuffle = True,
        num_workers = 8
    )

    for i,(data,label) in enumerate(dataloader):
        print('data',data.shape) # [8, 1, 28, 28]
        print('label',label.shape) # [8]
        print('label',label) # [2, 6, 1, 7, 2, 8, 1, 6]
        break