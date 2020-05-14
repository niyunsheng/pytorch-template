import os

import Augmentor
from torchvision import transforms
from torch.utils.data import DataLoader
from torch import nn
from torch import optim

from models.NYS import NYS
from datasets.dataset import MNIST
from config import myConfig
from trainer import Trainer




def main(opt):

    if not os.path.exists(opt.log_dir): 
        os.mkdir(opt.log_dir)
    # model
    model = NYS()
    # model = VGG16()
    # train_loader

    p = Augmentor.Pipeline()
    p.random_contrast(probability=0.5,min_factor=0.6,max_factor=1.4)

    train_dataset = MNIST(opt.train_data_root,opt.train_data_index,transforms.Compose([
        p.torch_transform(),
        transforms.Resize((28,28)),
        transforms.ToTensor()
    ]))
    trainloader = DataLoader(train_dataset,batch_size = opt.batch_size,
        shuffle = opt.shuffle,
        num_workers = opt.num_workers
    )
    # test_loader
    test_dataset = MNIST(opt.test_data_root,opt.test_data_index,transforms.Compose([
        transforms.Resize((28,28)),
        transforms.ToTensor()
    ]))
    testloader = DataLoader(test_dataset,batch_size = opt.batch_size,
        shuffle = opt.shuffle,
        num_workers = opt.num_workers
    )
    # loss，一般分类模型最后一层不加sigmoid，因为nn.CrossEntropyLoss()已经包含了sigmoid操作
    loss_fn = nn.CrossEntropyLoss()

    # optimizer
    optimizer = optim.Adam(params=model.parameters(),lr=opt.lr,weight_decay=opt.weight_decay)
    # trainer
    trainer = Trainer(train_loader=trainloader, val_loader=testloader, model=model, loss_fn=loss_fn, optimizer=optimizer, cuda=opt.cuda, load_model_path=opt.load_model_path, log_interval=opt.log_interval)

    for epoch_idx in range(trainer.start_epoch,opt.n_epoch):
        print('--------------------epoch ',epoch_idx)
        trainer.train_epoch(epoch_idx)
        trainer.test_epoch(epoch_idx)

if __name__ == '__main__':
    opt = myConfig()
    main(opt)