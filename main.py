import argparse, os, time, logging
import Augmentor
from torchvision import transforms
from torch.utils.data import DataLoader
from torch import nn
from torch import optim

from models.NYS import NYS
from datasets.dataset import CommomDataset
from trainer import Trainer
from utils import get_model

def main(opt):
    if not os.path.exists(args.log_dir): 
        os.mkdir(args.log_dir)
    logfile = '{}/{}.log'.format(args.log_dir, time.strftime("%Y_%m_%d", time.localtime()))
    logging.basicConfig(filename=logfile, level=logging.DEBUG)
    logger = logging.getLogger(__name__)

    # model
    model = get_model(args.modelname)
    # train_loader

    # p = Augmentor.Pipeline()
    # p.random_contrast(probability=0.5,min_factor=0.6,max_factor=1.4)

    train_dataset = CommomDataset(args.train_data_root,args.train_data_index,transforms.Compose([
        # p.torch_transform(),
        transforms.Resize((28,28)),
        transforms.ToTensor()
    ]))
    trainloader = DataLoader(train_dataset,batch_size = args.batch_size,
        shuffle = args.shuffle,
        num_workers = args.num_workers
    )
    # test_loader
    test_dataset = CommomDataset(args.test_data_root,args.test_data_index,transforms.Compose([
        transforms.Resize((28,28)),
        transforms.ToTensor()
    ]))
    testloader = DataLoader(test_dataset,batch_size = args.batch_size,
        shuffle = args.shuffle,
        num_workers = args.num_workers
    )
    # loss，一般分类模型最后一层不加sigmoid，因为nn.CrossEntropyLoss()已经包含了sigmoid操作
    loss_fn = nn.CrossEntropyLoss()

    # optimizer
    optimizer = optim.Adam(params=model.parameters(),lr=args.lr,weight_decay=args.weight_decay)
    # trainer
    trainer = Trainer(train_loader=trainloader, val_loader=testloader, model=model, loss_fn=loss_fn, optimizer=optimizer, load_model_path=args.load_model_path, log_interval=args.log_interval)

    for epoch_idx in range(trainer.start_epoch,args.end_epoch):
        print('-'*12,'epoch ',epoch_idx,'-'*12)
        trainer.train_epoch(epoch_idx)
        trainer.test_epoch(epoch_idx)

if __name__ == '__main__':
    parser = argparse.ArgumentParser("PyTorch Pipeline")
    arg = parser.add_argument
    arg('--train-data-root', type=str, default='data/tiny-imagenet/tiny-imagenet-200')
    arg('--train-data-index', type=str, default='tinyimagenet_train.csv')
    arg('--test-data-root', type=str, default='data/tiny-imagenet/tiny-imagenet-200')
    arg('--test-data-index', type=str, default='tinyimagenet_val.csv')
    arg('--log-dir', type=str, default='log')
    arg('--gpus', type=str, default='')

    arg('--batch-size', type=int, default=32)
    arg('--num-workers', type=int, default=8)
    arg('--shuffle', type=bool, default=True)
    arg('--lr', type=float, default=0.001)
    arg('--lr-decay', type=float, default=0.95)
    arg('--weight-decay', type=float, default=0)
    arg('--log_interval', type=int, default=20)

    arg("--best-acc", type=float, default=0)
    arg("--start-epoch", type=int, default=0)
    arg("--end-epoch", type=int, default=10)
    arg("--load-model-path", type=str, default=None)

    arg("--modelname", type=str, default="nys")

    arg("--seed", type=int, default=777)
    args = parser.parse_args()
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpus
    main(args)