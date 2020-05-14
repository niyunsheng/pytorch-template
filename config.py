import warnings
class myConfig(object):
    train_data_root = '/home/nys/commonData/MNIST/images'
    train_data_index = 'train.csv'
    test_data_root = '/home/nys/commonData/MNIST/images'
    test_data_index = 'test.csv'

    log_dir = 'log'
    class_num = 10

    load_model_path = None #'./checkpoint/NYS_401.pth' # 为None时表示不加载

    batch_size = 64
    cuda = True
    num_workers = 4 # dataloader 的线程数
    shuffle = False
    log_interval = 20
    
    lr = 1e-4
    lr_decay = 0.95
    weight_decay = 0

    best_acc = 0
    start_epoch = 0
    n_epoch = 500

    def parse(self,kwargs):
        for k,v in kwargs.items():
            if not hasattr(self,k):
                warnings.warn('warning: opt has not attribute {}'.format(k))
            setattr(self,k,v)

if __name__ == '__main__':
    opt = myConfig()
    new_config = {'lr':1e-1,'device':'cpu'}
    print(opt.lr)
    opt.parse(new_config)
    print(opt.lr)