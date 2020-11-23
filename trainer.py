from tqdm import tqdm
import numpy as np
import torch
import os

# tensorboardX
# from tensorboardX import SummaryWriter

class Trainer(object):
    def __init__(self,train_loader=None, val_loader=None, model=None, loss_fn=None, optimizer=None, scheduler=None, load_model_path=None, log_interval=20):
        self.train_loader=train_loader
        self.val_loader=val_loader
        self.model=model
        self.loss_fn=loss_fn
        self.optimizer=optimizer
        self.scheduler=scheduler
        self.load_model_path = load_model_path
        self.log_interval=log_interval
        self.best_acc = 0
        self.start_epoch = 0
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.modelname = model.name
        # self.writer = SummaryWriter()

        if self.load_model_path is not None:
            print('==> Resuming from checkpoint..')
            checkpoint = torch.load(self.load_model_path)
            self.model.load_state_dict(checkpoint['net'])
            self.best_acc = checkpoint['acc']
            self.start_epoch = checkpoint['epoch']
            print(self.best_acc)
        
        if torch.cuda.is_available():
            if torch.cuda.device_count() > 1:
                self.model = torch.nn.DataParallel(self.model)
            self.model = self.model.cuda()

    def train_epoch(self,epoch):
        self.model.train()
        if self.scheduler:
            self.scheduler.step()
        losses = []
        total_loss = 0
        correct = 0
        total = 0
        for batch_idx,(data,label) in enumerate(self.train_loader):
            data = data.to(self.device)
            label = label.to(self.device)
            # 梯度归零
            self.optimizer.zero_grad()
            # 前向传播
            outputs = self.model(data)
            _, predicted = outputs.max(1)
            total += label.size(0)
            correct += predicted.eq(label.view_as(predicted)).sum().item()
            # 计算损失
            loss = self.loss_fn(outputs,label)
            losses.append(loss.item())
            total_loss += loss.item()
            # 反向传播
            loss.backward()
            self.optimizer.step()
            # 打印信息
            if batch_idx % self.log_interval == 0:
                message = 'Train: [{}/{} ({:.0f}%)]\tAccuracy: {:.6f}%\tLoss: {:.6f}'.format(
                        total, len(self.train_loader.dataset),
                        100. * batch_idx / len(self.train_loader), 
                        100.*correct/total,
                        np.mean(losses))
                print(message)
                losses = []
        total_loss /= (batch_idx+1)
        message = 'Train: Accuracy: {:.6f}\tLoss: {:.6f}'.format(
                        100.*correct/total,
                        total_loss)
        # self.writer.add_scalar('data/train_loss',total_loss,epoch)
        # 查看网络参数
        # for name,param in self.model.named_parameters():
        #     self.writer.add_histogram(name,param,epoch)
        print(message)

    def test_epoch(self,epoch):
        self.model.eval()
        val_loss = 0
        correct = 0
        total = 0
        with torch.no_grad():
            for batch_idx,(data,label) in tqdm(enumerate(self.val_loader)):
                data = data.to(self.device)
                label = label.to(self.device)
                outputs = self.model(data)
                _, predicted = outputs.max(1)
                total += label.size(0)
                correct += predicted.eq(label.view_as(predicted)).sum().item()
                loss = self.loss_fn(outputs,label)
                val_loss += loss.item()
        message = 'Val: Accuracy: {:.6f}\tLoss: {:.6f}'.format(
                        100.*correct/total,
                        val_loss/(batch_idx+1))
        # self.writer.add_scalar('data/val_loss',val_loss/(batch_idx+1),epoch)
        print(message)
        # Save checkpoint.
        acc = 100.*correct/total
        if self.best_acc<acc:
            print('Saving..')
            if torch.cuda.device_count() > 1:
                state_dict = self.model.module.state_dict()
            else:
                state_dict = self.model.state_dict()
            state = {
                'net': state_dict,
                'acc': acc,
                'epoch':epoch
            }
            os.makedirs('checkpoint', exist_ok=True)
            os.makedirs('checkpoint/{}'.format(self.modelname), exist_ok=True)
            torch.save(state, './checkpoint/{}/{}.pth'.format(self.modelname,epoch))
            self.best_acc = acc
