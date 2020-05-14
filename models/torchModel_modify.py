from torch import nn
import torch
import torchvision

class VGG16(nn.Module):
    def __init__(self):
        super(VGG16,self).__init__()
        self.name = 'vgg16'
        self.net = torchvision.models.vgg16(num_classes=10,pretrained=False)
        self.net.features[0] = nn.Conv2d(1,64,3,padding=1)
        nn.init.normal_(self.net.features[0].weight,mean=0,std=0.01)
    def forward(self,x):
        out = self.net(x)
        out = torch.sigmoid(out)
        return out
if __name__ == "__main__":
    model = VGG16()
    print(model.net.features[0])
    print(model.net.classifier[-1])
    print(model)