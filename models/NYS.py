from torch import nn

class NYS(nn.Module):
    def __init__(self, class_num=1000):
        self.name = 'NYS'
        super(NYS,self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3,8,3,padding=1),
            nn.LeakyReLU(),
            nn.Conv2d(8,16,3,padding=1),
            nn.LeakyReLU()
        )
        self.classifier = nn.Sequential(
            nn.Linear(28*28*16,256),
            nn.ReLU(),
            nn.Linear(256, class_num),
        )

    def forward(self,x):
        out = self.features(x)
        out = out.view(out.size(0),-1)
        out = self.classifier(out)
        return out