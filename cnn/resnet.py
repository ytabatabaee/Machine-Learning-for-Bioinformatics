from torch import nn

class ResidualBlock(nn.Module):
    def __init__(self, in_c, out_c, k, s, bn_flag, dp_flag):
        super(ResidualBlock, self).__init__()
        layers = []
        ds_layers = []
        layers += [nn.Conv2d(in_c, out_c, kernel_size=k, stride=s, padding = (k-s+2)//2, bias=True)]
        if bn_flag:
          layers += [nn.BatchNorm2d(out_c)]   
        layers +=  [nn.ReLU()]
        if dp_flag:
          layers += [nn.Dropout(p = 0.3)]
        layers += [nn.Conv2d(out_c, out_c, kernel_size=k, stride=1, padding = (k-s+2)//2, bias=True)]
        if bn_flag:
          layers += [nn.BatchNorm2d(out_c)]  
        layers +=  [nn.ReLU()]
        if dp_flag:
          layers += [nn.Dropout(p = 0.3)]
        
        ds_layers += [nn.Conv2d(in_c, out_c, kernel_size=1, stride=s, padding=(1-s+2)//2, bias=True)]
        if bn_flag:
            ds_layers += [nn.BatchNorm2d(out_c)]

        self.net = nn.Sequential(*layers)
        self.downsample = nn.Sequential(*ds_layers)

    def forward(self, x):
        o = self.net(x)
        o += self.downsample(x)
        return o


class ResNet(nn.Module):
    def __init__(self, num_classes, bn_flag, dp_flag):
        super(ResNet, self).__init__()
        self.block1 = ResidualBlock(in_c=1, out_c=8, k=7, s=3, bn_flag=bn_flag, dp_flag=dp_flag)
        self.block2 = ResidualBlock(in_c=8, out_c=16, k=5, s=3, bn_flag=bn_flag, dp_flag=dp_flag)
        self.block3 = ResidualBlock(in_c=16, out_c=32, k=3, s=3, bn_flag=bn_flag, dp_flag=dp_flag)
        self.fc = nn.Linear(2592 , num_classes, bias=True)
        self.softmax = nn.Softmax(dim=1)


    def forward(self, x):
        o = self.block1(x) 
        o = self.block2(o)  
        o = self.block3(o)  
        o = o.view(-1, 2592) 
        o = self.fc(o)
        o = self.softmax(o)
        return o
