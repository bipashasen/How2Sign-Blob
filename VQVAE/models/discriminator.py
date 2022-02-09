import torch.nn as nn

class discriminator(nn.Module):
    def __init__(self):
        super(discriminator, self).__init__()
        k = 64
        
        self.model = nn.Sequential(
            # C64 -- 48
            nn.Conv2d(in_channels=3, out_channels=k, kernel_size=4, stride=2), # in_channel, out_channel, kernel, stride, padding 
            nn.LeakyReLU(0.2),
            # C128 -- 23
            nn.Conv2d(in_channels=k, out_channels=k*2, kernel_size=4, stride=2),
            nn.InstanceNorm2d(k*2, affine=True),
            nn.LeakyReLU(0.2),
            # C256 -- 20
            nn.Conv2d(in_channels=k*2, out_channels=k*4, kernel_size=4),
            nn.InstanceNorm2d(k*4, affine=True),
            nn.LeakyReLU(0.2),
            # C512 -- 18
            nn.Conv2d(in_channels=k*4, out_channels=k*8, kernel_size=3),
            nn.InstanceNorm2d(k*8, affine=True),
            nn.LeakyReLU(0.2),
            # C1 -- 16 
            nn.Conv2d(in_channels=k*8, out_channels=1, kernel_size=3),
        )
        
    def forward(self, x):
        return self.model(x)