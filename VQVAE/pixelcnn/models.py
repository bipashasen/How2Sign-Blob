
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions.normal import Normal
from torch.distributions import kl_divergence

from tqdm import tqdm

def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        try:
            nn.init.xavier_uniform_(m.weight.data)
            m.bias.data.fill_(0)
        except AttributeError:
            print("Skipping initialization of ", classname)



class GatedActivation(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        x, y = x.chunk(2, dim=1)
        return F.tanh(x) * F.sigmoid(y)


class GatedMaskedConv2d(nn.Module):
    def __init__(self, mask_type, dim, kernel, residual=True, n_classes=10):
        super().__init__()

        # initialized 15 times (default)
        # i=0: A, 625, 7, False, 10
        # i=1: B, 625, 3, True, 10

        assert kernel % 2 == 1, print("Kernel size must be odd")
        self.mask_type = mask_type
        self.residual = residual

        self.class_cond_embedding = nn.Embedding(
            n_classes, 2 * dim
        )

        kernel_shp = (kernel // 2 + 1, kernel)  # (ceil(n/2), n)
        padding_shp = (kernel // 2, kernel // 2)
        self.vert_stack = nn.Conv2d(
            dim, dim * 2, # 625, 1250
            kernel_shp, 1, padding_shp
        )

        self.vert_to_horiz = nn.Conv2d(2 * dim, 2 * dim, 1) # 1250, 1250

        kernel_shp = (1, kernel // 2 + 1)
        padding_shp = (0, kernel // 2)
        self.horiz_stack = nn.Conv2d(
            dim, dim * 2, # 625, 1250
            kernel_shp, 1, padding_shp
        )

        self.horiz_resid = nn.Conv2d(dim, dim, 1) # 625, 625

        self.gate = GatedActivation() # divides the channel dimension in half

    def make_causal(self):
        # both are nn.Conv2d
        self.vert_stack.weight.data[:, :, -1].zero_()  # Mask final row
        self.horiz_stack.weight.data[:, :, :, -1].zero_()  # Mask final column

    def forward(self, x_v, x_h, h):
        if self.mask_type == 'A':
            self.make_causal()

        h_vert = self.vert_stack(x_v) # B x 625 x 25 x 25 -> B x 1250 x 26 x 25
        h_vert = h_vert[:, :, :x_v.size(-1), :] # B x 1250 x 25 x 25
        out_v = self.gate(h_vert + h[:, :, None, None]) 
        # !!! here the condition is used.
        # h[:, :, None, None]: B x 1250 -> B x 1250 x 1 x 1
        # out_v: B x 625 x 25 x 25

        h_horiz = self.horiz_stack(x_h) # B x 625 x 25 x 25 -> B x 1250 x 25 x 26
        h_horiz = h_horiz[:, :, :, :x_h.size(-2)] # B x 1250 x 25 x 25
        v2h = self.vert_to_horiz(h_vert) # NOTICE INPUT! B x 1250 x 25 x 25

        out = self.gate(v2h + h_horiz + h[:, :, None, None])
        # !!! here the condition is used again.
        # out: B x 625 x 25 x 25
        # hvert -> v2h + h_horiz + h

        if self.residual:
            out_h = self.horiz_resid(out) + x_h
        else:
            out_h = self.horiz_resid(out) # B x 625 x 25 x 25

        return out_v, out_h


class GatedPixelCNN(nn.Module):
    def __init__(self, input_dim=256, dim=64, n_layers=15, n_classes=10):
        super().__init__()
        self.dim = dim
        self.n_layers = n_layers

        # Create embedding layer to embed input
        self.embedding = nn.Embedding(input_dim, dim)

        # Building the PixelCNN layer by layer
        self.layers = nn.ModuleList()

        # Initial block with Mask-A convolution
        # Rest with Mask-B convolutions
        for i in range(n_layers): # n_layers: 15
            mask_type = 'A' if i == 0 else 'B'
            kernel = 7 if i == 0 else 3
            residual = False if i == 0 else True
            # i=0: A, 625, 7, False, 10
            # i=1: B, 625, 3, True, 10
            self.layers.append(
                GatedMaskedConv2d(mask_type, dim, kernel, residual, n_classes)
            )

        # Add the output layer
        self.output_conv = nn.Sequential(
            nn.Conv2d(dim, 512, 1), # 625 -> 512 
            nn.ReLU(True),
            nn.Conv2d(512, input_dim, 1) # 512 -> 256
        )

        self.apply(weights_init)

    def forward(self, x, label):
        shp = x.size() + (-1, )
        x = self.embedding(x.view(-1)).view(shp)  # (B, H, W, C)
        x = x.permute(0, 3, 1, 2)  # (B, C, W, H)

        x_v, x_h = (x, x)
        for i, layer in enumerate(self.layers):
            x_v, x_h = layer(x_v, x_h, label)
            # if i < self.n_layers-10:
            #     x_v = x_v + x
            #     x_h = x_h + x

        out = self.output_conv(x_h)

        return out