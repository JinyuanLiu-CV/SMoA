import torch
import torch.nn as nn
import torch.nn.functional as F
OPS = {
    'none': lambda C, stride, affine: Zero(stride),
    'sep_conv_3x3': lambda C, stride, affine: SepConv(C, C, 3, stride, 1, affine=affine),
    'sep_conv_5x5': lambda C, stride, affine: SepConv(C, C, 5, stride, 2, affine=affine),
    'dil_conv_3x3': lambda C, stride, affine: DilConv(C, C, 3, stride, 2, 2, affine=affine),  # 2,2
    'dil_conv_5x5': lambda C, stride, affine: DilConv(C, C, 5, stride, 4, 2, affine=affine),  # 4,2
    'NonLocalattention': lambda C, stride, affine: NLBasicBlock(C, stride),
    'Spatialattention': lambda C, stride, affine: Spatial_BasicBlock(C, stride),
    'Denseblocks': lambda C, stride, affine: ResidualDenseBlock(C, stride),
    'Residualblocks': lambda C, stride, affine: ResidualModule(C, stride),
}


class BasicConv(nn.Module):
    def __init__(self, in_planes, out_planes, stride=1, dilation=1, groups=1, relu=True, bn=False,
                 bias=False):
        super(BasicConv, self).__init__()
        # judge
        # stride = 1
        padding = 0
        kernel_size = 3
        if kernel_size == 3 and dilation == 1:
            padding = 1
        if kernel_size == 3 and dilation == 2:
            padding = 2
        if kernel_size == 5 and dilation == 1:
            padding = 2
        if kernel_size == 5 and dilation == 2:
            padding = 4
        if kernel_size == 7 and dilation == 1:
            padding = 3
        if kernel_size == 7 and dilation == 2:
            padding = 6
        self.out_channels = out_planes
        self.conv = nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=padding,
                              dilation=dilation, groups=groups, bias=bias, padding_mode='reflect')
        self.bn = nn.BatchNorm2d(out_planes, eps=1e-5, momentum=0.01, affine=True) if bn else None
        self.relu = nn.PReLU() if relu else None

    def forward(self, x):
        x = self.conv(x)
        if self.bn is not None:
            x = self.bn(x)
        if self.relu is not None:
            x = self.relu(x)
        return x


class NonLocalBlock2D(nn.Module):
    def __init__(self, in_channels, inter_channels, bias=False):
        super(NonLocalBlock2D, self).__init__()

        self.in_channels = in_channels
        self.inter_channels = inter_channels

        self.g = nn.Conv2d(in_channels=self.in_channels, out_channels=self.inter_channels, kernel_size=1, stride=1,
                           padding=0, bias=bias)

        self.W = nn.Conv2d(in_channels=self.inter_channels, out_channels=self.in_channels, kernel_size=1, stride=1,
                           padding=0, bias=bias)
        # for pytorch 0.3.1
        # nn.init.constant(self.W.weight, 0)
        # nn.init.constant(self.W.bias, 0)
        # for pytorch 0.4.0
        nn.init.constant_(self.W.weight, 0)
        # nn.init.constant_(self.W.bias, 0)
        self.theta = nn.Conv2d(in_channels=self.in_channels, out_channels=self.inter_channels, kernel_size=1, stride=1,
                               padding=0, bias=bias)

        self.phi = nn.Conv2d(in_channels=self.in_channels, out_channels=self.inter_channels, kernel_size=1, stride=1,
                             padding=0, bias=bias)

    def forward(self, x):
        batch_size = x.size(0)

        g_x = self.g(x).view(batch_size, self.inter_channels, -1)

        g_x = g_x.permute(0, 2, 1)

        theta_x = self.theta(x).view(batch_size, self.inter_channels, -1)

        theta_x = theta_x.permute(0, 2, 1)

        phi_x = self.phi(x).view(batch_size, self.inter_channels, -1)

        f = torch.matmul(theta_x, phi_x)

        f_div_C = F.softmax(f, dim=1)

        y = torch.matmul(f_div_C, g_x)

        y = y.permute(0, 2, 1).contiguous()

        y = y.view(batch_size, self.inter_channels, *x.size()[2:])
        W_y = self.W(y)
        z = W_y + x
        return z


def conv3x3(in_planes, out_planes, stride=1):
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False, padding_mode='reflect')


class NLBasicBlock(nn.Module):
    def __init__(self, inplanes, stride=1, with_norm=False):
        super(NLBasicBlock, self).__init__()
        self.with_norm = with_norm
        kernel = 3
        self.conv1 = conv3x3(inplanes, inplanes, stride)
        self.conv2 = BasicConv(inplanes, inplanes, relu=False)
        self.se = NonLocalBlock2D(inplanes, inplanes)
        self.relu = nn.PReLU()
        if self.with_norm:
            self.bn1 = nn.BatchNorm2d(inplanes)
            self.bn2 = nn.BatchNorm2d(inplanes)

    def forward(self, x):
        out = x = self.conv1(x)
        if self.with_norm:
            out = self.bn1(out)
        out = self.relu(out)

        out = self.se(out)
        out += x
        out = self.conv2(out)
        if self.with_norm:
            out = self.bn2(out)
        out = self.relu(out)
        # print(out.shape)
        return out


class ChannelPool(nn.Module):
    def forward(self, x):
        return torch.cat((torch.max(x, 1)[0].unsqueeze(1), torch.mean(x, 1).unsqueeze(1)), dim=1)


class spatial_attn_layer(nn.Module):
    def __init__(self, kernel_size=5):
        super(spatial_attn_layer, self).__init__()
        self.compress = ChannelPool()
        self.spatial = BasicConv(2, 1, relu=False)

    def forward(self, x):
        # import pdb;pdb.set_trace()
        x_compress = self.compress(x)
        x_out = self.spatial(x_compress)
        scale = torch.sigmoid(x_out)  # broadcasting
        return x * scale


class Spatial_BasicBlock(nn.Module):
    def __init__(self, inplanes, stride=1, reduction=64, with_norm=False):
        super(Spatial_BasicBlock, self).__init__()
        self.with_norm = with_norm
        kernel = 3
        self.conv1 = conv3x3(inplanes, inplanes, stride)
        self.conv2 = BasicConv(inplanes, inplanes, relu=False)
        self.se = spatial_attn_layer(kernel)
        self.relu = nn.PReLU()
        if self.with_norm:
            self.bn1 = nn.BatchNorm2d(inplanes)
            self.bn2 = nn.BatchNorm2d(inplanes)

    def forward(self, x):
        out = x = self.conv1(x)
        if self.with_norm:
            out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        if self.with_norm:
            out = self.bn2(out)
        out = self.se(out)
        out += x
        out = self.relu(out)
        return out

class ResidualDenseBlock(nn.Module):
    def __init__(self, in_channels, stride):
        super(ResidualDenseBlock, self).__init__()
        # gc: growth channel, i.e. intermediate channels

        self.conv1 = BasicConv(in_channels, in_channels, stride, relu=False)
        self.conv2 = BasicConv(in_channels * 2, in_channels, stride, relu=False)
        self.conv3 = BasicConv(in_channels * 3, in_channels, stride, relu=False)

        self.lrelu = nn.PReLU()

    def forward(self, x):
        x1 = self.lrelu(self.conv1(x))
        x2 = self.lrelu(self.conv2(torch.cat((x, x1), 1)))
        x3 = self.lrelu(self.conv3(torch.cat((x, x1, x2), 1)))
        return x3 * 0.333333 + x


class ResidualModule(nn.Module):
    def __init__(self, in_channels, stride, dialtions=1):
        super(ResidualModule, self).__init__()
        self.op = nn.Sequential(
            BasicConv(in_channels, in_channels, stride, dilation=dialtions, relu=False, groups=in_channels),
            nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=1, padding=2, dilation=2, groups=in_channels,
                      bias=False, padding_mode='reflect'),
            nn.Conv2d(in_channels, in_channels, kernel_size=1, padding=0, bias=False),
            # nn.BatchNorm2d(in_channels),
            nn.PReLU(),
        )

    def forward(self, x):
        res = self.op(x)
        return x + res


class ReLUConvBN(nn.Module):

    def __init__(self, C_in, C_out, kernel_size, stride, padding, affine=True):
        super(ReLUConvBN, self).__init__()
        self.op = nn.Sequential(
            nn.ReLU(inplace=False),
            nn.Conv2d(C_in, C_out, kernel_size, stride=stride, padding=padding, bias=False, padding_mode='reflect'),
            # nn.BatchNorm2d(C_out, affine=affine)
        )

    def forward(self, x):
        return self.op(x)


class DilConv(nn.Module):

    def __init__(self, C_in, C_out, kernel_size, stride, padding, dilation, affine=True):
        super(DilConv, self).__init__()
        self.op = nn.Sequential(
            nn.ReLU(inplace=False),
            nn.Conv2d(C_in, C_in, kernel_size=kernel_size, stride=stride, padding=padding, dilation=dilation,
                      groups=C_in, bias=False, padding_mode='reflect'),
            nn.Conv2d(C_in, C_out, kernel_size=1, padding=0, bias=False),
            # nn.BatchNorm2d(C_out, affine=affine),
        )

    def forward(self, x):
        return self.op(x)

# 深度可分离卷积

class SepConv(nn.Module):

    def __init__(self, C_in, C_out, kernel_size, stride, padding, affine=True):
        super(SepConv, self).__init__()
        self.op = nn.Sequential(
            nn.ReLU(inplace=False),
            nn.Conv2d(C_in, C_in, kernel_size=kernel_size, stride=stride, padding=padding, groups=C_in, bias=False, padding_mode='reflect'),
            nn.Conv2d(C_in, C_in, kernel_size=1, padding=0, bias=False),
            # nn.BatchNorm2d(C_in, affine=affine),
            nn.ReLU(inplace=False),
            nn.Conv2d(C_in, C_in, kernel_size=kernel_size, stride=1, padding=padding, groups=C_in, bias=False, padding_mode='reflect'),
            nn.Conv2d(C_in, C_out, kernel_size=1, padding=0, bias=False),
            # nn.BatchNorm2d(C_out, affine=affine),
        )

    def forward(self, x):
        return self.op(x)


class Identity(nn.Module):

    def __init__(self):
        super(Identity, self).__init__()

    def forward(self, x):
        return x


class Zero(nn.Module):

    def __init__(self, stride):
        super(Zero, self).__init__()
        self.stride = stride

    def forward(self, x):
        if self.stride == 1:
            return x.mul(0.)
        return x[:, :, ::self.stride, ::self.stride].mul(0.)

