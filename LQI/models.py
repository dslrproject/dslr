import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import torch.autograd as autograd
import torch.optim as optim
import torchvision, numpy as np, pdb
from utils import *

class Doubleconv(nn.Module):
    __doc__ = '(convolution => [BN] => ReLU) * 2'

    def __init__(self, in_channels, out_channels):
        super(Doubleconv, self).__init__()
        self.double_conv = nn.Sequential(nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1), nn.BatchNorm2d(num_features=out_channels), nn.LeakyReLU(negative_slope=0.2, inplace=True), nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1), nn.BatchNorm2d(num_features=out_channels), nn.LeakyReLU(negative_slope=0.2, inplace=True))

    def forward(self, x):
        return self.double_conv(x)


class Down(nn.Module):
    __doc__ = 'Downscaling with maxpool'

    def __init__(self):
        super(Down, self).__init__()
        self.down = nn.Sequential(nn.MaxPool2d(kernel_size=(2, 4)))

    def forward(self, x):
        return self.down(x)


class Up(nn.Module):
    __doc__ = 'Upscaling with ConvTranspose2d'

    def __init__(self, in_channels, out_channels):
        super(Up, self).__init__()
        self.up = nn.Sequential(nn.ConvTranspose2d(in_channels, out_channels, kernel_size=(4,
                                                                                           6), stride=(2,
                                                                                                       4), padding=1), nn.BatchNorm2d(num_features=out_channels), nn.LeakyReLU(negative_slope=0.2, inplace=True))

    def forward(self, x):
        return self.up(x)


class Out(nn.Module):

    def __init__(self, in_channels, out_channels):
        super(Out, self).__init__()
        self.out = nn.Sequential(nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1), nn.Tanh())

    def forward(self, x):
        return self.out(x)


class DownBlock(nn.Module):
    __doc__ = 'Down and then Doubleconv'

    def __init__(self, in_channels, out_channels):
        super(DownBlock, self).__init__()
        self.down_double_conv = nn.Sequential(Down(), Doubleconv(in_channels, out_channels))

    def forward(self, x):
        return self.down_double_conv(x)


class UpBlock(nn.Module):
    __doc__ = 'Up, concat and then Doubleconv'

    def __init__(self, in_channels, out_channels):
        super(UpBlock, self).__init__()
        self.up = Up(in_channels, out_channels)
        self.conv = Doubleconv(2 * out_channels, out_channels)

    def forward(self, x1, x2):
        x_up = self.up(x1)
        x_cat = torch.cat([x2, x_up], dim=1)
        x_conv = self.conv(x_cat)
        return x_conv


class Unet(nn.Module):

    def __init__(self, args):
        super(Unet, self).__init__()
        self.args = args
        nc = 3 if args.no_polar else 2
        ndf = 64
        lf = ff = (1, 4)
        self.encoder_conv1 = Doubleconv(nc, ndf)
        self.encoder_down1 = DownBlock(ndf, ndf * 2)
        self.encoder_down2 = DownBlock(ndf * 2, ndf * 4)
        self.encoder_down3 = DownBlock(ndf * 4, ndf * 8)
        self.encoder_down4 = DownBlock(ndf * 8, ndf * 16)
        nz = ndf * 32
        self.encoder_down5 = nn.Sequential(nn.Conv2d((ndf * 16), nz, lf, 1, 0, bias=False), nn.BatchNorm2d(num_features=nz), nn.LeakyReLU(negative_slope=0.2, inplace=True))
        self.encoder_conv2 = nn.Sequential(nn.Conv2d(nz, nz, 1, 1, 0, bias=False), nn.BatchNorm2d(num_features=nz), nn.LeakyReLU(negative_slope=0.2, inplace=True), nn.Conv2d(nz, nz, 1, 1, 0, bias=False), nn.BatchNorm2d(num_features=nz), nn.LeakyReLU(negative_slope=0.2, inplace=True))
        self.decoder_up1 = nn.Sequential(nn.ConvTranspose2d(nz, (ndf * 16), ff, 1, 0, bias=False), nn.BatchNorm2d(num_features=(ndf * 16)), nn.LeakyReLU(negative_slope=0.2, inplace=True))
        self.decoder_conv3 = Doubleconv(ndf * 16 * 2, ndf * 16)
        self.decoder_up2 = UpBlock(ndf * 16, ndf * 8)
        self.decoder_up3 = UpBlock(ndf * 8, ndf * 4)
        self.decoder_up4 = UpBlock(ndf * 4, ndf * 2)
        self.decoder_up5 = UpBlock(ndf * 2, ndf)
        self.out = Out(ndf, nc)

    def forward(self, x):
        x_conv1 = self.encoder_conv1(x)
        x_down1 = self.encoder_down1(x_conv1)
        x_down2 = self.encoder_down2(x_down1)
        x_down3 = self.encoder_down3(x_down2)
        x_down4 = self.encoder_down4(x_down3)
        x_down5 = self.encoder_down5(x_down4)
        x_conv2 = self.encoder_conv2(x_down5)
        z = x_conv2
        x_up1 = self.decoder_up1(x_conv2)
        xcat = torch.cat([x_up1, x_down4], dim=1)
        x_conv3 = self.decoder_conv3(xcat)
        x_up2 = self.decoder_up2(x_conv3, x_down3)
        x_up3 = self.decoder_up3(x_up2, x_down2)
        x_up4 = self.decoder_up4(x_up3, x_down1)
        x_up5 = self.decoder_up5(x_up4, x_conv1)
        xout = self.out(x_up5)
        recon = xout
        return (
         recon, None, z)

class CNNIQAnet(nn.Module):
    def __init__(self,args, ker_size=(7,5), n_kers=50, n1_nodes=800, n2_nodes=800):
        super(CNNIQAnet, self).__init__()
        self.args = args
        self.conv1  = nn.Sequential(nn.Conv2d(2, n_kers, ker_size))
        #self.conv2  = nn.Sequential(nn.Conv2d(n_kers//2, n_kers, ker_size))
        self.fc1    = nn.Sequential(nn.Linear(2 * n_kers, n1_nodes))
        self.fc1    = nn.Sequential(nn.Linear(2 * n_kers, n1_nodes))
        self.fc2    = nn.Sequential(nn.Linear(n1_nodes, n2_nodes))
        self.fc3    = nn.Sequential(nn.Linear(n2_nodes, 1))
        #self.fc4    = nn.Sequential(nn.Linear(10,1, bias = False))
        self.dropout = nn.Sequential(nn.Dropout())

    def forward(self, x):
        x = x.cuda()
        x  = x.view(-1, x.size(-3), x.size(-2), x.size(-1))  #
        h = self.conv1(x)
        #h  = self.conv2(h0)

        # h1 = F.adaptive_max_pool2d(h, 1)
        # h2 = -F.adaptive_max_pool2d(-h, 1)
        h1 = F.max_pool2d(h, (h.size(-2), h.size(-1)))
        h2 = -F.max_pool2d(-h, (h.size(-2), h.size(-1)))
        h  = torch.cat((h1, h2), 1)  # max-min pooling
        h  = h.squeeze(3).squeeze(2)

        h  = F.relu(self.fc1(h))
        h  = self.dropout(h)
        h  = F.relu(self.fc2(h))
        q  = F.relu(self.fc3(h))
        #q = F.relu(self.fc4(q))
        #q =  self.fc4(q)
        return q


class VAE_dynseg(nn.Module):

    def __init__(self, args, return_last_layer=False):
        super(VAE_dynseg, self).__init__()
        self.args = args
        self.return_last_layer = return_last_layer
        mult = 1
        nz = args.z_dim * mult
        nc = 3 if args.no_polar else 2
        ndf = 16
        lf = (5, 64)
        self.encoder_conv2d_a = nn.Sequential(nn.Conv2d(nc, ndf, 3, 1, 1, bias=False))
        self.encoder_leakyrelu_b = nn.Sequential(nn.LeakyReLU(0.2, inplace=True))
        self.encoder_conv2d_A = nn.Sequential(nn.Conv2d(ndf, ndf, 3, 1, 1, bias=False))
        self.encoder_leakyrelu_B = nn.Sequential(nn.LeakyReLU(0.2, inplace=True))
        self.encoder_conv2d_1 = nn.Sequential(nn.Conv2d(ndf, ndf, 3, 1, 1, bias=False))
        self.encoder_leakyrelu_2 = nn.Sequential(nn.LeakyReLU(0.2, inplace=True))
        self.encoder_conv2d_c = nn.Sequential(nn.Conv2d(ndf, ndf, 3, 1, 1, bias=False))
        self.encoder_leakyrelu_d = nn.Sequential(nn.LeakyReLU(0.2, inplace=True))
        self.encoder_conv2d_C = nn.Sequential(nn.Conv2d(ndf, ndf, 3, 1, 1, bias=False))
        self.encoder_leakyrelu_D = nn.Sequential(nn.LeakyReLU(0.2, inplace=True))
        self.encoder_conv2d_3 = nn.Sequential(nn.Conv2d(ndf, (ndf * 2), 3, 2, 1, bias=False))
        self.encoder_batchnorm2d_4 = nn.Sequential(nn.BatchNorm2d(ndf * 2))
        self.encoder_leakyrelu_5 = nn.Sequential(nn.LeakyReLU(0.2, inplace=True))
        self.encoder_conv2d_e = nn.Sequential(nn.Conv2d((ndf * 2), (ndf * 2), 3, 1, 1, bias=False))
        self.encoder_leakyrelu_f = nn.Sequential(nn.LeakyReLU(0.2, inplace=True))
        self.encoder_conv2d_E = nn.Sequential(nn.Conv2d((ndf * 2), (ndf * 2), 3, 1, 1, bias=False))
        self.encoder_leakyrelu_F = nn.Sequential(nn.LeakyReLU(0.2, inplace=True))
        self.encoder_conv2d_6 = nn.Sequential(nn.Conv2d((ndf * 2), (ndf * 4), 3, 2, 1, bias=False))
        self.encoder_batchnorm2d_7 = nn.Sequential(nn.BatchNorm2d(ndf * 4))
        self.encoder_leakyrelu_8 = nn.Sequential(nn.LeakyReLU(0.2, inplace=True))
        self.encoder_conv2d_g = nn.Sequential(nn.Conv2d((ndf * 4), (ndf * 4), 3, 1, 1, bias=False))
        self.encoder_leakyrelu_h = nn.Sequential(nn.LeakyReLU(0.2, inplace=True))
        self.encoder_conv2d_G = nn.Sequential(nn.Conv2d((ndf * 4), (ndf * 4), 3, 1, 1, bias=False))
        self.encoder_leakyrelu_H = nn.Sequential(nn.LeakyReLU(0.2, inplace=True))
        self.encoder_conv2d_9 = nn.Sequential(nn.Conv2d((ndf * 4), (ndf * 8), (3, 3), 2, (1,
                                                                                          1), bias=False))
        self.encoder_batchnorm2d_10 = nn.Sequential(nn.BatchNorm2d(ndf * 8))
        self.encoder_leakyrelu_11 = nn.Sequential(nn.LeakyReLU(0.2, inplace=True))
        self.encoder_conv2d_12 = nn.Sequential(nn.Conv2d((ndf * 8), nz, lf, 1, 0, bias=False))
        self.drop1 = nn.Sequential(nn.Dropout(0.5))
        self.dense = nn.Sequential(nn.Linear(nz, 256))
        self.encoder_leakyrelu_12 = nn.Sequential(nn.LeakyReLU(0.2, inplace=True))
        self.drop2 = nn.Sequential(nn.Dropout(0.5))
        self.dense2 = nn.Sequential(nn.Linear(256,5))
        self.dense3 = nn.Sequential(nn.Linear(5,1))
        ngf = ndf
        base = 4
        ff = (5, 64)
        nz = args.z_dim
        nc = 3 if args.no_polar else 2
#        self.decoder_convtranspose2d_13 = nn.Sequential(nn.ConvTranspose2d(nz, (ngf * 8), ff, 1, 0, bias=False))
#        self.decoder_batchnorm2d_14 = nn.Sequential(nn.BatchNorm2d(ngf * 8))
#        self.decoder_relu_15 = nn.Sequential(nn.ReLU(True))
#        self.decoder_conv2d_i = nn.Sequential(nn.Conv2d((ngf * 8), (ngf * 8), 3, 1, 1, bias=False))
#        self.decoder_relu_j = nn.Sequential(nn.ReLU(True))
#        self.decoder_convtranspose2d_16 = nn.Sequential(nn.ConvTranspose2d((ngf * 8), (ngf * 4), (4,
#                                                                                                  4), stride=2, padding=(1,
#                                                                                                                         1), bias=False))
#        self.decoder_batchnorm2d_17 = nn.Sequential(nn.BatchNorm2d(ngf * 4))
#        self.decoder_relu_18 = nn.Sequential(nn.ReLU(True))
#        self.decoder_conv2d_k = nn.Sequential(nn.Conv2d((ngf * 4 * 2), (ngf * 4), 3, 1, 1, bias=False))
#        self.decoder_relu_l = nn.Sequential(nn.ReLU(True))
#        self.decoder_conv2d_K = nn.Sequential(nn.Conv2d((ngf * 4), (ngf * 4), 3, 1, 1, bias=False))
#        self.decoder_relu_L = nn.Sequential(nn.ReLU(True))
#        self.decoder_convtranspose2d_19 = nn.Sequential(nn.ConvTranspose2d((ngf * 4), (ngf * 2), (4,
#                                                                                                  4), stride=2, padding=(1,
#                                                                                                                         1), bias=False))
#        self.decoder_batchnorm2d_20 = nn.Sequential(nn.BatchNorm2d(ngf * 2))
#        self.decoder_relu_21 = nn.Sequential(nn.ReLU(True))
#        self.decoder_conv2d_m = nn.Sequential(nn.Conv2d((ngf * 2 * 2), (ngf * 2), 3, 1, 1, bias=False))
#        self.decoder_relu_n = nn.Sequential(nn.ReLU(True))
#        self.decoder_conv2d_M = nn.Sequential(nn.Conv2d((ngf * 2), (ngf * 2), 3, 1, 1, bias=False))
#        self.decoder_relu_N = nn.Sequential(nn.ReLU(True))
#        self.decoder_convtranspose2d_22 = nn.Sequential(nn.ConvTranspose2d((ngf * 2), (ngf * 1), (4,
#                                                                                                  4), stride=2, padding=(1,
#                                                                                                                         1), bias=False))
#        self.decoder_batchnorm2d_23 = nn.Sequential(nn.BatchNorm2d(ngf * 1))
#        self.decoder_relu_24 = nn.Sequential(nn.ReLU(True))
#        self.decoder_conv2d_o = nn.Sequential(nn.Conv2d((ngf * 1 * 2), (ngf * 1), 3, 1, 1, bias=False))
#        self.decoder_relu_p = nn.Sequential(nn.ReLU(True))
#        self.decoder_conv2d_O = nn.Sequential(nn.Conv2d((ngf * 1), (ngf * 1), 3, 1, 1, bias=False))
#        self.decoder_relu_P = nn.Sequential(nn.ReLU(True))
#        self.decoder_convtranspose2d_25 = nn.Sequential(nn.ConvTranspose2d(ngf, ngf, 3, 1, 1, bias=False))
#        self.decoder_relu_26 = nn.Sequential(nn.ReLU(True))
#        self.decoder_conv2d_q = nn.Sequential(nn.Conv2d((ngf * 2), ngf, 3, 1, 1, bias=False))
#        self.decoder_relu_R = nn.Sequential(nn.ReLU(True))
#        self.decoder_conv2d_Q = nn.Sequential(nn.Conv2d(ngf, nc, 3, 1, 1, bias=False))
#        self.tanh_27 = nn.Sequential(nn.Tanh())
        self.softmax = nn.Sequential(nn.Softmax(dim=1))
#        self.sigmoid = nn.Sequential(nn.Sigmoid())
    def forward(self, x):
        xa = self.encoder_conv2d_a(x)
        xb = self.encoder_leakyrelu_b(xa)
        xA = self.encoder_conv2d_A(xb)
        xB = self.encoder_leakyrelu_B(xA)
        x1 = self.encoder_conv2d_1(xB)
        x2 = self.encoder_leakyrelu_2(x1)
        xc = self.encoder_conv2d_c(x2)
        xd = self.encoder_leakyrelu_d(xc)
        xC = self.encoder_conv2d_C(xd)
        xD = self.encoder_leakyrelu_D(xC)
        x3 = self.encoder_conv2d_3(xD)
        x4 = self.encoder_batchnorm2d_4(x3)
        x5 = self.encoder_leakyrelu_5(x4)
        xe = self.encoder_conv2d_e(x5)
        xf = self.encoder_leakyrelu_f(xe)
        xE = self.encoder_conv2d_E(xf)
        xF = self.encoder_leakyrelu_F(xE)
        x6 = self.encoder_conv2d_6(xF)
        x7 = self.encoder_batchnorm2d_7(x6)
        x8 = self.encoder_leakyrelu_8(x7)
        xg = self.encoder_conv2d_g(x8)
        xh = self.encoder_leakyrelu_h(xg)
        xG = self.encoder_conv2d_G(xh)
        xH = self.encoder_leakyrelu_H(xG)
        x9 = self.encoder_conv2d_9(xH)
        x10 = self.encoder_batchnorm2d_10(x9)
        x11 = self.encoder_leakyrelu_11(x10)
        x12 = self.encoder_conv2d_12(x11)
        z = x12
        while z.dim() != 2:
            z = z.squeeze(-1)
        x13 = self.drop1(self.dense(z))
        x14 = self.encoder_leakyrelu_12(x13)
        x15 = self.drop2(self.dense2(x14))
        x16 = self.dense3(x15)
        return x16
#        x13 = self.decoder_convtranspose2d_13(x12)
#        x14 = self.decoder_batchnorm2d_14(x13)
#        x15 = self.decoder_relu_15(x14)
#        x16 = self.decoder_convtranspose2d_16(x15)
#        x17 = self.decoder_batchnorm2d_17(x16)
#        x18 = self.decoder_relu_18(x17)
#        x_cat1 = torch.cat([x18, xH], dim=1)
#        xk = self.decoder_conv2d_k(x_cat1)
#        xl = self.decoder_relu_l(xk)
#        xK = self.decoder_conv2d_K(xl)
#        xL = self.decoder_relu_L(xK)
#        x19 = self.decoder_convtranspose2d_19(xL)
#        x20 = self.decoder_batchnorm2d_20(x19)
#        x21 = self.decoder_relu_21(x20)
#        x_cat2 = torch.cat([x21, xF], dim=1)
#        xm = self.decoder_conv2d_m(x_cat2)
#        xn = self.decoder_relu_n(xm)
#        xM = self.decoder_conv2d_M(xn)
#        xN = self.decoder_relu_N(xM)
#        x22 = self.decoder_convtranspose2d_22(xN)
#        x23 = self.decoder_batchnorm2d_23(x22)
#        x24 = self.decoder_relu_24(x23)
#        x_cat3 = torch.cat([x24, xD], dim=1)
#        xo = self.decoder_conv2d_o(x_cat3)
#        xp = self.decoder_relu_p(xo)
#        xO = self.decoder_conv2d_O(xp)
#        xP = self.decoder_relu_P(xO)
#        x25 = self.decoder_convtranspose2d_25(xP)
#        x26 = self.decoder_relu_26(x25)
#        x_cat4 = torch.cat([x26, xB], dim=1)
#        xq = self.decoder_conv2d_q(x_cat4)
#        xR = self.decoder_relu_R(xq)
#        xQ = self.decoder_conv2d_Q(xR)
#        x27 = self.sigmoid(xQ)
#        recon = xQ
        if self.return_last_layer:
            return (
             recon, xR)
        return (recon, xQ)

    def sample(self, nb_samples=16, tmp=1):
        noise = torch.cuda.FloatTensor(nb_samples, self.args.z_dim).normal_(0, tmp)
        return self.decode(noise)

    @staticmethod
    def gaussian_kl(mu, logvar):
        return -0.5 * torch.sum((1 + logvar - mu.pow(2) - logvar.exp()), dim=(-1))

    @staticmethod
    def log_gauss(z, params):
        mu, std = params
        return -0.5 * (t.pow(z - mu, 2) * t.pow(std + 1e-08, -2) + 2 * t.log(std + 1e-08) + math.log(2 * math.pi)).sum(1)


class VAE(nn.Module):

    def __init__(self, args, return_last_layer=False):
        super(VAE, self).__init__()
        self.args = args
        self.return_last_layer = return_last_layer
        mult = 1
        nz = args.z_dim * mult
        nc = 3 if args.no_polar else 2
        ndf = 64
        lf = (1, 64)
        self.encoder_conv2d_a = nn.Sequential(nn.Conv2d(nc, ndf, 3, 1, 1, bias=False))
        self.encoder_leakyrelu_b = nn.Sequential(nn.LeakyReLU(0.2, inplace=True))
        self.encoder_conv2d_A = nn.Sequential(nn.Conv2d(ndf, ndf, 3, 1, 1, bias=False))
        self.encoder_leakyrelu_B = nn.Sequential(nn.LeakyReLU(0.2, inplace=True))
        self.encoder_conv2d_1 = nn.Sequential(nn.Conv2d(ndf, ndf, 3, 2, 1, bias=False))
        self.encoder_leakyrelu_2 = nn.Sequential(nn.LeakyReLU(0.2, inplace=True))
        self.encoder_conv2d_c = nn.Sequential(nn.Conv2d(ndf, ndf, 3, 1, 1, bias=False))
        self.encoder_leakyrelu_d = nn.Sequential(nn.LeakyReLU(0.2, inplace=True))
        self.encoder_conv2d_C = nn.Sequential(nn.Conv2d(ndf, ndf, 3, 1, 1, bias=False))
        self.encoder_leakyrelu_D = nn.Sequential(nn.LeakyReLU(0.2, inplace=True))
        self.encoder_conv2d_3 = nn.Sequential(nn.Conv2d(ndf, (ndf * 2), 3, 2, 1, bias=False))
        self.encoder_batchnorm2d_4 = nn.Sequential(nn.BatchNorm2d(ndf * 2))
        self.encoder_leakyrelu_5 = nn.Sequential(nn.LeakyReLU(0.2, inplace=True))
        self.encoder_conv2d_e = nn.Sequential(nn.Conv2d((ndf * 2), (ndf * 2), 3, 1, 1, bias=False))
        self.encoder_leakyrelu_f = nn.Sequential(nn.LeakyReLU(0.2, inplace=True))
        self.encoder_conv2d_E = nn.Sequential(nn.Conv2d((ndf * 2), (ndf * 2), 3, 1, 1, bias=False))
        self.encoder_leakyrelu_F = nn.Sequential(nn.LeakyReLU(0.2, inplace=True))
        self.encoder_conv2d_6 = nn.Sequential(nn.Conv2d((ndf * 2), (ndf * 4), 3, 2, 1, bias=False))
        self.encoder_batchnorm2d_7 = nn.Sequential(nn.BatchNorm2d(ndf * 4))
        self.encoder_leakyrelu_8 = nn.Sequential(nn.LeakyReLU(0.2, inplace=True))
        self.encoder_conv2d_g = nn.Sequential(nn.Conv2d((ndf * 4), (ndf * 4), 3, 1, 1, bias=False))
        self.encoder_leakyrelu_h = nn.Sequential(nn.LeakyReLU(0.2, inplace=True))
        self.encoder_conv2d_G = nn.Sequential(nn.Conv2d((ndf * 4), (ndf * 4), 3, 1, 1, bias=False))
        self.encoder_leakyrelu_H = nn.Sequential(nn.LeakyReLU(0.2, inplace=True))
        self.encoder_conv2d_9 = nn.Sequential(nn.Conv2d((ndf * 4), (ndf * 8), (3, 3), 2, (1,
                                                                                          1), bias=False))
        self.encoder_batchnorm2d_10 = nn.Sequential(nn.BatchNorm2d(ndf * 8))
        self.encoder_leakyrelu_11 = nn.Sequential(nn.LeakyReLU(0.2, inplace=True))
        self.encoder_conv2d_12 = nn.Sequential(nn.Conv2d((ndf * 8), nz, lf, 1, 0, bias=False))
        ngf = ndf
        base = 4
        ff = (1, 64)
        nz = args.z_dim
        nc = 3 if args.no_polar else 2
        self.decoder_convtranspose2d_13 = nn.Sequential(nn.ConvTranspose2d(nz, (ngf * 8), ff, 1, 0, bias=False))
        self.decoder_batchnorm2d_14 = nn.Sequential(nn.BatchNorm2d(ngf * 8))
        self.decoder_relu_15 = nn.Sequential(nn.ReLU(True))
        self.decoder_conv2d_i = nn.Sequential(nn.Conv2d((ngf * 8), (ngf * 8), 3, 1, 1, bias=False))
        self.decoder_relu_j = nn.Sequential(nn.ReLU(True))
        self.decoder_convtranspose2d_16 = nn.Sequential(nn.ConvTranspose2d((ngf * 8), (ngf * 4), (4,
                                                                                                  4), stride=2, padding=(1,
                                                                                                                         1), bias=False))
        self.decoder_batchnorm2d_17 = nn.Sequential(nn.BatchNorm2d(ngf * 4))
        self.decoder_relu_18 = nn.Sequential(nn.ReLU(True))
        self.decoder_conv2d_k = nn.Sequential(nn.Conv2d((ngf * 4 * 2), (ngf * 4), 3, 1, 1, bias=False))
        self.decoder_relu_l = nn.Sequential(nn.ReLU(True))
        self.decoder_conv2d_K = nn.Sequential(nn.Conv2d((ngf * 4), (ngf * 4), 3, 1, 1, bias=False))
        self.decoder_relu_L = nn.Sequential(nn.ReLU(True))
        self.decoder_convtranspose2d_19 = nn.Sequential(nn.ConvTranspose2d((ngf * 4), (ngf * 2), (4,
                                                                                                  4), stride=2, padding=(1,
                                                                                                                         1), bias=False))
        self.decoder_batchnorm2d_20 = nn.Sequential(nn.BatchNorm2d(ngf * 2))
        self.decoder_relu_21 = nn.Sequential(nn.ReLU(True))
        self.decoder_conv2d_m = nn.Sequential(nn.Conv2d((ngf * 2 * 2), (ngf * 2), 3, 1, 1, bias=False))
        self.decoder_relu_n = nn.Sequential(nn.ReLU(True))
        self.decoder_conv2d_M = nn.Sequential(nn.Conv2d((ngf * 2), (ngf * 2), 3, 1, 1, bias=False))
        self.decoder_relu_N = nn.Sequential(nn.ReLU(True))
        self.decoder_convtranspose2d_22 = nn.Sequential(nn.ConvTranspose2d((ngf * 2), (ngf * 1), (4,
                                                                                                  4), stride=2, padding=(1,
                                                                                                                         1), bias=False))
        self.decoder_batchnorm2d_23 = nn.Sequential(nn.BatchNorm2d(ngf * 1))
        self.decoder_relu_24 = nn.Sequential(nn.ReLU(True))
        self.decoder_conv2d_o = nn.Sequential(nn.Conv2d((ngf * 1 * 2), (ngf * 1), 3, 1, 1, bias=False))
        self.decoder_relu_p = nn.Sequential(nn.ReLU(True))
        self.decoder_conv2d_O = nn.Sequential(nn.Conv2d((ngf * 1), (ngf * 1), 3, 1, 1, bias=False))
        self.decoder_relu_P = nn.Sequential(nn.ReLU(True))
        self.decoder_convtranspose2d_25 = nn.Sequential(nn.ConvTranspose2d(ngf, ngf, 4, 2, 1, bias=False))
        self.decoder_relu_26 = nn.Sequential(nn.ReLU(True))
        self.decoder_conv2d_q = nn.Sequential(nn.Conv2d((ngf * 2), ngf, 3, 1, 1, bias=False))
        self.decoder_relu_R = nn.Sequential(nn.ReLU(True))
        self.decoder_conv2d_Q = nn.Sequential(nn.Conv2d(ngf, nc, 3, 1, 1, bias=False))
        self.tanh_27 = nn.Sequential(nn.Tanh())

    def forward(self, x):
        xa = self.encoder_conv2d_a(x)
        xb = self.encoder_leakyrelu_b(xa)
        xA = self.encoder_conv2d_A(xb)
        xB = self.encoder_leakyrelu_B(xA)
        x1 = self.encoder_conv2d_1(xB)
        x2 = self.encoder_leakyrelu_2(x1)
        xc = self.encoder_conv2d_c(x2)
        xd = self.encoder_leakyrelu_d(xc)
        xC = self.encoder_conv2d_C(xd)
        xD = self.encoder_leakyrelu_D(xC)
        x3 = self.encoder_conv2d_3(xD)
        x4 = self.encoder_batchnorm2d_4(x3)
        x5 = self.encoder_leakyrelu_5(x4)
        xe = self.encoder_conv2d_e(x5)
        xf = self.encoder_leakyrelu_f(xe)
        xE = self.encoder_conv2d_E(xf)
        xF = self.encoder_leakyrelu_F(xE)
        x6 = self.encoder_conv2d_6(xF)
        x7 = self.encoder_batchnorm2d_7(x6)
        x8 = self.encoder_leakyrelu_8(x7)
        xg = self.encoder_conv2d_g(x8)
        xh = self.encoder_leakyrelu_h(xg)
        xG = self.encoder_conv2d_G(xh)
        xH = self.encoder_leakyrelu_H(xG)
        x9 = self.encoder_conv2d_9(xH)
        x10 = self.encoder_batchnorm2d_10(x9)
        x11 = self.encoder_leakyrelu_11(x10)
        x12 = self.encoder_conv2d_12(x11)
        z = x12
        while z.dim() != 2:
            z = z.squeeze(-1)

        x13 = self.decoder_convtranspose2d_13(x12)
        x14 = self.decoder_batchnorm2d_14(x13)
        x15 = self.decoder_relu_15(x14)
        x16 = self.decoder_convtranspose2d_16(x15)
        x17 = self.decoder_batchnorm2d_17(x16)
        x18 = self.decoder_relu_18(x17)
        x_cat1 = torch.cat([x18, xH], dim=1)
        xk = self.decoder_conv2d_k(x_cat1)
        xl = self.decoder_relu_l(xk)
        xK = self.decoder_conv2d_K(xl)
        xL = self.decoder_relu_L(xK)
        x19 = self.decoder_convtranspose2d_19(xL)
        x20 = self.decoder_batchnorm2d_20(x19)
        x21 = self.decoder_relu_21(x20)
        x_cat2 = torch.cat([x21, xF], dim=1)
        xm = self.decoder_conv2d_m(x_cat2)
        xn = self.decoder_relu_n(xm)
        xM = self.decoder_conv2d_M(xn)
        xN = self.decoder_relu_N(xM)
        x22 = self.decoder_convtranspose2d_22(xN)
        x23 = self.decoder_batchnorm2d_23(x22)
        x24 = self.decoder_relu_24(x23)
        x_cat3 = torch.cat([x24, xD], dim=1)
        xo = self.decoder_conv2d_o(x_cat3)
        xp = self.decoder_relu_p(xo)
        xO = self.decoder_conv2d_O(xp)
        xP = self.decoder_relu_P(xO)
        x25 = self.decoder_convtranspose2d_25(xP)
        x26 = self.decoder_relu_26(x25)
        x_cat4 = torch.cat([x26, xB], dim=1)
        xq = self.decoder_conv2d_q(x_cat4)
        xR = self.decoder_relu_R(xq)
        xQ = self.decoder_conv2d_Q(xR)
        x27 = self.tanh_27(xQ)
        recon = x27
        if self.return_last_layer:
            return (
             recon, xR)
        return recon

    def sample(self, nb_samples=16, tmp=1):
        noise = torch.cuda.FloatTensor(nb_samples, self.args.z_dim).normal_(0, tmp)
        return self.decode(noise)

    @staticmethod
    def gaussian_kl(mu, logvar):
        return -0.5 * torch.sum((1 + logvar - mu.pow(2) - logvar.exp()), dim=(-1))

    @staticmethod
    def log_gauss(z, params):
        mu, std = params
        return -0.5 * (t.pow(z - mu, 2) * t.pow(std + 1e-08, -2) + 2 * t.log(std + 1e-08) + math.log(2 * math.pi)).sum(1)


class VAE_filtered(nn.Module):

    def __init__(self, args):
        super(VAE_filtered, self).__init__()
        self.args = args
        nc = 3 if args.no_polar else 2
        ndf = 32
        self.unet = VAE(args, return_last_layer=True)
        self.out = nn.Sequential(nn.Conv2d((nc * 2 + ndf), (ndf * 2), 3, 1, 1, bias=False), nn.BatchNorm2d(num_features=(ndf * 2)), nn.LeakyReLU(negative_slope=0.2, inplace=True), nn.Conv2d((ndf * 2), 2, 3, 1, 1, bias=False), nn.Softmax(dim=1))

    def forward(self, x):
        recon, x_last = self.unet(x)
        xcat = torch.cat([x, recon, x_last], dim=1)
        xmask = self.out(xcat)
        return (recon, xmask)


class PointNetfeat_(nn.Module):

    def __init__(self, num_points=10240, global_feat=True):
        super(PointNetfeat_, self).__init__()
        self.conv1 = torch.nn.Conv1d(3, 64, 1)
        self.conv2 = torch.nn.Conv1d(64, 128, 1)
        self.conv3 = torch.nn.Conv1d(128, 1024, 1)
        self.bn1 = torch.nn.BatchNorm1d(64)
        self.bn2 = torch.nn.BatchNorm1d(128)
        self.bn3 = torch.nn.BatchNorm1d(1024)
        self.num_points = num_points
        self.global_feat = global_feat

    def forward(self, x):
        batchsize = x.size()[0]
        x = F.relu(self.bn1(self.conv1(x)))
        pointfeat = x
        x = F.relu(self.bn2(self.conv2(x)))
        x = self.bn3(self.conv3(x))
        x, _ = torch.max(x, 2)
        x = x.view(-1, 1024)
        return x


class PointGenCon(nn.Module):

    def __init__(self, bottleneck_size=128):
        self.bottleneck_size = bottleneck_size
        super(PointGenCon, self).__init__()
        self.conv1 = torch.nn.Conv1d(self.bottleneck_size, self.bottleneck_size, 1)
        self.conv2 = torch.nn.Conv1d(self.bottleneck_size, self.bottleneck_size // 2, 1)
        self.conv3 = torch.nn.Conv1d(self.bottleneck_size // 2, self.bottleneck_size // 4, 1)
        self.conv4 = torch.nn.Conv1d(self.bottleneck_size // 4, 3, 1)
        self.th = nn.Tanh()
        self.bn1 = torch.nn.BatchNorm1d(self.bottleneck_size)
        self.bn2 = torch.nn.BatchNorm1d(self.bottleneck_size // 2)
        self.bn3 = torch.nn.BatchNorm1d(self.bottleneck_size // 4)

    def forward(self, x):
        batchsize = x.size()[0]
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = self.th(self.conv4(x))
        return x


class AE_AtlasNet(nn.Module):

    def __init__(self, num_points=10240, bottleneck_size=1024, nb_primitives=2, AE=True):
        super(AE_AtlasNet, self).__init__()
        bot_enc = bottleneck_size if AE else 2 * bottleneck_size
        self.num_points = num_points
        self.bottleneck_size = bottleneck_size
        self.nb_primitives = nb_primitives
        self.encoder = nn.Sequential(PointNetfeat_(num_points, global_feat=True), nn.Linear(1024, bot_enc), nn.BatchNorm1d(bot_enc), nn.ReLU())
        self.decoder = nn.ModuleList([PointGenCon(bottleneck_size=(2 + self.bottleneck_size)) for i in range(0, self.nb_primitives)])

    def encode(self, x):
        if x.dim() == 4:
            if x.size(1) != 3:
                assert x.size(-1) == 3
                x = x.permute(0, 3, 1, 2).contiguous()
            x = x.reshape(x.size(0), 3, -1)
        else:
            if x.size(1) != 3:
                assert x.size(-1) == 3
                x = x.transpose(-1, -2).contiguous()
        x = self.encoder(x)
        return x

    def decode(self, x):
        outs = []
        for i in range(0, self.nb_primitives):
            rand_grid = torch.cuda.FloatTensor(x.size(0), 2, self.num_points // self.nb_primitives)
            rand_grid.data.uniform_(0, 1)
            y = x.unsqueeze(2).expand(x.size(0), x.size(1), rand_grid.size(2)).contiguous()
            y = torch.cat((rand_grid, y), 1).contiguous()
            outs.append(self.decoder[i](y))

        return torch.cat(outs, 2).contiguous().transpose(2, 1).contiguous()


if __name__ == '__main__':
    points = torch.cuda.FloatTensor(10, 3, 40, 256).normal_()
    AE = AE_AtlasNet(num_points=10240).cuda()
    out = AE(points)
    loss = get_chamfer_dist()(points, out)
    x = 1

class PointGenPSG2(nn.Module):

    def __init__(self, nz=100, num_points=10240):
        super(PointGenPSG2, self).__init__()
        self.num_points = num_points
        self.fc1 = nn.Linear(nz, 256)
        self.fc2 = nn.Linear(256, 512)
        self.fc3 = nn.Linear(512, 1024)
        self.fc4 = nn.Linear(1024, self.num_points * 3 // 2)
        self.fc11 = nn.Linear(nz, 256)
        self.fc21 = nn.Linear(256, 512)
        self.fc31 = nn.Linear(512, 1024)
        self.fc41 = nn.Linear(1024, self.num_points * 3 // 2)
        self.th = nn.Tanh()
        self.nz = nz

    def forward(self, x):
        batchsize = x.size()[0]
        x1 = x
        x2 = x
        x1 = F.relu(self.fc1(x1))
        x1 = F.relu(self.fc2(x1))
        x1 = F.relu(self.fc3(x1))
        x1 = self.th(self.fc4(x1))
        x1 = x1.view(batchsize, 3, -1)
        x2 = F.relu(self.fc11(x2))
        x2 = F.relu(self.fc21(x2))
        x2 = F.relu(self.fc31(x2))
        x2 = self.th(self.fc41(x2))
        x2 = x2.view(batchsize, 3, -1)
        return torch.cat([x1, x2], 2)
