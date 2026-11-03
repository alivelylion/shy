# source: https://github.com/CompVis/taming-transformers

import torch
import torch.nn as nn
import torchvision
import functools
import torch.nn.functional as F
import numpy as np
from collections import namedtuple
from .kepler import KeplerQuantizer, KeplerLoss




def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)


class ScalingLayer(nn.Module):
    def __init__(self):
        super(ScalingLayer, self).__init__()
        self.register_buffer('shift', torch.Tensor([-.030, -.088, -.188])[None, :, None, None])
        self.register_buffer('scale', torch.Tensor([.458, .448, .450])[None, :, None, None])

    def forward(self, inp):
        return (inp - self.shift) / self.scale


class NetLinLayer(nn.Module):
    """ A single linear layer which does a 1x1 conv """
    def __init__(self, chn_in, chn_out=1, use_dropout=False):
        super(NetLinLayer, self).__init__()
        layers = [nn.Dropout(), ] if (use_dropout) else []
        layers += [nn.Conv2d(chn_in, chn_out, 1, stride=1, padding=0, bias=False), ]
        self.model = nn.Sequential(*layers)


class vgg16(torch.nn.Module):
    def __init__(self, requires_grad=False, pretrained=True):
        super(vgg16, self).__init__()
        vgg_pretrained_features = torchvision.models.vgg16(pretrained=pretrained).features
        self.slice1 = torch.nn.Sequential()
        self.slice2 = torch.nn.Sequential()
        self.slice3 = torch.nn.Sequential()
        self.slice4 = torch.nn.Sequential()
        self.slice5 = torch.nn.Sequential()
        self.N_slices = 5
        for x in range(4):
            self.slice1.add_module(str(x), vgg_pretrained_features[x])
        for x in range(4, 9):
            self.slice2.add_module(str(x), vgg_pretrained_features[x])
        for x in range(9, 16):
            self.slice3.add_module(str(x), vgg_pretrained_features[x])
        for x in range(16, 23):
            self.slice4.add_module(str(x), vgg_pretrained_features[x])
        for x in range(23, 30):
            self.slice5.add_module(str(x), vgg_pretrained_features[x])
        if not requires_grad:
            for param in self.parameters():
                param.requires_grad = False

    def forward(self, X):
        h = self.slice1(X)
        h_relu1_2 = h
        h = self.slice2(h)
        h_relu2_2 = h
        h = self.slice3(h)
        h_relu3_3 = h
        h = self.slice4(h)
        h_relu4_3 = h
        h = self.slice5(h)
        h_relu5_3 = h
        vgg_outputs = namedtuple("VggOutputs", ['relu1_2', 'relu2_2', 'relu3_3', 'relu4_3', 'relu5_3'])
        out = vgg_outputs(h_relu1_2, h_relu2_2, h_relu3_3, h_relu4_3, h_relu5_3)
        return out


def normalize_tensor(x,eps=1e-10):
    norm_factor = torch.sqrt(torch.sum(x**2,dim=1,keepdim=True))
    return x/(norm_factor+eps)


def spatial_average(x, keepdim=True):
    return x.mean([2,3],keepdim=keepdim)


URL_MAP = {
    "vgg_lpips": "https://huggingface.co/24yearsold/lpip/resolve/main/vgg.pth"
}


class LPIPS(nn.Module):
    # Learned perceptual metric
    def __init__(self, use_dropout=True):
        super().__init__()
        self.scaling_layer = ScalingLayer()
        self.chns = [64, 128, 256, 512, 512]  # vg16 features
        self.net = vgg16(pretrained=True, requires_grad=False)
        self.lin0 = NetLinLayer(self.chns[0], use_dropout=use_dropout)
        self.lin1 = NetLinLayer(self.chns[1], use_dropout=use_dropout)
        self.lin2 = NetLinLayer(self.chns[2], use_dropout=use_dropout)
        self.lin3 = NetLinLayer(self.chns[3], use_dropout=use_dropout)
        self.lin4 = NetLinLayer(self.chns[4], use_dropout=use_dropout)
        self.load_from_pretrained()
        for param in self.parameters():
            param.requires_grad = False

    def load_from_pretrained(self, name="vgg_lpips"):
        self.load_state_dict(torch.hub.load_state_dict_from_url(URL_MAP[name], map_location=torch.device("cpu")), strict=False)

    def forward(self, input, target):
        in0_input, in1_input = (self.scaling_layer(input), self.scaling_layer(target))
        outs0, outs1 = self.net(in0_input), self.net(in1_input)
        feats0, feats1, diffs = {}, {}, {}
        lins = [self.lin0, self.lin1, self.lin2, self.lin3, self.lin4]
        for kk in range(len(self.chns)):
            feats0[kk], feats1[kk] = normalize_tensor(outs0[kk]), normalize_tensor(outs1[kk])
            diffs[kk] = (feats0[kk] - feats1[kk]) ** 2

        res = [spatial_average(lins[kk].model(diffs[kk]), keepdim=True) for kk in range(len(self.chns))]
        val = res[0]
        for l in range(1, len(self.chns)):
            val += res[l]
        return val


def adopt_weight(weight, global_step, threshold=0, value=0.):
    if global_step < threshold:
        weight = value
    return weight


def hinge_d_loss(logits_real, logits_fake, weight_mat=1):
    loss_real = torch.mean(F.relu(1. - logits_real) * weight_mat)
    loss_fake = torch.mean(F.relu(1. + logits_fake) * weight_mat)
    d_loss = 0.5 * (loss_real + loss_fake)
    return d_loss


def vanilla_d_loss(logits_real, logits_fake):
    d_loss = 0.5 * (
        torch.mean(torch.nn.functional.softplus(-logits_real)) +
        torch.mean(torch.nn.functional.softplus(logits_fake)))
    return d_loss



def group_norm(in_channels, num_groups=32):
    if in_channels < num_groups:
        num_groups = in_channels
    return nn.GroupNorm(num_groups=num_groups, num_channels=in_channels, eps=1e-6, affine=True)


def modpad_tensor(tensor, mods, mode, **kwargs):
    H, W = tensor.shape[-2:]
    modH, modW = mods
    leftH = H % modH
    padH = modH - leftH if leftH else 0
    leftW = W % modW
    padW = modW - leftW if leftW else 0
    pads = (0, padW, 0, padH)
    padded = F.pad(tensor, pads, mode, **kwargs)
    return padded

def tensor2grey(img_tensor):
    r, g, b = img_tensor[:, [0]], img_tensor[:, [1]], img_tensor[:, [2]]
    l_img = (0.2989 * r + 0.587 * g + 0.114 * b).to(img_tensor.dtype)
    return l_img

def normalize_tensor(x,eps=1e-6):
    norm_factor = torch.sqrt(torch.sum(x**2,dim=1,keepdim=True))
    return x/(norm_factor+eps)



class ConcatTupleLayer(nn.Module):
    def forward(self, x):
        assert isinstance(x, tuple)
        x_l, x_g = x
        assert torch.is_tensor(x_l) or torch.is_tensor(x_g)
        if not torch.is_tensor(x_g):
            return x_l
        return torch.cat(x, dim=1)



class HakuNLayerDiscriminator(nn.Module):
    """
    Modern patch of NLayerDiscriminator
    LeakyReLU -> Mish/SiLU
    BatchNorm2d/ActNorm -> LayerNorm/GroupNorm
    """

    def __init__(self, input_nc=3, ndf=64, n_layers=3, gruops=1):
        """Construct a PatchGAN discriminator
        Parameters:
            input_nc (int)  -- the number of channels in input images
            ndf (int)       -- the number of filters in the last conv layer
            n_layers (int)  -- the number of conv layers in the discriminator
            groups (int)    -- the number of groups in GroupNorm, 1 = LayerNorm
        """
        super(HakuNLayerDiscriminator, self).__init__()

        kw = 4
        padw = 1
        sequence = [
            nn.Conv2d(input_nc, ndf, kernel_size=kw, stride=2, padding=padw),
            nn.Mish(),
        ]
        nf_mult = 1
        nf_mult_prev = 1
        for n in range(1, n_layers):  # gradually increase the number of filters
            nf_mult_prev = nf_mult
            nf_mult = min(2**n, 8)
            sequence += [
                nn.Conv2d(
                    ndf * nf_mult_prev,
                    ndf * nf_mult,
                    kernel_size=kw,
                    stride=2,
                    padding=padw,
                ),
                nn.GroupNorm(gruops, ndf * nf_mult),
                nn.Mish(),
            ]

        nf_mult_prev = nf_mult
        nf_mult = min(2**n_layers, 8)
        sequence += [
            nn.Conv2d(
                ndf * nf_mult_prev,
                ndf * nf_mult,
                kernel_size=kw,
                stride=1,
                padding=padw,
            ),
            nn.GroupNorm(gruops, ndf * nf_mult),
            nn.Mish(),
        ]

        sequence += [
            nn.Conv2d(ndf * nf_mult, 1, kernel_size=kw, stride=1, padding=padw)
        ]  # output 1 channel prediction map
        self.main = nn.Sequential(*sequence)

    def forward(self, input):
        """Standard forward."""
        return self.main(input)



class NLayerDiscriminator(nn.Module):
    """Defines a PatchGAN discriminator as in Pix2Pix
        --> see https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix/blob/master/models/networks.py
    """
    def __init__(self, input_nc=3, ndf=64, n_layers=3, use_actnorm=False, norm_layer = group_norm):
        """Construct a PatchGAN discriminator
        Parameters:
            input_nc (int)  -- the number of channels in input images
            ndf (int)       -- the number of filters in the last conv layer
            n_layers (int)  -- the number of conv layers in the discriminator
            norm_layer      -- normalization layer
        """
        super(NLayerDiscriminator, self).__init__()

        if type(norm_layer) == functools.partial:  # no need to use bias as BatchNorm2d has affine parameters
            use_bias = norm_layer.func != nn.BatchNorm2d
        else:
            use_bias = norm_layer != nn.BatchNorm2d

        # if not use_actnorm:
        #     norm_layer = nn.BatchNorm2d
        # else:
        #     norm_layer = ActNorm
        # if type(norm_layer) == functools.partial:  # no need to use bias as BatchNorm2d has affine parameters
        #     use_bias = norm_layer.func != nn.BatchNorm2d
        # else:
        #     use_bias = norm_layer != nn.BatchNorm2d

        kw = 4
        padw = 1
        sequence = [nn.Conv2d(input_nc, ndf, kernel_size=kw, stride=2, padding=padw), nn.LeakyReLU(0.2, True)]
        nf_mult = 1
        nf_mult_prev = 1
        for n in range(1, n_layers):  # gradually increase the number of filters
            nf_mult_prev = nf_mult
            nf_mult = min(2 ** n, 8)
            sequence += [
                nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult, kernel_size=kw, stride=2, padding=padw, bias=use_bias),
                norm_layer(ndf * nf_mult),
                nn.LeakyReLU(0.2, True)
            ]

        nf_mult_prev = nf_mult
        nf_mult = min(2 ** n_layers, 8)
        sequence += [
            nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult, kernel_size=kw, stride=1, padding=padw, bias=use_bias),
            norm_layer(ndf * nf_mult),
            nn.LeakyReLU(0.2, True)
        ]

        sequence += [
            nn.Conv2d(ndf * nf_mult, 1, kernel_size=kw, stride=1, padding=padw)]  # output 1 channel prediction map
        self.main = nn.Sequential(*sequence)

    def forward(self, input):
        """Standard forward."""
        return self.main(input)
    

def tensorgrad(t: torch.Tensor):
    if t.shape[1] == 3:
        t = t.mean(dim=1, keepdim=True)
    v = torch.diff(torch.nn.functional.pad(t, [0, 1, 0, 0], mode='circular'), dim=3)
    h = torch.diff(torch.nn.functional.pad(t, [0, 0, 0, 1], mode='circular'), dim=2)
    grad = torch.cat([v, h], dim=1)
    return grad
    

class Discriminator_UNet(nn.Module):
    """Defines a U-Net discriminator with spectral normalization (SN)"""

    def __init__(self, input_nc=3, ndf=64, disc_grad: bool = False):
        super(Discriminator_UNet, self).__init__()

        self.disc_grad = disc_grad

        if disc_grad:
            input_nc = 5

        print('init unet sn, ', disc_grad)

        from torch.nn.utils import spectral_norm
        norm = spectral_norm

        self.conv0 = nn.Conv2d(input_nc, ndf, kernel_size=3, stride=1, padding=1)

        self.conv1 = norm(nn.Conv2d(ndf, ndf * 2, 4, 2, 1, bias=False))
        self.conv2 = norm(nn.Conv2d(ndf * 2, ndf * 4, 4, 2, 1, bias=False))
        self.conv3 = norm(nn.Conv2d(ndf * 4, ndf * 8, 4, 2, 1, bias=False))
        # upsample
        self.conv4 = norm(nn.Conv2d(ndf * 8, ndf * 4, 3, 1, 1, bias=False))
        self.conv5 = norm(nn.Conv2d(ndf * 4, ndf * 2, 3, 1, 1, bias=False))
        self.conv6 = norm(nn.Conv2d(ndf * 2, ndf, 3, 1, 1, bias=False))

        # extra
        self.conv7 = norm(nn.Conv2d(ndf, ndf, 3, 1, 1, bias=False))
        self.conv8 = norm(nn.Conv2d(ndf, ndf, 3, 1, 1, bias=False))

        self.conv9 = nn.Conv2d(ndf, 1, 3, 1, 1)
        print('using the UNet discriminator')

    def forward(self, x):

        if self.disc_grad:
            x = torch.cat([tensorgrad(x), x], dim=1)


        x0 = F.leaky_relu(self.conv0(x), negative_slope=0.2, inplace=True)
        x1 = F.leaky_relu(self.conv1(x0), negative_slope=0.2, inplace=True)
        x2 = F.leaky_relu(self.conv2(x1), negative_slope=0.2, inplace=True)
        x3 = F.leaky_relu(self.conv3(x2), negative_slope=0.2, inplace=True)

        # upsample
        x3 = F.interpolate(x3, scale_factor=2, mode='bilinear', align_corners=False)
        x4 = F.leaky_relu(self.conv4(x3), negative_slope=0.2, inplace=True)

        x4 = x4 + x2
        x4 = F.interpolate(x4, scale_factor=2, mode='bilinear', align_corners=False)
        x5 = F.leaky_relu(self.conv5(x4), negative_slope=0.2, inplace=True)

        x5 = x5 + x1
        x5 = F.interpolate(x5, scale_factor=2, mode='bilinear', align_corners=False)
        x6 = F.leaky_relu(self.conv6(x5), negative_slope=0.2, inplace=True)

        x6 = x6 + x0

        # extra
        out = F.leaky_relu(self.conv7(x6), negative_slope=0.2, inplace=True)
        out = F.leaky_relu(self.conv8(out), negative_slope=0.2, inplace=True)
        out = self.conv9(out)

        return out


class LPIPSWithDiscriminator(nn.Module):
    def __init__(self, disc_start=1, logvar_init=0.0, kl_weight=0.000001, pixelloss_weight=1.0,
                 disc_num_layers=3, disc_in_channels=3, disc_factor=1.0, disc_weight=0.5,
                 perceptual_weight=1.0, use_actnorm=False,
                 disc_loss="hinge", discriminator='nlayer',
                 disc_grad=False):

        super().__init__()
        assert disc_loss in ["hinge", "vanilla"]
        self.kl_weight = kl_weight
        self.pixel_weight = pixelloss_weight
    

        if perceptual_weight > 0:
            self.perceptual_loss = LPIPS().eval()
        self.perceptual_weight = perceptual_weight

        # output log variance
        self.logvar = nn.Parameter(torch.ones(size=()) * logvar_init)

        if discriminator is None or discriminator == 'nlayer':
            self.discriminator = NLayerDiscriminator(input_nc=disc_in_channels,
                                                    n_layers=disc_num_layers,
                                                    use_actnorm=use_actnorm
                                                    ).apply(weights_init)
        elif discriminator == 'unet_sn':
            self.discriminator = Discriminator_UNet(disc_in_channels, disc_grad=disc_grad).apply(weights_init)
        else:
            raise ValueError(f'invalid discriminator {discriminator}')
        self.discriminator_iter_start = disc_start
        self.disc_loss = hinge_d_loss if disc_loss == "hinge" else vanilla_d_loss
        self.disc_factor = disc_factor
        self.discriminator_weight = disc_weight


    def calculate_adaptive_weight(self, nll_loss, g_loss, last_layer=None):
        if last_layer is not None:
            nll_grads = torch.autograd.grad(nll_loss, last_layer, retain_graph=True)[0]
            g_grads = torch.autograd.grad(g_loss, last_layer, retain_graph=True)[0]
        else:
            nll_grads = torch.autograd.grad(nll_loss, self.last_layer[0], retain_graph=True)[0]
            g_grads = torch.autograd.grad(g_loss, self.last_layer[0], retain_graph=True)[0]

        d_weight = torch.norm(nll_grads) / (torch.norm(g_grads) + 1e-4)
        d_weight = torch.clamp(d_weight, 0.0, 1e4).detach()
        d_weight = d_weight * self.discriminator_weight
        return d_weight

    def forward(self, inputs, reconstructions, optimizer_idx,
                global_step, posteriors=None, last_layer=None, cond=None, split="train",
                weights=None, line_input=None, line_rec=None, gen_disc_loss=True, rgb_cond=None, alpha_reweight=1.0):


        # now the GAN part

        def _reweight_mean(loss_mat, weight_mat):
            if alpha_reweight == 1:
                return loss_mat.mean()
            _, c, h, w = weight_mat.shape
            if h != loss_mat.shape[-2] or w != loss_mat.shape[-1]:
                weight_mat = torch.nn.functional.interpolate(weight_mat, loss_mat.shape[-2:], mode='bilinear')
            if loss_mat.shape[1] == 4:
                return loss_mat[:, :1].mean() / 4 + (loss_mat[:, 1:] * weight_mat).mean() * 3 / 4
            else:
                return (loss_mat * weight_mat).mean()

        if alpha_reweight != 1:
            alpha_weight = torch.clip(inputs[:, [0]], 0, 1) * alpha_reweight
        else:
            alpha_weight = None

        if optimizer_idx == 0 or optimizer_idx is None:
            # generator update

            rec_loss = _reweight_mean(torch.abs(inputs.float() - reconstructions.float()), alpha_weight)
            if self.perceptual_weight > 0:
                p_loss = self.perceptual_loss(inputs.contiguous(), reconstructions.contiguous())
                rec_loss = rec_loss + self.perceptual_weight * p_loss

            # nll_loss = rec_loss / torch.exp(self.logvar.float()) + self.logvar.float()
            # weighted_nll_loss = nll_loss
            # if weights is not None:
            #     weighted_nll_loss = weights*nll_loss
            # weighted_nll_loss = torch.sum(weighted_nll_loss) / weighted_nll_loss.shape[0]
            # nll_loss = torch.sum(nll_loss) / nll_loss.shape[0]
            nll_loss = weighted_nll_loss = rec_loss

            if posteriors is not None:
                kl_loss = posteriors.kl()
                kl_loss = torch.sum(kl_loss) / kl_loss.shape[0]
            else:
                kl_loss = None

            if gen_disc_loss and self.disc_factor > 0.0:
                # print(reconstructions.shape)
                if rgb_cond is not None:
                    logits_fake = self.discriminator(torch.cat((reconstructions.contiguous(), rgb_cond), dim=1))
                else:
                    logits_fake = self.discriminator(reconstructions.contiguous())
                g_loss = _reweight_mean(-logits_fake.float(), alpha_weight)

                try:
                    d_weight = self.calculate_adaptive_weight(nll_loss, g_loss, last_layer=last_layer)
                except RuntimeError:
                    assert not self.training
                    d_weight = torch.tensor(0.0)
            else:
                d_weight = torch.tensor(0.0, device=nll_loss.device)
                g_loss = torch.tensor(0.0, device=nll_loss.device)

            disc_factor = adopt_weight(self.disc_factor, global_step, threshold=self.discriminator_iter_start)
            loss = weighted_nll_loss + d_weight * disc_factor * g_loss
            if kl_loss is not None:
                loss += self.kl_weight * kl_loss

            log = {"{}/total_loss".format(split): loss.clone().detach().mean(), "{}/logvar".format(split): self.logvar.detach().to(device=loss.device),
                   "{}/nll_loss".format(split): nll_loss.detach().mean(),
                   "{}/rec_loss".format(split): rec_loss.detach().mean(),
                   "{}/d_weight".format(split): d_weight.detach(),
                   "{}/disc_factor".format(split): torch.tensor(disc_factor, device=loss.device),
                   "{}/g_loss".format(split): g_loss.detach().mean(),
                   }
            if kl_loss is not None:
                log["{}/kl_loss".format(split)] = kl_loss.detach().mean()

            return loss, log

        if optimizer_idx == 1:
            # second pass for discriminator update
            if rgb_cond is None:
                logits_real = self.discriminator(inputs.contiguous().detach())
                logits_fake = self.discriminator(reconstructions.contiguous().detach())
            else:
                logits_real = self.discriminator(torch.cat((inputs.contiguous().detach(), rgb_cond), dim=1))
                logits_fake = self.discriminator(torch.cat((reconstructions.contiguous().detach(), rgb_cond), dim=1))

            disc_factor = self.disc_factor
            weight_mat = 1
            if alpha_reweight != 1:
                weight_mat = torch.nn.functional.interpolate(alpha_weight, logits_fake.shape[-2:], mode='bilinear')
            d_loss = disc_factor * self.disc_loss(logits_real.float(), logits_fake.float(), weight_mat=weight_mat)

            log = {"{}/disc_loss".format(split): d_loss.clone().detach().mean(),
                   "{}/logits_real".format(split): logits_real.detach().mean(),
                   "{}/logits_fake".format(split): logits_fake.detach().mean()
                   }
            return d_loss, log



import torch
import torch.nn as nn
import torch.nn.functional as F
import lpips
from torchvision import models
from convnext_perceptual_loss import ConvNextPerceptualLoss, ConvNextType


class VGGFeatureExtractor(nn.Module):
    def __init__(self, layer_index=23):
        super(VGGFeatureExtractor, self).__init__()
        vgg16 = models.vgg16(pretrained=True)
        self.features = nn.Sequential(*list(vgg16.features.children())[:layer_index])
        for param in self.features.parameters():
            param.requires_grad = False

    def forward(self, x):
        return self.features(x)


class PerceptualLoss(nn.Module):
    def __init__(self, layer_index=23, loss_type="mse"):
        super(PerceptualLoss, self).__init__()
        self.feature_extractor = VGGFeatureExtractor(layer_index=layer_index)
        self.loss_type = loss_type

    def forward(self, x_real, x_recon):
        feat_real = self.feature_extractor(x_real)
        feat_recon = self.feature_extractor(x_recon)

        if self.loss_type == "l1":
            loss = F.l1_loss(feat_recon, feat_real)
        else:
            loss = F.mse_loss(feat_recon, feat_real)

        return loss


class LPIPSLoss(nn.Module):
    def __init__(self, net="alex"):
        super(LPIPSLoss, self).__init__()
        self.lpips_model = lpips.LPIPS(net=net)
        self.lpips_model.eval().requires_grad_(False)

    def forward(self, x_real, x_recon):
        loss = self.lpips_model(x_real, x_recon)
        return loss.mean()


class ConvNeXtPerceptualLoss(nn.Module):
    def __init__(
        self,
        model_type=ConvNextType.BASE,
        feature_layers=[0, 2, 4, 6, 8, 10, 12, 14],
        feature_weights=None,
        use_gram=False,
        input_range=(0, 1),
        layer_weight_decay=0.99,
        device="cpu",
    ):
        super(ConvNeXtPerceptualLoss, self).__init__()
        self.model = ConvNextPerceptualLoss(
            model_type=model_type,
            feature_layers=feature_layers,
            feature_weights=feature_weights,
            use_gram=use_gram,
            input_range=input_range,
            layer_weight_decay=layer_weight_decay,
            device=device,
        )

    def forward(self, x_real, x_recon):
        loss = self.model(x_real, x_recon)
        return loss



loss_table = {
    "mse": nn.MSELoss,
    "l1": nn.L1Loss,
    "huber": nn.HuberLoss,
    "gnll": nn.GaussianNLLLoss,
}

def srgb_to_oklab(srgb: torch.Tensor) -> torch.Tensor:
    # Convert to linear RGB space.
    rgb = torch.where(
        srgb <= 0.04045,
        srgb / 12.92,
        # Clamping avoids NaNs in backwards pass
        ((torch.clamp(srgb, min=0.04045) + 0.055) / 1.055) ** 2.4
    )

    # Convert RGB to LMS (cone response)
    t_rgb_lms = torch.tensor([
        [0.4122214708, 0.5363325363, 0.0514459929],
        [0.2119034982, 0.6806995451, 0.1073969566],
        [0.0883024619, 0.2817188376, 0.6299787005]
    ], dtype=srgb.dtype, device=srgb.device)

    lin_lms = torch.tensordot(rgb, t_rgb_lms, dims=([1], [0]))

    # Cone response cuts off at low light and we rely on rods, but assume a
    # linear response in low light to preserve differentiablity.
    # (2/255) / 12.92, which is roughly in the range of scotopic vision
    # (2e-6 cd/m^2) given a bright 800x600 CRT at 250 cd/m^2.

    # Apply nonlinearity to LMS
    X = 6e-4
    A = (X ** (1/3)) / X

    lms = torch.where(
        lin_lms <= X,
        lin_lms * A,
        # Clamping avoids NaNs in backwards pass
        torch.clamp(lin_lms, min=X) ** (1/3)
    )

    # Convert LMS to Oklab
    t_lms_oklab = torch.tensor([
        [0.2104542553, 0.7936177850, -0.0040720468],
        [1.9779984951, -2.4285922050, 0.4505937099],
        [0.0259040371, 0.7827717662, -0.8086757660]
    ], dtype=srgb.dtype, device=srgb.device)

    return torch.tensordot(lms, t_lms_oklab, dims=([1], [0]))


def noise_alpha_blend(x_srgba, noise=None, concat_alpha=True):
    x_alpha = torch.clip(x_srgba[:, [0]], 0, 1)
    x_srgb = x_srgba[:, 1:]
    if noise is None:
        noise = torch.randn_like(x_srgb)
    blended = noise * (1 - x_alpha) + x_srgb * x_alpha
    if concat_alpha:
        blended = torch.concat([x_srgba[:, [0]], blended], dim=1)
    return blended, noise


class ReconLoss(nn.Module):
    def __init__(
        self,
        loss_type="mse",
        lpips_net="alex",
        loss_colorspace="srgb",
        convnext_type=None,
        convnext_kwargs={},
        loss_weights={},
        noise_compose=False
    ):
        super(ReconLoss, self).__init__()
        self.loss = loss_table[loss_type]()
        self.loss_weight = loss_weights.get(loss_type, 1.0)
        self.loss_colorspace = loss_colorspace
        if lpips_net is not None:
            self.lpips_loss = LPIPSLoss(lpips_net)
            self.lpips_weight = loss_weights.get("lpips", 1.0)
        else:
            self.lpips_loss = None
        if convnext_type is not None:
            self.convn_loss = ConvNeXtPerceptualLoss(
                model_type=convnext_type, **convnext_kwargs
            )
            self.convn_weight = loss_weights.get("convnext", 1.0)
        else:
            self.convn_loss = None
        self.noise_compose = noise_compose

    def forward(self, x_real, x_recon):
        if isinstance(self.loss, nn.GaussianNLLLoss):
            x_recon, var = torch.split(
                x_recon, (x_real.size(1), x_recon.size(1) - x_real.size(1)), dim=1
            )
            # var = var.expand(-1, x_real.size(1), -1, -1)

        # losses relying on trained networks need to stay as sRGB
        x_real_srgb = x_real
        x_recon_srgb = x_recon
        if self.noise_compose and x_real_srgb.shape[1] == 4 and x_recon_srgb.shape[1] == 4:
            x_real_srgb, noise_map = noise_alpha_blend(x_real_srgb, concat_alpha=True)
            x_recon_srgb, _ = noise_alpha_blend(x_recon_srgb, concat_alpha=True, noise=noise_map)


        if self.loss_colorspace == "srgb":
            pass # assumed that pixel data is in sRGB space
        elif self.loss_colorspace == "oklab":
            x_real = srgb_to_oklab(x_real)
            x_recon = srgb_to_oklab(x_recon)
        else:
            raise NotImplementedError

        if isinstance(self.loss, nn.GaussianNLLLoss):
            base = self.loss(x_recon, x_real, torch.abs(var) + 1) * self.loss_weight
        else:
            base = self.loss(x_recon, x_real) * self.loss_weight


        # if self.noise_compose and x_real_srgb.shape[1] == 4 and x_recon_srgb.shape[1] == 4:
        #     x_real_alpha = x_real_srgb[:, [0]]
        #     x_real_srgb = x_real_srgb[:, 1:]
        #     noise = torch.randn_like(x_real_srgb)
        #     x_real_srgb = noise * (1 - x_real_alpha) + x_real_srgb * x_real_alpha

        #     x_recon_alpha = torch.clip(x_recon_srgb[:, [0]], 0, 1)
        #     x_recon_srgb = x_recon_srgb[:, 1:]
        #     x_recon_srgb = noise * (1 - x_recon_alpha) + x_recon_srgb * x_recon_alpha

        if x_real_srgb.shape[1] == 4:
            x_real_srgb = x_real_srgb[:, 1:]
            x_recon_srgb = x_recon_srgb[:, 1:]
            # print(x_real_srgb.shape, x_recon_srgb.shape)
            # from utils.torch_utils import img2tensor, tensor2img
            # from utils.io_utils import save_tmp_img
            # xr = tensor2img(x_real_srgb[1], denormalize=True)
            # xrecon = tensor2img(x_recon_srgb[1], denormalize=True)
            # save_tmp_img(np.concatenate([xr, xrecon]))
            # raise

        if self.lpips_loss is not None:
            lpips = self.lpips_loss(x_recon_srgb, x_real_srgb)
            base += lpips * self.lpips_weight
        if self.convn_loss is not None:
            convn = self.convn_loss(x_recon_srgb, x_real_srgb)
            base += convn * self.convn_weight
        return base




class KeplerQuantizerRegLoss(nn.Module):
    def __init__(
        self,
        embed_dim,
        num_embed=1024,
        partitions=4,
        scale=1,
        beta=0.25,
        use_kepler_loss=False,
        kl_weight=1e-8,
    ):
        super(KeplerQuantizerRegLoss, self).__init__()
        self.quantizer = KeplerQuantizer(
            embed_dim=embed_dim,
            scale=scale,
            partitions=partitions,
            n_embed=num_embed,
            beta=beta,
            kepler_loss=KeplerLoss(
                use=use_kepler_loss, kl_weight=kl_weight, n_e=int(num_embed * scale)
            ),
            legacy=True,
        )

    def forward(self, z):
        quantized, loss = self.quantizer(z)
        return loss.mean()