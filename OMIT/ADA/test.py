from augment import AugmentPipe
from torch_utils import  misc,training_stats
import torch
import numpy as np
augpipe_specs = {
    'blit': dict(xflip=1, rotate90=1, xint=1),
    'geom': dict(scale=1, rotate=1, aniso=1, xfrac=1),
    'color': dict(brightness=1, contrast=1, lumaflip=1, hue=1, saturation=1),
    'filter': dict(imgfilter=1),
    'noise': dict(noise=1),
    'cutout': dict(cutout=1),
    'bg': dict(xflip=1, rotate90=1, xint=1, scale=1, rotate=1, aniso=1, xfrac=1),
    'bgc': dict(xflip=1, rotate90=1, xint=1, scale=1, rotate=1, aniso=1, xfrac=1, brightness=1, contrast=1, lumaflip=1,
                hue=1, saturation=1),
    'bgcf': dict(xflip=1, rotate90=1, xint=1, scale=1, rotate=1, aniso=1, xfrac=1, brightness=1, contrast=1, lumaflip=1,
                 hue=1, saturation=1, imgfilter=1),
    'bgcfn': dict(xflip=1, rotate90=1, xint=1, scale=1, rotate=1, aniso=1, xfrac=1, brightness=1, contrast=1,
                  lumaflip=1, hue=1, saturation=1, imgfilter=1, noise=1),
    'bgcfnc': dict(xflip=1, rotate90=1, xint=1, scale=1, rotate=1, aniso=1, xfrac=1, brightness=1, contrast=1,
                   lumaflip=1, hue=1, saturation=1, imgfilter=1, noise=1, cutout=1),
}
ada_stats = None
augment_pipe = AugmentPipe(**augpipe_specs['bgc']).train().requires_grad_(False).to("cuda")
augment_p=0
ada_interval=4
ada_target=0.6
augment_pipe.p.copy_(torch.as_tensor(augment_p))
print(augment_pipe.p)
ada_stats = training_stats.Collector(regex='Loss/signs/real')
input1=torch.randn(8,3,256,256).to("cuda")
print(augment_pipe(input1).shape)
ada_stats.update()
adjust = np.sign(1 - ada_target) * (8 * ada_interval) / (500 * 1000)
augment_pipe.p.copy_((augment_pipe.p + adjust).max(misc.constant(0, device="cuda")))
print(augment_pipe.p)