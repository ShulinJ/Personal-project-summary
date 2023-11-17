"""
StarGAN v2
Copyright (c) 2020-present NAVER Corp.

This work is licensed under the Creative Commons Attribution-NonCommercial
4.0 International License. To view a copy of this license, visit
http://creativecommons.org/licenses/by-nc/4.0/ or send a letter to
Creative Commons, PO Box 1866, Mountain View, CA 94042, USA.
"""
#
import os
from os.path import join as ospj
import time
import datetime
from munch import Munch
from copy import deepcopy
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parallel import DataParallel, DistributedDataParallel
from core.model import build_model
from core.checkpoint import CheckpointIO
from core.data_loader import InputFetcher
import core.utils as utils
from metrics.eval import calculate_metrics
from core.face_reg_irnet import get_model
from ADA.augment import AugmentPipe
import cv2
import numpy as np
class Solver(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        # self.network_identity = utils.ResNetArcFace("IRBlock", [2, 2, 2, 2], False)
        # self.load_network(self.network_identity, "arcface_resnet18.pth", True, None)
        # self.network_identity.eval()
        # for param in self.network_identity.parameters():
        #     param.requires_grad = False
        self.network_identity  = get_model("r50", fp16=False).to("cuda")
        self.network_identity.load_state_dict(torch.load("backbone.pth"))
        self.network_identity.eval()
        for param in self.network_identity.parameters():
            param.requires_grad = False
######################
        self.augpipe_specs = {
            'blit': dict(xflip=1, rotate90=1, xint=1),
            'geom': dict(scale=1, rotate=1, aniso=1, xfrac=1),
            'color': dict(brightness=1, contrast=1, lumaflip=1, hue=1, saturation=1),
            'filter': dict(imgfilter=1),
            'noise': dict(noise=1),
            'cutout': dict(cutout=1),
            'bg': dict(xflip=1, rotate90=1, xint=1, scale=1, rotate=1, aniso=1, xfrac=1),
            'bgc': dict(xflip=1, rotate90=1, xint=1, scale=1, rotate=1, aniso=1, xfrac=1, brightness=1, contrast=1,
                        lumaflip=1,
                        hue=1, saturation=1),
            'bgcf': dict(xflip=1, rotate90=1, xint=1, scale=1, rotate=1, aniso=1, xfrac=1, brightness=1, contrast=1,
                         lumaflip=1,
                         hue=1, saturation=1, imgfilter=1),
            'bgcfn': dict(xflip=1, rotate90=1, xint=1, scale=1, rotate=1, aniso=1, xfrac=1, brightness=1, contrast=1,
                          lumaflip=1, hue=1, saturation=1, imgfilter=1, noise=1),
            'bgcfnc': dict(xflip=1, rotate90=1, xint=1, scale=1, rotate=1, aniso=1, xfrac=1, brightness=1, contrast=1,
                           lumaflip=1, hue=1, saturation=1, imgfilter=1, noise=1, cutout=1),
        }
        self.augment_pipe = AugmentPipe(**self.augpipe_specs['bgc']).train().requires_grad_(False).to("cuda")
        self.augment_p = 0.3
        self.augment_pipe.p.copy_(torch.as_tensor(self.augment_p))





        self.cri_perceptual = utils.PerceptualLoss()
        self.criterionL2 = utils.L1Loss()
        self.nets, self.nets_ema = build_model(args)

        # below setattrs are to make networks be children of Solver, e.g., for self.to(self.device)
        for name, module in self.nets.items():
            utils.print_network(module, name)
            setattr(self, name, module)
        for name, module in self.nets_ema.items():
            setattr(self, name + '_ema', module)

        if args.mode == 'train':
            self.optims = Munch()
            for net in self.nets.keys():
                if net == 'fan' :
                    continue
                self.optims[net] = torch.optim.Adam(
                    params=self.nets[net].parameters(),
                    lr=args.f_lr if net == 'mapping_network' else args.lr,
                    betas=[args.beta1, args.beta2],
                    weight_decay=args.weight_decay)


            self.ckptios = [
                CheckpointIO(ospj(args.checkpoint_dir, '{}_nets.ckpt'), data_parallel=True, **self.nets),
                CheckpointIO(ospj(args.checkpoint_dir, '{}_nets_ema.ckpt'), data_parallel=True, **self.nets_ema),
                CheckpointIO(ospj(args.checkpoint_dir, '{}_optims.ckpt'), **self.optims)]
        else:
            self.ckptios = [CheckpointIO(ospj(args.checkpoint_dir, '{:06d}_nets_ema.ckpt'), data_parallel=True, **self.nets_ema)]

        self.to(self.device)
        for name, network in self.named_children():
            # Do not initialize the FAN parameters
            if ('ema' not in name) and ('fan' not in name) and ('network_identity' not in name) and  ('cri_perceptual' not in name):
                print('Initializing %s...' % name)
                network.apply(utils.he_init)
    def get_bare_model(self, net):
        """Get bare model, especially under wrapping with
        DistributedDataParallel or DataParallel.
        """
        if isinstance(net, (DataParallel, DistributedDataParallel)):
            net = net.module
        return net
    def load_network(self, net, load_path, strict=True, param_key='params'):
        """Load network.

        Args:
            load_path (str): The path of networks to be loaded.
            net (nn.Module): Network.
            strict (bool): Whether strictly loaded.
            param_key (str): The parameter key of loaded network. If set to
                None, use the root 'path'.
                Default: 'params'.
        """
        net = self.get_bare_model(net)
        load_net = torch.load(load_path, map_location=lambda storage, loc: storage)
        if param_key is not None:
            if param_key not in load_net and 'params' in load_net:
                param_key = 'params'
            load_net = load_net[param_key]
        # remove unnecessary 'module.'
        for k, v in deepcopy(load_net).items():
            if k.startswith('module.'):
                load_net[k[7:]] = v
                load_net.pop(k)
        net.load_state_dict(load_net, strict=strict)
    def _save_checkpoint(self, step):
        for ckptio in self.ckptios:
            ckptio.save(step)

    def _load_checkpoint(self, step):
        for ckptio in self.ckptios:
            ckptio.load(step)

    def _reset_grad(self):
        for optim in self.optims.values():
            optim.zero_grad()

    def train(self, loaders):
        args = self.args
        nets = self.nets
        nets_ema = self.nets_ema
        optims = self.optims

        # fetch random validation images for debugging
        fetcher = InputFetcher(loaders.src, loaders.ref, args.latent_dim, 'train')
        fetcher_val = InputFetcher(loaders.val, None, args.latent_dim, 'val')
        inputs_val = next(fetcher_val)

        # resume training if necessary
        if args.resume_iter > 0:
            self._load_checkpoint(args.resume_iter)

        # remember the initial value of ds weight
        initial_lambda_ds = args.lambda_ds

        print('Start training...')
        start_time = time.time()
        for i in range(args.resume_iter, args.total_iters):
            # fetch images and labels
            inputs = next(fetcher)
            x_real, y_org = inputs.x_src, inputs.y_src
            x_ref, x_ref2, y_trg = inputs.x_ref, inputs.x_ref2, inputs.y_ref
            z_trg, z_trg2 = inputs.z_trg, inputs.z_trg2
            # print(y_org,y_trg)
            masks = nets.fan.get_heatmap(x_real) if args.w_hpf > 0 else None

            # train the discriminator
            d_loss, d_losses_latent = compute_d_loss(
                nets, args, x_real, y_org, y_trg,z_trg=z_trg,  x_ref=x_ref, masks=masks)
            self._reset_grad()
            d_loss.backward()
            optims.discriminator.step()

            d_loss, d_losses_ref = compute_d_loss1(
                nets, args, x_real, y_org, y_trg,z_trg=x_ref2, x_ref=x_ref, masks=masks)
            self._reset_grad()
            d_loss.backward()
            optims.discriminator.step()

            # train the generator
            g_loss, g_losses_latent = compute_g_loss(nets, args, x_real, y_org, y_trg, self.network_identity,self.criterionL2,self.cri_perceptual,z_trg=z_trg,x_ref=x_ref, masks=masks)
            self._reset_grad()
            g_loss.backward()
            optims.generator.step()
            optims.style_encoder.step()
            optims.mapping_network.step()

            g_loss, g_losses_ref = compute_g_loss1(nets, args, x_real, y_org, y_trg, self.network_identity,self.criterionL2,self.cri_perceptual,z_trg=x_ref2,x_ref=x_ref, masks=masks)
            self._reset_grad()
            g_loss.backward()
            optims.generator.step()

            # compute moving average of network parameters
            moving_average(nets.generator, nets_ema.generator, beta=0.999)
            moving_average(nets.mapping_network, nets_ema.mapping_network, beta=0.999)
            moving_average(nets.style_encoder, nets_ema.style_encoder, beta=0.999)

            # decay weight for diversity sensitive loss
            if args.lambda_ds > 0:
                args.lambda_ds -= (initial_lambda_ds / args.ds_iter)

            # print out log info
            if (i+1) % args.print_every == 0:
                # print("mapping_loss_____",mapping_loss)
                elapsed = time.time() - start_time
                elapsed = str(datetime.timedelta(seconds=elapsed))[:-7]
                log = "Elapsed time [%s], Iteration [%i/%i], " % (elapsed, i+1, args.total_iters)
                all_losses = dict()
                for loss, prefix in zip([d_losses_latent,  g_losses_latent,],
                                        ['D/latent_', 'G/latent_']):
                    for key, value in loss.items():
                        all_losses[prefix + key] = value
                all_losses['G/lambda_ds'] = args.lambda_ds
                log += ' '.join(['%s: [%.4f]' % (key, value) for key, value in all_losses.items()])
                print(log)

            # generate images for debugging
            if (i+1) % args.sample_every == 0:
                os.makedirs(args.sample_dir, exist_ok=True)
                utils.debug_image(nets_ema, args, self.network_identity, inputs=inputs_val,step=i+1)

            # save model checkpoints
            if (i+1) % args.save_every == 0:
                self._save_checkpoint(step=i+1)

            # compute FID and LPIPS if necessary
            if (i+1) % args.eval_every == 0:
                calculate_metrics(nets_ema, args, i+1, mode='reference')

    @torch.no_grad()
    def sample(self, loaders):
        args = self.args
        nets_ema = self.nets_ema
        os.makedirs(args.result_dir, exist_ok=True)
        self._load_checkpoint(args.resume_iter)

        src = next(InputFetcher(loaders.src, None, args.latent_dim, 'test'))
        ref = next(InputFetcher(loaders.ref, None, args.latent_dim, 'test'))

        fname = ospj(args.result_dir, 'reference.jpg')
        print('Working on {}...'.format(fname))
        utils.translate_using_reference(nets_ema, args, src.x, ref.x, ref.y, fname)

        fname = ospj(args.result_dir, 'video_ref.mp4')
        print('Working on {}...'.format(fname))
        utils.video_ref(nets_ema, args, src.x, ref.x, ref.y, fname)

    @torch.no_grad()
    def evaluate(self):
        args = self.args
        nets_ema = self.nets_ema
        resume_iter = args.resume_iter
        self._load_checkpoint(args.resume_iter)
        calculate_metrics(nets_ema, args, step=resume_iter, mode='reference')
#########cycle
def compute_d_loss(nets, args, x_real, y_org, y_trg, z_trg=None, x_ref=None, masks=None):
    x_ref.requires_grad_()
    out = nets.discriminator(x_ref, y_trg)
    loss_real = adv_loss(out, 1)
    loss_reg = r1_reg(out, x_ref)
    with torch.no_grad():
        # face_vector_real = network_identity(x_real).detach()
        # s_trg = nets.style_encoder(x_real, y_trg).detach()
        s_trg = nets.mapping_network(z_trg, y_trg)
        x_fake = nets.generator(x_real, s_trg, masks=masks)
    out = nets.discriminator(x_fake, y_trg)
    loss_fake = adv_loss(out, 0)
    out = nets.discriminator(x_real, y_trg)
    loss_fake1 = adv_loss(out, 0)
    loss = loss_real + loss_fake+loss_fake1 + args.lambda_reg * loss_reg
    return loss, Munch(real=loss_real.item(),
                       fake=loss_fake.item(),
                       reg=loss_reg.item())
def compute_d_loss1(nets, args, x_real, y_org, y_trg, z_trg=None, x_ref=None, masks=None):
    # assert (z_trg is None) != (x_ref is None)
    x_ref.requires_grad_()
    out = nets.discriminator(x_ref, y_trg)
    loss_real = adv_loss(out, 1)
    loss_reg = r1_reg(out, x_ref)
    with torch.no_grad():
        # face_vector_real = network_identity(x_real).detach()
        s_trg = nets.style_encoder(z_trg, y_trg).detach()
        # s_trg = nets.mapping_network(face_vector_real, y_trg)
        x_fake = nets.generator(x_real, s_trg, masks=masks)
    out = nets.discriminator(x_fake, y_trg)
    loss_fake = adv_loss(out, 0)
    loss = loss_real + loss_fake + args.lambda_reg * loss_reg
    return loss, Munch(real=loss_real.item(),
                       fake=loss_fake.item(),
                       reg=loss_reg.item())
def compute_g_loss(nets, args, x_real, y_org, y_trg,network_identity,criterionL2,cri_perceptual,z_trg=None, x_ref=None, masks=None):
    s_trg = nets.mapping_network(z_trg, y_trg)
    x_fake = nets.generator(x_real, s_trg, masks=masks)
    face_vector_zh_real = network_identity(x_ref).detach()
    face_vector_real = network_identity(x_real).detach()
    s_trg2 = nets.style_encoder(x_real, y_org)
    x_fake2 = nets.generator(x_real, s_trg2, masks=masks)
    loss_cyc = torch.mean(torch.abs(x_fake2 - x_real))
    out = nets.discriminator(x_fake, y_trg)
    loss_adv = adv_loss(out, 1)
    # _,loss_vgg_style = cri_perceptual(x_real, x_fake)
    face_vector_fake= network_identity(x_fake)
    loss_id1 = criterionL2(face_vector_zh_real, face_vector_fake)
    # loss_id2 = criterionL2(face_vector_real, face_vector_fake)

    s_trg3 = nets.style_encoder(x_fake, y_trg)
    loss_sty = torch.mean(torch.abs(s_trg - s_trg3))
    loss = loss_adv + loss_cyc + loss_id1 * 4 + loss_sty  #+ loss_sty*2+loss_sty2*2#+loss_vgg*0.05+ loss_cyc+loss_vgg_style*0.05#+ args.lambda_cyc * loss_cyc       #+ loss_id*(1.0-args.lambda_ds)+loss_vgg*0.01#(1.1-args.lambda_ds)*
    return loss, Munch(adv=loss_adv.item(),
                       sty=loss_cyc.item(),
                       ds=loss_id1.item() * 2 * (2-args.lambda_ds),
                       cyc=loss_sty.item()* args.lambda_ds)
def compute_g_loss1(nets, args, x_real, y_org, y_trg,network_identity,criterionL2,cri_perceptual,z_trg=None, x_ref=None, masks=None):
    s_trg = nets.style_encoder(z_trg, y_trg)
    x_fake = nets.generator(x_real, s_trg, masks=masks)
    face_vector_zh_real = network_identity(x_ref).detach()
    face_vector_real = network_identity(x_real).detach()
    s_trg2 = nets.style_encoder(x_real, y_org)
    x_fake2 = nets.generator(x_real, s_trg2, masks=masks)
    loss_cyc = torch.mean(torch.abs(x_fake2 - x_real))
    out = nets.discriminator(x_fake, y_trg)
    loss_adv = adv_loss(out, 1)
    # _,loss_vgg_style = cri_perceptual(x_real, x_fake)
    face_vector_fake = network_identity(x_fake)
    loss_id1 = criterionL2(face_vector_zh_real, face_vector_fake)
    # loss_id2 = criterionL2(face_vector_real, face_vector_fake)
    s_trg3 = nets.style_encoder(x_fake, y_trg)
    loss_sty = torch.mean(torch.abs(s_trg - s_trg3))
    loss = loss_adv + loss_cyc + loss_id1 * 4  + loss_sty  # + loss_sty*2+loss_sty2*2#+loss_vgg*0.05+ loss_cyc+loss_vgg_style*0.05#+ args.lambda_cyc * loss_cyc       #+ loss_id*(1.0-args.lambda_ds)+loss_vgg*0.01#(1.1-args.lambda_ds)*
    return loss, Munch(adv=loss_adv.item(),
                       sty=loss_cyc.item(),
                       ds=loss_id1.item() * 2 * (2 - args.lambda_ds),
                       cyc=loss_sty.item() * args.lambda_ds)

def moving_average(model, model_test, beta=0.999):
    for param, param_test in zip(model.parameters(), model_test.parameters()):
        param_test.data = torch.lerp(param.data, param_test.data, beta)
def adv_loss(logits, target):
    assert target in [1, 0]
    targets = torch.full_like(logits, fill_value=target)
    loss = F.binary_cross_entropy_with_logits(logits, targets)
    return loss
def r1_reg(d_out, x_in):
    # zero-centered gradient penalty for real images
    batch_size = x_in.size(0)
    grad_dout = torch.autograd.grad(
        outputs=d_out.sum(), inputs=x_in,
        create_graph=True, retain_graph=True, only_inputs=True
    )[0]
    grad_dout2 = grad_dout.pow(2)
    assert(grad_dout2.size() == x_in.size())
    reg = 0.5 * grad_dout2.view(batch_size, -1).sum(1).mean(0)
    return reg