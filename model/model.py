from collections import OrderedDict
import os
import torch.functional as F
from torch.optim import lr_scheduler
import logging
import torch.nn as nn
import model.networks as networks
from model.base_model import BaseModel
from copy import deepcopy
import torch

eps = torch.finfo(torch.float32).eps

logger = logging.getLogger('base')


class EMA(nn.Module):
    def __init__(self, model, decay=0.9999, device=None):
        super(EMA, self).__init__()
        self.module = deepcopy(model)  # 保留一个模型的备份，model的参数更新很快，我们只使用model的一部分参数，dacay是旧参数，1-decay是model的新参数
        self.module.eval()
        self.decay = decay
        self.device = device  # 如果设置device就会在不同的设备上执行ema
        if self.device is not None:
            self.module.to(device=device)

    def _update(self, model, update_fn):
        with torch.no_grad():
            for ema_v, model_v in zip(self.module.state_dict().values(), model.state_dict().values()):
                if self.device is not None:
                    model_v = model_v.to(device=self.device)
                ema_v.copy_(update_fn(ema_v, model_v))

    def update(self, model):
        self._update(model, update_fn=lambda e, m: self.decay * e + (1 - self.decay) * m)

    def set(self, model):
        self._update(model, update_fn=lambda e, m: m)


class DDPM(BaseModel):
    def __init__(self, opt):
        super(DDPM, self).__init__(opt)
        self.optG = None
        self.scheduler = None
        self.netG = self.set_device(networks.define_G(opt))
        self.spatical_model = self.set_device(networks.define_spatical(opt))
        self.spectral_model = self.set_device(networks.define_spectral(opt))
        self.schedule_phase = None
        self.set_loss()
        self.loss = nn.L1Loss()
        self.up = nn.Upsample(scale_factor=4, mode='nearest')

        if self.opt['phase'] == 'train':  # 如果有预训练模型，之前初始化的参数，这里初始化的scheduler和optG会被覆盖掉
            self.netG.train()
            optim_params = list(self.netG.parameters())  # 加载所有参数进行训练
            self.optG = torch.optim.AdamW(
                optim_params, lr=opt['train']["optimizer"]["lr"])
            self.scheduler = lr_scheduler.StepLR(self.optG, step_size=500000000)
            self.log_dict = OrderedDict()

        if self.opt['phase'] == 'train':
            self.spatical_model.train()
            self.spectral_model.train()
            optim_params = list(self.spatical_model.parameters())  # 加载所有参数进行训练
            self.optSpatical = torch.optim.AdamW(
                optim_params, lr=opt['train']["optimizer"]["lr"])
            self.schedulerSpatical = lr_scheduler.StepLR(self.optSpatical, step_size=500000000)
            optim_params = list(self.spectral_model.parameters())  # 加载所有参数进行训练
            self.optSpectral = torch.optim.AdamW(
                optim_params, lr=opt['train']["optimizer"]["lr"])
            self.schedulerSpectral = lr_scheduler.StepLR(self.optSpectral, step_size=500000000)

        self.load_network()
        self.print_network()

    def feed_data(self, data):
        self.data = self.set_device(data)

    def optimize_observation_models(self, Z):
        self.optSpatical.zero_grad()
        self.optSpectral.zero_grad()
        X2 = self.spatical_model(Z)
        Y1 = self.spectral_model(Z)

        loss1 = self.loss(X2, self.data['LR'])
        loss2 = self.loss(Y1, self.data['PAN'])
        loss3 = self.loss(self.spectral_model(X2), self.spatical_model(Y1))
        loss = loss1 + loss2 + loss3
        loss.backward()
        self.optSpectral.step()
        self.optSpatical.step()
        self.log_dict['l_observe'] = loss.item()

    def fine_tune_paramter(self):
        self.spatical_model.eval()
        self.spectral_model.eval()
        with torch.no_grad():
            X3 = self.spatical_model(self.data['LR'])
            Y2 = self.spatical_model(self.data['PAN'])
            X3_UP = self.up(X3)
        data = {'LR': X3, 'PAN': Y2, 'MS': X3_UP, 'HR': self.data['LR'] * 2 - 1,
                'Index': self.data['Index']}
        # print(X3.shape, Y2.shape, X3_UP.shape, data['HR'].shape,data['HR'].type)
        self.optG.zero_grad()
        l_pix = self.netG(data)  # 调用forward函数，计算扩散loss
        b, c, h, w = self.data['HR'].shape
        l_pix = l_pix.sum() / int(b * c * h * w)
        l_pix.backward()
        self.optG.step()
        self.log_dict['l_pix'] = l_pix.item()
        self.spatical_model.train()
        self.spectral_model.train()

    def optimize_parameters(self):
        self.optG.zero_grad()
        l_pix = self.netG(self.data)  # 调用forward函数，计算扩散loss
        b, c, h, w = self.data['HR'].shape
        l_pix = l_pix.sum() / int(b * c * h * w)
        l_pix.backward()
        self.optG.step()
        self.log_dict['l_pix'] = l_pix.item()

    def test(self, continous=False, w=3.0):  # 传入的continous为True
        self.netG.eval()
        with torch.no_grad():
            if isinstance(self.netG, nn.DataParallel):
                self.SR = self.netG.module.super_resolution(
                    self.data, continous, w=w)
            else:
                self.SR = self.netG.super_resolution(  # SR是连续采样得到的结果
                    self.data, continous, w=w)
        self.netG.train()

    def set_loss(self):
        if isinstance(self.netG, nn.DataParallel):
            self.netG.module.set_loss(self.device)
        else:
            self.netG.set_loss(self.device)

    def set_new_noise_schedule(self, schedule_opt, schedule_phase='train'):
        if self.schedule_phase is None or self.schedule_phase != schedule_phase:
            self.schedule_phase = schedule_phase
            if isinstance(self.netG, nn.DataParallel):
                self.netG.module.set_new_noise_schedule(
                    schedule_opt, self.device)
            else:
                self.netG.set_new_noise_schedule(schedule_opt, self.device)

    def get_current_log(self):
        return self.log_dict

    def get_current_visuals(self):
        out_dict = OrderedDict()
        out_dict['SR'] = self.SR.detach().float().cpu()  # 预测的MS
        out_dict['MS'] = self.data['MS'].detach().float().cpu()  # 上采样的MS，pan2ms的GT
        out_dict['PAN'] = self.data['PAN'].float().cpu()
        out_dict['LR'] = self.data['LR'].detach().float().cpu()
        out_dict['HR'] = self.data['HR'].detach().float().cpu()
        return out_dict

    def print_network(self):
        s, n = self.get_network_description(self.netG)
        if isinstance(self.netG, nn.DataParallel):
            net_struc_str = '{} - {}'.format(self.netG.__class__.__name__,
                                             self.netG.module.__class__.__name__)
        else:
            net_struc_str = '{}'.format(self.netG.__class__.__name__)

        logger.info(
            'Network G structure: {}, with parameters: {:,d}'.format(net_struc_str, n))
        logger.info(s)

    def save_network(self, epoch, iter_step):
        gen_path = os.path.join(
            self.opt['path']['checkpoint'], 'I{}_E{}_gen.pth'.format(iter_step, epoch))
        opt_path = os.path.join(
            self.opt['path']['checkpoint'], 'I{}_E{}_opt.pth'.format(iter_step, epoch))
        # gen
        network = self.netG
        if isinstance(network, nn.DataParallel):
            network = network.module
        state_dict = network.state_dict()
        for key, param in state_dict.items():
            state_dict[key] = param.cpu()
        torch.save(state_dict, gen_path)
        # opt
        opt_state = {'epoch': epoch, 'iter': iter_step,
                     'scheduler': self.scheduler.state_dict(),
                     'optimizer': self.optG.state_dict()}
        torch.save(opt_state, opt_path)

        logger.info(
            'Saved model in [{:s}] ...'.format(gen_path))

    def load_network(self):
        load_path = self.opt['path']['resume']
        if load_path is not None:
            logger.info(
                'Loading pretrained model for G [{:s}] ...'.format(load_path))
            gen_path = '{}_gen.pth'.format(load_path)
            opt_path = '{}_opt.pth'.format(load_path)
            # gen
            network = self.netG
            if isinstance(self.netG, nn.DataParallel):
                network = network.module

            network.load_state_dict(torch.load(gen_path), False)

            if self.opt['phase'] == 'train':
                opt = torch.load(opt_path)
                self.begin_step = opt['iter']
                self.begin_epoch = opt['epoch']
                self.optG.load_state_dict(opt['optimizer'])
                self.scheduler.load_state_dict(opt['scheduler'])
