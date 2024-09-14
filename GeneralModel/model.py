from collections import OrderedDict
import os
from torch.optim import lr_scheduler
import logging
import torch.nn as nn
import GeneralModel.networks as networks
from .base_model import BaseModel
from copy import deepcopy
import torch
from transformers import get_scheduler

logger = logging.getLogger('base')

class DDPM(BaseModel):
    def __init__(self, opt):
        super(DDPM, self).__init__(opt)
        self.optG = None
        self.scheduler = None
        self.netG = self.set_device(networks.define_General(opt))
        self.schedule_phase = None
        self.set_loss()
        if self.opt['phase'] == 'train':  # 如果有预训练模型，之前初始化的参数，这里初始化的scheduler和optG会被覆盖掉
            self.netG.train()
            optim_params = []
            for name, param in self.netG.named_parameters():
                if "clip_text" in name:  # Frozen clip model
                    continue
                print(name, param.shape)
                optim_params.append(param)
            self.optG = torch.optim.AdamW(
                optim_params, lr=opt['train']["optimizer"]["lr"], weight_decay=1e-4)
            self.scheduler = get_scheduler("linear",self.optG,num_warmup_steps=100,num_training_steps=opt['train']["max_iter"])
            self.log_dict = OrderedDict()
        self.load_network()
        self.print_network()

    def feed_data(self, data):
        self.data = self.set_device(data)

    def optimize_parameters(self, prompt=None):
        l_pix = self.netG(self.data, prompt).sum()
        l_pix.backward()
        self.optG.step()
        self.scheduler.step()
        self.optG.zero_grad()
        self.log_dict['l_pix'] = l_pix.detach()
        self.log_dict['lr'] = self.optG.state_dict()['param_groups'][0]['lr']

    def test(self, continous=False, prompt="QB", guidance=3.0):  # 传入的continous为True
        self.netG.eval()
        with torch.no_grad():
            if isinstance(self.netG, nn.DataParallel):
                self.SR = self.netG.module.super_resolution(
                    self.data, continous, prompt, guidance)
            else:
                self.SR = self.netG.super_resolution(  # SR是连续采样得到的结果
                    self.data, continous, prompt, guidance)
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
        out_dict['HR'] = self.data['HR'].detach().float().cpu()  # 目前只用了前两个 [-1,1]分布
        out_dict['MS'] = self.data['MS'].detach().float().cpu()  # 上采样的MS，pan2ms的GT
        out_dict['PAN'] = self.data['PAN'].float().cpu()
        out_dict['LR'] = self.data['LR'].detach().float().cpu()

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

    def save_network(self, iter_step):
        gen_path = os.path.join(
            self.opt['path']['checkpoint'], 'I{}_gen.pth'.format(iter_step))
        opt_path = os.path.join(
            self.opt['path']['checkpoint'], 'I{}_opt.pth'.format(iter_step))
        # gen
        network = self.netG
        if isinstance(network, nn.DataParallel):
            network = network.module
        state_dict = network.state_dict()
        for key, param in state_dict.items():
            state_dict[key] = param.cpu()
        torch.save(state_dict, gen_path)
        # opt
        opt_state = {'iter': iter_step,
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
                # self.optG.load_state_dict(opt['optimizer'])
                # self.scheduler.load_state_dict(opt['scheduler'])
