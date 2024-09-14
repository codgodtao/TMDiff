import random
import shutil
from copy import deepcopy

import math
import torch
import data as Data
import GeneralModel as Model
import argparse
import logging
import core.logger as Logger
import os.path as osp
import os
import numpy as np
import scipy.io as scio
from core.metrics import SSIM_numpy, SAM_numpy
from utils.util import get_data_generator
import wandb
from core.mylib import add_prefix, dct2str

eps = torch.finfo(torch.float32).eps


def seed_torch(seed=3407):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)  # 为了禁止hash随机化，使得实验可复现
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU.
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = True


def copy_source(file, output_dir):
    shutil.copyfile(file, os.path.join(output_dir, os.path.basename(file)))


def normlization(data, min_max=(-1.0, 1.0)):
    data = data.squeeze().cpu().clamp_(*min_max).numpy()
    data = (data - min_max[0]) / (min_max[1] - min_max[0])
    return np.transpose(data, (1, 2, 0))  # H W C


def sample_data(qb, gf2):
    # print("sample prob is :", qb_prob, gf2_prob, 1 - qb_prob - gf2_prob)
    p = random.random()
    if p < qb:
        return next(train_qb_generator), "QB"
    elif p < qb + gf2:
        return next(train_gf2_generator), "GF2"
    else:
        return next(train_wv3_generator), "WV3"


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--config', type=str, default='config/general_finetune.json',
                        help='JSON file for configuration')
    parser.add_argument('-p', '--phase', type=str, choices=['train', 'val'],
                        help='Run either train(training) or val(generation)', default='val')
    parser.add_argument('-gpu', '--gpu_ids', type=str, default=None)
    parser.add_argument('-debug', '-d', action='store_true')
    parser.add_argument('-log_eval', action='store_true')

    # parse configs
    args = parser.parse_args()
    opt = Logger.parse(args)
    opt = Logger.dict_to_nonedict(opt)
    result_path = opt['path']['log']
    os.makedirs(result_path, exist_ok=True)
    seed_torch()
    # wandb.init(dir=os.path.abspath(result_path), project='joint_wavelet', job_type='train', mode="online")

    Logger.setup_logger(None, opt['path']['log'], 'train', level=logging.INFO, screen=True)
    Logger.setup_logger('val', opt['path']['log'], 'val', level=logging.INFO)
    logger = logging.getLogger('base')
    logger.info(opt['info'])
    logger.info(Logger.dict2str(opt))

    for phase, dataset_opt in opt['datasets'].items():
        if phase == 'train_qb' and args.phase != 'val':
            train_qb_set = Data.create_dataset2(dataset_opt, 'train')
            train_qb_loader = Data.create_dataloader(
                train_qb_set, dataset_opt, phase)
            train_qb_generator = get_data_generator(train_qb_loader, enable_tqdm=True, desc="train_qb")
        elif phase == 'train_gf2' and args.phase != 'val':
            train_gf2_set = Data.create_dataset2(dataset_opt, 'train')
            train_gf2_loader = Data.create_dataloader(
                train_gf2_set, dataset_opt, phase)
            train_gf2_generator = get_data_generator(train_gf2_loader, enable_tqdm=True, desc="train_gf2")
        elif phase == 'train_wv3' and args.phase != 'val':
            train_wv3_set = Data.create_dataset2(dataset_opt, 'train')
            train_wv3_loader = Data.create_dataloader(
                train_wv3_set, dataset_opt, phase)
            train_wv3_generator = get_data_generator(train_wv3_loader, enable_tqdm=True, desc="train_wv3")
        elif phase == 'val_QB':
            val_set_qb = Data.create_dataset2(dataset_opt, 'val')
            val_loader_qb = Data.create_dataloader(
                val_set_qb, dataset_opt, phase)
        elif phase == 'val_GF2':
            val_set_gf2 = Data.create_dataset2(dataset_opt, 'val')
            val_loader_gf2 = Data.create_dataloader(
                val_set_gf2, dataset_opt, phase)
        elif phase == 'val_WV3':
            val_set_wv3 = Data.create_dataset2(dataset_opt, 'val')
            val_loader_wv3 = Data.create_dataloader(
                val_set_wv3, dataset_opt, phase)
    logger.info('Initial Dataset Finished')
    diffusion = Model.create_model(opt)
    logger.info('Initial Model Finished')
    current_step = diffusion.begin_step
    diffusion.set_new_noise_schedule(  # 初始化扩散模型需要的，无需学习的参数
        opt['model']['beta_schedule'][opt['phase']], schedule_phase=opt['phase'])

    log_path = '{}'.format(opt['path']['results'])
    if not os.path.exists(log_path):
        os.makedirs(log_path)
    copy_source(__file__, log_path)
    shutil.copytree('GeneralModel',
                    os.path.join(log_path, 'GeneralModel'))
    shutil.copytree('config',
                    os.path.join(log_path, 'config'))


    def val_dataset(dataset, val_loader):
        import time
        sam_val = ssim_val = 0.
        result_path = '{}/{}'.format(opt['path']['results'], dataset)
        os.makedirs(result_path, exist_ok=True)
        t1 = time.time()
        for idx, val_data in enumerate(val_loader):
            diffusion.feed_data(val_data)
            diffusion.test(continous=False, prompt=dataset)  # continous=True,得到中间预测结果共计9张图片，否则只得到最后一张
            visuals = diffusion.get_current_visuals()
            img_scale = 1023.0 if dataset == "GF2" else 2047.0

            result = normlization(visuals['SR'][-1], min_max=(0, 1))  # 这里是获取最后一个，因为continous为Ture是返回一个图片数组
            hr = normlization(visuals['HR'], min_max=(0, 1))

            scio.savemat(osp.join(result_path, f'output_mulExm_{str(idx)}.mat'),
                         {"sr": result * img_scale})  # H*W*C

            ssim_val += SSIM_numpy(result, hr, 1)
            sam_val += SAM_numpy(result, hr)
        t2 = time.time()
        print((t2 - t1) / val_loader_qb.__len__())
        ssim_val = float(ssim_val / val_loader_qb.__len__())
        sam_val = float(sam_val / val_loader_qb.__len__())
        eval_score = {f"ssim_{dataset}": ssim_val, f"sam_{dataset}": sam_val}
        # wandb.log(add_prefix(eval_score, "train"), step=current_step)
        print(current_step, dataset, eval_score)


    if opt['phase'] == 'train':
        if opt['path']['resume_state']:  # resume修改为choise+pan2ms或ms2pan的路径
            logger.info('Resuming training from iter: {}.'.format(current_step))
        total_size = 4 * len(train_qb_loader) + 4 * len(train_gf2_loader) + 8 * len(train_wv3_loader)
        qb_prob = 4 * len(train_qb_loader) / total_size
        gf2_prob = 4 * len(train_gf2_loader) / total_size
        while current_step < opt['train']['max_iter']:
            train_data, prompt = sample_data(qb_prob, gf2_prob)
            current_step += 1
            diffusion.feed_data(train_data)
            diffusion.optimize_parameters(prompt)
            # log
            if current_step % opt['train']['print_freq'] == 0:
                logs = diffusion.get_current_log()
                logger.info(dct2str(logs))
                # wandb.log(add_prefix(logs, "train"), step=current_step)
            if current_step % opt['train']['val_freq'] == 0:
                val_dataset("QB", val_loader_qb)
                val_dataset("GF2", val_loader_gf2)
                val_dataset("WV3", val_loader_wv3)
                diffusion.save_network(current_step)
    else:
        logger.info('Begin Model Evaluation.')
        val_dataset("QB", val_loader_qb)
        # val_dataset("GF2", val_loader_gf2)
        # val_dataset("WV3", val_loader_wv3)