import random

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


def normlization(data, min_max=(-1.0, 1.0)):
    data = data.squeeze().cpu().clamp_(*min_max).numpy()
    data = (data - min_max[0]) / (min_max[1] - min_max[0])
    return np.transpose(data, (1, 2, 0))  # H W C


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--config', type=str, default='config/general_multi.json',
                        help='JSON file for configuration')
    parser.add_argument('-p', '--phase', type=str, choices=['train', 'val'],
                        help='Run either train(training) or val(generation)', default='val')
    parser.add_argument('-gpu', '--gpu_ids', type=str, default=None)
    parser.add_argument('-debug', '-d', action='store_true')
    parser.add_argument('-log_eval', action='store_true')

    # parse configs
    args = parser.parse_args()
    opt = Logger.parse(args)
    # Convert to NoneDict, which return None for missing key. training到后期的时候使用detach即可
    opt = Logger.dict_to_nonedict(opt)
    result_path = opt['path']['log']
    os.makedirs(result_path, exist_ok=True)
    seed_torch()
    wandb.init(dir=os.path.abspath(result_path), project='joint', job_type='train', mode="offline")

    Logger.setup_logger(None, opt['path']['log'],
                        'train', level=logging.INFO, screen=True)
    Logger.setup_logger('val', opt['path']['log'], 'val', level=logging.INFO)
    logger = logging.getLogger('base')
    logger.info(opt['info'])
    logger.info(Logger.dict2str(opt))

    for phase, dataset_opt in opt['datasets'].items():
        if phase == 'train_qb' and args.phase != 'val':
            train_qb_set = Data.create_dataset(dataset_opt, 'train')
            train_qb_loader = Data.create_dataloader(
                train_qb_set, dataset_opt, phase)
            train_qb_generator = get_data_generator(train_qb_loader, enable_tqdm=True, desc="train_qb")
        elif phase == 'train_gf2' and args.phase != 'val':
            train_gf2_set = Data.create_dataset(dataset_opt, 'train')
            train_gf2_loader = Data.create_dataloader(
                train_gf2_set, dataset_opt, phase)
            train_gf2_generator = get_data_generator(train_gf2_loader, enable_tqdm=True, desc="train_gf2")
        elif phase == 'train_wv3' and args.phase != 'val':
            train_wv3_set = Data.create_dataset(dataset_opt, 'train')
            train_wv3_loader = Data.create_dataloader(
                train_wv3_set, dataset_opt, phase)
            train_wv3_generator = get_data_generator(train_wv3_loader, enable_tqdm=True, desc="train_wv3")
        elif phase == 'train_wv2' and args.phase != 'val':
            train_wv2_set = Data.create_dataset(dataset_opt, 'train')
            train_wv2_loader = Data.create_dataloader(
                train_wv2_set, dataset_opt, phase)
            train_wv2_generator = get_data_generator(train_wv2_loader, enable_tqdm=True, desc="train_wv2")
        elif phase == 'val_QB':
            val_set_qb = Data.create_dataset(dataset_opt, 'val')
            val_loader_qb = Data.create_dataloader(
                val_set_qb, dataset_opt, phase)
        elif phase == 'val_GF2':
            val_set_gf2 = Data.create_dataset(dataset_opt, 'val')
            val_loader_gf2 = Data.create_dataloader(
                val_set_gf2, dataset_opt, phase)
        elif phase == 'val_WV3':
            val_set_wv3 = Data.create_dataset(dataset_opt, 'val')
            val_loader_wv3 = Data.create_dataloader(
                val_set_wv3, dataset_opt, phase)
    logger.info('Initial Dataset Finished')

    # model
    diffusion = Model.create_model(opt)
    logger.info('Initial Model Finished')

    # Train
    current_step = diffusion.begin_step
    diffusion.set_new_noise_schedule(  # 初始化扩散模型需要的，无需学习的参数
        opt['model']['beta_schedule'][opt['phase']], schedule_phase=opt['phase'])
    if opt['phase'] == 'train':
        if opt['path']['resume_state']:  # resume修改为choise+pan2ms或ms2pan的路径
            logger.info('Resuming training from iter: {}.'.format(current_step))
        best_ssim_val = 0.0
        while True:
            # batch交替训练的方案
            if current_step % 3 == 0:
                train_data = next(train_qb_generator)
                prompt = "QB"
            elif current_step % 3 == 1:
                train_data = next(train_gf2_generator)
                prompt = "GF2"
            else:
                train_data = next(train_wv3_generator)
                prompt = "WV3"
            current_step += 1
            diffusion.feed_data(train_data)
            diffusion.optimize_parameters(prompt)
            diffusion.scheduler.step()
            # log
            if current_step % opt['train']['print_freq'] == 0:
                logs = diffusion.get_current_log()
                logger.info(dct2str(logs))
                wandb.log(add_prefix(logs, "train"), step=current_step)

            if current_step % opt['train']['val_freq'] == 0:
                sam_val = ssim_val = 0.
                idx = 0
                result_path = '{}/{}/{}'.format(opt['path']
                                                ['results'], "GF2", current_step)
                os.makedirs(result_path, exist_ok=True)

                for _, val_data in enumerate(val_loader_gf2):
                    idx += 1
                    diffusion.feed_data(val_data)
                    diffusion.test(continous=False, prompt="GF2",
                                   guidance=3.0)  # continous=True,得到中间预测结果共计9张图片，否则只得到最后一张
                    visuals = diffusion.get_current_visuals()

                    img_scale = 2047.0
                    result = normlization(visuals['SR'], min_max=(0, 1))  # 这里是获取最后一个，因为continous为Ture是返回一个图片数组
                    hr = normlization(visuals['HR'], min_max=(0, 1))

                    scio.savemat(osp.join(result_path, f'output_mulExm_{str(idx - 1)}.mat'),
                                 {"sr": result * img_scale})  # H*W*C

                    ssim_val += SSIM_numpy(result, hr, 1)
                    sam_val += SAM_numpy(result, hr)

                ssim_val = float(ssim_val / val_loader_gf2.__len__())
                sam_val = float(sam_val / val_loader_gf2.__len__())
                eval_score = {"ssim_gf2": ssim_val, "sam_gf2": sam_val}
                print(eval_score)
                wandb.log(add_prefix(eval_score, "val"), step=current_step)

            if current_step % opt['train']['val_freq'] == 0:
                sam_val = ssim_val = 0.
                idx = 0
                result_path = '{}/{}/{}'.format(opt['path']
                                                ['results'], "WV3", current_step)
                os.makedirs(result_path, exist_ok=True)

                for _, val_data in enumerate(val_loader_wv3):
                    idx += 1
                    diffusion.feed_data(val_data)
                    diffusion.test(continous=False, prompt="WV3",
                                   guidance=3.0)  # continous=True,得到中间预测结果共计9张图片，否则只得到最后一张
                    visuals = diffusion.get_current_visuals()

                    img_scale = 2047.0
                    result = normlization(visuals['SR'], min_max=(0, 1))  # 这里是获取最后一个，因为continous为Ture是返回一个图片数组
                    hr = normlization(visuals['HR'], min_max=(0, 1))

                    scio.savemat(osp.join(result_path, f'output_mulExm_{str(idx - 1)}.mat'),
                                 {"sr": result * img_scale})  # H*W*C

                    ssim_val += SSIM_numpy(result, hr, 1)
                    sam_val += SAM_numpy(result, hr)

                ssim_val = float(ssim_val / val_loader_wv3.__len__())
                sam_val = float(sam_val / val_loader_wv3.__len__())
                eval_score = {"ssim_wv3": ssim_val, "sam_wv3": sam_val}
                print(eval_score)
                wandb.log(add_prefix(eval_score, "val"), step=current_step)

            if current_step % opt['train']['val_freq'] == 0:
                sam_val = ssim_val = 0.
                result_path = '{}/{}/{}'.format(opt['path']
                                                ['results'], "QB", current_step)
                os.makedirs(result_path, exist_ok=True)
                idx = 0
                for _, val_data in enumerate(val_loader_qb):
                    idx += 1
                    diffusion.feed_data(val_data)
                    diffusion.test(continous=False, prompt="QB",
                                   guidance=3.0)  # continous=True,得到中间预测结果共计9张图片，否则只得到最后一张
                    visuals = diffusion.get_current_visuals()

                    img_scale = 2047.0
                    result = normlization(visuals['SR'], min_max=(0, 1))  # 这里是获取最后一个，因为continous为Ture是返回一个图片数组
                    hr = normlization(visuals['HR'], min_max=(0, 1))

                    scio.savemat(osp.join(result_path, f'output_mulExm_{str(idx - 1)}.mat'),
                                 {"sr": result * img_scale})  # H*W*C

                    ssim_val += SSIM_numpy(result, hr, 1)
                    sam_val += SAM_numpy(result, hr)

                ssim_val = float(ssim_val / val_loader_qb.__len__())
                sam_val = float(sam_val / val_loader_qb.__len__())
                eval_score = {"ssim": ssim_val, "sam": sam_val}
                print(eval_score)
                wandb.log(add_prefix(eval_score, "val"), step=current_step)

            if current_step % opt['train']['save_checkpoint_freq'] == 0:
                if best_ssim_val < ssim_val:
                    best_ssim_val = ssim_val
                    diffusion.save_network(current_step)
    else:
        logger.info('Begin Model Evaluation.')
        for guidance in range(2, 10, 1):
            idx = 0
            sam_val = ssim_val = 0.
            result_path = '{}/{}/{}'.format(opt['path']
                                            ['results'], "QB", guidance)
            os.makedirs(result_path, exist_ok=True)
            for _, val_data in enumerate(val_loader_qb):
                idx += 1
                diffusion.feed_data(val_data)
                diffusion.test(continous=False, prompt="QB",
                               guidance=guidance)  # continous=True,得到中间预测结果共计9张图片，否则只得到最后一张
                visuals = diffusion.get_current_visuals()

                img_scale = 2047.0

                result = normlization(visuals['SR'], min_max=(0, 1))  # 这里是获取最后一个，因为continous为Ture是返回一个图片数组
                hr = normlization(visuals['HR'], min_max=(0, 1))

                scio.savemat(osp.join(result_path, f'output_mulExm_{str(idx - 1)}.mat'),
                             {"sr": result * img_scale})  # H*W*C

                ssim_val += SSIM_numpy(result, hr, 1)
                sam_val += SAM_numpy(result, hr)
            ssim_val = float(ssim_val / val_loader_qb.__len__())
            sam_val = float(sam_val / val_loader_qb.__len__())
            eval_score = {"ssim": ssim_val, "sam": sam_val}
            print(guidance, "QB", eval_score)
            sam_val = ssim_val = 0.
            idx = 0
            result_path = '{}/{}/{}'.format(opt['path']
                                            ['results'], "GF2", guidance)
            os.makedirs(result_path, exist_ok=True)
            for _, val_data in enumerate(val_loader_gf2):
                idx += 1
                diffusion.feed_data(val_data)
                diffusion.test(continous=False, prompt="GF2",
                               guidance=guidance)  # continous=True,得到中间预测结果共计9张图片，否则只得到最后一张
                visuals = diffusion.get_current_visuals()

                img_scale = 2047.0
                result = normlization(visuals['SR'], min_max=(0, 1))  # 这里是获取最后一个，因为continous为Ture是返回一个图片数组
                hr = normlization(visuals['HR'], min_max=(0, 1))

                scio.savemat(osp.join(result_path, f'output_mulExm_{str(idx - 1)}.mat'),
                             {"sr": result * img_scale})  # H*W*C
                ssim_val += SSIM_numpy(result, hr, 1)
                sam_val += SAM_numpy(result, hr)
            ssim_val = float(ssim_val / val_loader_qb.__len__())
            sam_val = float(sam_val / val_loader_qb.__len__())
            eval_score = {"ssim": ssim_val, "sam": sam_val}
            print(guidance, "GF2", eval_score)
            sam_val = ssim_val = 0.
            idx = 0
            result_path = '{}/{}/{}'.format(opt['path']
                                            ['results'], "WV3", guidance)
            os.makedirs(result_path, exist_ok=True)
            for _, val_data in enumerate(val_loader_wv3):
                idx += 1
                diffusion.feed_data(val_data)
                diffusion.test(continous=False, prompt="WV3",
                               guidance=guidance)  # continous=True,得到中间预测结果共计9张图片，否则只得到最后一张
                visuals = diffusion.get_current_visuals()

                img_scale = 2047.0
                result = normlization(visuals['SR'], min_max=(0, 1))  # 这里是获取最后一个，因为continous为Ture是返回一个图片数组
                hr = normlization(visuals['HR'], min_max=(0, 1))

                scio.savemat(osp.join(result_path, f'output_mulExm_{str(idx - 1)}.mat'),
                             {"sr": result * img_scale})  # H*W*C
                ssim_val += SSIM_numpy(result, hr, 1)
                sam_val += SAM_numpy(result, hr)
            ssim_val = float(ssim_val / val_loader_qb.__len__())
            sam_val = float(sam_val / val_loader_qb.__len__())
            eval_score = {"ssim": ssim_val, "sam": sam_val}
            print(guidance, "WV3", eval_score)
