import copy
import torch
import data as Data
import model as Model
import argparse
import logging
import core.logger as Logger
from tensorboardX import SummaryWriter
import os.path as osp
import os
import numpy as np
import scipy.io as scio
from core.metrics import SSIM_numpy, SAM_numpy

eps = torch.finfo(torch.float32).eps


def normlization(data, min_max=(-1, 1)):
    size = data.shape[-1]
    data = data.squeeze().float().cpu().clamp_(*min_max)
    data = (data - min_max[0]) / \
           (min_max[1] - min_max[0])
    return data.reshape((4, size, size))


def normlization_Z(data, min_max=(-1, 1)):
    size = data.shape[-1]
    data = data.clamp_(*min_max)
    data = (data - min_max[0]) / \
           (min_max[1] - min_max[0])
    return data.reshape((1, 4, size, size))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--config', type=str, default='config/M_cycle.json',
                        help='JSON file for configuration')
    parser.add_argument('-p', '--phase', type=str, choices=['train', 'val'],
                        help='Run either train(training) or val(generation)', default='train')
    parser.add_argument('-gpu', '--gpu_ids', type=str, default=None)
    parser.add_argument('-debug', '-d', action='store_true')
    parser.add_argument('-log_eval', action='store_true')

    # parse configs
    args = parser.parse_args()
    opt = Logger.parse(args)
    opt = Logger.dict_to_nonedict(opt)
    # logging
    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = True

    Logger.setup_logger(None, opt['path']['log'],
                        'train', level=logging.INFO, screen=True)
    Logger.setup_logger('val', opt['path']['log'], 'val', level=logging.INFO)
    logger = logging.getLogger('base')
    logger.info(opt['info'])
    logger.info(Logger.dict2str(opt))
    tb_logger = SummaryWriter(log_dir=opt['path']['tb_logger'])

    # dataset
    for phase, dataset_opt in opt['datasets'].items():
        if phase == 'reduced' and args.phase != 'val':
            reduced_set = Data.create_dataset(dataset_opt, phase)
            reduced_train_loader = Data.create_dataloader(
                reduced_set, dataset_opt, phase)
        elif phase == 'full':
            full_set = Data.create_dataset(dataset_opt, phase)
            full_loader = Data.create_dataloader(
                full_set, dataset_opt, phase)
    logger.info('Initial Dataset Finished')

    # model
    diffusion = Model.create_model(opt)
    logger.info('Initial Model Finished')

    diffusion.set_new_noise_schedule(  # 初始化扩散模型需要的，无需学习的参数
        opt['model']['beta_schedule'][opt['phase']], schedule_phase=opt['phase'])

    n_epochs = opt['train']['n_epochs']
    n_fine_tune_epochs = opt['train']['fine_tune']
    n_cycle = opt['train']['cycle']
    best_psnr_val = 0.
    # zero-shot方法，先循环数据，在循环cycle轮次
    for current_cycle in range(n_cycle):
        result_path = '{}/{}'.format(opt['path']
                                     ['results'], current_cycle)
        os.makedirs(result_path, exist_ok=True)
        # 第一步，基于观测图像采样得到Z
        idx = current_epoch = fine_tune_epoch = 0
        ssim_val = sam_val = 0.
        for _, val_data in enumerate(reduced_train_loader):
            idx += 1
            diffusion.feed_data(val_data)
            diffusion.test(continous=False, w=1.0)
            Z = normlization_Z(diffusion.SR.detach())  # 转换为[0-1]分布的采样图像,E step

            while current_epoch < n_epochs:
                current_epoch += 1
                diffusion.optimize_observation_models(Z)  # M step1
                logger.info("optimize_observation_models")
                # print some logs
                logs = diffusion.get_current_log()
                message = '<epoch:{:3d}, iter:{:8,d}> '.format(
                    current_cycle, current_epoch)
                for k, v in logs.items():
                    message += '{:s}: {:.4e} '.format(k, v)
                    tb_logger.add_scalar(k, v, current_epoch)
                    logger.info(message)

            while fine_tune_epoch < n_fine_tune_epochs:
                fine_tune_epoch += 1
                diffusion.fine_tune_paramter()  # M step2
                logger.info("fine_tune_paramter")
                logs = diffusion.get_current_log()
                message = '<epoch:{:3d}, iter:{:8,d}> '.format(
                    fine_tune_epoch, current_cycle)
                for k, v in logs.items():
                    message += '{:s}: {:.4e} '.format(k, v)
                    tb_logger.add_scalar(k, v, fine_tune_epoch)
                logger.info(message)

            diffusion.test(continous=False, w=1.0)
            visuals = diffusion.get_current_visuals()

            img_scale = 2047.0
            result = np.transpose(normlization(visuals['SR']).numpy() * img_scale, (1, 2, 0))
            hr = np.transpose(normlization(visuals['HR']).numpy() * img_scale, (1, 2, 0))

            scio.savemat(osp.join(result_path, f'output_mulExm_{str(idx - 1)}.mat'),
                         {"sr": np.transpose(result * img_scale, (1, 2, 0))})  # H*W*C

            ssim_val += SSIM_numpy(result, hr, 2047)
            sam_val += SAM_numpy(result, hr)
        ssim_val = float(ssim_val / reduced_train_loader.__len__())
        sam_val = float(sam_val / reduced_train_loader.__len__())
        print('SAM on validation data', sam_val, current_cycle)
        print('SSIM on validation data', ssim_val, current_cycle)
    #     # save model
    #     if best_psnr_val < psnr_val:
    #         best_psnr_val = psnr_val
    #         logger.info('Saving models and training states.')
    #         diffusion.save_network(current_cycle, current_epoch)
    # logger.info('End of training.')
