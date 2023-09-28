import copy
import torch
import data as Data
import model as Model
import argparse
import logging
import core.logger as Logger
import core.metrics as Metrics
from tensorboardX import SummaryWriter
import os.path as osp
import os
import numpy as np
import scipy.io as scio
from core.metrics import SCC_numpy, SSIM_numpy

eps = torch.finfo(torch.float32).eps


def normlization(data, min_max=(-1, 1)):
    size = data.shape[-1]
    data = data.squeeze().float().cpu().clamp_(*min_max)
    data = (data - min_max[0]) / \
           (min_max[1] - min_max[0])
    return data.reshape((4, size, size))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--config', type=str, default='config/sr_sr3_64_512_band4_linux.json',
                        help='JSON file for configuration')
    parser.add_argument('-p', '--phase', type=str, choices=['train', 'val'],
                        help='Run either train(training) or val(generation)', default='train')
    parser.add_argument('-gpu', '--gpu_ids', type=str, default=None)
    parser.add_argument('-debug', '-d', action='store_true')
    parser.add_argument('-log_eval', action='store_true')

    # parse configs
    args = parser.parse_args()
    opt = Logger.parse(args)
    # Convert to NoneDict, which return None for missing key.
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
        if phase == 'train' and args.phase != 'val':
            train_set = Data.create_dataset(dataset_opt, phase)
            train_loader = Data.create_dataloader(
                train_set, dataset_opt, phase)
        elif phase == 'val':
            val_set = Data.create_dataset(dataset_opt, phase)
            val_loader = Data.create_dataloader(
                val_set, dataset_opt, phase)
    logger.info('Initial Dataset Finished')

    # model
    diffusion = Model.create_model(opt)
    logger.info('Initial Model Finished')

    # Train
    current_step = diffusion.begin_step
    current_epoch = diffusion.begin_epoch
    n_epochs = opt['train']['n_epochs']

    diffusion.set_new_noise_schedule(  # 初始化扩散模型需要的，无需学习的参数
        opt['model']['beta_schedule'][opt['phase']], schedule_phase=opt['phase'])

    if opt['phase'] == 'train':
        if opt['path']['resume_state']:  # resume修改为choise+pan2ms或ms2pan的路径
            logger.info('Resuming training from epoch: {}, iter: {}.'.format(
                current_epoch, current_step))
        best_psnr_val, best_ssim_val = 0., 0.
        while current_epoch < n_epochs:
            current_epoch += 1
            for _, train_data in enumerate(train_loader):
                current_step += opt['datasets']['train']['batch_size']
                if current_epoch > n_epochs:
                    break
                diffusion.feed_data(train_data)
                diffusion.optimize_parameters()
                # log
                if current_epoch % opt['train']['print_freq'] == 0:
                    logs = diffusion.get_current_log()
                    message = '<epoch:{:3d}, iter:{:8,d}> '.format(
                        current_epoch, current_step)
                    for k, v in logs.items():
                        message += '{:s}: {:.4e} '.format(k, v)
                        tb_logger.add_scalar(k, v, current_step)
                    logger.info(message)
            diffusion.scheduler.step()

            # validation
            psnr_val = 0.
            ssim_val = 0.
            if current_epoch % opt['train']['val_freq'] == 0:
                idx = 0
                result_path = '{}/{}'.format(opt['path']
                                             ['results'], current_epoch)
                os.makedirs(result_path, exist_ok=True)

                diffusion.set_new_noise_schedule(
                    opt['model']['beta_schedule']['val'], schedule_phase='val')
                for _, val_data in enumerate(val_loader):
                    idx += 1
                    diffusion.feed_data(val_data)
                    diffusion.test(continous=False, w=1.0)
                    visuals = diffusion.get_current_visuals()

                    img_scale = 2047.0
                    result = normlization(visuals['SR']).numpy()  # 这里是获取最后一个，因为continous为Ture是返回一个图片数组
                    hr = normlization(visuals['HR']).numpy()

                    scio.savemat(osp.join(result_path, f'output_mulExm_{str(idx - 1)}.mat'),
                                 {"sr": np.transpose(result * img_scale, (1, 2, 0))})  # H*W*C

                    psnr_val += SCC_numpy(result, hr)
                    ssim_val += SSIM_numpy(result, hr)

                psnr_val = float(psnr_val / val_loader.__len__())
                ssim_val = float(ssim_val / val_loader.__len__())
                logger.info(str(psnr_val), str(ssim_val))
                tb_logger.add_scalar('PSNR on validation data', psnr_val, current_epoch)
                tb_logger.add_scalar('SSIM on validation data', ssim_val, current_epoch)
                diffusion.set_new_noise_schedule(
                    opt['model']['beta_schedule']['train'], schedule_phase='train')
            # save model
            if current_epoch % opt['train']['save_checkpoint_freq'] == 0:
                if best_psnr_val < psnr_val:
                    best_psnr_val = psnr_val
                    best_ssim_val = ssim_val
                    logger.info('Saving models and training states.')
                    diffusion.save_network(current_epoch, current_step)
        logger.info('End of training.')
    else:
        logger.info('Begin Model Evaluation.')
        idx = 0
        result_path = '{}'.format(opt['path']['results'])
        os.makedirs(result_path, exist_ok=True)
        for _, val_data in enumerate(val_loader):
            idx += 1
            diffusion.feed_data(val_data)
            diffusion.test(continous=True, w=1.0)  # 传入的continues为True,一次只传入一个sample进行串行采样
            visuals = diffusion.get_current_visuals()

            # 保存需要的sr结果
            img_scale = 2047.0
            result = normlization(visuals['SR'][-1])
            scio.savemat(osp.join(result_path, f'output_mulExm_{str(idx - 1)}.mat'),
                         {"sr": np.transpose(result * img_scale, (1, 2, 0)).detach().numpy()})  # H*W*C

            # 中间的其他辅助信息，保存为png格式
            sr_img = Metrics.tensor2img_4C(visuals['SR'])  # uint8
            hr_img = Metrics.tensor2img_4C(visuals['HR'])  # uint8
            lr_img = Metrics.tensor2img_4C(visuals['LR'])  # uint8
            PAN_img = Metrics.tensor2img_4C(visuals['PAN'])  # uint8
            sr_img_mode = 'grid'
            if sr_img_mode == 'single':
                # single img series
                sr_img = visuals['SR']  # uint8
                sample_num = sr_img.shape[0]
                for iter in range(0, sample_num):
                    Metrics.save_img(
                        Metrics.tensor2img_4C(sr_img[iter]),
                        '{}/{}_{}_sr_{}.png'.format(result_path, current_step, idx, iter))
            else:
                sr_img = Metrics.tensor2img_4C(visuals['SR'])  # uint8
                Metrics.save_img(
                    sr_img, '{}/{}_{}_sr_process.png'.format(result_path, current_step, idx))
                Metrics.save_img(
                    Metrics.tensor2img_4C(visuals['SR'][-1]),
                    '{}/{}_{}_sr.png'.format(result_path, current_step, idx))
            Metrics.save_img(
                hr_img, '{}/{}_{}_hr.png'.format(result_path, current_step, idx))
            Metrics.save_img(
                lr_img, '{}/{}_{}_lr.png'.format(result_path, current_step, idx))
            Metrics.save_img(
                PAN_img, '{}/{}_{}_pan.png'.format(result_path, current_step, idx))
