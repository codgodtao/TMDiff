import h5py
from torch.utils.data import Dataset
import torch
import numpy as np
from utils import util


class LRHRDataset(Dataset):
    def __init__(self, dataroot, data_len=-1):
        self.data_len = data_len

        data = h5py.File(dataroot)
        if data.get('gt', None) is None:
            gt1 = data['lms'][...]
        else:
            gt1 = data['gt'][...]

        img_scale = 2047.0
        gt1 = np.array(gt1, dtype=np.float32) / img_scale
        self.gt = torch.from_numpy(gt1)

        ms1 = data["ms"][...]  # convert to np tpye for CV2.filter
        ms1 = np.array(ms1, dtype=np.float32) / img_scale
        self.ms = torch.from_numpy(ms1)

        lms1 = data["lms"][...]  # convert to np tpye for CV2.filter
        lms1 = np.array(lms1, dtype=np.float32) / img_scale
        self.lms = torch.from_numpy(lms1)

        pan1 = data['pan'][...]  # Nx1xHxW
        pan1 = np.array(pan1, dtype=np.float32) / img_scale  # Nx1xHxW
        self.pan = torch.from_numpy(pan1)  # Nx1xHxW:
        # logger.info(pan1.shape, lms1.shape, gt1.shape, ms1.shape)
        self.dataset_len = self.ms.shape[0]  # 目录下所有图像的数量
        if self.data_len <= 0:
            self.data_len = self.dataset_len
        else:
            self.data_len = min(self.data_len, self.dataset_len)
        print(self.pan.shape, self.lms.shape, self.gt.shape, self.ms.shape)

    def __len__(self):
        return self.data_len

    def __getitem__(self, index):
        img_HR = self.gt[index, :, :, :].float()
        img_MS = self.lms[index, :, :, :].float()
        img_PAN = self.pan[index, :, :, :].float()
        img_LR = self.ms[index, :, :, :].float()

        # img_HR = Util.img2res(img_HR, img_MS)  # 残差信息作为输入输出，最终预测结果也是残差信息[-1,1]，需要加上[0,1]的lms
        [img_HR_norm] = util.transform_augment(
            [img_HR], min_max=(-1, 1))  # [-1,1]
        return {'LR': img_LR, 'PAN': img_PAN, 'MS': img_MS, 'HR': img_HR_norm,
                'Index': index}
