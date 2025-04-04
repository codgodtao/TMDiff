import h5py
from torch.utils.data import Dataset
import torch
import numpy as np
from utils import util
import torchvision.transforms as transforms


def _transform(p):
    return transforms.Compose([
        transforms.RandomHorizontalFlip(p),
        transforms.RandomVerticalFlip(p),
        transforms.RandomRotation(90)
    ])


def invPatch(img_MS):
    # receive (1,c,h,w) and return (4,c,h/2,w/2),将一个大图像划分为4个小图像，降低分辨率为原来的1/4
    b, c, h, w = img_MS.shape
    patch = torch.zeros((4, c, h // 2, w // 2))
    patch[0] = img_MS[:, :, :h // 2, :w // 2]
    patch[1] = img_MS[:, :, :h // 2, w // 2:]
    patch[2] = img_MS[:, :, h // 2:, :w // 2]
    patch[3] = img_MS[:, :, h // 2:, w // 2:]
    return patch


def patch_16(img_MSs):
    # receive (16, c, h, w) and return (c, 4h, 4w) cube
    b, c, h, w = img_MSs.shape
    patch = np.zeros((c, 4 * h, 4 * w))  # Set the shape of the patch tensor
    for i in range(4):
        for j in range(4):
            patch[:, i * h:(i + 1) * h, j * w:(j + 1) * w] = img_MSs[
                i * 4 + j]  # Assign image values to the correct regions in the patch tensor
    return patch


def unpatch_16(patch):
    # 假设patch是一个形状为(c, 4h, 4w)的三维数组
    # 目标是将其分割回16个原始分辨率的小图像，形状为(16, c, h, w)
    c, ph, pw = patch.shape
    h, w = ph // 4, pw // 4  # 计算原始图片的高度和宽度

    # 初始化一个新的数组来存储分割后的图像
    img_MSs = np.zeros((16, c, h, w), dtype=patch.dtype)

    # 遍历patch中的每个小块，将其分割回原始的小图像
    for i in range(4):
        for j in range(4):
            img_MSs[i * 4 + j] = patch[:, i * h:(i + 1) * h, j * w:(j + 1) * w]

    return img_MSs


import cv2


def Upsample(img_MSs, resolution):
    """
    上采样函数，将图像从 [B, C, H, W] 分辨率上采样到 [B, C, resolution, resolution]。
    """
    # 确保输入图像数据是四维的
    if img_MSs.ndim != 4:
        raise ValueError("img_MSs must be a 4-dimensional array with shape [B, C, H, W].")

    # 初始化一个空列表来存储上采样后的图像
    upsampled_images = []

    # 对每个图像进行上采样
    for b in range(img_MSs.shape[0]):
        # 将当前图像从 [C, H, W] 转换为 [H, W, C]，因为 OpenCV 期望的格式是 [H, W, C]
        img = img_MSs[b].transpose((1, 2, 0))

        # 使用 OpenCV 的 resize 函数进行上采样
        upsampled_img = cv2.resize(img, (resolution, resolution), interpolation=cv2.INTER_LINEAR)

        # 将上采样后的图像添加到列表中
        upsampled_images.append(upsampled_img)

    # 将列表转换为 NumPy 数组，并确保通道顺序为 [B, C, resolution, resolution]
    upsampled_images_array = np.stack(upsampled_images, axis=0).transpose((0, 3, 1, 2))

    return upsampled_images_array


class LRHRDataset(Dataset):
    def __init__(self, dataroot, data_len=-1, phase='train'):
        self.data_len = data_len
        self.phase = phase
        data = h5py.File(dataroot)
        if data.get('gt', None) is None:
            gt1 = data['lms'][...]
        else:
            gt1 = data['gt'][...]
        if "gf2" in dataroot:
            img_scale = 1023.0
        else:
            img_scale = 2047.0
        print(img_scale, dataroot)
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
        self.pan = torch.from_numpy(pan1)  #

        self.dataset_len = self.ms.shape[0]  # 目录下所有图像的数量
        if self.data_len <= 0:
            self.data_len = self.dataset_len
        else:
            self.data_len = min(self.data_len, self.dataset_len)
        print(self.pan.shape, self.lms.shape, self.gt.shape, self.ms.shape, self.data_len)
        print(torch.max(self.gt), torch.min(self.gt))

    def __len__(self):
        return self.data_len

    def __getitem__(self, index):
        img_HR = self.gt[index, :, :, :].float()
        img_MS = self.lms[index, :, :, :].float()
        img_PAN = self.pan[index, :, :, :].float()
        img_LR = self.ms[index, :, :, :].float()
        img_Res = util.img2res(img_HR, img_MS)  # 变成[-1,1];
        return {'LR': img_LR, 'PAN': img_PAN, 'MS': img_MS, 'HR': img_HR, 'Res': img_Res}


def read_h5(file_path, new_file_name):
    data = h5py.File(file_path, 'r')
    gt1 = data['gt'][...]
    gt1 = np.array(gt1, dtype=np.float32)
    ms1 = data["ms"][...]  # convert to np tpye for CV2.filter
    ms1 = np.array(ms1, dtype=np.float32)
    lms1 = data["lms"][...]  # convert to np tpye for CV2.filter
    lms1 = np.array(lms1, dtype=np.float32)
    pan1 = data['pan'][...]  # Nx1xHxW
    pan1 = np.array(pan1, dtype=np.float32)

    batch_size = 16
    new_gt = []
    new_ms = []
    new_lms = []
    new_pan = []
    # 遍历整个数组，每次取出16个样本
    for i in range(0, gt1.shape[0], batch_size):
        # 使用切片选择当前批次的数据
        batch_gt1 = gt1[i:i + batch_size]
        batch_ms1 = ms1[i:i + batch_size]
        batch_lms1 = lms1[i:i + batch_size]
        batch_pan1 = pan1[i:i + batch_size]
        # 确保批次大小为16，如果不足16个，则取剩余的所有样本
        if batch_size > gt1.shape[0] - i:
            break
        new_gt.append(patch_16(batch_gt1))
        new_ms.append(patch_16(batch_ms1))
        new_lms.append(patch_16(batch_lms1))
        new_pan.append(patch_16(batch_pan1))
        # return (c, 4h, 4w)

        print(f"Batch {i // batch_size + 1} has shape: {batch_gt1.shape}")
    new_gt = np.stack(new_gt, axis=0)
    new_ms = np.stack(new_ms, axis=0)
    new_lms = np.stack(new_lms, axis=0)
    new_pan = np.stack(new_pan, axis=0)
    print(new_gt.shape, new_ms.shape, new_lms.shape, new_pan.shape)
    f = h5py.File(new_file_name, 'w')
    f.create_dataset('ms', data=new_ms)
    f.create_dataset('gt', data=new_gt)
    f.create_dataset('pan', data=new_pan)
    f.create_dataset('lms', data=new_lms)


if __name__ == "__main__":
    root = "/data/qlt/pancollection/training_data/train_wv3_data.h5"
    new_file_name = "/data/qlt/pancollection/training_data/train_wv3256_data.h5"
    read_h5(root, new_file_name)
