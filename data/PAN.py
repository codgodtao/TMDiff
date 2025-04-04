import random
import torch
import h5py
import torch.utils.data as data
import torch.nn.functional as F
import torchvision.transforms as T
import cv2
import numpy as np
import pywt

from utils import util


class Identity:
    def __call__(self, *args):
        # args is a tuple
        # return is also a tuple
        return args


def RandomEraseChannel(n_channel=8):
    def _RandomEraseChannel(x):
        if x.shape[0] != n_channel:
            return x
        channel = np.where(np.random.rand(1, n_channel) < 0.5)[0]
        # print('erase channel {}'.format(channel))
        x[channel, :, :] = 0.0
        return x

    return _RandomEraseChannel


class PanDataset(data.Dataset):
    def __init__(
            self,
            data_root,
            aug_prob=0.0,  # 不需要使用
            hp=False,  # don't change
            hp_ksize=(5, 5),  # don't change
            norm_range=True,
            wavelets=False,
            *,
            constrain_channel=False,  # don't change
            data_len=-1
    ):
        """

        :param d: h5py.File or dict
        :param aug_prob: augmentation probability
        :param hp: high pass for ms and pan. x = x - cv2.boxFilter(x)
        :param hp_ksize: cv2.boxFiler kernel size
        :param norm_range: normalize data range to [-1, 1]
        :param full_res: use full resolution data or not
                            for full resolution data, there is no gt
                            for synthetic data, there is gt to compute reference metrics(e.g. PSNR)

        """
        super(PanDataset, self).__init__()
        # FIXME: should pass @path rather than @file which is h5py.File object to avoid can not be pickled error
        d = h5py.File(data_root)
        if "gf2" in data_root:
            division = 1023.0
        else:
            division = 2047.0
        self.wavelets = wavelets

        if constrain_channel:
            print(
                "warning: @constrain_n_channel is only used for test code",
                "do not use it if you do not fully know about this.",
            )
            self.slice_channel = [1, 2, 5]
        else:
            self.slice_channel = slice(None)

        self.gt, self.ms, self.lms, self.pan = self.get_divided(d)

        if wavelets:
            print("processing wavelets...")
            lms_main, (lms_h, lms_v, lms_d) = pywt.wavedec2(
                self.lms, "db1", level=1
            )
            pan_main, (pan_h, pan_v, pan_d) = pywt.wavedec2(
                self.pan, "db1", level=1, axes=[-2, -1]
            )
            print("done.")

        print("datasets shape:")
        print("{:^20}{:^20}{:^20}{:^20}".format("pan", "ms", "lms", "gt"))
        print(
            "{:^20}{:^20}{:^20}{:^20}".format(
                str(self.pan.shape),
                str(self.ms.shape),
                str(self.lms.shape),
                str(self.gt.shape),
            )
        )

        self.size = data_len
        if self.size <= 0:
            self.size = self.ms.shape[0]
        else:
            self.size = min(self.ms.shape[0], self.size)
        print("data_len:", self.size)

        # highpass filter
        self.hp = hp
        self.hp_ksize = hp_ksize
        if hp and hp_ksize is not None:
            self.group_high_pass(hp_ksize)

        # to tensor
        if norm_range:
            print("output data ranging in [-1, 1]")
        else:
            print("output data ranging in [0, 1]")

        def norm_func(x):
            if not norm_range:
                x = x / division  # near [0, 1]
            else:
                x = x - x.min()
                x = x / x.max()
                x = 2 * x - 1  # [-1, 1]
            return torch.tensor(x, dtype=torch.float32)

        self.pan = norm_func(self.pan)
        self.ms = norm_func(self.ms)
        self.lms = norm_func(self.lms)
        if wavelets:
            self.wavelets_dcp = torch.cat(
                [*map(norm_func, [lms_main, pan_h, pan_d, pan_v])], dim=1
            )

        self.gt = norm_func(self.gt)

        # geometrical transformation
        self.aug_prob = aug_prob
        self.random_erase_channel = RandomEraseChannel(self.lms.shape[1])
        self.geo_trans = (
            T.Compose(
                [T.RandomHorizontalFlip(p=aug_prob), T.RandomVerticalFlip(p=aug_prob)]
            )
            if aug_prob != 0.0
            else Identity()
        )
        self.erase_trans = T.RandomChoice(
            [T.Lambda(self.random_erase_channel)], p=[aug_prob]
        )

    def get_divided(self, d):
        if d.get('gt', None) is not None:
            return (
                np.asarray(d["gt"], dtype=float)[:, self.slice_channel],
                np.asarray(d["ms"], dtype=float)[:, self.slice_channel],
                np.asarray(d["lms"], dtype=float)[:, self.slice_channel],
                np.asarray(d["pan"], dtype=float),
            )
        else:
            return (
                np.asarray(d["lms"], dtype=float)[:, self.slice_channel],
                np.asarray(d["ms"], dtype=float)[:, self.slice_channel],
                np.asarray(d["lms"], dtype=float)[:, self.slice_channel],
                np.asarray(d["pan"], dtype=float),
            )

    @staticmethod
    def _get_high_pass(data, k_size):
        for i, img in enumerate(data):
            hp = cv2.boxFilter(img.transpose(1, 2, 0), -1, k_size)
            if hp.ndim == 2:
                hp = hp[..., np.newaxis]
            data[i] = img - hp.transpose(2, 0, 1)
        return data

    def group_high_pass(self, k_size):
        self.ms = self._get_high_pass(self.ms, k_size)
        self.pan = self._get_high_pass(self.pan, k_size)

    def aug_trans(self, *data):
        data_list = []
        seed = torch.random.seed()
        for d in data:
            torch.manual_seed(seed)
            random.seed(seed)
            d = self.geo_trans(d)
            data_list.append(d)

        return data_list

    def __getitem__(self, item):
        tuple_data = {'LR': self.ms[item], 'PAN': self.pan[item], 'MS': self.lms[item], 'HR': self.gt[item],
                      'Res': util.img2res(self.gt[item], self.lms[item]), "wav": self.wavelets_dcp[item]}
        return tuple_data

    def __len__(self):
        return self.size

    def __repr__(self):
        return (
            f"num: {self.size} \n "
            f"augmentation: {self.geo_trans} \n"
            f"get high pass ms and pan: {self.hp} \n "
            f"filter kernel size: {self.hp_ksize}"
        )


if __name__ == "__main__":
    import h5py
    from torch.utils.data import DataLoader
    import matplotlib.pyplot as plt

    ds_valid = PanDataset(
        "/data/qlt/pancollection/training_data/train_wv3_data.h5",
        constrain_channel=False,
        aug_prob=0,
        norm_range=False,
        wavelets=True
    )
    dl_train = DataLoader(
        ds_valid, batch_size=1, shuffle=True, pin_memory=False, num_workers=0
    )
    for data in dl_train:
        print(data)  # 8+3  4+3这样的情况，额外有三个通道上的信息
    # train_qb_generator = get_data_generator(train_qb_loader, enable_tqdm=True, desc="train_qb")

    # fig_name = ["pan", "lms", "hr", "m", "h", "v", "d"]
    # j = 0
    # for pan, lms, hr, wavelets in dl_train:
    #     m = wavelets[:, :8] 4
    #     h, v, d = wavelets[:, 8:].chunk(3, dim=1) 32*32 hr-ms的1/2? xt pan-ms wavelets
    #
    #     fig, axes = plt.subplots(ncols=2, nrows=4, figsize=(2 * 3, 4 * 3))
    #     axes = axes.flatten()
    #     for i, x in enumerate([pan, lms, hr, m, h, v, d]):
    #         ax = axes[i]
    #         print(x.shape)
    #         x.clip_(0, 1)
    #         if x.shape[1] > 3:
    #             ax.imshow(x[0, [4, 2, 0]].numpy().transpose(1, 2, 0))
    #         else:
    #             ax.imshow(x[0].numpy().transpose(1, 2, 0))
    #         ax.set_axis_off()
    #         ax.set_title(fig_name[i])
    #     plt.subplots_adjust(hspace=0.1, wspace=0.1)
    #     fig.savefig(f"./{j}.jpg", dpi=200)
    #     j += 1
