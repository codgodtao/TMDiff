import numpy as np
import matplotlib.pyplot as plt
import os
import cv2

# from cv2 import imread, resize
from torch.nn import functional as F
# from tqdm import tqdm
# from collections import Counter
import torch
import torch.nn as nn
from torch.autograd import Variable
import math

from utils.util import get_gaussian_kernel


def add_prefix(dct, prefix):
    return {f'{prefix}/{key}': val for key, val in dct.items()}


def dct2str(dct):
    return str({k: f'{v:.6g}' for k, v in dct.items()})


'''
mav_value = 1023  #  GF:1023  QB:2047
img = img.astype(np.float32) / mav_value    # 归一化处理  最开始 mav_value = 1023  #  GF:1023  QB:2047
x = (x * mav_value).astype(np.uint16)
'''


# args = args_parser()


# def denorm(x):
#     x = (x * mav_value).astype(np.uint16)
#     return x


# def eval_img_save(x, name, k):
#     x = x.numpy()
#     x = np.transpose(x, (0, 2, 3, 1))  # [batch_size,512,512,4]
#     if name == 'real_images':
#         array2raster(join(evalsample_dir, 'real_images_{}_epoch{}.tif'.format(k + 1, total_epochs)),
#                      [0, 0], 8, 8, denorm(x[0].transpose(2, 0, 1)), 4)
#     else:
#         array2raster(join(evalsample_dir,
#                           '{}_v{}_eval_fused_images_{}_epoch{}.tif'.format(method, version, k + 1, total_epochs)),
#                      [0, 0], 8, 8, denorm(x[0].transpose(2, 0, 1)), 4)
#
#
# def test_img_save(x, name, epoch):
#     x = np.transpose(x, (0, 2, 3, 1))
#     x = x.numpy()  # [batch_size,512,512,4]
#     if name == 'test_fused_images':
#         array2raster(join(testsample_dir, 'test_fused_images_9_epoch{}.tif'.format(epoch)),
#                      [0, 0], 8, 8, denorm(x[0].transpose(2, 0, 1)), 4)
#     elif name == 'real_images':
#         array2raster(join(testsample_dir, 'real_images_9_epoch{}.tif'.format(epoch)),
#                      [0, 0], 8, 8, denorm(x[0].transpose(2, 0, 1)), 4)
#     elif name == 'test_pan_images':
#         array2raster(join(testsample_dir, 'test_pan_images_9_epoch{}.tif'.format(epoch)),
#                      [0, 0], 8, 8, denorm(x[0].reshape(x.shape[1], x.shape[2])), 1)
#     else:
#         array2raster(join(testsample_dir, 'test_lrms_images_9_epoch{}.tif'.format(epoch)),
#                      [0, 0], 8, 8, denorm(x[0].transpose(2, 0, 1)), 4)
#
#
# def array2raster(newRasterfn, rasterOrigin, pixelWidth, pixelHeight, array, bandSize):
#     if (bandSize == 4):
#         cols = array.shape[2]
#         rows = array.shape[1]
#         originX = rasterOrigin[0]
#         originY = rasterOrigin[1]
#
#         driver = gdal.GetDriverByName('GTiff')  # #存的数据格式
#
#         outRaster = driver.Create(newRasterfn, cols, rows, 4, gdal.GDT_UInt16)
#         outRaster.SetGeoTransform((originX, pixelWidth, 0, originY, 0, pixelHeight))
#         for i in range(1, 5):
#             outband = outRaster.GetRasterBand(i)
#             outband.WriteArray(array[i - 1, :, :])
#         outRasterSRS = osr.SpatialReference()
#         outRasterSRS.ImportFromEPSG(4326)
#         outRaster.SetProjection(outRasterSRS.ExportToWkt())
#         outband.FlushCache()
#     elif (bandSize == 1):
#         cols = array.shape[1]
#         rows = array.shape[0]
#         originX = rasterOrigin[0]
#         originY = rasterOrigin[1]
#
#         driver = gdal.GetDriverByName('GTiff')
#
#         outRaster = driver.Create(newRasterfn, cols, rows, 1, gdal.GDT_UInt16)
#         outRaster.SetGeoTransform((originX, pixelWidth, 0, originY, 0, pixelHeight))
#
#         outband = outRaster.GetRasterBand(1)
#         outband.WriteArray(array)


def normalized(X):
    maxX = np.max(X)
    # print('maxX:', maxX)
    if maxX == 0:
        return X
    else:
        minX = np.min(X)
        X = (X - minX) / (maxX - minX)
    return X


def setRange(X, maxX=1, minX=0):
    X = (X - minX) / (maxX - minX)
    return X


def get3band_of_tensor(outX, nbanch=0, nframe=[0, 1, 2]):
    X = outX[:, :, :, nframe]
    X = X[nbanch, :, :, :]
    return X


def imshow(X):
    plt.close()
    X = np.maximum(X, 0)
    X = np.minimum(X, 1)
    plt.imshow(X[:, :, ::-1])
    plt.axis('off')
    plt.show()


# def imwrite(X, saveload='./tempIm/1.png'):  #################### zy
#     plt.imsave(saveload, ML.normalized(X[:, :, ::-1]))


def mkdir(path):
    folder = os.path.exists(path)

    if not folder:
        os.makedirs(path)
        print("---  new folder...  ---")
        print("---  " + path + "  ---")
    else:
        print("---  There exsits folder " + path + " !  ---")


# SSRNT
def to_var(x, volatile=False):
    if torch.cuda.is_available():
        x = x.cuda().float()
    return Variable(x, volatile=volatile)


class AverageMeter(object):
    """
    Keeps track of most recent, average, sum, and count of a metric.
    """

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


# SSRNET
def adjust_learning_rate(optimizer, shrink_factor):
    """
    Shrinks learning rate by a specified factor.

    :param optimizer: optimizer whose learning rate must be shrunk.
    :param shrink_factor: factor in interval (0, 1) to multiply learning rate with.
    """

    print("\nDECAYING learning rate.")
    for param_group in optimizer.param_groups:
        param_group['lr'] = param_group['lr'] * shrink_factor
    print("The new learning rate is %f\n" % (optimizer.param_groups[0]['lr'],))


def get_lp(data):
    r""" get the low-frequency of input images,
    calculate the avg_filter of the input as low-frequency

    Args:
        data (torch.Tensor): image matrix, shape of [N, C, H, W]
    Returns:
        torch.Tensor: low-frequency part of input, shape of [N, C, H, W]
    """
    rs = F.avg_pool2d(data, kernel_size=5, stride=1, padding=2)
    return rs


def channel_pooling(imgs, mode='avg'):
    r""" average or maximum pooling at channel-dim

    Args:
        imgs (torch.Tensor): input images, shape of [N, C, H, W]
        mode (str): 'avg' or 'max', Default: 'avg'
    Returns:
        torch.Tensor: images after pooling, shape of [N, 1, H, W]
    """
    if mode == 'avg':
        return torch.mean(imgs, dim=1, keepdim=True)
    elif mode == 'max':
        return torch.max(imgs, dim=1, keepdim=True)[0]
    else:
        raise SystemExit(f'no such pooling mode \"{mode}\"')


def get_hp_gussain(data, spectral_num):
    x_hat_guass = get_gaussian_kernel(5, 2, spectral_num).cuda()
    rs = x_hat_guass(data)
    rs = data - rs
    return rs


def get_hp(data):
    r""" get the high-frequency of input images,
    first calculate the avg_filter of the input as low-frequency,
    subtract the low-frequency to get the high-frequency

    Args:
        data (torch.Tensor): image matrix, shape of [N, C, H, W]
    Returns:
        torch.Tensor: high-frequency part of input, shape of [N, C, H, W]
    """
    rs = F.avg_pool2d(data, kernel_size=5, stride=1, padding=2)
    rs = data - rs
    return rs


# wang shang
def adjust_learning_rate2(optimizer, epoch, schedule, gamma):
    """Sets the learning rate to the initial LR decayed by schedule"""
    if epoch in schedule:
        for param_group in optimizer.param_groups:
            param_group['lr'] *= gamma
    return optimizer.state_dict()['param_groups'][0]['lr']


def sobel_gradient(input):
    filter_x = torch.from_numpy(
        np.array([[-1., 0., 1.], [-2., 0., 2.], [-1., 0., 1.]], dtype='float32').reshape([1, 1, 3, 3]))
    filter_y = torch.from_numpy(
        np.array([[-1., -2., -1.], [0., 0., 0.], [1., 2., 1.]], dtype='float32').reshape([1, 1, 3, 3]))
    conv_x = nn.Conv2d(1, 1, 3, 1, 1, bias=False)
    conv_x.weight.data = filter_x.cuda()
    conv_y = nn.Conv2d(1, 1, 3, 1, 1, bias=False)
    conv_y.weight.data = filter_y.cuda()
    n, c, h, w = input.shape
    g_x = torch.from_numpy(np.zeros((n, c, h, w))).type(torch.FloatTensor)
    g_y = torch.from_numpy(np.zeros((n, c, h, w))).type(torch.FloatTensor)
    for i in range(c):
        d1_x = conv_x(input[:, i, :, :].unsqueeze(1))
        d1_y = conv_y(input[:, i, :, :].unsqueeze(1))
        # print('d1_x:{}______g_x[:, i, :, :]:{}'.format(d1_x.squeeze().shape, g_x[:, i, :, :].shape))
        g_x[:, i, :, :] = d1_x.squeeze()
        g_y[:, i, :, :] = d1_y.squeeze()

    return g_x, g_y


def lpls_gradient(input):
    filter = torch.from_numpy(
        np.array([[1., 1., 1.], [1., -8., 1.], [1., 1., 1.]], dtype='float32').reshape([1, 1, 3, 3]))
    conv = nn.Conv2d(1, 1, 3, 1, 1, bias=False)
    conv.weight.data = filter
    n, c, h, w = input.shape
    g = torch.from_numpy(np.zeros((n, c, h, w))).type(torch.FloatTensor)
    for i in range(c):
        d = conv(input[:, i, :, :].unsqueeze(1))
        g[:, i, :, :] = d

    return g


'''input b c h w    output b c h w    function gaussian blur tesor'''


def Gaussian_Blur(input):
    img_np = input.detach().cpu().numpy()
    count = img_np.shape[0]
    output = torch.ones(img_np.shape)
    # print('img_np.shape: ', img_np.shape, 'count: ', count, 'output.shape', output.shape)

    for i in range(count):
        img_hwc = np.squeeze(img_np[i, :, :, :]).transpose(1, 2, 0)
        img_hwc_blur = cv2.GaussianBlur(img_hwc, (5, 5), 2)
        img_chw = img_hwc_blur.transpose(2, 0, 1)
        output[i, :, :, :] = torch.from_numpy(img_chw)
        # print('img_hwc.shape: ', img_hwc.shape, 'img_hwc_blur.shape: ', img_hwc_blur.shape,
        # 'img_chw.shape: ', img_chw.shape)
    output = Variable(output).cuda()
    # print('output.shape: ', output.shape, 'output.type: ', type(output))
    return output


class loss_func1(nn.Module):
    def __init__(self):
        super(loss_func1, self).__init__()

    def forward(self, ref, pan, ms, out):
        # channel mean
        out2pan = torch.mean(out, dim=1).unsqueeze(1)
        # blur and resize
        # out2ms = F.interpolate(out, scale_factor=0.25, mode="bicubic")  # ms upsampled already!
        # print('out2ms:{}__ms:{}'.format(out2ms.shape,ms.shape))
        # loss
        pan_gradient_x, pan_gradient_y = sobel_gradient(pan)
        out2pan_gradient_x, out2pan_gradient_y = sobel_gradient(out2pan)
        loss_ms_out = torch.mean(torch.abs(out - ms))
        loss_ref_out = torch.mean(torch.abs(out - ref))
        loss_pan_out = torch.mean(torch.abs(pan_gradient_x - out2pan_gradient_x)) + torch.mean(
            torch.abs(pan_gradient_y - out2pan_gradient_y))
        # gai weighted loss
        a = 1
        b = 1
        c = 1
        # d = 1d*()
        loss_total = a * loss_ms_out + b * loss_ref_out + c * loss_pan_out
        print('loss_ms_out{}: __loss_ref_out{}: __loss_pan_out{}: __loss_total{}:'.format(loss_ms_out, loss_ref_out,
                                                                                          loss_pan_out, loss_total))
        return loss_total


'''add blur part : out2ms'''


class loss_func2(nn.Module):
    def __init__(self):
        super(loss_func2, self).__init__()

    def forward(self, ref, pan, ms, out):
        # channel mean
        out2pan = torch.mean(out, dim=1).unsqueeze(1)
        # blur
        # out2ms = F.interpolate(out, scale_factor=0.25, mode="bicubic")  # ms upsampled already!
        # print('out2ms:{}__ms:{}'.format(out2ms.shape,ms.shape))
        out2ms = Gaussian_Blur(out)
        # loss
        pan_gradient_x, pan_gradient_y = sobel_gradient(pan)
        out2pan_gradient_x, out2pan_gradient_y = sobel_gradient(out2pan)
        loss_ms_out = torch.mean(torch.abs(out2ms - ms))
        loss_ref_out = torch.mean(torch.abs(out - ref))
        loss_pan_out = torch.mean(torch.abs(pan_gradient_x - out2pan_gradient_x)) + torch.mean(
            torch.abs(pan_gradient_y - out2pan_gradient_y))
        # gai weighted loss
        a = 1
        b = 1
        c = 1
        # d = 1d*()
        loss_total = a * loss_ms_out + b * loss_ref_out + c * loss_pan_out
        print('loss_ms_out{}: __loss_ref_out{}: __loss_pan_out{}: __loss_total{}:'.format(loss_ms_out, loss_ref_out,
                                                                                          loss_pan_out, loss_total))
        return loss_total


'''torch.mean(torch.abs ()) changed as nn.L1Loss()'''


class loss_func3(nn.Module):
    def __init__(self):
        super(loss_func3, self).__init__()

    def forward(self, ref, pan, ms, out):
        # channel mean
        out2pan = torch.mean(out, dim=1).unsqueeze(1)
        # blur
        # out2ms = F.interpolate(out, scale_factor=0.25, mode="bicubic")  # ms upsampled already!
        # print('out2ms:{}__ms:{}'.format(out2ms.shape,ms.shape))
        out2ms = Gaussian_Blur(out)
        # loss
        loss = nn.L1Loss()
        loss2 = nn.CosineEmbeddingLoss(margin=0.2)
        pan_gradient_x, pan_gradient_y = sobel_gradient(pan)
        out2pan_gradient_x, out2pan_gradient_y = sobel_gradient(out2pan)
        print(out2ms.shape, ms.shape)
        # b, _, ww, hh = out2ms.shape
        # tar = torch.ones(b, 1, ww, hh).cuda()
        loss_ms_out = loss(out2ms, ms)  # loss2(out2ms, ms, tar)
        loss_pan_out = loss(out2pan_gradient_x, pan_gradient_x) + loss(out2pan_gradient_y, pan_gradient_y)

        loss_ref_out = loss(out, ref)  # loss2(out, ref, tar) +
        # gai weighted loss
        # a = 2
        # b = 2
        # c = 1
        # d = 1d*()
        # loss_total = a * loss_ms_out + b * loss_ref_out + c * loss_pan_out
        loss_total = loss_ms_out + loss_ref_out + loss_pan_out
        # loss_total = loss_ref_out + loss_pan_out
        print('loss_ms_out{}: __loss_ref_out{}: __loss_pan_out{}: __loss_total{}:'.format(loss_ms_out, loss_ref_out,
                                                                                          loss_pan_out, loss_total))
        # print('loss_ref_out{}: __loss_pan_out{}: __loss_total{}:'.format(loss_ref_out, loss_pan_out, loss_total))
        return loss_total


class loss_func4(nn.Module):
    def __init__(self):
        super(loss_func4, self).__init__()
        self.triplet_margin = 12  # 5

    def forward(self, ref, pan, ms, out, quary, key, value):
        # channel mean
        out2pan = torch.mean(out, dim=1).unsqueeze(1)
        # blur
        # out2ms = F.interpolate(out, scale_factor=0.25, mode="bicubic")  # ms upsampled already!
        # print('out2ms:{}__ms:{}'.format(out2ms.shape,ms.shape))
        out2ms = Gaussian_Blur(out)
        # loss
        loss = nn.L1Loss()
        loss2 = nn.CosineEmbeddingLoss(margin=0.2)
        loss_qkv = self.similarity_based_triple_loss(quary, key, value)
        pan_gradient_x, pan_gradient_y = sobel_gradient(pan)
        out2pan_gradient_x, out2pan_gradient_y = sobel_gradient(out2pan)
        print(out2ms.shape, ms.shape)
        # b, _, ww, hh = out2ms.shape
        # tar = torch.ones(b, 1, ww, hh).cuda()
        loss_ms_out = loss(out2ms, ms)  # loss2(out2ms, ms, tar)
        loss_pan_out = loss(out2pan_gradient_x, pan_gradient_x) + loss(out2pan_gradient_y, pan_gradient_y)

        loss_ref_out = loss(out, ref)  # loss2(out, ref, tar) +
        # gai weighted loss
        # a = 2
        # b = 2
        # c = 1
        # d = 1d*()
        # loss_total = a * loss_ms_out + b * loss_ref_out + c * loss_pan_out
        loss_total = loss_ms_out + loss_ref_out + loss_pan_out + loss_qkv
        # loss_total = loss_ref_out + loss_pan_out
        print('loss_ms_out{}: __loss_ref_out{}: __loss_pan_out{}: __loss_qkv{}: __loss_total{}:'.format(loss_ms_out,
                                                                                                        loss_ref_out,
                                                                                                        loss_pan_out,
                                                                                                        loss_qkv,
                                                                                                        loss_total))
        # print('loss_ref_out{}: __loss_pan_out{}: __loss_total{}:'.format(loss_ref_out, loss_pan_out, loss_total))
        return loss_total

    def similarity_based_triple_loss(self, anchor, positive, negative):
        distance = self.scaled_dot_product(anchor, positive) - self.scaled_dot_product(anchor,
                                                                                       negative) + self.triplet_margin
        loss = torch.mean(torch.max(distance, torch.zeros_like(distance)))
        return loss

        # https://www.quantumdl.com/entry/11%EC%A3%BC%EC%B0%A82-Attention-is-All-You-Need-Transformer

    def scaled_dot_product(self, query, key, mask=None, dropout=None):
        "Compute 'Scaled Dot Product Attention'"
        d_k = query.size(-1)
        scores = torch.matmul(query, key.transpose(-2, -1)) \
                 / math.sqrt(d_k)
        return scores


class loss_func5(nn.Module):
    def __init__(self):
        super(loss_func5, self).__init__()
        self.triplet_margin = 12  # 5

    def forward(self, ref, pan, ms, out, quary, key, value):
        # loss
        loss = nn.L1Loss()
        loss2 = nn.CosineEmbeddingLoss(margin=0.2)
        loss3 = nn.MSELoss()
        loss_qkv = self.similarity_based_triple_loss(quary, key, value)

        loss_ref_out = loss(out, ref)  # loss2(out, ref, tar) +
        # gai weighted loss
        # a = 2
        # b = 2
        # c = 1
        # d = 1d*()
        # loss_total = a * loss_ms_out + b * loss_ref_out + c * loss_pan_out
        loss_total = loss_ref_out + loss_qkv
        # loss_total = loss_ref_out + loss_pan_out
        print('loss_ref_out{}:  __loss_qkv{}: __loss_total{}:'.format(loss_ref_out, loss_qkv, loss_total))

        # print('loss_ref_out{}: __loss_pan_out{}: __loss_total{}:'.format(loss_ref_out, loss_pan_out, loss_total))
        return loss_total

    def similarity_based_triple_loss(self, anchor, positive, negative):
        distance = self.scaled_dot_product(anchor, positive) - self.scaled_dot_product(anchor,
                                                                                       negative) + self.triplet_margin
        loss = torch.mean(torch.max(distance, torch.zeros_like(distance)))
        return loss

        # https://www.quantumdl.com/entry/11%EC%A3%BC%EC%B0%A82-Attention-is-All-You-Need-Transformer

    def scaled_dot_product(self, query, key, mask=None, dropout=None):
        "Compute 'Scaled Dot Product Attention'"
        d_k = query.size(-1)
        scores = torch.matmul(query, key.transpose(-2, -1)) \
                 / math.sqrt(d_k)
        return scores


class loss_func6(nn.Module):
    def __init__(self):
        super(loss_func6, self).__init__()
        self.triplet_margin = 12  # 5

    def forward(self, ref, pan, ms, out, quary, key, value, quary_h1, key_h1, value_h1):  # , quary_h1, key_h1, value_h1

        # loss
        loss = nn.L1Loss()
        loss2 = nn.CosineEmbeddingLoss(margin=0.2)
        loss3 = nn.MSELoss()
        loss_qkv = self.similarity_based_triple_loss(quary, key, value)
        # loss_qkvh2 = self.similarity_based_triple_loss(quary_h2, key_h2, value_h2)
        loss_qkvh1 = self.similarity_based_triple_loss(quary_h1, key_h1, value_h1)
        style_loss = loss(self.gram_matrix(out), self.gram_matrix(ref))

        loss_ref_out = loss(out, ref)  # loss2(out, ref, tar) +
        # gai weighted loss
        # a = 2
        # b = 2
        # c = 1
        # d = 1d*()
        # loss_total = a * loss_ms_out + b * loss_ref_out + c * loss_pan_out
        loss_total = loss_ref_out + loss_qkv + loss_qkvh1 + style_loss  # + loss_qkvh2
        # loss_total = loss_ref_out + loss_pan_out
        print('loss_ref_out:{}  __loss_qkv:{}  __loss_qkvh1:{}  __style_loss:{} __loss_total:{}'
              .format(loss_ref_out, loss_qkv, loss_qkvh1, style_loss, loss_total))

        # with open('train_WV2.txt', 'a') as f:
        #     f.write('loss_ref_out:' + str(loss_ref_out) + '_' + 'loss_qkv:' + str(loss_qkv) + '_'
        #             + 'loss_qkvh1:' + str(loss_qkvh1) + '_' + 'style_loss:' + str(style_loss) + '_' + 'loss_total'
        #             + str(loss_total) + '\n')
        # f.close()

        # print('loss_ref_out{}:  __loss_qkv{}:  __loss_qkvh2{}: __loss_qkvh1{}: __style_loss{}: __loss_total{}:'
        #       .format(loss_ref_out, loss_qkv, loss_qkvh2, loss_qkvh1, style_loss, loss_total))

        # print('loss_ref_out{}: __loss_pan_out{}: __loss_total{}:'.format(loss_ref_out, loss_pan_out, loss_total))
        return loss_total

    def similarity_based_triple_loss(self, anchor, positive, negative):
        distance = self.scaled_dot_product(anchor, positive) - self.scaled_dot_product(anchor,
                                                                                       negative) + self.triplet_margin
        loss = torch.mean(torch.max(distance, torch.zeros_like(distance)))
        return loss

        # https://www.quantumdl.com/entry/11%EC%A3%BC%EC%B0%A82-Attention-is-All-You-Need-Transformer

    def scaled_dot_product(self, query, key, mask=None, dropout=None):
        "Compute 'Scaled Dot Product Attention'"
        d_k = query.size(-1)
        scores = torch.matmul(query, key.transpose(-2, -1)) \
                 / math.sqrt(d_k)
        return scores

    def gram_matrix(self, y):
        (b, ch, h, w) = y.size()
        features = y.view(b, ch, w * h)
        features_t = features.transpose(1, 2)
        gram = features.bmm(features_t) / (ch * h * w)
        return gram


class loss_func7(nn.Module):
    def __init__(self):
        super(loss_func7, self).__init__()
        self.triplet_margin = 12  # 5

    def forward(self, ref, pan, ms, out, quary, key, value, quary_h2, key_h2, value_h2, quary_h1, key_h1, value_h1):  #

        # loss
        loss = nn.L1Loss()
        # loss2 = nn.CosineEmbeddingLoss(margin=0.2)
        # loss3 = nn.MSELoss()
        loss_qkv = self.similarity_based_triple_loss(quary, key, value)
        loss_qkvh2 = self.similarity_based_triple_loss(quary_h2, key_h2, value_h2)
        loss_qkvh1 = self.similarity_based_triple_loss(quary_h1, key_h1, value_h1)
        style_loss = loss(self.gram_matrix(out), self.gram_matrix(ref))

        loss_ref_out = loss(out, ref)  # loss2(out, ref, tar) +
        # gai weighted loss
        # a = 2
        # b = 2
        # c = 1
        # d = 1d*()
        # loss_total = a * loss_ms_out + b * loss_ref_out + c * loss_pan_out
        loss_total = loss_ref_out + loss_qkv + loss_qkvh2 + style_loss + loss_qkvh1
        # loss_total = loss_ref_out + loss_pan_out
        # print('loss_ref_out:{}  __loss_qkv:{}  __loss_qkvh2:{}  __style_loss:{} __loss_total:{}'
        #       .format(loss_ref_out, loss_qkv, loss_qkvh2, style_loss, loss_total))
        print('loss_ref_out{}:  __loss_qkv{}:  __loss_qkvh2{}: __loss_qkvh1{}: __style_loss{}: __loss_total{}:'
              .format(loss_ref_out, loss_qkv, loss_qkvh2, loss_qkvh1, style_loss, loss_total))

        # print('loss_ref_out{}: __loss_pan_out{}: __loss_total{}:'.format(loss_ref_out, loss_pan_out, loss_total))
        return loss_total

    def similarity_based_triple_loss(self, anchor, positive, negative):
        distance = self.scaled_dot_product(anchor, positive) - self.scaled_dot_product(anchor,
                                                                                       negative) + self.triplet_margin
        loss = torch.mean(torch.max(distance, torch.zeros_like(distance)))
        return loss

    def scaled_dot_product(self, query, key, mask=None, dropout=None):
        "Compute 'Scaled Dot Product Attention'"
        d_k = query.size(-1)
        scores = torch.matmul(query, key.transpose(-2, -1)) \
                 / math.sqrt(d_k)
        return scores

    def gram_matrix(self, y):
        (b, ch, h, w) = y.size()
        features = y.view(b, ch, w * h)
        features_t = features.transpose(1, 2)
        gram = features.bmm(features_t) / (ch * h * w)
        return gram


class loss_func8(nn.Module):
    def __init__(self):
        super(loss_func8, self).__init__()

    def forward(self, pan, ms, out):
        # channel mean
        out2pan = torch.mean(out, dim=1).unsqueeze(1)
        # blur
        # out2ms = F.interpolate(out, scale_factor=0.25, mode="bicubic")  # ms upsampled already!
        # print('out2ms:{}__ms:{}'.format(out2ms.shape,ms.shape))
        out2ms = Gaussian_Blur(out)
        # loss
        loss = nn.L1Loss()
        loss2 = nn.CosineEmbeddingLoss(margin=0.2)
        pan_gradient_x, pan_gradient_y = sobel_gradient(pan)
        out2pan_gradient_x, out2pan_gradient_y = sobel_gradient(out2pan)
        print(out2ms.shape, ms.shape)
        # b, _, ww, hh = out2ms.shape
        # tar = torch.ones(b, 1, ww, hh).cuda()
        loss_ms_out = loss(out2ms, ms)  # loss2(out2ms, ms, tar)
        loss_pan_out = loss(out2pan_gradient_x, pan_gradient_x) + loss(out2pan_gradient_y, pan_gradient_y)

        # loss_ref_out = loss(out, ref)  # loss2(out, ref, tar) +

        loss_total = loss_ms_out + loss_pan_out

        print('loss_ms_out{}: __loss_pan_out{}: __loss_total{}:'.format(loss_ms_out, loss_pan_out, loss_total))
        return loss_total


'''display image learn from matlab'''


class loss_func9(nn.Module):
    def __init__(self):
        super(loss_func9, self).__init__()

    def forward(self, pan, ms, out):
        # channel mean
        out2pan = torch.mean(out, dim=1).unsqueeze(1)
        # blur
        # out2ms = F.interpolate(out, scale_factor=0.25, mode="bicubic")  # ms upsampled already!
        # print('out2ms:{}__ms:{}'.format(out2ms.shape,ms.shape))
        out2ms = Gaussian_Blur(out)
        # loss
        loss = nn.L1Loss()
        loss2 = nn.CosineEmbeddingLoss(margin=0.2)
        pan_gradient_x, pan_gradient_y = sobel_gradient(pan)
        out2pan_gradient_x, out2pan_gradient_y = sobel_gradient(out2pan)
        print(out2ms.shape, ms.shape)
        b, _, ww, hh = out2ms.shape
        tar = torch.ones(b, 1, ww, hh).cuda()
        loss_ms_out = loss2(out2ms, ms, tar)  # loss2(out2ms, ms, tar)
        loss_pan_out = loss(out2pan_gradient_x, pan_gradient_x) + loss(out2pan_gradient_y, pan_gradient_y)

        # loss_ref_out = loss(out, ref)  # loss2(out, ref, tar) +

        loss_total = loss_ms_out + loss_pan_out

        print('loss_ms_out{}: __loss_pan_out{}: __loss_total{}:'.format(loss_ms_out, loss_pan_out, loss_total))
        return loss_total


'''display image learn from matlab'''


def display_img(img):
    try:
        count = img.shape[2]
        for i in range(count):
            m = np.min(img[:, :, i])
            img[:, :, i] = img[:, :, i] - m
            img[:, :, i] = img[:, :, i] / np.max(img[:, :, i])
    except IndexError:
        m = np.min(img[:, :])
        img[:, :] = img[:, :] - m
        img[:, :] = img[:, :] / np.max(img[:, :])
    return img


def save_img(img, i, name):
    try:
        count = img.shape[2]
    except IndexError:
        count = 1
    if count == 4:
        print("多光谱")
        out_red = img[:, :, 2][:, :, np.newaxis]
        out_green = img[:, :, 1][:, :, np.newaxis]
        out_blue = img[:, :, 0][:, :, np.newaxis]
        out = np.concatenate((out_blue, out_green, out_red), axis=2)
    elif count == 8:
        print("多光谱")
        out_red = img[:, :, 4][:, :, np.newaxis]
        out_green = img[:, :, 2][:, :, np.newaxis]
        out_blue = img[:, :, 1][:, :, np.newaxis]
        out = np.concatenate((out_blue, out_green, out_red), axis=2)
    else:
        out = img
        print("单通道")
    out = 255 * display_img(out)
    # out = 255 * out
    cv2.imwrite('figs/{}_'.format(i) + name + '.jpg', out)


def save_fea(img, i, name):
    try:
        count = img.shape[2]
        print("特征")
        out = np.zeros((img.shape[0], img.shape[1]))
        for j in range(count):
            out += img[:, :, j]
    except IndexError:
        out = img
    out = 255 * display_img(out)
    # out = 255 * out
    cv2.imwrite('figs/{}_'.format(i) + name + '.jpg', out)


class loss_func10(nn.Module):
    def __init__(self):
        super(loss_func10, self).__init__()

    def forward(self, ref, out):
        loss_total = torch.mean(torch.abs(ref - out) * torch.square(ref - out))
        print('______________________________________loss_total{}:'.format(loss_total))
        return loss_total


# class loss_func11(nn.Module):
#     def __init__(self):
#         super(loss_func11, self).__init__()
#         self.dwt = DWT_2D(wavename='haar')
#
#     def forward(self, ref, out):
#         ref_l, ref_h1, ref_h2, ref_h3 = self.dwt(ref)
#         out_l, out_h1, out_h2, out_h3 = self.dwt(out)
#         ref_h = torch.cat((ref_h1, ref_h2, ref_h3), 1)
#         out_h = torch.cat((out_h1, out_h2, out_h3), 1)
#         loss_h = torch.mean(torch.abs(ref_h - out_h) * torch.square(ref_h - out_h))
#         loss_l = torch.mean(torch.abs(ref_l - out_l) * torch.square(ref_l - out_l))
#         loss_total = loss_h + loss_l
#         print('loss_h{}: __loss_l{}: __loss_total{}:'.format(loss_h, loss_l, loss_total))
#         return loss_total


class loss_func12(nn.Module):
    def __init__(self):
        super(loss_func12, self).__init__()

    def forward(self, out_l, out_h, ms_l, pan_h):
        loss_h = torch.mean(torch.abs(pan_h - out_h) * torch.square(pan_h - out_h))
        loss_l = torch.mean(torch.abs(ms_l - out_l) * torch.square(ms_l - out_l))
        loss_total = loss_h + loss_l
        print('loss_h{}: __loss_l{}: __loss_total{}:'.format(loss_h, loss_l, loss_total))
        return loss_total


class loss_func13(nn.Module):
    def __init__(self):
        super(loss_func13, self).__init__()

    def forward(self, out_l, out_h, ms_l, pan_h):
        loss = nn.L1Loss()
        loss_h = loss(pan_h, out_h)
        loss_l = loss(ms_l, out_l)
        loss_total = loss_h + loss_l
        print('loss_h{}: __loss_l{}: __loss_total{}:'.format(loss_h, loss_l, loss_total))
        return loss_total


# class loss_func14(nn.Module):
#     def __init__(self):
#         super(loss_func14, self).__init__()
#
#     def forward(self, ref, out):
#         loss = FFL(loss_weight=1.0, alpha=1.0)
#
#         loss_total = loss(out, ref)
#         print('________loss_total:{}'.format(loss_total))
#         return loss_total


def off_diagonal(x):
    # return a flattened view of the off-diagonal elements of a square matrix
    n, m = x.shape
    assert n == m
    return x.flatten()[:-1].view(n - 1, n + 1)[:, 1:].flatten()


# batch and batch similar
class FLoss(nn.Module):
    # MLP to change high dimension feature to low dimension feature for CC loss calculation
    def __init__(self, dv, do, lambd=1):
        # dv is the M*N*C number of input feature-(B C M N)
        super(FLoss, self).__init__()

        self.layer1 = nn.Linear(dv, do)
        self.layer2 = nn.Linear(dv, do)
        self.bn = nn.BatchNorm1d(do, affine=False)
        self.lambd = lambd
        # self.layer3 = nn.Linear(dv, dv)

    def forward(self, F1, F2):
        # change the shape from (B C M N) to (B M*N*C) for nn.linear function
        F1 = torch.reshape(F1, (F1.size(0), F1.size(1) * F1.size(2) * F1.size(3)))
        F2 = torch.reshape(F2, (F2.size(0), F2.size(1) * F2.size(2) * F2.size(3)))
        # print("F1.shape:", F1.shape, F2.shape)

        # reduce dimension from (B M*N*C) to (B M*N*C)
        F1_1 = self.bn(self.layer1(F1))
        F2_1 = self.bn(self.layer2(F2))

        # print('F1_1:', F1_1.shape)
        # print('F2_1:', F2_1.shape)

        # empirical cross-correlation matrix, size-(batch_size, size)
        c = self.bn(F1_1).T @ self.bn(F2_1)  # mean-0 variation-1

        on_diag = torch.diagonal(c).add_(-1).pow_(2).sum()
        off_diag = off_diagonal(c).pow_(2).sum()
        loss = on_diag + self.lambd * off_diag

        return loss


# only make low part similar
class loss_func15(nn.Module):
    def __init__(self, dv=8192, do=64, lambd=0.005):
        super(loss_func15, self).__init__()
        # dv = M*N*C = 64*64*32=131072

        self.cc_loss = FLoss(dv, do, lambd)

    def forward(self, ref, out, pc_2, mc_2):
        loss = nn.L1Loss()
        loss_ref = loss(ref, out)
        # print('pc_2.shape:', pc_2.shape, mc_2.shape)# torch.Size([32, 32, 16, 16]) torch.Size([32, 32, 16, 16])
        loss_l = self.cc_loss(pc_2, mc_2)
        loss_total = loss_ref + loss_l
        print('loss_ref{}: __loss_l{}: __loss_total{}:'.format(loss_ref, loss_l, loss_total))
        return loss_total


# only make low part similar
class loss_func16(nn.Module):
    def __init__(self, dv=8192, do=64, lambd=0.005):
        super(loss_func16, self).__init__()
        # dv = M*N*C = 64*64*32=131072

        self.cc_loss = FLoss(dv, do, lambd)

    def forward(self, ref, out, pc_2, mc_2):
        loss = nn.L1Loss()
        loss_ref = loss(ref, out)
        # print('pc_2.shape:', pc_2.shape, mc_2.shape)# torch.Size([32, 32, 16, 16]) torch.Size([32, 32, 16, 16])
        loss_l = self.cc_loss(pc_2, mc_2) / 4096.
        loss_total = 0.7 * loss_ref + 0.3 * loss_l
        print('loss_ref{}: __loss_l{}: __loss_total{}:'.format(loss_ref, loss_l, loss_total))
        return loss_total


# only make high part similar
class loss_func17(nn.Module):
    def __init__(self, dv=8192, do=64, lambd=0.005):
        super(loss_func17, self).__init__()
        # dv = M*N*C = 64*64*32=131072  16*16*32=8192 16*16*32*3

        self.cc_lossh2 = FLoss(3 * dv, do, lambd)
        self.cc_lossh1 = FLoss(3 * 4 * dv, do, lambd)

    def forward(self, ref, out, pgc_2, mgc_2, pgc_1, mgc_1):
        loss = nn.L1Loss()
        loss_ref = loss(ref, out)
        # print('pgc_2.shape:', pgc_2.shape, mgc_2.shape)
        # print('pgc_1.shape:', pgc_1.shape, mgc_1.shape)
        # print('pc_2.shape:', pc_2.shape, mc_2.shape)# torch.Size([32, 32, 16, 16]) torch.Size([32, 32, 16, 16])
        loss_h1 = self.cc_lossh1(pgc_1, mgc_1) / 4096.
        loss_h2 = self.cc_lossh2(pgc_2, mgc_2) / 4096.
        loss_total = loss_ref + loss_h1 + loss_h2
        print('loss_ref{}: __loss_h1{}: __loss_h2{}: __loss_total{}:'.format(loss_ref, loss_h1, loss_h2, loss_total))
        return loss_total


# only make high part similar
class loss_func18(nn.Module):
    def __init__(self, dv=8192, do=64, lambd=0.005):
        super(loss_func18, self).__init__()
        # dv = M*N*C = 64*64*32=131072  16*16*32=8192 16*16*32*3

        self.cc_lossl = FLoss(dv, do, lambd)
        self.cc_lossh2 = FLoss(3 * dv, do, lambd)
        self.cc_lossh1 = FLoss(3 * 4 * dv, do, lambd)

    def forward(self, ref, out, pc_2, mc_2, pgc_2, mgc_2, pgc_1, mgc_1):
        loss = nn.L1Loss()
        loss_ref = loss(ref, out)
        # print('pc_2.shape:', pc_2.shape, mc_2.shape)# torch.Size([32, 32, 16, 16]) torch.Size([32, 32, 16, 16])
        loss_l = self.cc_lossl(pc_2, mc_2) / 4096.
        loss_h1 = self.cc_lossh1(pgc_1, mgc_1) / 4096.
        loss_h2 = self.cc_lossh2(pgc_2, mgc_2) / 4096.
        loss_total = loss_ref + loss_l + loss_h1 + loss_h2
        print(
            'loss_ref{}: __loss_l{}: loss_h1{}: __loss_h2{}: __loss_total{}:'.format(loss_ref, loss_l, loss_h1, loss_h2,
                                                                                     loss_total))
        return loss_total


# data and data similar
class FLoss2(nn.Module):
    # MLP to change high dimension feature to low dimension feature for CC loss calculation
    def __init__(self, dv, dv1, do, lambd=1):
        # dv is the M*N*C number of input feature-(B C M N)
        super(FLoss2, self).__init__()

        self.layer1 = nn.Linear(dv, do)
        self.layer2 = nn.Linear(dv1, do)
        self.bn = nn.BatchNorm1d(do, affine=False)
        self.lambd = lambd
        # self.layer3 = nn.Linear(dv, dv)

    def forward(self, F1, F2):
        # change the shape from (B C M N) to (B M*N*C) for nn.linear function
        F1 = torch.reshape(F1, (F1.size(0), F1.size(1) * F1.size(2) * F1.size(3)))
        F2 = torch.reshape(F2, (F2.size(0), F2.size(1) * F2.size(2) * F2.size(3)))
        # print("F1.shape:", F1.shape, F2.shape)

        # reduce dimension from (B M*N*C) to (B M*N*C)
        F1_1 = self.bn(self.layer1(F1))
        F2_1 = self.bn(self.layer2(F2))

        # print('F1_1:', F1_1.shape)
        # print('F2_1:', F2_1.shape)

        # empirical cross-correlation matrix, size-(batch_size, size)
        c = self.bn(F1_1) @ self.bn(F2_1).T  # mean-0 variation-1

        on_diag = torch.diagonal(c).add_(-1).pow_(2).mean()
        off_diag = off_diagonal(c).pow_(2).mean()
        loss = on_diag + self.lambd * off_diag

        return loss


# class loss_func19(nn.Module):
#     def __init__(self, dv=4096, dv1=4096 * args.bands, do=16, lambd=0.005):
#         super(loss_func19, self).__init__()
#         self.cc_loss_pan = FLoss2(dv, dv1, do, lambd)
#         self.cc_loss_ms = FLoss2(dv1, dv1, do, lambd)
#
#     def forward(self, ref, out, pan, ms):
#         # channel mean
#         loss = nn.L1Loss()
#         loss_ref_out = loss(ref, out)
#         #
#         loss_ms_out = self.cc_loss_pan(pan, out)
#         loss_pan_out = self.cc_loss_ms(ms, out)
#         loss_total = loss_ref_out + 0.001 * loss_pan_out + 0.001 * loss_ms_out
#         # loss_total = loss_ref_out + loss_pan_out
#         print('loss_ref_out:{}_loss_ms_out:{}_loss_pan_out:{}_loss_total:{}'.format(loss_ref_out, loss_ms_out,
#                                                                                     loss_pan_out, loss_total))
#         return loss_total


class loss_func20(nn.Module):
    def __init__(self):
        super(loss_func20, self).__init__()

    def forward(self, ref, out, ms_2, out_2):
        # channel mean
        loss = nn.L1Loss()
        loss_ref = loss(ref, out)
        loss_ms2 = loss(ms_2, out_2)

        loss_total = loss_ref + loss_ms2
        # loss_total = loss_ref_out + loss_pan_out
        print('loss_ref:{}_loss_ms2:{}_loss_total:{}'.format(loss_ref, loss_ms2, loss_total))
        return loss_total


class loss_func21(nn.Module):
    def __init__(self):
        super(loss_func21, self).__init__()

    def forward(self, ref, out, ms_2, out_2, ms_1, out_1):
        # channel mean
        loss = nn.L1Loss()
        loss_ref = loss(ref, out)
        loss_ms2 = loss(ms_2, out_2)
        loss_ms1 = loss(ms_1, out_1)

        loss_total = loss_ref + loss_ms2 + loss_ms1
        # loss_total = loss_ref_out + loss_pan_out
        print('loss_ref:{}_loss_ms2:{}_loss_ms1:{}_loss_total:{}'.format(loss_ref, loss_ms2, loss_ms1, loss_total))
        return loss_total


# arccos(x) make x near to 1,then sam near to 0
class SAMLoss(nn.Module):
    # MLP to change high dimension feature to low dimension feature for CC loss calculation
    def __init__(self):
        # dv is the M*N*C number of input feature-(B C M N)
        super(SAMLoss, self).__init__()

    def forward(self, t1, t2):
        t1 = torch.reshape(t1, (t1.size(1) * t1.size(2), -1))
        t2 = torch.reshape(t2, (t2.size(1) * t2.size(2), -1))
        t11 = (torch.sum(t1 ** 2, dim=0)).sqrt()
        t22 = (torch.sum(t2 ** 2, dim=0)).sqrt()
        # print(e * f)
        t12 = torch.sum(t1 * t2, dim=0)
        result = t12 / (t11 * t22 + 0.0000000001)
        print(result.shape)
        loss = result.add_(-1).pow_(2).mean()

        return loss


def dwt(y):
    """
    DWT (Discrete Wavelet Transform) function implementation according to
    "Multi-level Wavelet Convolutional Neural Networks"
    by Pengju Liu, Hongzhi Zhang, Wei Lian, Wangmeng Zuo
    https://arxiv.org/abs/1907.03128
       x shape - BCHW (channel first)88ik.
    """
    x = y.permute(0, 2, 3, 1)

    x1 = x[:, 0::2, 0::2, :]  # x(2i−1, 2j−1)
    x2 = x[:, 1::2, 0::2, :]  # x(2i, 2j-1)
    x3 = x[:, 0::2, 1::2, :]  # x(2i−1, 2j)
    x4 = x[:, 1::2, 1::2, :]  # x(2i, 2j)

    x_LL = x1 + x2 + x3 + x4
    x_LH = -x1 - x3 + x2 + x4
    x_HL = -x1 + x3 - x2 + x4
    x_HH = x1 - x3 - x2 + x4

    LL = x_LL.permute(0, 3, 1, 2)
    LH = x_LH.permute(0, 3, 1, 2)
    HL = x_HL.permute(0, 3, 1, 2)
    HH = x_HH.permute(0, 3, 1, 2)

    return LL, LH, HL, HH


class loss_func22(nn.Module):
    def __init__(self):
        super(loss_func22, self).__init__()

    def forward(self, ref, out):
        loss = nn.L1Loss()
        samloss = SAMLoss()
        # channel mean
        ref_dwt = dwt(ref)
        out_dwt = dwt(out)
        loss_l = samloss(ref_dwt[0], out_dwt[0])

        ref_gradient = torch.cat((ref_dwt[1], ref_dwt[2], ref_dwt[3]), 1)
        out_gradient = torch.cat((out_dwt[1], out_dwt[2], out_dwt[3]), 1)

        loss_h = loss(ref_gradient, out_gradient)
        loss_whole = loss(ref, out)

        # weight1 = loss_l / (loss_l + loss_h)
        # weight2 = loss_h / (loss_l + loss_h)

        loss_total = loss_whole + 0.1 * loss_l + 0.5 * loss_h
        # loss_total = loss_ref_out + loss_pan_out
        print('loss_whole:{}_loss_l:{}_loss_h:{}_loss_total:{}'.format(loss_whole, loss_l, loss_h, loss_total))
        return loss_total


class loss_func23(nn.Module):
    def __init__(self):
        super(loss_func23, self).__init__()

    def forward(self, ref, out, ms_2, out_2):
        loss = nn.L1Loss()
        samloss = SAMLoss()
        # channel mean
        ref_dwt = dwt(ref)
        out_dwt = dwt(out)
        loss_l = samloss(ref_dwt[0], out_dwt[0])

        ref_gradient = torch.cat((ref_dwt[1], ref_dwt[2], ref_dwt[3]), 1)
        out_gradient = torch.cat((out_dwt[1], out_dwt[2], out_dwt[3]), 1)

        loss_h = loss(ref_gradient, out_gradient)
        loss_whole = loss(ref, out)
        loss_1_2 = loss(ms_2, out_2)

        # weight1 = loss_l / (loss_l + loss_h)
        # weight2 = loss_h / (loss_l + loss_h)

        loss_total = loss_whole + 0.1 * loss_l + 0.3 * loss_h + 0.5 * loss_1_2
        # loss_total = loss_ref_out + loss_pan_out
        print('loss_whole:{}_loss_1_2:{}_loss_l:{}_loss_h:{}_loss_total:{}'.format(loss_whole, loss_1_2, loss_l, loss_h,
                                                                                   loss_total))
        return loss_total


# data and data similar
class FLoss3(nn.Module):
    # MLP to change high dimension feature to low dimension feature for CC loss calculation
    def __init__(self, dv, dv1, do, lambd=1):
        # dv is the M*N*C number of input feature-(B C M N)
        super(FLoss3, self).__init__()

        self.relu = nn.LeakyReLU(0.2)
        num_hid = int(math.sqrt(dv))

        self.bn0 = nn.BatchNorm1d(num_hid, affine=False)

        self.layer1 = nn.Sequential(nn.Linear(dv, num_hid), self.bn0, self.relu, nn.Linear(num_hid, do))
        self.layer2 = nn.Sequential(nn.Linear(dv1, num_hid), self.bn0, self.relu, nn.Linear(num_hid, do))
        self.bn = nn.BatchNorm1d(do, affine=False)

        self.lambd = lambd
        # self.layer3 = nn.Linear(dv, dv)

    def forward(self, F1, F2):
        # change the shape from (B C M N) to (B M*N*C) for nn.linear function
        F1 = torch.reshape(F1, (F1.size(0), F1.size(1) * F1.size(2) * F1.size(3)))
        F2 = torch.reshape(F2, (F2.size(0), F2.size(1) * F2.size(2) * F2.size(3)))
        # print("F1.shape:", F1.shape, F2.shape)

        # reduce dimension from (B M*N*C) to (B M*N*C)
        F1_1 = self.bn(self.layer1(F1))
        F2_1 = self.bn(self.layer2(F2))

        # print('F1_1:', F1_1.shape)
        # print('F2_1:', F2_1.shape)

        # empirical cross-correlation matrix, size-(batch_size, size)
        c = self.bn(F1_1) @ self.bn(F2_1).T  # mean-0 variation-1

        on_diag = torch.diagonal(c).add_(-1).pow_(2).mean()
        off_diag = off_diagonal(c).pow_(2).mean()
        loss = on_diag + self.lambd * off_diag

        return loss


# class loss_func24(nn.Module):
#     def __init__(self, dv=4096, dv1=4096 * args.bands, do=16, lambd=0.005):
#         super(loss_func24, self).__init__()
#         self.cc_loss_pan = FLoss3(dv, dv1, do, lambd)
#         self.cc_loss_ms = FLoss3(dv1, dv1, do, lambd)
#
#     def forward(self, ref, out, pan, ms):
#         # channel mean
#         loss = nn.L1Loss()
#         loss_ref_out = loss(ref, out)
#         #
#         loss_ms_out = self.cc_loss_pan(pan, out)
#         loss_pan_out = self.cc_loss_ms(ms, out)
#         loss_total = loss_ref_out + 0.001 * loss_pan_out + 0.001 * loss_ms_out
#         # loss_total = loss_ref_out + loss_pan_out
#         print('loss_ref_out:{}_loss_ms_out:{}_loss_pan_out:{}_loss_total:{}'.format(loss_ref_out, loss_ms_out,
#                                                                                     loss_pan_out, loss_total))
#         return loss_total


# data and data similar ours  lambd=0.005     lambd=0.0005  lambd=0.05
def FLoss4(v1, v2, lambd=0.005):
    c = v1 @ v2.T  # mean-0 variation-1
    # c = v1.T @ v2  # mean-0 variation-1 YUAN
    on_diag = torch.diagonal(c).add_(-1).pow_(2).mean()
    print('on_diag', on_diag)
    off_diag = off_diagonal(c).pow_(2).mean()
    print('off_diag', off_diag)
    loss = on_diag + lambd * off_diag

    return loss


# batch and batch similar yuan
def FLoss5(v1, v2, lambd=0.005):
    c = v1.T @ v2  # mean-0 variation-1 YUAN
    on_diag = torch.diagonal(c).add_(-1).pow_(2).mean()
    print('on_diag', on_diag)
    off_diag = off_diagonal(c).pow_(2).mean()
    print('off_diag', off_diag)
    loss = on_diag + lambd * off_diag
    return loss


class loss_func25(nn.Module):
    def __init__(self):
        super(loss_func25, self).__init__()

    def forward(self, ref, out, pan_, out_pan, ms_, out_ms):
        # channel mean
        loss = nn.L1Loss()
        loss_ref_out = loss(ref, out)
        #
        loss_ms_out = FLoss4(pan_, out_pan)
        loss_pan_out = FLoss4(ms_, out_ms)
        loss_total = loss_ref_out + 0.0001 * loss_pan_out + 0.0001 * loss_ms_out
        # loss_total = loss_ref_out + loss_pan_out
        # 0.1 0.01 0.001 makes net not merge
        print('loss_ref_out:{}_loss_ms_out:{}_loss_pan_out:{}_loss_total:{}'.format(loss_ref_out, loss_ms_out,
                                                                                    loss_pan_out, loss_total))
        return loss_total


class loss_func26(nn.Module):
    def __init__(self):
        super(loss_func26, self).__init__()

    def forward(self, ref, out, pan_, out_pan, ms_, out_ms):
        # channel mean
        loss = nn.L1Loss()
        loss_ref_out = loss(ref, out)
        #
        loss_ms_out = loss(pan_, out_pan)
        print('pan_.shape:', pan_.shape, 'out_pan.shape:', out_pan.shape)
        loss_pan_out = loss(ms_, out_ms)
        print('ms_.shape:', ms_.shape, 'out_ms.shape:', out_ms.shape)
        '''
        pan_.shape: torch.Size([32, 16]) out_pan.shape: torch.Size([32, 16])
        ms_.shape: torch.Size([32, 16]) out_ms.shape: torch.Size([32, 16])
        '''
        loss_total = loss_ref_out + 0.5 * loss_pan_out + 0.1 * loss_ms_out
        # loss_total = loss_ref_out + loss_pan_out
        # 0.1 0.01 0.001 makes net not merge
        print('loss_ref_out:{}_loss_ms_out:{}_loss_pan_out:{}_loss_total:{}'.format(loss_ref_out, loss_ms_out,
                                                                                    loss_pan_out, loss_total))
        return loss_total


class loss_func27(nn.Module):
    def __init__(self):
        super(loss_func27, self).__init__()

    def forward(self, ref, out, pan_, out_pan, ms_, out_ms):
        # channel mean
        loss = nn.L1Loss()
        loss_ref_out = loss(ref, out)
        #
        loss_ms_out = loss(pan_, out_pan)
        print('pan_.shape:', pan_.shape, 'out_pan.shape:', out_pan.shape)
        loss_pan_out = loss(ms_, out_ms)
        print('ms_.shape:', ms_.shape, 'out_ms.shape:', out_ms.shape)
        '''
        pan_.shape: torch.Size([32, 16]) out_pan.shape: torch.Size([32, 16])
        ms_.shape: torch.Size([32, 16]) out_ms.shape: torch.Size([32, 16])
        '''
        loss_total = loss_ref_out + 0.1 * loss_pan_out + 0.1 * loss_ms_out
        # loss_total = loss_ref_out + loss_pan_out
        # 0.1 0.01 0.001 makes net not merge
        print('loss_ref_out:{}_loss_ms_out:{}_loss_pan_out:{}_loss_total:{}'.format(loss_ref_out, loss_ms_out,
                                                                                    loss_pan_out, loss_total))
        return loss_total


# only make low part similar
class loss_func28(nn.Module):
    def __init__(self):
        super(loss_func28, self).__init__()

    def forward(self, ref, out, panl_latent, msl_latent):
        loss = nn.L1Loss()
        loss_ref = loss(ref, out)
        # print('pc_2.shape:', pc_2.shape, mc_2.shape)# torch.Size([32, 32, 16, 16]) torch.Size([32, 32, 16, 16])
        loss_l = FLoss4(panl_latent, msl_latent)
        loss_total = 10 * loss_ref + 0.1 * loss_l
        print('loss_ref{}: __loss_l{}: __loss_total{}:'.format(loss_ref, loss_l, loss_total))
        return loss_total


# only make low part similar ours
class loss_func29(nn.Module):
    def __init__(self):
        super(loss_func29, self).__init__()

    def forward(self, ref, out, panl_latent, msl_latent, panl_latent1, msl_latent1):
        loss = nn.L1Loss()
        loss_ref = loss(ref, out)
        # print('pc_2.shape:', pc_2.shape, mc_2.shape)# torch.Size([32, 32, 16, 16]) torch.Size([32, 32, 16, 16])
        loss_ll = FLoss4(panl_latent, msl_latent)
        loss_l = FLoss4(panl_latent1, msl_latent1)
        loss_total = loss_ref + 20 * loss_l + 20 * loss_ll  # 0.1  0.5 1 loss_l-0.1 * loss_ll  0.1  0.5 1 loss_ll-0.1 * loss_l
        # loss_total = loss_l + loss_ll
        # 0.5 + 0.5  1 + 1
        # 4-21 0.1 1 to 0.5 1
        print('loss_ref:{} __loss_l:{} __loss_ll:{} __loss_total:{}'.format(loss_ref, loss_l, loss_ll, loss_total))
        return loss_total


# only make low part similar ours
class loss_func29_h(nn.Module):
    def __init__(self):
        super(loss_func29_h, self).__init__()

    def forward(self, ref, out, panh_latent, msh_latent, panlh_latent, mslh_latent):
        loss = nn.L1Loss()
        loss_ref = loss(ref, out)
        # print('pc_2.shape:', pc_2.shape, mc_2.shape)# torch.Size([32, 32, 16, 16]) torch.Size([32, 32, 16, 16])
        loss_lh = FLoss4(panlh_latent, mslh_latent)
        loss_h = FLoss4(panh_latent, msh_latent)
        loss_total = loss_ref + 50 * (
                loss_h + loss_lh)  # 0.1  0.5 1 loss_l-0.1 * loss_ll  0.1  0.5 1 loss_ll-0.1 * loss_l

        print('loss_ref:{} __loss_h:{} __loss_lh:{} __loss_total:{}'.format(loss_ref, loss_h, loss_lh, loss_total))
        return loss_total


class loss_func29_h_3(nn.Module):
    def __init__(self):
        super(loss_func29_h_3, self).__init__()

    def forward(self, ref, out, panh_latent, msh_latent, panlh_latent, mslh_latent, panllh_latent, msllh_latent):
        loss = nn.L1Loss()
        loss_ref = loss(ref, out)
        # print('pc_2.shape:', pc_2.shape, mc_2.shape)# torch.Size([32, 32, 16, 16]) torch.Size([32, 32, 16, 16])
        loss_lh = FLoss4(panlh_latent, mslh_latent)
        loss_h = FLoss4(panh_latent, msh_latent)
        loss_llh = FLoss4(panllh_latent, msllh_latent)
        loss_total = loss_ref + 20 * (
                loss_h + loss_lh + loss_llh)  # 0.1  0.5 1 loss_l-0.1 * loss_ll  0.1  0.5 1 loss_ll-0.1 * loss_l
        # 0.5 + 0.5  1 + 1
        # 4-21 0.1 1 to 0.5 1
        print('loss_ref:{} __loss_h:{} __loss_lh:{} __loss_llh:{} __loss_total:{}'.format(loss_ref, loss_h, loss_lh,
                                                                                          loss_llh, loss_total))
        return loss_total


# only make low part similar ours
class loss_func29_lh(nn.Module):
    def __init__(self):
        super(loss_func29_lh, self).__init__()

    def forward(self, ref, out, panl_latent1, msl_latent1, panl_latent, msl_latent, panh_latent, msh_latent,
                panlh_latent, mslh_latent):
        loss = nn.L1Loss()
        loss_ref = loss(ref, out)
        # print('pc_2.shape:', pc_2.shape, mc_2.shape)# torch.Size([32, 32, 16, 16]) torch.Size([32, 32, 16, 16])
        loss_ll = FLoss4(panl_latent, msl_latent)
        loss_l = FLoss4(panl_latent1, msl_latent1)

        loss_lh = FLoss4(panlh_latent, mslh_latent)
        loss_h = FLoss4(panh_latent, msh_latent)
        loss_total = loss_ref + 20 * (loss_h + loss_lh + loss_l + loss_ll)
        # 0.5 + 0.5  1 + 1
        # 4-21 0.1 1 to 0.5 1
        print('loss_ref:{} __loss_l:{} __loss_ll:{} __loss_h:{} __loss_lh:{} __loss_total:{}'.format(loss_ref, loss_l,
                                                                                                     loss_ll, loss_h,
                                                                                                     loss_lh,
                                                                                                     loss_total))
        return loss_total


# only make low part similar ours

class loss_func29_c(nn.Module):
    def __init__(self):
        super(loss_func29_c, self).__init__()

    def forward(self, ref, out, panl_latent, msl_latent, panl_latent1, msl_latent1):
        loss = nn.L1Loss()
        loss_ref = loss(ref, out)
        # print('pc_2.shape:', pc_2.shape, mc_2.shape)# torch.Size([32, 32, 16, 16]) torch.Size([32, 32, 16, 16])
        loss_ll = FLoss4(panl_latent, msl_latent)
        loss_l = FLoss4(panl_latent1, msl_latent1)
        loss_total = loss_ref + 0.1 * loss_l + 1 * loss_ll  # 0.1  0.5 1 loss_l-0.1 * loss_ll  0.1  0.5 1 loss_ll-0.1 * loss_l
        # 0.5 + 0.5  1 + 1
        # 4-21 0.1 1 to 0.5 1
        print('loss_ref:{} __loss_l:{} __loss_ll:{} __loss_total:{}'.format(loss_ref, loss_l, loss_ll, loss_total))
        return loss_total


class loss_func29_3layer(nn.Module):
    def __init__(self):
        super(loss_func29_3layer, self).__init__()

    def forward(self, ref, out, panl_latent, msl_latent, panll_latent, msll_latent, panlll_latent, mslll_latent):
        loss = nn.L1Loss()
        loss_ref = loss(ref, out)
        # print('pc_2.shape:', pc_2.shape, mc_2.shape)# torch.Size([32, 32, 16, 16]) torch.Size([32, 32, 16, 16])
        loss_l = FLoss4(panl_latent, msl_latent)
        loss_ll = FLoss4(panll_latent, msll_latent)
        loss_lll = FLoss4(panlll_latent, mslll_latent)
        loss_total = loss_ref + 0.1 * loss_l + 1 * loss_ll + 1 * loss_lll

        print('loss_ref:{} __loss_l:{} __loss_ll:{} __loss_lll:{} __loss_total:{}'.format(loss_ref, loss_l, loss_ll,
                                                                                          loss_lll, loss_total))
        return loss_total


class loss_func29_a(nn.Module):
    def __init__(self):
        super(loss_func29_a, self).__init__()

    def forward(self, ref, out, panl_latent, msl_latent, panl_latent1, msl_latent1):
        loss = nn.L1Loss()
        loss_ref = loss(ref, out)
        # print('pc_2.shape:', pc_2.shape, mc_2.shape)# torch.Size([32, 32, 16, 16]) torch.Size([32, 32, 16, 16])
        loss_ll = loss(panl_latent, msl_latent)
        loss_l = loss(panl_latent1, msl_latent1)
        loss_total = loss_ref + 1 * loss_l + 1 * loss_ll  # 0.1  0.5 1 loss_l-0.1 * loss_ll  0.1  0.5 1 loss_ll-0.1 * loss_l
        # 0.5 + 0.5  1 + 1
        # 4-21 0.1 1 to 0.5 1
        print('loss_ref:{} __loss_l:{} __loss_ll:{} __loss_total:{}'.format(loss_ref, loss_l, loss_ll, loss_total))
        return loss_total


# only make low part similar yuan cc loss
class loss_func29_1(nn.Module):
    def __init__(self):
        super(loss_func29_1, self).__init__()

    def forward(self, ref, out, panl_latent, msl_latent, panl_latent1, msl_latent1):
        loss = nn.L1Loss()
        loss_ref = loss(ref, out)
        # print('pc_2.shape:', pc_2.shape, mc_2.shape)# torch.Size([32, 32, 16, 16]) torch.Size([32, 32, 16, 16])
        loss_ll = FLoss5(panl_latent, msl_latent)
        loss_l = FLoss5(panl_latent1, msl_latent1)
        loss_total = loss_ref + 0.1 * loss_l + 1 * loss_ll  # 0.1  0.5 1 loss_l-0.1 * loss_ll  0.1  0.5 1 loss_ll-0.1 * loss_l
        # 0.5 + 0.5  1 + 1
        print('loss_ref:{} __loss_l:{} __loss_ll:{} __loss_total:{}'.format(loss_ref, loss_l, loss_ll, loss_total))
        return loss_total


# only make high part similar
class loss_func30(nn.Module):
    def __init__(self):
        super(loss_func30, self).__init__()

    def forward(self, ref, out, panl_latent, msl_latent, panl_latent1, msl_latent1):
        loss = nn.L1Loss()
        loss_ref = loss(ref, out)
        # print('pc_2.shape:', pc_2.shape, mc_2.shape)# torch.Size([32, 32, 16, 16]) torch.Size([32, 32, 16, 16])
        loss_h2 = FLoss4(panl_latent, msl_latent)
        loss_h1 = FLoss4(panl_latent1, msl_latent1)
        loss_total = loss_ref + 0.1 * loss_h1 + 0.1 * loss_h2
        print('loss_ref:{} __loss_h1:{} __loss_h2:{} __loss_total:{}'.format(loss_ref, loss_h1, loss_h2, loss_total))
        return loss_total


# make high and low part similar
class loss_func31(nn.Module):
    def __init__(self):
        super(loss_func31, self).__init__()

    def forward(self, ref, out, panll_latent, msll_latent, panl_latent, msl_latent, panh2_latent, msh2_latent,
                panh1_latent, msh1_latent):
        loss = nn.L1Loss()
        loss_ref = loss(ref, out)
        # print('pc_2.shape:', pc_2.shape, mc_2.shape)# torch.Size([32, 32, 16, 16]) torch.Size([32, 32, 16, 16])
        loss_h2 = FLoss4(panh2_latent, msh2_latent)
        loss_h1 = FLoss4(panh1_latent, msh1_latent)
        loss_l = FLoss4(panl_latent, msl_latent)
        loss_ll = FLoss4(panll_latent, msll_latent)
        loss_total = loss_ref + 0.1 * loss_h1 + 0.1 * loss_h2 + 0.1 * loss_l + 0.1 * loss_ll
        print('loss_ref:{} __loss_l:{} __loss_ll:{} __loss_h1:{} __loss_h2:{} __loss_total:{}'.format(loss_ref, loss_l,
                                                                                                      loss_ll, loss_h1,
                                                                                                      loss_h2,
                                                                                                      loss_total))
        return loss_total


# def dwt(y):
#     """
#     DWT (Discrete Wavelet Transform) function implementation according to
#     "Multi-level Wavelet Convolutional Neural Networks"
#     by Pengju Liu, Hongzhi Zhang, Wei Lian, Wangmeng Zuo
#     https://arxiv.org/abs/1907.03128
#        x shape - BCHW (channel first)88ik.
#     """
#     x = y.permute(0, 2, 3, 1)
#
#     x1 = x[:, 0::2, 0::2, :]  # x(2i−1, 2j−1)
#     x2 = x[:, 1::2, 0::2, :]  # x(2i, 2j-1)
#     x3 = x[:, 0::2, 1::2, :]  # x(2i−1, 2j)
#     x4 = x[:, 1::2, 1::2, :]  # x(2i, 2j)
#
#     x_LL = x1 + x2 + x3 + x4
#     x_LH = -x1 - x3 + x2 + x4
#     x_HL = -x1 + x3 - x2 + x4
#     x_HH = x1 - x3 - x2 + x4
#
#     LL = x_LL.permute(0, 3, 1, 2)
#     LH = x_LH.permute(0, 3, 1, 2)
#     HL = x_HL.permute(0, 3, 1, 2)
#     HH = x_HH.permute(0, 3, 1, 2)
#
#     return LL, LH, HL, HH


class loss_func32(nn.Module):
    def __init__(self):
        super(loss_func32, self).__init__()

    def forward(self, ref, out, panh_latent, msh_latent, panlh_latent, mslh_latent):
        loss = nn.L1Loss()

        # channel mean
        ref_dwt = dwt(ref)
        out_dwt = dwt(out)

        loss_l_fer = loss(ref_dwt[0], out_dwt[0])

        ref_gradient = torch.cat((ref_dwt[1], ref_dwt[2], ref_dwt[3]), 1)
        out_gradient = torch.cat((out_dwt[1], out_dwt[2], out_dwt[3]), 1)

        loss_h_fer = loss(ref_gradient, out_gradient)
        loss_ref = loss(ref, out)

        loss_lh = FLoss4(panlh_latent, mslh_latent)
        loss_h = FLoss4(panh_latent, msh_latent)
        loss_total = loss_ref + loss_h_fer + loss_l_fer + 20 * (
                loss_h + loss_lh)

        print('loss_ref:{}  __loss_h_fer:{} __loss_l_fer:{} __loss_h:{} __loss_lh:{} __loss_total:{}'.format(loss_ref,
                                                                                                             loss_h_fer,
                                                                                                             loss_l_fer,
                                                                                                             loss_h,
                                                                                                             loss_lh,
                                                                                                             loss_total))

        return loss_total


# def apply_colormap_on_image(org_im, activation, colormap_name):
#     """
#         Apply heatmap on image
#     Args:
#         org_img (PIL img): Original image
#         activation_map (numpy arr): Activation map (grayscale) 0-255
#         colormap_name (str): Name of the colormap
#     """
#     # Get colormap
#     color_map = mpl_color_map.get_cmap(colormap_name)
#     no_trans_heatmap = color_map(activation)
#     # Change alpha channel in colormap to make sure original image is displayed
#     heatmap = copy.copy(no_trans_heatmap)
#     heatmap[:, :, 3] = 0.4
#     heatmap = Image.fromarray((heatmap * 255).astype(np.uint8))
#     no_trans_heatmap = Image.fromarray((no_trans_heatmap * 255).astype(np.uint8))
#
#     # Apply heatmap on image
#     heatmap_on_image = Image.new("RGBA", org_im.size)
#     heatmap_on_image = Image.alpha_composite(heatmap_on_image, org_im.convert('RGBA'))
#     heatmap_on_image = Image.alpha_composite(heatmap_on_image, heatmap)
#     return no_trans_heatmap, heatmap_on_image


# def apply_heatmap(R, sx, sy):
#     """
#         Heatmap code stolen from https://git.tu-berlin.de/gmontavon/lrp-tutorial
#
#         This is (so far) only used for LRP
#     """
#     b = 10 * ((np.abs(R) ** 3.0).mean() ** (1.0 / 3))
#     my_cmap = plt.cm.seismic(np.arange(plt.cm.seismic.N))
#     my_cmap[:, 0:3] *= 0.85
#     my_cmap = ListedColormap(my_cmap)
#     plt.figure(figsize=(sx, sy))
#     plt.subplots_adjust(left=0, right=1, bottom=0, top=1)
#     plt.axis('off')
#     heatmap = plt.imshow(R, cmap=my_cmap, vmin=-b, vmax=b, interpolation='nearest')
#     return heatmap
#     # plt.show()


if __name__ == '__main__':
    from PIL import Image

    # 读入图片
    img_path = ('1.jpeg')
    img = Image.open(img_path)
    # 转换格式，变成numpy.ndarray
    img_arr = np.array(img)
    print(img_arr.shape)  # (197, 316, 3)

    # 归一化
    max = np.max(img_arr)
    min = np.min(img_arr)
    img_arr = ((img - min) / (max - min + 0.0))  # 255 *
    plt.figure()
    plt.subplot(1, 2, 1)
    plt.imshow(img - np.mean(img))

    image_hwc = img_arr
    # 从hwc排列顺序变成chw
    image_chw = np.transpose(image_hwc, (2, 0, 1))
    print(image_chw.shape)  # (3, 197, 316)
    c, h, w = image_chw.shape
    # 加一维，代表图片数量，和函数输入格式[N,C,H,W]保持一致
    img_nchw = image_chw.reshape([1, c, h, w])
    print(img_nchw.shape)  # (1, 3, 197, 316)

    # pan = torch.from_numpy(img_nchw[:, 0, :, :]).float().unsqueeze(1)
    #
    # plt.subplot(1, 4, 2)
    # plt.imshow(np.squeeze(pan.detach().numpy()))

    # 为了可以进行运算，将即将输入小波层的图片格式再做调整；一个是把byte数据格式变成float，再者变成GPU运算的格式，与变量matrix_low_0他们保持一致
    img_nchw = torch.from_numpy(img_nchw).float()

    # ms = F.interpolate(img_nchw, scale_factor=0.25)
    # print("ms:{}".format(ms.shape))
    # plt.subplot(1, 4, 3)
    # plt.imshow(np.squeeze(ms.detach().numpy()).transpose(1, 2, 0))
    # print("ms:{}--pan:{}".format(ms.shape, pan.shape))

    # 单通道测试
    # img_nchw = torch.from_numpy(img_nchw[:, 0, :, :]).float().unsqueeze(1)

    print(img_nchw.shape)
    # 开始调用函数
    x, y = sobel_gradient(img_nchw.cuda())  # lpls_gradient   sobel_gradient  x, y
    z = np.squeeze((x + y).detach().numpy()).transpose(1, 2, 0)  # x+y  z

    # 单通道测试
    # z=np.squeeze((x+y).detach().numpy())

    plt.subplot(1, 2, 2)
    plt.imshow(z)
    plt.show()

    # # 开始调用函数
    # loss = loss_func1(img_nchw, pan, ms, img_nchw)
    # print(loss)
