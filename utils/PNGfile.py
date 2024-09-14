# 1. read file or tokenize, using tif format for preview and get answer
# 2. answer.json分析
import h5py
import numpy as np


def read_file(file_path):
    data = h5py.File(file_path)
    if data.get('gt', None) is None:
        gt1 = data['lms'][...]
    else:
        gt1 = data['gt'][...]

    gt1 = np.array(gt1, dtype=np.float32)
    ms1 = np.array(data["ms"][...], dtype=np.float32)  # convert to np tpye for CV2.filter
    lms1 = np.array(data["lms"][...], dtype=np.float32)  # convert to np tpye for CV2.filter
    pan1 = np.array(data['pan'][...], dtype=np.float32)  # Nx1xHxW
    return gt1, ms1, lms1, pan1


from PIL import Image

import json
import cv2
import skimage.io as io


def save_img(img, img_path, mode='RGB'):
    if mode == 'gray':
        cv2.imwrite(img_path, img)
    else:
        cv2.imwrite(img_path, cv2.cvtColor(img, cv2.COLOR_RGB2BGR))


def tif_format(output_dir, gt1, ms1, lms1, pan1):
    import os
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    answer_file = os.path.join(output_dir, "question.jsonl")
    ans_file = open(answer_file, "w")
    for i in range(gt1.shape[0]):
        # 获取单张图片 gt1[i][[0,2,4],:,:]
        image = gt1[i][[0,2,4],:,:]
        image = image.transpose(1, 2, 0)  # 转换为 (H, W, C) 格式
        print(image.ndim, image.shape)

        # 确保图像数据的类型为Pillow可以处理的类型
        image = (image - np.min(image)) / (np.max(image) - np.min(image)) * 255.0

        # 指定图片的文件名
        filename = f'output_{i}.png'

        # 构建完整的文件路径
        file_path = os.path.join(output_dir, filename)

        # 保存图像到TIFF格式
        save_img(image, file_path)

        question_id = file_path
        image_id = file_path
        text = "Classify the given image in one of the following classes. Classes: dense residential, river, overpass, medium residential, tennis court, agricultural, intersection, buildings, freeway, runway, chaparral, storage tanks, parking lot, sparse residential, beach, forest, baseball diamond, golf course, mobile home park, airplane, harbor. \nAnswer in one word or a short phrase."
        ground_truth = "GaoFen2"
        ans_file.write(json.dumps({
            "question_id": question_id,
            "image": image_id,
            "text": text,
            "ground_truth": ground_truth
        }) + "\n")
        ans_file.flush()

    print(f'所有图片已成功保存为TIFF格式到 {output_dir}')
    ans_file.close()




if __name__ == "__main__":
    file_path = "/data/qlt/h5/test_data/WV2/test_wv2_data_FR.h5"
    output_dir = "WV2_full"
    gt1, ms1, lms1, pan1 = read_file(file_path)
    tif_format(output_dir, gt1, ms1, lms1, pan1)
