"""
@author:      Swing
@create:      2020-05-05 17:06
@desc:
"""

from torch.utils.data import Dataset, DataLoader
import torchvision
import os
from PIL import Image
import math
import numpy as np
import config
import utils

LABEL_FILE_PATH = 'D:\datasets\yolodata\label.txt'
IMG_BASE_DIR = 'D:\datasets\yolodata\images'

transforms = torchvision.transforms.Compose([
    torchvision.transforms.ToTensor()
])


def one_hot(class_num, i):
    b = np.zeros(class_num)
    b[i] = 1.
    return b


class MyDataSet(Dataset):
    def __init__(self):
        with open(LABEL_FILE_PATH) as f:
            self.dataset = f.readlines()

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, item):
        labels = {}
        line = self.dataset[item]
        strs = line.split()
        img_data = Image.open(os.path.join(IMG_BASE_DIR, strs[0]))

        boxes = np.array(list(map(float, strs[1:])))
        boxes = np.split(boxes, len(boxes) // 5)
        boxes = np.stack(boxes)

        # 原始图片和标记框转换成符合出入要求的尺寸
        img_data, boxes = utils.img_preprocess(img_data, boxes)
        img_data = transforms(img_data)

        for feature_size, anchors in config.ANCHORS_GROUP_KMEANS.items():
            labels[feature_size] = np.zeros(shape=(feature_size, feature_size, 3, 5 + config.CLASS_NUM))
            for box in boxes:
                cls, cx, cy, w, h = box
                cx_offset, cx_index = math.modf(cx * feature_size / config.IMG_WIDTH)
                cy_offset, cy_index = math.modf(cy * feature_size / config.IMG_HEIGHT)

                for i, anchor in enumerate(anchors):
                    anchor_area = config.ANCHORS_GROUP_KMEANS_AREA[feature_size][i]
                    p_w, p_h = w / anchor[0], h / anchor[1]
                    p_area = w * h
                    iou = min(p_area, anchor_area) / max(p_area, anchor_area)
                    labels[feature_size][int(cy_index), int(cx_index), i] = np.array(
                        [iou, cx_offset, cy_offset, np.log(p_w), np.log(p_h), *one_hot(config.CLASS_NUM, int(cls))]
                    )

        return labels[13], labels[26], labels[52], img_data


if __name__ == '__main__':
    x = one_hot(10, 2)
    print(x)

    data = MyDataSet()
    dataLoader = DataLoader(data, 1, shuffle=True)

    for t_13, t_26, t_52, img in dataLoader:
        print(t_13.shape)
        print(t_26.shape)
        print(t_52.shape)
