"""
@author:      Swing
@create:      2020-05-11 15:57
@desc:
"""

import numpy as np
import torch


def iou(box, boxes, is_min=False):
    box_area = (box[3] - box[1]) * (box[4] - box[2])
    area = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 4] - boxes[:, 2])
    xx1 = torch.max(box[1], boxes[:, 1])
    yy1 = torch.max(box[2], boxes[:, 2])
    xx2 = torch.min(box[3], boxes[:, 3])
    yy2 = torch.min(box[4], boxes[:, 4])

    w = torch.clamp(xx2 - xx1, min=0)
    h = torch.clamp(yy2 - yy1, min=0)

    inter = w * h
    if is_min:
        ovr = inter / torch.min(box_area, area)
    else:
        ovr = inter / (box_area + area - inter)

    return ovr


def nms(boxes, thresh=0.6, is_min=True):

    if boxes.shape[0] == 0:
        return torch.tensor(np.array([]))
    _boxes = boxes[(-boxes[:, 0]).argsort()]
    r_boxes = []

    while _boxes.shape[0] > 1:
        a_box = _boxes[0]
        b_boxes = _boxes[1:]

        r_boxes.append(a_box)

        index = np.where(iou(a_box, b_boxes, is_min) < thresh)
        _boxes = b_boxes[index]

    if _boxes.shape[0] > 0:
        r_boxes.append(_boxes[0])

    return torch.stack(r_boxes)


if __name__ == '__main__':

    bs = torch.tensor([[1.0, 1.0, 10.0, 10.0, 40.0,8.0], [1.0, 1.0, 9.0, 9.0, 10.0,9.0], [9.0, 8.0, 13.0, 20.0, 15.0,3.0], [6.0, 11.0, 13.0, 17.0, 18.0,2.0]])
    print(bs[:,3].argsort())
    print(nms(bs))