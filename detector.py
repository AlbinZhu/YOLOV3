"""
@author:      Swing
@create:      2020-05-11 15:55
@desc:
"""

from model import *
import config
import torch
import numpy as np
from PIL import Image, ImageDraw
import tool

class Detector(torch.nn.Module):

    def __init__(self, save_path):
        super(Detector, self).__init__()

        self.net = Net()
        self.net.load_state_dict(torch.load(save_path))
        self.net.eval()

    def forward(self, input, threshold, anchors):
        output_13, output_26, output_52 = self.net(input)

        idxs_13, vecs_13 = self._filter(output_13, threshold)
        boxes_13 = self._parse(idxs_13, vecs_13, 32, anchors[13])

        idxs_26, vecs_26 = self._filter(output_26, threshold)
        boxes_26 = self._parse(idxs_26, vecs_26, 16, anchors[26])

        idxs_52, vecs_52 = self._filter(output_52, threshold)
        boxes_52 = self._parse(idxs_52, vecs_52, 8, anchors[52])

        return torch.cat([boxes_13, boxes_26, boxes_52], dim=1)

    def _filter(self, output, threshold):
        output = output.permute(0, 2, 3, 1)
        output = output.reshape(output.size(0), output.size(1), output.size(2), 3, -1)
        mask = output[..., 0] > threshold
        idxs = mask.nonzeros()
        vecs = output[mask]
        return idxs, vecs

    def _parse(self, idxs, vecs, t, anchors):
        anchors = torch.Tensor(anchors)
        a = idxs[:, 3]
        confidence = vecs[:, 0]
        _classify = vecs[:, 5:]

        if len(_classify) == 0:
            classify = torch.Tensor([])
        else:
            classify = torch.argmax(_classify, dim=1).float()

        cy = (idxs[:, 1].float() + vecs[:, 2]) * t
        cx = (idxs[:, 2].float() + vecs[:, 1]) * t
        w = anchors[a, 0] * torch.exp(vecs[:, 3])
        h = anchors[1, 1] * torch.exp(vecs[:, 4])
        x1 = cx - w / 2
        y1 = cy - h / 2
        x2 = x1 + w
        y2 = y1 + h
        output = torch.stack([confidence, x1, y1, x2, y2, classify], dim=1)
        return output


if __name__ == '__main__':
    save_path = r''
    detector = Detector(save_path)

    img1 = Image.open(r'')
    img = img1.convert("RGB")
    img = np.array(img) / 255
    img = torch.Tensor(img)
    img = img.unsqueeze(0)
    img = img.permute(0, 3, 1, 2)
    img = img.cuda()

    out_value = detector(img, 0.3, config.ANCHORS_GROUP)
    boxes = []

    for j in range(10):
        classify_mask = (out_value[..., -1] == j)
        _boxes = out_value[classify_mask]
        boxes.append(tool.nms(_boxes))

    for box in boxes:
        try:
            img_drwa = ImageDraw.ImageDraw(img1)
            c, x1, y1, x2, y2 = box[0, 0: 5]
            print(c, x1, y1, x2, y2)
            img_drwa.rectangle((x1, y1, x2, y2))
        except:
            continue
    img1.show()