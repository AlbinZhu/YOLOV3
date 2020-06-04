"""
@author:      Swing
@create:      2020-05-11 15:55
@desc:
"""

from model import *
import config
import torch
from PIL import Image, ImageDraw
import tool
from utils import img_preprocess
import torchvision


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

        return torch.cat([boxes_13, boxes_26, boxes_52], dim=0)

    @staticmethod
    def _filter(output, threshold):
        output = output.permute(0, 2, 3, 1)
        output = output.reshape(output.size(0), output.size(1), output.size(2), 3, -1)
        mask = output[..., 0] > threshold
        idxs = torch.nonzero(mask)
        vecs = output[mask]
        return idxs, vecs

    @staticmethod
    def _parse(idxs, vecs, t, anchors):
        anchors = torch.tensor(anchors)
        a = idxs[:, 3]
        confidence = vecs[:, 0]
        _classify = vecs[:, 5:]

        if len(_classify) == 0:
            classify = torch.tensor([])
        else:
            classify = torch.argmax(_classify, dim=1).float()

        cy = (idxs[:, 1].float() + vecs[:, 2]) * t
        cx = (idxs[:, 2].float() + vecs[:, 1]) * t
        w = anchors[a, 0] * torch.exp(vecs[:, 3])
        h = anchors[a, 1] * torch.exp(vecs[:, 4])
        x1 = cx - w / 2
        y1 = cy - h / 2
        x2 = x1 + w
        y2 = y1 + h
        output = torch.stack([confidence, x1, y1, x2, y2, classify], dim=1)
        return output


if __name__ == '__main__':

    class_map = {0: 'Cat', 1: 'Person', 2: 'Horse'}

    save_path = r'models/net_yolo.pth'
    detector = Detector(save_path)

    transforms = torchvision.transforms.Compose([
        torchvision.transforms.ToTensor()
    ])

    img1 = Image.open(r'test.jpg')
    img1, _ = img_preprocess(img1, size=416)

    # img = np.array(img) / 255
    # img = torch.tensor(img)
    # img = img.unsqueeze(0)
    # img = img.permute(0, 3, 1, 2)
    # img = img.cuda()

    out_value = detector(transforms(img1).unsqueeze(0), 0.3, config.ANCHORS_GROUP_KMEANS)
    boxes = []

    for j in range(10):
        classify_mask = (out_value[..., -1] == j)
        _boxes = out_value[classify_mask]
        boxes.append(tool.nms(_boxes))

    for box in boxes:
        try:
            img_draw = ImageDraw.ImageDraw(img1)
            c, x1, y1, x2, y2, cls = box[0, 0: 6]
            print(c.item(), x1.item(), y1.item(), x2.item(), y2.item())
            img_draw.rectangle((x1.item(), y1.item(), x2.item(), y2.item()), outline='red')
            img_draw.text((x1.item(), y1.item()), class_map[int(cls.item())], fill='red')
        except:
            continue
    img1.show()