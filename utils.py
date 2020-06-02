"""
@author:      Swing
@create:      2020-06-02 17:33
@desc:
"""

from PIL import Image, ImageDraw
import numpy as np


def img_preprocess(image: Image, boxes: np.ndarray = None, size=416):
    if image.size == (size, size):
        return image, boxes

    iw, ih = image.size
    if iw != ih:
        iw = ih = max(iw, ih)

    new_image = Image.new('RGB', (iw, ih), (128, 128, 128))
    x_offset = int((iw - image.size[0]) / 2)
    y_offset = int((ih - image.size[1]) / 2)
    anchor = (x_offset, y_offset)
    new_image.paste(image, anchor)
    new_image = new_image.resize((size, size), Image.BICUBIC)

    scale = size / iw
    if boxes is not None:
        boxes[:, 0] = boxes[:, 0] + x_offset
        boxes[:, 1] = boxes[:, 1] + y_offset
        boxes = (boxes * scale).astype(np.int)
    return new_image, boxes


if __name__ == '__main__':
    file = r'D:\datasets\yolodata\001.jpg'
    # file = r"C:\Users\Administrator\Desktop\test.png"
    test_img = Image.open(file)

    test_boxes = np.array([
        [5, 187, 392, 793],
        [372, 241, 657, 796],
        [635, 449, 1046, 796],
        [958, 135, 1258, 788]
    ])

    n = np.zeros((4, 4))
    n[:, 2] = test_boxes[:, 2] - test_boxes[:, 0]
    n[:, 3] = test_boxes[:, 3] - test_boxes[:, 1]
    n[:, 0] = test_boxes[:, 0] + n[:, 2] / 2
    n[:, 1] = test_boxes[:, 1] + n[:, 3] / 2

    n = n.astype(np.int)
    img, b = img_preprocess(test_img, n)
    draw = ImageDraw.Draw(img)
    for i in range(b.shape[0]):
        box = b[i]
        # w, h = box[2], box[3]
        cx, cy, w, h = box
        x1 = int(cx - w / 2)
        y1 = int(cy - h / 2)
        x2 = x1 + w
        y2 = y1 + h

        draw.rectangle((x1, y1, x2, y2), outline='red')
    img.show()
