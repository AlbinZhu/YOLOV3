"""
@author:      Swing
@create:      2020-06-02 23:48
@desc:
"""

import os
from xml.etree import ElementTree

if __name__ == '__main__':

    class_map = {'Cat': '0', 'Person': '1', 'Horse': '2'}

    label_path = r'D:\datasets\yolodata\label.txt'
    xml_dir = r'D:\datasets\yolodata\outputs'

    with open(label_path, 'a') as label_file:
        xml_list = os.listdir(xml_dir)

        for xml_file in xml_list:
            label_arr = []
            xml_path = os.path.join(xml_dir, xml_file)
            xml = ElementTree.parse(xml_path)
            root = xml.getroot()
            image_name = root.find('filename').text
            label_arr.append(image_name)

            for obj in root.findall('object'):
                cls_name = obj.find('name').text
                box = obj.find('bndbox')
                x_min = int(box.find('xmin').text)
                y_min = int(box.find('ymin').text)
                x_max = int(box.find('xmax').text)
                y_max = int(box.find('ymax').text)

                w = x_max - x_min
                h = y_max - y_min
                cx = x_min + int(w / 2)
                cy = y_min + int(h / 2)

                label_arr += [class_map[cls_name], str(cx), str(cy), str(w), str(h)]

            label_str = ' '.join(label_arr) + '\n'
            label_file.write(label_str)