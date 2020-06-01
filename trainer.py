"""
@author:      Swing
@create:      2020-05-11 11:08
@desc:
"""

from dataset import MyDataSet
from model import *
import torch
from torch.utils.data import DataLoader
from torch.optim import Adam
import os

def loss_func(output, target, alpha):

    conf_loss = torch.nn.BCEWithLogitsLoss()
    crood_loss = torch.nn.MSELoss()
    cls_loss_fn = torch.nn.CrossEntropyLoss()

    output = output.permute(0, 2, 3, 1)


    output = output.reshape(output.size(0), output.size(1), output.size(2), 3, -1)
    output = output.cpu().double()
    mask_obj = target[..., 0] > 0
    output_obj = output[mask_obj]
    target_obj = target[mask_obj]

    loss_obj_conf = conf_loss(output_obj[:, 0], target_obj[:, 0])
    loss_obj_crood = crood_loss(output_obj[:, 1: 5], target_obj[:, 1: 5])
    loss_obj_cls = cls_loss_fn(output_obj[:, 5:], target_obj[:, 5:])
    loss_obj = loss_obj_conf + loss_obj_crood + loss_obj_cls

    mask_noobj = target[..., 0] == 0
    output_noobj = output[mask_noobj]
    target_noobj = target[mask_noobj]
    loss_noobj = conf_loss(output_noobj[:, 0], target_noobj[:, 0])
    loss = alpha * loss_obj + (1 - alpha) * loss_noobj

    return loss

if __name__ == '__main__':
    save_path = 'models/net_yolo.pth'
    dataset = MyDataSet()
    train_loader = DataLoader(dataset, batch_size=2, shuffle=True)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    net = Net().to(device)

    if os.path.exists(save_path):
        net.load_state_dict(torch.load(save_path))
    else:
        print('No save data')

    net.train()
    opt = Adam(net.parameters())

    epoch = 0
    while True:
        for target_13, target_26, target_52, img_data in train_loader:
            img_data = img_data.to(device)
            output_13, output_26, output_52 = net(img_data)
            loss_13 = loss_func(output_13, target_13, 0.9)
            loss_26 = loss_func(output_26, target_26, 0.9)
            loss_52 = loss_func(output_52, target_52, 0.9)
            loss = loss_13 + loss_26 + loss_52

            opt.zero_grad()
            loss.backward()
            if epoch % 10 == 0:
                torch.save(net.state_dict(), save_path)
                print('save {}'. format(epoch))
            print(loss.item())
        epoch += 1