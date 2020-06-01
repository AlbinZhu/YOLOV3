"""
@author:      Swing
@create:      2020-05-06 17:46
@desc:
"""

import torch
import torch.nn.functional as func


class ConvolutionLayer(torch.nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, bias=False):
        super(ConvolutionLayer, self).__init__()

        self.sub_module = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=bias),
            torch.nn.BatchNorm2d(out_channels),
            torch.nn.LeakyReLU(0.1)
        )

    def forward(self, x):
        return self.sub_module(x)


class UpSampleLayer(torch.nn.Module):
    def __init__(self):
        super(UpSampleLayer, self).__init__()

    def forward(self, x):
        return func.interpolate(x, scale_factor=2, mode='nearest')


class DownSampleLayer(torch.nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DownSampleLayer, self).__init__()

        self.sub_module = torch.nn.Sequential(
            ConvolutionLayer(in_channels, out_channels, 3, 2, 1)
        )

    def forward(self, x):
        return self.sub_module(x)


class ResidualLayer(torch.nn.Module):
    def __init__(self, in_channels):
        super(ResidualLayer, self).__init__()

        self.sub_module = torch.nn.Sequential(
            ConvolutionLayer(in_channels, in_channels // 2, 1, 1, 0),
            ConvolutionLayer(in_channels // 2, in_channels, 3, 1, 1),
        )

    def forward(self, x):
        return x + self.sub_module(x)


class ConvolutionSet(torch.nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ConvolutionSet, self).__init__()

        self.sub_module = torch.nn.Sequential(
            ConvolutionLayer(in_channels, out_channels, 1, 1, 0),
            ConvolutionLayer(out_channels, in_channels, 3, 1, 1),

            ConvolutionLayer(in_channels, out_channels, 1, 1, 0),
            ConvolutionLayer(out_channels, in_channels, 3, 1, 1),

            ConvolutionLayer(in_channels, out_channels, 1, 1, 0)
        )

    def forward(self, x):
        return self.sub_module(x)


class Net(torch.nn.Module):

    def __init__(self):
        super(Net, self).__init__()

        self.trunk_52 = torch.nn.Sequential(
            ConvolutionLayer(3, 32, 3, 1, 1),
            DownSampleLayer(32, 64),
            ResidualLayer(64),
            DownSampleLayer(64, 128),
            ResidualLayer(128),
            ResidualLayer(128),

            DownSampleLayer(128, 256),
            ResidualLayer(256),
            ResidualLayer(256),
            ResidualLayer(256),
            ResidualLayer(256),
            ResidualLayer(256),
            ResidualLayer(256),
            ResidualLayer(256),
            ResidualLayer(256)
        )

        self.trunk_26 = torch.nn.Sequential(
            DownSampleLayer(256, 512),
            ResidualLayer(512),
            ResidualLayer(512),
            ResidualLayer(512),
            ResidualLayer(512),
            ResidualLayer(512),
            ResidualLayer(512),
            ResidualLayer(512),
            ResidualLayer(512)
        )

        self.trunk_13 = torch.nn.Sequential(
            DownSampleLayer(512, 1024),
            ResidualLayer(1024),
            ResidualLayer(1024),
            ResidualLayer(1024),
            ResidualLayer(1024)
        )

        self.con_13 = torch.nn.Sequential(
            ConvolutionSet(1024, 512)
        )

        self.detection_13 = torch.nn.Sequential(
            ConvolutionLayer(512, 1024, 3, 1, 1),
            ConvolutionLayer(1024, 45, 1, 1, 0)
        )

        self.up_26 = torch.nn.Sequential(
            ConvolutionLayer(512, 256, 1, 1, 0),
            UpSampleLayer()
        )

        self.con_26 = torch.nn.Sequential(
            ConvolutionSet(768, 256)
        )

        self.detection_26 = torch.nn.Sequential(
            ConvolutionLayer(256, 512, 3, 1, 1),
            ConvolutionLayer(512, 45, 1, 1, 0)
        )

        self.up_52 = torch.nn.Sequential(
            ConvolutionLayer(256, 128, 1, 1, 0),
            UpSampleLayer()
        )

        self.con_52 = torch.nn.Sequential(
            ConvolutionSet(384, 128),
        )

        self.detection_52 = torch.nn.Sequential(
            ConvolutionLayer(128, 256, 3, 1, 1),
            ConvolutionLayer(256, 45, 1, 1, 0)
        )

    def forward(self, x):
        h_52 = self.trunk_52(x)
        h_26 = self.trunk_26(h_52)
        h_13 = self.trunk_13(h_26)

        con_13 = self.con_13(h_13)
        detection_13 = self.detection_13(con_13)

        up_26 = self.up_26(con_13)
        cat_26 = torch.cat((up_26, h_26), dim=1)
        con_26 = self.con_26(cat_26)
        detection_26 = self.detection_26(con_26)

        up_52 = self.up_52(con_26)
        cat_52 = torch.cat((up_52, h_52), dim=1)
        con_52 = self.con_52(cat_52)
        detection_52 = self.detection_52(con_52)

        return detection_13, detection_26, detection_52


if __name__ == '__main__':
    trunk = Net()

    x = torch.randn([2, 3, 416, 416], dtype=torch.float32)

    y_13, y_26, y_52 = trunk(x)

    print(y_13.shape)
    print(y_26.shape)
    print(y_52.shape)

    pass
