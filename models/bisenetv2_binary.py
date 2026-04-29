# bisenetv2_binary.py

import torch
import torch.nn as nn
import torch.nn.functional as F


def upsample(x, size):
    return F.interpolate(x, size=size, mode="bilinear", align_corners=False)


class ConvBNReLU(nn.Module):
    def __init__(self, in_ch, out_ch, k=3, s=1, p=1):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=k, stride=s, padding=p, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.block(x)


class DWConvBNReLU(nn.Module):
    def __init__(self, in_ch, out_ch, k=3, s=1, p=1):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, in_ch, kernel_size=k, stride=s, padding=p, groups=in_ch, bias=False),
            nn.BatchNorm2d(in_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_ch, out_ch, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.block(x)


class StemBlock(nn.Module):
    def __init__(self, in_ch=3, out_ch=16):
        super().__init__()
        self.conv1 = ConvBNReLU(in_ch, out_ch, k=3, s=2, p=1)
        self.left = nn.Sequential(
            ConvBNReLU(out_ch, out_ch // 2, k=1, s=1, p=0),
            ConvBNReLU(out_ch // 2, out_ch, k=3, s=2, p=1),
        )
        self.right = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.fuse = ConvBNReLU(out_ch * 2, out_ch, k=3, s=1, p=1)

    def forward(self, x):
        x = self.conv1(x)
        x_left = self.left(x)
        x_right = self.right(x)
        x = torch.cat([x_left, x_right], dim=1)
        x = self.fuse(x)
        return x


class GELayer(nn.Module):
    def __init__(self, in_ch, out_ch, stride=1, exp_ratio=6):
        super().__init__()
        mid_ch = in_ch * exp_ratio

        self.conv1 = ConvBNReLU(in_ch, in_ch, k=3, s=1, p=1)

        self.dwconv = nn.Sequential(
            nn.Conv2d(in_ch, mid_ch, kernel_size=3, stride=stride, padding=1,
                      groups=in_ch, bias=False),
            nn.BatchNorm2d(mid_ch),
            nn.ReLU(inplace=True),
        )

        self.project = nn.Sequential(
            nn.Conv2d(mid_ch, out_ch, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_ch),
        )

        if stride == 2 or in_ch != out_ch:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_ch, in_ch, kernel_size=3, stride=stride, padding=1,
                          groups=in_ch, bias=False),
                nn.BatchNorm2d(in_ch),
                nn.Conv2d(in_ch, out_ch, kernel_size=1, bias=False),
                nn.BatchNorm2d(out_ch),
            )
        else:
            self.shortcut = nn.Identity()

        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        identity = self.shortcut(x)
        out = self.conv1(x)
        out = self.dwconv(out)
        out = self.project(out)
        out = out + identity
        out = self.relu(out)
        return out


class CEBlock(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.bn = nn.BatchNorm2d(in_ch)
        self.conv_gap = ConvBNReLU(in_ch, out_ch, k=1, s=1, p=0)
        self.conv_last = ConvBNReLU(out_ch, out_ch, k=3, s=1, p=1)

    def forward(self, x):
        feat = torch.mean(x, dim=(2, 3), keepdim=True)
        feat = self.bn(feat)
        feat = self.conv_gap(feat)
        feat = feat + x
        feat = self.conv_last(feat)
        return feat


class DetailBranch(nn.Module):
    def __init__(self):
        super().__init__()
        self.s1 = nn.Sequential(
            ConvBNReLU(3, 64, 3, 2, 1),
            ConvBNReLU(64, 64, 3, 1, 1),
        )
        self.s2 = nn.Sequential(
            ConvBNReLU(64, 64, 3, 2, 1),
            ConvBNReLU(64, 64, 3, 1, 1),
            ConvBNReLU(64, 64, 3, 1, 1),
        )
        self.s3 = nn.Sequential(
            ConvBNReLU(64, 128, 3, 2, 1),
            ConvBNReLU(128, 128, 3, 1, 1),
            ConvBNReLU(128, 128, 3, 1, 1),
        )

    def forward(self, x):
        x = self.s1(x)  # 1/2
        x = self.s2(x)  # 1/4
        x = self.s3(x)  # 1/8
        return x


class SemanticBranch(nn.Module):
    def __init__(self):
        super().__init__()
        self.stem = StemBlock(3, 16)        # 1/4
        self.s3 = nn.Sequential(            # 1/8
            GELayer(16, 32, stride=2),
            GELayer(32, 32, stride=1),
        )
        self.s4 = nn.Sequential(            # 1/16
            GELayer(32, 64, stride=2),
            GELayer(64, 64, stride=1),
        )
        self.s5 = nn.Sequential(            # 1/32
            GELayer(64, 128, stride=2),
            GELayer(128, 128, stride=1),
            GELayer(128, 128, stride=1),
            GELayer(128, 128, stride=1),
        )
        self.ce = CEBlock(128, 128)

    def forward(self, x):
        x = self.stem(x)
        x = self.s3(x)
        x = self.s4(x)
        x = self.s5(x)
        x = self.ce(x)
        return x


class BGALayer(nn.Module):
    def __init__(self, detail_ch=128, semantic_ch=128, out_ch=128):
        super().__init__()

        self.detail_dw = nn.Sequential(
            nn.Conv2d(detail_ch, detail_ch, 3, 1, 1, groups=detail_ch, bias=False),
            nn.BatchNorm2d(detail_ch),
            nn.Conv2d(detail_ch, out_ch, 1, bias=False),
            nn.BatchNorm2d(out_ch),
        )

        self.detail_down = nn.Sequential(
            nn.Conv2d(detail_ch, detail_ch, 3, 2, 1, groups=detail_ch, bias=False),
            nn.BatchNorm2d(detail_ch),
            nn.Conv2d(detail_ch, out_ch, 1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.AvgPool2d(kernel_size=3, stride=2, padding=1),
        )

        self.semantic_conv = nn.Sequential(
            nn.Conv2d(semantic_ch, out_ch, 3, 1, 1, bias=False),
            nn.BatchNorm2d(out_ch),
        )

        self.semantic_dw = nn.Sequential(
            nn.Conv2d(semantic_ch, semantic_ch, 3, 1, 1, groups=semantic_ch, bias=False),
            nn.BatchNorm2d(semantic_ch),
            nn.Conv2d(semantic_ch, out_ch, 1, bias=False),
            nn.BatchNorm2d(out_ch),
        )

        self.fuse = nn.Sequential(
            ConvBNReLU(out_ch, out_ch, 3, 1, 1),
        )

    def forward(self, detail, semantic):
        semantic_up = upsample(semantic, detail.shape[2:])
        semantic_up_conv = self.semantic_conv(semantic_up)
        detail_feat = self.detail_dw(detail)
        left = detail_feat * torch.sigmoid(semantic_up_conv)

        detail_down = self.detail_down(detail)
        semantic_feat = self.semantic_dw(semantic)
        right = detail_down * torch.sigmoid(semantic_feat)
        right = upsample(right, detail.shape[2:])

        out = left + right
        out = self.fuse(out)
        return out


class SegHead(nn.Module):
    def __init__(self, in_ch, mid_ch, num_classes=1):
        super().__init__()
        self.block = nn.Sequential(
            ConvBNReLU(in_ch, mid_ch, 3, 1, 1),
            nn.Dropout2d(0.1),
            nn.Conv2d(mid_ch, num_classes, kernel_size=1, bias=True),
        )

    def forward(self, x):
        return self.block(x)


class BiSeNetV2Binary(nn.Module):
    def __init__(self, num_classes=1):
        super().__init__()
        self.detail = DetailBranch()
        self.semantic = SemanticBranch()
        self.bga = BGALayer(128, 128, 128)
        self.head = SegHead(128, 128, num_classes=num_classes)

    def forward(self, x):
        h, w = x.shape[2:]
        feat_d = self.detail(x)
        feat_s = self.semantic(x)
        feat = self.bga(feat_d, feat_s)
        logits = self.head(feat)
        logits = upsample(logits, (h, w))
        return logits


def build_model(num_classes=1):
    return BiSeNetV2Binary(num_classes=num_classes)