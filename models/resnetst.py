import torch
import torch.nn as nn
from mmcls.models.backbones import ResNet
from mmcls.models.builder import BACKBONES

from .swin_transformer import SwinTransformer


@BACKBONES.register_module()
class ResNetST(SwinTransformer):
    def __init__(self, *args, **kwargs):
        super(ResNetST, self).__init__(out_indices=(2, ), *args, **kwargs)

        self.resnet = ResNet(depth=50, in_channels=3)
        self.alignment = nn.Sequential(
            nn.Conv2d(512, 192, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(192), nn.ReLU())

        self.absolute_pos_embed = nn.Parameter(torch.zeros(1, 28 * 28, 192))

    def forward(self, x):
        # use resnet to extract the cnn feature first
        x = self.resnet.conv1(x)
        x = self.resnet.norm1(x)
        x = self.resnet.relu(x)
        x = self.resnet.maxpool(x)

        featrue = None
        for i, layer_name in enumerate(self.resnet.res_layers):
            res_layer = getattr(self.resnet, layer_name)
            x = res_layer(x)
            if i == 1:
                featrue = x
        resnet_final_feature = x
        resnet_mid_feature = featrue
        x = resnet_mid_feature
        x = self.alignment(x)
        x = x.permute((0, 2, 1, 3)).permute((0, 1, 3, 2))
        B, H, W, C = x.shape
        x = x.view((B, H * W, C))

        # add the absolute position embed
        x = x + self.absolute_pos_embed

        outs = []
        for i, stage in enumerate(self.stages):
            if i == 1 or i == 2:
                x = stage(x)
                if i in self.out_indices:
                    norm_layer = getattr(self, f'norm{i}')
                    out = norm_layer(x)
                    out = out.view(-1, *stage.out_resolution,
                                   stage.out_channels).permute(0, 3, 1,
                                                               2).contiguous()
                    outs.append(out)
        x = outs[-1]
        return x
