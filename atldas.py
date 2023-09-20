import math
from functools import partial

import torch
import torch.nn as nn
import torch.nn.functional as F

from timm.data import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from timm.layers import SelectAdaptivePool2d, Linear, make_divisible
from ._builder import build_model_with_cfg
from ._efficientnet_blocks import SqueezeExcite, ConvBnAct
from ._manipulate import checkpoint_seq
from ._registry import register_model, generate_default_cfgs

__all__ = ['ATLDAS']


_SE_LAYER = partial(SqueezeExcite, gate_layer='hard_sigmoid', rd_round_fn=partial(make_divisible, divisor=4))


def separable_conv(inp, oup):
    return nn.Sequential(
        nn.Conv2d(inp, inp, 3, 1, 1, groups=inp, bias=False),
        nn.BatchNorm2d(inp),
        nn.ReLU(),
        nn.Conv2d(inp, oup, 1, 1, 0, bias=False),
        nn.BatchNorm2d(oup),
    )


def stem(inp, oup, stride):
    return nn.Sequential(
        nn.Conv2d(inp, oup, 3, stride, 1, bias=False),
        nn.BatchNorm2d(oup),
        nn.ReLU()
    )

class InvertedResidual(nn.Module):
    def __init__(self, inp, oup, kernel_size, stride, expand_ratio,train=False):
        super(InvertedResidual, self).__init__()
        assert stride in [1, 2]
        self.stride = stride
        padding = kernel_size // 2
        hidden_dim = round(inp * expand_ratio)
        self.use_res_connect = self.stride == 1 and inp == oup
        self.conv1 = nn.Conv2d(inp, hidden_dim, 1, 1, 0, bias=False)
        self.bn1 = nn.BatchNorm2d(hidden_dim,affine=train)
        self.conv2 = nn.Conv2d(hidden_dim, hidden_dim, kernel_size, stride, padding, groups=hidden_dim, bias=False)
        self.bn2 = nn.BatchNorm2d(hidden_dim,affine=train)
        self.conv3 = nn.Conv2d(hidden_dim, oup, 1, 1, 0, bias=False)
        self.bn3 = nn.BatchNorm2d(oup,affine=train)

    def forward(self, x):
        inputs = x
        x = self.conv1(x)
        x = self.bn1(x)
        x = F.relu(x, inplace=True)
        x = self.conv2(x)
        x = self.bn2(x)
        x = F.relu(x, inplace=True)
        x = self.conv3(x)
        x = self.bn3(x)
        if self.use_res_connect:
            return inputs + x
        else:
            return x


class ATLDAS(nn.Module):
    def __init__(
            self,
            num_classes=1000,
            in_chans=3,
            output_stride=32,
            global_pool='avg',
            drop_rate=0.2,
    ):
        super(ATLDAS, self).__init__()
        # setting of inverted residual blocks
        assert output_stride == 32, 'only output_stride==32 is valid, dilation not supported'
        self.cfgs = []
        self.num_classes=num_classes
        choice=[4, 4, 5, 4, 1, 5, 5, 4, 5, 2, 4, 4, 2, 5, 5, 5, 5, 5, 5]
        mb_config = [
            [3, 32, 7, 2,0,0],
            [3, 32, 3, 1,0,0],
            [3, 40, 7, 2,1,0],
            [6, 40, 3, 1,1,0],
            [6, 40, 7, 1,1,0],
            [3, 40, 3, 1,0,1],
            [3, 80, 3, 2,0,1],
            [6, 80, 7, 1,0,1],
            [6, 80, 7, 1,0,1],
            [3, 80, 5, 1,0,1],
            [6, 96, 3, 1,0,1],
            [3, 96, 5, 1,0,1],
            [3, 96, 5, 1,0,1],
            [6, 96, 3, 1,0,1],
            [6, 192, 3, 2, 1, 1 ],
            [6, 192, 7, 1, 1, 1 ],
            [6, 192, 3, 1, 1, 1 ],
            [6, 192, 7, 1, 1, 1 ],
            [6, 320, 5, 1, 1, 1 ],
        ]
        for id,x in enumerate(choice):
            mb_config[id][0]=(x//3)*3+3
            mb_config[id][2]=x%3*2+3
        input_channel = 16
        last_channel = 1280
        self.drop_rate = 0.2
        self.last_channel = last_channel
        self.stem = stem(3, 32, 2)
        self.separable_conv = separable_conv(32, 16)
        self.mb_module = list()

        for t, c, k, s,se,hs in mb_config:
            output_channel = c
            self.mb_module.append(InvertedResidual(input_channel,output_channel, k, s,t,True))
            input_channel = output_channel
        self.mb_module = nn.Sequential(*self.mb_module)
        self.global_pool = SelectAdaptivePool2d(pool_type=global_pool)
        self.conv_head = nn.Conv2d(320, last_channel, 1, 1, 0, bias=True)
        self.act2 = nn.ReLU(inplace=True)
        self.flatten = nn.Flatten(1) if global_pool else nn.Identity()
        self.classifier = Linear(last_channel, num_classes) if num_classes > 0 else nn.Identity()

        # FIXME init

    @torch.jit.ignore
    def group_matcher(self, coarse=False):
        matcher = dict(
            stem=r'^conv_stem|bn1',
            blocks=[
                (r'^blocks\.(\d+)' if coarse else r'^blocks\.(\d+)\.(\d+)', None),
                (r'conv_head', (99999,))
            ]
        )
        return matcher

    @torch.jit.ignore
    def set_grad_checkpointing(self, enable=True):
        self.grad_checkpointing = enable

    @torch.jit.ignore
    def get_classifier(self):
        return self.classifier

    def reset_classifier(self, num_classes, global_pool='avg'):
        self.num_classes = num_classes
        # cannot meaningfully change pooling of efficient head after creation
        self.global_pool = SelectAdaptivePool2d(pool_type=global_pool)
        self.flatten = nn.Flatten(1) if global_pool else nn.Identity()  # don't flatten if pooling disabled
        self.classifier = Linear(self.pool_dim, num_classes) if num_classes > 0 else nn.Identity()

    def forward_features(self, x):
        x = self.stem(x)
        x = self.separable_conv(x)
        x = self.mb_module(x)

        return x

    def forward_head(self, x):
        x = self.global_pool(x)
        x = self.conv_head(x)
        x = self.act2(x)
        x = self.flatten(x)
        if self.drop_rate > 0.:
            x = F.dropout(x, p=self.drop_rate, training=self.training)
        x = self.classifier(x)
        return x

    def forward(self, x):
        x = self.forward_features(x)
        x = self.forward_head(x)
        return x



def _create_atldas(variant='atldas',pretrained=False, **kwargs):
    variant='atldas'

    return build_model_with_cfg(
        ATLDAS,
        variant,
        pretrained,
        **kwargs,
    )


def _cfg(url='', **kwargs):
    return {
        'url': url, 'num_classes': 1000, 'input_size': (3, 224, 224), 'pool_size': (7, 7),
        'crop_pct': 0.875, 'interpolation': 'bicubic',
        'mean': IMAGENET_DEFAULT_MEAN, 'std': IMAGENET_DEFAULT_STD,
        'first_conv': 'conv_stem', 'classifier': 'classifier',
        **kwargs
    }


default_cfgs = generate_default_cfgs({
    'atldas.untrained': _cfg(),
    'atldas.atldas': _cfg(),
})


@register_model
def atldas(pretrained=False, **kwargs) -> ATLDAS:
    model = _create_atldas('atldas',  pretrained=pretrained, **kwargs)
    return model




