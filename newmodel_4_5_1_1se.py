from collections import OrderedDict
from functools import partial
from typing import Callable, Optional
import torch.nn as nn
import torch
from torch import Tensor


def drop_path(x, drop_prob: float = 0., training: bool = False):
    """
    Drop paths (Stochastic Depth) per sample (when applied in main path of residual blocks).
    "Deep Networks with Stochastic Depth", https://arxiv.org/pdf/1603.09382.pdf

    This function is taken from the rwightman.
    It can be seen here:
    https://github.com/rwightman/pytorch-image-models/blob/master/timm/models/layers/drop.py#L140
    """
    if drop_prob == 0. or not training:
        return x
    keep_prob = 1 - drop_prob
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)  # work with diff dim tensors, not just 2D ConvNets
    random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
    random_tensor.floor_()  # binarize
    output = x.div(keep_prob) * random_tensor
    return output


class DropPath(nn.Module):
    """
    Drop paths (Stochastic Depth) per sample  (when applied in main path of residual blocks).
    "Deep Networks with Stochastic Depth", https://arxiv.org/pdf/1603.09382.pdf
    """
    def __init__(self, drop_prob=None):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        return drop_path(x, self.drop_prob, self.training)


class ConvBNAct(nn.Module):
    def __init__(self,
                 in_c: int,
                 out_c: int,
                 kernel_size ,
                 stride: int = 1,
                 groups: int = 1,
                 norm_layer: Optional[Callable[..., nn.Module]] = None,
                 activation_layer: Optional[Callable[..., nn.Module]] = None):
        super(ConvBNAct, self).__init__()

        if kernel_size == 1:
            padding = 0
        elif kernel_size == (1, 5):
            padding = (0, 2)
        elif kernel_size == (5, 1):
            padding = (2, 0)
        elif kernel_size == 3:
            padding = 1


        # padding = (kernel_size - 1) // 2
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        if activation_layer is None:
            activation_layer = nn.GELU  # alias Swish  (torch>=1.7)

        self.conv = nn.Conv2d(in_channels=in_c,
                              out_channels=out_c,
                              kernel_size=kernel_size,
                              stride=stride,
                              padding=padding,
                              groups=groups,
                              bias=False)

        self.bn = norm_layer(out_c)
        self.act = activation_layer()

    def forward(self, x):
        result = self.conv(x)
        result = self.bn(result)
        result = self.act(result)

        return result


class ConvBNAct1(nn.Module):
    def __init__(self,
                 in_c: int,
                 out_c: int,
                 kernel_size,
                 stride: int = 1,
                 groups: int = 1,
                 norm_layer: Optional[Callable[..., nn.Module]] = None,
                 activation_layer: Optional[Callable[..., nn.Module]] = None):
        super(ConvBNAct1, self).__init__()

        if kernel_size == 1:
            padding = 0
        elif kernel_size == (1, 5):
            padding = (0, 2)
        elif kernel_size == (5, 1):
            padding = (2, 0)

        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        if activation_layer is None:
            activation_layer = nn.GELU
        self.conv = nn.Conv2d(in_channels=in_c,
                              out_channels=out_c,
                              kernel_size=kernel_size,
                              stride=stride,
                              padding=padding,
                              groups=groups,
                              bias=False)

        self.bn = norm_layer(out_c)
        self.act = activation_layer()

    def forward(self, x):
        result = self.conv(x)
        result = self.bn(result)
        result = self.act(result)
        return result

class SqueezeExcite(nn.Module):
    def __init__(self,
                 input_c: int,   # block input channel
                 expand_c: int,  # block expand channel
                 se_ratio: float = 0.25):
        super(SqueezeExcite, self).__init__()
        squeeze_c = int(input_c * se_ratio)
        self.conv_reduce = nn.Conv2d(expand_c, squeeze_c, 1)
        self.act1 = nn.GELU()  # alias Swish
        self.conv_expand = nn.Conv2d(squeeze_c, expand_c, 1)
        self.act2 = nn.Sigmoid()

    def forward(self, x: Tensor) -> Tensor:
        scale = x.mean((2, 3), keepdim=True)
        scale = self.conv_reduce(scale)
        scale = self.act1(scale)
        scale = self.conv_expand(scale)
        scale = self.act2(scale)
        return scale * x


class MBConv(nn.Module):
    def __init__(self,
                 kernel_size ,
                 input_c: int,
                 out_c: int,
                 expand_ratio: int,
                 stride: int,
                 se_ratio: float,
                 drop_rate: float,
                 group_width: int ,              # 这里改动
                 norm_layer: Callable[..., nn.Module]):
        super(MBConv, self).__init__()

        if stride not in [1, 2]:
            raise ValueError("illegal stride value.")

        self.has_shortcut = (stride == 1 and input_c == out_c)

        activation_layer = nn.GELU  # alias Swish
        expanded_c = input_c * expand_ratio

        # 在EfficientNetV2中，MBConv中不存在expansion=1的情况所以conv_pw肯定存在
        assert expand_ratio != 1
        # Point-wise expansion


        # Depth-wise convolution
        self.dwconv = ConvBNAct(input_c,
                                input_c,
                                kernel_size=kernel_size,
                                stride=stride,
                                groups=input_c ,        #  expanded_c = input_c * expand_ratio
                                norm_layer=norm_layer,
                                activation_layer=activation_layer)
        self.bwconv = ConvBNAct(input_c,
                                input_c,
                                kernel_size=(5, 1),
                                stride=stride,
                                groups=input_c,        #  expanded_c = input_c * expand_ratio
                                norm_layer=norm_layer,
                                activation_layer=activation_layer)
        self.conv = ConvBNAct(input_c,
                              input_c,
                              kernel_size =1,
                              stride= stride,
                              groups= input_c,
                              norm_layer=norm_layer,
                              activation_layer=activation_layer)

        self.se = SqueezeExcite(input_c, expanded_c, se_ratio) if se_ratio > 0 else nn.Identity()

        self.expand_conv = ConvBNAct(input_c,
                                     expanded_c,
                                     kernel_size=1,
                                     norm_layer=norm_layer,
                                     activation_layer=activation_layer)

        # Point-wise linear projection
        self.project_conv = ConvBNAct(expanded_c,
                                      out_c,
                                      kernel_size=1,

                                      norm_layer=norm_layer,
                                      activation_layer=nn.Identity)  # 注意这里没有激活函数，所有传入Identity

        self.out_channels = out_c

        # 只有在使用shortcut连接时才使用dropout层
        self.drop_rate = drop_rate
        if self.has_shortcut and drop_rate > 0:
            self.dropout = DropPath(drop_rate)

    def forward(self, x: Tensor) -> Tensor:

        result_1 = self.dwconv(x)
        result_2 = self.bwconv(x)
        result_3 = self.conv(x)
        result = result_2 + result_1 +result_3

        result = self.expand_conv(result)
        result = self.se(result)
        result = self.project_conv(result)

        if self.has_shortcut:
            if self.drop_rate > 0:
                result = self.dropout(result)
            result += x

        return result


class FusedMBConv(nn.Module):
    def __init__(self,
                 kernel_size,
                 input_c: int,
                 out_c: int,
                 expand_ratio: int,
                 stride: int,
                 group_width:int,
                 se_ratio: float,
                 drop_rate: float,
                 norm_layer: Callable[..., nn.Module]):
        super(FusedMBConv, self).__init__()

        assert stride in [1, 2]
        # assert se_ratio == 0

        self.has_shortcut = stride == 1 and input_c == out_c
        self.drop_rate = drop_rate

        self.has_expansion = expand_ratio != 1

        activation_layer = nn.GELU  # alias Swish
        expanded_c = input_c * expand_ratio

        # 只有当expand ratio不等于1时才有expand conv
        if self.has_expansion:
            # Expansion convolution
            self.project_aconv = ConvBNAct(input_c,
                                         input_c,
                                         kernel_size=kernel_size,

                                         stride=stride,
                                         norm_layer=norm_layer,
                                         activation_layer=activation_layer)

            self.project_bconv = ConvBNAct(input_c,
                                         input_c,
                                         kernel_size=(5, 1),
                                         stride=stride,
                                         norm_layer=norm_layer,
                                         activation_layer=activation_layer)

            self.conv_a = ConvBNAct(input_c,
                                  input_c,
                                  kernel_size=1,
                                  stride=stride,
                                  norm_layer=norm_layer,
                                  activation_layer=activation_layer)

            self.se = SqueezeExcite(input_c, input_c, se_ratio) if se_ratio > 0 else nn.Identity()

            self.expand_aconv = ConvBNAct(input_c,
                                          expanded_c,
                                          kernel_size=1,

                                          norm_layer=norm_layer,
                                          activation_layer=nn.Identity)  # 注意没有激活函数

            self.expand_bconv = ConvBNAct(expanded_c,
                                          out_c,
                                          kernel_size=1,

                                          norm_layer=norm_layer,
                                          activation_layer=nn.Identity)  # 注意没有激活函数


        else:
            # 当只有project_conv时的情况
            self.project_cconv = ConvBNAct(input_c,
                                          out_c,
                                          kernel_size=kernel_size,
                                          stride=stride,
                                          norm_layer=norm_layer,

                                          activation_layer=activation_layer)  # 注意有激活函数

            self.project_dconv = ConvBNAct(input_c,
                                          out_c,
                                          kernel_size=(5, 1),
                                          stride=stride,
                                          norm_layer=norm_layer,

                                          activation_layer=activation_layer)  # 注意有激活函数

            self.conv_b = ConvBNAct(input_c,
                                    out_c,
                                    kernel_size=1,
                                    stride=stride,
                                    norm_layer=norm_layer,
                                    activation_layer=activation_layer)



        self.out_channels = out_c

        # 只有在使用shortcut连接时才使用dropout层
        self.drop_rate = drop_rate
        if self.has_shortcut and drop_rate > 0:
            self.dropout = DropPath(drop_rate)

    def forward(self, x: Tensor) -> Tensor:
        if self.has_expansion:
            result_1 = self.project_aconv(x)
            result_2 = self.project_bconv(x)
            result_3 = self.conv_a(x)
            result = result_2 + result_1+result_3
            result = self.se(result)
            result = self.expand_aconv(result)
            result = self.expand_bconv(result)
        else:
            result_1 = self.project_cconv(x)
            result_2 = self.project_dconv(x)
            result_3 = self.conv_b
            result = result_2 + result_1+result_3
            result = self.se(result)   #??

        if self.has_shortcut:
            if self.drop_rate > 0:
                result = self.dropout(result)

            result += x

        return result


class EfficientNetV2(nn.Module):
    def __init__(self,
                 model_cnf: list,
                 num_classes: int = 1000,
                 num_features: int = 1280,
                 dropout_rate: float = 0.2,
                 drop_connect_rate: float = 0.2,
                 zero_init_last_bn : bool= True):
        super(EfficientNetV2, self).__init__()

        for cnf in model_cnf:
            assert len(cnf) == 9

        norm_layer = partial(nn.BatchNorm2d, eps=1e-3, momentum=0.1)

        stem_filter_num = model_cnf[0][4]

        self.stem = ConvBNAct(1,
                              stem_filter_num,
                              kernel_size=3,

                              stride=2,
                              norm_layer=norm_layer)  # 激活函数默认是SiLU

        total_blocks = sum([i[0] for i in model_cnf])
        block_id = 0
        blocks = []
        for cnf in model_cnf:
            repeats = cnf[0]
            if cnf[-2] == 1:
                op = MBConv

            elif cnf[-2] == 2:
                op = FusedMBConv

            #op = Bottleneck if cnf[-2] == 0 else MBConv
            for i in range(repeats):
                blocks.append(op(kernel_size=cnf[1],
                                 input_c=cnf[4] if i == 0 else cnf[5],
                                 out_c=cnf[5],
                                 group_width=cnf[-3],
                                 expand_ratio=cnf[3],  # 这里没有bottleneck卷积
                                 stride=cnf[2] if i == 0 else 1,
                                 se_ratio=cnf[-1],
                                 drop_rate=drop_connect_rate * block_id / total_blocks,  # efficientv2
                                 norm_layer=norm_layer)) #这里B没有
                                                   #这里B没有
                block_id += 1
        self.blocks = nn.Sequential(*blocks)

        head_input_c = model_cnf[-1][-4]
        head = OrderedDict()

        # head.update({"project_conv": ConvBNAct(head_input_c,
        #                                        1280,
        #                                        kernel_size=1,
        #                                        norm_layer=norm_layer)})  # 激活函数默认是SiLU
        head.update({"layerNorm": nn.BatchNorm2d(head_input_c, eps=1e-3)})
        head.update({"avgpool": nn.AdaptiveAvgPool2d(1)})
        head.update({"flatten": nn.Flatten()})
        if dropout_rate > 0:
            head.update({"dropout": nn.Dropout(p=dropout_rate, inplace=True)})
        
        head.update({"classifier": nn.Linear(head_input_c, num_classes)})

        self.head = nn.Sequential(head)

        # initial weights
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.zeros_(m.bias)
        if zero_init_last_bn:      # 直接添加上去的不知道啥意思 bottleneck
            for m in self.modules():
                if hasattr(m, "zero_init_last_bn"):
                    m.zero_init_last_bn()
    def forward(self, x: Tensor) -> Tensor:
        x = self.stem(x)
        x = self.blocks(x)
        x = self.head(x)

        return x


def efficientnetv2_s(num_classes: int = 1000):
    """
    EfficientNetV2
    https://arxiv.org/abs/2104.00298
    """
    # train_size: 300, eval_size: 384

    # repeat, kernel, stride, expansion, in_c, out_c,group_width ,operator, se_ratio
    model_config = [
                                                     #注意在fusedMBC 这里的group_width 为了方便传入直接等于了groups，仅实用FMBC
                    [3, (1,5), 2, 4, 24, 48, 1, 2, 0],   # 一定要注意expansion
                    [3, (1,5), 2, 4, 48, 64, 1, 2, 0],
                    [9, (1,5), 2, 4, 64, 128, 1, 1, 0.25],
                    [3, (1,5), 2, 4, 128, 256, 1, 1, 0.25]]

    model = EfficientNetV2(model_cnf=model_config,
                           num_classes=num_classes,
                           dropout_rate=0.1)
    return model


def efficientnetv2_m(num_classes: int = 1000):
    """
    EfficientNetV2
    https://arxiv.org/abs/2104.00298
    """
    # train_size: 384, eval_size: 480

    # repeat, kernel, stride, expansion, in_c, out_c, operator, se_ratio
    model_config = [[3, 3, 1, 1, 24, 24, 0, 0],
                    [5, 3, 2, 4, 24, 48, 0, 0],
                    [5, 3, 2, 4, 48, 80, 0, 0],
                    [7, 3, 2, 4, 80, 160, 1, 0.25],
                    [14, 3, 1, 6, 160, 176, 1, 0.25],
                    [18, 3, 2, 6, 176, 240, 1, 0.25],
                    [5, 3, 1, 6, 240, 256, 1, 0.25]]

    model = EfficientNetV2(model_cnf=model_config,
                           num_classes=num_classes,
                           dropout_rate=0.3)
    return model


def efficientnetv2_l(num_classes: int = 1000):
    """
    EfficientNetV2
    https://arxiv.org/abs/2104.00298
    """
    # train_size: 384, eval_size: 480

    # repeat, kernel, stride, expansion, in_c, out_c, operator, se_ratio
    model_config = [[4, 3, 1, 1, 32, 32, 0, 0],
                    [7, 3, 2, 4, 32, 64, 0, 0],
                    [7, 3, 2, 4, 64, 96, 0, 0],
                    [10, 3, 2, 4, 96, 192, 1, 0.25],
                    [19, 3, 1, 6, 192, 224, 1, 0.25],
                    [25, 3, 2, 6, 224, 384, 1, 0.25],
                    [7, 3, 1, 6, 384, 640, 1, 0.25]]

    model = EfficientNetV2(model_cnf=model_config,
                           num_classes=num_classes,
                           dropout_rate=0.4)
    return model
