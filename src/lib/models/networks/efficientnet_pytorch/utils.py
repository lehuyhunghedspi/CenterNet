"""
This file contains helper functions for building the model and for loading model parameters.
These helper functions are built to mirror those in the official TensorFlow implementation.
"""

import re
import math
import collections
from functools import partial
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils import model_zoo

########################################################################
############### HELPERS FUNCTIONS FOR MODEL ARCHITECTURE ###############
########################################################################


# Parameters for the entire model (stem, all blocks, and head)
GlobalParams = collections.namedtuple('GlobalParams', [
    'batch_norm_momentum', 'batch_norm_epsilon', 'dropout_rate',
    'num_classes', 'width_coefficient', 'depth_coefficient',
    'depth_divisor', 'min_depth', 'drop_connect_rate', 'image_size'])

# Parameters for an individual model block
BlockArgs = collections.namedtuple('BlockArgs', [
    'kernel_size', 'num_repeat', 'input_filters', 'output_filters',
    'expand_ratio', 'id_skip', 'stride', 'se_ratio'])

# Change namedtuple defaults
GlobalParams.__new__.__defaults__ = (None,) * len(GlobalParams._fields)
BlockArgs.__new__.__defaults__ = (None,) * len(BlockArgs._fields)


class SwishImplementation(torch.autograd.Function):
    @staticmethod
    def forward(ctx, i):
        result = i * torch.sigmoid(i)
        ctx.save_for_backward(i)
        return result

    @staticmethod
    def backward(ctx, grad_output):
        i = ctx.saved_variables[0]
        sigmoid_i = torch.sigmoid(i)
        return grad_output * (sigmoid_i * (1 + i * (1 - sigmoid_i)))


class MemoryEfficientSwish(nn.Module):
    def forward(self, x):
        return SwishImplementation.apply(x)

class Swish(nn.Module):
    def forward(self, x):
        return x * torch.sigmoid(x)


def round_filters(filters, global_params):
    """ Calculate and round number of filters based on depth multiplier. """
    multiplier = global_params.width_coefficient
    if not multiplier:
        return filters
    divisor = global_params.depth_divisor
    min_depth = global_params.min_depth
    filters *= multiplier
    min_depth = min_depth or divisor
    new_filters = max(min_depth, int(filters + divisor / 2) // divisor * divisor)
    if new_filters < 0.9 * filters:  # prevent rounding by more than 10%
        new_filters += divisor
    return int(new_filters)


def round_repeats(repeats, global_params):
    """ Round number of filters based on depth multiplier. """
    multiplier = global_params.depth_coefficient
    if not multiplier:
        return repeats
    return int(math.ceil(multiplier * repeats))


def drop_connect(inputs, p, training):
    """ Drop connect. """
    if not training: return inputs
    batch_size = inputs.shape[0]
    keep_prob = 1 - p
    random_tensor = keep_prob
    random_tensor += torch.rand([batch_size, 1, 1, 1], dtype=inputs.dtype, device=inputs.device)
    binary_tensor = torch.floor(random_tensor)
    output = inputs / keep_prob * binary_tensor
    return output


def get_same_padding_conv2d(image_size=None):
    """ Chooses static padding if you have specified an image size, and dynamic padding otherwise.
        Static padding is necessary for ONNX exporting of models. """
    if image_size is None:
        return Conv2dDynamicSamePadding
    else:
        return partial(Conv2dStaticSamePadding, image_size=image_size)


class Conv2dDynamicSamePadding(nn.Conv2d):
    """ 2D Convolutions like TensorFlow, for a dynamic image size """

    def __init__(self, in_channels, out_channels, kernel_size, stride=1, dilation=1, groups=1, bias=True):
        super().__init__(in_channels, out_channels, kernel_size, stride, 0, dilation, groups, bias)
        self.stride = self.stride if len(self.stride) == 2 else [self.stride[0]] * 2

    def forward(self, x):
        ih, iw = x.size()[-2:]
        kh, kw = self.weight.size()[-2:]
        sh, sw = self.stride
        oh, ow = math.ceil(ih / sh), math.ceil(iw / sw)
        pad_h = max((oh - 1) * self.stride[0] + (kh - 1) * self.dilation[0] + 1 - ih, 0)
        pad_w = max((ow - 1) * self.stride[1] + (kw - 1) * self.dilation[1] + 1 - iw, 0)
        if pad_h > 0 or pad_w > 0:
            x = F.pad(x, [pad_w // 2, pad_w - pad_w // 2, pad_h // 2, pad_h - pad_h // 2])
        return F.conv2d(x, self.weight, self.bias, self.stride, self.padding, self.dilation, self.groups)


class Conv2dStaticSamePadding(nn.Conv2d):
    """ 2D Convolutions like TensorFlow, for a fixed image size"""

    def __init__(self, in_channels, out_channels, kernel_size, image_size=None, **kwargs):
        super().__init__(in_channels, out_channels, kernel_size, **kwargs)
        self.stride = self.stride if len(self.stride) == 2 else [self.stride[0]] * 2

        # Calculate padding based on image size and save it
        assert image_size is not None
        ih, iw = image_size if type(image_size) == list else [image_size, image_size]
        kh, kw = self.weight.size()[-2:]
        sh, sw = self.stride
        oh, ow = math.ceil(ih / sh), math.ceil(iw / sw)
        pad_h = max((oh - 1) * self.stride[0] + (kh - 1) * self.dilation[0] + 1 - ih, 0)
        pad_w = max((ow - 1) * self.stride[1] + (kw - 1) * self.dilation[1] + 1 - iw, 0)
        if pad_h > 0 or pad_w > 0:
            self.static_padding = nn.ZeroPad2d((pad_w // 2, pad_w - pad_w // 2, pad_h // 2, pad_h - pad_h // 2))
        else:
            self.static_padding = Identity()

    def forward(self, x):
        x = self.static_padding(x)
        x = F.conv2d(x, self.weight, self.bias, self.stride, self.padding, self.dilation, self.groups)
        return x


class Identity(nn.Module):
    def __init__(self, ):
        super(Identity, self).__init__()

    def forward(self, input):
        return input


########################################################################
############## HELPERS FUNCTIONS FOR LOADING MODEL PARAMS ##############
########################################################################


def efficientnet_params(model_name):
    """ Map EfficientNet model name to parameter coefficients. """
    params_dict = {
        # Coefficients:   width,depth,res,dropout
        'efficientnet-b0': (1.0, 1.0, 224, 0.2),
        'efficientnet-b1': (1.0, 1.1, 240, 0.2),
        'efficientnet-b2': (1.1, 1.2, 260, 0.3),
        'efficientnet-b3': (1.2, 1.4, 300, 0.3),
        'efficientnet-b4': (1.4, 1.8, 380, 0.4),
        'efficientnet-b5': (1.6, 2.2, 456, 0.4),
        'efficientnet-b6': (1.8, 2.6, 528, 0.5),
        'efficientnet-b7': (2.0, 3.1, 600, 0.5),
    }
    return params_dict[model_name]


class BlockDecoder(object):
    """ Block Decoder for readability, straight from the official TensorFlow repository """

    @staticmethod
    def _decode_block_string(block_string):
        """ Gets a block through a string notation of arguments. """
        assert isinstance(block_string, str)

        ops = block_string.split('_')
        options = {}
        for op in ops:
            splits = re.split(r'(\d.*)', op)
            if len(splits) >= 2:
                key, value = splits[:2]
                options[key] = value

        # Check stride
        assert (('s' in options and len(options['s']) == 1) or
                (len(options['s']) == 2 and options['s'][0] == options['s'][1]))

        return BlockArgs(
            kernel_size=int(options['k']),
            num_repeat=int(options['r']),
            input_filters=int(options['i']),
            output_filters=int(options['o']),
            expand_ratio=int(options['e']),
            id_skip=('noskip' not in block_string),
            se_ratio=float(options['se']) if 'se' in options else None,
            stride=[int(options['s'][0])])

    @staticmethod
    def _encode_block_string(block):
        """Encodes a block to a string."""
        args = [
            'r%d' % block.num_repeat,
            'k%d' % block.kernel_size,
            's%d%d' % (block.strides[0], block.strides[1]),
            'e%s' % block.expand_ratio,
            'i%d' % block.input_filters,
            'o%d' % block.output_filters
        ]
        if 0 < block.se_ratio <= 1:
            args.append('se%s' % block.se_ratio)
        if block.id_skip is False:
            args.append('noskip')
        return '_'.join(args)

    @staticmethod
    def decode(string_list):
        """
        Decodes a list of string notations to specify blocks inside the network.

        :param string_list: a list of strings, each string is a notation of block
        :return: a list of BlockArgs namedtuples of block args
        """
        assert isinstance(string_list, list)
        blocks_args = []
        for block_string in string_list:
            blocks_args.append(BlockDecoder._decode_block_string(block_string))
        return blocks_args

    @staticmethod
    def encode(blocks_args):
        """
        Encodes a list of BlockArgs to a list of strings.

        :param blocks_args: a list of BlockArgs namedtuples of block args
        :return: a list of strings, each string is a notation of block
        """
        block_strings = []
        for block in blocks_args:
            block_strings.append(BlockDecoder._encode_block_string(block))
        return block_strings


def efficientnet(width_coefficient=None, depth_coefficient=None, dropout_rate=0.2,
                 drop_connect_rate=0.2, image_size=None, num_classes=1000):
    """ Creates a efficientnet model. """

    blocks_args = [
        'r1_k3_s11_e1_i32_o16_se0.25', 'r2_k3_s22_e6_i16_o24_se0.25',
        'r2_k5_s22_e6_i24_o40_se0.25', 'r3_k3_s22_e6_i40_o80_se0.25',
        'r3_k5_s11_e6_i80_o112_se0.25', 'r4_k5_s22_e6_i112_o192_se0.25',
        'r1_k3_s11_e6_i192_o320_se0.25',
    ]
    blocks_args = BlockDecoder.decode(blocks_args)

    global_params = GlobalParams(
        batch_norm_momentum=0.99,
        batch_norm_epsilon=1e-3,
        dropout_rate=dropout_rate,
        drop_connect_rate=drop_connect_rate,
        # data_format='channels_last',  # removed, this is always true in PyTorch
        num_classes=num_classes,
        width_coefficient=width_coefficient,
        depth_coefficient=depth_coefficient,
        depth_divisor=8,
        min_depth=None,
        image_size=image_size,
    )

    return blocks_args, global_params


def get_model_params(model_name, override_params):
    """ Get the block args and global params for a given model """
    if model_name.startswith('efficientnet'):
        w, d, s, p = efficientnet_params(model_name)
        # note: all models have drop connect rate = 0.2
        blocks_args, global_params = efficientnet(
            width_coefficient=w, depth_coefficient=d, dropout_rate=p, image_size=s)
    else:
        raise NotImplementedError('model name is not pre-defined: %s' % model_name)
    if override_params:
        # ValueError will be raised here if override_params has fields not included in global_params.
        global_params = global_params._replace(**override_params)
    return blocks_args, global_params


url_map = {
    'efficientnet-b0': 'http://storage.googleapis.com/public-models/efficientnet/efficientnet-b0-355c32eb.pth',
    'efficientnet-b1': 'http://storage.googleapis.com/public-models/efficientnet/efficientnet-b1-f1951068.pth',
    'efficientnet-b2': 'http://storage.googleapis.com/public-models/efficientnet/efficientnet-b2-8bb594d6.pth',
    'efficientnet-b3': 'http://storage.googleapis.com/public-models/efficientnet/efficientnet-b3-5fb5a3c3.pth',
    'efficientnet-b4': 'http://storage.googleapis.com/public-models/efficientnet/efficientnet-b4-6ed6700e.pth',
    'efficientnet-b5': 'http://storage.googleapis.com/public-models/efficientnet/efficientnet-b5-b6417697.pth',
    'efficientnet-b6': 'http://storage.googleapis.com/public-models/efficientnet/efficientnet-b6-c76e70fd.pth',
    'efficientnet-b7': 'http://storage.googleapis.com/public-models/efficientnet/efficientnet-b7-dcc49843.pth',
}


fixbug=["_conv_stem.weight", "_bn0.weight", "_bn0.bias", "_bn0.running_mean", "_bn0.running_var", "_blocks.0._depthwise_conv.weight", "_blocks.0._bn1.weight", "_blocks.0._bn1.bias", "_blocks.0._bn1.running_mean", "_blocks.0._bn1.running_var", "_blocks.0._se_reduce.weight", "_blocks.0._se_reduce.bias", "_blocks.0._se_expand.weight", "_blocks.0._se_expand.bias", "_blocks.0._project_conv.weight", "_blocks.0._bn2.weight", "_blocks.0._bn2.bias", "_blocks.0._bn2.running_mean", "_blocks.0._bn2.running_var", "_blocks.1._expand_conv.weight", "_blocks.1._bn0.weight", "_blocks.1._bn0.bias", "_blocks.1._bn0.running_mean", "_blocks.1._bn0.running_var", "_blocks.1._depthwise_conv.weight", "_blocks.1._bn1.weight", "_blocks.1._bn1.bias", "_blocks.1._bn1.running_mean", "_blocks.1._bn1.running_var", "_blocks.1._se_reduce.weight", "_blocks.1._se_reduce.bias", "_blocks.1._se_expand.weight", "_blocks.1._se_expand.bias", "_blocks.1._project_conv.weight", "_blocks.1._bn2.weight", "_blocks.1._bn2.bias", "_blocks.1._bn2.running_mean", "_blocks.1._bn2.running_var", "_blocks.2._expand_conv.weight", "_blocks.2._bn0.weight", "_blocks.2._bn0.bias", "_blocks.2._bn0.running_mean", "_blocks.2._bn0.running_var", "_blocks.2._depthwise_conv.weight", "_blocks.2._bn1.weight", "_blocks.2._bn1.bias", "_blocks.2._bn1.running_mean", "_blocks.2._bn1.running_var", "_blocks.2._se_reduce.weight", "_blocks.2._se_reduce.bias", "_blocks.2._se_expand.weight", "_blocks.2._se_expand.bias", "_blocks.2._project_conv.weight", "_blocks.2._bn2.weight", "_blocks.2._bn2.bias", "_blocks.2._bn2.running_mean", "_blocks.2._bn2.running_var", "_blocks.3._expand_conv.weight", "_blocks.3._bn0.weight", "_blocks.3._bn0.bias", "_blocks.3._bn0.running_mean", "_blocks.3._bn0.running_var", "_blocks.3._depthwise_conv.weight", "_blocks.3._bn1.weight", "_blocks.3._bn1.bias", "_blocks.3._bn1.running_mean", "_blocks.3._bn1.running_var", "_blocks.3._se_reduce.weight", "_blocks.3._se_reduce.bias", "_blocks.3._se_expand.weight", "_blocks.3._se_expand.bias", "_blocks.3._project_conv.weight", "_blocks.3._bn2.weight", "_blocks.3._bn2.bias", "_blocks.3._bn2.running_mean", "_blocks.3._bn2.running_var", "_blocks.4._expand_conv.weight", "_blocks.4._bn0.weight", "_blocks.4._bn0.bias", "_blocks.4._bn0.running_mean", "_blocks.4._bn0.running_var", "_blocks.4._depthwise_conv.weight", "_blocks.4._bn1.weight", "_blocks.4._bn1.bias", "_blocks.4._bn1.running_mean", "_blocks.4._bn1.running_var", "_blocks.4._se_reduce.weight", "_blocks.4._se_reduce.bias", "_blocks.4._se_expand.weight", "_blocks.4._se_expand.bias", "_blocks.4._project_conv.weight", "_blocks.4._bn2.weight", "_blocks.4._bn2.bias", "_blocks.4._bn2.running_mean", "_blocks.4._bn2.running_var", "_blocks.5._expand_conv.weight", "_blocks.5._bn0.weight", "_blocks.5._bn0.bias", "_blocks.5._bn0.running_mean", "_blocks.5._bn0.running_var", "_blocks.5._depthwise_conv.weight", "_blocks.5._bn1.weight", "_blocks.5._bn1.bias", "_blocks.5._bn1.running_mean", "_blocks.5._bn1.running_var", "_blocks.5._se_reduce.weight", "_blocks.5._se_reduce.bias", "_blocks.5._se_expand.weight", "_blocks.5._se_expand.bias", "_blocks.5._project_conv.weight", "_blocks.5._bn2.weight", "_blocks.5._bn2.bias", "_blocks.5._bn2.running_mean", "_blocks.5._bn2.running_var", "_blocks.6._expand_conv.weight", "_blocks.6._bn0.weight", "_blocks.6._bn0.bias", "_blocks.6._bn0.running_mean", "_blocks.6._bn0.running_var", "_blocks.6._depthwise_conv.weight", "_blocks.6._bn1.weight", "_blocks.6._bn1.bias", "_blocks.6._bn1.running_mean", "_blocks.6._bn1.running_var", "_blocks.6._se_reduce.weight", "_blocks.6._se_reduce.bias", "_blocks.6._se_expand.weight", "_blocks.6._se_expand.bias", "_blocks.6._project_conv.weight", "_blocks.6._bn2.weight", "_blocks.6._bn2.bias", "_blocks.6._bn2.running_mean", "_blocks.6._bn2.running_var", "_blocks.7._expand_conv.weight", "_blocks.7._bn0.weight", "_blocks.7._bn0.bias", "_blocks.7._bn0.running_mean", "_blocks.7._bn0.running_var", "_blocks.7._depthwise_conv.weight", "_blocks.7._bn1.weight", "_blocks.7._bn1.bias", "_blocks.7._bn1.running_mean", "_blocks.7._bn1.running_var", "_blocks.7._se_reduce.weight", "_blocks.7._se_reduce.bias", "_blocks.7._se_expand.weight", "_blocks.7._se_expand.bias", "_blocks.7._project_conv.weight", "_blocks.7._bn2.weight", "_blocks.7._bn2.bias", "_blocks.7._bn2.running_mean", "_blocks.7._bn2.running_var", "_blocks.8._expand_conv.weight", "_blocks.8._bn0.weight", "_blocks.8._bn0.bias", "_blocks.8._bn0.running_mean", "_blocks.8._bn0.running_var", "_blocks.8._depthwise_conv.weight", "_blocks.8._bn1.weight", "_blocks.8._bn1.bias", "_blocks.8._bn1.running_mean", "_blocks.8._bn1.running_var", "_blocks.8._se_reduce.weight", "_blocks.8._se_reduce.bias", "_blocks.8._se_expand.weight", "_blocks.8._se_expand.bias", "_blocks.8._project_conv.weight", "_blocks.8._bn2.weight", "_blocks.8._bn2.bias", "_blocks.8._bn2.running_mean", "_blocks.8._bn2.running_var", "_blocks.9._expand_conv.weight", "_blocks.9._bn0.weight", "_blocks.9._bn0.bias", "_blocks.9._bn0.running_mean", "_blocks.9._bn0.running_var", "_blocks.9._depthwise_conv.weight", "_blocks.9._bn1.weight", "_blocks.9._bn1.bias", "_blocks.9._bn1.running_mean", "_blocks.9._bn1.running_var", "_blocks.9._se_reduce.weight", "_blocks.9._se_reduce.bias", "_blocks.9._se_expand.weight", "_blocks.9._se_expand.bias", "_blocks.9._project_conv.weight", "_blocks.9._bn2.weight", "_blocks.9._bn2.bias", "_blocks.9._bn2.running_mean", "_blocks.9._bn2.running_var", "_blocks.10._expand_conv.weight", "_blocks.10._bn0.weight", "_blocks.10._bn0.bias", "_blocks.10._bn0.running_mean", "_blocks.10._bn0.running_var", "_blocks.10._depthwise_conv.weight", "_blocks.10._bn1.weight", "_blocks.10._bn1.bias", "_blocks.10._bn1.running_mean", "_blocks.10._bn1.running_var", "_blocks.10._se_reduce.weight", "_blocks.10._se_reduce.bias", "_blocks.10._se_expand.weight", "_blocks.10._se_expand.bias", "_blocks.10._project_conv.weight", "_blocks.10._bn2.weight", "_blocks.10._bn2.bias", "_blocks.10._bn2.running_mean", "_blocks.10._bn2.running_var", "_blocks.11._expand_conv.weight", "_blocks.11._bn0.weight", "_blocks.11._bn0.bias", "_blocks.11._bn0.running_mean", "_blocks.11._bn0.running_var", "_blocks.11._depthwise_conv.weight", "_blocks.11._bn1.weight", "_blocks.11._bn1.bias", "_blocks.11._bn1.running_mean", "_blocks.11._bn1.running_var", "_blocks.11._se_reduce.weight", "_blocks.11._se_reduce.bias", "_blocks.11._se_expand.weight", "_blocks.11._se_expand.bias", "_blocks.11._project_conv.weight", "_blocks.11._bn2.weight", "_blocks.11._bn2.bias", "_blocks.11._bn2.running_mean", "_blocks.11._bn2.running_var", "_blocks.12._expand_conv.weight", "_blocks.12._bn0.weight", "_blocks.12._bn0.bias", "_blocks.12._bn0.running_mean", "_blocks.12._bn0.running_var", "_blocks.12._depthwise_conv.weight", "_blocks.12._bn1.weight", "_blocks.12._bn1.bias", "_blocks.12._bn1.running_mean", "_blocks.12._bn1.running_var", "_blocks.12._se_reduce.weight", "_blocks.12._se_reduce.bias", "_blocks.12._se_expand.weight", "_blocks.12._se_expand.bias", "_blocks.12._project_conv.weight", "_blocks.12._bn2.weight", "_blocks.12._bn2.bias", "_blocks.12._bn2.running_mean", "_blocks.12._bn2.running_var", "_blocks.13._expand_conv.weight", "_blocks.13._bn0.weight", "_blocks.13._bn0.bias", "_blocks.13._bn0.running_mean", "_blocks.13._bn0.running_var", "_blocks.13._depthwise_conv.weight", "_blocks.13._bn1.weight", "_blocks.13._bn1.bias", "_blocks.13._bn1.running_mean", "_blocks.13._bn1.running_var", "_blocks.13._se_reduce.weight", "_blocks.13._se_reduce.bias", "_blocks.13._se_expand.weight", "_blocks.13._se_expand.bias", "_blocks.13._project_conv.weight", "_blocks.13._bn2.weight", "_blocks.13._bn2.bias", "_blocks.13._bn2.running_mean", "_blocks.13._bn2.running_var", "_blocks.14._expand_conv.weight", "_blocks.14._bn0.weight", "_blocks.14._bn0.bias", "_blocks.14._bn0.running_mean", "_blocks.14._bn0.running_var", "_blocks.14._depthwise_conv.weight", "_blocks.14._bn1.weight", "_blocks.14._bn1.bias", "_blocks.14._bn1.running_mean", "_blocks.14._bn1.running_var", "_blocks.14._se_reduce.weight", "_blocks.14._se_reduce.bias", "_blocks.14._se_expand.weight", "_blocks.14._se_expand.bias", "_blocks.14._project_conv.weight", "_blocks.14._bn2.weight", "_blocks.14._bn2.bias", "_blocks.14._bn2.running_mean", "_blocks.14._bn2.running_var", "_blocks.15._expand_conv.weight", "_blocks.15._bn0.weight", "_blocks.15._bn0.bias", "_blocks.15._bn0.running_mean", "_blocks.15._bn0.running_var", "_blocks.15._depthwise_conv.weight", "_blocks.15._bn1.weight", "_blocks.15._bn1.bias", "_blocks.15._bn1.running_mean", "_blocks.15._bn1.running_var", "_blocks.15._se_reduce.weight", "_blocks.15._se_reduce.bias", "_blocks.15._se_expand.weight", "_blocks.15._se_expand.bias", "_blocks.15._project_conv.weight", "_blocks.15._bn2.weight", "_blocks.15._bn2.bias", "_blocks.15._bn2.running_mean", "_blocks.15._bn2.running_var", "_conv_head.weight", "_bn1.weight", "_bn1.bias", "_bn1.running_mean", "_bn1.running_var", "_fc.weight", "_fc.bias"]

dont_need=["base2.0._bn0.num_batches_tracked", "base2.0._blocks.0._bn1.num_batches_tracked", "base2.0._blocks.0._bn2.num_batches_tracked", "base2.0._blocks.1._bn0.num_batches_tracked", "base2.0._blocks.1._bn1.num_batches_tracked", "base2.0._blocks.1._bn2.num_batches_tracked", "base2.0._blocks.2._bn0.num_batches_tracked", "base2.0._blocks.2._bn1.num_batches_tracked", "base2.0._blocks.2._bn2.num_batches_tracked", "base2.0._blocks.3._bn0.num_batches_tracked", "base2.0._blocks.3._bn1.num_batches_tracked", "base2.0._blocks.3._bn2.num_batches_tracked", "base2.0._blocks.4._bn0.num_batches_tracked", "base2.0._blocks.4._bn1.num_batches_tracked", "base2.0._blocks.4._bn2.num_batches_tracked", "base2.0._blocks.5._bn0.num_batches_tracked", "base2.0._blocks.5._bn1.num_batches_tracked", "base2.0._blocks.5._bn2.num_batches_tracked", "base2.0._blocks.6._bn0.num_batches_tracked", "base2.0._blocks.6._bn1.num_batches_tracked", "base2.0._blocks.6._bn2.num_batches_tracked", "base2.0._blocks.7._bn0.num_batches_tracked", "base2.0._blocks.7._bn1.num_batches_tracked", "base2.0._blocks.7._bn2.num_batches_tracked", "base2.0._blocks.8._bn0.num_batches_tracked", "base2.0._blocks.8._bn1.num_batches_tracked", "base2.0._blocks.8._bn2.num_batches_tracked", "base2.0._blocks.9._bn0.num_batches_tracked", "base2.0._blocks.9._bn1.num_batches_tracked", "base2.0._blocks.9._bn2.num_batches_tracked", "base2.0._blocks.10._bn0.num_batches_tracked", "base2.0._blocks.10._bn1.num_batches_tracked", "base2.0._blocks.10._bn2.num_batches_tracked", "base2.0._blocks.11._bn0.num_batches_tracked", "base2.0._blocks.11._bn1.num_batches_tracked", "base2.0._blocks.11._bn2.num_batches_tracked", "base2.0._blocks.12._bn0.num_batches_tracked", "base2.0._blocks.12._bn1.num_batches_tracked", "base2.0._blocks.12._bn2.num_batches_tracked", "base2.0._blocks.13._bn0.num_batches_tracked", "base2.0._blocks.13._bn1.num_batches_tracked", "base2.0._blocks.13._bn2.num_batches_tracked", "base2.0._blocks.14._bn0.num_batches_tracked", "base2.0._blocks.14._bn1.num_batches_tracked", "base2.0._blocks.14._bn2.num_batches_tracked", "base2.0._blocks.15._bn0.num_batches_tracked", "base2.0._blocks.15._bn1.num_batches_tracked", "base2.0._blocks.15._bn2.num_batches_tracked", "base2.0._bn1.num_batches_tracked"]

def load_pretrained_weights(model, model_name, load_fc=True):
    """ Loads pretrained weights, and downloads if loading for the first time. """
    state_dict = model_zoo.load_url(url_map[model_name])
    if load_fc:
        from collections import OrderedDict 
        state_dict_new = OrderedDict() 

        print("use load fc!!!!!!!")
        print(type(state_dict))
        for key, value in state_dict.items(): 
            # print(key,type(key),type(value))
            # if key in fixbug:
            #     state_dict_new[key]=value
            # elif 'base2.0.'+key in dont_need:
            #     continue 
            # else:
            if key in ['_fc.weight','_fc.bias']:
                state_dict_new['base2.0.'+key]=value
            else:
                state_dict_new['base2.0.'+key]=value
        from torchsummary import summary
        def torch_summarize(model, show_weights=True, show_parameters=True):
            """Summarizes torch model by showing trainable parameters and weights."""
            tmpstr = model.__class__.__name__ + ' (\n'
            for key, module in model._modules.items():
                # if it contains layers let call it recursively to get params and weights
                if type(module) in [
                    torch.nn.modules.container.Container,
                    torch.nn.modules.container.Sequential
                ]:
                    modstr = torch_summarize(module)
                else:
                    modstr = module.__repr__()
                modstr = _addindent(modstr, 2)

                params = sum([np.prod(p.size()) for p in module.parameters()])
                weights = tuple([tuple(p.size()) for p in module.parameters()])

                tmpstr += '  (' + key + '): ' + modstr 
                if show_weights:
                    tmpstr += ', weights={}'.format(weights)
                if show_parameters:
                    tmpstr +=  ', parameters={}'.format(params)
                tmpstr += '\n'   

            tmpstr = tmpstr + ')'
            return tmpstr
        print(torch_summarize(model))
        model.load_state_dict(state_dict)
    else:
        print("not use load fc!!!!!!!")
        state_dict.pop('_fc.weight')
        state_dict.pop('_fc.bias')
        res = model.load_state_dict(state_dict, strict=False)
        assert set(res.missing_keys) == set(['_fc.weight', '_fc.bias']), 'issue loading pretrained weights'
    print('Loaded pretrained weights for {}'.format(model_name))
