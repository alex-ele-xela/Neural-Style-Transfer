"""
Contains VGG16 and VGG19 nets that output the feature maps from specific pre-defined layers relevant to the Nerual Style Transfer task
"""

from collections import namedtuple
import torch
from torchvision import models


class Vgg16(torch.nn.Module):
    """
    Class to implement VGG16's CNN with Pre-trained weights and return output feature maps of of relevant layers

    Args:
        requires_grad (bool): Do the params require grad, i.e., if they are trainable
        show_progress (bool): Should the progress of downloading VGG16 weights be shown
    """

    def __init__(self, requires_grad=False, show_progress=False):
        super().__init__()

        # download weights
        vgg_pretrained_features = models.vgg16(weights=models.VGG16_Weights.DEFAULT, progress=show_progress).features
        
        # naming relevant layers to extract feature maps from
        self.layer_names = ['relu1_2', 'relu2_2', 'relu3_3', 'relu4_3']
        self.content_feature_maps_index = 1  # relu2_2
        self.style_feature_maps_indices = list(range(len(self.layer_names)))

        # slicing into 4 slices to extract the relevant feature map output of each slice
        self.slice1 = torch.nn.Sequential()
        self.slice2 = torch.nn.Sequential()
        self.slice3 = torch.nn.Sequential()
        self.slice4 = torch.nn.Sequential()
        for x in range(4):
            self.slice1.add_module(str(x), vgg_pretrained_features[x])
        for x in range(4, 9):
            self.slice2.add_module(str(x), vgg_pretrained_features[x])
        for x in range(9, 16):
            self.slice3.add_module(str(x), vgg_pretrained_features[x])
        for x in range(16, 23):
            self.slice4.add_module(str(x), vgg_pretrained_features[x])
        if not requires_grad:
            for param in self.parameters():
                param.requires_grad = False

    def forward(self, x):
        x = self.slice1(x)
        relu1_2 = x

        x = self.slice2(x)
        relu2_2 = x

        x = self.slice3(x)
        relu3_3 = x

        x = self.slice4(x)
        relu4_3 = x

        # returning feature maps as namedtuple
        vgg_outputs = namedtuple("VggOutputs", self.layer_names)
        out = vgg_outputs(relu1_2, relu2_2, relu3_3, relu4_3)
        return out


class Vgg19(torch.nn.Module):
    """
    Class to implement VGG19's CNN with Pre-trained weights and return output feature maps of of relevant layers

    Args:
        requires_grad (bool): Do the params require grad, i.e., if they are trainable
        show_progress (bool): Should the progress of downloading VGG19 weights be shown
        use_relu (bool): Use ouput of relu layer function or not (use output of conv if not)
    """

    def __init__(self, requires_grad=False, show_progress=False, use_relu=True):
        super().__init__()

        # download weights
        vgg_pretrained_features = models.vgg19(weights=models.VGG19_Weights.DEFAULT, progress=show_progress).features
        
        # naming relevant layers to extract feature maps from
        if use_relu:
            self.layer_names = ['relu1_1', 'relu2_1', 'relu3_1', 'relu4_1', 'conv4_2', 'relu5_1']
            self.offset = 1
        else:
            self.layer_names = ['conv1_1', 'conv2_1', 'conv3_1', 'conv4_1', 'conv4_2', 'conv5_1']
            self.offset = 0
        self.content_feature_maps_index = 4  # conv4_2
        
        # all layers used for style representation except conv4_2
        self.style_feature_maps_indices = list(range(len(self.layer_names)))
        self.style_feature_maps_indices.remove(4)  # conv4_2

        # slicing into 6 slices to extract the relevant feature map output of each slice
        self.slice1 = torch.nn.Sequential()
        self.slice2 = torch.nn.Sequential()
        self.slice3 = torch.nn.Sequential()
        self.slice4 = torch.nn.Sequential()
        self.slice5 = torch.nn.Sequential()
        self.slice6 = torch.nn.Sequential()
        for x in range(1+self.offset):
            self.slice1.add_module(str(x), vgg_pretrained_features[x])
        for x in range(1+self.offset, 6+self.offset):
            self.slice2.add_module(str(x), vgg_pretrained_features[x])
        for x in range(6+self.offset, 11+self.offset):
            self.slice3.add_module(str(x), vgg_pretrained_features[x])
        for x in range(11+self.offset, 20+self.offset):
            self.slice4.add_module(str(x), vgg_pretrained_features[x])
        for x in range(20+self.offset, 22):
            self.slice5.add_module(str(x), vgg_pretrained_features[x])
        for x in range(22, 29+self.offset):
            self.slice6.add_module(str(x), vgg_pretrained_features[x])
        if not requires_grad:
            for param in self.parameters():
                param.requires_grad = False

    def forward(self, x):
        x = self.slice1(x)
        layer1_1 = x
        x = self.slice2(x)
        layer2_1 = x
        x = self.slice3(x)
        layer3_1 = x
        x = self.slice4(x)
        layer4_1 = x
        x = self.slice5(x)
        conv4_2 = x
        x = self.slice6(x)
        layer5_1 = x

        # returning feature maps as namedtuple
        vgg_outputs = namedtuple("VggOutputs", self.layer_names)
        out = vgg_outputs(layer1_1, layer2_1, layer3_1, layer4_1, conv4_2, layer5_1)
        return out