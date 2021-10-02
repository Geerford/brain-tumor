import pretrainedmodels
import torch
from efficientnet_pytorch import EfficientNet
from torchvision import models

from models.efficientnet3d import EfficientNet3D


def densenet(params):
    """
        types = ['dense121', 'dense161', 'dense169', 'dense201']
    """
    if params['model_type'] == 'dense121':
        return models.densenet121(pretrained=True)
    elif params['model_type'] == 'dense161':
        return models.densenet161(pretrained=True)
    elif params['model_type'] == 'dense169':
        return models.densenet169(pretrained=True)
    else:
        return models.densenet201(pretrained=True)


def resnet(params):
    """
        types = ['resnet50', 'resnet101']
    """
    if params['model_type'] == 'resnet50':
        return pretrainedmodels.__dict__[params['model_type']](num_classes=1000, pretrained='imagenet')
    else:
        return models.resnet101(pretrained=True)


def inception_v3(params):
    return models.inception_v3(pretrained=True)


def se_resnext(params):
    """
        types = ['se_resnext50_32x4d', 'se_resnext101_32x4d']
    """
    return pretrainedmodels.__dict__[params['model_type']](num_classes=1000, pretrained='imagenet')


def se_resnet(params):
    """
        types = ['se_resnet50', 'se_resnet101', 'se_resnet152']
    """
    return pretrainedmodels.__dict__[params['model_type']](num_classes=1000, pretrained='imagenet')


def senet154(params):
    """
        types = ['senet154']
    """
    return pretrainedmodels.__dict__[params['model_type']](num_classes=1000, pretrained='imagenet')


def polynet(params):
    """
        types = ['polynet']
    """
    return pretrainedmodels.__dict__[params['model_type']](num_classes=1000, pretrained='imagenet')


def dpn(params):
    """
        types = ['dpn68b', 'dpn92']
    """
    return pretrainedmodels.__dict__[params['model_type']](num_classes=1000, pretrained='imagenet+5k')


def nasnetamobile(params):
    """
        types = ['nasnetamobile']
    """
    return pretrainedmodels.__dict__[params['model_type']](num_classes=1000, pretrained='imagenet')


def resnext101(params):
    """
        types = ['resnext101_32x8d_wsl', 'resnext101_32x16d_wsl', 'resnext101_32x32d_wsl', 'resnext101_32x48d_wsl',
                'resnext101_64x4d', 'resnext101_32x4d']
    """
    if params['model_type'] in ['resnext101_32x8d_wsl', 'resnext101_32x16d_wsl', 'resnext101_32x32d_wsl', 'resnext101_32x48d_wsl']:
        return torch.hub.load('facebookresearch/WSL-Images', params['model_type'])
    elif params['model_type'] == 'resnext101_64x4d':
        return pretrainedmodels.__dict__[params['model_type']](num_classes=1000, pretrained='imagenet')
    else:
        return pretrainedmodels.__dict__[params['model_type']](num_classes=1000, pretrained='imagenet')


def efficientnet(params):
    """
        types = ['efficientnet-b0', 'efficientnet-b2', 'efficientnet-b3', 'efficientnet-b4', 'efficientnet-b5',
                'efficientnet-b6', 'efficientnet-b7']
    """
    return EfficientNet.from_pretrained(params['model_type'], num_classes=params['num_classes'], in_channels=params['input_channels'])


def efficientnet_3d(params):
    """
    types = ['efficientnet-b0', 'efficientnet-b2', 'efficientnet-b3', 'efficientnet-b4', 'efficientnet-b5',
            'efficientnet-b6', 'efficientnet-b7']
    """
    return EfficientNet3D.from_name(params['model_type'],
                                    override_params={'num_classes': params['num_classes']},
                                    in_channels=params['input_channels'])
