import glob
import os
import random

import albumentations as A
import imageio
import numpy as np
import pydicom
import torch
import torch.backends.cudnn as cudnn
# import cv2
import yaml
from PIL import Image, ImageOps
from albumentations.pytorch import ToTensorV2
from pydicom.pixel_data_handlers.util import apply_voi_lut


def set_seed(seed):
    if seed is None:
        seed = np.random.randint(low=0, high=(2 ** 31 - 1))
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True


def transform_sample(sample, transform, size, grayscale):
    if grayscale:
        sample = np.asarray(ImageOps.grayscale(sample))
        in_channels = 1
    else:
        sample = np.asarray(sample)
        in_channels = 3
    # pad_size = 256 if size > 256 else 224
    resize_image = A.Compose([
        # A.PadIfNeeded(min_height=pad_size, min_width=pad_size, value=0),
        A.Resize(size, size)
    ])
    sample = resize_image(image=sample)['image']
    sample = sample.reshape(in_channels, sample.shape[0], sample.shape[1])
    if transform:
        normalize = A.Normalize(mean=[0.5], std=[0.5]) if grayscale else A.Normalize(mean=[0.485, 0.456, 0.406],
                                                                                     std=[0.229, 0.224, 0.225])
        transform_image = A.Compose([
            A.HorizontalFlip(p=0.5),
            A.OneOf([
                A.GaussNoise(p=0.3),
                A.Blur(blur_limit=3, p=0.2)
            ], p=0.5),
            normalize,
            ToTensorV2()
        ])
        return transform_image(image=sample)['image']
    else:
        return torch.from_numpy(sample)


def read_image(path):
    dicom = pydicom.read_file(path)
    sample = apply_voi_lut(dicom.pixel_array, dicom)
    if dicom.PhotometricInterpretation == "MONOCHROME1":
        sample = np.amax(sample) - sample
    sample = sample - np.min(sample)
    with np.errstate(divide='ignore', invalid='ignore'):
        sample = sample / np.max(sample)
    sample = (sample * 255).astype(np.uint8)
    # Save png to storage
    png_path = path[:-4] + '.png'
    imageio.imsave(png_path, sample)
    # sample = Image.open(png_path)
    # sample = sample.resize((size, size), Image.BILINEAR)
    # if mode == 'grayscale':
    #     # sample = cv2.imread(path, 0)
    #     sample = ImageOps.grayscale(sample)
    # else:
    #     sample = cv2.imread(png_path)
    # return cv2.resize(sample, (size, size))
    return Image.open(png_path)


def load_seq_samples(seq_path, params: dict):
    samples = torch.stack(
        [transform_sample(read_image(item_path), params['transform'], params['input_size'], params['grayscale'])
         for item_path in seq_path])
    if params['num_images_seq'] == -1:
        samples = samples[len(samples) // 2]
    elif len(samples) == 0:
        samples = np.zeros((params['num_images_seq'], params['input_size'], params['input_size']))
    elif len(samples) < params['num_images_seq']:
        samples = np.concatenate((samples, np.zeros((params['num_images_seq'] - len(samples),
                                                     params['input_size'], params['input_size']))))
    elif len(samples) > params['num_images_seq']:
        samples = samples[len(samples) // 2 - (int(params['num_images_seq'] / 2)):
                          len(samples) // 2 + (int(params['num_images_seq'] / 2))]

    return samples


def load_samples(path, params: dict):
    if params['seq_type'] == 'FLAIR':
        flair = sorted(glob.glob(f"{path}\\{params['mri_types'][0]}\\*{params['sample_format']}"))
        flair_samples = load_seq_samples(seq_path=flair, params=params)
        return flair_samples
    elif params['seq_type'] == 'T1w':
        t1w = sorted(glob.glob(f"{path}\\{params['mri_types'][1]}\\*{params['sample_format']}"))
        t1w_samples = load_seq_samples(seq_path=t1w, params=params)
        return t1w_samples
    elif params['seq_type'] == 'T1wCE':
        t1wce = sorted(glob.glob(f"{path}\\{params['mri_types'][2]}\\*{params['sample_format']}"))
        t1wce_samples = load_seq_samples(seq_path=t1wce, params=params)
        return t1wce_samples
    elif params['seq_type'] == 'T2w':
        t2w = sorted(glob.glob(f"{path}\\{params['mri_types'][3]}\\*{params['sample_format']}"))
        t2w_samples = load_seq_samples(seq_path=t2w, params=params)
        return t2w_samples

    # return np.concatenate([flair_samples, t1w_samples, t1wce_samples, t2w_samples]).reshape(
    #     len(params['mri_types']), params['num_images_seq'], params['input_size'], params['input_size'])


def save_yaml(params: dict):
    with open('config/config.yaml', 'w') as stream:
        try:
            yaml.safe_dump(params, stream, default_flow_style=False)
        except yaml.YAMLError as e:
            print(e)


def load_yaml():
    with open("config/config.yaml", "r") as stream:
        try:
            params = yaml.safe_load(stream)
        except yaml.YAMLError as e:
            print(e)
    return params
