import glob
import os
import random

import albumentations as A
import numpy as np
import torch
import torch.backends.cudnn as cudnn
# import cv2
from PIL import Image, ImageOps
from albumentations.pytorch import ToTensorV2


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


def transform_sample(sample, transform, size, mode):
    pad_size = 256 if size > 256 else 224
    resize_image = A.Compose([
        A.PadIfNeeded(min_height=pad_size, min_width=pad_size, value=0),
        A.Resize(size, size)
    ])
    sample = resize_image(image=sample)['image']
    if mode == 'grayscale':
        sample = ImageOps.grayscale(sample)
    if transform:
        normalize = A.Normalize(mean=[0.5], std=[0.5]) if mode == 'grayscale' else A.Normalize(mean=[0.485, 0.456, 0.406],
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
    # sample = Image.open(path)
    # sample = sample.resize((size, size), Image.BILINEAR)
    # if mode == 'grayscale':
    #     # sample = cv2.imread(path, 0)
    #     sample = ImageOps.grayscale(sample)
    # else:
        # sample = cv2.imread(path)
    # return cv2.resize(sample, (size, size))
    return np.asarray(Image.open(path))


def load_seq_samples(seq_path, params: dict):
    samples = torch.stack(
        [transform_sample(read_image(item_path), params['transform'], params['input_size'], params['image_read_mode'])
         for item_path in seq_path])
    # if len(samples) == 0:
    #     samples = np.zeros((params['num_images_seq'], params['input_size'], params['input_size']))
    # elif len(samples) < params['num_images_seq']:
    #     samples = np.concatenate((samples, np.zeros((params['num_images_seq'] - len(samples),
    #                                                  params['input_size'], params['input_size']))))
    # elif len(samples) > params['num_images_seq']:
    #     samples = samples[len(samples) // 2 - (int(params['num_images_seq'] / 2)):
    #                       len(samples) // 2 + (int(params['num_images_seq'] / 2))]
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
