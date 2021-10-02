import glob

import pandas as pd
import torch
from torch.utils import data

from utils import load_samples


class BrainTumorDataset(torch.utils.data.Dataset):
    def __init__(self, labels: pd.DataFrame, root_dir: str, params: dict):
        """
        Args:
            labels (list): List indexes of dataset.
            root_dir (string): Directory with all the images.
            params (dict, optional): Optional params to be applied
                on a sample load.
        """
        self.labels = labels
        self.root_dir = root_dir
        self.params = params
        if self.params['type'] == 'test':
            self.ids = [item.split("\\")[-1] for item in sorted(glob.glob(f"{self.root_dir}*"))]
        else:
            self.ids = [item.split("\\")[-1] for item in sorted(glob.glob(f"{self.root_dir}*")) if
                        item.split("\\")[-1] in [str(item_id).zfill(5) for item_id in labels['BraTS21ID']]]

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        samples = load_samples(path=self.root_dir + self.ids[idx],
                               params=self.params)
        if self.params['type'] == 'test':
            return samples.float(), idx
        else:
            return samples.float(), torch.tensor(
                self.labels.loc[self.labels['BraTS21ID'] == int(self.ids[idx])].values[0][1], dtype=torch.long)
