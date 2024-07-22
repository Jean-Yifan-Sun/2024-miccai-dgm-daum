import numpy as np
import torch
import os
import pytorch_lightning as pl
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
import logging


class CardiacMRIDatset(torch.utils.data.Dataset):
    def __init__(self, data_path, subset='training', transform=None):
        super().__init__()
        valid_subsets = ['training', 'validation', 'testing']
        if subset not in valid_subsets:
            raise ValueError(f'Unknown subset {subset}. Valid subsets are: {valid_subsets}')

        self.transform = transform

        self.paths = np.load(os.path.join(data_path, f'{subset}.npy'))

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        path = self.paths[idx]

        try:
            data = np.load(path)
            image = data['image']
            labels = data['labels']
        except Exception as e:
            print("####### Error at: ", idx)
            logging.info("####### Error at: " + str(idx))
            raise e

        # replace nan with -1
        labels = np.nan_to_num(labels, nan=-1)

        if self.transform:
            image = self.transform(image)

        return image, labels

class CardiacMRILoader(pl.LightningDataModule):
    def __init__(self, batch_size, data_path, num_workers=2, *args, **kwargs):
        super().__init__()
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.data_path = data_path

        # Create training, validation and test datasets
        self.train_dataset = CardiacMRIDatset(self.data_path, subset='training', transform=None)
        self.val_dataset = CardiacMRIDatset(self.data_path, subset='validation', transform=None)
        self.test_dataset = CardiacMRIDatset(self.data_path, subset='testing', transform=None)

    def train_dataloader(self):
        train_dl = DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            drop_last=True,
            pin_memory=True,
            num_workers=4
        )

        return train_dl

    def val_dataloader(self):
        val_dl = DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            drop_last=True,
            pin_memory=True,
            num_workers=4
        )

        return val_dl

    def test_dataloader(self):
        val_dl = DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            drop_last=True,
            pin_memory=True,
            num_workers=4
        )

        return val_dl



class DistCardiacMRILoader(pl.LightningDataModule):
    def __init__(self, rank, world_size, args):
        super().__init__()
        akeys = args.keys()
        self.target_size = args['target_size'] if 'target_size' in akeys else (64, 64)
        self.batch_size = args['batch_size'] if 'batch_size' in akeys else 8
        self.num_workers = args['num_workers'] if 'num_workers' in akeys else 2
        self.data_path = args['data_path']

        # Create training, validation and test datasets
        self.train_dataset = CardiacMRIDatset(self.data_path, subset='training', transform=None)
        self.val_dataset = CardiacMRIDatset(self.data_path, subset='validation', transform=None)
        self.test_dataset = CardiacMRIDatset(self.data_path, subset='testing', transform=None)

        self.train_sampler = torch.utils.data.distributed.DistributedSampler(
            self.train_dataset,
            num_replicas=world_size,
            rank=rank
        )

        self.val_sampler = torch.utils.data.distributed.DistributedSampler(
            self.val_dataset,
            num_replicas=world_size,
            rank=rank
        )

        self.test_sampler = torch.utils.data.distributed.DistributedSampler(
            self.test_dataset,
            num_replicas=world_size,
            rank=rank
        )

    def train_dataloader(self):
        train_dl = DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            drop_last=False,
            pin_memory=True,
            num_workers=4,
            sampler=self.train_sampler
        )

        return train_dl

    def val_dataloader(self):
        val_dl = DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            drop_last=True,
            pin_memory=True,
            num_workers=4,
            sampler=self.val_sampler
        )

        return val_dl

    def test_dataloader(self):
        test_dl = DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            drop_last=True,
            pin_memory=True,
            num_workers=4,
            sampler=self.test_sampler
        )

        return test_dl

