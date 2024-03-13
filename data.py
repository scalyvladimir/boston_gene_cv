import pytorch_lightning as pl
from torch.utils.data import DataLoader, Subset, WeightedRandomSampler
from sklearn.model_selection import train_test_split
from torchvision import datasets, transforms as TT
import torch
import os

import augmentations

class BGDataModule(pl.LightningDataModule):
    def __init__(self, data_dir= "./", batch_size=1):
        super().__init__()
        self.save_hyperparameters()

        self.data_dir = data_dir
        assert os.path.isdir(data_dir)

        # Later we will deal with Subsets of dataset, those are sharing exact same transform -- that's is highly undesirable scenario
        # In order to avoid deepcopying dataset and replacing transforms we will do something slighly more elegant here 
        self.train_dataset = datasets.ImageFolder(
            root=self.data_dir,
            transform=augmentations.get_train_transform()
        )

        self.val_dataset = datasets.ImageFolder(
            root=self.data_dir,
            transform=augmentations.get_val_transform()
        )

        self.seed = 0xC0FFEE

        self.train_idx, self.val_idx = train_test_split(
            torch.arange(len(self.train_dataset), dtype=torch.int),
            test_size=0.2,
            random_state=self.seed,
            stratify=self.train_dataset.targets
        )

        self.batch_size = batch_size

    def setup(self, stage: str):
        if stage == 'fit':
            self.train_dataset = Subset(self.train_dataset, self.train_idx)
            self.val_dataset = Subset(self.val_dataset, self.val_idx)

            t_tgts = torch.tensor(self.train_dataset.targets)

            _, class_cnts = torch.unique(t_tgts[self.train_idx], return_counts=True)
    
            sample_weights = torch.zeros(len(t_tgts[self.train_idx]))
            
            for i, label_id in enumerate(self.train_idx):
                sample_weights[i] = 1. / class_cnts[t_tgts[label_id]]
            
            N = int(max(class_cnts) * len(class_cnts))
            
            self.train_sampler = WeightedRandomSampler(sample_weights, num_samples=N//2)

        if stage == "test":
            self.val_dataset = Subset(self.val_dataset, self.val_idx)

    def train_dataloader(self):
        return DataLoader(
            dataset=self.train_dataset,
            batch_size=self.batch_size,
            pin_memory=True,
            sampler=self.train_sampler,
            num_workers=11
        )

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size, pin_memory=True, num_workers=11, shuffle=False)

    def test_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size, pin_memory=True, num_workers=11, shuffle=False)