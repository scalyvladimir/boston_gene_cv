import pytorch_lightning as pl
from torch.utils.data import DataLoader, Subset, WeightedRandomSampler
from sklearn.model_selection import train_test_split
from torchvision import datasets, transforms as TT
import torch
import os


class BGDataModule(pl.LightningDataModule):
    def __init__(self, data_dir= "./", batch_size=1):
        super().__init__()
        self.save_hyperparameters()

        self.data_dir = data_dir
        assert os.path.isdir(data_dir)

        self.dataset = datasets.ImageFolder(
            root=self.data_dir,
            transform=TT.Compose([
                TT.Resize((128, 128)),
                TT.ToTensor(),
                TT.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225]
                )
            ])
        )

        # self.transform = None #transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
        
        self.seed = 0xC0FFEE

        self.train_idx, self.val_idx = train_test_split(
            torch.arange(len(self.dataset), dtype=torch.int),
            test_size=0.2,
            random_state=self.seed,
            stratify=self.dataset.targets
        )

        self.batch_size = batch_size

    def setup(self, stage: str):
        if stage == 'fit':
            self.train_dataset = Subset(self.dataset, self.train_idx)
            self.val_dataset = Subset(self.dataset, self.val_idx)

            t_tgts = torch.tensor(self.dataset.targets)

            _, class_cnts = torch.unique(t_tgts[self.train_idx], return_counts=True)
    
            sample_weights = torch.zeros(len(t_tgts[self.train_idx]))
            
            for i, label_id in enumerate(self.train_idx):
                sample_weights[i] = 1. / class_cnts[t_tgts[label_id]]
            
            N = int(max(class_cnts) * len(class_cnts))
            
            self.train_sampler = WeightedRandomSampler(sample_weights, num_samples=N//2)

        if stage == "test":
            self.val_dataset = Subset(self.dataset, self.val_idx)

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