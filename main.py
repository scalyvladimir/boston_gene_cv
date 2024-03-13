import torch
from pytorch_lightning.cli import LightningCLI
from models import ClassificationNet
from data import BGDataModule


def cli_main():
    cli = LightningCLI(
        model_class=ClassificationNet,
        datamodule_class=BGDataModule,
        parser_kwargs={"parser_mode": "omegaconf"},
        save_config_callback=None
    )
    
if __name__ == '__main__':
    cli_main()