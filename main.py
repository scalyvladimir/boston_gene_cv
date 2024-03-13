import torch
from pytorch_lightning.cli import LightningCLI
from models import ClassificationNet
from data import BGDataModule


def cli_main():
    LightningCLI(
        model_class=ClassificationNet,
        datamodule_class=BGDataModule
        # parser_kwargs={
            # parser
        #     'default_config_files': ['configs/fit.yaml']
        # }
    )
    
if __name__ == '__main__':
    cli_main()