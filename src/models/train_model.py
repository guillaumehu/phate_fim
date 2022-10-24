import sys
import torch
import torch.nn as nn
import numpy as np
import argparse
from argparse import ArgumentParser
import pytorch_lightning as pl

from pytorch_lightning import Trainer
from pytorch_lightning.loggers import WandbLogger

sys.path.append("../../")
from src.data.make_dataset import train_dataloader
from src.fim_noemb import torch_phate
from src.models.lit_losses import phate_loss, loss_dist
from src.models.lit_encoder import LitAutoencoder


# TODO add early stopping https://pytorch-lightning.readthedocs.io/en/#stable/common/early_stopping.html


# Hyperparameters

parser = ArgumentParser()

# add Program level
parser.add_argument("--run_name", default=None)
parser.add_argument("--dataset", type=str, default="sphere")
parser.add_argument("--n_obs", type=int, default=1000)
parser.add_argument("--n_dim", type=int, default=5)
parser.add_argument("--batch_size", type=int, default=50)
parser.add_argument("--wandb", default=True, action=argparse.BooleanOptionalAction)

# add Model arg
parser = LitAutoencoder.add_model_specific_args(parser)

# add Trainer args
parser = Trainer.add_argparse_args(parser)

args = parser.parse_args()
dict_args = vars(args)


if __name__ == "__main__":
    seed = torch.randint(0, 1000, size=(1,))
    pl.utilities.seed.seed_everything(seed=seed)

    if args.run_name is None:
        args.run_name = f"data_" + args.dataset + ""

    if args.wandb:
        logger = WandbLogger(project="fim_phate", name=args.run_name)
    else:
        logger = False
    # deterministic=True for reproducibility
    train_loader = train_dataloader(
        args.dataset, args.n_obs, args.n_dim, args.batch_size
    )

    # To test
    # trainer = Trainer(fast_dev_run=True, accelerator="gpu", devices=1)

    # trainer = Trainer(
    #     max_epochs=num_epochs,
    #     fast_dev_run=False,
    #     accelerator="gpu",
    #     devices=1,
    #     logger=wandb_logger,
    #     log_every_n_steps=5,
    # )
    # model = LitAutoencoder(
    #     encoder_layer=encoder_layer,
    #     bandwidth=bandwidth,
    #     t=t,
    # )

    trainer = Trainer.from_argparse_args(
        args, accelerator="gpu", devices=1, logger=logger
    )
    model = LitAutoencoder(input_dim=args.n_dim, **dict_args)
    trainer.fit(model, train_dataloaders=train_loader)
