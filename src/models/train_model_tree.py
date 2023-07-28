import sys
import torch
import torch.nn as nn
import numpy as np
import argparse
from argparse import ArgumentParser
import pytorch_lightning as pl
import os

from pytorch_lightning import Trainer
from pytorch_lightning.loggers import WandbLogger

sys.path.append("../../")
from src.data.make_dataset import train_dataloader
from src.fim_noemb import torch_phate
#from src.models.lit_losses import phate_loss, loss_dist
from src.models.lit_encoder import LitAutoencoder
from src.data.make_dataset import make_tree
from src.fim_noemb import FIM


# TODO add early stopping https://pytorch-lightning.readthedocs.io/en/#stable/common/early_stopping.html


# Hyperparameters

parser = ArgumentParser()

# add Program level
parser.add_argument("--run_name", default=None)
parser.add_argument("--dataset", type=str, default="tree")
parser.add_argument("--n_obs", type=int, default=1600)
parser.add_argument("--n_dim", type=int, default=10)
parser.add_argument("--batch_size", type=int, default=150)
parser.add_argument("--lr", type=int, default=0.0001)
parser.add_argument("--knn", type=int, default=5)
parser.add_argument("--max_epochs", type=int, default=150)
parser.add_argument("--wandb", default=False, action=argparse.BooleanOptionalAction)
parser.add_argument("--activation", type=str, default="ReLU")
parser.add_argument("--kernel_type", type=str, default="phate")
parser.add_argument("--loss", type=str, default="loss_dist")
parser.add_argument("--encoder_layer",type=lambda s: [int(item) for item in s.split(",")],default=[100, 100, 50],help="List of layers excluding the input dimension. Default to `[10,10,10]`. In the command line, it takes values separated by a comma, e.g. `10,10,10`.",)
parser.add_argument("--scale", type=int, default=0.0005)
parser.add_argument("--inference",action='store_true')
parser.add_argument("--inference_obs",type=int,default=1600)

args = parser.parse_args()
dict_args = vars(args)

if __name__ == "__main__":
    seed = torch.randint(0, 1000, size=(1,))
    emb_dim = args.encoder_layer[-1]
    pl.utilities.seed.seed_everything(seed=seed)

    if args.run_name is None:
        args.run_name = f"data_" + args.dataset + ""

    if args.wandb:
        logger = WandbLogger(project="fim_phate", name=args.run_name)
    else:
        logger = False
    # deterministic=True for reproducibility
    train_loader = train_dataloader(
        args.dataset, args.n_obs, args.n_dim, emb_dim, args.batch_size,args.knn)

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

    #1) Train model with specified settings or load previous model


    if not args.inference:
        trainer = Trainer.from_argparse_args(
            args, accelerator="gpu", devices=1, logger=logger
        )
        model = LitAutoencoder(input_dim=args.n_dim, emb_dim=emb_dim, **dict_args)
        trainer.fit(model, train_dataloaders=train_loader)
        model_name = os.getcwd() + args.dataset + '.pt'
        torch.save(model.state_dict(), model_name)
    else:
        model_name = os.getcwd() + args.dataset + '.pt'
        model = LitAutoencoder(input_dim=args.n_dim, emb_dim=emb_dim, **dict_args)
        model.load_state_dict(torch.load(model_name))

    #2) Inference


    tree_data, tree_phate, tree_clusters = make_tree(n_obs=args.inference_obs, dim=args.n_dim,emb_dim=2,knn=args.knn)
    model.cuda()
    model.eval()
    tree_data = tree_data.to('cuda')
    pred = model.encode(tree_data).detach().cpu().numpy()
   

    #3) Compute FIM

    model.cuda()
    fcn = model.encode
    fisher = FIM(tree_data,fcn,args.inference_obs,args.n_dim,emb_dim,pred)
    fishermat, J = fisher.fit()

    print("Number of Observations for inference",args.inference_obs)
    print("Shape of FIM output",fishermat.shape)

    




