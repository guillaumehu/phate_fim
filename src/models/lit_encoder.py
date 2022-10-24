import sys
import torch
import torch.nn as nn
import pytorch_lightning as pl
import argparse

sys.path.append("../../")
from src.models.lit_losses import phate_loss, loss_dist

# TODO: add train file, and function to train only one epoch.
#       - with PHATE distance (DONE)
#       - with PHATE Embedding (DONE)
#       - with reconstruction loss.


# TODO add early stopping https://pytorch-lightning.readthedocs.io/en/#stable/common/early_stopping.html


class LitAutoencoder(pl.LightningModule):
    def __init__(
        self,
        input_dim,
        encoder_layer,
        activation="ReLU",
        lr=0.001,
        kernel_type="phate",
        loss_emb=True,
        bandwidth=10,
        t=1,
        scale=0.05,
        **kwargs,
    ) -> None:
        super().__init__()
        encoder_layer.insert(0, input_dim)
        encoder = []
        for i0, i1 in zip(encoder_layer, encoder_layer[1:]):
            encoder.append(nn.Linear(i0, i1))
            if i1 == 10:
                encoder.append(getattr(nn, activation)())
            else:
                encoder.append(getattr(nn, activation)())
        self.encoder = nn.Sequential(*encoder)

        self.lr = lr
        self.kernel_type = kernel_type
        self.loss_emb = loss_emb
        self.bandwidth=bandwidth
        self.t=t
        self.scale = scale
        # decoder=[]
        # for i0,i1 in zip(decoder_layer,decoder_layer[1:]):
        #   decoder.append(nn.Linear(i0, i1))
        #   decoder.append(getattr(nn, activation)())
        # self.decoder = nn.Sequential(*decoder).to(device)

    # def encode(self,x):
    #   return self.encoder(x)

    # def decoder(self,x):
    #   return self.decoder(x)
    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = parent_parser.add_argument_group("LitModel")
        parser.add_argument("--encoder_layer", default=[10, 10, 10])
        parser.add_argument("--activation", type=str, default="ReLU")
        parser.add_argument("--kernel_type", type=str, default="decay")
        parser.add_argument(
            "--loss_emb", default=True, action=argparse.BooleanOptionalAction
        )
        return parent_parser

    def forward(self, x):
        x = self.encoder(x)
        return x  # self.decoder(x)

    def configure_optimizers(self):
        return torch.optim.AdamW(self.parameters(), lr=self.lr)

    def training_step(self, batch, batch_idx):
        sample, _ = batch

        noise = self.scale * torch.randn(sample.size()).to(
            sample.device
        ) 
        encode_sample = self.forward(sample + noise)

        loss_d, loss_e = loss_dist(
            encode_sample, sample, kernel_type=self.kernel_type, loss_emb=self.loss_emb, bandwidth=self.bandwidth, t=self.t
        )
        loss = loss_d + loss_e
        tensorboard_log = {"train_loss": loss}
        self.log("training_losses", {"loss_d": loss_d, "loss_e": loss_e, "loss": loss})
        return {"loss": loss, "log": tensorboard_log}
