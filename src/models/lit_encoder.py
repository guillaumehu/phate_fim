import sys
import torch
import torch.nn as nn
import pytorch_lightning as pl
import argparse

sys.path.append("../../")
from src.models.lit_losses import loss_dist

# TODO: add train file, and function to train only one epoch.
#       - with PHATE distance (DONE)
#       - with PHATE Embedding (DONE)
#       - with reconstruction loss.


# TODO add early stopping https://pytorch-lightning.readthedocs.io/en/#stable/common/early_stopping.html


class LitAutoencoder(pl.LightningModule):
    def __init__(
        self,
        input_dim,
        encoder_layer=[10, 10, 10],
        decoder_layer=[10, 10, 10]
        decoder = False
        activation="ReLU",
        lr=0.001,
        kernel_type="phate",
        loss_emb=True,
        loss_dist=True,
        bandwidth=10,
        t=1,
        scale=0.05,
        **kwargs,
    ) -> None:
        super().__init__()
        encoder_layer.insert(0, input_dim)
        encoder = []
        # encoder.append(nn.BatchNorm1d(encoder_layer[0]))
        for i0, i1 in zip(encoder_layer, encoder_layer[1:]):
            encoder.append(nn.Linear(i0, i1))
            if i1 != encoder_layer[-1]:
                encoder.append(getattr(nn, activation)())
        self.encoder = nn.Sequential(*encoder)

        self.lr = lr
        self.kernel_type = kernel_type
        self.loss_emb = loss_emb
        self.loss_dist = loss_dist
        self.bandwidth = bandwidth
        self.t = t
        self.scale = scale
        
        if decoder:
            decoder_layer.insert(0, input_dim)
            decoder=[]
            for i0,i1 in zip(decoder_layer,decoder_layer[1:]):
               decoder.append(nn.Linear(i0, i1))
               decoder.append(getattr(nn, activation)())
            print(decoder)
            self.decoder = nn.Sequential(*decoder).to(device)

    def encode(self,x):
       return self.encoder(x)

    def decoder(self,x):
       return self.decoder(x)
    
    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = parent_parser.add_argument_group("LitModel")
        parser.add_argument(
            "--encoder_layer",
            type=lambda s: [int(item) for item in s.split(",")],
            default=[10, 10, 10],
            help="List of layers excluding the input dimension. Default to `[10,10,10]`. In the command line, it takes values separated by a comma, e.g. `10,10,10`.",
        )
        parser.add_argument("--activation", type=str, default="ReLU")
        parser.add_argument("--kernel_type", type=str, default="decay")
        parser.add_argument(
            "--loss_emb", default=True, action=argparse.BooleanOptionalAction
        )
        parser.add_argument("--decoder", action='store_true')
        return parent_parser

    def forward(self, x):
        x = self.encoder(x)
        return x  # self.decoder(x)

    def configure_optimizers(self):
        return torch.optim.AdamW(self.parameters(), lr=self.lr)

    def training_step(self, batch, batch_idx):
        sample, target = batch

        noise = self.scale * torch.randn(sample.size()).to(sample.device)
        encode_sample = self.forward(sample + noise)

        loss_d, loss_e = loss_dist(
            encode_sample,
            sample,
            kernel_type=self.kernel_type,
            loss_emb=self.loss_emb,
            loss_dist=self.loss_dist,
            bandwidth=self.bandwidth,
            t=self.t,
            target=target,
        )

        loss = loss_e + loss_d  # Loss distances and loss embedding
        tensorboard_log = {"train_loss": loss}
        self.log("training_losses", {"loss_d": loss_d, "loss_e": loss_e, "loss": loss})
        return {"loss": loss, "log": tensorboard_log}

