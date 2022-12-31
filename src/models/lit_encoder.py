import sys
import torch
import torch.nn as nn
import pytorch_lightning as pl
import argparse

sys.path.append("../../")
from src.models.lit_losses import loss_fn


# TODO: add train file, and function to train only one epoch.
#       - with PHATE distance (DONE)
#       - with PHATE Embedding (DONE)
#       - with reconstruction loss.


# TODO add early stopping https://pytorch-lightning.readthedocs.io/en/#stable/common/early_stopping.html (use Hydra and callbacks)


class LitAutoencoder(pl.LightningModule):
    def __init__(
        self,
        input_dim,
        emb_dim,
        encoder_layer=[10, 10, 10],
        decoder_layer=[10, 10, 10],
        activation="ReLU",
        lr=0.001,
        kernel_type="phate",
        loss_emb=False,
        loss_dist=True,
        loss_rec=False,
        bandwidth=10,
        t=1,
        scale=0.05,
        knn=5,
        logp=False,
        **kwargs,
    ) -> None:
        
        
        self.logp = logp
        #Specify encoder
        super().__init__()
        encoder_layer.insert(0, input_dim)
        encoder = []
        # encoder.append(nn.BatchNorm1d(encoder_layer[0]))
        for i0, i1 in zip(encoder_layer, encoder_layer[1:]):
            encoder.append(nn.Linear(i0, i1))
            if i1 != encoder_layer[-1]:
                encoder.append(getattr(nn, activation)())
        if self.logp:
            encoder.append(nn.Softmax(dim=1))
        self.encoder = nn.Sequential(*encoder)
        
        print(encoder)
        #Specify decoder
        
        decoder_layer.insert(0, emb_dim)
        decoder=[]
        for i0,i1 in zip(decoder_layer,decoder_layer[1:]):
           decoder.append(nn.Linear(i0, i1))
           decoder.append(getattr(nn, activation)())
        print(decoder)
        self.decoder = nn.Sequential(*decoder)
            
 
        self.lr = lr
        self.kernel_type = kernel_type
        self.loss_emb = loss_emb
        self.loss_dist = loss_dist
        self.bandwidth = bandwidth
        self.t = t
        self.scale = scale
        self.knn = knn
        self.loss_rec = loss_rec
        
        
    def decode(self,x):
        return self.decoder(x)

    def encode(self,x):
        if self.logp:
            x = torch.log(self.encoder(x) + 1e-6)
        else:
            x = self.encoder(x)
        return x

    def forward(self,x):
        if self.logp:
            x = torch.log(self.encoder(x) + 1e-6)# NOTE: 1e-6 to avoid log of 0. 
        else:
            x = self.encoder(x) 
        return x #self.decoder(x)

    
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


    def configure_optimizers(self):
        return torch.optim.AdamW(self.parameters(), lr=self.lr)

    def training_step(self, batch, batch_idx):
        sample, target = batch

        noise = self.scale * torch.randn(sample.size()).to(sample.device)
        if self.logp:
            encoded_sample = self.encode(sample + noise)
        else:
            encoded_sample = self.encode(sample + noise)
        decoded_sample = self.decode(encoded_sample)

        if self.loss_rec:
            decoded_sample = self.decode(encoded_sample)
        else:
            decoded_sample = torch.tensor(0.0).float().to(sample.device)
            

        
        loss_d, loss_e, loss_r = loss_fn(
            encoded_sample = encoded_sample,
            decoded_sample = decoded_sample,
            sample=sample,
            target=target,
            kernel_type=self.kernel_type,
            loss_emb=self.loss_emb,
            loss_dist=self.loss_dist,
            loss_recon = self.loss_rec,
            bandwidth=self.bandwidth,
            t=self.t,
            knn=self.knn,
        )
        
        
        loss = loss_e + loss_d + loss_r # Loss distances and loss embedding
        
        
        tensorboard_log = {"train_loss": loss}
        self.log("training_losses", {"loss_d": loss_d, "loss_e": loss_e, "loss": loss})
        return {"loss": loss, "log": tensorboard_log}



class LitDistEncoder(pl.LightningModule):
    def __init__(
        self,
        input_dim,
        emb_dim,
        encoder_layer=[10, 10, 10],
        decoder_layer=[10, 10, 10],
        activation="ReLU",
        lr=0.001,
        kernel_type="phate",
        loss_emb=False,
        loss_dist=True,
        loss_rec=False,
        bandwidth=10,
        t=1,
        scale=0.05,
        knn=5,
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
        encoder.append(nn.Softmax(dim=1))
        self.encoder = nn.Sequential(*encoder)
        print(encoder)
       
        decoder_layer.insert(0, emb_dim)
        decoder=[]
        for i0,i1 in zip(decoder_layer,decoder_layer[1:]):
            decoder.append(nn.Linear(i0, i1))
            decoder.append(getattr(nn, activation)())
        print(decoder)
        self.decoder = nn.Sequential(*decoder)

        self.lr = lr
        self.kernel_type = kernel_type
        self.loss_emb = loss_emb
        self.loss_dist = loss_dist
        self.loss_rec = loss_rec
        self.bandwidth = bandwidth
        self.t = t
        self.scale = scale
        self.knn = knn
        
    def decode(self,x):
        return self.decoder(x)
    
    def encode(self,x):
        return self.encoder(x)
    
    def forward(self,x):
        x = torch.log(self.encoder(x) + 1e-6) # NOTE: 1e-6 to avoid log of 0. 
        return #self.decoder(x)
    

    
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

    def configure_optimizers(self):
        return torch.optim.AdamW(self.parameters(), lr=self.lr)

    def training_step(self, batch, batch_idx):
        sample, target = batch

        noise = self.scale * torch.randn(sample.size()).to(sample.device)
        encoded_sample = self.encode(sample + noise)
        
        
        if self.loss_rec:
            decoded_sample = self.decode(encoded_sample)
        else:
            decoded_sample = torch.tensor(0.0).float().to(sample.device)

        loss_d, loss_e, loss_r = loss_fn(
            encoded_sample = encoded_sample,
            decoded_sample = decoded_sample,
            sample = sample,
            target = target,
            kernel_type=self.kernel_type,
            loss_emb=self.loss_emb,
            loss_dist=self.loss_dist,
            loss_recon=self.loss_rec,
            bandwidth=self.bandwidth,
            t=self.t,
            knn=self.knn,
        )

        loss = loss_d# + loss_e + loss_r # Loss distance
        tensorboard_log = {"train_loss": loss}
        self.log("training_losses", {"loss": loss})
        return {"loss": loss, "log": tensorboard_log}

