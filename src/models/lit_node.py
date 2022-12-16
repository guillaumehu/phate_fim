import sys
import torch
import torch.nn as nn
import pytorch_lightning as pl
import argparse
from torchdiffeq import odeint_adjoint as odeint
from typing import Union, List, Any
import itertools
import torch.nn.functional as F
import numpy as np

sys.path.append("../../")
from src.models.lit_losses import loss_fn


class NODE(nn.Module):
    def __init__(
        self,
        fn_ode: nn.Module,
        method: str = "rk4",
        atol: float = 0.001,
        rtol: float = 0.001,
        *args: Any,
        **kwargs: Any
    ) -> None:
        super().__init__(*args, **kwargs)
        self.fn_ode = fn_ode
        self.method = method
        self.atol = atol
        self.rtol = rtol
        self.path = []

    def forward(self, x, n_steps=10, end_time=1):
        self.path =[]
        """
        x (tensor): initial point
        n_steps (int): discretisation steps between the two points"""
        device = x.device
        t = torch.tensor(list(np.linspace(0,end_time,n_steps)), device=device).float()
        x = odeint(
            self.fn_ode, x, t, method=self.method, atol=self.atol, rtol=self.rtol
        )
        for time, point in zip(t,x):
            self.path.append(self.fn_ode(time,point))
        return x # NOTE add Sigmoid if we do ode on images.


class ToyODE(nn.Module):
    """ODE derivative network.
    feature_dims (int) default '5': dimension of the inputs, either in ambient space or embedded space.
    layer (list of int) default ''[64]'': the hidden layers of the network.
    activation (torch.nn) default '"ReLU"': activation function applied in between layers.
    scales (NoneType|list of float) default 'None': the initial scale for the noise in the trajectories. One scale per bin, add more if using an adaptative ODE solver.
    n_aug (int) default '1': number of added dimensions to the input of the network. Total dimensions are features_dim + 1 (time) + n_aug.
    Method
    forward (Callable)
        forward pass of the ODE derivative network.
        Parameters:
        t (torch.tensor): time of the evaluation.
        x (torch.tensor): position of the evaluation.
        Return:
        derivative at time t and position x.
    """

    def __init__(
        self,
        feature_dims: int = 5,
        layers: List[int] = [64],
        activation: str = "ReLU",
        n_aug: int = 0,
        batch_norm: bool = False,
    ):
        super().__init__()
        steps = [
            feature_dims + n_aug + 1,
            *layers,
            feature_dims,
        ]  # NOTE added n_aug in the last layer.
        pairs = zip(steps, steps[1:])

        chain = list(
            itertools.chain(
                *list(
                    zip(
                        map(lambda e: nn.Linear(*e), pairs),
                        itertools.repeat(getattr(nn, activation)()),
                    )
                )
            )
        )[:-1]
        if batch_norm:
            chain.insert(0, nn.BatchNorm1d(feature_dims + n_aug + 1))

        self.chain = chain
        self.seq = nn.Sequential(*chain)
        self.n_aug = n_aug
        self.path = []

    def reset_path(self):
        self.path = []

    def forward(self, t, x):
        # TODO: augmented dimensions.
        # NOTE the forward pass when we use torchdiffeq must be forward(self,t,x)
        # zero = torch.tensor([0]).to(x.device)
        # zeros = zero.repeat(x.size()[0],self.n_aug)
        time = t.repeat(x.size()[0], 1)
        aug = torch.cat((x, time), dim=1)
        x = self.seq(aug)
        # if self.alpha is not None:
        #     z = torch.randn(x.size(),requires_grad=False).to(x.device)
        # dxdt = x + z*self.alpha[int(t-1)] if self.alpha is not None else x
        return x
