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
from math import pi

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
        self.lenght = 0.0
        # if we want to add a network after the neural ode.
        #self.net = nn.Sequential(nn.Linear(2,40),nn.LeakyReLU(),nn.Linear(40,40),nn.LeakyReLU(),nn.Linear(40,2)) 

    def forward(self, x, n_steps=10, end_time=1):
        self.path = []
        """
        x (tensor): initial point
        n_steps (int): discretisation steps between the two points"""
        device = x.device
        # TODO add one dimension to keep track of the lenght.
        #zero = torch.tensor([[0]]).to(device)
        #x = torch.hstack((x,zero))
        t = torch.tensor(list(np.linspace(0,end_time,n_steps)), device=device).float()
        x = odeint(
            self.fn_ode, x, t, method=self.method, atol=self.atol, rtol=self.rtol
        )

        for time, point in zip(t,x):
            self.path.append(self.fn_ode(time,point))
        #TODO: change the last activation function.
        #x = torch.tanh(torch.tensor([0.2])*x)
        #return torch.tensor([pi/2,pi])*x + torch.tensor([pi/2,pi])
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
        #metric,
        feature_dims: int = 5,
        layers: List[int] = [64],
        activation: str = "ReLU",
        n_aug: int = 0,
        metric = lambda x: 0.0*torch.rand((x.shape[0],x.shape[1],x.shape[1])),
        sqrt_metric=False
    ):
        super().__init__()
        steps = [
            feature_dims + n_aug + 1, # +1 for time 
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

        self.chain = chain
        self.seq = nn.Sequential(*chain)
        self.n_aug = n_aug
        self.path = []
        self.metric = metric
        self.cost = 0.0
        self.sqrt_metric=sqrt_metric

    def reset_cost(self):
        self.cost = 0.0

    def forward(self, t, x):
        # last col of x is the metric loss, i.e. x[:,-1].
        # NOTE the forward pass when we use torchdiffeq must be forward(self,t,x)
        position = x[:,:-1] 
        time = t.repeat(x.size()[0], 1)
        aug = torch.cat((position, time), dim=1)
        x = self.seq(aug)

        # current cost

        c = torch.matmul(torch.matmul(x.unsqueeze(1),self.metric(position)),x.unsqueeze(1).transpose(1,2)).squeeze(-1)
        c = torch.sqrt(c) if self.sqrt_metric else c 
        self.cost += torch.mean(c)
        x = torch.hstack((x,c))
        return x 


