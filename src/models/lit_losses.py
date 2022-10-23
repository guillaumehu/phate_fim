# Loss functions for one training epoch.

import phate
import torch

def loss_dist(encode_sample, sample, kernel_type="phate", loss_emb=True, t=1, bandwidth=5):
    """"Compute the distance loss, either using the Gaussian kernel or PHATE's alpha-decay."""
    loss_e = torch.tensor(0.0).float().to(sample.device)
    if kernel_type.lower() == "phate":
        _, dim = encode_sample.shape
        sample_np = sample.detach().cpu().numpy()
        phate_op = phate.PHATE(n_components=dim, verbose=False, n_landmark=10000).fit(sample_np)
        phate_dist = torch.tensor(phate_op.diff_potential).float().to(sample.device)
        if loss_emb:
            emb = torch.tensor(phate_op.transform(sample_np)).float().to(sample.device)
            loss_e = torch.nn.MSELoss()(encode_sample,emb)
    
    elif kernel_type.lower() == "gaussian":
        dists = torch.norm(sample[:, None] - sample, dim=2, p="fro")
        kernel = torch.exp(-(dists**2) / bandwidth)
        p = kernel / kernel.sum(axis=0)[:, None]
        pt = torch.matrix_power(p, t)
        diff_pot = torch.log(pt)
        phate_dist = torch.cdist(diff_pot, diff_pot) 
    
    encode_dist = torch.cdist(encode_sample, encode_sample) 
    loss_d = torch.nn.MSELoss()(encode_dist,phate_dist)
    return loss_d, loss_e

def phate_loss(encode_sample, sample, loss_emb=False):
    _, dim = encode_sample.shape

    sample_np = sample.detach().cpu().numpy()
    phate_op = phate.PHATE(n_components=dim, verbose=False, n_landmark=10000)
    diff_pot = torch.tensor(phate_op.fit(sample_np).diff_potential).float().to(sample.device)

    encode_dist = torch.cdist(encode_sample, encode_sample) 
    phate_dist = torch.cdist(diff_pot, diff_pot) 
    loss_d = torch.nn.MSELoss()(encode_dist,phate_dist)

    loss_e = torch.tensor(0.0, requires_grad=True).float().to(sample.device)
    if loss_emb:
        emb = torch.tensor(phate_op.transform(sample_np)).float().to(sample.device)
        loss_e = torch.nn.MSELoss()(encode_sample,emb)

    return loss_d, loss_e


