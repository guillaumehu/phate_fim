# Loss functions for one training/eval batch.

import phate
import torch


def loss_fn(
    encoded_sample, 
    decoded_sample,
    sample,
    target,
    kernel_type="phate",
    loss_emb=True,
    loss_dist=True,
    loss_recon=True,
    bandwidth=10,
    t=1,
    knn=5,
):
    """ "Compute the distance loss, either using the Gaussian kernel or PHATE's alpha-decay."""
    loss_e = torch.tensor(0.0).float().to(sample.device)
    loss_d = torch.tensor(0.0).float().to(sample.device)
    loss_r = torch.tensor(0.0).float().to(sample.device)
    if loss_dist:
        if kernel_type.lower() == "phate":
            _, dim = encoded_sample.shape
            sample_np = sample.detach().cpu().numpy()
            phate_op = phate.PHATE(n_components=dim, verbose=False, n_pca=19, knn=knn).fit(
                sample_np
            )
            diff_pot = torch.tensor(phate_op.diff_potential).float().to(sample.device)

        elif kernel_type.lower() == "gaussian":
            dists = torch.norm(sample[:, None] - sample, dim=2, p="fro")
            kernel = torch.exp(-(dists**2) / bandwidth)
            p = kernel / kernel.sum(axis=0)[:, None]
            pt = torch.matrix_power(p, t)
            diff_pot = torch.log(pt)

        phate_dist = torch.cdist(diff_pot, diff_pot)
        encoded_dist = torch.cdist(encoded_sample, encoded_sample)
        loss_d = torch.nn.MSELoss()(encoded_dist, phate_dist)

    if loss_emb:
        loss_e = torch.nn.MSELoss()(encoded_sample, target)
    if loss_recon:
        loss_r = torch.nn.MSELoss()(decoded_sample, sample)
        
    return loss_d, loss_e, loss_r

