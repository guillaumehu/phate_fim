# Loss functions for one training/eval batch.

import phate
import torch
import sys

"""

def computeJSD(X):
    
    #NOTE: Input for this function is matrix where the rows are log-transformed probabilites

    jsd = torch.zeros((X.shape[0],X.shape[0]))
    for i in range(X.shape[0]):

        p = X[i,:]
        q = X
        m = (0.5 * (p.exp() + q.exp()))
        

        kone = 0.5 *( (p.exp() * (p - m.log() ) ).sum(dim=1))
        ktwo = 0.5 *( (q.exp() * (q - m.log() ) ).sum(dim=1))
        k = kone + ktwo

        jsd[i,:] = k
     
    jsd.fill_diagonal_(0) #setting diagonal to 0 to avoid nan's
            
    return jsd.to('cuda')
    
"""

def computeJSD(X):
    
    #NOTE: Input for this function is matrix where rows are probabilites

    jsd = torch.zeros((X.shape[0],X.shape[0]))
    for i in range(X.shape[0]):

        p = X[i,:]
        q = X
        m = (0.5 * (p + q))
        

        kone = 0.5 *( (p * (p.log() - m.log() ) ).sum(dim=1))
        ktwo = 0.5 *( (q * (q.log() - m.log() ) ).sum(dim=1))
        k = kone + ktwo

        jsd[i,:] = k
     
    jsd.fill_diagonal_(0) #setting diagonal to 0 to avoid nan's
            
    return jsd.to('cuda')


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
            diff_op = torch.tensor(phate_op.diff_op).float().to(sample.device) 

        elif kernel_type.lower() == "gaussian":
            dists = torch.norm(sample[:, None] - sample, dim=2, p="fro")
            kernel = torch.exp(-(dists**2) / bandwidth)
            p = kernel / kernel.sum(axis=0)[:, None]
            pt = torch.matrix_power(p, t)
            diff_pot = torch.log(pt)
                        
        elif kernel_type.lower() == "ipsc_phate":
            _, dim = encoded_sample.shape
            sample_np = sample.detach().cpu().numpy()
            phate_op = phate.PHATE(verbose=False, n_components=dim, knn=knn,t=250,decay=10).fit(
                sample_np
            )
            diff_pot = torch.tensor(phate_op.diff_potential).float().to(sample.device)
            diff_op = torch.tensor(phate_op.diff_op).float().to(sample.device) 
            
        elif kernel_type.lower() == "pbmc_phate":
            _, dim = encoded_sample.shape
            sample_np = sample.detach().cpu().numpy()
            phate_op = phate.PHATE(verbose=False, n_components=dim, knn=knn).fit(
                sample_np
            )
            diff_pot = torch.tensor(phate_op.diff_potential).float().to(sample.device)
            diff_op = torch.tensor(phate_op.diff_op).float().to(sample.device) 
        

        #phate_dist = torch.cdist(diff_pot, diff_pot)
        #encoded_dist = torch.cdist(encoded_sample, encoded_sample)
        #loss_d = torch.nn.MSELoss()(encoded_dist, phate_dist)
        
        #JSD loss
        phate_dist = torch.sqrt( torch.abs(computeJSD(diff_op + 1e-7)) )       
        encoded_dist = torch.sqrt( torch.abs(computeJSD(encoded_sample + 1e-7)) )
        loss_d = torch.nn.MSELoss()(encoded_dist, phate_dist)

        

    if loss_emb:
        loss_e = torch.nn.MSELoss()(encoded_sample, target)
    if loss_recon:
        loss_r = torch.nn.MSELoss()(decoded_sample, sample)
        

        
     
        
    return loss_d, loss_e, loss_r

