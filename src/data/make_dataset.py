import sys
import os
from tabnanny import verbose
import numpy as np
import phate
import torch
from torch.utils.data import Dataset
import scipy
import scanpy as sc


def make_live_seq(PATH, emb_dim=20, label=False):
    adata_liveseq = sc.read_h5ad(os.path.join(PATH,"Liveseq.h5ad"))
    #adata_rnaseq = sc.read_h5ad(os.path.join(PATH,"scRNA.h5ad"))
    X = adata_liveseq.X
    phate_operator = phate.PHATE(random_state=42, verbose=False, n_components=emb_dim)
    phate_live_seq = phate_operator.fit_transform(X)
    
    if label:
        return torch.tensor(X, requires_grad=True).float(), phate_live_seq, adata_liveseq.obs['celltype_treatment']
    else:
        return torch.tensor(X, requires_grad=True).float(), phate_live_seq


def make_n_sphere(n_obs=150, dim=3, emb_dim=2):
    """Make an N-sphere with Muller's method. return a Tensor `requires_grad=True`."""
    norm = np.random.normal
    normal_deviates = norm(size=(dim, n_obs))
    radius = np.sqrt((normal_deviates**2).sum(axis=0))
    X = (normal_deviates / radius).T
    # if train_dataset:
    #     phate_sphere = None
    # else:
    phate_operator = phate.PHATE(random_state=42, verbose=False, n_components=emb_dim)
    phate_sphere = phate_operator.fit_transform(X)
    phate_sphere = scipy.stats.zscore(phate_sphere) 

    return torch.tensor(X, requires_grad=True).float(), phate_sphere


def make_tree(n_obs=150, dim=10, emb_dim=2):
    """Make a tree dataset. Return a Tensor `requires_grad=True` and tree_phate"""
    tree_data, _ = phate.tree.gen_dla(n_dim=dim, n_branch=5, branch_length=30)
    # if train_dataset:
    #     tree_phate = None
    # else:
    phate_operator = phate.PHATE(random_state=42, verbose=False, n_components=emb_dim)
    tree_phate = phate_operator.fit_transform(tree_data)
    tree_phate = scipy.stats.zscore(tree_phate) 

    return torch.tensor(tree_data, requires_grad=True).float(), tree_phate


class torch_dataset(Dataset):
    def __init__(self, X, Y) -> None:
        self.X = X
        self.Y = Y
    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, index):
        target = self.Y[index, :]
        sample = self.X[index, :]
        return sample, target


def train_dataloader(name, n_obs, dim, emb_dim, batch_size, PATH=None):
    """Create a Torch data loader for training."""

    # TODO: add warning if `name` is not implemented.
    if name.lower() == "sphere":
        X, Y = make_n_sphere(n_obs, dim, emb_dim)
        Y = torch.tensor(Y).float()
        train_dataset = torch_dataset(X, Y)
        train_loader = torch.utils.data.DataLoader(
            dataset=train_dataset, batch_size=batch_size, shuffle=True
        )

    elif name.lower() == "tree":
        X, Y = make_tree(n_obs, dim, emb_dim)
        Y = torch.tensor(Y).float()
        train_dataset = torch_dataset(X, Y)
        train_loader = torch.utils.data.DataLoader(
            dataset=train_dataset, batch_size=batch_size, shuffle=True
        )

    elif name.lower() == "live_seq":
        X, Y = make_live_seq(PATH, emb_dim, label=False)
        Y = torch.tensor(Y).float()
        train_dataset = torch_dataset(X, Y)
        train_loader = torch.utils.data.DataLoader(
            dataset=train_dataset, batch_size=batch_size, shuffle=True
        )

    return train_loader


# if __name__ == '__main__':
#     log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
#     logging.basicConfig(level=logging.INFO, format=log_fmt)

#     # not used in this stub but often useful for finding various files
#     project_dir = Path(__file__).resolve().parents[2]

#     # find .env automagically by walking up directories until it's found, then
#     # load up the .env entries as environment variables
#     load_dotenv(find_dotenv())

#     main()
