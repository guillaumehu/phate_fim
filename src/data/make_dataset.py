import sys
import os
from tabnanny import verbose
import numpy as np
import phate
import torch
from torch.utils.data import Dataset
import scipy
import scanpy as sc

def rotation_transform(
    X : np.ndarray, # The input matrix, of size n x d (d is # dimensions)
    tilt_angles # a list of d-1 values in [0,2pi] specifying how much to tilt in d-1 the xy, yz (...) planes
):
    # Tilt matrix into arbitrary dimensions
    d = X.shape[1]
    assert len(tilt_angles) == d - 1
    # construct Tilting Matrices TM!
    tilting_matrices_tm = []
    for i in range(d-1):
        A = np.eye(d)
        A[i][i] = np.cos(tilt_angles[i])
        A[i+1][i+1] = np.cos(tilt_angles[i])
        A[i][i+1] = np.sin(tilt_angles[i])
        A[i+1][i] = - np.sin(tilt_angles[i])
        tilting_matrices_tm.append(A)
    X_tilted = X
    for tilter in tilting_matrices_tm:
        # print(X_tilted)
        X_tilted = X_tilted @ tilter
    return X_tilted, tilting_matrices_tm


def make_live_seq(PATH, emb_dim=20, knn=5, label=False):
    adata_liveseq = sc.read_h5ad(os.path.join(PATH,"Liveseq.h5ad"))
    #adata_rnaseq = sc.read_h5ad(os.path.join(PATH,"scRNA.h5ad"))
    X = adata_liveseq.X
    phate_operator = phate.PHATE(random_state=42, verbose=False, n_components=emb_dim, knn=knn)
    phate_live_seq = phate_operator.fit_transform(X)
    phate_live_seq = scipy.stats.zscore(phate_live_seq) 
    if label:
        return torch.tensor(X, requires_grad=True).float(), phate_live_seq, adata_liveseq.obs['celltype_treatment']
    else:
        return torch.tensor(X, requires_grad=True).float(), phate_live_seq


def make_n_sphere(n_obs=150, dim=3, emb_dim=2, knn=5):
    """Make an N-sphere with Muller's method. return a Tensor `requires_grad=True`."""
    norm = np.random.normal
    normal_deviates = norm(size=(dim, n_obs))
    radius = np.sqrt((normal_deviates**2).sum(axis=0))
    X = (normal_deviates / radius).T
    # if train_dataset:
    #     phate_sphere = None
    # else:
    phate_operator = phate.PHATE(random_state=42, verbose=False, n_components=emb_dim, knn=knn)
    phate_sphere = phate_operator.fit_transform(X)
    phate_sphere = scipy.stats.zscore(phate_sphere) 

    return torch.tensor(X, requires_grad=True).float(), phate_sphere



def make_n_sphere_two(n_obs=150, dim=10,emb_dim=2,knn=5):
    
    sphere = [] #Create sphere in 3D
    for i in range(n_obs):
        x = np.random.normal(0,1,3)
        sphere.append(x/(np.sqrt(np.sum(x**2))))

    nsphere = np.array(sphere)
    zerovec = np.zeros((n_obs,dim-3)) #add vector of zeros onto first 3 dimensions
    highdsphere = np.concatenate((nsphere,zerovec),axis=1) #Create high-d sphere
    
    angles = list(np.repeat(90,highdsphere.shape[1]-1)) #can insert angle you wish to rotate sphere by
    rotatesphere, _ = rotation_transform(highdsphere, angles)
    
    #run phate on rotated sphere
    phate_operator = phate.PHATE(random_state=42, verbose=False, n_components=emb_dim, knn=knn)
    phate_sphere_rot = phate_operator.fit_transform(rotatesphere)
    phate_sphere_rot = scipy.stats.zscore(phate_sphere_rot)
    
    return torch.tensor(rotatesphere, requires_grad=True).float(), phate_sphere_rot


def make_tree(n_obs=150, dim=10, emb_dim=2, knn=5):
    """Make a tree dataset. Return a Tensor `requires_grad=True` and tree_phate"""
    n_obs = int(n_obs/5)
    tree_data, _ = phate.tree.gen_dla(n_dim=dim, n_branch=5, branch_length=n_obs)
    # if train_dataset:
    #     tree_phate = None
    # else:
    phate_operator = phate.PHATE(random_state=42, verbose=False, n_components=emb_dim, knn=knn)
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


def train_dataloader(name, n_obs, dim, emb_dim, batch_size, knn, PATH=None):
    """Create a Torch data loader for training."""

    # TODO: add warning if `name` is not implemented.
    if name.lower() == "sphere":
        X, Y = make_n_sphere_two(n_obs, dim, emb_dim, knn)
        Y = torch.tensor(Y).float()
        train_dataset = torch_dataset(X, Y)
        train_loader = torch.utils.data.DataLoader(
            dataset=train_dataset, batch_size=batch_size, shuffle=True
        )

    elif name.lower() == "tree":
        X, Y = make_tree(n_obs, dim, emb_dim, knn)
        Y = torch.tensor(Y).float()
        train_dataset = torch_dataset(X, Y)
        train_loader = torch.utils.data.DataLoader(
            dataset=train_dataset, batch_size=batch_size, shuffle=True
        )

    elif name.lower() == "live_seq":
        X, Y = make_live_seq(PATH, emb_dim, knn, label=False)
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
