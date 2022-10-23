import sys
from tabnanny import verbose
import numpy as np
import phate
import torch
from torch.utils.data import Dataset


def make_n_sphere(n_obs=150, dim=3, train_dataset=False):
    """Make an N-sphere with Muller's method. return a Tensor `requires_grad=True`."""
    norm = np.random.normal
    normal_deviates = norm(size=(dim, n_obs))
    radius = np.sqrt((normal_deviates**2).sum(axis=0))
    X = (normal_deviates / radius).T
    if train_dataset:
        phate_sphere = None
    else:
        phate_operator = phate.PHATE(random_state=42, verbose=False)
        phate_sphere = phate_operator.fit_transform(X)

    return phate_sphere, torch.tensor(X, requires_grad=True).float()


def make_tree(n_obs=150, dim=10, train_dataset=False):
    """Make a tree dataset. Return a Tensor `requires_grad=True` and tree_phate"""
    tree_data, _ = phate.tree.gen_dla(n_dim=dim, n_branch=5, branch_length=30)
    if train_dataset:
        tree_phate = None
    else:
        phate_operator = phate.PHATE(random_state=42, verbose=False)
        tree_phate = phate_operator.fit_transform(tree_data)

    return tree_phate, torch.tensor(tree_data, requires_grad=True).float()


class torch_dataset(Dataset):
    def __init__(self, X) -> None:
        self.X = X

    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, index):
        _ = 0
        sample = self.X[index, :]
        return sample, _


def train_dataloader(name, n_obs, dim, batch_size):
    """Create a Torch data loader for training."""

    # TODO: add warning if `name` is not implemented.
    if name.lower() == "sphere":
        _, X = make_n_sphere(n_obs, dim, train_dataset=True)
        train_dataset = torch_dataset(X)
        train_loader = torch.utils.data.DataLoader(
            dataset=train_dataset, batch_size=batch_size, shuffle=True
        )

    elif name.lower() == "tree":
        _, X = make_tree(n_obs, dim, train_dataset=True)
        train_dataset = torch_dataset(X)
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
