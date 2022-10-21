# # -*- coding: utf-8 -*-
# from tabnanny import verbose
# import click
# import logging
# from pathlib import Path
# from dotenv import find_dotenv, load_dotenv
from tabnanny import verbose
import numpy as np
import phate
import torch


# @click.command()
# @click.argument('input_filepath', type=click.Path(exists=True))
# @click.argument('output_filepath', type=click.Path())
# def main(input_filepath, output_filepath):
#     """ Runs data processing scripts to turn raw data from (../raw) into
#         cleaned data ready to be analyzed (saved in ../processed).
#     """
#     logger = logging.getLogger(__name__)
#     logger.info('making final data set from raw data')


def make_n_sphere(n_obs=150, dim=3):
    """Make an N-sphere with Muller's method. return a Tensor `requires_grad=True`."""
    norm = np.random.normal
    normal_deviates = norm(size=(dim, n_obs))
    radius = np.sqrt((normal_deviates**2).sum(axis=0))
    X = (normal_deviates / radius).T

    phate_operator = phate.PHATE(random_state=42, verbose=False)
    phate_sphere = phate_operator.fit_transform(X)

    return phate_sphere, torch.tensor(X, requires_grad=True)


def make_tree(n_obs=150, dim=10):
    """Make a tree dataset. Return a Tensor `requires_grad=True` and tree_phate"""
    tree_data, _ = phate.tree.gen_dla(n_dim=dim, n_branch=5, branch_length=30)
    phate_operator = phate.PHATE(random_state=42, verbose=False)
    tree_phate = phate_operator.fit_transform(tree_data)

    return tree_phate, torch.tensor(tree_data, requires_grad=True)


# if __name__ == '__main__':
#     log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
#     logging.basicConfig(level=logging.INFO, format=log_fmt)

#     # not used in this stub but often useful for finding various files
#     project_dir = Path(__file__).resolve().parents[2]

#     # find .env automagically by walking up directories until it's found, then
#     # load up the .env entries as environment variables
#     load_dotenv(find_dotenv())

#     main()
