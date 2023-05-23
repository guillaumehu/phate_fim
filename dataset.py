from scipy.stats import special_ortho_group
from sklearn.metrics import pairwise_distances

import graphtools
import itertools
import numpy as np
import pygsp
import sklearn.datasets as skd

# Modified https://github.com/atong01/MultiscaleEMD and https://homepages.ecs.vuw.ac.nz/~marslast/Code/Ch6/lle.py


class Dataset:
    """Dataset class for Optimal Transport."""

    def __init__(self):
        super().__init__()
        self.X = None
        self.labels = None
        self.graph = None

    def get_labels(self):
        return self.labels

    def get_data(self):
        return self.X

    def standardize_data(self):
        """Standardize data putting it in a unit box around the origin.
        This is necessary for quadtree type algorithms
        """
        X = self.X
        minx = np.min(self.X, axis=0)
        maxx = np.max(self.X, axis=0)
        self.std_X = (X - minx) / (maxx - minx)
        return self.std_X

    def rotate_to_dim(self, dim):
        """Rotate dataset to a different dimensionality."""
        self.rot_mat = special_ortho_group.rvs(dim)[: self.X.shape[1]]
        self.high_X = np.dot(self.X, self.rot_mat)
        return self.high_X


class SwissRoll(Dataset):
    def __init__(
        self,
        n_points=100,
        manifold_noise=0.05,
        width=1,
        random_state=42,
        rotate=False,
        rotate_dim=None,
    ):
        super().__init__()
        # rng = np.random.default_rng(random_state)

        # self.mean_t = (
        #     t_scale * np.pi * (1 + 2 * rng.uniform(size=(1, n_distributions)))
        # )  # NOTE: ground truth coordinate  euclidean in (y,t) is geo on 3d
        # self.mean_y = width * rng.uniform(size=(1, n_distributions))
        # t_noise = (
        #     manifold_noise
        #     # * 3
        #     * rng.normal(size=(n_distributions, n_points_per_distribution))
        # )
        # y_noise = (
        #     manifold_noise
        #     # * 7
        #     * rng.normal(size=(n_distributions, n_points_per_distribution))
        # )
        # ts = np.reshape(t_noise + self.mean_t.T, -1)
        # ys = np.reshape(y_noise + self.mean_y.T, -1)
        # xs = ts * np.cos(ts)
        # zs = ts * np.sin(ts)
        # X = np.stack((xs, ys, zs))
        # X += noise * rng.normal(size=(3, n_distributions * n_points_per_distribution))
        # self.X = X.T
        # self.ts = np.squeeze(ts)
        # self.labels = np.repeat(
        #     np.eye(n_distributions), n_points_per_distribution, axis=0
        # )
        # self.t = self.mean_t[0]
        # mean_x = self.mean_t * np.cos(self.mean_t)
        # mean_z = self.mean_t * np.sin(self.mean_t)
        # self.means = np.concatenate((mean_x, self.mean_y, mean_z)).T
        np.random.seed(random_state)
        t = 3 * np.pi / 2 * (1 + 2 * np.random.rand(1, n_points))
        h = width * np.random.rand(1, n_points)
        data = np.concatenate(
            (t * np.cos(t), h, t * np.sin(t))
        ) + manifold_noise * np.random.randn(3, n_points)
        self.X = data.T
        if rotate and rotate_dim is not None:
            self.X = self.rotate_to_dim(rotate_dim)
        self.t = t
        self.h = h

    def get_graph(self):
        """Create a graphtools graph if does not exist."""
        if self.graph is None:
            self.graph = graphtools.Graph(self.X, use_pygsp=True)
        return self.graph

    def get_geodesic(self):
        # true_coords = np.stack([self.means[:, 1], self.t / 10], axis=1)
        true_coords = np.concatenate((self.t, self.h)).T
        geodesic_dist = pairwise_distances(true_coords, metric="euclidean")
        return geodesic_dist
