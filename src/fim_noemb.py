import torch
import numpy as np


def torch_phate(X, kernel, bandwidth, t):

    """logarithm of the PHATE transition matrix."""
    dists = torch.norm(X[:, None] - X, dim=2, p="fro")

    def gaussian_kernel(x):
        return torch.exp(-(dists**2) / bandwidth)

    kernel = gaussian_kernel(dists)
    p = kernel / kernel.sum(axis=0)[:, None]
    pt = torch.matrix_power(p, t)
    log_p = torch.log(pt)
    return log_p


class FIM_noemb:
    """Fisher information matrix from a dataset. Using PHATE embedding with a Gaussian kernel."""

    def __init__(self, kernel="gaussian", bandwidth=10, t=10) -> None:
        self.kernel = kernel
        self.bandwidth = bandwidth
        self.t = t

    def fit(self, X):

        """logarithm of the PHATE transition matrix, and computation of FIM"""
        n_obs, n_dim = X.shape

        # Jacobian
        self.log_p = torch_phate(
            X, kernel=self.kernel, bandwidth=self.bandwidth, t=self.t
        )
        fn = lambda x: torch_phate(
            x, kernel=self.kernel, bandwidth=self.bandwidth, t=self.t
        )
        J = torch.autograd.functional.jacobian(fn, X)

        jac = torch.empty((n_obs, n_obs, n_dim))
        for i in range(n_obs):
            jac[i] = J[i][i]

        # FIM
        prod = torch.ones((n_obs, n_dim, n_dim, n_obs))
        for i in range(n_dim):
            for j in range(i, n_dim):
                prod[:, i, j, :] = prod[:, j, i, :] = (
                    jac[:, :, i] * jac[:, :, j] * torch.exp(self.log_p)
                )

        fim = torch.sum(prod, dim=3)

        return fim

    def get_volume(self, X):
        fim = self.fit(X)
        V = np.sqrt(np.linalg.det(fim.detach().cpu()))
        return V
    
class FIM:
    def __init__(self,X,fn,n_obs,in_dims,out_dims,X_out):
        self.X = X 
        self.fn = fn
        self.n_obs = n_obs
        self.in_dims = in_dims
        self.out_dims = out_dims
        self.X_out = X_out
        
    def fit(self):
        "Computes Fisher information metric"
        
        #Initializae Jacobian matrix
        Jacob  = np.zeros((self.n_obs,self.out_dims,self.in_dims))
        
        #Get Jacobian of each sample
        for i in range(self.n_obs):
            X_sample = torch.unsqueeze(self.X[i].float().cuda(),0)    
            J = torch.autograd.functional.jacobian(self.fn,X_sample).squeeze()
            Jacob[i,:,:] = J.cpu().detach().numpy()

        #Get FIM of each sample
        FIMetric = np.zeros((self.n_obs,self.in_dims,self.in_dims)) #FIM is square matrix of size of original dimensions
        for k in range(self.n_obs):
            prod = np.empty((self.in_dims,self.in_dims)) #Initialize empty FIM

            #Compute FIM
            for i in range(self.in_dims):
                for j in range(self.in_dims):
                    prod[i,j] = np.sum(Jacob[k,:,i] * Jacob[k,:,j] *np.exp(self.X_out[k,:]))

            FIMetric[k,:,:] = prod
            
        return FIMetric, Jacob

    
    def get_volume(self):
        "Computes Volume"
        fim, _ = self.fit()
        V = np.sqrt(np.abs(np.linalg.det(fim)))
        return V
    
    def get_eigs(self):
        
        "Eigendecomposition of FIM"
        fim, _ = self.fit()
        FIMeigvec = np.zeros((self.n_obs,self.in_dims,self.in_dims))
        FIMeigval = np.zeros((self.n_obs,self.in_dims))
        for i in range(self.n_obs):
            eigval, eigvec = np.linalg.eig(fim[i])
            FIMeigvec[i,:,:] = eigvec
            FIMeigval[i,:] = eigval
        return FIMeigval, FIMeigvec
    
  
    def get_quadform(self,vone):
        "Computes Quadratic form with input vector of size (1,self.in_dims)"
        
        fim, _ = self.fit()
        quadforms = np.zeros((self.n_obs))
        for i in range(self.n_obs):
            quadforms[i] = vone @ fim[i,:,:] @ vone.T
        
        return quadforms
    
class FIM_torch:
    def __init__(self,X,fn,n_obs,in_dims,out_dims,X_out):
        self.X = X
        self.fn = fn
        self.n_obs = n_obs
        self.in_dims = in_dims
        self.out_dims = out_dims
        self.X_out = X_out
        
    def fit(self):
        "Computes Fisher information metric"
        
        #Initializae Jacobian matrix
        Jacob  = torch.zeros((self.n_obs,self.out_dims,self.in_dims)).cuda()
        
        #Get Jacobian of each sample
        for i in range(self.n_obs):
            X_sample = torch.unsqueeze(self.X[i].float().cuda(),0)    
            J = torch.autograd.functional.jacobian(self.fn,X_sample).squeeze()
            Jacob[i,:,:] = J

        #Get FIM of each sample
        FIMetric = torch.zeros((self.n_obs,self.in_dims,self.in_dims)).cuda() #FIM is square matrix of size of original dimensions
        for k in range(self.n_obs):
            prod = torch.empty((self.in_dims,self.in_dims)).cuda() #Initialize empty FIM

            #Compute FIM
            for i in range(self.in_dims):
                for j in range(self.in_dims):
                    prod[i,j] = torch.sum(Jacob[k,:,i] * Jacob[k,:,j] *torch.exp(self.X_out[k,:].float().cuda()))

            FIMetric[k,:,:] = prod
            
        return FIMetric
    
class FIM_cpu:
    def __init__(self,X,fn,n_obs,in_dims,out_dims,X_out):
        self.X = X
        self.fn = fn
        self.n_obs = n_obs
        self.in_dims = in_dims
        self.out_dims = out_dims
        self.X_out = X_out
        
    def fit(self):
        "Computes Fisher information metric"
        
        #Initializae Jacobian matrix
        Jacob  = torch.zeros((self.n_obs,self.out_dims,self.in_dims)).cpu()
        
        #Get Jacobian of each sample
        for i in range(self.n_obs):
            X_sample = torch.unsqueeze(self.X[i].float().cpu(),0)    
            J = torch.autograd.functional.jacobian(self.fn,X_sample).squeeze()
            Jacob[i,:,:] = J

        #Get FIM of each sample
        FIMetric = torch.zeros((self.n_obs,self.in_dims,self.in_dims)).cpu() #FIM is square matrix of size of original dimensions
        for k in range(self.n_obs):
            prod = torch.empty((self.in_dims,self.in_dims)).cpu() #Initialize empty FIM

            #Compute FIM
            for i in range(self.in_dims):
                for j in range(self.in_dims):
                    prod[i,j] = torch.sum(Jacob[k,:,i] * Jacob[k,:,j] *torch.exp(self.X_out[k,:].float().cpu()))

            FIMetric[k,:,:] = prod
            
        return FIMetric
    
    def get_volume(self):
        "Computes Volume"
        fim, _ = self.fit()
        V = np.sqrt(np.abs(np.linalg.det(fim)))
        return V
    
    def get_eigs(self):
        
        "Eigendecomposition of FIM"
        fim, _ = self.fit()
        FIMeigvec = np.zeros((self.n_obs,self.in_dims,self.in_dims))
        FIMeigval = np.zeros((self.n_obs,self.in_dims))
        for i in range(self.n_obs):
            eigval, eigvec = np.linalg.eig(fim[i])
            FIMeigvec[i,:,:] = eigvec
            FIMeigval[i,:] = eigval
        return FIMeigval, FIMeigvec
    
  
    def get_quadform(self,vone):
        "Computes Quadratic form with input vector of size (1,self.in_dims)"
        
        fim, _ = self.fit()
        quadforms = np.zeros((self.n_obs))
        for i in range(self.n_obs):
            quadforms[i] = vone @ fim[i,:,:] @ vone.T
        
        return quadforms