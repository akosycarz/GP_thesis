import torch
import random
import gpytorch

from gpytorch.distributions import MultivariateNormal
from gpytorch.kernels import  NewtonGirardAdditiveKernel, RBFKernel
from gpytorch.means import ConstantMean, ZeroMean
from gpytorch.models import ExactGP
import numpy as np


def postprocess_rbf(dist_mat):
    return dist_mat.div_(-2).exp_()

# Base Kernel definitions
class ConstrainedRBFKernel(gpytorch.kernels.Kernel):
    has_lengthscale = True

    def forward(self, x1, x2, diag=False, **params):
        scaling_factor = (self.lengthscale * torch.sqrt(self.lengthscale**2 + 2)).div(self.lengthscale**2 + 1)
        mu = torch.tensor(0.0).view(1,1)
        # x1 = x1.unsqueeze(1)  # Shape: [B1, 1, D], in our case it should be [1 batch, 160 examples, 10 features]
        # x2 = x2.unsqueeze(0)  # Shape: [1, B2, D]
        x1_ = x1.div(self.lengthscale**2)
        x2_ = x2.div(self.lengthscale**2)
        x1_constrain = x1.div(self.lengthscale**2 + 1)
        x2_constrain = x2.div(self.lengthscale**2 + 1)
        constrain = postprocess_rbf(
            self.covar_dist(x1_constrain, mu, square_dist=True, diag=diag, **params)
            + self.covar_dist(
                x2_constrain, mu, square_dist=True, diag=diag, **params
            )
        )
        scaled_constrain = scaling_factor * constrain
        base = postprocess_rbf(self.covar_dist(x1_, x2_, square_dist=True, diag=diag, **params)) - scaled_constrain

        return base

class ConstrainedRBFKernel_NotWorking(gpytorch.kernels.Kernel):
    # has_lengthscale = True
    
    def __init__(self, mu=0, var=1, **kwargs):
        super(ConstrainedRBFKernelNotWorking, self).__init__(**kwargs)
        self.mu = torch.tensor(mu, dtype=torch.float)
        self.var = torch.tensor(var, dtype=torch.float)
        self.register_parameter(name="raw_lengthscale",  parameter= torch.nn.Parameter(torch.tensor(1.0).view(1, 1, 1))) #initialize the lenghthsacel to 1 #torch.nn.Parameter(torch.ones(*self.batch_shape, 1, 1)))
        self.register_constraint("raw_lengthscale", gpytorch.constraints.Positive())

    @property
    def lengthscale(self):
        return self.raw_lengthscale_constraint.transform(self.raw_lengthscale)
 
    @lengthscale.setter
    def lengthscale(self, value):
        # Transform the value using the inverse of the constraint and set it to raw_lengthscale
        if not torch.is_tensor(value):
            value = torch.as_tensor(value).to(self.raw_lengthscale)
        self.initialize(raw_lengthscale=self.raw_lengthscale_constraint.inverse_transform(torch.tensor(value).view(1, 1, 1)))

    def forward(self, x1, x2, diag = False, **params):
        x1_ = x1.unsqueeze(1)  # Shape: [B1, 1, D], in our case it should be [1 batch, 160 examples, 10 features]
        x2_ = x2.unsqueeze(0)  # Shape: [1, B2, D]
        mu_ = self.mu
        
        l = self.lengthscale
        l_sq = l**2#.pow(2)
        variance = self.var#.pow(2)

        # Base RBF kernel calculation
        diff = x1_ - x2_
        dists = torch.sum(diff ** 2, -1)
        base = torch.exp(-0.5 * dists / l_sq)

        # Constraint term calculation
        term1 = torch.sum((x1_ - mu_)**2, -1) + torch.sum((x2_ - mu_)**2, -1)
        scaled_l_sq = l_sq + variance
        constraint = torch.exp(-0.5 * term1 / scaled_l_sq)
        scaling_factor = (l* torch.sqrt(l_sq +2*variance)) / scaled_l_sq

        # Constrained kernel
        constrained_kernel = base - scaling_factor * constraint

#
        return constrained_kernel if not diag else constrained_kernel.diag()

    def evaluate(self, x1, x2=None):
        if x2 is None:
            x2 = x1
        return self.forward(x1, x2)
    


# Creating  AGP model with Newton Girard additive kernel
class AGPModel(ExactGP):
    def __init__(self, train_x, train_y, likelihood):
        super(AGPModel, self).__init__(train_x, train_y, likelihood)
        self.mean_module = ZeroMean()
        self.base_kernel = ConstrainedRBFKernel()  # You can replace this with ConstrainedRBFKernel if desired
        self.covar_module = NewtonGirardAdditiveKernel(base_kernel=self.base_kernel, num_dims=train_x.shape[-1])

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return MultivariateNormal(mean_x, covar_x)
    


# reating the Omega function for the shapley value calculation
def Omega(X, i,sigmas,q_additivity=None):
    
    n, d = X.shape
    if q_additivity is None:
        q_additivity = d
    
    # Reorder columns so that the i-th column is first
    idx = torch.arange(d)
    idx[i] = 0
    idx[0] = i
    X = X[:, idx]

    # Initialize dp array
    dp = torch.zeros((q_additivity, d, n))

    # Initial sum of features across the dataset
    sum_current = torch.zeros((n,))
    
    # Fill the first order dp (base case)
    for j in range(d):
        dp[0, j, :] = X[:, j]
        sum_current += X[:, j]

    # Fill the dp table for higher orders
    for i in range(1, q_additivity):
        temp_sum = torch.zeros((n,))
        for j in range(d):
            # Subtract the previous contribution of this feature when moving to the next order
            sum_current -= dp[i - 1, j, :]

            dp[i, j, :] =  X[:,j] * sum_current
            dp[i, j, :] = dp[i, j, :] * (i/(i+1)) 
            dp[i,j,:] = dp[i,j,:]
            temp_sum += dp[i, j, :]
        
        sum_current = temp_sum
    for i in range(q_additivity):
        dp[i,:,:] = dp[i,:,:] * sigmas[i]
    # Sum the first row of each slice
    omega = torch.sum(dp[:, 0, :], axis=0)

    return omega , dp


# Creating the K_s_isntance (so the k(x*,X), where the x* is a specific instance featuer)
def K_s_instance(X, sigmas,q_additivity=None):
        
    n, d = X.shape
    if q_additivity is None:
        q_additivity = d
    # Initialize dp array
    dp = torch.zeros((q_additivity, d, n))

    # Initial sum of features across the dataset
    sum_current = torch.zeros((n,))
    
    # Fill the first order dp (base case)
    for j in range(d):
        dp[0, j, :] = X[:, j]
        sum_current += X[:, j]

    # Fill the dp table for higher orders
    for i in range(1, q_additivity):
        temp_sum = torch.zeros((n,))
        for j in range(d):
            # Subtract the previous contribution of this feature when moving to the next order
            sum_current -= dp[i - 1, j, :]

            dp[i, j, :] =  X[:,j] * sum_current
            dp[i, j, :] = dp[i, j, :] 
            dp[i,j,:] = dp[i,j,:]
            temp_sum += dp[i, j, :]
        
        sum_current = temp_sum
    for i in range(q_additivity):
        dp[i,:,:] = dp[i,:,:] * sigmas[i]
     # here i would like to some all dimentions d
    result = torch.zeros(n)
    for i in range(d):
        result += torch.sum(dp[:,i,:], axis=0)

    return result, dp

# Compute K_per_feature
def K_per_feature(model, instance_features ,train_x):
    n, d = train_x.shape
    K_per_feature = torch.zeros(n, d)

    with torch.no_grad():
        model.eval()
        
        # # Get lengthscale from model's kernel if it's already defined
        # l = model.covar_module.base_kernel.lengthscale.item()
        
        # Define the ConstrainedRBFKernel

        # constrained_kernel = ConstrainedRBFKernel()
        # constrained_kernel.lengthscale = l
        constrained_kernel = model.covar_module.base_kernel # this one was used to compare if the constrained_kernel form ConstrainedRBFKernel() is set with correct l
        
        # Extract the instance's features; assuming you want the 4th sample (index 3)
        instance_features = instance_features.unsqueeze(0)  # Shape (1, d)

        # Loop over each feature dimension
        for i in range(d):
            # Extract the i-th feature across all samples
            feature_column = train_x[:, i].unsqueeze(1)  # Shape (n, 1)
            instance_feature = instance_features[:, i].unsqueeze(1)  # Shape (1, 1)
        
            # Compute the kernel matrix for the i-th feature
            K_per_feature[:, i] = constrained_kernel.evaluate(instance_feature, feature_column)#.evaluate()
        
    return K_per_feature
