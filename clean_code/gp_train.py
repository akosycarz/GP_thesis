import torch
import random
import gpytorch

from gpytorch.distributions import MultivariateNormal
from gpytorch.kernels import  NewtonGirardAdditiveKernel, RBFKernel
from gpytorch.likelihoods import GaussianLikelihood
from gpytorch.mlls import ExactMarginalLogLikelihood
from sklearn.preprocessing import StandardScaler
import numpy as np
from sklearn.model_selection import train_test_split

from gp_functions import AGPModel, ConstrainedRBFKernel, K_s_instance, Omega, K_per_feature

SEED = 8
np.random.seed(SEED)
torch.manual_seed(SEED)
random.seed(SEED)

# Generate Normlally distributed data
# Define the number of samples
num_samples = 200


# Generate features X1 to X10 from N(0,1)
X = np.random.normal(0, 1, (num_samples, 10))
y = np.sin(X[:, 0]**2) + 2 * X[:, 0] + X[:, 1] + np.exp(X[:, 0]**2 * X[:, 1])

# Split the data into training and testing sets
train_x, test_x, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42) # 20% data as test

# Initialize the scaler for features and target
scaler_x = StandardScaler()
# Fit and transform the training data
train_x_scaled = scaler_x.fit_transform(train_x)
# Transform the test data
test_x_scaled = scaler_x.transform(test_x)

# Convert to torch tensors
train_x = torch.tensor(train_x_scaled, dtype=torch.float32)
test_x = torch.tensor(test_x_scaled, dtype=torch.float32)
y_train = torch.tensor(y_train, dtype=torch.float32)
y_test = torch.tensor(y_test, dtype=torch.float32)

likelihood = GaussianLikelihood()
model = AGPModel(train_x, y_train, likelihood)

# Training procedure
model.train()
likelihood.train()

optimizer = torch.optim.Adam(model.parameters(), lr=0.1)
mll = ExactMarginalLogLikelihood(likelihood, model)

for i in range(1000):
    optimizer.zero_grad()
    output = model(train_x)
    with gpytorch.settings.cholesky_jitter(1e-4*2):
        loss = -mll(output, y_train)
    loss.backward()
    optimizer.step()
    print(f'Iteration {i+1}/1000 - Loss: {loss.item()}')

model.eval()
likelihood.eval()
for param_name, param in model.named_parameters():
    print(f'Parameter name: {param_name:42} value = {param.data}')

# Making prediction

# creaitng the base kernel with the lengthscale after the optimalization
trained_kernel = model.covar_module.base_kernel

outputscale = model.covar_module.outputscale # save the optimized outputscales
kernel = NewtonGirardAdditiveKernel(base_kernel=trained_kernel, num_dims=train_x.shape[-1])
kernel.outputscale = outputscale

noise_variance = likelihood.noise.data
# Compute training and test kernel matrices

with torch.no_grad():
    model.eval()

    likelihood.eval()
    K_train = kernel(train_x, train_x).evaluate() + noise_variance * torch.eye(train_x.size(0))
    K_s = kernel(test_x,train_x).evaluate()
    K_ss = kernel(test_x, test_x).evaluate() 
    K_inv =  torch.inverse(K_train)
    alpha = torch.matmul(K_inv, y_train) #this is the alpha used for all calcualtions
    predicted_mean =torch.matmul(K_s,alpha)
    v = torch.linalg.solve_triangular(torch.linalg.cholesky(K_train), K_s.t(), upper=False)
    predicted_covariance_matrix = K_ss - v.t().matmul(v)


# Start of Shapley value calculations
instance_features = train_x[3] # Shape (1, d)
# K_per_feature
K_per_feature = K_per_feature(model, instance_features ,train_x)
print(K_per_feature)

n,d = train_x.shape
sigmas = outputscale.data.unsqueeze(-1)
val = torch.zeros(d)
for i in range(d):
    omega_dp, dp = Omega(K_per_feature, i, sigmas,q_additivity=None)
    val[i] = torch.matmul(omega_dp, alpha)
sum_shpaley = torch.sum(val)
print('Sum of Shapley values',sum_shpaley)
print('shapley values', val)


# Making a prediction based on the base kernel
K_instance_trainx, _ = K_s_instance(K_per_feature, sigmas)
prediction = torch.matmul(K_instance_trainx.unsqueeze(0),  alpha.unsqueeze(1))
print(prediction)


