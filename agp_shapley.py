import torch
import random
import gpytorch

from gpytorch.distributions import MultivariateNormal
from gpytorch.kernels import  NewtonGirardAdditiveKernel
from gpytorch.likelihoods import GaussianLikelihood
from gpytorch.means import ConstantMean
from gpytorch.mlls import ExactMarginalLogLikelihood
from gpytorch.models import ExactGP
from sklearn.preprocessing import StandardScaler
import numpy as np
from sklearn.model_selection import train_test_split


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
scaler_y = StandardScaler()
# Fit and transform the training data
train_x_scaled = scaler_x.fit_transform(train_x)
y_train_scaled = scaler_y.fit_transform(y_train.reshape(-1, 1)).ravel()

# Transform the test data
test_x_scaled = scaler_x.transform(test_x)
y_test_scaled = scaler_y.transform(y_test.reshape(-1, 1)).ravel()

# Convert to torch tensors
train_x = torch.tensor(train_x_scaled, dtype=torch.float32)
test_x = torch.tensor(test_x_scaled, dtype=torch.float32)
y_train = torch.tensor(y_train_scaled, dtype=torch.float32)
y_test = torch.tensor(y_test_scaled, dtype=torch.float32)


# Base Kernel definitions
class ConstrainedRBFKernel(gpytorch.kernels.Kernel):
    has_lengthscale = True
    
    def __init__(self, mu=0, var=1, **kwargs):
        super(ConstrainedRBFKernel, self).__init__(**kwargs)
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
class TestGPModel(ExactGP):
    def __init__(self, train_x, train_y, likelihood):
        super().__init__(train_x, train_y, likelihood)
        self.mean_module = gpytorch.means.ZeroMean()
        self.covar_module = NewtonGirardAdditiveKernel(base_kernel=ConstrainedRBFKernel(), num_dims=train_x.shape[-1])

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return MultivariateNormal(mean_x, covar_x)

likelihood = GaussianLikelihood()
model = TestGPModel(train_x, y_train, likelihood)

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



'''
This part of code was to check if the used parameters give as accurate predictions caompared to predictions made by gputorch

for param_name, param in model.named_parameters():
    print(f'Parameter name: {param_name:42} value = {param.data}')
    # Evaluating with standard deviations
model.eval()
likelihood.eval()
with torch.no_grad():
    output = model(test_x)
    
    # Extracting means and standard deviations
    predicted_means = output.loc
    predicted_covariance_matrix = output.covariance_matrix
    predicted_stddevs = output.stddev.numpy()  # Extract standard deviations
print("Predicted Means:")
print(predicted_means)

print("Predicted Standard Deviations:")
print(predicted_stddevs.shape)


print(predicted_covariance_matrix.shape)

'''
# Calculation of the posterior mean nad covariance matrix (especially important as we need 'alpha' in later calculations)
trained_kernel = model.covar_module.base_kernel
outputscale = model.covar_module.outputscale
kernel = NewtonGirardAdditiveKernel(base_kernel=trained_kernel, num_dims=train_x.shape[-1])
kernel.outputscale = outputscale

noise_variance = likelihood.noise.data
# Compute training and test kernel matrices

with torch.no_grad():
    model.eval()

    likelihood.eval()
    K_train = kernel(train_x, train_x).evaluate() + noise_variance * torch.eye(train_x.size(0))
    K_s = kernel(test_x,train_x).evaluate()
    K_ss = kernel(test_x, test_x).evaluate() +  torch.eye(test_x.size(0))  # Added jitter for numerical stability

    K_inv =  torch.inverse(K_train)
    alpha = torch.matmul(K_inv, y_train)
    predicted_mean =torch.matmul(K_s,alpha)
    v = torch.linalg.solve_triangular(torch.linalg.cholesky(K_train), K_s.t(), upper=False)
    predicted_covariance_matrix = K_ss - v.t().matmul(v)


# Shapley values calculations

# Creating the K_per_feature matrix
n_samples, n_features = train_x.shape

K_per_feature = torch.zeros(n_samples, n_features)

with torch.no_grad():
    model.eval()
    likelihood.eval()
    
    # Get lengthscale from model's kernel if it's already defined
    l = model.covar_module.base_kernel.lengthscale.item()
    
    # Define the ConstrainedRBFKernel

    constrained_kernel = ConstrainedRBFKernel()
    constrained_kernel.lengthscale = l
    #constrained_kernel = trained_kernel # this one was used to compare if the constrained_kernel form ConstrainedRBFKernel() is set with correct l
    
    # Extract the instance's features; assuming you want the 4th sample (index 3)
    instance_features = train_x[3].unsqueeze(0)  # Shape (1, d)

    # Loop over each feature dimension
    for i in range(n_features):
        # Extract the i-th feature across all samples
        feature_column = train_x[:, i].unsqueeze(1)  # Shape (n, 1)
        instance_feature = instance_features[:, i].unsqueeze(1)  # Shape (1, 1)
       
        # Compute the kernel matrix for the i-th feature
        K_per_feature[:, i] = constrained_kernel.evaluate(instance_feature, feature_column)
       
print(K_per_feature)

# Calculating The Shapley value
# Creating the Omega function
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

sigmas = outputscale.data.unsqueeze(-1)
val = torch.zeros(n_features)
for i in range(n_features):
    omega_dp, dp = Omega(K_per_feature, i, sigmas,q_additivity=None)
    val[i] = torch.matmul(omega_dp, alpha)


sum_shapley_values = torch.sum(val)
print(sum_shapley_values)
print(val)

# Prediction using only base kernel (which is calculated in K_per_feature for all features seperately)

def K_S_instance(X, sigmas,q_additivity=None):
    
    n, d = X.shape
    if q_additivity is None:
        q_additivity = d
    
    # Reorder columns so that the i-th column is first

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
    result = torch.zeros(n_samples)
    for i in range(d):
        result += torch.sum(dp[:,i,:], axis=0)

    return result, dp

sigmas = outputscale.data.unsqueeze(-1)

K_s_instance , dp= K_S_instance(K_per_feature, sigmas)

prediction = torch.matmul(K_s_instance.unsqueeze(0),  alpha.unsqueeze(1))
print(prediction)