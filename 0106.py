import torch
import gpytorch

import pandas as pd
import numpy as np
from sklearn.model_selection import cross_val_score,train_test_split
from gpytorch.models import ExactGP
from gpytorch.distributions import MultivariateNormal

from gpytorch.constraints import Positive

from sklearn.preprocessing import StandardScaler

from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split



np.random.seed(42)

# Define the number of samples
num_samples = 200

# Generate features X1 to X10 from N(0,1)
X = np.random.normal(0, 1, (num_samples, 10))
y = np.sin(X[:, 0]**2) - 2 * X[:, 0] * X[:, 1] + np.exp(X[:, 0]**2 * X[:, 2])


# Split the data into training and testing sets
from sklearn.model_selection import train_test_split
train_x, test_x, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42) # 20% data as test

# Initialize the scaler for features and target
scaler_x = StandardScaler()
#not scalling the target values
# scaler_y = StandardScaler()

# Fit and transform the training data
train_x_scaled = scaler_x.fit_transform(train_x)
# y_train_scaled = scaler_y.fit_transform(y_train.reshape(-1, 1)).ravel()

# Transform the test data
test_x_scaled = scaler_x.transform(test_x)
# y_test_scaled = scaler_y.transform(y_test.reshape(-1, 1)).ravel()

# Convert to torch tensors
train_x = torch.tensor(train_x_scaled, dtype=torch.float32)
test_x = torch.tensor(test_x_scaled, dtype=torch.float32)
# y_train = torch.tensor(y_train_scaled, dtype=torch.float32)
# y_test = torch.tensor(y_test_scaled, dtype=torch.float32)
y_train = torch.tensor(y_train, dtype=torch.float32)
y_test = torch.tensor(y_test, dtype=torch.float32)
]


class OrthogonalRBF(gpytorch.kernels.Kernel):
    has_lengthscale = True

    def __init__(self, mu=0, var=1):
        super(OrthogonalRBF, self).__init__()
        self.mu = torch.tensor(mu, dtype=torch.float)
        self.var = torch.tensor(var, dtype=torch.float)
        self.register_parameter(name="raw_lengthscale", parameter=torch.nn.Parameter(torch.tensor(1.0)))
        self.register_constraint("raw_lengthscale", gpytorch.constraints.Positive())

    def cov_X_s(self, X):
        """
        Computes covariance between X and the Gaussian measure (mu, var).
        """
        l = self.lengthscale
        mu, var = self.mu, self.var
        return (l / torch.sqrt(l ** 2 + var)
                * torch.exp(-0.5 * ((X - mu) ** 2) / (l ** 2 + var)))

    def var_s(self):
        """
        Computes variance of the Gaussian measure (mu, var).
        """
        l = self.lengthscale
        return l / torch.sqrt(l ** 2 + 2 * self.var)

    def forward(self, x1, x2, diag=False, **kwargs):
        """
        Computes the kernel matrix using the base RBF kernel adjusted by the Gaussian measure's influence.
        """
        x1_ = x1.unsqueeze(1) - x2.unsqueeze(0)
        r2 = x1_.pow(2).sum(-1).div(self.lengthscale.pow(2))
        base_kernel = torch.exp(-0.5 * r2)

        cov_X1_s = self.cov_X_s(x1)
        cov_X2_s = self.cov_X_s(x2)
        
        adjustment = torch.mm(cov_X1_s, cov_X2_s.T) / self.var_s()

        result = base_kernel - adjustment
        return result.diag() if diag else result

class DPadditveKernel(gpytorch.kernels.Kernel):
    def __init__(self, base_kernel, num_dims, q_additivity, **kwargs):
        super().__init__(**kwargs)
        self.base_kernel = base_kernel
        self.num_dims = num_dims
        self.q_additivity = q_additivity
        self.register_parameter(
            name="raw_outputscale", 
            parameter=torch.nn.Parameter(torch.ones(1, self.q_additivity))
        )
        self.outputscale_constraint = gpytorch.constraints.Positive()
        self.register_constraint("raw_outputscale", self.outputscale_constraint)

    @property
    def outputscale(self):
        return self.outputscale_constraint.transform(self.raw_outputscale).squeeze()

    @outputscale.setter
    def outputscale(self, value):
        if not torch.is_tensor(value):
            value = torch.tensor(value, device=self.raw_outputscale.device)
        self.initialize(raw_outputscale=self.outputscale_constraint.inverse_transform(value))

    def forward(self, x1, x2, diag=False, **params):
    # Determine sizes based on input matrices
        n,d = x1.shape
      
       # Initialize dp array
        dp = torch.zeros(self.q_additivity, d, n, n)
            # Initialize the matrix K with zeros
        current_sum = torch.zeros(n,n)
        #first calculate the first order kernels
    

        # Compute the covariance matrix for each feature
        for i in range(d):
            # Extract the i-th feature and reshape to (n, 1)
            feature_column = train_x[:, i].unsqueeze(1)
            # Apply the kernel to the feature column
            # Since kernel operates on (n, 1) input, it outputs an (n, n) covariance matrix
            covariance_matrix = self.base_kernel(feature_column).evaluate()

            # Store the computed covariance matrix in the tensor
            dp[0,i, :, :] = covariance_matrix

            current_sum +=  dp[0,i, :, :]

        #for the higher orders
        for i in range(1, self.q_additivity):
            temp_sum = torch.zeros(n,n)
            for j in range(self.num_dims):
                current_sum -= dp[i-1,j,:,:] 
                if (current_sum > 0).all():
                    dp[i,j,:,:] = dp[i-1,j,:,:]  * current_sum
                    temp_sum =+  dp[i,j,:,:]
            current_sum = temp_sum
        
 
        for i in range(self.q_additivity):
            for j in range(self.num_dims):
                dp[i,j,:,:] = self.outputscale[i] * dp[i,j,:,:] 
        dp_summed = torch.sum(dp, dim=1)
        result = torch.sum(dp_summed, dim= 0)
      
        result += 1

        return result.diag() if diag else result




class AGP_model(gpytorch.models.ExactGP): # i need to find a diferent model
    def __init__(self, train_x, train_y, likelihood):
        super(AGP_model, self).__init__(train_x, train_y, likelihood)
        self.mean_module = gpytorch.means.ZeroMean()
        # self.base_kernel = gpytorch.kernels.RBFKernel()
        self.base_kernel = OrthogonalRBF()
        self.covar_module = DPadditveKernel(base_kernel=self.base_kernel, num_dims=train_x.size(-1), q_additivity=train_x.size(-1))

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x,x)  # Make sure to pass x twice WHY
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)

# Create the GP model
likelihood = gpytorch.likelihoods.GaussianLikelihood()


model = AGP_model(train_x, y_train.squeeze(-1), likelihood)
optimizer = torch.optim.Adam(model.parameters(), lr=0.1)

# "Loss" for GPs - the marginal log likelihood
mll = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood, model)
model.train()
likelihood.train()
# Training loop
training_iter = 1000
for i in range(training_iter):
    optimizer.zero_grad()
    output = model(train_x)
    # print(output)
    loss = -mll(output, y_train)
    loss = loss.mean() 
    loss.backward()
    
    optimizer.step()

model.eval()

for param_name, param in model.named_parameters():
    print(f'Parameter name: {param_name:42} value = {param.data}')



















       

