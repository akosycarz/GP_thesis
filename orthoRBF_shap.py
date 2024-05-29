import pandas as pd
import numpy as np
from sklearn.model_selection import cross_val_score,train_test_split
import torch
import gpytorch
from gpytorch.models import ExactGP
from gpytorch.distributions import MultivariateNormal

from gpytorch.constraints import Positive

from sklearn.preprocessing import StandardScaler

from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split

# Load data
diabetes = load_diabetes()
X, y = diabetes.data, diabetes.target
X = X[:, :3]
# Split the data into training and testing sets
train_x, test_x, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42) # 20% data as test


# Create a StandardScaler object
scaler = StandardScaler()

# Fit the scaler to the training data and transform it
train_x_scaled = scaler.fit_transform(train_x)

# Transform the test data using the same scaler
test_x_scaled = scaler.transform(test_x)
#scale target
scaler_y = StandardScaler()
y_train = scaler_y.fit_transform(y_train.reshape(-1, 1)).ravel()  # Flatten back to 1D array
y_test = scaler_y.transform(y_test.reshape(-1, 1)).ravel()

# Convert to torch tensors
train_x = torch.tensor(train_x_scaled, dtype=torch.float32)
test_x = torch.tensor(test_x_scaled, dtype=torch.float32)
y_train = torch.tensor(y_train, dtype=torch.float32)
y_test = torch.tensor(y_test, dtype=torch.float32)


class OrthoRBF(gpytorch.kernels.Kernel):
    has_lengthscale = True

    def __init__(self, mu=0, var=1):
        super(OrthoRBF, self).__init__()
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

class DPkernel(gpytorch.kernels.Kernel):
    def __init__(self, base_kernel, num_dims, q_additivity, **kwargs):
        super().__init__(**kwargs)
        self.base_kernel = base_kernel
        self.num_dims = num_dims
        self.q_additivity = q_additivity
        self.register_parameter(
            name="raw_outputscale", 
            parameter=torch.nn.Parameter(torch.zeros(1, self.q_additivity))
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
        x1_size = x1.size(0)
        x2_size = x2.size(0)
        
        # Initialize matrices based on input sizes
        result = torch.zeros(x1_size, x2_size, device=x1.device) #initialize the result matrix
        sum_order_b = torch.zeros(x1_size, x2_size, device=x1.device) # initialize the matrix for the matrix for a single order
        kernels =[] # list were the z1, z2,... would be stored

       
        
        #calculations for first order
        #calcualte the kernels for each dimentions
        for d in range(self.num_dims):
            x1_d = x1[:, d:d+1]
            x2_d = x2[:, d:d+1]
            k_d = self.base_kernel(x1_d, x2_d).evaluate() # change thek to k0
            kernels.append(k_d) #save them in order in the kernels list
            # print(f"Kernel k_d at dim {d} shape: {k_d.shape}, sum_order_b shape: {sum_order_b.shape}")

            sum_order_b += k_d # add each one dimension kernels to one matrix for first order
    
        # first_kernels = kernels
        outputscale = self.outputscale.unsqueeze(0) if len(self.outputscale.shape) == 0 else self.outputscale
        result += sum_order_b * self.outputscale[0] #add the first order kernel miltiplied by first outputscale

        # Compute higher order interactions
        for i in range(1, self.q_additivity):
            temp_sum = torch.zeros(x1_size, x2_size, device=x1.device)
            new_kernels = []
            for j in range(self.num_dims):
                for k in range(j + 1, self.num_dims):
                    new_kernel = kernels[j] * kernels[k]
                    new_kernels.append(new_kernel)
                    temp_sum += new_kernel

            kernels = new_kernels  # update kernels list with new order interactions
            result += temp_sum * self.outputscale[i]

        return result

# Example usage in a GP model
class MyGP(gpytorch.models.ExactGP): # i need to find a diferent model
    def __init__(self, train_x, train_y, likelihood):
        super(MyGP, self).__init__(train_x, train_y, likelihood)
        self.mean_module = gpytorch.means.ZeroMean()
        # self.base_kernel = gpytorch.kernels.RBFKernel()
        # self.base_kernel = CustomRBFKernel()
        self.base_kernel = OrthoRBF()
        self.covar_module = DPkernel(base_kernel=self.base_kernel, num_dims=train_x.size(-1), q_additivity=train_x.size(-1))

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x,x)  # Make sure to pass x twice WHY
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)

# Create the GP model
likelihood = gpytorch.likelihoods.GaussianLikelihood()


model = MyGP(train_x, y_train.squeeze(-1), likelihood)
model.eval()

optimizer = torch.optim.Adam(model.parameters(), lr=0.1)
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


#evaluating there is a problem when the test_y and test_x have float numbers
with torch.no_grad():
    model.eval()  # Set the model to evaluation mode (mode is for computing predictions through the model posterior.)
    likelihood.eval()
    output = likelihood(model(test_x))  # Make predictions on new data 
    


# for 
# Extracting means and standard deviations
predicted_means = output.mean.numpy() 
predicted_stddevs = output.stddev.numpy()  # Extract standard deviations

# print("Predicted Means:")
# print(predicted_means)

# print("Predicted Standard Deviations:")
# print(predicted_stddevs)


# calculate teh alpha_hat_eta
model.eval()
likelihood.eval()
with torch.no_grad():
    # Evaluate the kernel matrix
    t_k_matrix = model.covar_module(train_x).evaluate()
    
    # Ensure the noise variance is non-zero and sufficiently large to avoid singularity
    noise = likelihood.noise_covar.noise
    # Check the value of noise and adjust accordingly
    if 0 < noise < 1e-6:
        noise_variance = 1e-6
    elif -1e-6 < noise < 0:
        noise_variance = -1e-6
    else:
        noise_variance = noise
    n_matrix = noise_variance * torch.eye(t_k_matrix.size(-1), device=t_k_matrix.device) 
    
    # Add regularization to avoid singular matrix
    inside = t_k_matrix + n_matrix
    K_inv = torch.inverse(inside)#(t_k_matrix + n_matrix) #+ torch.eye(t_k_matrix.size(-1), device=t_k_matrix.device)

    # Compute alpha_hat_eta using the inverse (dot product)
    alpha_hat_eta = torch.matmul(K_inv, y_train.unsqueeze(1))#.unsqueeze(-1)

    print(alpha_hat_eta.shape)
    # calculate teh alpha_hat_eta
    

# print(torch.sum(alpha_hat_eta))
# the vector with sigma values
sigmas = model.covar_module.raw_outputscale.data.T
print(sigmas.shape)
print(sigmas[0])
sigma_1 = sigmas[0]

n, d = train_x.shape
model.eval()
likelihood.eval()

with torch.no_grad():
    # kernel = model.covar_module
    kernel = OrthoRBF()

    # Initialize the matrix K with zeros
    K_per_feature = torch.zeros((n, d))

    # Extracting a specific instance's features
    instance_features = train_x[3].unsqueeze(0)  # Shape (1, d)

    # Loop over each feature dimension
    for i in range(d):
        # Reshape the specific feature across all samples to match the input shape required by the kernel
        feature_column = train_x[:, i].unsqueeze(1)  # Shape (n, 1)
        instance_feature = instance_features[:, i].unsqueeze(1)  # Shape (1, 1)

        K_per_feature[:, i] = kernel(instance_feature, feature_column).evaluate().squeeze()
        #

print("Matrix K (n*d):")
print(K_per_feature)

n_samples, n_features = train_x.size()
def Omega(X, i, q_additivity=None, feature_type='numerical'):
    
    
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
            dp[i, j, :] = (i / (i + 1)) * (X[:,j]* sum_current)
            temp_sum += dp[i, j, :]
        
        sum_current = temp_sum
  
    dp[:, 0, :] = sigmas * dp[:, 0, :] # multiply each row of the matrix by corresponding sigma value
    # Sum up all contributions from the first dimension of each feature to get the final values
    omega = torch.sum(dp[:, 0, :], axis=0)
    
    return omega, dp


val = torch.zeros(n_features)
for i in range(n_features):
    omega_dp, _ = Omega(K_per_feature, i, q_additivity=None, feature_type='numerical')
    val[i] = torch.matmul(omega_dp, alpha_hat_eta)

# print('This is val:', val)

sum_shap = torch.sum(val)
# sumee = sumee
print('sum of shapley values', sum_shap)

#this is the 1^T*alpha_hat_eta
alpha = torch.sum(alpha_hat_eta)
print('alpha',alpha)

# Assuming train_x is a tensor and model & likelihood have been defined appropriately
instance_features = train_x[3].unsqueeze(0)  # Ensure it's in the correct shape

model.eval()  # Set the model to evaluation mode
likelihood.eval()  # Also set the likelihood to evaluation mode

# Make a prediction using the model and the likelihood
with torch.no_grad():  # Ensures gradients are not tracked
    prediction = likelihood(model(instance_features))

# Extract mean and variance of the predictive distribution
predicted_mean = prediction.mean
predicted_variance = prediction.variance

# Print out the results
print("Predicted Mean:", predicted_mean)
print("Predicted Variance:", predicted_variance)


print(predicted_mean - alpha == sum_shap)
