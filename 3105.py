import pandas as pd
import numpy as np
from sklearn.model_selection import cross_val_score,train_test_split
import torch
import gpytorch
from gpytorch.models import ExactGP
from gpytorch.distributions import MultivariateNormal

from gpytorch.constraints import Positive

from sklearn.preprocessing import StandardScaler

# from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split


np.random.seed(42)

# Define the number of samples
num_samples = 200

# Generate features X1 to X10 from N(0,1)
X = np.random.normal(0, 1, (num_samples, 10))
y = np.sin(X[:, 0]**2) - 2 * X[:, 0] * X[:, 1] + np.exp(X[:, 0]**2 * X[:, 1])


# Split the data into training and testing sets
from sklearn.model_selection import train_test_split
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

    #
import gpytorch
import torch

class DP_additive_kernel(gpytorch.kernels.Kernel):
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
        x1_size = x1.size(0)
        x2_size = x2.size(0)
        result = torch.zeros(x1_size, x2_size, device=x1.device)
        sum_order_b = torch.zeros(x1_size, x2_size, device=x1.device)
        kernels = []

        for d in range(self.num_dims):
            x1_d = x1[:, d:d+1]
            x2_d = x2[:, d:d+1]
            k_d = self.base_kernel(x1_d, x2_d).evaluate()
            kernels.append(k_d)
            sum_order_b += k_d

        result += sum_order_b * self.outputscale[0]

        for i in range(1, self.q_additivity):
            temp_sum = torch.zeros(x1_size, x2_size, device=x1.device)
            new_kernels = []
            for j in range(self.num_dims):
                for k in range(j + 1, self.num_dims):
                    new_kernel = kernels[j] * kernels[k]
                    new_kernels.append(new_kernel)
                    temp_sum += new_kernel

            kernels = new_kernels
            result += temp_sum * self.outputscale[i]

        result += 1

        return result

    def predict(self, train_x, y_train, test_x, likelihood):
        self.eval()
        likelihood.eval()

        with torch.no_grad(), gpytorch.settings.fast_pred_var():
            # Get into evaluation (predictive posterior) mode
            observed_pred = likelihood(self(train_x, test_x))
            mean = observed_pred.mean
            lower, upper = observed_pred.confidence_region()

        return mean, lower, upper


# Example usage in a GP model
class AGP_model(gpytorch.models.ExactGP): # i need to find a diferent model
    def __init__(self, train_x, train_y, likelihood):
        super(AGP_model, self).__init__(train_x, train_y, likelihood)
        self.mean_module = gpytorch.means.ZeroMean()
        self.base_kernel = gpytorch.kernels.RBFKernel()
        # self.base_kernel = CustomRBFKernel()
        # self.base_kernel = OrthogonalRBF()
        self.covar_module = DP_additve_kernel(base_kernel=self.base_kernel, num_dims=train_x.size(-1), q_additivity=train_x.size(-1))

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x,x)  # Make sure to pass x twice WHY
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)

likelihood = gpytorch.likelihoods.GaussianLikelihood()
model = AGP_model(train_x, y_train.squeeze(-1), likelihood)
optimizer = torch.optim.Adam([
    {'params': model.parameters()},  # includes kernel parameters
], lr=0.1)
mll = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood, model)




# Training loop
model.train()
likelihood.train()
outputscale_history = []
loss_history = []
training_iter = 1000
for i in range(training_iter):
    optimizer.zero_grad()
    output = model(train_x)
    loss = -mll(output, y_train)
    loss.backward()
    optimizer.step()

    # Store parameter value
    outputscale_history.append(model.covar_module.outputscale.detach().cpu().numpy())
    loss_history.append(loss.item())

# Switch to evaluation mode after training
model.eval()
likelihood.eval()

# Plotting the outputscales and loss
import matplotlib.pyplot as plt
plt.figure(figsize=(10, 5))  # Increase figure size for better visibility
plt.plot(outputscale_history, label='Outputscale', color='blue')  # Plot outputscale
plt.plot(loss_history, label='Loss', color='red')  # Plot loss
plt.title('Training Metrics Over Iterations')
plt.xlabel('Iteration')
plt.ylabel('Values')
plt.legend()  # Include legend to identify the lines

plt.show()



model.eval()
likelihood.eval()
for param_name, param in model.named_parameters():
    print(f'Parameter name: {param_name:42} value = {param.data}')


#evaluating there is a problem when the test_y and test_x have float numbers
with torch.no_grad():
    model.eval()  # Set the model to evaluation mode (mode is for computing predictions through the model posterior.)
    likelihood.eval()
    output = likelihood(model(test_x))  # Make predictions on new data 
    

# Extracting means and standard deviations
predicted_means = output.mean.numpy() 
predicted_stddevs = output.stddev.numpy()  # Extract standard deviations

print("Predicted Means:")
print(predicted_means)

print("Predicted Standard Deviations:")
print(predicted_stddevs)

#calculate teh alpha_hat_eta
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



n, d = train_x.shape

with torch.no_grad():
    # kernel = model.covar_module
    kernel =  model.base_kernel

    # Initialize the matrix K with zeros
    K_per_feature = torch.zeros((n, d))

    # Extracting a specific instance's features
    instance_features = train_x[3].unsqueeze(0)  # Shape (1, d)

    # Loop over each feature dimension
    for i in range(d):
        # Reshape the specific feature across all samples to match the input shape required by the kernel
        feature_column = train_x[:, i].unsqueeze(1)  # Shape (n, 1)
        instance_feature = instance_features[:, i].unsqueeze(1)  # Shape (1, 1)

        K_per_feature[:, i] = kernel(instance_feature, feature_column).evaluate().squeeze(0)


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
  
    # dp[:, 0, :] = model.covar_module.raw_outputscale.data.T * dp[:, 0, :] # multiply each row of the matrix by corresponding sigma value

    # Sum up all contributions from the first dimension of each feature to get the final values
    omega = torch.sum(dp[:, 0, :], axis=0)
    
    return omega, dp


val = torch.zeros(n_features)
for i in range(n_features):
    omega_dp, _ = Omega(K_per_feature, i, q_additivity=None, feature_type='numerical')
    val[i] = torch.matmul(omega_dp, alpha_hat_eta)

# print('This is val:', val)

sum_shap = torch.sum(val)
print("shapley values sum", sum_shap)
#this is the 1^T*alpha_hat_eta
bias = torch.sum(alpha_hat_eta, dim=0)


# Assuming train_x is a tensor and model & likelihood have been defined appropriately
# instance_features = train_x[3].unsqueeze(0)  # Ensure it's in the correct shape

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


# print(predicted_mean - alpha == sum_shap)
print(predicted_mean - bias)