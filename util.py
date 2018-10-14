import numpy as np

def sample(distribution, theta, sigma, reshape_param=[], number=1):
    if distribution == 'gaussian':
        distribution = np.random.multivariate_normal
    if isinstance(sigma,int):
        theta_k =  distribution(theta.reshape(theta.shape[0]*theta.shape[1]), (sigma*sigma)*np.identity(theta.shape[0]*theta.shape[1]), 1)
        return theta_k.reshape(reshape_param[0], reshape_param[1])
    else:
        theta_k = distribution(theta.reshape(theta.shape[0]), sigma, number)
        return theta_k.reshape(number, reshape_param[0], reshape_param[1])

def get_init(state_space, action_space, sigma, condition=False):
    if condition:
        theta = np.random.rand(state_space, action_space)
        return theta.reshape(state_space, action_space)    
    theta = np.random.rand(state_space, action_space)
    theta = theta.reshape(state_space*action_space, 1)
    shape = theta.shape[0]
    sigma_diag = sigma
    sigma_ = (sigma_diag*sigma_diag)*np.identity(shape)
    return theta, sigma_

def generate_new_distribution(distribution, mean_vector, values, best_k, epsilon):
    if distribution == 'gaussian':
        distribution = np.random.multivariate_normal
    values = values[:best_k]
    sum_vector = np.array([element[0] for element in values])
    average_vector = np.sum(sum_vector, axis=0)/best_k
    sum_vector -= mean_vector
    sigma = (epsilon*np.identity(values[0][0].shape[0]))
    for vector in sum_vector:
        sigma += vector.dot(vector.T)
    sigma /= (epsilon + best_k)

    return average_vector, sigma
