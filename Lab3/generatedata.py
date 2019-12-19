import numpy as np

def get_dataset(mean,cov,numbers):
    dataset = np.random.multivariate_normal(mean,cov,numbers)
    return np.array(dataset)
