'''example program to cluster iris data using hbc'''
import numpy as np
from sklearn import datasets
from pyBHC import bhc
from pyBHC.dists import NormalInverseWishart
if __name__ == '__main__':
    feature, ground_truth = datasets.load_iris(return_X_y = True)
    dim = feature.shape[1]
    hypers = {
        'mu_0': np.zeros(dim),
        'nu_0': 3.0,
        'kappa_0': 1.0,
        'lambda_0': np.eye(dim)
    }
    data_model = NormalInverseWishart(**hypers)    
    bhc_instance = bhc(feature, data_model)
    asgn = bhc_instance.assignments
    print(asgn)
