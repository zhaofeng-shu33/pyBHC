'''example program to cluster iris data using hbc
   pyBHC library not support python 3 currently
   use python 2.7 to run this example script
   have patience to wait for the results
'''
import numpy as np
from sklearn import datasets
from pyBHC import bhc
from pyBHC.dists import NormalInverseWishart
if __name__ == '__main__':
    feature, ground_truth = datasets.load_iris(return_X_y = True)
    dim = feature.shape[1]
    hypers = {
        'mu_0': np.zeros(dim),
        'nu_0': 4.0,
        'kappa_0': 1.0,
        'lambda_0': np.eye(dim)
    }
    data_model = NormalInverseWishart(**hypers)    
    bhc_instance = bhc(feature, data_model)
    asgn = bhc_instance.assignments
    print(asgn)
