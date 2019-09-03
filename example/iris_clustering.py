'''example program to cluster iris data using hbc
   pyBHC library not support python 3 currently
   use python 2.7 to run this example script
   have patience to wait for the results
'''
import numpy as np
from sklearn import datasets
from pyBHC import bhc
from pyBHC.dists import NormalInverseWishart
from scipy.cluster.hierarchy import dendrogram
import matplotlib.pyplot as plt

def makeLinkageMatrix(asgn,lmls):
    N = len(asgn[0])
    Z = np.zeros((N-1,4),dtype=np.double)
    parents=dict([(i,i) for i in range(N)])
    nleaves = dict([(i,1.0) for i in range(N)])
    for i in range(N-1):
        L0 = asgn[i]
        L1 = asgn[i+1]
        for j in range(N):
            if L0[j] != L1[j] and parents[L0[j]]!= parents[L1[j]]:
                Z[i,0] = parents[L0[j]]
                Z[i,1] = parents[L1[j]]

                Z[i,2] = np.fabs(lmls[i]) #-lmls[i+1] if i<N-2 else lmls[N-2]-lmls[N-3]
                Z[i,3] = np.double(nleaves[parents[L0[j]]] + nleaves[parents[L1[j]]])
                
                parents[L0[j]]= N+i
                parents[L1[j]]= N+i
                nleaves[N+i] = Z[i,3]
    print(Z)
    return Z
    
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
    lmls = bhc_instance.lmls
    print('len of data', asgn.shape[0], 'len of lml', len(lmls))
    Z = makeLinkageMatrix(asgn, lmls)
    dn = dendrogram(Z)
    plt.show()
