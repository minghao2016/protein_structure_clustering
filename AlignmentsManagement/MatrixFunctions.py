
import numpy as np
import sklearn.preprocessing as pp
from sklearn.metrics.pairwise import euclidean_distances

def symmetrizeMatrix(a):
        return a + a.T - np.diag(a.diagonal())
