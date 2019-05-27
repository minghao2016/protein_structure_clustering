
import numpy as np

def symmetrizeMatrix(a):
        return a + a.T - np.diag(a.diagonal())
