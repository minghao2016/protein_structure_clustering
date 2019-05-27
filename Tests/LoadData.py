
import numpy as np

def loadMatrixFromFile(sample, measure):
    path_to_matrix = 'C:/ShareSSD/scop/data/matrix_'+sample+'_'+measure
    matrix = np.load(path_to_matrix)
    return matrix

def loadDomainListFromFile(sample):
    path_to_domains = 'C:/ShareSSD/scop/data/domains_'+sample
    domains = []
    with open(path_to_domains, 'r') as fp:
        line = fp.readline()
        while 'END' not in line:
            domains.append(str(line).strip())
            line = fp.readline()
    return domains