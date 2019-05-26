



import ClusteringEvaluation as ce
import MatrixFunctions as mf
import UtilitiesSCOP as scop
import LoadData as ld
import numpy as np
from sklearn.mixture import GMM
from sklearn.mixture import GaussianMixture
domains = ld.loadDomainListFromFile('a.3.')

matrix = np.load('C:/ShareSSD/scop/data/matrix_a.3._rmsd')
n_labels = scop.getUniqueClassifications('a.3')

matrix = mf.minMaxScale(matrix)
matrix = mf.calculateDistances(matrix)
ground_truth = scop.getDomainLabels(domains)

gmm = GaussianMixture(n_components=n_labels).fit(matrix)
labels = gmm.predict(matrix)

metrics = ce.clusterEvaluation(matrix, labels, ground_truth)

print(metrics)
print()