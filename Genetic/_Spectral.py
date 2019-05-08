

import ClusteringEvaluation as ce
import MatrixFunctions as mf
from sklearn.cluster import DBSCAN
from sklearn.cluster import SpectralClustering
from sklearn.cluster import AgglomerativeClustering
import UtilitiesSCOP as scop

import numpy as np

matrix = np.load('C:/ShareSSD/scop/data/matrix_a.1._rmsd')
n_labels = scop.getUniqueClassifications('a.1')

matrix = mf.minMaxScale(matrix)
#matrix = mf.calculateDistances(matrix)

#beta = 20
#matrix = np.exp( beta-matrix / matrix.std())
# testar 1-matrix
#matrix = 1-matrix
print(matrix)
#matrix = 1-matrix
#matrix = np.exp(- matrix ** 2 / (2. * (matrix.std()) ** 2))

#for n_clusters in np.arange(4,10,1):

#mf.calculateDistances(matrix)

sc = SpectralClustering(n_clusters=n_labels, affinity='nearest_neighbors', assign_labels="discretize", random_state=100).fit(matrix)
metrics = ce.clusterEvaluationNoLabels(matrix, sc.labels_)
print(metrics)

sc = SpectralClustering(n_clusters=n_labels, affinity='nearest_neighbors', assign_labels="kmeans", random_state=100).fit(matrix)
metrics = ce.clusterEvaluationNoLabels(matrix, sc.labels_)
print(metrics)

sc = SpectralClustering(n_clusters=n_labels, affinity='rbf', assign_labels="discretize", random_state=100).fit(matrix)
metrics = ce.clusterEvaluationNoLabels(matrix, sc.labels_)
print(metrics)

agglomerative = AgglomerativeClustering(affinity='precomputed', n_clusters=5, linkage="complete").fit(matrix)
metrics = ce.clusterEvaluationNoLabels(matrix, agglomerative.labels_)
print(metrics)

print("")


