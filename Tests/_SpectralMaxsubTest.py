

import ClusteringEvaluation as ce
import MatrixFunctions as mf
from sklearn.cluster import SpectralClustering
import UtilitiesSCOP as scop
import LoadData as ld
import numpy as np

domains = ld.loadDomainListFromFile('a.1.')

matrix = np.load('C:/ShareSSD/scop/data/matrix_a.1._maxsub')
n_labels = scop.getUniqueClassifications('a.1')

matrix = mf.minMaxScale(matrix)
matrix = mf.calculateDistances(matrix)
ground_truth = scop.getDomainLabels(domains)

sc = SpectralClustering(n_clusters=n_labels, affinity='precomputed', assign_labels="kmeans", random_state=100, n_jobs=-1).fit(matrix)
metrics = ce.clusterEvaluationNoLabels(matrix, sc.labels_)
print(metrics)
metrics = ce.clusterEvaluation(matrix, sc.labels_, ground_truth)
print(metrics)

sc = SpectralClustering(n_clusters=n_labels, affinity='precomputed', assign_labels="discretize", random_state=100, n_jobs=-1).fit(matrix)
metrics = ce.clusterEvaluationNoLabels(matrix, sc.labels_)
print(metrics)

sc = SpectralClustering(n_clusters=n_labels, affinity='nearest_neighbors', assign_labels="kmeans", random_state=100, n_jobs=-1).fit(matrix)
metrics = ce.clusterEvaluationNoLabels(matrix, sc.labels_)
print(metrics)

sc = SpectralClustering(n_clusters=n_labels, affinity='nearest_neighbors', assign_labels="discretize", random_state=100, n_jobs=-1).fit(matrix)
metrics = ce.clusterEvaluationNoLabels(matrix, sc.labels_)
print(metrics)

sc = SpectralClustering(n_clusters=n_labels, affinity='rbf', assign_labels="discretize", random_state=100, n_jobs=-1).fit(matrix)
metrics = ce.clusterEvaluationNoLabels(matrix, sc.labels_)
print(metrics)

#agglomerative = AgglomerativeClustering(affinity='precomputed', n_clusters=5, linkage="complete").fit(matrix)
#metrics = ce.clusterEvaluationNoLabels(matrix, agglomerative.labels_)
#print(metrics)

print("")


