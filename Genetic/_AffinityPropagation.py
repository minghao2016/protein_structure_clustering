

from sklearn.cluster import AffinityPropagation
from sklearn import metrics

import MatrixFunctions as mf
import ClusteringEvaluation as ce
import LoadData as ld
import UtilitiesSCOP as scop
import numpy as np

domains = ld.loadDomainListFromFile('a.1.')

matrix = np.load('C:/ShareSSD/scop/data/matrix_a.1._rmsd')

matrix = mf.minMaxScale(matrix)
matrix = mf.calculateDistances(matrix)
ground_truth = scop.getDomainLabels(domains)

unique_ = set(ground_truth)

best_pref = -2000
best_damp = 0.5

max = [0,0,0,0,0,0,0]

for preference in np.arange(-2000, 2000, 50):
    for damping in np.arange(0.5,1.0,0.1):

        af = AffinityPropagation(preference=preference, damping=damping).fit(matrix)
        labels = af.labels_
        metrics = ce.clusterEvaluation(matrix, labels, ground_truth)
        unique = set(labels)
        if metrics[4] > max[4] and unique == unique_:
            max = metrics
            best_damp = damping
            best_pref = preference

        print(metrics)

print('..........................')
print(max)
print(best_damp)
print(best_pref)
print('..........................')