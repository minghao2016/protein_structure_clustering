

from sklearn.cluster import AffinityPropagation
from sklearn import metrics

import MatrixFunctions as mf
import ClusteringEvaluation as ce

import numpy as np

matrix = np.load('C:/Users/pedro.arguelles/Desktop/scripts/matrix_a.3._rmsd')

matrix = mf.minMaxScale(matrix)
#matrix = mf.calculateDistances(matrix)

max = [0,0]
best_preference = 0
best_damping = 0
best_labels = 0

for preference in np.arange(-1000, 1000, 50):
    for damping in np.arange(0.5,1.0,0.1):

        af = AffinityPropagation(preference=preference).fit(matrix)
        labels = af.labels_
        metrics = ce.clusterEvaluationNoLabels(matrix, labels)
        if metrics[1] > max[1]:
            max = metrics
            best_damping = damping
            best_preference = preference
            best_labels = len(set(labels))

        print(metrics)

#a.3 rmsd dist 0.5 -800
#a.3 maxsub dist 0.5 -50

print(max)
print(best_damping)
print(best_preference)
print(best_labels)