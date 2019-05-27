import ClusteringEvaluation as ce
import MatrixFunctions as mf


from sklearn.neighbors import NearestNeighbors
import numpy as np

matrix = np.load('C:/Users/pedro.arguelles/Desktop/scripts/matrix_a.3._rmsd')

best_n = 0
best_metrics = []

for n in np.arange(2,100,1):
    try:
        nbrs = NearestNeighbors(n_neighbors=n, algorithm='ball_tree').fit(matrix)
        distances, indices = nbrs.kneighbors(matrix)
        labels = [col[1] for col in indices]
        metrics = ce.clusterEvaluationNoLabels(matrix, labels)
        if metrics[1] > best_metrics[1]:
            best_n = n
            best_metrics = metrics
    except Exception:
        pass

print(best_n)
print(best_metrics)