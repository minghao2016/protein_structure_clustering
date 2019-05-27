import numpy as np
import ClusteringEvaluation as ce
import MatrixFunctions as mf
from sklearn.cluster import DBSCAN
from sklearn import metrics
import LoadData as ld

matrix = np.load('C:/Users/pedro.arguelles/Desktop/scripts/matrix_a.3._maxsub')

matrix = mf.minMaxScale(matrix)
matrix = mf.calculateDistances(matrix)

best_e = 0
best_ms = 0
best_metrics = [0,0]
best_labels = 0

for e in np.arange(0.1,50,0.5):
    for ms in np.arange(5, 500, 5):

        try:
            db = DBSCAN(eps=e, min_samples=ms).fit(matrix)
            labels = db.labels_
            metrics = ce.clusterEvaluationNoLabels(matrix, labels)
            if metrics[1] > best_metrics[1]:
                best_metrics = metrics
                best_e = e
                best_ms = ms
                best_labels = len(set(labels))

            print('------------------------------------------------------------------')
            print(metrics)
            print(str(e)+'\n')
            print(str(ms))    
        except Exception:
            print('------------------------------------------------------------------')
            print(str(e)+'\n')
            print(str(ms))    
            pass

print('------------------------------------------------------------------')
print(best_e)
print(best_ms)
print(best_metrics)
print(best_labels)