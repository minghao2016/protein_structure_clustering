
import ClusteringEvaluation as ce
import MatrixFunctions as mf
import LoadData as ld
import UtilitiesSCOP as scop
import numpy as np

from sklearn.mixture import GaussianMixture


combinations = [('rmsd','gdt_2'),('rmsd','gdt_4'),
                ('rmsd','maxsub'),('rmsd','tm'),
                ('gdt_2','gdt_4'),('gdt_2','maxsub'),
                ('gdt_2','tm'), ('gdt_4','maxsub'),
                ('gdt_4','tm'), ('maxsub','tm')]

for m1, m2 in combinations:
    for spl in ['a.1','a.3','b.2','b.3']:

        # load protein data before loop
        path_to_results = 'C:/ShareSSD/scop/clustering_results_combined/'
        measure1 = m1
        measure2 = m2
        measure3 = 'seq'

        sample_for_domains = spl
        sample = str(spl)+'.'
        
        matrix1 = ld.loadMatrixFromFile(sample, measure1)
        matrix2 = ld.loadMatrixFromFile(sample, measure2)
        matrix3 = ld.loadMatrixFromFile(sample, measure3)

        domains = ld.loadDomainListFromFile(sample)

        n_labels = scop.getUniqueClassifications(sample_for_domains)

        ground_truth = scop.getDomainLabels(domains)

        matrix1 = mf.minMaxScale(matrix1)
        matrix2 = mf.minMaxScale(matrix2)
        matrix3 = mf.minMaxScale(matrix3)

        matrix1 = mf.calculateDistances(matrix1)
        matrix2 = mf.calculateDistances(matrix2)
        matrix3 = mf.calculateDistances(matrix3)

        for w1 in np.arange(0.00,1.01,0.01):
            w2 = 1-w1
            w3 = 0

            corr = mf.calculateCorrelationMatrix(matrix1, matrix2, matrix3, w1, w2, w3)

            # Gaussian Mixture Models
            gmm = GaussianMixture(n_components=n_labels).fit(corr)
            labels = gmm.predict(corr)
            metrics = ce.clusterEvaluation(corr, labels, ground_truth)
            ce.saveResultsCombined(measure1, measure2, w1, w2, w3, 'gmm', sample, metrics)
            print(metrics)
