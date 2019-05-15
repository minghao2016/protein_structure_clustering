
import random
import numpy as np
import matplotlib.pyplot as plt

# GENETIC ALGORITHM
from deap import algorithms
from deap import base
from deap import creator
from deap import tools

# PROTEIN CLUSTERING
import ClusteringEvaluation as ce
import LoadData as ld
import MatrixFunctions as mf
import UtilitiesSCOP as scop
from sklearn import metrics
from sklearn.cluster import AffinityPropagation

samples = ['a.1','a.3','b.2','b.3']

clustering = ['complete','average']

combinations = [('rmsd','gdt_2'),('rmsd','gdt_4'),
                ('rmsd','maxsub'),('rmsd','tm'),
                ('gdt_2','gdt_4'),('gdt_2','maxsub'),
                ('gdt_2','tm'), ('gdt_4','maxsub'),
                ('gdt_4','tm'), ('maxsub','tm')]

NGENERATIONS = 30
POPSIZE = 100

for measures in combinations:
    for spl in samples:

        #####################################################
        # LOAD PROTEIN DATA
        #####################################################
        measure1 = measures[0]
        measure2 = measures[1]
        measure3 = 'seq'
        sample_for_domains = spl
        sample = spl+'.'    
        path_to_results = 'C:/ShareSSD/scop/genetic_afp/genafp_'+sample+'_'+measure1+'_'+measure2        
        matrix1 = ld.loadMatrixFromFile(sample, measure1)
        matrix2 = ld.loadMatrixFromFile(sample, measure2)
        matrix3 = ld.loadMatrixFromFile(sample, measure3)
        matrix1 = mf.minMaxScale(matrix1)
        matrix2 = mf.minMaxScale(matrix2)
        matrix3 = mf.minMaxScale(matrix3)
        matrix1 = mf.calculateDistances(matrix1)
        matrix2 = mf.calculateDistances(matrix2)
        matrix3 = mf.calculateDistances(matrix3)
        domains = ld.loadDomainListFromFile(sample_for_domains)
        n_labels = scop.getUniqueClassifications(sample_for_domains)
        ground_truth = scop.getDomainLabels(domains)

        current_individual = 0
        current_generation = 0
        writer = open(path_to_results, 'w') 
        writer.write('Gen: '+str(current_generation)+'\n')

        #####################################################
        # GENERATE INDIVIDUALS
        #####################################################
        def generateIndividualAfp():
            w1 = round(random.uniform(0,1),2)
            w2 = round(random.uniform(0,1-w1),2)
            w3 = round(1-w2-w1,2)
            w4 = random.randint(-1000,500)
            w5 = round(random.uniform(0.5,1),1)
            return [w1,w2,w3,w4,w5]

        #####################################################
        # FITNESS
        #####################################################
        def evaluate(individual):

            indv = [round(x,2) for x in individual]
            w1 = indv[0]
            w2 = indv[1]
            w3 = indv[2]
            w4 = indv[3]
            w5 = indv[4]

            corr = mf.calculateCorrelationMatrix(matrix1, matrix2, matrix3, w1, w2, w3)

            af = AffinityPropagation(preference=w4, damping=w5).fit(corr)
            labels = af.labels_

            if len(set(labels)) != n_labels:
                return -1,-1,

            #return ari and silhouette 
            metrics = ce.clusterEvaluationNoLabels(corr, labels)
            return float(metrics[4]), float((metrics[6])),

        #####################################################
        # REGISTER COMPONENTS
        #####################################################
        # contains preference and damping
        IND_SIZE = 5  
        toolbox = base.Toolbox()
        # two weights - ari (most important) and silhouette 
        creator.create("FitnessMax", base.Fitness, weights=(1.0, 0.9))
        creator.create("Individual", list, fitness=creator.FitnessMax)
        toolbox.register('expr', generateIndividual)
        toolbox.register('individual', tools.initIterate, creator.Individual, toolbox.expr)
        toolbox.register('population', tools.initRepeat, list, toolbox.individual)
        # add the cluster number restriction
        toolbox.register("evaluate", evaluate)
        toolbox.register("mate", tools.cxTwoPoint)
        toolbox.register("mutate", tools.mutFlipBit, indpb=0.2)
        toolbox.register("select", tools.selTournament, tournsize=40)

        def main():
            # random.seed(94)

            global NGENERATIONS
            global POPSIZE
            global writer

            population = toolbox.population(n=POPSIZE)

            # CXPB - probabilidade de crossover
            # MUTPB - probabilidade de mutacao
            # NGEN - numero de geracoes
            CXPB, MUTPB = 0.9, 0.01
            
            # STATISTICS
            stats = tools.Statistics(key=lambda ind: ind.fitness.values)
            stats.register("min", np.min)
            stats.register("avg", np.mean)
            stats.register("max", np.max)

            # Run GA
            population, logbook = algorithms.eaSimple(population, toolbox, CXPB, MUTPB, NGENERATIONS, stats=stats)

            writer.close()

        if __name__ == "__main__":
            main()