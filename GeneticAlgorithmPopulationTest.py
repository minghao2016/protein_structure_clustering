


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
import MatrixFunctions as mf
import ReadSimilarities as rs
import UtilitiesSCOP as scop
from sklearn import metrics
from sklearn.cluster import AffinityPropagation

#####################################################
# GENETIC ALGORITHM
#####################################################

IND_SIZE = 1
toolbox = base.Toolbox()

def generateIndividual():
    w1 = round(random.uniform(0,1),2)
    w2 = round(random.uniform(0,1-w1),2)
    w3 = round(1-w2-w1,2)
    w4 = random.randint(-1000,200)
    w5 = round(random.uniform(0.5,1),2)
    return [w1,w2,w3,w4,w5]

# two weights - ari (most important) and silhouette 
creator.create("FitnessMax", base.Fitness, weights=(1.0, 0.9))
creator.create("Individual", list, fitness=creator.FitnessMax)
toolbox.register('expr', generateIndividual)
toolbox.register('individual', tools.initIterate, creator.Individual, toolbox.expr)
toolbox.register('population', tools.initRepeat, list, toolbox.individual)

def evaluate(individual):
    return sum(individual), sum(individual),

# add the cluster number restriction
toolbox.register("evaluate", evaluate)
toolbox.register("mate", tools.cxTwoPoint)
toolbox.register("mutate", tools.mutFlipBit, indpb=0.2)
toolbox.register("select", tools.selTournament, tournsize=40)

#toolbox.register("select", tools.selBest)
#toolbox.register("select", tools.selBest)

def main():
    random.seed(94)

    population = toolbox.population(n=10)

    # CXPB - probabilidade de crossover
    # MUTPB - probabilidade de mutacao
    # NGEN - numero de geracoes
    CXPB, MUTPB = 0.9, 0.01
    
    # STATISTICS
    stats = tools.Statistics(key=lambda ind: ind.fitness.values)
    #stats.register("std", np.std)
    stats.register("min", np.min)
    stats.register("avg", np.mean)
    stats.register("max", np.max)

    # Run GA
    population, logbook = algorithms.eaSimple(population, toolbox, CXPB, MUTPB, 10, stats=stats)

if __name__ == "__main__":
    main()