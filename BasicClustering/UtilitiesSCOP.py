import os
import numpy as np

def getSCOPSample(sample):
    scop_names_path = "C:/ShareSSD/scop/scope/dir.cla.scope.2.07-stable.txt"

    to_write = set()

    with open(scop_names_path) as fp:
        line = fp.readline()
        while line:
            scop_id = str(line).split('\t')
            if sample in str(scop_id[3]):
                to_write.add(scop_id[0]+'.ent '+scop_id[3])
            line = fp.readline()

    to_write = list(to_write)

    with open('C:/ShareSSD/scop/sample_'+sample, 'w') as nf:
        for pair in to_write:
            nf.write(pair+'\n')

def getUniqueClassifications(sample):
    scop_sample_path= "C:/ShareSSD/scop/samples/sample_"+sample #a.1 for instance

    superfamilies = set()

    with open(scop_sample_path) as fp:
        line = fp.readline()
        while line:
            #parsed = str(line).split(' ')[1].strip().split('.')[-1]
            parsed = str(line).split(' ')[1].strip()
            superfamilies.add(parsed)
            line = fp.readline()

    return len(superfamilies)

def getDomainLabels(domains):
    scop_names_path = "C:/ShareSSD/scop/scope/dir.cla.scope.2.07-stable.txt"

    labels = []

    with open(scop_names_path, 'r') as fp:
        for domain in domains:
            domain = str(domain).replace('.ent','')
            fp.seek(0)
            line = fp.readline()
            while line:
                if domain in line:
                    #parsed = str(line).strip().split('\t')[3].split('.')[-1]
                    parsed = str(line).strip().split('\t')[3]
                    labels.append(str(parsed))
                    break
                line = fp.readline()
    
    return labels
