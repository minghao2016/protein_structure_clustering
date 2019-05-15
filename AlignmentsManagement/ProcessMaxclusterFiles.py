

import MatrixFunctions as mf
import numpy as np
import os

def readSimilaritiesToMatrix(sample, measure):
    
    path_to_matrix = 'C:/ShareSSD/scop/data/values_'+sample+'_'+measure
    path_to_domains = 'C:/ShareSSD/scop/data/domains_'+sample

    counter = 0
    matrix = []
    row = []

    with open(path_to_matrix, 'r') as fp:

        domains = set()

        line = fp.readline()
        while line:

            if 'END' in line:
                break

            parsed = str(line).strip().split(' ')
            structure1 = parsed[0]
            structure2 = parsed[1]
            value = float(parsed[2])

            domains.add(structure1)
            domains.add(structure2)

            # add the respective amount of zeroes to the current row
            counter += 1
            i = 0
            while i < counter:
                row.append(0)
                i += 1
            i = 0

            # track the current structure and read its alignments
            current_row = parsed[0]
            while current_row == parsed[0] and line:
                row.append(value)
                line = fp.readline()
                if 'END' in line:
                    break
                parsed = str(line).strip().split(' ')
                print(parsed)
                value = float(parsed[2])

            matrix.append(row)
            row = []

        line = fp.readline()

    counter += 1

    i = 0
    while i < counter:
        row.append(0)
        i += 1
    i = 0
    matrix.append(row)

    matrix = np.asmatrix(matrix)

    # symmetrize and write results to file
    matrix = mf.symmetrizeMatrix(matrix)
    matrix = np.matrix(matrix)
    matrix.dump("C:/ShareSSD/scop/data/matrix_"+sample+'_'+measure)

    # write domain list to file
    if not os.path.isfile(path_to_domains):
        with open(path_to_domains, 'w') as nf:
            domains = list(domains)
            for domain in domains:
                nf.write(domain+'\n')
            nf.write('END')

# Use this to read MaxSub and TM-score files
def readDistancesMaxsub(sample, measure):
    path_to_matrix = 'C:/ShareSSD/scop/data/sim_'+sample+'_'+measure
    path_to_values = 'C:/ShareSSD/scop/data/values_'+sample+'_'+measure

    counter = 0
    matrix = []
    row = []

    with open(path_to_matrix, 'r') as fp:

        size = 0
        domains = []

        #get structures
        line = fp.readline()
        while line:
            if 'PDB  :' in str(line) or 'PDB:' in str(line): 
                if '#' not in str(line):
                    size += 1
                    domain = str(line).strip().split()[-1].split('/')[-1]
                    domains.append(str(domain))
            if '# Maxsub records' in str(line):
                break
            line = fp.readline()

        while line:
            if 'MS :' in str(line): 
                if '#' not in str(line):

                    parsed = str(line).strip().split()
                    current_row = parsed[2]
                    value = float(parsed[4])

                    while current_row == parsed[2]:

                        row.append(value)
                        line = fp.readline()
                        parsed = str(line).strip().split()
                        # value is the average between the two
                        value = (float(parsed[4])+float(parsed[5]))/2

                    counter += 1
                    matrix.append(row)
                    row = []    

                    #REVER ORDEM DAS OPERACOES
                    i = 0
                    while i < counter:
                        row.append(0)
                        i += 1
                    i = 0
                    row.append(value)

            line = fp.readline()

    row = []
    i = 0
    while i < counter:
        row.append(0)
        i += 1
    i = 0
    row.append(1.0)
    matrix.append(row)

    matrix = np.asmatrix(matrix)
    print(matrix)

    # if the diagonal is 1 symmetrization does not work properly
    # set to 0 temporarilly
    n, m = matrix.shape
    for i in range(n):
        matrix[i,i] = 0

    matrix = mf.symmetrizeMatrix(matrix)

    for i in range(n):
        matrix[i,i] = 1

    print(matrix)

    # write values to be used in kernel density estimation
    with open(path_to_values, 'w') as nf:
        for i in range(0,n):
            for j in range(0,m):
                nf.write(str(matrix[i,j])+'\n')

    matrix.dump('C:/ShareSSD/scop/tests/matrix_'+sample+'_'+measure)

# Use this to read RMSD and GDT files
def readDistances(sample, measure):
    path_to_matrix = 'C:/ShareSSD/scop/data/sim_'+sample+'_'+measure
    path_to_values = 'C:/ShareSSD/scop/data/values_'+sample+'_'+measure

    counter = 0
    matrix = []

    with open(path_to_matrix, 'r') as fp:

        size = 0
        domains = []

        #get structures
        line = fp.readline()
        while line:
            if 'PDB  :' in str(line): 
                if '#' not in str(line):
                    size += 1
                    domain = str(line).strip().split()[-1].split('/')[-1]
                    domains.append(str(domain))
            if 'Distance records' in str(line):
                break
            line = fp.readline()

        #get distance matrix
        while line:
            if 'DIST :' in str(line): 
                if '#' not in str(line):
                    print(line)
                    parsed = str(line).strip().split()
                    current_row = parsed[2]
                    value = float(parsed[4])
            
                    while current_row == parsed[2]:
                    
                        matrix.append(value)
                        line = fp.readline()
                        parsed = str(line).strip().split()
                        value = float(parsed[4])

                    counter += 1

                    i = 0
                    while i < counter:
                        matrix.append(0)
                        i += 1
                    i = 0

                    matrix.append(value)

            line = fp.readline()

    counter += 1

    i = 0
    while i < counter:
        matrix.append(0)
        i += 1
    i = 0

    matrix = np.asmatrix(matrix)
    matrix = matrix.reshape(size,size-1)
    n, m = matrix.shape 
    X0 = np.zeros((n,1))
    matrix = np.hstack((X0,matrix))

    matrix = mf.symmetrizeMatrix(matrix)

    # save kernel values
    with open(path_to_values, 'w') as nf:
        for i in range(0,n):
            for j in range(0,m):
                nf.write(str(matrix[i,j])+'\n')

    matrix.dump('C:/ShareSSD/scop/data/matrix_'+sample+'_'+measure)
    return domains, matrix