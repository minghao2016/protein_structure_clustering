

import os

patterns = ['a.1.1','b.1.1']
filename = 'x'.join(patterns)

scop_names_path = "C:/ShareSSD/scop/scope/dir.cla.scope.2.07-stable.txt"
path_to_sample = 'C:/ShareSSD/scop/tests/'

# write sample list
nf1 = open(path_to_sample+'sample_'+filename,'w')
# write sample list with classifications
nf2 = open(path_to_sample+'sample_structures_'+filename,'w')

with open(scop_names_path, 'r') as fp:

    line = fp.readline()

    while line:

        parsed = str(line).strip().split('\t')
        structure = parsed[0]
        classification = parsed[3]

        if any(pattern in classification for pattern in patterns):
            nf1.write(structure+'.ent\n')
            nf2.write(structure+'.ent '+classification+'\n')
            
        line = fp.readline()

nf1.close()
nf2.close()


