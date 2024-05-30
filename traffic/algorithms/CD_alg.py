import numpy as np

from algorithms.modified_pc.pc_alg import pc
from causallearn.utils.GraphUtils import GraphUtils


def get_structure_pc(data1, data2):
    row_n = np.array(data2).shape[1]
    data_input = np.concatenate((np.array(data2), np.array(data1)), axis=1)
    cg = pc(data_input, alpha=0.01, indep_test='kci')
    return np.absolute(cg.G.graph[ :row_n, row_n: ])
