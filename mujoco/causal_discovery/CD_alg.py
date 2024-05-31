import numpy as np

from causal_discovery.modified_pc.pc_alg import pc
from causallearn.utils.GraphUtils import GraphUtils


def get_structure_pc(data1, data2):
    row_n = np.array(data2).shape[1]
    data_input = np.concatenate((np.array(data2), np.array(data1)), axis=1)
    num_rows_to_select = 512
    random_indices = np.random.choice(data_input.shape[0], num_rows_to_select, replace=False)
    selected_data = data_input[random_indices]
    cg = pc(selected_data, alpha=0.05, indep_test='kci')
    return np.absolute(cg.G.graph[ :row_n, row_n: ])
