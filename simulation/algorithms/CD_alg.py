import time
import numpy as np
import matplotlib.pyplot as plt

from utils.summary import Summarizer
from utils.sampler import select_coreset

from algorithms.modified_pc.pc_alg import pc
from causallearn.utils.GraphUtils import GraphUtils


def select_batch_data(data_list, next_data_list, n_samples, predictor, top_k, has_factorize, sel_method='core_set'):
    rs = np.random.RandomState(0)
    if sel_method == 'core_set':
        sel_id = select_coreset(np.array(data_list), np.array(next_data_list), has_factorize, model=predictor)
    else:
        summarizer = Summarizer.factory(sel_method, rs)
        sel_id = summarizer.build_summary(np.array(data_list), np.array(next_data_list), n_samples, method=sel_method, model=predictor)    
    return sel_id[:top_k]


def get_structure_pc(data1, data2, thresholds=0.05, kci_test=False):
    row_n = data2.shape[1]
    data_input = np.concatenate((data2, data1), axis=1)

    if kci_test:
        cg, _ = pc(data_input, alpha=thresholds, indep_test='kci')
    else:
        cg, _ = pc(data_input, alpha=thresholds)

    return np.absolute(cg.G.graph[ :row_n, row_n: ])
