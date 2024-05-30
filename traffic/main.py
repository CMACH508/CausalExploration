# -*- coding: utf-8 -*-

'''
SEED: random number for initializing the experiment
setting_memo: the folder name for this experiment
The conf, data files will should be placed in conf/setting_memo, data/setting_memo respectively
The records, model files will be generated in records/setting_memo, model/setting_memo respectively
'''

import os
import json

import time
import random

import exploration
import numpy as np
import tensorflow as tf

SEED = 2023
random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)

setting_memo = "one_run"

# first column: for train, second column: for spre_train
list_traffic_files = [
    [["cross.2phases_rou1_switch_rou0.xml"], ["cross.2phases_rou1_switch_rou0.xml"]],
    [["cross.2phases_rou01_equal_300s.xml"], ["cross.2phases_rou01_equal_300s.xml"]],
    [["cross.2phases_rou01_unequal_5_300s.xml"], ["cross.2phases_rou01_unequal_5_300s.xml"]],
    [["cross.all_synthetic.rou.xml"], ["cross.all_synthetic.rou.xml"]],
]

list_model_name = [
                   "Deeplight",
                   ]

PATH_TO_CONF = os.path.join("conf", setting_memo)

sumoBinary = r"/usr/bin/sumo-gui"
sumoCmd = [sumoBinary,
           '-c',
           r'{0}/data/{1}/cross.sumocfg'.format(os.path.split(os.path.realpath(__file__))[0], setting_memo)]
sumoCmd_pretrain = [sumoBinary,
                    '-c',
                    r'{0}/data/{1}/cross_pretrain.sumocfg'.format(
                        os.path.split(os.path.realpath(__file__))[0], setting_memo)]

sumoBinary_nogui = r"/usr/bin/sumo"
sumoCmd_nogui = [sumoBinary_nogui,
                 '-c',
                 r'{0}/data/{1}/cross.sumocfg'.format(
                     os.path.split(os.path.realpath(__file__))[0], setting_memo)]
sumoCmd_nogui_pretrain = [sumoBinary_nogui,
                          '-c',
                          r'{0}/data/{1}/cross_pretrain.sumocfg'.format(
                              os.path.split(os.path.realpath(__file__))[0], setting_memo)]

data_list, next_data_list = [],[]

for model_name in list_model_name:
    for traffic_file, traffic_file_pretrain in list_traffic_files:
        dic_exp = json.load(open(os.path.join(PATH_TO_CONF, "exp.conf"), "r"))
        dic_exp["MODEL_NAME"] = model_name
        dic_exp["TRAFFIC_FILE"] = traffic_file
        dic_exp["TRAFFIC_FILE_PRETRAIN"] = traffic_file_pretrain
        if "real" in traffic_file[0]:
            dic_exp["RUN_COUNTS"] = 864
        elif "2phase" in traffic_file[0]:
            dic_exp["RUN_COUNTS"] = 1586
        elif "synthetic" in traffic_file[0]:
            dic_exp["RUN_COUNTS"] = 216
        json.dump(dic_exp, open(os.path.join(PATH_TO_CONF, "exp.conf"), "w"), indent=4)

        dic_sumo = json.load(open(os.path.join(PATH_TO_CONF, "sumo_agent.conf"), "r"))
        if model_name == "Deeplight":
            dic_sumo["MIN_ACTION_TIME"] = 5
        else:
            dic_sumo["MIN_ACTION_TIME"] = 1
        json.dump(dic_sumo, open(os.path.join(PATH_TO_CONF, "sumo_agent.conf"), "w"), indent=4)

        prefix = "{0}_{1}_{2}_{3}".format(
            dic_exp["MODEL_NAME"],
            dic_exp["TRAFFIC_FILE"],
            dic_exp["TRAFFIC_FILE_PRETRAIN"],
            time.strftime('%m_%d_%H_%M_%S_', time.localtime(time.time())) + "seed_%d" % SEED
        )

        if traffic_file != ['cross.all_synthetic.rou.xml']:
            temp_data_list, temp_next_data_list = exploration.main(memo=setting_memo, f_prefix=prefix, sumo_cmd_str=sumoCmd_nogui, sumo_cmd_pretrain_str=sumoCmd_nogui_pretrain)
            data_list += (temp_data_list)
            next_data_list += (temp_next_data_list)
            print("finished {0}".format(traffic_file))
        else:
            exploration.causal_discovery(data_list, next_data_list)
            exploration.main(memo=setting_memo, f_prefix=prefix, sumo_cmd_str=sumoCmd_nogui, sumo_cmd_pretrain_str=sumoCmd_nogui_pretrain, collect_data=False)
        
    print ("finished {0}".format(model_name))
