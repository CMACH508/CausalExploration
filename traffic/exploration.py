# -*- coding: utf-8 -*-

import os
import time
import math
import copy
import json
import shutil
import numpy as np

import xml.etree.ElementTree as ET

from algorithms.sumo_agent import SumoAgent
from algorithms.CD_alg import get_structure_pc
from algorithms.explore_agent import DeeplightAgent

from models.CausalWorldModel import World_model


class TrafficLightDQN:

    DIC_AGENTS = {
        "Deeplight": DeeplightAgent,
    }

    NO_PRETRAIN_AGENTS = []

    class ParaSet:

        def __init__(self, dic_paras):
            for key, value in dic_paras.items():
                setattr(self, key, value)

    class PathSet:

        # ======================================= conf files ========================================
        EXP_CONF = "exp.conf"
        SUMO_AGENT_CONF = "sumo_agent.conf"
        PATH_TO_CFG_TMP = os.path.join("data", "tmp")
        # ======================================= conf files ========================================

        def __init__(self, path_to_conf, path_to_data, path_to_output, path_to_model):

            self.PATH_TO_CONF = path_to_conf
            self.PATH_TO_DATA = path_to_data
            self.PATH_TO_OUTPUT = path_to_output
            self.PATH_TO_MODEL = path_to_model

            if not os.path.exists(self.PATH_TO_OUTPUT):
                os.makedirs(self.PATH_TO_OUTPUT)
            if not os.path.exists(self.PATH_TO_MODEL):
                os.makedirs(self.PATH_TO_MODEL)

            dic_paras = json.load(open(os.path.join(self.PATH_TO_CONF, self.EXP_CONF), "r"))
            self.AGENT_CONF = "{0}_agent.conf".format(dic_paras["MODEL_NAME"].lower())
            self.TRAFFIC_FILE = dic_paras["TRAFFIC_FILE"]
            self.TRAFFIC_FILE_PRETRAIN = dic_paras["TRAFFIC_FILE_PRETRAIN"]

    def __init__(self, memo, f_prefix, if_test=False):

        self.path_set = self.PathSet(os.path.join("conf", memo),
                                     os.path.join("data", memo),
                                     os.path.join("outputs", "records", memo, f_prefix),
                                     os.path.join("outputs", "model", memo, f_prefix))

        self.para_set = self.load_conf(conf_file=os.path.join(self.path_set.PATH_TO_CONF, self.path_set.EXP_CONF))
        shutil.copy(
            os.path.join(self.path_set.PATH_TO_CONF, self.path_set.EXP_CONF),
            os.path.join(self.path_set.PATH_TO_OUTPUT, self.path_set.EXP_CONF))

        self.agent = self.DIC_AGENTS[self.para_set.MODEL_NAME](num_phases=2,
                                                               num_actions=2,
                                                               path_set=self.path_set,
                                                               if_test=if_test)

    def load_conf(self, conf_file):

        dic_paras = json.load(open(conf_file, "r"))
        return self.ParaSet(dic_paras)

    def check_if_need_pretrain(self):

        if self.para_set.MODEL_NAME in self.NO_PRETRAIN_AGENTS:
            return False
        else:
            return True

    def _generate_pre_train_ratios(self, phase_min_time, em_phase):
        phase_traffic_ratios = [phase_min_time]

        # generate how many varients for each phase
        for i, phase_time in enumerate(phase_min_time):
            if i == em_phase:
                for j in range(1, 5, 1):
                    gen_phase_time = copy.deepcopy(phase_min_time)
                    gen_phase_time[i] += j
                    phase_traffic_ratios.append(gen_phase_time)
            else:
                # pass
                for j in range(1, 5, 1):
                    gen_phase_time = copy.deepcopy(phase_min_time)
                    gen_phase_time[i] += j
                    phase_traffic_ratios.append(gen_phase_time)
            for j in range(5, 20, 5):
                gen_phase_time = copy.deepcopy(phase_min_time)
                gen_phase_time[i] += j
                phase_traffic_ratios.append(gen_phase_time)

        return phase_traffic_ratios

    @staticmethod
    def _set_traffic_file(sumo_config_file_tmp_name, sumo_config_file_output_name, list_traffic_file_name):

        # update sumocfg
        sumo_cfg = ET.parse(sumo_config_file_tmp_name)
        config_node = sumo_cfg.getroot()
        input_node = config_node.find("input")
        for route_files in input_node.findall("route-files"):
            input_node.remove(route_files)
        input_node.append(
            ET.Element("route-files", attrib={"value": ",".join(list_traffic_file_name)}))
        sumo_cfg.write(sumo_config_file_output_name)

    def set_traffic_file(self):

        self._set_traffic_file(
            os.path.join(self.path_set.PATH_TO_DATA, "cross_pretrain.sumocfg"),
            os.path.join(self.path_set.PATH_TO_DATA, "cross_pretrain.sumocfg"),
            self.para_set.TRAFFIC_FILE_PRETRAIN)
        self._set_traffic_file(
            os.path.join(self.path_set.PATH_TO_DATA, "cross.sumocfg"),
            os.path.join(self.path_set.PATH_TO_DATA, "cross.sumocfg"),
            self.para_set.TRAFFIC_FILE)
        for file_name in self.path_set.TRAFFIC_FILE_PRETRAIN:
            shutil.copy(
                    os.path.join(self.path_set.PATH_TO_DATA, file_name),
                    os.path.join(self.path_set.PATH_TO_OUTPUT, file_name))
        for file_name in self.path_set.TRAFFIC_FILE:
            shutil.copy(
                os.path.join(self.path_set.PATH_TO_DATA, file_name),
                os.path.join(self.path_set.PATH_TO_OUTPUT, file_name))

    def train(self, sumo_cmd_str, if_pretrain, use_average, collect_data, causal_graph=None):

        if if_pretrain:
            total_run_cnt = self.para_set.RUN_COUNTS_PRETRAIN
            phase_traffic_ratios = self._generate_pre_train_ratios(self.para_set.BASE_RATIO, em_phase=0)  # en_phase=0
            pre_train_count_per_ratio = math.ceil(total_run_cnt / len(phase_traffic_ratios))
            ind_phase_time = 0
        else:
            total_run_cnt = self.para_set.RUN_COUNTS


        # generate World model
        if not collect_data:
            w_model = World_model()
            w_model._build(causal_graph)

        # start sumo
        s_agent = SumoAgent(sumo_cmd_str,
                            self.path_set)
        current_time = s_agent.get_current_time()  # in seconds

        data_list, next_data_list = [], []

        # start experiment
        while current_time < total_run_cnt:

            if if_pretrain:
                if current_time > pre_train_count_per_ratio:
                    print("Terminal occured. Episode end.")
                    s_agent.end_sumo()
                    ind_phase_time += 1
                    if ind_phase_time >= len(phase_traffic_ratios):
                        break

                    s_agent = SumoAgent(sumo_cmd_str,
                            self.path_set)
                    current_time = s_agent.get_current_time()  # in seconds

                phase_time_now = phase_traffic_ratios[ind_phase_time]

            # get state
            state = s_agent.get_observation()
            state = self.agent.get_state(state, current_time)

            if if_pretrain:
                _, q_values = self.agent.choose(count=current_time, if_pretrain=if_pretrain)
                if state.time_this_phase[0][0] < phase_time_now[state.cur_phase[0][0]]:
                    action_pred = 0
                else:
                    action_pred = 1
            else:
                # get action based on e-greedy, combine current state
                action_pred, q_values = self.agent.choose(count=current_time, if_pretrain=if_pretrain)
            
            # get reward from sumo agent
            reward, action = s_agent.take_action(action_pred)

            # get next state
            next_state = s_agent.get_observation()
            next_state = self.agent.get_next_state(next_state, current_time)

            if collect_data:
                for i in range (12):
                    temp_state = np.hstack((
                        state.queue_length[0][i],
                        state.num_of_vehicles[0][i],
                        state.waiting_time[0][i],
                        state.cur_phase[0],
                        state.next_phase[0],
                        action_pred
                    ))
                    if np.count_nonzero(temp_state) > 2:
                        data_list.append(temp_state)
                        next_data_list.append(
                            np.hstack((
                                next_state.queue_length[0][i],
                                next_state.num_of_vehicles[0][i],
                                next_state.waiting_time[0][i],
                                next_state.cur_phase[0],
                                next_state.next_phase[0]
                            ))
                        )
            else:
                itsc_r = w_model.get_reward(
                    np.hstack((
                        state.queue_length,
                        state.num_of_vehicles,
                        state.waiting_time,
                        state.cur_phase,
                        state.next_phase,
                        action_pred.reshape(1, -1)
                    )),
                    np.hstack((
                        next_state.queue_length,
                        next_state.num_of_vehicles,
                        next_state.waiting_time,
                        next_state.cur_phase,
                        next_state.next_phase
                    ))
                )

            # remember
            self.agent.remember(state, action, reward, next_state)
            if not collect_data:
                w_model.store_data(
                    np.hstack((
                        state.queue_length,
                        state.num_of_vehicles,
                        state.waiting_time, state.cur_phase,
                        state.next_phase,
                        action_pred.reshape(1, -1)
                    )),
                    np.hstack((
                        next_state.queue_length,
                        next_state.num_of_vehicles,
                        next_state.waiting_time,
                        next_state.cur_phase,
                        next_state.next_phase
                    ))
                )

            current_time = s_agent.get_current_time()  # in seconds

            if not if_pretrain:
                # update network
                self.agent.update_network(if_pretrain, use_average, current_time)
                self.agent.update_network_bar()

                #update world model
                if not collect_data:
                    w_model.train(current_time)

        if if_pretrain:
            self.agent.set_update_outdated()
            self.agent.update_network(if_pretrain, use_average, current_time)
            self.agent.update_network_bar()
        self.agent.reset_update_count()

        s_agent.end_sumo()
        print("END")
        return data_list, next_data_list


def causal_discovery(data_list, next_data_list):
    causal_graph_pred = get_structure_pc(np.array(data_list).reshape(-1, 6), np.array(next_data_list).reshape(-1, 5))


def main(memo, f_prefix, sumo_cmd_str, sumo_cmd_pretrain_str, collect_data=True):
    player = TrafficLightDQN(memo, f_prefix)
    player.set_traffic_file()

    if collect_data:
        data_list, next_data_list = player.train(sumo_cmd_str, if_pretrain=False, use_average=False, collect_data=True)
        return data_list, next_data_list
    else:
        causal_graph = np.loadtxt("./pre_trained/estd_graph.txt")
        player.train(sumo_cmd_str, if_pretrain=False, use_average=False, collect_data=False, causal_graph=causal_graph)
