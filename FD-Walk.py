# -*- coding: UTF-8 -*-
from __future__ import print_function
import numpy as np
import pandas as pd
import os
import time
import datetime
import sys
import pickle
import csv
from scipy.optimize import dual_annealing

class Span:
    """This class represents an edge in the graph

    """
    def __init__(self, caller_turple, callee_turple, metric, is_entry=False):
        self.caller_turple = caller_turple
        self.callee_turple = callee_turple
        self.caller_node = Node(caller_turple)
        self.callee_node = Node(callee_turple)
        
        self.metric = metric

        self.is_entry = is_entry
    
    def __eq__(self, obj):
        if obj == None or obj == 0:
            return False
        return self.caller_turple == obj.caller_turple and self.callee_turple == obj.callee_turple
    
    def __hash__(self):
        return hash(self.caller_turple) + hash(self.callee_turple)

    def normalize_metric(self):
        """transfer raw data to data list
        """
        try:
            request_list = split_raw_data(self.metric[0])
        except:
            request_list = [np.nan]*1440

        try:
            duration_list = split_raw_data(self.metric[1])
        except:
            duration_list = [np.nan]*1440

        try:
            exception_list = split_raw_data(self.metric[2])
        except:
            exception_list = [np.nan]*1440
        
        try:
            timeout_list = split_raw_data(self.metric[3])
        except:
            timeout_list = [np.nan]*1440

        self.qpm = request_list
        self.ec = [np.nan] * 1440
        self.rt = [np.nan] * 1440

        for exception, timeout, index in zip(exception_list, timeout_list, range(0,1440)):
            if (not np.isnan(exception)) and (not np.isnan(timeout)):
                self.ec[index] = exception + timeout
        
        for duration, request, index in zip(duration_list, request_list, range(0, 1440)):
            if (not np.isnan(duration)) and (not np.isnan(duration)):
                self.rt[index] = duration/request

class Graph:
    """This class represents a connected component in the whole graph

    """
    def __init__(self, span_set):
        self.span_set = span_set
        self.span_list = list(span_set)

        self.entry_span_list = []
        for span in self.span_list:
            if span.is_entry:
                self.entry_span_list.append(span)
       
        self.matrix_construct()
        self.random_walk_setup()

    def matrix_construct(self):
        """construct adjancy matrix for the call graph

        """
        temp_method_list = []
        for span in self.span_list:
            temp_method_list.append(span.caller_turple)
            temp_method_list.append(span.callee_turple)
        method_list = list(set(temp_method_list))
        self.node_list = []
        for method in method_list:
            self.node_list.append(Node(method))
        
        for index, node in enumerate(self.node_list):
            self.node_list[index].index = index

        node_dict = {}
        for node in self.node_list:
            node_dict[node.turple] = node

        for span in self.span_list:
            span.caller = node_dict[span.caller_turple]
            span.callee = node_dict[span.callee_turple]

        # construct matrix
        self.adjancy_matrix = []
        for i in range(0, len(self.node_list)):
            temp_list = [0] * len(self.node_list)
            self.adjancy_matrix.append(temp_list)
        for span in self.span_list:
            self.adjancy_matrix[span.caller.index][span.callee.index] = span


    def random_walk_setup(self):
        """Setup for the random walk procedure
        
        """
        self.random_walk_begin_span_list = []
        self.walk_begin_probability_list = []
        for span in self.entry_span_list:
            col_index = span.caller.index
            flag = True
            for row in range(0, len(self.node_list)):
                if self.adjancy_matrix[row][col_index] != 0:
                    flag = False
            if flag:
                self.random_walk_begin_span_list.append(span)
                span.traffic_sum = 0.0
                for i in range(alarm_time-detection_window_width, alarm_time):
                    if not np.isnan(span.qpm[i]):
                        span.traffic_sum += span.qpm[i]
                self.walk_begin_probability_list.append(span.traffic_sum)

        traffic_sum = sum(self.walk_begin_probability_list)
        for index in range(0, len(self.walk_begin_probability_list)):
            self.walk_begin_probability_list[index] /= traffic_sum

        self.walk_begin_probability_list = np.asarray(self.walk_begin_probability_list)

class Node:
    """This class represents one end of an edge(caller or callee)
    
    """
    def __init__(self, turple):
        # turple = (server, service, method, set)
        self.turple = turple
        self.visited_time = 0
    
    def __hash__(self):
        return hash(self.turple)

    def __eq__(self, obj):
        if obj == None:
            return False
        else:
            return self.turple == obj.turple

def compute_regression(graph, upstream_node, current_node, downstream_node):
    upstream_turple_list = callee_data[current_node.turple]
    upstream_qpm_list = []
    result_index = -1
    for index, upstream_turple in enumerate(upstream_turple_list):
        if upstream_node.turple == upstream_turple:
            result_index = index

        upstream_qpm = split_raw_data(caller_callee0[upstream_turple+current_node.turple][0])
        upstream_qpm_list.append(upstream_qpm)

    downstream_qpm = graph.adjancy_matrix[current_node.index][downstream_node.index].qpm

    array_u = list()
    for i in range(0, len(upstream_qpm_list)):
        array_u.append(list())
    array_d = list()

    for index_qpm in range(0, 1440):
        valid_flag = True
        for upstream_qpm in upstream_qpm_list:
            if np.isnan(upstream_qpm[index_qpm]):
                valid_flag = False
        if np.isnan(downstream_qpm[index_qpm]):
            valid_flag = False
        
        if valid_flag:
            for index_up_qpm, upstream_qpm in enumerate(upstream_qpm_list):
                array_u[index_up_qpm].append(upstream_qpm[index_qpm])
            array_d.append(downstream_qpm[index_qpm])

    # lack of data
    if len(array_d) == 0:
        return 0.0
    else:
        array_d = np.array(array_d)
        array_us = np.array(array_u)
        X = np.transpose(array_us)
        result = linear_regression_func(X, array_d)
        if result[result_index]>=1:
            return 1
        else:
            return result[result_index]
    
def linear_regression_func(A1,b1): 
    # with constraint: the regression coefficient is not less than 0
    num_x = np.shape(A1)[1]
    global regression_param
    theta = []
    for i in range(0, num_x):
        theta.append(1/num_x)

    def my_func(x):
        # loss
        ls = (b1-np.dot(A1,x))**2
        result = np.sum(ls)
        return result
    bnds = [(0,regression_param)]
    for i in range(num_x-1):
        bnds.append((0,regression_param))

    res1 = dual_annealing(my_func, bounds = bnds, maxiter=1000)
    return res1.x

class Root_Cause:

    def __init__(self, turple, visited_count):
        self.turple = turple
        self.visited_count = visited_count

class Root_Cause_Server:

    def __init__(self, server, root_cause_list):
        self.server = server
        self.root_cause_list = root_cause_list

        self.visited_count_sum = 0
        self.visited_count_average = 0

        for root_cause in self.root_cause_list:
            self.visited_count_sum += root_cause.visited_count
        
        self.visited_count_average = (self.visited_count_sum * 1.0) / len(self.root_cause_list)

class Random_Walk_Choice:

    def __init__(self, current_node, downstream_node, probability, reflexive=False):
        self.current_node = current_node
        self.downstream_node = downstream_node
        self.probability = probability
        self.reflexive = reflexive

def read_files():
    """Read the raw data into memory

    """
    files0 = os.listdir(data_path0)[1:]
    for file in files0:
        file_path = data_path0 + file
        try:
            for i, row in pd.read_csv(file_path, sep='|', on_bad_lines='skip').iterrows():
                caller_turple = (row[1], row[2], row[3], row[4])
                callee_turple = (row[5], row[6], row[7], row[8])
                caller_callee_turple = caller_turple + callee_turple

                time_series = [row[9], row[10], row[12], row[13]]
                
                if caller_data.get(caller_turple) == None:
                    caller_data[caller_turple] = []
                caller_data[caller_turple].append(callee_turple)

                if callee_data.get(callee_turple) == None:
                    callee_data[callee_turple] = []
                callee_data[callee_turple].append(caller_turple)

                caller_callee0[caller_callee_turple] = time_series
        except:
            pass

def split_raw_data(data):
    time_list = [np.nan]*1440
    data_points = data.split(',')

    for data_point in data_points:
        time_list[int(data_point.split(':')[0])] = float(data_point.split(':')[1])

    return time_list

def get_downstream_node_list(graph, node):
    return_node_list = []
    row = node.index
    for col in range(0, len(graph.node_list)):
        if graph.adjancy_matrix[row][col] != 0:
            return_node_list.append(graph.adjancy_matrix[row][col].callee)

    return return_node_list

def get_root_cause_server_list(node_list):
    root_cause_list = []
    for node in node_list:
        root_cause_list.append(Root_Cause(node.turple, node.visited_time))
    
    root_cause_dict = {}
    root_cause_server_list = []
    for root_cause in root_cause_list:
        if root_cause_dict.get(root_cause.turple[0]) == None:
            root_cause_dict[root_cause.turple[0]] = list()
        root_cause_dict[root_cause.turple[0]].append(root_cause)
    for server, temp_root_cause_list in list(root_cause_dict.items()):
        root_cause_server_list.append(Root_Cause_Server(server, temp_root_cause_list))

    root_cause_server_list.sort(key=lambda x:x.visited_count_sum, reverse=True)
    return root_cause_server_list

def random_walk_start(graph):
    # choose start node
    if len(graph.random_walk_begin_span_list) == 0:
        return
    begin_span = np.random.choice(graph.random_walk_begin_span_list, p=graph.walk_begin_probability_list)
    random_walk(graph, last_node=begin_span.caller, current_node=begin_span.callee)

def random_walk(graph, last_node, current_node):

    if probability_cache_dict.get(current_node.turple) != None:
        transfer_probability_list = probability_cache_dict[current_node.turple][0]
        transfer_choice_list = probability_cache_dict[current_node.turple][1]
    else:
        downstream_node_list = get_downstream_node_list(graph, current_node)
        # no downstream nodes => end
        if len(downstream_node_list) == 0:
            current_node.visited_time += 1
            return

        transfer_probability_list = []
        transfer_choice_list = []

        for downstream_node in downstream_node_list:
            downstream_span = graph.adjancy_matrix[current_node.index][downstream_node.index]
            downstream_span.ec_sum = 0.0
            downstream_span.traffic_sum = 0.0
            for i in range(alarm_time-detection_window_width, alarm_time):
                if not np.isnan(downstream_span.ec[i]):
                    downstream_span.ec_sum += downstream_span.ec[i]
                if not np.isnan(downstream_span.qpm[i]):
                    downstream_span.traffic_sum += downstream_span.qpm[i]

            if downstream_span.traffic_sum != 0:
                downstream_span.ec_beta = downstream_span.ec_sum/downstream_span.traffic_sum
                transfer_probability = 1 - ((1 - downstream_span.ec_sum/downstream_span.traffic_sum)**compute_regression(graph=graph, upstream_node=last_node, current_node=current_node, downstream_node=downstream_node))
            else:
                transfer_probability = 0.0
                downstream_span.ec_beta = 0.0

            transfer_choice = Random_Walk_Choice(current_node=current_node, downstream_node=downstream_node, probability=transfer_probability)
            transfer_probability_list.append(transfer_probability)
            transfer_choice_list.append(transfer_choice)
        
        upstream_span = graph.adjancy_matrix[last_node.index][current_node.index]
        upstream_span.traffic_sum = 0.0
        upstream_span.ec_sum = 0.0
        for i in range(alarm_time - detection_window_width, alarm_time):
            if not np.isnan(upstream_span.qpm[i]):
                upstream_span.traffic_sum += upstream_span.qpm[i]
            if not np.isnan(upstream_span.ec[i]):
                upstream_span.ec_sum += upstream_span.ec[i]
        
        left_part = (upstream_span.traffic_sum - upstream_span.ec_sum) / upstream_span.traffic_sum
        right_part = 1.0
        for index, downstream_node in enumerate(downstream_node_list):
            downstream_span = graph.adjancy_matrix[current_node.index][downstream_node.index]
            right_part *= ((1 - downstream_span.ec_beta)**compute_regression(graph=graph, upstream_node=last_node, current_node=current_node, downstream_node=downstream_node))
        if right_part == 0:
            transfer_to_itself_probability = 0.0
        else:
            transfer_to_itself_probability = -(left_part / right_part -1)
            
        if transfer_to_itself_probability < 0:
            transfer_to_itself_probability = 0.0

        transfer_probability_list.append(transfer_to_itself_probability)
        transfer_choice = Random_Walk_Choice(current_node=current_node, downstream_node=current_node, probability=transfer_to_itself_probability, reflexive=True)
        transfer_choice_list.append(transfer_choice)

        # normalize
        sum_probability = 0.0
        for transfer_probability in transfer_probability_list:
            sum_probability += transfer_probability
        if sum_probability == 0:
            current_node.visited_time += 1
            probability_cache_dict[current_node.turple] = [transfer_probability_list, transfer_choice_list]
            return

        for index in range(0, len(transfer_probability_list)):
            transfer_probability_list[index] /= sum_probability
            transfer_choice_list[index].probability /= sum_probability

        probability_cache_dict[current_node.turple] = [transfer_probability_list, transfer_choice_list]

    if sum(transfer_probability_list) == 0:
        current_node.visited_time += 1
        return
    transfer_choice = np.random.choice(transfer_choice_list, p=transfer_probability_list)
    if transfer_choice.reflexive:
        current_node.visited_time += 1
        return
    else:
        # recursion
        random_walk(graph=graph, last_node=transfer_choice.current_node, current_node=transfer_choice.downstream_node)
    
def ec_valid_in_window(span):
    sum_ec = 0
    for i in range(alarm_time-detection_window_width, alarm_time):
        if not np.isnan(span.ec[i]):
            sum_ec += span.ec[i]
    if sum_ec >= ec_count_threshold:
        return True
    else:
        return False

if __name__ == '__main__':
    
    # number of randoms walks to be performed
    random_walk_iters = 10000

    # ec filter threshold
    ec_count_threshold = 2

    # width of the detection window
    detection_window_width = 10

    # up limit for linear regression
    regression_param = sys.maxsize

    # slo alarm information
    alarm_server_str = ''
    root_cause_server_str = ''
    alarm_time = ''
    alarm_date = None
    
    # record invoking rekationships
    caller_data = {}
    callee_data = {}

    # record time series data
    caller_callee0 = {}

    # raw data path
    data_path0 = ''

    read_files()

    entry_span_list = []
    for caller_turple in list(caller_data.keys()):
        if caller_turple[0] == alarm_server_str:
            for callee_turple in caller_data[caller_turple]:
                span = Span(caller_turple, callee_turple, caller_callee0[caller_turple+callee_turple], is_entry=True)
                span.normalize_metric()
                if ec_valid_in_window(span):
                    entry_span_list.append(span)
                
    for callee_turple in list(callee_data.keys()):
        if callee_turple[0] == alarm_server_str:
            for caller_turple in callee_data[callee_turple]:
                span = Span(caller_turple, callee_turple, caller_callee0[caller_turple+callee_turple], is_entry=True)
                span.normalize_metric()
                if ec_valid_in_window(span):
                    entry_span_list.append(span) 

    # extend from the entry calls
    all_graph_span_set = set()
    for entry_span in entry_span_list:
        cache_span_queue = [entry_span]
        cache_span_set = set()
        cache_span_set.add(entry_span)
        while len(cache_span_queue):
            top_span = cache_span_queue.pop(0)
            all_graph_span_set.add(top_span)
            downstream_span_list = []
            try:
                downstream_turple_list = caller_data[top_span.callee_turple]
                for downstream_turple in downstream_turple_list:
                    span = Span(top_span.callee_turple, downstream_turple, caller_callee0[top_span.callee_turple + downstream_turple])
                    span.normalize_metric()
                    downstream_span_list.append(span)
            except KeyError:
                downstream_span_list = []

            for span in downstream_span_list:
                if span not in cache_span_set:
                    cache_span_set.add(span)
                    if ec_valid_in_window(span):
                        cache_span_queue.append(span)
                        all_graph_span_set.add(span)

    np.random.seed(0)
    all_graph = Graph(all_graph_span_set)
    probability_cache_dict = {}
    for i in range(0, random_walk_iters):
        random_walk_start(all_graph)

    all_root_cause_server_list = get_root_cause_server_list(all_graph.node_list)
    all_index = -1
    for index, root_cause_server in enumerate(all_root_cause_server_list):
        if root_cause_server.server == root_cause_server_str:
            all_index = index+1
            print('Found root cause:', index+1,'/'+str(len(all_root_cause_server_list)))
    print('#########')

    csv_file = open('result.csv','a')
    result_writer = csv.writer(csv_file)
    line = []
    if all_index != -1:
        line.append(str(all_index)+'/'+str(len(all_root_cause_server_list)))
    else:
        line.append('no')
    result_writer.writerow(line)
    csv_file.close()
