import math
from scipy.sparse import csr_matrix
import networkx as nx
import torch
from matplotlib import pyplot as plt
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import random
from utils.datareader import GraphData, DataReader
import torch.nn.functional as F
import math
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from utils.batch import collate_batch
from config import parse_args
# 设置设备
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

args = parse_args()
seed = args.seed
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)




def updated_dr(args, dr):

    indices_to_remove = []
    for index, (adj_matrix, label, features) in enumerate(zip(dr.data['adj_list'], dr.data['labels'], dr.data['features'])):
        num_nodes = len(adj_matrix)
        if num_nodes < 5:
            indices_to_remove.append(index)

    for index in reversed(indices_to_remove):
        del dr.data['adj_list'][index]
        dr.data['labels'] = np.delete(dr.data['labels'], index, axis=0)
        del dr.data['features'][index]

        # for i in range(len(indices_to_remove)):
        #     if indices_to_remove[i] > index:
        #         indices_to_remove[i] -= 1

    total_indices = list(range(len(dr.data['adj_list'])))
    random.seed(args.seed)
    random.shuffle(total_indices)

    half_length = len(total_indices) // 2
    dr.data['splits']['train'] = total_indices[:half_length]
    dr.data['splits']['test'] = total_indices[half_length:]

    dr.filtered_indices = indices_to_remove

    return dr


def Select_trigger_inj_position(args, gdata_train, feat_dim):
    trans = int(feat_dim * args.inject_position_transfer)
    rand_num = random.randint(0, trans)

    Graph_Class_0_features = {}
    Graph_Class_1_features = {}
    for i in range(len(gdata_train)):
        if gdata_train.labels[i] == 0:
            Graph_Class_0_features[i] = np.sum(gdata_train.features[i], axis=0)
        elif gdata_train.labels[i] == 1:
            Graph_Class_1_features[i] = np.sum(gdata_train.features[i], axis=0)

    Class_0_idx = {}
    Class_1_idx = {}
    if args.inject_position == 'MIA':
        for k, v in Graph_Class_0_features.items():
            sorted_idx = np.argsort(v)
            Class_0_idx[k] = sorted_idx[-args.trig_size-rand_num:-rand_num]
        for k, v in Graph_Class_1_features.items():
            sorted_idx = np.argsort(v)
            Class_1_idx[k] = sorted_idx[-args.trig_size-rand_num:-rand_num]
    elif args.inject_position == 'LIA':
        for k, v in Graph_Class_0_features.items():
            sorted_idx = np.argsort(v)
            Class_0_idx[k] = sorted_idx[rand_num:args.trig_size+rand_num]
        for k, v in Graph_Class_1_features.items():
            sorted_idx = np.argsort(v)
            Class_1_idx[k] = sorted_idx[rand_num:args.trig_size+rand_num]

    def count_occurrences(idx_dict):
        idx_count = {}
        for idx_list in idx_dict.values():
            for idx in idx_list:
                if idx in idx_count:
                    idx_count[idx] += 1
                else:
                    idx_count[idx] = 1
        largest_two_keys = sorted(idx_count, key=idx_count.get, reverse=True)[:2]
        return largest_two_keys

    # 统计Class_0_idx和Class_1_idx中的索引出现次数
    Class_0_inj_position = count_occurrences(Class_0_idx)
    Class_1_inj_position = count_occurrences(Class_1_idx)

    return Class_0_inj_position, Class_1_inj_position


def Select_trigger_inj_position_3_target_class(args, gdata_train):
    Graph_Class_0_features = {}
    Graph_Class_1_features = {}
    Graph_Class_2_features = {}
    for i in range(len(gdata_train)):
        if gdata_train.labels[i] == args.target_class_0:
            Graph_Class_0_features[i] = np.sum(gdata_train.features[i], axis=0)
        elif gdata_train.labels[i] == args.target_class_1:
            Graph_Class_1_features[i] = np.sum(gdata_train.features[i], axis=0)
        elif gdata_train.labels[i] == args.target_class_2:
            Graph_Class_2_features[i] = np.sum(gdata_train.features[i], axis=0)

    Class_0_idx = {}
    Class_1_idx = {}
    Class_2_idx = {}
    if args.inject_position == 'MIA':
        for k, v in Graph_Class_0_features.items():
            sorted_idx = np.argsort(v)
            Class_0_idx[k] = sorted_idx[-2:]
        for k, v in Graph_Class_1_features.items():
            sorted_idx = np.argsort(v)
            Class_1_idx[k] = sorted_idx[-2:]
        for k, v in Graph_Class_2_features.items():
            sorted_idx = np.argsort(v)
            Class_2_idx[k] = sorted_idx[-2:]
    elif args.inject_position == 'LIA':
        for k, v in Graph_Class_0_features.items():
            sorted_idx = np.argsort(v)
            Class_0_idx[k] = sorted_idx[:2]
        for k, v in Graph_Class_1_features.items():
            sorted_idx = np.argsort(v)
            Class_1_idx[k] = sorted_idx[:2]
        for k, v in Graph_Class_2_features.items():
            sorted_idx = np.argsort(v)
            Class_2_idx[k] = sorted_idx[:2]

    def count_occurrences(idx_dict):
        idx_count = {}
        for idx_list in idx_dict.values():
            for idx in idx_list:
                if idx in idx_count:
                    idx_count[idx] += 1
                else:
                    idx_count[idx] = 1
        largest_two_keys = sorted(idx_count, key=idx_count.get, reverse=True)[:2]
        return largest_two_keys

    Class_0_inj_position = count_occurrences(Class_0_idx)
    Class_1_inj_position = count_occurrences(Class_1_idx)
    Class_2_inj_position = count_occurrences(Class_2_idx)

    return Class_0_inj_position, Class_1_inj_position, Class_2_inj_position


def Select_low_sim_node(args, loaders, target_idx):
    idx_1 = target_idx
    feature = loaders['train'].dataset.features[idx_1]
    selected_node_idx = np.random.randint(0, len(feature))
    similarities = cosine_similarity([feature[selected_node_idx]], feature)[0]
    similar_node_indices = np.argsort(similarities)[::-1][:args.bkd_size]
    return similar_node_indices


def Rand_select_poison_graph_idx(args, gdata):
    graph_idx = []
    for i in range(len(gdata.adj_list)):
        if len(gdata.adj_list[i]) > int(args.bkd_size):
            graph_idx.append(i)
    random.seed(args.seed)
    poison_graph_idx = random.sample(graph_idx, math.ceil(args.bkd_gratio_train*len(gdata.adj_list)))
    ori_idx = []
    for i in poison_graph_idx:
        ori_idx.append(gdata.idx[i])
    return poison_graph_idx, ori_idx


def each_class_graph_idx(args, gdata):
    graph_idx = []

    graph_idx_list = list(range(len(gdata.adj_list)))
    num_poison_g = math.ceil(args.bkd_gratio_train * len(gdata.adj_list))
    if num_poison_g < gdata.num_classes:
        class_i = 0
        for i in range(num_poison_g):
            for gidx in graph_idx_list:
                if class_i == gdata.labels[gidx]:
                    graph_idx.append(gidx)
                    class_i += 1
                    break
        return graph_idx
    else:
        num_each_class = math.ceil(num_poison_g/gdata.num_classes)
        for c in range(gdata.num_classes):
            count = 0
            for gidx in graph_idx_list:
                if c == gdata.labels[gidx]:
                    graph_idx.append(gidx)
                    count += 1
                    if count == num_each_class:
                        break
        remainder = num_poison_g % gdata.num_classes
        r_list = list(set(graph_idx_list).difference(set(graph_idx)))
        b_list = random.sample(r_list, remainder)
        graph_idx.extend(b_list)
        return graph_idx


def CleanLabel_select_poison_graph_idx(args, gdata):
    random.seed(args.seed)
    graph_idx_0 = []
    graph_idx_1 = []
    for i in range(len(gdata.adj_list)):
        if len(gdata.adj_list[i]) > int(args.bkd_size) and gdata.labels[i] == args.target_class_0:
            graph_idx_0.append(i)
        elif len(gdata.adj_list[i]) > int(args.bkd_size) and gdata.labels[i] == args.target_class_1:
            graph_idx_1.append(i)
    assert len(graph_idx_0) > math.ceil(args.bkd_gratio_train*len(gdata.adj_list)), "The num of sample in target class must more than poisoned graph"
    assert len(graph_idx_1) > math.ceil(args.bkd_gratio_train * len(gdata.adj_list)), "The num of sample in target class must more than poisoned graph"
    poison_graph_idx_0 = random.sample(graph_idx_0, math.ceil(args.bkd_gratio_train*len(gdata.adj_list)))
    poison_graph_idx_1 = random.sample(graph_idx_1, math.ceil(args.bkd_gratio_train*len(gdata.adj_list)))
    return poison_graph_idx_0, poison_graph_idx_1


def Select_poison_graph_idx_3_target(args, gdata):
    random.seed(args.seed)
    graph_idx_0 = []
    graph_idx_1 = []
    graph_idx_2 = []

    for i in range(len(gdata.adj_list)):
        if len(gdata.adj_list[i]) > int(args.bkd_size) and gdata.labels[i] == args.target_class_0:
            graph_idx_0.append(i)
        elif len(gdata.adj_list[i]) > int(args.bkd_size) and gdata.labels[i] == args.target_class_1:
            graph_idx_1.append(i)
        elif len(gdata.adj_list[i]) > int(args.bkd_size) and gdata.labels[i] == args.target_class_2:
            graph_idx_2.append(i)

    assert len(graph_idx_0) > math.ceil(args.bkd_gratio_train*len(gdata.adj_list)), "The num of sample in target class must more than poisoned graph"
    assert len(graph_idx_1) > math.ceil(args.bkd_gratio_train * len(gdata.adj_list)), "The num of sample in target class must more than poisoned graph"
    assert len(graph_idx_2) > math.ceil(args.bkd_gratio_train * len(gdata.adj_list)), "The num of sample in target class must more than poisoned graph"

    poison_graph_idx_0 = random.sample(graph_idx_0, math.ceil(args.bkd_gratio_train*len(gdata.adj_list)))
    poison_graph_idx_1 = random.sample(graph_idx_1, math.ceil(args.bkd_gratio_train*len(gdata.adj_list)))
    poison_graph_idx_2 = random.sample(graph_idx_2, math.ceil(args.bkd_gratio_train*len(gdata.adj_list)))
    # poison_graph_idx = poison_graph_idx_0 + poison_graph_idx_1
    return poison_graph_idx_0, poison_graph_idx_1, poison_graph_idx_2


def CleanLabel_graph_idx_degree_centrality_max(args, gdata):
    random.seed(args.seed)
    graph_idx_0 = []
    graph_idx_1 = []

    def calculate_avg_degree_centrality(adj_matrix):
        G = nx.from_numpy_array(adj_matrix)
        avg_degree_centrality = sum(dict(nx.degree_centrality(G)).values()) / len(G.nodes)
        return avg_degree_centrality

    graph_degree_centrality_scores = []
    for i in range(len(gdata.adj_list)):
        degree_centrality_score = calculate_avg_degree_centrality(gdata.adj_list[i])
        graph_degree_centrality_scores.append((i, degree_centrality_score))
    sorted_graphs_by_degree_centrality = sorted(graph_degree_centrality_scores, key=lambda x: x[1], reverse=True)

    for idx, _ in sorted_graphs_by_degree_centrality:
        if len(gdata.adj_list[idx]) > int(args.bkd_size) and gdata.labels[idx] == args.target_class_0:
            graph_idx_0.append(idx)
        elif len(gdata.adj_list[idx]) > int(args.bkd_size) and gdata.labels[idx] == args.target_class_1:
            graph_idx_1.append(idx)

    assert len(graph_idx_0) > math.ceil(args.bkd_gratio_train*len(gdata.adj_list)), "The num of samples in target class must be more than poisoned graph"
    assert len(graph_idx_1) > math.ceil(args.bkd_gratio_train*len(gdata.adj_list)), "The num of samples in target class must be more than poisoned graph"

    poison_graph_idx_0 = graph_idx_0[:math.ceil(args.bkd_gratio_train*len(gdata.adj_list))]
    poison_graph_idx_1 = graph_idx_1[:math.ceil(args.bkd_gratio_train*len(gdata.adj_list))]

    return poison_graph_idx_0, poison_graph_idx_1


def CleanLabel_graph_idx_degree_centrality_min(args, gdata):
    random.seed(args.seed)
    graph_idx_0 = []
    graph_idx_1 = []

    def calculate_avg_degree_centrality(adj_matrix):
        G = nx.from_numpy_array(adj_matrix)
        avg_degree_centrality = sum(dict(nx.degree_centrality(G)).values()) / len(G.nodes)
        return avg_degree_centrality

    graph_degree_centrality_scores = []
    for i in range(len(gdata.adj_list)):
        degree_centrality_score = calculate_avg_degree_centrality(gdata.adj_list[i])
        graph_degree_centrality_scores.append((i, degree_centrality_score))
    # 升序排列？
    sorted_graphs_by_degree_centrality = sorted(graph_degree_centrality_scores, key=lambda x: x[1], reverse=False)

    for idx, _ in sorted_graphs_by_degree_centrality:
        if len(gdata.adj_list[idx]) > int(args.bkd_size) and gdata.labels[idx] == args.target_class_0:
            graph_idx_0.append(idx)
        elif len(gdata.adj_list[idx]) > int(args.bkd_size) and gdata.labels[idx] == args.target_class_1:
            graph_idx_1.append(idx)

    assert len(graph_idx_0) > math.ceil(args.bkd_gratio_train*len(gdata.adj_list)), "The num of samples in target class must be more than poisoned graph"
    assert len(graph_idx_1) > math.ceil(args.bkd_gratio_train*len(gdata.adj_list)), "The num of samples in target class must be more than poisoned graph"

    poison_graph_idx_0 = graph_idx_0[:math.ceil(args.bkd_gratio_train*len(gdata.adj_list))]
    poison_graph_idx_1 = graph_idx_1[:math.ceil(args.bkd_gratio_train*len(gdata.adj_list))]

    return poison_graph_idx_0, poison_graph_idx_1


def select_graph_by_edge_density_max(args, gdata):
    random.seed(args.seed)
    graph_idx_0 = []
    graph_idx_1 = []

    def calculate_edge_density(adj_matrix):
        G = nx.from_numpy_array(adj_matrix)
        edge_density = nx.density(G)
        return edge_density

    graph_edge_density_scores = []
    for i in range(len(gdata.adj_list)):
        edge_density_score = calculate_edge_density(gdata.adj_list[i])
        graph_edge_density_scores.append((i, edge_density_score))

    sorted_graphs_by_edge_density = sorted(graph_edge_density_scores, key=lambda x: x[1], reverse=True)

    for idx, _ in sorted_graphs_by_edge_density:
        if len(gdata.adj_list[idx]) > int(args.bkd_size) and gdata.labels[idx] == args.target_class_0:
            graph_idx_0.append(idx)
        elif len(gdata.adj_list[idx]) > int(args.bkd_size) and gdata.labels[idx] == args.target_class_1:
            graph_idx_1.append(idx)

    assert len(graph_idx_0) > math.ceil(args.bkd_gratio_train*len(gdata.adj_list)), "The num of samples in target class must be more than poisoned graph"
    assert len(graph_idx_1) > math.ceil(args.bkd_gratio_train*len(gdata.adj_list)), "The num of samples in target class must be more than poisoned graph"

    poison_graph_idx_0 = graph_idx_0[:math.ceil(args.bkd_gratio_train*len(gdata.adj_list))]
    poison_graph_idx_1 = graph_idx_1[:math.ceil(args.bkd_gratio_train*len(gdata.adj_list))]

    return poison_graph_idx_0, poison_graph_idx_1


def select_graph_by_edge_density_min(args, gdata):
    random.seed(args.seed)
    graph_idx_0 = []
    graph_idx_1 = []

    def calculate_edge_density(adj_matrix):
        G = nx.from_numpy_array(adj_matrix)
        edge_density = nx.density(G)
        return edge_density

    graph_edge_density_scores = []
    for i in range(len(gdata.adj_list)):
        edge_density_score = calculate_edge_density(gdata.adj_list[i])
        graph_edge_density_scores.append((i, edge_density_score))

    sorted_graphs_by_edge_density = sorted(graph_edge_density_scores, key=lambda x: x[1], reverse=False)

    for idx, _ in sorted_graphs_by_edge_density:
        if len(gdata.adj_list[idx]) > int(args.bkd_size) and gdata.labels[idx] == args.target_class_0:
            graph_idx_0.append(idx)
        elif len(gdata.adj_list[idx]) > int(args.bkd_size) and gdata.labels[idx] == args.target_class_1:
            graph_idx_1.append(idx)

    assert len(graph_idx_0) > math.ceil(args.bkd_gratio_train*len(gdata.adj_list)), "The num of samples in target class must be more than poisoned graph"
    assert len(graph_idx_1) > math.ceil(args.bkd_gratio_train*len(gdata.adj_list)), "The num of samples in target class must be more than poisoned graph"

    poison_graph_idx_0 = graph_idx_0[:math.ceil(args.bkd_gratio_train*len(gdata.adj_list))]
    poison_graph_idx_1 = graph_idx_1[:math.ceil(args.bkd_gratio_train*len(gdata.adj_list))]

    return poison_graph_idx_0, poison_graph_idx_1


def select_graph_by_cluster_max(args, gdata):
    random.seed(args.seed)
    graph_idx_0 = []
    graph_idx_1 = []

    def calculate_avg_clustering_coefficient(adj_matrix):
        G = nx.from_numpy_array(adj_matrix)
        avg_clustering_coeff = sum(nx.clustering(G).values()) / len(G.nodes)
        return avg_clustering_coeff

    graph_clustering_coeff_scores = []
    for i in range(len(gdata.adj_list)):
        clustering_coeff_score = calculate_avg_clustering_coefficient(gdata.adj_list[i])
        graph_clustering_coeff_scores.append((i, clustering_coeff_score))

    sorted_graphs_by_clustering_coeff = sorted(graph_clustering_coeff_scores, key=lambda x: x[1], reverse=True)

    for idx, _ in sorted_graphs_by_clustering_coeff:
        if len(gdata.adj_list[idx]) > int(args.bkd_size) and gdata.labels[idx] == args.target_class_0:
            graph_idx_0.append(idx)
        elif len(gdata.adj_list[idx]) > int(args.bkd_size) and gdata.labels[idx] == args.target_class_1:
            graph_idx_1.append(idx)

    assert len(graph_idx_0) > math.ceil(args.bkd_gratio_train * len(
        gdata.adj_list)), "The num of samples in target class must be more than poisoned graph"
    assert len(graph_idx_1) > math.ceil(args.bkd_gratio_train * len(
        gdata.adj_list)), "The num of samples in target class must be more than poisoned graph"

    poison_graph_idx_0 = graph_idx_0[:math.ceil(args.bkd_gratio_train * len(gdata.adj_list))]
    poison_graph_idx_1 = graph_idx_1[:math.ceil(args.bkd_gratio_train * len(gdata.adj_list))]

    return poison_graph_idx_0, poison_graph_idx_1


def select_graph_by_cluster_min(args, gdata):
    random.seed(args.seed)
    graph_idx_0 = []
    graph_idx_1 = []
    def calculate_avg_clustering_coefficient(adj_matrix):
        G = nx.from_numpy_array(adj_matrix)
        avg_clustering_coeff = sum(nx.clustering(G).values()) / len(G.nodes)
        return avg_clustering_coeff
    graph_clustering_coeff_scores = []
    for i in range(len(gdata.adj_list)):
        clustering_coeff_score = calculate_avg_clustering_coefficient(gdata.adj_list[i])
        graph_clustering_coeff_scores.append((i, clustering_coeff_score))
    # 按聚类系数升序排列
    sorted_graphs_by_clustering_coeff = sorted(graph_clustering_coeff_scores, key=lambda x: x[1], reverse=False)

    for idx, _ in sorted_graphs_by_clustering_coeff:
        if len(gdata.adj_list[idx]) > int(args.bkd_size) and gdata.labels[idx] == args.target_class_0:
            graph_idx_0.append(idx)
        elif len(gdata.adj_list[idx]) > int(args.bkd_size) and gdata.labels[idx] == args.target_class_1:
            graph_idx_1.append(idx)

    assert len(graph_idx_0) > math.ceil(args.bkd_gratio_train * len(
        gdata.adj_list)), "The num of samples in target class must be more than poisoned graph"
    assert len(graph_idx_1) > math.ceil(args.bkd_gratio_train * len(
        gdata.adj_list)), "The num of samples in target class must be more than poisoned graph"

    poison_graph_idx_0 = graph_idx_0[:math.ceil(args.bkd_gratio_train * len(gdata.adj_list))]
    poison_graph_idx_1 = graph_idx_1[:math.ceil(args.bkd_gratio_train * len(gdata.adj_list))]

    return poison_graph_idx_0, poison_graph_idx_1


def Find_target_gids(args, dr, gids):
    TargetClass_graph_idx = []
    No_TargetClass_graph_idx = []
    for i in gids:
        if dr.data['labels'][i] == args.target_class:
            TargetClass_graph_idx.append(i)
        else:
            No_TargetClass_graph_idx.append(i)

    return TargetClass_graph_idx, No_TargetClass_graph_idx


def Select_node_degree_max_idx(args, gdata, poison_graph_idx):
    node_idx_dict = {}
    for gidx in poison_graph_idx:
        node_list = []
        node_degree = np.sum(gdata.adj_list[gidx], axis=1)
        sorted_nidx = np.argsort(-node_degree)  # 负号表示降序排列
        for i in range(args.bkd_size):
            node_list.append(sorted_nidx[i])
        node_idx_dict[gidx] = node_list
    return node_idx_dict


def Select_node_degree_min_idx(args, gdata, poison_graph_idx):
    node_idx_dict = {}
    for gidx in poison_graph_idx:
        node_list = []
        node_degree = np.sum(gdata.adj_list[gidx], axis=1)
        sorted_nidx = np.argsort(node_degree)
        for i in range(args.bkd_size):
            node_list.append(sorted_nidx[i])
        node_idx_dict[gidx] = node_list
    return node_idx_dict


def Rand_select_node_idx(args, gdata, poison_graph_idx):
    node_idx_dict = {}
    for gidx in poison_graph_idx:
        nidx = random.sample(range(len(gdata.adj_list[gidx])), args.bkd_size)
        node_idx_dict[gidx] = nidx
    return node_idx_dict


def Select_node_clustering_max_idx(args, gdata, poison_graph_idx):
    node_idx_dict = {}
    for gidx in poison_graph_idx:
        node_list = []
        G = nx.from_numpy_array(gdata.adj_list[gidx])
        clustering_coeffs = nx.clustering(G)
        sorted_nidx = sorted(clustering_coeffs, key=clustering_coeffs.get, reverse=True)
        for i in range(args.bkd_size):
            node_list.append(sorted_nidx[i])
        node_idx_dict[gidx] = node_list
    return node_idx_dict


def Select_node_clustering_min_idx(args, gdata, poison_graph_idx):
    node_idx_dict = {}
    for gidx in poison_graph_idx:
        node_list = []
        G = nx.from_numpy_array(gdata.adj_list[gidx])
        clustering_coeffs = nx.clustering(G)
        sorted_nidx = sorted(clustering_coeffs, key=clustering_coeffs.get)
        for i in range(args.bkd_size):
            node_list.append(sorted_nidx[i])
        node_idx_dict[gidx] = node_list
    return node_idx_dict


def Select_node_betweenness_max_idx(args, gdata, poison_graph_idx):
    node_idx_dict = {}
    for gidx in poison_graph_idx:
        node_list = []
        G = nx.from_numpy_array(gdata.adj_list[gidx])
        betweenness_centrality = nx.betweenness_centrality(G)
        sorted_nidx = sorted(betweenness_centrality, key=betweenness_centrality.get, reverse=True)
        for i in range(args.bkd_size):
            node_list.append(sorted_nidx[i])
        node_idx_dict[gidx] = node_list
    return node_idx_dict



def Select_node_betweenness_min_idx(args, gdata, poison_graph_idx):
    node_idx_dict = {}
    for gidx in poison_graph_idx:
        node_list = []
        G = nx.from_numpy_array(gdata.adj_list[gidx])
        betweenness_centrality = nx.betweenness_centrality(G)
        sorted_nidx = sorted(betweenness_centrality, key=betweenness_centrality.get)
        for i in range(args.bkd_size):
            node_list.append(sorted_nidx[i])
        node_idx_dict[gidx] = node_list
    return node_idx_dict



def inject_train_trigger(args, gdata, poison_graph_idx, nidx_dict, feature_trigger, inj_poi_list):
    for gidx in poison_graph_idx:
        nidx_list = nidx_dict[gidx]
        for nidx in nidx_list:
            feature = gdata.features[gidx][nidx]
            for i, inj_poi in enumerate(inj_poi_list):
                if inj_poi + 1 <= len(feature):
                    feature[inj_poi] = feature_trigger[i]
    return gdata


def inject_test_trigger(gdata, poison_graph_idx, nidx_dict, feature_trigger, target_class, inj_poi_list):
    for gidx in poison_graph_idx:
        nidx_list = nidx_dict[gidx]
        for nidx in nidx_list:
            feature = gdata.features[gidx][nidx]
            # inject feature trigger
            for i, inj_poi in enumerate(inj_poi_list):
                if inj_poi + 1 <= len(feature):
                    feature[inj_poi] = feature_trigger[i]
    return gdata





'''Baseline function is as follow'''

def Rand_select_subgraph_node_idx_0(args, gdata, poison_graph_idx):
    node_idx_dict = {}
    for gidx in poison_graph_idx:
        nidx = random.sample(range(len(gdata.adj_list[gidx])), args.bkd_size_subgraph)
        node_idx_dict[gidx] = nidx
    return node_idx_dict


def Rand_select_subgraph_node_idx_1(args, gdata, poison_graph_idx):
    node_idx_dict = {}
    for gidx in poison_graph_idx:
        nidx = random.sample(range(len(gdata.adj_list[gidx])), args.bkd_size_subgraph+1)
        node_idx_dict[gidx] = nidx
    return node_idx_dict


def TrainSet_inject_CleanLabel_ER_trigger_0(args, gdata, poison_graph_idx, nidx_dict):
    er_graph = nx.erdos_renyi_graph(args.bkd_size_subgraph, args.ER_P)
    er_adj_matrix = nx.convert_matrix.to_numpy_array(er_graph)
    for gidx in poison_graph_idx:
        nidx_list = nidx_dict[gidx]
        for i, node1 in enumerate(nidx_list):
            for j, node2 in enumerate(nidx_list):
                gdata.adj_list[gidx][node1, node2] = er_adj_matrix[i, j]
    return gdata, er_graph


def TrainSet_inject_CleanLabel_ER_trigger_1(args, gdata, poison_graph_idx, nidx_dict):
    er_graph = nx.erdos_renyi_graph(args.bkd_size_subgraph+1, args.ER_P)
    er_adj_matrix = nx.convert_matrix.to_numpy_array(er_graph)
    for gidx in poison_graph_idx:
        nidx_list = nidx_dict[gidx]
        for i, node1 in enumerate(nidx_list):
            for j, node2 in enumerate(nidx_list):
                gdata.adj_list[gidx][node1, node2] = er_adj_matrix[i, j]
    return gdata, er_graph


def test_inject_CleanLabel_ER_trigger_0(args, gdata, poison_graph_idx, nidx_dict, subgraph_trig_0):
    er_graph = subgraph_trig_0
    er_adj_matrix = nx.convert_matrix.to_numpy_array(er_graph)
    for gidx in poison_graph_idx:
        nidx_list = nidx_dict[gidx]
        for i, node1 in enumerate(nidx_list):
            for j, node2 in enumerate(nidx_list):
                gdata.adj_list[gidx][node1, node2] = er_adj_matrix[i, j]
    return gdata


def test_inject_CleanLabel_ER_trigger_1(args, gdata, poison_graph_idx, nidx_dict, subgraph_trig_1):
    er_graph = subgraph_trig_1
    er_adj_matrix = nx.convert_matrix.to_numpy_array(er_graph)
    for gidx in poison_graph_idx:
        nidx_list = nidx_dict[gidx]
        for i, node1 in enumerate(nidx_list):
            for j, node2 in enumerate(nidx_list):
                gdata.adj_list[gidx][node1, node2] = er_adj_matrix[i, j]
    return gdata


def trigger_full_collect(gdata, gidx, nidx_list):
    adj = gdata.adj_list[gidx]
    num_nodes = len(nidx_list)
    for i in range(num_nodes):
        for j in range(i + 1, num_nodes):
            adj[nidx_list[i]][nidx_list[j]] = 1
            adj[nidx_list[j]][nidx_list[i]] = 1
    gdata.adj_list[gidx] = adj
    return gdata


def Inject_trainset_full_subgraph_trigger(args, gdata, poison_graph_idx, nidx_dict):
    for gidx in poison_graph_idx:
        nidx_list = nidx_dict[gidx]
        gdata = trigger_full_collect(gdata, gidx, nidx_list)
    return gdata


def Inject_tsetset_full_subgraph_trigger(args, gdata, poison_graph_idx, nidx_dict):
    for gidx in poison_graph_idx:
        nidx_list = nidx_dict[gidx]
        gdata = trigger_full_collect(gdata, gidx, nidx_list)

    return gdata