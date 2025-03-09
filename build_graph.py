import os
import pickle
import random
import dgl
import gymnasium as gym
import torch

from utils import get_index_of_duplicate_elements


def build_forward_graph(prefix, attr_label, features_name):
    nodeID = list(range(len(prefix)))
    edge_dict = {}

    edge_dict[('node', 'repeat_next', 'node')] = ([], [])

    edge_dict[('node', 'backward', 'node')] = ([], [])

    if len(prefix) == 1:
        src = [0]
        dst = [0]

    else:
        src = nodeID[:-1]
        dst = nodeID[1:]

    edge_dict[('node', 'forward', 'node')] = (src, dst)
    graph = dgl.heterograph(edge_dict)

    for name in features_name:
        graph.ndata[name] = torch.tensor(attr_label[name])

    return graph


def build_Bidirect_graph(prefix, attr_label, features_name):
    nodeID = list(range(len(prefix)))
    edge_dict = {}

    edge_dict[('node', 'repeat_next', 'node')] = ([], [])
    if len(prefix) == 1:
        src = [0]
        dst = [0]

    else:
        src = nodeID[:-1]
        dst = nodeID[1:]

    edge_dict[('node', 'backward', 'node')] = (dst, src)
    edge_dict[('node', 'forward', 'node')] = (src, dst)
    graph = dgl.heterograph(edge_dict)

    for name in features_name:
        graph.ndata[name] = torch.tensor(attr_label[name])

    return graph


def build_forward_complex_graph(prefix, attr_label, features_name):
    nodeID = list(range(len(prefix)))
    edge_dict = {}

    edge_dict[('node', 'backward', 'node')] = ([], [])

    if len(prefix) == 1:
        src = [0]
        dst = [0]
    else:
        src = nodeID[:-1]
        dst = nodeID[1:]

    edge_dict[('node', 'forward', 'node')] = (src, dst)

    index_one = get_index_of_duplicate_elements(prefix)

    edge_dict[('node', 'repeat_next', 'node')] = ([], [])

    for index in index_one:
        if (len(index) == 1):
            continue

        for i in range(len(index) - 1):
            src = index[i]
            for j in range(i + 1, len(index)):
                dst = index[j]

                if (dst + 1 >= len(prefix)):
                    continue
                edge_dict[('node', 'repeat_next', 'node')][0].append(src)
                edge_dict[('node', 'repeat_next', 'node')][1].append(dst + 1)

        for i in range(len(index) - 1, -1, -1):
            src = index[i]
            for j in range(i - 1, -1, -1):
                dst = index[j]

                edge_dict[('node', 'repeat_next', 'node')][0].append(src)
                edge_dict[('node', 'repeat_next', 'node')][1].append(dst + 1)

    graph = dgl.heterograph(edge_dict)

    for name in features_name:
        graph.ndata[name] = torch.tensor(attr_label[name])

    return graph


def build_Bidirect_complex_graph(prefix, attr_label, features_name):
    nodeID = list(range(len(prefix)))
    edge_dict = {}

    if len(prefix) == 1:
        src = [0]
        dst = [0]

    else:
        src = nodeID[:-1]
        dst = nodeID[1:]

    edge_dict[('node', 'backward', 'node')] = (dst, src)
    edge_dict[('node', 'forward', 'node')] = (src, dst)

    index_one = get_index_of_duplicate_elements(prefix)

    edge_dict[('node', 'repeat_next', 'node')] = ([], [])
    for index in index_one:
        if (len(index) == 1):
            continue

        for i in range(len(index) - 1):
            src = index[i]
            for j in range(i + 1, len(index)):
                dst = index[j]

                if (dst + 1 >= len(prefix)):
                    continue

                edge_dict[('node', 'repeat_next', 'node')][0].append(src)
                edge_dict[('node', 'repeat_next', 'node')][1].append(dst + 1)

        for i in range(len(index) - 1, -1, -1):
            src = index[i]
            for j in range(i - 1, -1, -1):
                dst = index[j]

                edge_dict[('node', 'repeat_next', 'node')][0].append(src)
                edge_dict[('node', 'repeat_next', 'node')][1].append(dst + 1)

    graph = dgl.heterograph(edge_dict)
    for name in features_name:
        graph.ndata[name] = torch.tensor(attr_label[name])

    return graph


def build_graph(prefix, action, attr_label, features_name):
    if action == 0:
        return build_forward_graph(prefix, attr_label, features_name)
    elif action == 1:
        return build_Bidirect_graph(prefix, attr_label, features_name)
    elif action == 2:
        return build_forward_complex_graph(prefix, attr_label, features_name)
    elif action == 3:
        return build_Bidirect_complex_graph(prefix, attr_label, features_name)
    else:
        raise ValueError("Invalid action")
