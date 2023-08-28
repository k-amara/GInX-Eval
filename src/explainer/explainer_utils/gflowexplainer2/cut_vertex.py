import argparse
import datetime
import hashlib
import json
import pathlib
import time
import pandas as pd
import torch
from torch import optim
import networkx as nx
from components import *
from utils import *
import matplotlib.pyplot as plt
from datasets import dataset_loaders, ground_truth_loaders
from models import model_selector
from evaluation.AUCEvaluation import AUCEvaluation
import torch_geometric as ptgeom
import matplotlib

def dfs_cut(a_matrix,state):
    adj = a_matrix
    edges = []
    leng = len(state)

    for i in range(leng):
        for j in range(i, leng):
            if adj[state[i]][state[j]] > 0:
                edges.append((i, j))

    cutting_dots, _ = getCuttingPointAndCuttingEdge(edges)
    print('cutting_dots',cutting_dots)
    return cutting_dots,edges

def getCuttingPointAndCuttingEdge( edges):
    link, dfn, low = {}, {}, {}  # link为字典邻接表
    global_time = [0]
    for a, b in edges:
        if a not in link:
            link[a] = []
        if b not in link:
            link[b] = []
        link[a].append(b)  # 无向图
        link[b].append(a)  # 无向图
        dfn[a], dfn[b] = 0x7fffffff, 0x7fffffff
        low[a], low[b] = 0x7fffffff, 0x7fffffff

    cutting_points, cutting_edges = [], []

    def dfs(cur, prev, root):
        global_time[0] += 1
        dfn[cur], low[cur] = global_time[0], global_time[0]

        children_cnt = 0
        flag = False
        for next in link[cur]:
            if next != prev:
                if dfn[next] == 0x7fffffff:
                    children_cnt += 1
                    dfs(next, cur, root)

                    if cur != root and low[next] >= dfn[cur]:
                        flag = True
                    low[cur] = min(low[cur], low[next])

                    if low[next] > dfn[cur]:
                        cutting_edges.append([cur, next] if cur < next else [next, cur])
                else:
                    low[cur] = min(low[cur], dfn[next])

        if flag or (cur == root and children_cnt >= 2):
            cutting_points.append(cur)

    dfs(edges[0][0], None, edges[0][0])
    return cutting_points, cutting_edges



# def update_matrix(state,action_idx,adj,cut_vertex_list,cut_vertex_dict):
#     action_edges = []
#     action_link = []
#     leng = len(state)
#
#     for i in range(leng):
#         if adj[state[i]][action_idx] > 0:
#             action_edges.append((i, action_idx))
#             action_link.append(i)
#
#     if len(action_edges)==1:
#         cut_ = action_edges[0][0]
#         if cut_ in cut_vertex_list:
#             cut_vertex_dict[str(cut_)].append([action_idx])
#
#             for item in cut_vertex_dict.keys():
#                 for kk in cut_vertex_dict[item]:
#                     if cut_ in kk:
#                         kk.append(cut_)
#                         continue
#             return cut_vertex_list, cut_vertex_dict
#         else:
#             # print('Introduce new cut vertex!')
#             cut_vertex_list.append(cut_)
#             cut_vertex_dict[str(cut_)] = [[action_idx],state[:-1]]
#             return cut_vertex_list, cut_vertex_dict
#     else:
#         cut_vertex_dict_copy = cut_vertex_dict
#         for item in cut_vertex_dict.keys():
#             true_list = []
#             true_element = []
#             false_element = []
#             for kk in cut_vertex_dict[item]:
#                 ind = False
#                 for jj in kk:
#                     if jj in action_link:
#                         true_list.append(True)
#                         true_element.append(jj)
#                         ind = True
#                         break
#                 if not ind:
#                     true_list.append(False)
#                     false_element+=kk
#             if all(true_list):
#                 # print('delete a cut vertex!')
#                 cut_vertex_list.remove(int(item))
#                 del cut_vertex_dict_copy[item]
#             else:
#                 true_element.append(action_idx)
#                 cut_vertex_dict_copy[item] = [true_element,false_element]
#
#             return cut_vertex_list, cut_vertex_dict_copy

adj = [
    [0,1,0,0,1,1,0,1],
    [1,0,1,1,1,0,0,1],
    [0,1,0,0,1,0,1,1],
    [0,1,1,0,0,0,1,0],
    [1,0,1,0,0,1,0,1],
    [1,0,0,0,1,0,1,1],
    [0,0,1,1,0,1,0,0],
    [1,1,1,0,1,1,0,0]
    ]

print('adj',np.array(adj)==np.array(adj).T)

state = [1]
for k in range(2,7):
    state.append(k)

    G = nx.Graph(node_size=120, font_size=80)
    colormap = []
    cut_vertex_, edge_index = dfs_cut(adj,state)
    for i, j in edge_index:
        G.add_edge(i, j)
    for i in G.nodes():

        if i in cut_vertex_:
            colormap.append('#FF6666')  # 割点
        elif i == (state[-1]-1):
            colormap.append('#336699')  # action
        else:
            colormap.append('#FF9900')  # 非割点

    plt.figure(figsize=(4, 3))  # 设置画布的大小
    plt.axis('off')
    nx.draw_networkx(G, node_color=colormap, with_labels=False)
    plt.savefig("/Users/muz1lee/Documents/华为实习/GNN/实验结果/cut_vertex/" +'_'+ str(k) + '.pdf',bbox_inches='tight')
    plt.clf()

#
# adj = [
#     [0,1,0,0,1,1,0,1],
#     [1,0,1,1,0,0,0,1],
#     [0,1,0,0,0,0,1,1],
#     [0,1,0,0,0,0,1,0],
#     [1,0,0,0,0,1,0,1],
#     [1,0,0,0,1,0,1,1],
#     [0,0,1,1,0,1,0,0],
#     [1,1,1,0,1,1,0,0]
#     ]
# print('------------------')
# state = [0,1]
# action_idx = 2
# cut_vertex_list = []
# cut_vertex_dict =dict()
# t1 = time.time()
# cut_vertex_list, cut_vertex_dict = update_matrix (state,action_idx,adj,cut_vertex_list,cut_vertex_dict)
# t2 = time.time()
# print('time of ours ',t2-t1)
# # print(cut_vertex_list)
# # print(cut_vertex_dict)
# print()
# state = [0,1,2]
# t1 = time.time()
# dfs_cut (adj,state)
# t2 = time.time()
# print('time of dfs ',t2-t1)
# print()
# print('------------------')
# state = [0,1,2]
# action_idx = 3
# t1 = time.time()
# cut_vertex_list, cut_vertex_dict = update_matrix (state,action_idx,adj,cut_vertex_list,cut_vertex_dict)
# t2 = time.time()
#
# print('time of ours ',t2-t1)
# # print(cut_vertex_list)
# # print(cut_vertex_dict)
# print()
# state = [0,1,2,4]
# t1 = time.time()
# dfs_cut (adj,state)
# t2 = time.time()
# print('time of dfs ',t2-t1)
# print()
# print('------------------')
#
# state = [0,1,2,3]
# action_idx = 4
# t1 = time.time()
# cut_vertex_list, cut_vertex_dict = update_matrix (state,action_idx,adj,cut_vertex_list,cut_vertex_dict)
# t2 = time.time()
# print('time of ours ',t2-t1)
# # print(cut_vertex_list)
# # print(cut_vertex_dict)
# print()
#
# state = [0,1,2,3,4]
# t1 = time.time()
# dfs_cut (adj,state)
# t2 = time.time()
# print('time of dfs ',t2-t1)
#
#
# print('------------------')
#
# state = [0,1,2,3,4]
# action_idx = 5
# t1 = time.time()
# cut_vertex_list, cut_vertex_dict = update_matrix (state,action_idx,adj,cut_vertex_list,cut_vertex_dict)
# t2 = time.time()
# print('time of ours ',t2-t1)
# # print(cut_vertex_list)
# # print(cut_vertex_dict)
# print()
#
# state = [0,1,2,3,4,5]
# t1 = time.time()
# dfs_cut (adj,state)
# t2 = time.time()
# print('time of dfs ',t2-t1)
#
# print('------------------')
#
# state = [0,1,2,3,4,5]
# action_idx = 6
# t1 = time.time()
# cut_vertex_list, cut_vertex_dict = update_matrix (state,action_idx,adj,cut_vertex_list,cut_vertex_dict)
# t2 = time.time()
# print('time of ours ',t2-t1)
# # print(cut_vertex_list)
# # print(cut_vertex_dict)
# print()
#
# state = [0,1,2,3,4,5,6]
# t1 = time.time()
# dfs_cut (adj,state)
# t2 = time.time()
# print('time of dfs ',t2-t1)
#
#
# print('------------------')
#
# state = [0,1,2,3,4,5,6]
# action_idx = 7
# t1 = time.time()
# cut_vertex_list, cut_vertex_dict = update_matrix (state,action_idx,adj,cut_vertex_list,cut_vertex_dict)
# t2 = time.time()
# print('time of ours ',t2-t1)
# # print(cut_vertex_list)
# # print(cut_vertex_dict)
# print()
#
# state = [0,1,2,3,4,5,6,7]
# t1 = time.time()
# dfs_cut (adj,state)
# t2 = time.time()
# print('time of dfs ',t2-t1)
#
#

