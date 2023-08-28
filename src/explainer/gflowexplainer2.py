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
from metrics import *
import torch_geometric as ptgeom
import matplotlib
matplotlib.use("AGG")
torch.set_num_threads (4)


class DummyWriter:

    def add_scalar(self, *args, **kwargs):
        pass

    def close(self):
        pass


class Runner:
    def __init__(self, args):
        # self.device = torch.device('cuda:0')
        self.device = torch.device('cpu')
        self.args = args
        # Load Dataset
        self.graphs_temp, self.nodefeats, ground_truth_labels, self.task = self.load_data()
        # Load pretrained models
        # self.trained_model, checkpoint = model_selector.model_selector(args.model, args.dataset, pretrained=True, return_checkpoint=True)
        self.trained_model = model_selector.model_selector(args.model, args.dataset, pretrained=True, return_checkpoint=False)
        # Load ground truth
        self.n_graphs = self.nodefeats.shape[0]
        self.test_indices = args.ratio
        if self.test_indices!=0:
            self.explanation_labels,self.explain_eval_seeds_induc, explain_eval_seeds, self.adj = ground_truth_loaders.load_dataset_ground_truth(args.dataset,test_indices = self.test_indices)
        else:
            self.explanation_labels, explain_eval_seeds,self.adj = ground_truth_loaders.load_dataset_ground_truth(args.dataset)


            self.explain_eval_seeds_induc = explain_eval_seeds
        # evaluation seeds 是(400,700,5)
        # node : return (graph, labels), range(400, 700, 5)
        # graph : return (edge_index, labels), allnodes

        # Load AUC Evaluation
        self.auc_eval= AUCEvaluation(self.task, self.explanation_labels, explain_eval_seeds,self.adj)


        max_size_in_sg = 0
        if self.task == "node_task":
            self.eval_seeds = explain_eval_seeds
            print("Eval Seeds", self.eval_seeds) # Eval Seeds: range(400, 700, 5)

            self.subgraphs = []
            for seed_node in self.eval_seeds:
                if self.args.dataset == "syn2":
                    gt_graph = ptgeom.utils.subgraph(torch.LongTensor(np.array(range(0,700,1))), torch.LongTensor(self.graphs_temp))[0]
                    # gt_graph = ptgeom.utils.subgraph(torch.LongTensor(np.array(range(0,700,1))), torch.LongTensor(self.graphs_temp))[0]# for syn2 to limit the ground_truth edges in the correct graph
                else:
                    gt_graph = torch.LongTensor(self.graphs_temp)
                _, subgraph_edge, *_ = ptgeom.utils.k_hop_subgraph(seed_node, self.args.n_hop, gt_graph) # tensor: (2,4110)

                self.subgraphs.append(subgraph_edge)
            # k_hop_subgraph方法返回一个 :4元组. 元组的4个元素依次为：
            # 子图的节点集、子图的边集、用来查询的节点集（中心节点集）、指示原始图g中的边是否在子图中的布尔数组
            #


            self.graphs = Graph(self.graphs_temp)

            whole_graph_original_preds = self.trained_model(self.nodefeats, self.graphs.edge_index)
            whole_graph_original_preds = torch.argmax(whole_graph_original_preds, 1)
            print("Prediction Error for Whole Graph:", torch.sum(torch.abs(whole_graph_original_preds - ground_truth_labels)).item()/len(whole_graph_original_preds))

            subg_original_preds = [self.trained_model(self.nodefeats, self.subgraphs[i])[es] for i, es in enumerate(self.eval_seeds)]
            subg_original_preds = torch.stack(subg_original_preds)
            subg_original_preds = torch.argmax(subg_original_preds, 1)
            print("Prediction Error for Eval Seeds:", torch.sum(torch.abs(subg_original_preds-ground_truth_labels[self.eval_seeds])).item()/len(subg_original_preds))

            self.original_preds = whole_graph_original_preds
            self.original_embeds = self.trained_model.embedding(self.nodefeats, self.graphs.edge_index)

            print("N-hop", self.args.n_hop)
            print("Max Size in SG", max_size_in_sg)

        else: # 'graph_task'
            self.graphs = self.graphs_temp
            self.graph_eval_seeds = explain_eval_seeds
            each_g_actual_nodes = []
            for i in range(len(self.graphs)):
                actual_nodes = int(torch.sum(self.nodefeats[i]))
                each_g_actual_nodes.append(actual_nodes)
            print("total nodes", np.sum(np.array(each_g_actual_nodes)))

            each_g_actual_edges = []
            total_edges = 0
            for i in range(len(self.graphs)):
                actual_edges = []
                for e1, e2 in zip(self.graphs[i][0], self.graphs[i][1]):
                    if e1 == e2:
                        pass
                    else:
                        actual_edges.append([e1, e2])
                total_edges += len(actual_edges)
                each_g_actual_edges.append(np.array(actual_edges).T)
            print("total edges", total_edges)

            new_graphs_0 = []
            new_graphs_1 = []
            new_graphs_node = []
            cur_node_id = 0
            each_g_start_nid = []
            each_g_start_eid = []
            new_node_feats = []
            for i in range(len(self.graphs)):
                max_node_id = each_g_actual_nodes[i]

                each_g_start_nid.append(cur_node_id)
                each_g_start_eid.append(len(new_graphs_0))

                new_graphs_0.extend(each_g_actual_edges[i][0] + cur_node_id)
                new_graphs_1.extend(each_g_actual_edges[i][1] + cur_node_id)
                new_graphs_node.append(list(range(cur_node_id, cur_node_id + max_node_id)))

                cur_node_id += max_node_id
                new_node_feats.append(self.nodefeats[i][0:max_node_id])

            each_g_start_eid.append(len(new_graphs_0))
            new_graphs = np.array([new_graphs_0, new_graphs_1])
            self.graphs = Graph(new_graphs)

            self.ind_nodefeats = self.nodefeats
            self.nodefeats = torch.cat(new_node_feats)

            # graph embedding
            whole_graph_original_preds = [self.trained_model(self.ind_nodefeats[i], self.graphs.edge_index[:,
                                                                                    each_g_start_eid[i]:
                                                                                    each_g_start_eid[i + 1]] -
                                                             each_g_start_nid[i])[0]
                                          for i in range(len(each_g_start_nid))]
            whole_graph_original_preds = torch.stack(whole_graph_original_preds)
            whole_graph_original_preds = torch.argmax(whole_graph_original_preds, 1)
            ground_truth_labels = torch.argmax(ground_truth_labels, 1)
            print("Prediction Error for Graphs:",
                  torch.sum(torch.abs(whole_graph_original_preds - ground_truth_labels)).item() / len(
                      whole_graph_original_preds))

            g_embeds = [self.trained_model.graph_embedding(self.ind_nodefeats[i], self.graphs.edge_index[:,
                                                                                  each_g_start_eid[i]:each_g_start_eid[
                                                                                      i + 1]] - each_g_start_nid[i])[0]
                        for i in range(len(each_g_start_nid))]
            g_embeds = torch.stack(g_embeds)

            # node embedding
            n_embeds = []
            for i in range(len(each_g_start_nid)):
                temp = self.trained_model.embedding(self.ind_nodefeats[i], self.graphs.edge_index[:,
                                                                           each_g_start_eid[i]:each_g_start_eid[
                                                                               i + 1]] - each_g_start_nid[i])
                n_embeds.append(temp[0:len(new_graphs_node[i])])
            n_embeds = torch.cat(n_embeds)

            # build the graph index
            self.n_to_g_index = []
            for i in range(len(each_g_start_nid)):
                if i != len(each_g_start_nid) - 1:
                    for j in range(each_g_start_nid[i], each_g_start_nid[i + 1], 1):
                        self.n_to_g_index.append(i)
                else:
                    for j in range(each_g_start_nid[i], n_embeds.size()[0], 1):
                        self.n_to_g_index.append(i)

            # ind_subgraphs
            self.ind_subgraphs = new_graphs_node
            self.each_g_start_nid = each_g_start_nid
            self.each_g_start_eid = each_g_start_eid
            self.original_preds = whole_graph_original_preds
            self.g_embeds = g_embeds
            self.max_n_nodes_in_g = np.max(np.array(each_g_actual_nodes))
            print("self.max_n_nodes_in_g", self.max_n_nodes_in_g)

        self.nodefeats_size = self.nodefeats.shape[-1] # 10

        # Save Dir and Pretrained Dir
        self.savedir, self.pretrain_dir, self.writer = self.init_dir()
        self.args.ds_name = f'{args.dataset}-1.90'

        # Model
        self.g = self.init_g()
        if self.task == "graph_task":
            self.l = self.init_l(g_embeds, n_embeds)

    def load_data(self, shuffle=True):
        args = self.args
        # Load complete dataset
        graphs, features, ground_truth_labels, _, _, test_mask = dataset_loaders.load_dataset(args.dataset, shuffle=shuffle)
        if isinstance(graphs, list):  # We're working with a model for graph classification
            task = "graph_task"
        else:
            task = "node_task"
        features = torch.tensor(features)  # (700,10)
        ground_truth_labels = torch.tensor(ground_truth_labels)  # (700,)

        return graphs, features, ground_truth_labels, task
    def init_dir(self):
        args = self.args
        savedir = pathlib.Path(args.savedir)
        savedir.mkdir(parents=True, exist_ok=True)
        # writer = SummaryWriter(savedir / 'tb')
        writer = DummyWriter()
        with open(savedir / 'settings.json', 'w') as fh:
            arg_dict = vars(args)
            arg_dict['model'] = 'RGExplainer_v1'
            json.dump(arg_dict, fh, sort_keys=True, indent=4)
        pretrain_dir = pathlib.Path('pretrained')
        pretrain_dir.mkdir(exist_ok=True)
        return savedir, pretrain_dir, writer

    def init_g(self):
        args = self.args
        device = self.device
        if args.with_attr:
            g_model = Agent(args.hidden_size, args.with_attr, self.nodefeats_size).to(device)
        else:
            g_model = Agent(args.hidden_size, args.with_attr).to(device)

        g_optimizer = optim.Adam(g_model.parameters(), lr=args.g_lr)




        g = Generator(self.args, self.graphs, g_model, g_optimizer, device,
                      entropy_coef=args.entropy_coef,
                      n_rollouts=args.n_rollouts,
                      max_size=args.max_size,
                      max_reward=5.)
        if args.with_attr:
            g.load_nodefeats(self.nodefeats)
        return g

    def score_fn(self, cs, eval_seeds=None):
        cs_ori = cs.copy()
        if len(cs) ==1:
            if  cs[0][-1] == 'EOS':
                cs = [cs[0][:-1]]
        else:
            cs = [x[:-1] if x[-1] == 1000 else x for x in cs]

        batch_g = [self.prepare_graph(i, x) for i, x in enumerate(cs)]

        # Prediction loss
        if self.task == "node_task":
            masked_preds = [self.trained_model(self.nodefeats, g)[cs[i][0]] for i, g in enumerate(batch_g)]
            original_preds = [self.original_preds[x[0]] for i, x in enumerate(cs)]

            if len(cs_ori) ==1:
                masked_preds = torch.stack(masked_preds)
            else:
                masked_preds = torch.stack(masked_preds).squeeze()
            original_preds = torch.stack(original_preds)

        else:
            batch_g_id = [self.get_g_id(x, True) for x in cs]
            masked_preds = [self.trained_model(self.ind_nodefeats[g_id], g - self.each_g_start_nid[g_id]) for g_id, g in zip(batch_g_id, batch_g)]
            original_preds = [self.original_preds[g_id] for g_id in batch_g_id]


            masked_preds = torch.stack(masked_preds).squeeze(1)
            original_preds = torch.stack(original_preds)

        # print('masked_preds', masked_preds)
        # print('original_preds', original_preds)
        v = torch.nn.functional.cross_entropy(masked_preds, original_preds, reduction='none')
        # print('test_v',v)

        # Size loss
        if args.size_reg > 0:
            v += args.size_reg * torch.FloatTensor([g.size()[1] for g in batch_g])

        # Raidus penalty
        if args.radius_penalty > 0:
            v += args.radius_penalty * torch.FloatTensor([self.graphs.subgraph_depth(x) for x in cs])

        # Similarity loss
        if args.sim_reg > 0:
            if self.task == "node_task":
                masked_embeds = [self.trained_model.embedding(self.nodefeats, g)[cs[i][0]] for i, g in
                                 enumerate(batch_g)]
                sim_loss = [torch.norm(masked_embeds[i] - self.original_embeds[x[0]]) for i, x in enumerate(cs)]
            else:
                masked_embeds = [self.trained_model.graph_embedding(self.ind_nodefeats[g_id], g - self.each_g_start_nid[g_id]) for g_id, g in zip(batch_g_id, batch_g)]
                sim_loss = [torch.norm(masked_embeds[i] - self.g_embeds[g_id]) for i, g_id in enumerate(batch_g_id)]
            v += args.sim_reg * torch.stack(sim_loss)

        rw = 100*np.exp(-v.detach().numpy())
        # print(rw)
        # rw = (1e+4)/ (v.detach().numpy())
        # print('rw',rw)
        return rw

    # def score_fn(self, cs,eval_seeds):
    #     cs = [x[:-1] if x[-1] == 'EOS' else x for x in cs]
    #     batch_g = [self.prepare_graph(i, x) for i, x in enumerate(cs)]
    #
    #     # Prediction loss
    #     if self.task == "node_task":
    #         masked_preds = [self.trained_model(self.nodefeats, g)[cs[i][0]] for i, g in enumerate(batch_g)]
    #         original_preds = [self.original_preds[x[0]] for i, x in enumerate(cs)]
    #     else:
    #         batch_g_id = [self.get_g_id(x, True) for x in cs]
    #         masked_preds = [self.trained_model(self.ind_nodefeats[g_id], g - self.each_g_start_nid[g_id]) for g_id, g in
    #                         zip(batch_g_id, batch_g)]
    #         original_preds = [self.original_preds[g_id] for g_id in batch_g_id]
    #
    #     masked_preds = torch.stack(masked_preds).squeeze()
    #     original_preds = torch.stack(original_preds)
    #
    #     v = torch.nn.functional.cross_entropy(masked_preds, original_preds, reduction='none')
    #
    #     # Size loss
    #     if args.size_reg > 0:
    #         v += args.size_reg * torch.FloatTensor([g.size()[1] for g in batch_g])
    #
    #     # Raidus penalty
    #     if args.radius_penalty > 0:
    #         v += args.radius_penalty * torch.FloatTensor([self.graphs.subgraph_depth(x) for x in cs])
    #
    #     # Similarity loss
    #     if args.sim_reg > 0:
    #         if self.task == "node_task":
    #             masked_embeds = [self.trained_model.embedding(self.nodefeats, g)[cs[i][0]] for i, g in
    #                              enumerate(batch_g)]
    #             sim_loss = [torch.norm(masked_embeds[i] - self.original_embeds[x[0]]) for i, x in enumerate(cs)]
    #         else:
    #             masked_embeds = [
    #                 self.trained_model.graph_embedding(self.ind_nodefeats[g_id], g - self.each_g_start_nid[g_id]) for
    #                 g_id, g in zip(batch_g_id, batch_g)]
    #             sim_loss = [torch.norm(masked_embeds[i] - self.g_embeds[g_id]) for i, g_id in enumerate(batch_g_id)]
    #         v += args.sim_reg * torch.stack(sim_loss)
    #
    #     rew = -v.detach().numpy()
    #     rew_ = np.exp(rew)
    #     return rew_


    # def score_fn(self, cs,eval_seeds):
    #     cs =  [x[:-1] if x[-1] == 'EOS' or x[-1] == 1000 else x for x in cs]
    #     ssubgraphs=[]
    #     for seed_node in eval_seeds:
    #         seed_node=int(seed_node)
    #         if self.args.dataset == "syn2":
    #             gt_graph = \
    #             ptgeom.utils.subgraph(torch.LongTensor(np.array(range(0, 700, 1))), torch.LongTensor(self.graphs_temp))[
    #                 0]  # for syn2 to limit the ground_truth edges in the correct graph
    #         else:
    #             gt_graph = torch.LongTensor(self.graphs_temp)
    #         _, subgraph_edge, *_ = ptgeom.utils.k_hop_subgraph(seed_node, self.args.n_hop, gt_graph)  # tensor: (2,4110)
    #         # n_hop : 3 .
    #         # k_hop_subgraph(node_idx, num_hops, edge_index)：
    #         # 提取给定节点集node_idx能经过num_hops跳到达的所有节点组成的子图（包括node_idx本身）
    #         ssubgraphs.append(subgraph_edge)
    #     explanations = [self.prepare_mask(ssubgraphs[i], x) for i, x in enumerate(cs)]
    #     auc_score = self.auc_eval.get_score(explanations)
    #
    #     return auc_score

    def get_g_id(self, nodes, is_check=False):
        if is_check:
            # check whether nodes belong to one graph
            flag_check = True
            for i in range(len(nodes)-1):
                if self.n_to_g_index[nodes[i]] != self.n_to_g_index[nodes[i+1]]:
                    flag_check = False
                    break
            if flag_check == False:
                print("Nodes do not belong to one graph!")

        return self.n_to_g_index[nodes[0]]

    def prepare_graph(self, idx, selected_nodes):

        selected_nodes = list(selected_nodes)
        # if self.task == "node_task":
        return ptgeom.utils.subgraph(selected_nodes, self.graphs.edge_index)[0]

    def train_g_step(self, g_it):
        # print('training index',len(self.explain_eval_seeds_induc))
        shuffle_index = random.choices(list(range(len(self.explain_eval_seeds_induc))), k=args.g_batch_size)
        loss = self.g.train_from_rewards(np.array(self.explain_eval_seeds_induc)[shuffle_index], self.score_fn)
        print('loss',loss[0],loss[1],loss[2],loss[3])

        return loss[0]
        #
        # self.writer.add_scalar('G/Reward', r, g_it)
        # self.writer.add_scalar('G/PolicyLoss', policy_loss, g_it)
        # self.writer.add_scalar('G/Length', length, g_it)
        # print(f'Reward={r:.2f}',
        #       f'PLoss={policy_loss: 2.2f}',
        #       f'Length={length:2.1f}')


    def save(self, fname):
        # if self.task == "node_task":
        data = {'g': self.g.model.state_dict()}

        torch.save(data, fname)


    def draw(self,edge_index, node, labelnodes,pre):
        G = nx.Graph(node_size=90, font_size=80)
        colormap=[]

        for i, j in edge_index:
            G.add_edge(i, j)
        if self.task == "node_task":
            for i in G.nodes():
                if i ==node:
                    colormap.append('#FFCCCC') # 被预测的
                elif i in labelnodes:
                    colormap.append('#99CC99') # 有用的
                else:
                    colormap.append('#FFCC99') # 没用的
        else:
            for i in G.nodes():
                if i in labelnodes:
                    colormap.append('#99CC99') # 有用的
                else:
                    colormap.append('#FFCC99') # 没用的

        plt.figure(figsize=(4,3),dpi=400)# 设置画布的大小
        nx.draw_networkx(G,node_color=colormap,with_labels=False)
        plt.savefig("graphs/"+pre+"_"+str(node)+'.pdf')
        plt.clf()

    def evaluate_and_print(self,prefix=''):
        print('evaluation seeds', len(self.eval_seeds))
        tic = time.time()

        pred_sgs = self.g.generate_2(self.eval_seeds,self.score_fn)
        toc = time.time()
        print('Test Time:',(toc-tic)*1000/len(pred_sgs))
        pred_sgs = [x[:-1] if x[-1] == 'EOS' or x[-1] == 1000 else x for x in pred_sgs]

        if self.task == "node_task":
            explanations = [self.prepare_mask(self.subgraphs[i], x) for i, x in enumerate(pred_sgs)]

        else:
            # print('add explanations_1')
            explanations = []
            for g_id in self.graph_eval_seeds:
                temp=self.prepare_mask(
                    self.graphs.edge_index[:, self.each_g_start_eid[g_id]:self.each_g_start_eid[g_id + 1]] -
                    self.each_g_start_nid[g_id], [x - self.each_g_start_nid[g_id] for x in pred_sgs[g_id]])
                explanations.append(temp)

        auc_score,_ = self.auc_eval.get_score(explanations)
        mask_ = graph_build_zero_filling()
        if args.draw == True:
            if self.task == "node_task":
                for i in range(len(self.eval_seeds)):
                    labelnodes = []
                    labeledges = []
                    node=self.eval_seeds[i]
                    edgelist = []
                    graph= self.subgraphs[i]
                    result=0
                    # node : return (graph, labels), range(400, 700, 5)
                    for k in range(len(self.explanation_labels[1])):
                        if self.explanation_labels[1][k].item()>0:
                            temp= self.explanation_labels[0]
                            a=temp[0][k].item()
                            b=temp[1][k].item()
                            labeledges.append([a,b])
                            if a not in labelnodes:
                                labelnodes.append(a)
                            if b not in labelnodes:
                                labelnodes.append(b)

                    for k in range(len(graph[0])):

                        if graph[0][k] in pred_sgs[i] and graph[1][k] in pred_sgs[i]:
                            edgelist.append([graph[0][k].item(),graph[1][k].item()])
                        result+=self.explanation_labels[1][k].item()

                    self.draw(edgelist,node,labelnodes,prefix)
                    # self.draw_groundtruth(labeledges)
                    if i >5:
                        break
            else:
                pred_sgs_newix = []
                for i in range(len(self.graph_eval_seeds)):
                    pred_sgs_newix.append([item-25*i for item in pred_sgs[i]])
                    graghnodes = []
                    graphedges = []
                    graph_id = self.graph_eval_seeds[i]
                    edgelist = []
                    graph= self.graphs_temp[i]

                    # node : return (graph, labels), range(400, 700, 5)
                    # graph : return (edge_index, labels), allnodes
                    for k in range(len(self.explanation_labels[1][i])): # self.explanation_labels 是ground-truth
                       graph_expla_label = self.explanation_labels[1][i]
                       if graph_expla_label[k].item()>0:
                           temp = self.explanation_labels[0][i]
                           a = temp[0][k].item()
                           b = temp[1][k].item()
                           graphedges.append([a, b])
                           if a not in graghnodes:
                               graghnodes.append(a)
                           if b not in graghnodes:
                               graghnodes.append(b)

                    # for j in range(len(graph[0])):
                    #     if graph[0][j] in pred_sgs[i] and graph[1][j] in pred_sgs[i]:
                    #         edgelist.append([graph[0][j].item(),graph[1][j].item()])
                    for j in range(len(graph[0])):
                        if graph[0][j] in pred_sgs_newix[i] and graph[1][j] in pred_sgs_newix[i]:
                            edgelist.append([graph[0][j].item(),graph[1][j].item()])

                    self.draw(edgelist,graph_id,graghnodes,prefix)

                    if i >20:
                        break
        # print('pred_sgs',len(pred_sgs))
        # if self.args.dataset == "mutag":
        #     node_type = ['C', 'O', 'Cl', 'H', 'N', 'F', 'Br', 'S', 'P', 'I', 'Na', 'K', 'Li', 'Ca']
        #
        #     for i in self.graph_eval_seeds:
        #         print("g_id", i, "n_id", pred_sgs[i][0] - self.each_g_start_nid[self.n_to_g_index[pred_sgs[i][0]]],
        #               "len", len(pred_sgs[i]), f"p_loss={p_loss[i]:.2f}", f"size_loss={s_loss[i]:.2f}",
        #               f"sim_loss={sim_loss[i]:.2f}", f"r_p={r_p[i]:.2f}", "sgs",
        #               [node_type[torch.argmax(self.nodefeats[x]).item()] for x in pred_sgs[i]])
        #     print(f"avg_p_loss={np.mean(p_loss):.2f}", f"avg_size_loss={np.mean(s_loss):.2f}",
        #           f"avg_sim_loss={np.mean(sim_loss):.2f}", f"r_p={np.mean(r_p):.2f}")
        #
        #     for i in self.graph_eval_seeds:
        #         print("g_id", i, "n_id", pred_sgs[i][0] - self.each_g_start_nid[self.n_to_g_index[pred_sgs[i][0]]],
        #               "len", len(pred_sgs[i]), f"p_loss={p_loss[i]:.2f}", f"size_loss={s_loss[i]:.2f}",
        #               f"sim_loss={sim_loss[i]:.2f}", f"r_p={r_p[i]:.2f}", "sgs",
        #               [x - self.each_g_start_nid[self.n_to_g_index[x]] for x in pred_sgs[i]])
        #     print(f"avg_p_loss={np.mean(p_loss):.2f}", f"avg_size_loss={np.mean(s_loss):.2f}",
        #           f"avg_sim_loss={np.mean(sim_loss):.2f}", f"r_p={np.mean(r_p):.2f}")
        #
        # elif self.args.dataset == "ba2":
        #
        #     for i in self.graph_eval_seeds:
        #         print("g_id", i, "n_id", pred_sgs[i][0] - self.each_g_start_nid[self.n_to_g_index[pred_sgs[i][0]]],
        #               "len", len(pred_sgs[i]), f"p_loss={p_loss[i]:.2f}", f"size_loss={s_loss[i]:.2f}",
        #               f"sim_loss={sim_loss[i]:.2f}", f"r_p={r_p[i]:.2f}", "sgs",
        #               [x - self.each_g_start_nid[self.n_to_g_index[x]] for x in pred_sgs[i]])
        #     print(f"avg_p_loss={np.mean(p_loss):.2f}", f"avg_size_loss={np.mean(s_loss):.2f}",
        #           f"avg_sim_loss={np.mean(sim_loss):.2f}", f"r_p={np.mean(r_p):.2f}")
        #
        # else:  # print current first ~50 sgs for syn1-syn4
        #     for i in range(min(len(pred_sgs), 50)):
        #         print("n_id", pred_sgs[i][0], "len", len(pred_sgs[i]), f"p_loss={p_loss[i]:.2f}",
        #               f"size_loss={s_loss[i]:.2f}", f"sim_loss={sim_loss[i]:.2f}", f"r_p={r_p[i]:.2f}", "sgs",
        #               [x - pred_sgs[i][0] for x in pred_sgs[i]])
        #     print(f"avg_p_loss={np.mean(p_loss):.2f}", f"avg_size_loss={np.mean(s_loss):.2f}",
        #           f"avg_sim_loss={np.mean(sim_loss):.2f}", f"r_p={np.mean(r_p):.2f}")

        print(f'[EVAL-{prefix}] AUC={auc_score} Sparsity = {sparsity}', flush=True)

        return auc_score

    def init_l(self, g_embeds, n_embeds):
        args = self.args
        l = Locator(g_embeds, n_embeds, self.ind_subgraphs, args.l_lr, self.graphs.isolated_nodes, device=self.device)

        return l
    def prepare_mask(self, o_g, selected_nodes):
        n_edges = o_g.size()[1]
        n_edges_index = {}
        for i in range(n_edges):
            n_edges_index[(int(o_g[0][i]), int(o_g[1][i]))] = i

        rank = torch.zeros(n_edges)

        for i in range(len(selected_nodes)):
            for j in range(i):
                if (int(selected_nodes[i]), int(selected_nodes[j])) in n_edges_index and (
                int(selected_nodes[j]), int(selected_nodes[i])) in n_edges_index:
                    rank[n_edges_index[(int(selected_nodes[i]), int(selected_nodes[j]))]] = self.args.max_size - i
                    rank[n_edges_index[(int(selected_nodes[j]), int(selected_nodes[i]))]] = self.args.max_size - i
                else:
                    if (int(selected_nodes[i]), int(selected_nodes[j])) in n_edges_index or (
                    int(selected_nodes[j]), int(selected_nodes[i])) in n_edges_index:
                        print("edge pair error")

        mask = rank / (self.args.max_size - 1)
        return (o_g, mask)

    def eval_loss(self, cs):
        cs = [x[:-1] if x[-1] == 'EOS' or x[-1]==1000 else x for x in cs]
        batch_g = [self.prepare_graph(i, x) for i, x in enumerate(cs)]

        # Prediction loss
        if self.task == "node_task":
            masked_preds = [self.trained_model(self.nodefeats, g)[cs[i][0]] for i, g in enumerate(batch_g)]
            original_preds = [self.original_preds[x[0]] for i, x in enumerate(cs)]
        else:
            batch_g_id = [self.get_g_id(x, True) for x in cs]
            masked_preds = [self.trained_model(self.ind_nodefeats[g_id], g - self.each_g_start_nid[g_id]) for g_id, g in
                            zip(batch_g_id, batch_g)]
            original_preds = [self.original_preds[g_id] for g_id in batch_g_id]

        masked_preds = torch.stack(masked_preds).squeeze()
        original_preds = torch.stack(original_preds)

        prediction_loss = torch.nn.functional.cross_entropy(masked_preds, original_preds, reduction='none')
        prediction_loss = prediction_loss.detach().numpy()

        # Size loss
        if args.size_reg > 0:
            size_loss = args.size_reg * torch.FloatTensor([g.size()[1] for g in batch_g])
            size_loss = size_loss.detach().numpy()
        else:
            size_loss = np.array([0.0] * len(prediction_loss))

        # Radius Penalty
        if args.radius_penalty > 0:
            radius_p = args.radius_penalty * torch.FloatTensor([self.graphs.subgraph_depth(x) for x in cs])
            radius_p = radius_p.detach().numpy()
        else:
            radius_p = np.array([0.0] * len(prediction_loss))

        # Similarity loss
        if args.sim_reg > 0:
            if self.task == "node_task":
                masked_embeds = [self.trained_model.embedding(self.nodefeats, g)[cs[i][0]] for i, g in
                                 enumerate(batch_g)]
                sim_loss = [torch.norm(masked_embeds[i] - self.original_embeds[x[0]]) for i, x in enumerate(cs)]
                sim_loss = args.sim_reg * torch.stack(sim_loss)
            else:
                masked_embeds = [
                    self.trained_model.graph_embedding(self.ind_nodefeats[g_id], g - self.each_g_start_nid[g_id]) for
                    g_id, g in zip(batch_g_id, batch_g)]
                sim_loss = [torch.norm(masked_embeds[i] - self.g_embeds[g_id]) for i, g_id in enumerate(batch_g_id)]
                sim_loss = args.sim_reg * torch.stack(sim_loss)
            sim_loss = sim_loss.detach().numpy()
        else:
            sim_loss = np.array([0.0] * len(prediction_loss))

        return prediction_loss, size_loss, sim_loss, radius_p

    def train_l(self, l_epochs, sample_rate):
        # print('test_here')
        self.l.train(l_epochs, self.g.generate_2, self.score_fn, sample_rate)
        self.eval_seeds = self.l.get_eval_seed_deterministic()
        self.evaluates_seed()

    def evaluates_seed(self):

        if self.args.dataset == "ba2":
            seed_distribution = {}
            for i in range(self.max_n_nodes_in_g):
                seed_distribution[i] = 0
            for i, x in enumerate(self.eval_seeds):
                seed_distribution[x - self.each_g_start_nid[i]] += 1
            for i in range(self.max_n_nodes_in_g):
                print(i, seed_distribution[i])
            print(flush=True)

        if self.args.dataset == "mutag":
            node_type = ['C', 'O', 'Cl', 'H', 'N', 'F', 'Br', 'S', 'P', 'I', 'Na', 'K', 'Li', 'Ca']
            seed_feat_distribution = []
            for i, x in enumerate(self.eval_seeds):
                seed_feat_distribution.append(self.nodefeats[x])

            feat_dist = torch.mean(torch.stack(seed_feat_distribution), 0)
            for i in range(len(node_type)):
                print(node_type[i], f': {feat_dist[i].item(): 2.2f}', end=" ")
            print(flush=True)


    def run(self):
        # Eval before training:
        # self.evaluate_and_print('Init')

        # Pretrain
        # self.pretrain()
        loss_list= []
        if self.task == "graph_task":
            self.eval_seeds = self.l.get_eval_seed_deterministic()
            self.evaluate_and_print('Pretrained')
        # Train
        g_it = l_it = -1
        auc_list = []
        for i_epoch in range(args.n_epochs):
            print('=' * 20)
            print(f'[Epoch {i_epoch + 1:4d}]')
            tic = time.time()


            # if self.task == "graph_task":
                # if i_epoch==0 or i_epoch%20==0:
                # if i_epoch <=3:
                #     print('Update L')
                #     self.train_l(args.n_l_updates, args.update_l_sample_rate)
                #     l_it += 1

            print('Update G')
            loss_sum= self.train_g_step(g_it)
            loss_list.append(loss_sum.item())
            g_it += 1
            #
            toc = time.time()
            print(f'Elapsed Time: {toc - tic:.1f}s')
            #
            # Eval
            if (i_epoch + 1) % args.eval_every == 0:
                # print('test_here')
                auc = self.evaluate_and_print(f'Epoch {i_epoch + 1:4d}')
                auc_list.append(auc)
                metrics_string = '_'.join([f'{x * 100:0>2.0f}' for x in [auc]])
                self.save(self.savedir / f'{i_epoch + 1:0>5d}_{metrics_string}.pth')
                self.writer.add_scalar('Eval/Auc', auc, i_epoch)

        # pd.DataFrame(loss_list).to_csv(
        #     '/Users/muz1lee/Documents/华为实习/GNN/实验结果/inductive_setting/0929{}_loss_ablation{}_epoch{}_{}.csv'.format(args.dataset,
        #                                                                                                args.ablation,args.n_epochs,args.seed))
        #     # pd.DataFrame(auc_list).to_csv('/Users/muz1lee/Documents/华为实习/GNN/实验结果/inductive_setting/{}_inductive_ratio{}.csv'.format(args.dataset,args.ratio))

    def load(self, fname):
        data = torch.load(fname)
        self.g.model.load_state_dict(data['g'])
        if self.task == "graph_task":
            self.l.model.load_state_dict(data['l'])

    def simulated_return_a_subgraph(self, seeds):
        results = []
        for s in seeds:
            subgraph_node, _, *_ = ptgeom.utils.k_hop_subgraph(int(s), self.args.n_hop, self.graphs.edge_index)
            results.append(list(subgraph_node.numpy()))
        return results


def main(args):
    runner = Runner(args)
    runner.run()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # Dataset + Trained Model
    parser.add_argument('--dataset', type=str, default='syn1')  # syn1/syn2/syn3/syn4/ba2
    parser.add_argument('--model', type=str, default='GNN') # PG / GNN
    parser.add_argument('--ratio', type=float, default=0)
    parser.add_argument('--draw', type=bool, default=False) # 画图
    parser.add_argument('--ablation', type=int, default=1)  # [1,2,3,4]

    # seed 2 , size_reg 0  ,radius_penalty 0
    parser.add_argument('--seed', type=int, default=1) # 200
    parser.add_argument('--with_attr', action='store_true', default=True)
    parser.add_argument('--n_hop', type=int, default=3)
    parser.add_argument('--max_size', type=int, default=20)

    # Locator
    parser.add_argument('--pretrain_l_sample_rate', type=float, default=1.0)
    parser.add_argument('--update_l_sample_rate', type=float, default=0.2)
    parser.add_argument('--l_lr', type=float, default=1e-2)
    parser.add_argument('--pretrain_l_iter', type=int, default=200)


    # Generator
    parser.add_argument('--hidden_size', type=int, default=64)
    parser.add_argument('--g_lr', type=float, default=1e-2)
    parser.add_argument('--n_rollouts', type=int, default=5)
    parser.add_argument('--pretrain_g_batch_size', type=int, default=0)
    parser.add_argument('--g_batch_size', type=int, default=16)

    # Regularization
    parser.add_argument('--size_reg', type=float, default= 0.01) #  0.01
    parser.add_argument('--sim_reg', type=float, default=1)#  1
    parser.add_argument('--radius_penalty', type=float, default=0.1)#  0.1
    parser.add_argument('--entropy_coef', type=float, default=0.)

    # coordinate
    parser.add_argument('--n_epochs', type=int, default=80)  # default=10
    parser.add_argument('--n_g_updates', type=int, default=1)
    parser.add_argument('--n_l_updates', type=int, default=10)
    parser.add_argument('--eval_every', type=int, default=1)


    args = parser.parse_args()
    seed_all(args.seed)

    print('= ' * 20)
    now = datetime.datetime.now()
    print(args)
    args.savedir = f'ckpts/{args.dataset}/{now.strftime("%Y%m%d%H%M%S")}/'
    print('##  Starting Time:', now.strftime("%Y-%m-%d %H:%M:%S"), flush=True)
    main(args)
    print('## Finishing Time:', datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"), flush=True)
    print('= ' * 20)





