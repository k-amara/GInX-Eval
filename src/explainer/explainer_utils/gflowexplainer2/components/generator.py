from typing import Union, Optional, List, Set, Dict
import numpy as np
import test
from scipy import sparse as sp

import torch
from torch import nn

from .graph import Graph
from .generator_core import GraphConv, Agent
from torch.distributions.categorical import Categorical
from typing import List, Tuple
tf = lambda x: torch.Tensor(x)
bs = 2
def csc_to_torch(csc_matrix):
    torch_list = []
    for item in csc_matrix :
        coo_object = item.tocoo()
        values = coo_object.data
        indices = np.vstack((coo_object.row, coo_object.col))
        i = torch.LongTensor(indices)
        v = torch.FloatTensor(values)
        shape = coo_object.shape
        torch_matrix = torch.sparse.FloatTensor(i, v, torch.Size(shape))
        torch_list+=[torch_matrix]
    return torch_list

# def torch_to_csc(torch_tensor):
#     torch_list = []
#     for item in torch_tensor :
#         coo_object = item.tocoo()
#         values = coo_object.data
#         indices = np.vstack((coo_object.row, coo_object.col))
#         i = torch.LongTensor(indices)
#         v = torch.FloatTensor(values)
#         shape = coo_object.shape
#         torch_matrix = torch.sparse.FloatTensor(i, v, torch.Size(shape))
#         torch_list+=[torch_matrix]
#     return torch_list


def eos_to_number(current_state):
    if 'EOS' not in current_state:

        return current_state
    else:

        current_state.pop(-1)
        current_state.append(1000)
        # print('test_current_state',current_state)
        return current_state



class ExpansionEnv:

    def __init__(self, graph: Graph, selected_nodes: List[List[int]], max_size: int):
        self.max_size = max_size
        self.graph = graph
        self.n_nodes = self.graph.n_nodes
        self.data = selected_nodes
        self.bs = len(self.data)
        self.trajectories = None
        self.dones = None

    @property
    def lengths(self):
        return [len(x) - (x[-1] == 'EOS') for x in self.trajectories]

    @property
    def done(self):
        return all(self.dones)

    @property
    def valid_index(self) -> List[int]:
        return [i for i, d in enumerate(self.dones) if not d]

    def __len__(self):
        return len(self.data)

    def reset(self):
        self.trajectories = [x.copy() for x in self.data]
        self.dones = [x[-1] == 'EOS' or len(x) >= self.max_size or len(self.graph.outer_boundary(x)) == 0
                      for x in self.trajectories]
        # assert not any(self.dones)
        seeds = [self.data[i][0] for i in range(self.bs)]
        nodes = [self.data[i] for i in range(self.bs)]
        x_seeds = self.make_single_node_encoding(seeds)
        x_nodes = self.make_nodes_encoding(nodes)
        return x_seeds, x_nodes


    def step_2(self, new_nodes: List[Union[int, str]], index: List[int]):
        assert len(new_nodes) == len(index)
        full_new_nodes: List[Optional[int]] = [None for _ in range(self.bs)]
        for i, v in zip(index, new_nodes):
            self.trajectories[i].append(v)
            if v == 'EOS':
                self.dones[i] = True
            elif len(self.trajectories[i]) == self.max_size:
                self.dones[i] = True
            elif self.graph.outer_boundary(self.trajectories[i]) == 0:
                self.dones[i] = True
            else:
                full_new_nodes[i] = v
        delta_x_nodes = self.make_single_node_encoding(full_new_nodes)
        return delta_x_nodes

    def step(self, new_nodes: List[Union[int, str]], index: List[int],fn,reward_list):
        assert len(new_nodes) == len(index)

        full_new_nodes: List[Optional[int]] = [None for _ in range(self.bs)]
        for i, v in zip(index, new_nodes):

            self.trajectories[i].append(v)

            if v == 'EOS' or v == 1000:
                self.dones[i] = True
                self.trajectories[i].pop(-1)
            elif len(self.trajectories[i]) == self.max_size:
                self.dones[i] = True
            elif self.graph.outer_boundary(self.trajectories[i]) == 0:
                self.dones[i] = True
            else:
                full_new_nodes[i] = v

            if self.dones[i] == True:
                # logps, values, entropys = [nn.utils.rnn.pad_sequence([torch.stack(x) for x in episode_xs], True)
                #                            for episode_xs in [episode_logps, episode_values, episode_entropys]]
                # print(self.trajectories[i])
                final_scores = fn([self.trajectories[i]],[self.data[i][0]]).item()
                # if final_scores>0.5:
                #     reward_list[i] = 0.5+np.power(final_scores-0.5,0.5)
                # else:
                #     reward_list[i] = 0.5 - np.power(0.5-final_scores , 0.5)
                reward_list[i] = final_scores
                # print(final_scores)

        delta_x_nodes = self.make_single_node_encoding(full_new_nodes)
        # full_new_nodes:[node_A, node_B, node_C ... ]
        return delta_x_nodes, self.trajectories , reward_list, self.dones

    def make_single_node_encoding_2(self, nodes: List[int]) :
        ind = [ [nodes[0]],[0]]
        # ind = [[nodes], [0]]
        ind = np.asarray(ind, dtype=np.int64)
        data = np.ones(ind.shape[1], dtype=np.float32)
        return sp.csc_matrix((data, ind),  shape = [self.n_nodes,1])

    def make_single_node_encoding(self, nodes: List[int]):
        bs = len(nodes)

        assert bs == self.bs
        ind = np.array([[v, i] for i, v in enumerate(nodes) if v is not None], dtype=np.int64).T

        if len(ind):
            data = np.ones(ind.shape[1], dtype=np.float32)

            return sp.csc_matrix((data, ind), shape=[self.n_nodes, bs])
        else:
            return sp.csc_matrix((self.n_nodes, bs), dtype=np.float32)

    def make_nodes_encoding(self, nodes: List[List[int]]):
        bs = len(nodes)
        assert bs == self.bs
        ind = [[v, i] for i, vs in enumerate(nodes) for v in vs]
        ind = np.asarray(ind, dtype=np.int64).T
        if len(ind):
            data = np.ones(ind.shape[1], dtype=np.float32)
            # test_b = sp.csc_matrix((data, ind), shape=[self.n_nodes, bs])

            return sp.csc_matrix((data, ind), shape=[self.n_nodes, bs])
        else:
            return sp.csc_matrix((self.n_nodes, bs), dtype=np.float32)


class Generator:

    def __init__(self,args, graph: Graph, model: Agent, optimizer,
                 device: Optional[torch.device] = None,
                 entropy_coef: float = 1e-6,
                 n_rollouts: int = 10,
                 max_size: int = 25,
                 k: int = 3,
                 alpha: float = 0.85,
                 max_reward: float = 1.,
                 ):
        self.args = args
        self.graph = graph
        self.model = model
        self.model_gflow = model
        self.optimizer = optimizer
        self.entropy_coef = entropy_coef
        self.max_reward = max_reward
        self.n_nodes = self.graph.n_nodes
        self.max_size = max_size # 设定的最大size是20个
        self.n_rollouts = n_rollouts
        self.conv = GraphConv(graph, k, alpha)
        self.nodefeats = None
        if device is None:
            self.device = torch.device('cpu')
        else:
            self.device = device

    def load_nodefeats(self, x):
        #self.nodefeats = torch.from_numpy(x).float().to(self.device)
        self.nodefeats = x.numpy()
        #self.nodefeats = x.astype(np.float32)

    def generate(self, seeds: List[int], fn_score, max_size: Optional[int] = None):
        max_size = self.max_size if max_size is None else max_size
        env = ExpansionEnv(self.graph, [[s] for s in seeds], max_size)

        self.model.eval()
        with torch.no_grad():
            episodes, *_ = self._sample_trajectories(env,fn_score)

        return episodes

    def generate_2(self, seeds: List[int], fn, max_size: Optional[int] = None):
        max_size = self.max_size if max_size is None else max_size
        env = ExpansionEnv(self.graph, [[s] for s in seeds], max_size)

        self.model.eval()
        with torch.no_grad():
            episodes, *_ = self._sample_trajectories_2(env,fn)

        return episodes

    def sample_episodes(self, seeds: List[int], fn,max_size: Optional[int] = None):
        max_size = self.max_size if max_size is None else max_size
        env = ExpansionEnv(self.graph, [[s] for s in seeds], max_size)
        return self._sample_trajectories(env,fn)

    def sample_rollouts(self, prefix: List[List[int]], max_size: Optional[int] = None):
        max_size = self.max_size if max_size is None else max_size
        bs = len(prefix)
        if bs * self.n_rollouts > 10000:
            return self._sample_rollouts_loop(prefix, max_size)
        else:
            return self._sample_rollouts_batch(prefix, max_size)


    def prepare_flow_calculation(self,selected_nodes,rewards):
        batch =[]
        for j in range(len(selected_nodes)):
            item = selected_nodes[j]

            for i in range(1,len(item)-1):
                # parent, action , current, reward , done
                batch+=[[item[:i],item[i],item[:i+1],rewards[j][i], True if i==len(item)-2 else False]]
        return batch



    def prepare_new_inputs(self, trajectory,z_nodes,z_seeds):
        vals_attr = [] if self.nodefeats is not None else None
        vals_node = []
        vals_seed = []
        indptr = []
        offset = 0
        batch_candidates = []
        # for i in valid_index:
        trajectory_nodes = trajectory.int().numpy().tolist()
        boundary_nodes = self.graph.outer_boundary(trajectory_nodes)
        candidate_nodes = list(boundary_nodes)
        # assert len(candidate_nodes)
        involved_nodes = candidate_nodes + trajectory_nodes

        if 1000 in involved_nodes:
            involved_nodes = involved_nodes[:-1]

        batch_candidates.append(candidate_nodes)
        if self.nodefeats is not None:
            vals_attr.append(self.nodefeats[involved_nodes])
        vals_node.append(z_nodes.T[:, involved_nodes].todense())
        # vals_node.append(z_nodes.T[i, involved_nodes].todense())
        vals_seed.append(z_seeds.T[:,involved_nodes].todense())

        indptr.append((offset, offset + len(involved_nodes), offset + len(candidate_nodes)))
        offset += len(involved_nodes)
        if self.nodefeats is not None:
            # vals_attr = torch.cat(vals_attr, 0)
            vals_attr = np.concatenate(vals_attr, 0)
            vals_attr = torch.from_numpy(vals_attr).to(self.device)

        vals_seed = np.array(np.concatenate(vals_seed, 1))[0]
        vals_seed = torch.from_numpy(vals_seed).to(self.device)

        vals_node = np.array(np.concatenate(vals_node, 1))[0]
        vals_node = torch.from_numpy(vals_node).to(self.device)
        indptr = np.array(indptr)

        return vals_attr, vals_seed, vals_node, indptr, batch_candidates

    def learn_from(self,batch):

        loginf = tf([1000])
        batch_parent = []
        batch_action = []
        batch_sp =[]
        batch_done =[]
        batch_reward = []

        new_model_inputs_parent = []
        new_model_inputs_current = []

        for (parents, parents_z_node, action, sp, sp_z_node, z_node_seed,reward, done) in batch:
            for i in range(len(parents)):
                batch_parent += [[parents[i],parents_z_node[i],z_node_seed]]
                batch_action += [action[i]]
            batch_sp += [[sp,sp_z_node,z_node_seed]]
            batch_done += [done]
            batch_reward += [reward]
           # *model_inputs, batch_candidates = self._prepare_inputs(valid_index, env.trajectories, z_nodes, z_seeds)
           #  # return vals_attr, vals_seed, vals_node, indptr, batch_candidates
           #  # model_inputs :  vals_attr, vals_seed, vals_node, indptr,
           #  # Todo
           #
           #
           #  # batch_logits, values = self.model(*model_inputs) # 跟原来self.model是一样的

        new_model_inputs_parent += [self.prepare_new_inputs(trajectory,z_nodes,z_node_seed) for (trajectory, z_nodes,z_node_seed) in batch_parent]
        new_model_inputs_current += [self.prepare_new_inputs(trajectory,z_nodes,z_node_seed) for (trajectory, z_nodes,z_node_seed) in batch_sp]
        # vals_attr, vals_seed, vals_node, indptr, batch_candidates
        batch_logits_parent_list = []
        batch_logits_current_list = []

        values_parent_list = []
        values_current_list = []

        parent_Qsa = []

        batch_idxs = torch.LongTensor(
            sum([[i] * len(parents) for i, (parents,_, _,_,_,_,_,_) in enumerate(batch)], []))
        for item, action_index in zip(new_model_inputs_parent,batch_action):
            # vals_attr, vals_seed, vals_node, indptr, batch_candidates
            batch_logits, values = self.model(item[0],item[1],item[2],item[3])
            candidates = item[4][0]
            action_item  = int(action_index.item())
            if action_item != 1000:
                action_ix = candidates.index(action_item)
            else:
                action_ix = -1
            batch_logits_parent_list += [batch_logits]
            values_parent_list += [values]
            parent_Qsa += [batch_logits[0][action_ix]]



        for item in new_model_inputs_current:
            if len(item[0])!= 0:
                batch_logits, values = self.model(item[0],item[1],item[2],item[3])
            else:
                batch_logits = 0
                values = 0
            batch_logits_current_list += [batch_logits[0]]
            values_current_list += [values]
        parents_Qsa = torch.stack(parent_Qsa)
        # in_flow = torch.log(torch.zeros((len(new_model_inputs_current),))
        #                     .index_add_(0, batch_idxs, torch.exp(parents_Qsa)))
        in_flow = torch.Tensor(torch.zeros((len(new_model_inputs_current),))
                            .index_add_(0, batch_idxs, torch.exp(parents_Qsa)))
        done = torch.stack((batch_done)).squeeze(1)
        r = torch.stack((batch_reward)).squeeze(1)


        # tl = torch.stack((batch_logits_current_list))
        test_list= [i[0].unsqueeze(0) for i in batch_logits_current_list]

        next_qd = [ (test_list[i] * (1 - done[i]) + (done[i] * (-loginf))) for i in range(len(test_list))]

        test_a = torch.log(r)[:, None]

        of_list = []
        for i in range(len(test_a)):
            # a1 = torch.log(r)[:, None][i]
            # a2 = next_qd[i]
            # test_b = torch.cat([torch.log(r)[:, None][i],next_qd[i]],0)

            test_c = torch.sum(torch.exp(torch.cat([torch.log(r)[:, None][i],next_qd[i]],0)), 0)
            of_list.append(test_c)

        out_flow = torch.stack(of_list)



        loss = (in_flow - out_flow).pow(2).mean()


        term_loss = ((in_flow - out_flow) * done).pow(2).sum() / (done.sum() + 1e-20)
        flow_loss = ((in_flow - out_flow) * (1 - done)).pow(2).sum() / ((1 - done).sum() + 1e-20)

        loss_total = term_loss*25+flow_loss
        # loss_total = term_loss
        return loss, term_loss,flow_loss,loss_total

    def train_from_lists(self, episodes: List[List[int]], max_size: Optional[int] = None):
        episodes = [(x + ['EOS']) if x[-1] != 'EOS' and x[-1] != 1000 else x for x in episodes]

        max_size = self.max_size if max_size is None else max_size
        self.model.train()
        self.optimizer.zero_grad()
        env = ExpansionEnv(self.graph, [[x[0]] for x in episodes], max_size)
        bs = env.bs
        x_seeds, delta_x_nodes = env.reset()
        z_seeds = self.conv(x_seeds)
        z_nodes = sp.csc_matrix((self.n_nodes, bs), dtype=np.float32)
        episode_logps = [[] for _ in range(bs)]
        episode_values = [[] for _ in range(bs)]
        k = 0
        while not env.done:
            k += 1
            z_nodes += self.conv(delta_x_nodes)
            valid_index = env.valid_index
            *model_inputs, batch_candidates = self._prepare_inputs(valid_index, env.trajectories, z_nodes, z_seeds)
            batch_logits, values = self.model(*model_inputs)
            logps = []
            actions = []
            for logits, candidates, i in zip(batch_logits, batch_candidates, valid_index):
                v = episodes[i][k]
                try:
                    action = candidates.index(v)
                except ValueError:
                    action = len(candidates)
                actions.append(action)
                logps.append(logits[action])
            new_nodes = [x[i] if i < len(x) else 'EOS' for i, x in zip(actions, batch_candidates)]
            delta_x_nodes = env.step_2(new_nodes, valid_index)
            for i, v1, v2 in zip(valid_index, logps, values):
                episode_logps[i].append(v1)
                episode_values[i].append(v2)
        # Stack and Padding
        logps, values = [nn.utils.rnn.pad_sequence([torch.stack(x) for x in episode_xs], True)
                         for episode_xs in [episode_logps, episode_values]]
        lengths = torch.LongTensor([len(x) for x in episodes]).to(self.device)
        mask = torch.arange(logps.size(1), device=self.device,
                            dtype=torch.int64).expand(bs, -1) < (lengths - 1).unsqueeze(1)
        mask = mask.float()
        n = mask.sum()
        # td_loss = ((values - self.max_reward) ** 2 * mask).sum() / n
        policy_loss = -(1. * logps * mask).sum() / n
        # loss = policy_loss + .5 * td_loss
        # loss.backward()
        policy_loss.backward()
        self.optimizer.step()
        return policy_loss.item()

    def train_from_rewards(self, seeds: List[int], fn):
        bs = len(seeds)
        self.model.train()
        selected_nodes, logps, values, entropys, batch= self.sample_episodes(seeds,fn)

        #
        # lengths = torch.LongTensor([len(x) for x in selected_nodes]).to(self.device)
        # # Compute Rewards Matrix
        # rewards = np.zeros(logps.shape, dtype=np.float32)
        # final_scores = fn(selected_nodes)
        # rewards[np.arange(logps.size(0)), lengths.cpu().numpy() - 2] = final_scores
        # rewards are negative number or 0  , ndarray (128,19)
        #
        # # To do : 把reward 放进batch里面
        # # batch_new = self.prepare_flow_calculation(selected_nodes,rewards)
        # print(batch)

        losses = self.learn_from(batch)

        if losses is not None:

            self.optimizer.zero_grad()
            # loss_ = losses[0].requires_grad_()
            # losses[3].backward()
            losses[0].backward()
            self.optimizer.step()
        # for name, parms in self.model.named_parameters():
        #     print('-->name:', name)
        #     print('-->para:', parms)
        #     print('-->grad_requirs:', parms.requires_grad)
        #     print('-->grad_value:', parms.grad)
        #     print("===")
        #
        # exit()
        # print('loss', losses[0])
        # print('loss_sum', losses[3])


        return losses
        # loss = policy_loss # + .5 * value_loss - self.entropy_coef * entropy_loss
        # loss.backward()
        # self.optimizer.step()
        # return (selected_nodes,
        #         np.mean(final_scores),
        #         policy_loss.item(),
        #         # value_loss.item(),
        #         # entropy_loss.item(),
        #         lengths.float().mean().item())


    # def train_from_lists(self, episodes: List[List[int]], max_size: Optional[int] = None):
    #     episodes = [(x + ['EOS']) if x[-1] != 'EOS' else x for x in episodes]
    #     max_size = self.max_size if max_size is None else max_size
    #     self.model.train()
    #     self.optimizer.zero_grad()
    #     env = ExpansionEnv(self.graph, [[x[0]] for x in episodes], max_size)
    #     bs = env.bs
    #     x_seeds, delta_x_nodes = env.reset()
    #     z_seeds = self.conv(x_seeds)
    #     z_nodes = sp.csc_matrix((self.n_nodes, bs), dtype=np.float32)
    #     episode_logps = [[] for _ in range(bs)]
    #     episode_values = [[] for _ in range(bs)]
    #     k = 0
    #     while not env.done:
    #         k += 1
    #         z_nodes += self.conv(delta_x_nodes)
    #         valid_index = env.valid_index
    #         *model_inputs, batch_candidates = self._prepare_inputs(valid_index, env.trajectories, z_nodes, z_seeds)
    #         batch_logits, values = self.model(*model_inputs)
    #         logps = []
    #         actions = []
    #         for logits, candidates, i in zip(batch_logits, batch_candidates, valid_index):
    #             v = episodes[i][k]
    #             try:
    #                 action = candidates.index(v)
    #             except ValueError:
    #                 action = len(candidates)
    #             actions.append(action)
    #             logps.append(logits[action])
    #         new_nodes = [x[i] if i < len(x) else 'EOS' for i, x in zip(actions, batch_candidates)]
    #         delta_x_nodes = env.step(new_nodes, valid_index)
    #         for i, v1, v2 in zip(valid_index, logps, values):
    #             episode_logps[i].append(v1)
    #             episode_values[i].append(v2)
    #     # Stack and Padding
    #     logps, values = [nn.utils.rnn.pad_sequence([torch.stack(x) for x in episode_xs], True)
    #                      for episode_xs in [episode_logps, episode_values]]
    #     lengths = torch.LongTensor([len(x) for x in episodes]).to(self.device)
    #     mask = torch.arange(logps.size(1), device=self.device,
    #                         dtype=torch.int64).expand(bs, -1) < (lengths - 1).unsqueeze(1)
    #     mask = mask.float()
    #     n = mask.sum()
    #     # td_loss = ((values - self.max_reward) ** 2 * mask).sum() / n
    #     policy_loss = -(1. * logps * mask).sum() / n
    #     # loss = policy_loss + .5 * td_loss
    #     # loss.backward()
    #     policy_loss.backward()
    #     self.optimizer.step()
    #     return policy_loss.item()

    def _sample_rollouts_loop(self, prefix, max_size=None):
        env = ExpansionEnv(self.graph, prefix, max_size)
        rollouts = []
        self.model.eval()
        with torch.no_grad():
            for _ in range(self.n_rollouts):
                trajectories, *_ = self._sample_trajectories(env)
                rollouts.append(trajectories)
        return list(zip(*rollouts))

    def _sample_rollouts_batch(self, prefix, max_size=None):
        bs = len(prefix)
        env = ExpansionEnv(self.graph, prefix * self.n_rollouts, max_size)
        self.model.eval()
        with torch.no_grad():
            trajectories, *_ = self._sample_trajectories(env)
        rollouts = []
        for i in range(self.n_rollouts):
            rollouts.append(trajectories[i*bs:(i+1)*bs])
        return list(zip(*rollouts))
    def getCuttingPointAndCuttingEdge(self,edges: List[Tuple]):
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

    def parent_transition(self,index, state,z_node_vector,env,reward,z_seed):
        parent_state = []
        parent_action = []
        parent_z_nodes = []
        current_z_nodes = z_node_vector
        current_reward = reward
        cutting_dots=[]


        if state[-1] == 1000 or 'EOS' in state:

            state.pop(-1)

        if self.args.ablation == 1:
        # ---------- Version 1 --------------
            if state[-1] != 1000 and 'EOS' not in state:
                if len(state) > 2:
                    adj = env.graph.adj_mat.toarray()

                    edges = []
                    leng = len(state)
                    for i in range(leng):
                        for j in range(i, leng):
                            if adj[state[i]][state[j]] > 0:
                                edges.append((i, j))

                    cutting_dots_index, _ = self.getCuttingPointAndCuttingEdge(edges)
                    cutting_dots = [state[item] for item in cutting_dots_index]

            # print('cutting_dots',cutting_dots)
            for i in range(len(state)):
                ori_state = state.copy() # [651,650 ]

                deleted_node = ori_state[i]
                # print(650 in seed_)
                if deleted_node in cutting_dots or deleted_node in env.data[index]:
                # if deleted_node in cutting_dots or deleted_node in env.data[index]:
                    continue

                # delete_ = ori_state[i]
                ori_state.pop(i)
                parent_z_node_vector = self.conv(env.make_single_node_encoding_2(ori_state))
                parent_state += [ori_state]
                parent_action += [deleted_node]
                parent_z_nodes += [parent_z_node_vector]

        elif self.args.ablation == 2:
            # ---------- Version 2 --------------
            for i in range(len(state)):
                ori_state = state.copy() # [651,650 ]

                deleted_node = ori_state[i]
                # print(650 in seed_)
                if deleted_node in env.data[index]:
                # if deleted_node in cutting_dots or deleted_node in env.data[index]:
                    continue

                # delete_ = ori_state[i]
                ori_state.pop(i)
                parent_z_node_vector = self.conv(env.make_single_node_encoding_2(ori_state))

                parent_state += [ori_state]
                parent_action += [deleted_node]
                parent_z_nodes += [parent_z_node_vector]

        elif self.args.ablation == 3:
                # ---------- Version 3 --------------
            for i in range(len(state)):
                ori_state = state.copy()  # [651,650 ]

                deleted_node = ori_state[i]
                # print(650 in seed_)

                # delete_ = ori_state[i]
                ori_state.pop(i)
                parent_z_node_vector = self.conv(env.make_single_node_encoding_2(ori_state))

                parent_state += [ori_state]
                parent_action += [deleted_node]
                parent_z_nodes += [parent_z_node_vector]

        else:

            ori_state = state.copy()  # [651,650 ]
            deleted_node = ori_state[-1]
            ori_state.pop(-1)
            parent_z_node_vector = self.conv(env.make_single_node_encoding_2(ori_state))
            parent_state += [ori_state]
            parent_action += [deleted_node]
            parent_z_nodes += [parent_z_node_vector]

        return parent_state, parent_z_nodes, parent_action, env.dones[index], current_z_nodes, current_reward,z_seed



    def _sample_trajectories(self, env: ExpansionEnv,fn):
        bs = env.bs

        x_seeds, delta_x_nodes = env.reset()
        # delta_x_nodes is the z_nodes here
        z_seeds = self.conv(x_seeds)# equation(2) in paper
        z_nodes = sp.csc_matrix((self.n_nodes, bs), dtype=np.float32) # (1231,32)


        # z_nodes_tensor = torch.sparse_coo_tensor ((self.n_nodes, bs), dtype=np.float32)# 在这里创建维度为 ( n_node, batch_size ) 的矩阵
        # print('z_nodes',z_nodes)
        z_nodes_update = sp.csc_matrix((self.n_nodes, bs), dtype=np.float32)
        episode_logps = [[] for _ in range(bs)]
        episode_values = [[] for _ in range(bs)]
        episode_entropys = [[] for _ in range(bs)]

        batch = []
        # done_l = [False] * bs
        reward_list = [0] * bs
        z_nodes_update += self.conv(delta_x_nodes)
        # ll_diff =  torch.zeros((bs,)) # .to(device)
        # Z = torch.zeros((1,))  # .to(device)
        # ll_diff += Z
        step=0
        while not env.done:
            step+=1
            z_nodes += self.conv(delta_x_nodes) # 这些delta_x_nodes就是纯粹的new_node_added

            valid_index = env.valid_index
            *model_inputs, batch_candidates = self._prepare_inputs(valid_index, env.trajectories, z_nodes, z_seeds)
            # return vals_attr, vals_seed, vals_node, indptr, batch_candidates
            # model_inputs :  vals_attr, vals_seed, vals_node, indptr,
            # Todo


            batch_logits, values = self.model(*model_inputs) # 跟原来self.model是一样的

            # if actions is not None:
            #     ll_diff[~done_list] -= back_logits.gather(1, action[action != ndim].unsqueeze(1)).squeeze(1)

            actions, logps, entropys = self._sample_actions(batch_logits,step,batch_candidates)

            new_nodes = [x[i] if i < len(x) else 'EOS' for i, x in zip(actions, batch_candidates)]
            # sample actions 其实是在sample它的index

            step_new = env.step(new_nodes, valid_index ,fn,reward_list)

            delta_x_nodes= step_new[0]
            z_nodes_update += self.conv(delta_x_nodes)




            p_a = [self.parent_transition(i,step_new[1][i],z_nodes_update[:,i],env,step_new[2][i],z_seeds[:,i]) for i in range(bs)]



            batch += [[item for item in (tf(parent_state).to(self.device), parent_z_nodes,
                                         tf(eos_to_number(action)).to(self.device),
                                         tf(eos_to_number(current_state)).to(self.device),
                                         current_z_nodes,
                                         z_node_seed,
                                         tf([reward]), tf([done]).to(self.device))]
                      for (parent_state, parent_z_nodes, action, done, current_z_nodes,reward,z_node_seed), current_state in
                      zip(p_a, step_new[1])]

            # step_new = delta_x_nodes, trajectory_current, reward_list, done_list



            # batch += [[item for item in (tf(parent_state).to(self.device),
            #                              tf(eos_to_number(action)).to(self.device),
            #
            #                              tf([done]).to(self.device))]
            #           for (parent_state, parent_z_nodes, action, done, current_z_nodes), current_state in
            #           zip(p_a, trajectory_current)]

            #
            #
            # batch += [ [ item for item in (tf(parent_state).to(self.device), csc_to_torch(parent_z_nodes),
            #        tf(action).to(self.device), tf([eos_to_number(current_state)]).to(self.device),csc_to_torch(current_z_nodes),tf([done]).to(self.device) )]
            #            for (parent_state,parent_z_nodes,action,done,current_z_nodes), current_state in zip(p_a, trajectory_current) ]
            # batch 里有 parent state list, p_z_node ( single column ) list , action list, current state, current_z_node list, done
            # To do: add current z_nodes
            for i, v1, v2, v3 in zip(valid_index, logps, values, entropys):
                episode_logps[i].append(v1)
                episode_values[i].append(v2)
                episode_entropys[i].append(v3)
        # Stack and Padding
        logps, values, entropys = [nn.utils.rnn.pad_sequence([torch.stack(x) for x in episode_xs], True)
                                   for episode_xs in [episode_logps, episode_values, episode_entropys]]

        return env.trajectories, logps, values, entropys , batch


    def _sample_trajectories_2(self, env: ExpansionEnv,fn):
        bs = env.bs

        x_seeds, delta_x_nodes = env.reset()
        # delta_x_nodes is the z_nodes here
        z_seeds = self.conv(x_seeds)# equation(2) in paper
        z_nodes = sp.csc_matrix((self.n_nodes, bs), dtype=np.float32) # (1231,32)


        z_nodes_update = sp.csc_matrix((self.n_nodes, bs), dtype=np.float32)

        batch = []

        reward_list = [0] * bs
        z_nodes_update += self.conv(delta_x_nodes)

        step=0
        while not env.done:
            step+=1
            z_nodes += self.conv(delta_x_nodes) # 这些delta_x_nodes就是纯粹的new_node_added

            valid_index = env.valid_index
            *model_inputs, batch_candidates = self._prepare_inputs(valid_index, env.trajectories, z_nodes, z_seeds)


            batch_logits, values = self.model(*model_inputs) # 跟原来self.model是一样的

            actions, logps, entropys = self._sample_actions(batch_logits,step,batch_candidates)

            new_nodes = [x[i] if i < len(x) else 'EOS' for i, x in zip(actions, batch_candidates)]

            step_new = env.step(new_nodes, valid_index ,fn,reward_list)

            delta_x_nodes= step_new[0]
            z_nodes_update += self.conv(delta_x_nodes)


        return env.trajectories, logps, values, entropys , batch

    def _prepare_inputs(self, valid_index: List[int], trajectories: List[List[int]],
                        z_nodes: sp.csc_matrix, z_seeds: sp.csc_matrix):
        vals_attr = [] if self.nodefeats is not None else None
        vals_seed = []
        vals_node = []
        indptr = []
        offset = 0
        batch_candidates = []
        for i in valid_index:
            boundary_nodes = self.graph.outer_boundary(trajectories[i])
            candidate_nodes = list(boundary_nodes)
            # assert len(candidate_nodes)
            involved_nodes = candidate_nodes + trajectories[i]  # 把当前graph的节点和它周边的邻居节点都找出来  # 1-ego net
            batch_candidates.append(candidate_nodes)  # candidates
            if self.nodefeats is not None:
                vals_attr.append(self.nodefeats[involved_nodes])
            vals_seed.append(z_seeds.T[i, involved_nodes].todense())
            vals_node.append(z_nodes.T[i, involved_nodes].todense())
            indptr.append((offset, offset + len(involved_nodes), offset + len(candidate_nodes)))
            offset += len(involved_nodes)
        if self.nodefeats is not None:
            # vals_attr = torch.cat(vals_attr, 0)
            vals_attr = np.concatenate(vals_attr, 0)
            vals_attr = torch.from_numpy(vals_attr).to(self.device)
        vals_seed = np.array(np.concatenate(vals_seed, 1))[0]
        vals_node = np.array(np.concatenate(vals_node, 1))[0]
        vals_seed = torch.from_numpy(vals_seed).to(self.device)
        vals_node = torch.from_numpy(vals_node).to(self.device)
        indptr = np.array(indptr)
        return vals_attr, vals_seed, vals_node, indptr, batch_candidates

    def _sample_actions(self, batch_logits: List,step,bc) -> (List, List, List):
        batch = []
        temp=0
        for logits in batch_logits:
            ps = torch.exp(logits) # 这个就是来计算action的probability，然后用multinomial那个函数来sample action
            entropy = -(ps * logits).sum()


            action = torch.multinomial(ps, 1).item()
            while(action>=len(bc[temp]) and step < 5 ):
                action = torch.multinomial(ps, 1).item()
            # if task == "node_task":
            #     while(action>=len(bc[temp]) and step < 5 ):
            #         action = torch.multinomial(ps, 1).item()
            # else:
            #     while (action >= len(bc[temp])and step < 5 ):
            #         action = torch.multinomial(ps, 1).item()
            logp = logits[action]
            batch.append([action, logp, entropy])
            temp+=1
        actions, logps, entropys = zip(*batch)
        actions = np.array(actions)
        return actions, logps, entropys
