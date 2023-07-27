import torch
from src.utils.gen_utils import from_edge_index_to_adj, padded_datalist
from torch_geometric.data import InMemoryDataset



class GraphDataset(InMemoryDataset):
    def __init__(self, data_list, root='.', name='NewDataset'):
        self.name = name.lower()
        super().__init__(root=root)
        #self.data_list = data_list
        self.data, self.slices = self.collate(data_list)

    """
    def process(self):
        new_data_list = []
        adj_list = []
        max_num_nodes = 0
        for data in self.data_list:
            max_num_nodes = max(max_num_nodes, data.num_nodes)
            adj = from_edge_index_to_adj(
                data.edge_index, data.edge_attr, data.num_nodes
            )
            adj_list.append(adj)
            new_data_list.append(data)
        new_data_list = padded_datalist(new_data_list, adj_list, max_num_nodes)
        self.data_list = new_data_list
        self.data, self.slices = self.collate(new_data_list)
    """