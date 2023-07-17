# https://github.com/flyingdoog/PGExplainer/blob/master/MUTAG.ipynb

from torch_geometric.data import InMemoryDataset


class GraphDataset(InMemoryDataset):
    def __init__(self, data_list, root='.', name='NewDataset'):
        self.name = name.lower()
        super().__init__(root=root)
        self.data, self.slices = self.collate(data_list)

    