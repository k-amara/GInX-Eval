import pickle as pkl
import numpy as np
import os
from numpy.random.mtrand import RandomState
import torch

from src.explainer.explainer_utils.gflowexplainer2.datasets.utils import preprocess_features, preprocess_adj, adj_to_edge_index
from src.gendata import get_dataset


def load_graph_dataset(_dataset, shuffle=True, **kwargs):
    """Load a graph dataset and optionally shuffle it.

    :param _dataset: Which dataset to load. Choose from "ba2" or "mutag"
    :param shuffle: Boolean. Wheter to suffle the loaded dataset.
    :returns: np.array
    """
    dataset_root = kwargs['data_save_dir']
    _dataset = kwargs['dataset_name']
    # Load the chosen dataset from the pickle file.
    dataset = get_dataset(dataset_root, **kwargs)
    n_graphs = dataset.data.y.shape[0]
    indices = np.arange(0, n_graphs)
    if shuffle:
        prng = RandomState(42) # Make sure that the permutation is always the same, even if we set the seed different
        indices = prng.permutation(indices)

    # Create shuffled data
    dataset = dataset[indices]
    #adjs = adjs[indices]
    edge_index = [np.array(data.edge_index) for data in dataset]
    features = torch.tensor([data.x_padded.detach().tolist() for data in dataset])
    b = np.zeros((len(dataset.data.y), kwargs["num_classes"]))
    b[np.arange(len(dataset.data.y)), dataset.data.y] = 1
    labels = torch.tensor(b)

    # Create masks
    train_indices = np.arange(0, int(n_graphs*0.8))
    val_indices = np.arange(int(n_graphs*0.8), int(n_graphs*0.9))
    test_indices = np.arange(int(n_graphs*0.9), n_graphs)
    train_mask = np.full((n_graphs), False, dtype=bool)
    train_mask[train_indices] = True
    val_mask = np.full((n_graphs), False, dtype=bool)
    val_mask[val_indices] = True
    test_mask = np.full((n_graphs), False, dtype=bool)
    test_mask[test_indices] = True


    # Transform to edge index
    #edge_index = adj_to_edge_index(adjs)

    return edge_index, features, labels, train_mask, val_mask, test_mask


def _load_node_dataset(_dataset):
    """Load a node dataset.

    :param _dataset: Which dataset to load. Choose from "syn1", "syn2", "syn3" or "syn4"
    :returns: np.array
    """
    dir_path = os.path.dirname(os.path.realpath(__file__))
    path = dir_path + '/pkls/' + _dataset + '.pkl'
    with open(path, 'rb') as fin:
        adj, features, y_train, y_val, y_test, train_mask, val_mask, test_mask, edge_label_matrix  = pkl.load(fin)
    labels = y_train #（700,4)
    labels[val_mask] = y_val[val_mask]
    labels[test_mask] = y_test[test_mask]

    return adj, features, labels, train_mask, val_mask, test_mask


def load_dataset(_dataset, skip_preproccessing=False, shuffle=True, **kwargs):
    """High level function which loads the dataset
    by calling others spesifying in nodes or graphs.

    Keyword arguments:
    :param _dataset: Which dataset to load. Choose from "syn1", "syn2", "syn3", "syn4", "ba2" or "mutag"
    :param skip_preproccessing: Whether or not to convert the adjacency matrix to an edge matrix.
    :param shuffle: Should the returned dataset be shuffled or not.
    :returns: multiple np.arrays
    """
    print(f"Loading {kwargs['dataset_name']} dataset")
    if _dataset[:3] == "syn": # Load node_dataset
        adj, features, labels, train_mask, val_mask, test_mask = _load_node_dataset(_dataset)
        # 700 nodes , 10 features, labels : (700,4)
        preprocessed_features = preprocess_features(features).astype('float32')
        if skip_preproccessing:
            graph = adj
        else:
            graph = preprocess_adj(adj)[0].astype('int64').T
            # 把矩阵里面有1的位置转化为两个ndarry来保存，比如（0，5）和（0，7）位置是1，那么就是[0,0...],[5,7...]
        labels = np.argmax(labels, axis=1)
        return graph, preprocessed_features, labels, train_mask, val_mask, test_mask
    else: # Load graph dataset
        return load_graph_dataset(_dataset, shuffle, **kwargs)
