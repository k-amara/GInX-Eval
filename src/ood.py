import os
from src.explain import get_mask_dir_path
from src.utils.mask_utils import transform_edge_masks
import torch
import yaml
import numpy as np
from src.dataset import GraphDataset
import pickle
from gendata import get_dataset
from utils.parser_utils import (
    arg_parse,
    create_args_group,
    fix_random_seed,
    get_data_args,
    get_graph_size_args,
)
import pandas as pd
from torch_geometric.utils import degree
from torch_geometric.data import Data
from evaluation.in_distribution.ood_stat import eval_graph_list
from torch_geometric.utils import to_networkx


def main(args, args_group):
    fix_random_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    dataset_params = args_group["dataset_params"]
    model_params = args_group["model_params"]

    # For active explainability
    args.train_params = args_group["train_params"]
    args.optimizer_params = args_group["optimizer_params"]

    dataset = get_dataset(
        dataset_root=args.data_save_dir,
        **dataset_params,
    )
    dataset.data.x = dataset.data.x.float()
    dataset.data.y = dataset.data.y.squeeze().long()
    args = get_data_args(dataset, args)
    model_params["edge_dim"] = args.edge_dim

    # Statistics of the dataset
    # Number of graphs, number of node features, number of edge features, average number of nodes, average number of edges
    info = {
        "num_graphs": len(dataset),
        "num_nf": args.num_node_features,
        "num_ef": args.edge_dim,
        "avg_num_nodes": np.mean([data.num_nodes for data in dataset])
        if eval(args.graph_classification)
        else dataset.data.num_nodes,
        "avg_num_edges": np.mean([data.edge_index.shape[1] for data in dataset])
        if eval(args.graph_classification)
        else dataset.data.num_edges,
        "avg_degree": np.mean(
            [degree(data.edge_index[0]).mean().item() for data in dataset]
        )
        if eval(args.graph_classification)
        else degree(dataset.data.edge_index[0]).mean().item(),
        "num_classes": args.num_classes,
    }
    print(info)

    if len(dataset) > 1:
        args.max_num_nodes = max([d.num_nodes for d in dataset])
    else:
        args.max_num_nodes = dataset.data.num_nodes


    ###### Load saved explanations ######
    mask_save_name = get_mask_dir_path(args)
    save_dir = os.path.join(args.mask_save_dir, args.dataset_name, args.explainer_name)
    save_path = os.path.join(save_dir, mask_save_name)
    with open(save_path, "rb") as f:
        w_list = pickle.load(f)
    list_explained_y, edge_masks, node_feat_masks, computation_time = tuple(w_list)


    #### Generate new datasets with explanatory masks ####
    # Modify dataset with the edge masks
    if args.explainer_name == "truth":
        thresh_edge_masks = edge_masks
    else: 
        thresh_edge_masks = transform_edge_masks(edge_masks, strategy="keep", threshold=0.2)
    #hard_exp = []
    test_graph = []
    pred_graph = []
    for i, data in enumerate(dataset):
        assert data.idx.detach().cpu().item() == list_explained_y[i]
        hard_edge_index = data.edge_index[:, thresh_edge_masks[i]>0]
        hard_edge_attr = data.edge_attr[thresh_edge_masks[i]>0]
        hard_edge_weight = torch.ones(hard_edge_attr.size(0), dtype=torch.float, device=device)
        hard_nodes = torch.sort(torch.unique(hard_edge_index))[0]
        hard_x = data.x[hard_nodes]
        dict = {hard_nodes[i].item(): i for i in range(len(hard_nodes))}
        hard_hard_edge_index = torch.tensor([[dict[hard_edge_index[0, j].item()], dict[hard_edge_index[1, j].item()]] for j in range(hard_edge_index.size(1))], dtype=torch.long, device=device).t()
        hard_data = Data(x = hard_x, edge_index = hard_hard_edge_index, edge_attr = hard_edge_attr, edge_weight = hard_edge_weight, y = data.y, idx = data.idx)
        #hard_exp.append(hard_data)

        G_ori = to_networkx(data, to_undirected=True)
        G_pred = to_networkx(hard_data, to_undirected=True)
        test_graph.append(G_ori)
        pred_graph.append(G_pred)
        if i > 500:
            break


    #hard_exp = GraphDataset(hard_exp)


    ##### Compare distribution of dataset and hard_dataset #####
    MMD = eval_graph_list(test_graph, pred_graph)
    MMD['dataset'] = args.dataset_name
    MMD['explainer']=args.explainer_name
    print('MMD: ', MMD)

    mmd_save_path = os.path.join(args.result_save_dir, "ood_mmd.csv")
    df_mmd = pd.DataFrame(MMD, index=[0])
    print(df_mmd)
    with open(mmd_save_path, 'a') as f:
        df_mmd.to_csv(f, header=f.tell()==0)

if __name__ == "__main__":
    parser, args = arg_parse()
    args = get_graph_size_args(args)

    # Get the absolute path to the parent directory of the current file
    parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

    # Load the config file
    config_path = os.path.join(parent_dir, "configs", "dataset.yaml")
    # read the configuration file
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    # loop through the config and add any values to the parser as arguments
    for key, value in config[args.dataset_name].items():
        setattr(args, key, value)

    args_group = create_args_group(parser, args)
    main(args, args_group)
