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
from gnn.model import get_gnnNets
from train_gnn import TrainModel
from gendata import get_dataset
from utils.mask_utils import transform_edge_masks
from utils.parser_utils import (
    arg_parse,
    create_args_group,
    fix_random_seed,
    get_data_args,
    get_graph_size_args,
)
from pathlib import Path
from torch_geometric.utils import degree
from torch_geometric.data import Data
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import seaborn as sns
import umap.umap_ as umap


pal = sns.color_palette("Paired", 15)
dict_names = {"random": 'Random', "sa":'Saliency', "ig": "IntegratedGrad", "gradcam": 'GradCAM', "occlusion": 'Occlusion',
              "basic_gnnexplainer": 'GNNExplainer', "gnnexplainer": 'GNNExplainer(E,NF)', 'gsat_active':'GSAT_Active',
              "pgmexplainer": 'PGMExplainer', "subgraphx": 'SubgraphX', "pgexplainer": 'PGExplainer', "graphcfe": "GraphCFE", "gflowexplainer": "GFlowExplainer", "rcexplainer": "RCExplainer", "gsat": "GSAT", "diffexplainer": "D4Explainer", "inverse": "Inverse", "truth": "Truth"}
dict_color = {"none":"dimgrey", "Random":'black', "Distance":pal[1], "PageRank":pal[2], "Saliency": 'olivedrab', "IntegratedGrad": 'forestgreen', "GradCAM": "yellowgreen", "Occlusion":'peru', 
              "GNNExplainer": pal[7], "GNNExplainer(E,NF)": 'orangered', 
              "PGMExplainer":'firebrick', "SubgraphX": "gold", "PGExplainer": 'orchid', "GraphCFE": "darkviolet", "GFlowExplainer": 'lightblue', "RCExplainer":'mediumpurple', "GSAT": "slategrey", "D4Explainer": "darkblue", "Inverse": "dodgerblue", "Truth": pal[5]}
dict_color = {"none":"dimgrey", "to_be_explained":'black', "GNNExplainer(E,NF)": pal[0], "GSAT": pal[1], "Truth": pal[5]}

dict_dataset = {'benzene':'Benzene', 'ba_2motifs':'BA-2Motifs', 'ba_house_grid':'BA-HouseGrid', 'ieee24_mc':'IEEE24', 'ieee39_mc':'IEEE39', 'ieee118_mc':'IEEE118', 'uk_mc': 'UK', "mutag": "MUTAG", "mutag_s":"Mutag_s", "mnist_bin": "MNIST", "bbbp":"BBBP", "ba_multishapes":"BA-MultiShapes"}

plt.rcParams['figure.dpi'] = 600
plt.rcParams['savefig.dpi'] = 600

def get_graph_embeddings(args, args_group):
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


    ##### Compare distribution of dataset and hard_dataset #####
    if eval(args.graph_classification):
        args.data_split_ratio = [args.train_ratio, args.val_ratio, args.test_ratio]
        dataloader_params = {
            "batch_size": args.batch_size,
            "random_split_flag": eval(args.random_split_flag),
            "data_split_ratio": args.data_split_ratio,
            "seed": args.seed,
        }
    model = get_gnnNets(args.num_node_features, args.num_classes, model_params)
    model_save_name = f"{args.model_name}_{args.num_layers}l_{args.seed}"
    trainer = TrainModel(
        model=model,
        dataset=dataset,
        device=device,
        graph_classification=eval(args.graph_classification),
        save_dir=os.path.join(args.model_save_dir, 'initial', args.dataset_name),
        save_name=model_save_name,
        dataloader_params=dataloader_params,
    )
    if Path(os.path.join(trainer.save_dir, f"{trainer.save_name}_best.pth")).is_file():
        trainer.load_model()
    else:
        trainer.train(
            train_params=args_group["train_params"],
            optimizer_params=args_group["optimizer_params"],
        )

    model.eval()

    graph_emb, graph_y, graph_explainer = [], [], []
    for i, data in enumerate(dataset):
        if i==0:
            print(data)
        data_emb = model.get_graph_rep(data)
        graph_emb.append(data_emb.detach().numpy().squeeze())
        graph_y.append(data.y.detach().cpu().item())
        graph_explainer.append('none')
        if i == 100:
            break

    for explainer in ['truth', 'gnnexplainer', 'gsat']:
        ###### Load saved explanations ######
        mask_save_name = "mask_{}_{}_{}_{}_{}.pkl".format(
            args.dataset_name,
            args.model_name,
            explainer,
            args.focus,
            args.seed,
        )
        save_dir = os.path.join(args.mask_save_dir, args.dataset_name, explainer)
        save_path = os.path.join(save_dir, mask_save_name)
        with open(save_path, "rb") as f:
            w_list = pickle.load(f)
        list_explained_y, init_edge_masks, node_feat_masks, computation_time = tuple(w_list)
        edge_masks = [] 
        for mask in init_edge_masks:
            if torch.is_tensor(mask):
                mask = mask.detach().numpy()
            edge_masks.append(mask)
        edge_masks = np.array(edge_masks)
        #### Generate new datasets with explanatory masks ####
        # Modify dataset with the edge masks
        if explainer == "truth":
            thresh_edge_masks = edge_masks
        else: 
            thresh_edge_masks = transform_edge_masks(edge_masks, strategy="keep", threshold=0.2)
        
        k = 0
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
            
            if data.y.detach().cpu().item() == 0:
                if k==0:
                    data_to_be_explained = i
                k += 1
                if k > 1:
                    break
                else:
                    graph_emb.append(model.get_graph_rep(hard_data).detach().numpy().squeeze())
                    graph_y.append(hard_data.y.detach().cpu().item())
                    graph_explainer.append(explainer)
                
        graph_explainer[data_to_be_explained]='to_be_explained'

    return graph_emb, graph_y, graph_explainer

def plot_t_sne(graph_emb, graph_y, graph_explainer, args):
    tsne = TSNE(n_components=2, verbose=1, random_state=123)
    z = tsne.fit_transform(graph_emb)
    df = pd.DataFrame()
    df["y"] = graph_y
    df["explainer"] = graph_explainer
    df["comp-1"] = z[:,0]
    df["comp-2"] = z[:,1]
    #df['dataset'] = df['dataset'].replace(dict_dataset)
    df['explainer'] = df['explainer'].replace(dict_names)

    sns.set_context("notebook", rc={"xtick.labelsize" : 16, "ytick.labelsize" : 16})
    fonttitle = 16
    fontlegend = 15

    fig, ax = plt.subplots(1, figsize=(7,6))
    dict_marker = {0: 'X', 1: 'o'}
    
    for i in dict_color.keys():
        df_subset = df[df['explainer']==i]
        size = 100 if i == 'none' else 250
        sns.scatterplot(data=df_subset, x="comp-1", y="comp-2", color=dict_color[i], style="y", markers=dict_marker, ax=ax, label=i, legend=True, s=size, alpha=0.8)
    
    ax.legend().set_visible(False)
    ax.set_xlabel('Dimension 1', fontsize=fontlegend)
    ax.set_ylabel('Dimension 2', fontsize=fontlegend)
    labels_color, handles_color = [], []
    full_labels_marker, full_handles_marker = [], []

    for handle, label in zip(*ax.get_legend_handles_labels()):
        if label in ['GNNExplainer(E,NF)', 'GSAT', 'Truth']:
            labels_color.append(label)
            handles_color.append(handle)
        elif label in ['0','1']:
            full_labels_marker.append(label)
            full_handles_marker.append(handle)
        
    labels_marker, index = np.unique(full_labels_marker, return_index=True)
    handles_marker = [full_handles_marker[i] for i in index]

    label_dict = {'0':'Toxic', '1':'Non-toxic'}

    leg_color = fig.legend(handles_color, labels_color, ncol=1, loc = 'upper left', bbox_to_anchor=(1, 0.7), fontsize=fontlegend, handlelength=1, title="Explainers", title_fontsize=fontlegend, labelspacing = 0.8, frameon=False, markerscale=0.8)
    leg_marker = fig.legend(handles_marker, list((pd.Series(labels_marker)).map(label_dict)), ncol=1, loc = 'upper left', bbox_to_anchor=(1, 0.9), fontsize=fontlegend, handlelength=1, title="Labels", title_fontsize=fontlegend, labelspacing = 0.8, frameon=False, markerscale=1.4)

    leg_color._legend_box.align = "left"
    leg_marker._legend_box.align = "left"

    
    ax.set_title('T-SNE plot of graph embeddings', fontsize=fonttitle)
    #sns.despine()
    plt.tight_layout(pad=1.8)
    plt.show()
    plt.savefig(f'/cluster/home/kamara/GInX-Eval/figures/projection/{args.dataset_name.upper()} data T-SNE projection.png', bbox_inches='tight')

    
def plot_umap(graph_emb, graph_y, graph_explainer, args):
    reducer = umap.UMAP()
    z = reducer.fit_transform(graph_emb)
    df = pd.DataFrame()
    df["y"] = graph_y
    df["explainer"] = graph_explainer
    df["comp-1"] = z[:,0]
    df["comp-2"] = z[:,1]
    #df['dataset'] = df['dataset'].replace(dict_dataset)
    df['explainer'] = df['explainer'].replace(dict_names)


    sns.set_context("notebook", rc={"xtick.labelsize" : 16, "ytick.labelsize" : 16})
    fonttitle = 16
    fontlegend = 15

    fig, ax = plt.subplots(1, figsize=(7,6))
    dict_marker = {0: 'X', 1: 'o'}
    for i in dict_color.keys():
        df_subset = df[df['explainer']==i]
        size = 100 if i == 'none' else 250
        sns.scatterplot(data=df_subset, x="comp-1", y="comp-2", color=dict_color[i], style="y", markers=dict_marker, ax=ax, label=i, legend=True, s=size, alpha=0.8)
        ax.legend().set_visible(False)

    ax.set_xlabel('Dimension 1', fontsize=fontlegend)
    ax.set_ylabel('Dimension 2', fontsize=fontlegend)
    labels_color, handles_color = [], []
    full_labels_marker, full_handles_marker = [], []

    for handle, label in zip(*ax.get_legend_handles_labels()):
        if label in ['GNNExplainer(E,NF)', 'GSAT', 'Truth']:
            labels_color.append(label)
            handles_color.append(handle)
        elif label in ['0','1']:
            full_labels_marker.append(label)
            full_handles_marker.append(handle)
        
    labels_marker, index = np.unique(full_labels_marker, return_index=True)
    handles_marker = [full_handles_marker[i] for i in index]

    label_dict = {'0':'Toxic', '1':'Non-toxic'}

    leg_color = fig.legend(handles_color, labels_color, ncol=1, loc = 'upper left', bbox_to_anchor=(1, 0.7), fontsize=fontlegend, handlelength=1, title="Explainers", title_fontsize=fontlegend, labelspacing = 0.8, frameon=False, markerscale=0.8)
    leg_marker = fig.legend(handles_marker, list((pd.Series(labels_marker)).map(label_dict)), ncol=1, loc = 'upper left', bbox_to_anchor=(1, 0.9), fontsize=fontlegend, handlelength=1, title="Labels", title_fontsize=fontlegend, labelspacing = 0.8, frameon=False, markerscale=1.4)

    leg_color._legend_box.align = "left"
    leg_marker._legend_box.align = "left"

    
    ax.set_title('UMAP plot of graph embeddings', fontsize=fonttitle)
    #sns.despine()
    plt.tight_layout(pad=1.8)
    plt.show()
    plt.savefig(f'/cluster/home/kamara/GInX-Eval/figures/projection/{args.dataset_name.upper()} data UMAP projection.png', bbox_inches='tight')



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
    graph_emb, graph_y, graph_exp = get_graph_embeddings(args, args_group)
    plot_t_sne(graph_emb, graph_y, graph_exp, args)
    plot_umap(graph_emb, graph_y, graph_exp, args)
