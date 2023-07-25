import os
from explain import explain_main
from src.dataset import GraphDataset
import torch
import yaml
import numpy as np
import pandas as pd
import copy
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


def main(args, args_group):
    fix_random_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    dataset_params = args_group["dataset_params"]
    model_params = args_group["model_params"]

    dataset = get_dataset(
        dataset_root=args.data_save_dir,
        **dataset_params,
    )
    dataset.data.x = dataset.data.x.float()
    dataset.data.y = dataset.data.y.squeeze().long()
    args = get_data_args(dataset, args)
    model_params["edge_dim"] = args.edge_dim

    data_y = dataset.data.y.cpu().numpy()
    if args.num_classes == 2:
        y_cf_all = 1 - data_y
    else:
        y_cf_all = []
        for y in data_y:
            y_cf_all.append(y + 1 if y < args.num_classes - 1 else 0)
    args.y_cf_all = torch.FloatTensor(y_cf_all).to(device)

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


    ###### Train Model ######
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
    if eval(args.graph_classification):
        trainer = TrainModel(
            model=model,
            dataset=dataset,
            device=device,
            graph_classification=eval(args.graph_classification),
            save_dir=os.path.join(args.model_save_dir, args.dataset_name),
            save_name=model_save_name,
            dataloader_params=dataloader_params,
        )
    else:
        trainer = TrainModel(
            model=model,
            dataset=dataset,
            device=device,
            graph_classification=eval(args.graph_classification),
            save_dir=os.path.join(args.model_save_dir, args.dataset_name),
            save_name=model_save_name,
        )
    if Path(os.path.join(trainer.save_dir, f"{trainer.save_name}_best.pth")).is_file():
        trainer.load_model()
    else:
        trainer.train(
            train_params=args_group["train_params"],
            optimizer_params=args_group["optimizer_params"],
        )
    scores, preds = trainer.test()
    scores['threshold'] = 0
    scores['seed'] = args.seed
    df_scores = pd.DataFrame(scores, index=[0])
    save_path = os.path.join(
        args.result_save_dir+'_sub', args.dataset_name, args.explainer_name
    )
    os.makedirs(save_path, exist_ok=True)
    scores_save_path = os.path.join(save_path, f"{model_save_name}_scores.csv")
    with open(scores_save_path, 'w') as f:
        df_scores.to_csv(f, header=f.tell()==0)



    ###### Generate Explanations ######
    list_explained_y, edge_masks, node_feat_masks, computation_time = explain_main(dataset, trainer.model, device, args)

    ###### Retrain with Graph degradation ######
    for t in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]:
        thresh_edge_masks = transform_edge_masks(edge_masks, strategy=args.retrain_strategy, threshold=t)
        # Modify dataset with the edge masks
        new_dataset = []
        for i, data in enumerate(dataset):
            assert data.idx.detach().cpu().item() == list_explained_y[i]
            new_data = copy.deepcopy(data)
            new_data.edge_index = data.edge_index[:, thresh_edge_masks[i]>0]
            new_data.edge_attr = data.edge_attr[thresh_edge_masks[i]>0]
            new_data.edge_weight = None
            new_data.x = data.x[np.unique(new_data.edge_index)]
            new_dataset.append(new_data)
        new_dataset = GraphDataset(new_dataset)

        model = get_gnnNets(args.num_node_features, args.num_classes, model_params)
        if eval(args.graph_classification):
            trainer = TrainModel(
                model=model,
                dataset=new_dataset,
                device=device,
                graph_classification=eval(args.graph_classification),
                save_dir=os.path.join(args.model_save_dir, args.dataset_name, args.explainer_name),
                save_name=model_save_name + f"_{args.explainer_name}_sub_thresh_{t}",
                dataloader_params=dataloader_params,
            )
        else:
            trainer = TrainModel(
                model=model,
                dataset=new_dataset,
                device=device,
                graph_classification=eval(args.graph_classification),
                save_dir=os.path.join(args.model_save_dir, args.dataset_name, args.explainer_name),
                save_name=model_save_name + f"_{args.explainer_name}_sub_thresh_{t}",
            )
        if Path(os.path.join(trainer.save_dir, f"{trainer.save_name}_best.pth")).is_file():
            trainer.load_model()
        else:
            trainer.train(
                train_params=args_group["train_params"],
                optimizer_params=args_group["optimizer_params"],
            )
        scores, preds = trainer.test()
        scores['threshold'] = t
        scores['seed'] = args.seed
        df_scores = pd.DataFrame(scores, index=[0])
        print(df_scores)
        with open(scores_save_path, 'a') as f:
            df_scores.to_csv(f, header=f.tell()==0)



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
