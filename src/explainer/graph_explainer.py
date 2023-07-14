import numpy as np
import torch
import os
import dill
import random
import time
from copy import deepcopy
from captum.attr import IntegratedGradients, Saliency
from torch_geometric.data import Data
from torch_geometric.utils import to_networkx
from gnn.model import GCNConv, GATConv, GINEConv, TransformerConv
from gendata import get_dataloader
from utils.io_utils import write_to_json

from explainer.gnnexplainer import TargetedGNNExplainer
from explainer.gradcam import GraphLayerGradCam
from explainer.rcexplainer import RCExplainer_Batch, train_rcexplainer
from explainer.explainer_utils.rcexplainer.rc_train import test_policy
from explainer.graphcfe import GraphCFE, train, test, baseline_cf, add_list_in_dict, compute_counterfactual
from explainer.gsat import GSAT, ExtractorMLP, gsat_get_config
from explainer.explainer_utils.gsat import Writer, init_metric_dict, save_checkpoint, load_checkpoint
from torch.optim.lr_scheduler import ReduceLROnPlateau

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def get_all_convolution_layers(model):
    layers = []
    for module in model.modules():
        if (
            isinstance(module, GCNConv)
            or isinstance(module, GATConv)
            or isinstance(module, GINEConv)
            or isinstance(module, TransformerConv)
        ):
            layers.append(module)
    return layers


def gpu_to_cpu(data, device):
    data.x = torch.FloatTensor(data.x.cpu().numpy().copy()).to(device)
    data.edge_index = torch.LongTensor(data.edge_index.cpu().numpy().copy()).to(device)
    data.edge_attr = torch.FloatTensor(data.edge_attr.cpu().numpy().copy()).to(device)
    return data


def model_forward_graph(x, model, edge_index, edge_attr):
    out = model(x, edge_index, edge_attr)
    return out


def node_attr_to_edge(edge_index, node_mask):
    edge_mask = np.zeros(edge_index.shape[1])
    edge_mask += node_mask[edge_index[0].cpu().numpy()]
    edge_mask += node_mask[edge_index[1].cpu().numpy()]
    return edge_mask


#### Baselines ####
def explain_random_graph(model, data, target, device, **kwargs):
    edge_mask = np.random.uniform(size=data.edge_index.shape[1])
    node_feat_mask = np.random.uniform(size=data.x.shape[1])
    return edge_mask.astype("float"), node_feat_mask.astype("float")


def explain_sa_graph(model, data, target, device, **kwargs):
    saliency = Saliency(model_forward_graph)
    input_mask = data.x.clone().requires_grad_(True).to(device)
    saliency_mask = saliency.attribute(
        input_mask,
        target=target,
        additional_forward_args=(model, data.edge_index, data.edge_attr),
        abs=False,
    )
    # 1 node feature mask per node.
    node_feat_mask = saliency_mask.cpu().numpy()
    node_attr = node_feat_mask.sum(axis=1)
    edge_mask = node_attr_to_edge(data.edge_index, node_attr)
    return edge_mask.astype("float"), node_feat_mask.astype("float")


def explain_ig_graph(model, data, target, device, **kwargs):
    ig = IntegratedGradients(model_forward_graph)
    input_mask = data.x.clone().requires_grad_(True).to(device)
    ig_mask = ig.attribute(
        input_mask,
        target=target,
        additional_forward_args=(model, data.edge_index, data.edge_attr),
        internal_batch_size=input_mask.shape[0],
    )
    node_feat_mask = ig_mask.cpu().detach().numpy()
    node_attr = node_feat_mask.sum(axis=1)
    edge_mask = node_attr_to_edge(data.edge_index, node_attr)
    return edge_mask.astype("float"), node_feat_mask.astype("float")


def explain_occlusion_graph(model, data, target, device, **kwargs):
    data = Data(x=data.x, edge_index=data.edge_index, edge_attr=data.edge_attr)
    if target is None:
        pred_probs = model(data)[0].cpu().detach().numpy()
        pred_prob = pred_probs.max()
        target = pred_probs.argmax()
    else:
        pred_prob = 1
    g = to_networkx(data)
    edge_occlusion_mask = np.ones(data.num_edges, dtype=bool)
    edge_mask = np.zeros(data.num_edges)
    edge_index_numpy = data.edge_index.cpu().numpy()
    for i in range(data.num_edges):
        # if include_edges is not None and not include_edges[i].item():
        # continue
        u, v = list(edge_index_numpy[:, i])
        if (u, v) in g.edges():
            edge_occlusion_mask[i] = False
            prob = model(
                data.x,
                data.edge_index[:, edge_occlusion_mask],
                data.edge_attr[edge_occlusion_mask],
            )[0][target].item()
            edge_mask[i] = pred_prob - prob
            edge_occlusion_mask[i] = True
    return edge_mask.astype("float"), None


def explain_basic_gnnexplainer_graph(model, data, target, device, **kwargs):
    data = gpu_to_cpu(data, device)
    explainer = TargetedGNNExplainer(
        model,
        num_hops=kwargs["num_layers"],
        return_type="prob",
        epochs=1000,
        edge_ent=kwargs["edge_ent"],
        edge_size=kwargs["edge_size"],
        allow_edge_mask=True,
        allow_node_mask=False,
        device=device,
    )
    _, edge_mask = explainer.explain_graph_with_target(
        x=data.x,
        edge_index=data.edge_index,
        edge_attr=data.edge_attr,
        target=target,
    )
    edge_mask = edge_mask.cpu().detach().numpy()
    return edge_mask.astype("float"), None


def explain_gnnexplainer_graph(model, data, target, device, **kwargs):
    data = gpu_to_cpu(data, device)
    explainer = TargetedGNNExplainer(
        model,
        num_hops=kwargs["num_layers"],
        return_type="prob",
        epochs=1000,
        edge_ent=kwargs["edge_ent"],
        edge_size=kwargs["edge_size"],
        allow_edge_mask=True,
        allow_node_mask=True,
        device=device,
    )
    node_feat_mask, edge_mask = explainer.explain_graph_with_target(
        x=data.x,
        edge_index=data.edge_index,
        edge_attr=data.edge_attr,
        target=target,
    )
    edge_mask = edge_mask.cpu().detach().numpy()
    node_feat_mask = node_feat_mask.cpu().detach().numpy()
    return edge_mask, node_feat_mask


def explain_gradcam_graph(model, data, target, device, **kwargs):
    # Captum default implementation of LayerGradCam does not average over nodes for different channels because of
    # different assumptions on tensor shapes
    input_mask = data.x.clone().requires_grad_(True).to(device)
    layers = get_all_convolution_layers(model)
    node_attrs = []
    for layer in layers:
        layer_gc = GraphLayerGradCam(model_forward_graph, layer)
        node_attr = layer_gc.attribute(
            input_mask,
            target=target,
            additional_forward_args=(
                model,
                data.edge_index,
                data.edge_attr,
            ),
        )
        node_attrs.append(node_attr.squeeze().cpu().detach().numpy())
    node_attr = np.array(node_attrs).mean(axis=0)
    edge_mask = sigmoid(node_attr_to_edge(data.edge_index, node_attr))
    return edge_mask.astype("float"), None


def explain_rcexplainer_graph(model, data, target, device, **kwargs):
    dataset_name = kwargs["dataset_name"]
    seed = kwargs["seed"]
    rcexplainer = RCExplainer_Batch(model, device, kwargs['num_classes'], hidden_size=kwargs['hidden_dim'])
    subdir = os.path.join(kwargs["model_save_dir"], "rcexplainer")
    os.makedirs(subdir, exist_ok=True)
    rcexplainer_saving_path = os.path.join(subdir, f"rcexplainer_{dataset_name}_{str(device)}_{seed}.pickle")
    if os.path.isfile(rcexplainer_saving_path):
        print("Load saved RCExplainer model...")
        rcexplainer_model = dill.load(open(rcexplainer_saving_path, "rb"))
        rcexplainer_model = rcexplainer_model.to(device)
    else:
       # data loader
        train_size = min(len(kwargs["dataset"]), 500)
        explain_dataset_idx = random.sample(range(len(kwargs["dataset"])), k=train_size)
        explain_dataset = kwargs["dataset"][explain_dataset_idx]
        dataloader_params = {
            "batch_size": kwargs["batch_size"],
            "random_split_flag": kwargs["random_split_flag"],
            "data_split_ratio": kwargs["data_split_ratio"],
            "seed": kwargs["seed"],
        }
        loader, train_dataset, _, test_dataset = get_dataloader(explain_dataset, **dataloader_params)
        t0 = time.time()
        lr, weight_decay, topk_ratio = 0.01, 1e-5, 1.0
        rcexplainer_model = train_rcexplainer(rcexplainer, train_dataset, test_dataset, loader, dataloader_params['batch_size'], lr, weight_decay, topk_ratio)
        train_time = time.time() - t0
        print("Save RCExplainer model...")
        dill.dump(rcexplainer_model, file = open(rcexplainer_saving_path, "wb"))
        train_time_file = os.path.join(subdir, f"rcexplainer_train_time.json")
        entry = {"dataset": dataset_name, "train_time": train_time, "seed": seed, "device": str(device)}
        write_to_json(entry, train_time_file)
    
    max_budget = data.num_edges
    state = torch.zeros(max_budget, dtype=torch.bool)
    data.batch = torch.zeros(data.num_nodes, dtype=torch.long)
    edge_ranking = test_policy(rcexplainer_model, model, data, device)
    edge_mask = 1 - edge_ranking/len(edge_ranking)
    # edge_mask[i]: indicate the i-th edge to be added in the search process, i.e. that gives the highest reward.
    return edge_mask, None
 
    

def explain_graphcfe_graph(model, data, target, device, **kwargs):
    dataset_name = kwargs["dataset_name"]
    y_cf_all = kwargs['y_cf_all']
    seed = kwargs["seed"]

    # data loader
    train_size = min(len(kwargs["dataset"]), 500)
    explain_dataset_idx = random.sample(range(len(kwargs["dataset"])), k=train_size)
    explain_dataset = kwargs["dataset"][explain_dataset_idx]
    dataloader_params = {
        "batch_size": kwargs["batch_size"],
        "random_split_flag": kwargs["random_split_flag"],
        "data_split_ratio": kwargs["data_split_ratio"],
        "seed": kwargs["seed"],
    }
    loader, _, _, _ = get_dataloader(explain_dataset, **dataloader_params)
    
    # metrics
    metrics = ['validity', 'proximity_x', 'proximity_a']

    subdir = os.path.join(kwargs["model_save_dir"], "graphcfe")
    os.makedirs(subdir, exist_ok=True)
    graphcfe_saving_path = os.path.join(subdir, f"graphcfe_{dataset_name}_{str(device)}_{seed}.pth")
     # model
    init_params = {'hidden_dim': kwargs["hidden_dim"], 'dropout': kwargs["dropout"], 'num_node_features': kwargs["num_node_features"], 'max_num_nodes': kwargs["max_num_nodes"]}
    graphcfe_model = GraphCFE(init_params=init_params, device=device)

    if os.path.isfile(graphcfe_saving_path):
        print("Load saved GraphCFE model...")
        state_dict = torch.load(graphcfe_saving_path)
        graphcfe_model.load_state_dict(state_dict)
        graphcfe_model = graphcfe_model.to(device)
    else:
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-5)
        graphcfe_model = graphcfe_model.to(device)
        train_params = {'epochs': 4000, 'model': graphcfe_model, 'pred_model': model, 'optimizer': optimizer,
                        'y_cf': y_cf_all,
                        'train_loader': loader['train'], 'val_loader': loader['eval'], 'test_loader': loader['test'],
                        'dataset': dataset_name, 'metrics': metrics, 'save_model': False}
        t0 = time.time()
        train(train_params)
        train_time = time.time() - t0
        print("Save GraphCFE model...")
        torch.save(graphcfe_model.state_dict(), graphcfe_saving_path)
        train_time_file = os.path.join(subdir, f"graphcfe_train_time.json")
        entry = {"dataset": dataset_name, "train_time": train_time, "seed": seed, "device": str(device)}
        write_to_json(entry, train_time_file)
    # test
    test_params = {'model': graphcfe_model, 'dataset': dataset_name, 'data_loader': loader['test'], 'pred_model': model,
                       'metrics': metrics, 'y_cf': y_cf_all}
    eval_results = test(test_params)
    results_all_exp = {}
    for k in metrics:
        results_all_exp = add_list_in_dict(k, results_all_exp, eval_results[k].detach().cpu().numpy())
    for k in eval_results:
        if isinstance(eval_results[k], list):
            print(k, ": ", eval_results[k])
        else:
            print(k, f": {eval_results[k]:.4f}")

    # baseline
    # num_rounds, type = 10, "random"
    # eval_results = baseline_cf(dataset_name, data, metrics, y_cf, model, device, num_rounds=num_rounds, type=type)
    
    if hasattr(data, 'y_cf'):
        y_cf = data.y_cf
    else:
        y_cf = 1 - data.y
    eval_results, edge_mask = compute_counterfactual(dataset_name, data, metrics, y_cf, graphcfe_model, model, device)
    results_all_exp = {}
    for k in metrics:
        results_all_exp = add_list_in_dict(k, results_all_exp, eval_results[k].detach().cpu().numpy())
    for k in eval_results:
        if isinstance(eval_results[k], list):
            print(k, ": ", eval_results[k])
        else:
            print(k, f": {eval_results[k]:.4f}")
    return edge_mask, None



##### Groundtruth Explanations #####

def explain_truth_graph(model, data, target, device, **kwargs):
    if not eval(kwargs["groundtruth"]):
        return None, None
    else: 
        return data.edge_mask.cpu().detach().numpy(), None
    
def explain_inverse_graph(model, data, target, device, **kwargs):
    if not eval(kwargs["groundtruth"]):
        return None, None
    else: 
        return (1-data.edge_mask).cpu().detach().numpy(), None
    


def explain_gsat_graph(model, data, target, device, **kwargs):
    dataset_name = kwargs["dataset_name"]
    seed = kwargs["seed"]
    num_class = kwargs["num_classes"]

    subdir = os.path.join(kwargs["model_save_dir"], "gsat")
    os.makedirs(subdir, exist_ok=True)
    gsat_saving_path = os.path.join(subdir, f"gsat_{dataset_name}_{str(device)}_{seed}.pt")

    # config gsat training
    shared_config, method_config = gsat_get_config()
    if dataset_name == 'mnist':
        multi_label = True
    else:
        multi_label = False
    extractor = ExtractorMLP(kwargs['hidden_dim'], shared_config).to(device)
    lr, wd = method_config['lr'], method_config.get('weight_decay', 0)
    optimizer = torch.optim.Adam(list(extractor.parameters()) + list(model.parameters()), lr=lr, weight_decay=wd)
    scheduler_config = method_config.get('scheduler', {})
    scheduler = None if scheduler_config == {} else ReduceLROnPlateau(optimizer, mode='max', **scheduler_config)

    # writer = Writer(log_dir=subdir)
    # hparam_dict = {"dataset": dataset_name, "seed": seed, "device": str(device), "model": kwargs['model_name']}
    metric_dict = deepcopy(init_metric_dict)
    # writer.add_hparams(hparam_dict=hparam_dict, metric_dict=metric_dict)

    # data loader
    train_size = min(len(kwargs["dataset"]), 500)
    explain_dataset_idx = random.sample(range(len(kwargs["dataset"])), k=train_size)
    explain_dataset = kwargs["dataset"][explain_dataset_idx]
    dataloader_params = {
        "batch_size": kwargs["batch_size"],
        "random_split_flag": kwargs["random_split_flag"],
        "data_split_ratio": kwargs["data_split_ratio"],
        "seed": kwargs["seed"],
    }
    loader, train_dataset, _, test_dataset = get_dataloader(explain_dataset, **dataloader_params)


    if os.path.isfile(gsat_saving_path):
        print("Load saved GSAT model...")
        load_checkpoint(extractor, subdir, model_name=f'gsat_{dataset_name}_{str(device)}_{seed}')
        extractor = extractor.to(device)
        gsat = GSAT(model, extractor, optimizer, scheduler, device, subdir, dataset_name, num_class, multi_label, seed, method_config, shared_config)

    else:
        print('====================================')
        print('[INFO] Training GSAT...')
        gsat = GSAT(model, extractor, optimizer, scheduler, device, subdir, dataset_name, num_class, multi_label, seed, method_config, shared_config)
        t0 = time.time()
        metric_dict = gsat.train(loader, test_dataset, metric_dict, use_edge_attr=True)
        train_time = time.time() - t0
        # writer.add_hparams(hparam_dict=hparam_dict, metric_dict=metric_dict)
        save_checkpoint(gsat.extractor, subdir, model_name=f'gsat_{dataset_name}_{str(device)}_{seed}')
        
        train_time_file = os.path.join(subdir, f"gsat_train_time.json")
        entry = {"dataset": dataset_name, "train_time": train_time, "seed": seed, "device": str(device)}
        write_to_json(entry, train_time_file)
        

    data.batch = torch.zeros(data.num_nodes, dtype=torch.long)
    gsat.extractor = gsat.extractor.to(device)
    data = data.to(device)
    edge_att, loss_dict, clf_logits = gsat.eval_one_batch(data, epoch=method_config['epochs'])
    edge_mask = edge_att # attention scores
    return edge_mask, None
