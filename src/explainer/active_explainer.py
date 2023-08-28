from src.train_gnn import TrainModel
from src.dataset import GraphDataset
from src.utils.mask_utils import keep_hard, remove_hard
import torch
import os
import random
import time
from copy import deepcopy
from gnn.model import get_gnnNets
from gendata import get_dataloader
from utils.io_utils import write_to_json
from utils.gen_utils import extract_hard_explanation
from explainer.gsat import GSAT, ExtractorMLP, gsat_get_config
from explainer.explainer_utils.gsat import init_metric_dict, save_checkpoint, load_checkpoint
from explainer.graphcfe import GraphCFE, train, test, add_list_in_dict, compute_counterfactual
from torch.optim.lr_scheduler import ReduceLROnPlateau


def explain_gsat_active_graph(model, data, target, device, **kwargs):
    dataset_name = kwargs["dataset_name"]
    explainer_name = kwargs["explainer_name"]
    seed = kwargs["seed"]
    num_class = kwargs["num_classes"]

    active_selection = kwargs["active_selection"]
    active_loss = kwargs["active_loss"]
    active_threshold = kwargs["active_threshold"]
    active_epochs = kwargs["active_epochs"]
    active_model = kwargs["active_model"]
    active_lambda = kwargs["active_lambda"]

    dataloader_params = {
            "batch_size": kwargs["batch_size"],
            "random_split_flag": kwargs["random_split_flag"],
            "data_split_ratio": kwargs["data_split_ratio"],
            "seed": kwargs["seed"],
        }

    subdir = os.path.join(kwargs["model_save_dir"], explainer_name)
    os.makedirs(subdir, exist_ok=True)
    gsat_active_saving_path = os.path.join(subdir, f"{explainer_name}_{dataset_name}_{str(device)}_{seed}.pt")

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

    metric_dict = deepcopy(init_metric_dict)

    gsat = GSAT(model, extractor, optimizer, scheduler, device, subdir, dataset_name, num_class, multi_label, seed, method_config, shared_config)


    if os.path.isfile(gsat_active_saving_path):
        print("Load saved Active GSAT model...")
        load_checkpoint(extractor, subdir, model_name=f'{explainer_name}_{dataset_name}_{str(device)}_{seed}')
        extractor = extractor.to(device)
        gsat = GSAT(model, extractor, optimizer, scheduler, device, subdir, dataset_name, num_class, multi_label, seed, method_config, shared_config)

    else:
        print('====================================')
        print('[INFO] Active Training GSAT...')
        t0 = time.time()
        
        ##### Start active Training of GSAT #####
        for epoch in range(active_epochs):
            # Pick 500 graphs to train and test
            train_size = min(len(kwargs["dataset"]), 500)
            explain_dataset_idx = random.sample(range(len(kwargs["dataset"])), k=train_size)
            explain_dataset = kwargs["dataset"][explain_dataset_idx]

            loader, train_dataset, _, test_dataset = get_dataloader(explain_dataset, **dataloader_params)
            metric_dict = gsat.train(loader, test_dataset, metric_dict, use_edge_attr=True)

            gsat.extractor = gsat.extractor.to(device)

            ## Compute explanatory subgraphs
            expl_subgraphs = []
            for explain_data in explain_dataset:
                explain_data.batch = torch.zeros(explain_data.num_nodes, dtype=torch.long)
                explain_data = explain_data.to(device)
                edge_att, loss_dict, clf_logits = gsat.eval_one_batch(explain_data, epoch=method_config['epochs'])
                edge_mask = edge_att # attention scores
                ### Remove the top 20% important edges
                if active_selection == 'rmv':
                    thresh_edge_mask = remove_hard(edge_mask, threshold=active_threshold)
                ### Keep the top 20% important edges
                elif active_selection == 'keep':
                    thresh_edge_mask = keep_hard(edge_mask, threshold=active_threshold)
                expl_subgraphs.append(extract_hard_explanation(explain_data, thresh_edge_mask, device))
            expl_subgraphs = GraphDataset(expl_subgraphs)

            # Train and test the explanations on new GNN
            if active_model == 'new':
                model = get_gnnNets(kwargs["num_node_features"], kwargs["num_classes"], kwargs)
            trainer = TrainModel(
                model=model,
                dataset=expl_subgraphs,
                device=device,
                graph_classification=eval(kwargs["graph_classification"]),
                save_dir=None,
                save_name=None,
                dataloader_params=dataloader_params,
            )
            trainer.train(
                    train_params=kwargs["train_params"],
                    optimizer_params=kwargs["optimizer_params"],
                )
            scores, preds = trainer.test()
            # Update the GSAT model loss
            sign = 1 if active_selection == 'keep' else -1
            score = -scores["test_acc"] if active_loss == 'acc' else scores["test_loss"]
            gsat.ginex_loss = active_lambda * sign * score
            print(f"GSAT active training epoch {epoch} loss: {gsat.ginex_loss}")

        train_time = time.time() - t0
        # writer.add_hparams(hparam_dict=hparam_dict, metric_dict=metric_dict)
        save_checkpoint(gsat.extractor, subdir, model_name=f'{explainer_name}_{dataset_name}_{str(device)}_{seed}')
        
        train_time_file = os.path.join(subdir, f"gsat_train_time.json")
        entry = {"name":explainer_name, "dataset": dataset_name, "train_time": train_time, "seed": seed, "device": str(device)}
        write_to_json(entry, train_time_file)

    data.batch = torch.zeros(data.num_nodes, dtype=torch.long)
    gsat.extractor = gsat.extractor.to(device)
    data = data.to(device)
    edge_att, loss_dict, clf_logits = gsat.eval_one_batch(data, epoch=method_config['epochs'])
    edge_mask = edge_att # attention scores

    return edge_mask, None











def explain_graphcfe_active_graph(model, data, target, device, **kwargs):


    active_selection = kwargs["active_selection"]
    active_loss = kwargs["active_loss"]
    active_threshold = kwargs["active_threshold"]
    active_epochs = kwargs["active_epochs"]
    active_model = kwargs["active_model"]

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
        print("Load saved Active GraphCFE model...")
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

        ##### Start active Training of GSAT #####
        for epoch in range(active_epochs):
            train(train_params)

            ## Compute explanatory subgraphs
            expl_subgraphs = []
            for explain_data in explain_dataset:
                explain_data.batch = torch.zeros(explain_data.num_nodes, dtype=torch.long)
                explain_data = explain_data.to(device)
                if hasattr(explain_data, 'y_cf'):
                    y_cf = explain_data.y_cf
                else:
                    y_cf = 1 - explain_data.y
                eval_results, edge_mask = compute_counterfactual(dataset_name, explain_data, metrics, y_cf, graphcfe_model, model, device)
                ### Remove the top 20% important edges
                if active_selection == 'rmv':
                    thresh_edge_mask = remove_hard(edge_mask, threshold=active_threshold)
                ### Keep the top 20% important edges
                elif active_selection == 'keep':
                    thresh_edge_mask = keep_hard(edge_mask, threshold=active_threshold)
                expl_subgraphs.append(extract_hard_explanation(explain_data, thresh_edge_mask, device))
            expl_subgraphs = GraphDataset(expl_subgraphs)

             # Train and test the explanations on new GNN
            if active_model == 'new':
                model = get_gnnNets(kwargs["num_node_features"], kwargs["num_classes"], kwargs)
            trainer = TrainModel(
                model=model,
                dataset=expl_subgraphs,
                device=device,
                graph_classification=eval(kwargs["graph_classification"]),
                save_dir=None,
                save_name=None,
                dataloader_params=dataloader_params,
            )
            trainer.train(
                    train_params=kwargs["train_params"],
                    optimizer_params=kwargs["optimizer_params"],
                )
            scores, preds = trainer.test()
            # Update the GSAT model loss
            sign = 1 if active_selection == 'keep' else -1
            score = -scores["test_acc"] if active_loss == 'acc' else scores["test_loss"]
            graphcfe_model.ginex_loss = sign * score
            print(f"GraphCFE active training epoch {epoch} loss: {trainer.ginex_loss}")

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