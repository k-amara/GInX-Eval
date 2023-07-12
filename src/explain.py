import os
import time
import sklearn.metrics
import torch
import pickle
import numpy as np
import pandas as pd
from evaluate.fidelity import (
    fidelity_acc,
    fidelity_acc_inv,
    fidelity_acc_inv_ext,
    fidelity_gnn_acc,
    fidelity_gnn_acc_inv,
    fidelity_gnn_acc_inv_ext,
    fidelity_gnn_prob,
    fidelity_gnn_prob_inv,
    fidelity_prob,
    fidelity_prob_inv,
)
from evaluate.accountability import (
    accountability,
    accountability_gnn,
    accountability_ext,
    accountability_gnn_ext,
    extend_mask,
)
from utils.io_utils import check_dir
from utils.gen_utils import list_to_dict
from dataset.syn_utils.gengroundtruth import get_ground_truth_syn
from evaluate.accuracy import (
    get_explanation_syn,
    get_scores,
)
from utils.mask_utils import (
    mask_to_shape,
    clean
)
from explainer.graph_explainer import *
from pathlib import Path


class Explain(object):
    def __init__(
        self,
        model,
        dataset,
        device,
        explainer_params,
        save_dir=None,
        save_name="mask",
    ):
        self.model = model
        self.dataset = dataset  # train_mask, eval_mask, test_mask
        self.data = dataset.data
        self.dataset_name = explainer_params["dataset_name"]
        self.device = device
        self.save = save_dir is not None
        self.save_dir = save_dir
        self.save_name = save_name
        if self.save_dir is not None:
            check_dir(self.save_dir)
        self.mu = explainer_params["mu"]

        self.explainer_params = explainer_params
        self.graph_classification = eval(explainer_params["graph_classification"])
        self.task = "_graph" if self.graph_classification else "_node"

        self.explainer_name = explainer_params["explainer_name"]
        self.list_explained_data = []

        self.focus = explainer_params["focus"]
        self.mask_nature = explainer_params["mask_nature"]
        self.groundtruth = eval(explainer_params["groundtruth"])
        if self.groundtruth:
            self.num_top_edges = explainer_params["num_top_edges"]


    def _eval_top_acc(self, edge_masks):
        print("Top Accuracy is being computed...")
        scores = []
        for i in range(len(self.list_explained_data)):
            edge_mask = torch.Tensor(edge_masks[i]).to(self.device)
            graph = (
                self.dataset[self.list_explained_data[i]]
                if self.graph_classification
                else self.dataset.data
            )
            if self.dataset_name.startswith(
                tuple(
                    [
                        "ba_2motifs",
                        "mutag",
                        "benzene"
                    ]
                )
            ):
                (
                    top_f1_score,
                    top_recall,
                    top_precision,
                    top_balanced_acc,
                    top_roc_auc_score,
                ) = (np.nan, np.nan, np.nan, np.nan, np.nan)
                edge_mask = edge_mask.cpu().numpy()
                if graph.edge_mask is not None:
                    true_explanation = graph.edge_mask.cpu().numpy()
                    n = len(np.where(true_explanation == 1)[0])
                    if n > 0:
                        pred_explanation = np.zeros(len(edge_mask))
                        mask = edge_mask.copy()
                        if eval(self.directed):
                            unimportant_indices = (-mask).argsort()[n + 1 :]
                            mask[unimportant_indices] = 0
                        else:
                            mask = mask_to_shape(mask, graph.edge_index, n)
                        top_roc_auc_score = sklearn.metrics.roc_auc_score(
                            true_explanation, mask
                        )
                        pred_explanation[mask > 0] = 1
                        top_precision = sklearn.metrics.precision_score(
                            true_explanation, pred_explanation, pos_label=1
                        )
                        top_recall = sklearn.metrics.recall_score(
                            true_explanation, pred_explanation, pos_label=1
                        )
                        top_f1_score = sklearn.metrics.f1_score(
                            true_explanation, pred_explanation, pos_label=1
                        )
                        top_balanced_acc = sklearn.metrics.balanced_accuracy_score(
                            true_explanation, pred_explanation
                        )
            else:
                raise ValueError("Unknown dataset name: {}".format(self.dataset_name))
            entry = {
                "top_roc_auc_score": top_roc_auc_score,
                "top_recall": top_recall,
                "top_precision": top_precision,
                "top_f1_score": top_f1_score,
                "top_balanced_acc": top_balanced_acc,
            }
            scores.append(entry)
        accuracy_scores = pd.DataFrame.from_dict(list_to_dict(scores))
        return accuracy_scores

    def _eval_acc(self, edge_masks):
        scores = []
        num_explained_data_with_acc = 0
        for i in range(len(self.list_explained_data)):
            edge_mask = torch.Tensor(edge_masks[i]).to(self.device)
            graph = (
                self.dataset[self.list_explained_data[i]]
                if self.graph_classification
                else self.dataset.data
            )
            if (self.dataset_name.startswith(tuple(["ba", "tree"]))) & (
                not self.graph_classification
            ):
                G_true, role, true_edge_mask = get_ground_truth_syn(
                    self.list_explained_data[i], self.data, self.dataset_name
                )
                G_expl = get_explanation_syn(
                    graph, edge_mask, num_top_edges=self.num_top_edges, top_acc=False
                )
                recall, precision, f1_score = get_scores(G_expl, G_true)
                balanced_acc, roc_auc_score = np.nan, np.nan
                num_explained_data_with_acc += 1

            elif self.dataset_name.startswith(
                tuple(
                    [
                        "ba_2motifs",
                        "mutag",
                        "benzene",
                    ]
                )
            ):
                f1_score, recall, precision, balanced_acc, roc_auc_score = (
                    np.nan,
                    np.nan,
                    np.nan,
                    np.nan,
                    np.nan,
                )
                edge_mask = edge_mask.cpu().numpy()
                if graph.get("edge_mask", None) is None:
                    print(
                        f"No true explanation available for this graph {graph.idx} with label {graph.y}."
                    )
                else:
                    true_explanation = graph.edge_mask.cpu().numpy()
                    n = len(np.where(true_explanation == 1)[0])
                    if n > 0:
                        roc_auc_score = sklearn.metrics.roc_auc_score(
                            true_explanation, edge_mask
                        )
                        pred_explanation = np.zeros(len(edge_mask))
                        pred_explanation[edge_mask > 0] = 1
                        precision = sklearn.metrics.precision_score(
                            true_explanation, pred_explanation, pos_label=1
                        )
                        recall = sklearn.metrics.recall_score(
                            true_explanation, pred_explanation, pos_label=1
                        )
                        f1_score = sklearn.metrics.f1_score(
                            true_explanation, pred_explanation, pos_label=1
                        )
                        balanced_acc = sklearn.metrics.balanced_accuracy_score(
                            true_explanation, pred_explanation
                        )
                        num_explained_data_with_acc += 1
            else:
                raise ValueError("Unknown dataset name: {}".format(self.dataset_name))
            entry = {
                "roc_auc_score": roc_auc_score,
                "recall": recall,
                "precision": precision,
                "f1_score": f1_score,
                "balanced_acc": balanced_acc,
            }
            scores.append(entry)
        accuracy_scores = pd.DataFrame.from_dict(list_to_dict(scores))
        accuracy_scores["num_explained_data_with_acc"] = num_explained_data_with_acc
        return accuracy_scores

    def _eval_fid(self, related_preds):
        if self.focus == "phenomenon":
            fidelity_scores = {
                "fidelity_acc+": fidelity_acc(related_preds),
                "fidelity_acc-": fidelity_acc_inv(related_preds),
                "fidelity_acc-_ext": fidelity_acc_inv_ext(related_preds),
                "fidelity_prob+": fidelity_prob(related_preds),
                "fidelity_prob-": fidelity_prob_inv(related_preds),
            }
        elif self.focus == "model":
            fidelity_scores = {
                "fidelity_gnn_acc+": fidelity_gnn_acc(related_preds),
                "fidelity_gnn_acc-": fidelity_gnn_acc_inv(related_preds),
                "fidelity_gnn_acc-_ext": fidelity_gnn_acc_inv_ext(related_preds),
                "fidelity_gnn_prob+": fidelity_gnn_prob(related_preds),
                "fidelity_gnn_prob-": fidelity_gnn_prob_inv(related_preds),
            }
        else:
            raise ValueError("Unknown focus: {}".format(self.focus))

        fidelity_scores = pd.DataFrame.from_dict(fidelity_scores)
        fidelity_scores["num_explained_data_fid"] = self.num_explained_data
        return fidelity_scores

    def _eval_act(self, related_preds):
        if self.focus == "phenomenon":
            accountability_scores = {
                "accountability": accountability(related_preds),
                "accountability_ext": accountability_ext(related_preds),
            }
        elif self.focus == "model":
            accountability_scores = {
                "accountability_gnn": accountability_gnn(related_preds),
                "accountability_gnn_ext": accountability_gnn_ext(related_preds),
            }
        else:
            raise ValueError("Unknown focus: {}".format(self.focus))
        accountability_scores = pd.DataFrame.from_dict(accountability_scores)
        accountability_scores["num_explained_data_act"] = self.num_explained_data
        return accountability_scores

    def eval(self, edge_masks, node_feat_masks):
        related_preds = eval("self.related_pred" + self.task)(
            edge_masks, node_feat_masks
        )
        if self.groundtruth:
            accuracy_scores = self._eval_acc(edge_masks)
            top_accuracy_scores = self._eval_top_acc(edge_masks)
        else:
            accuracy_scores, top_accuracy_scores = {}, {}
        fidelity_scores = self._eval_fid(related_preds)
        accountability_scores = self._eval_act(related_preds)
        return (
            top_accuracy_scores,
            accuracy_scores,
            fidelity_scores,
            accountability_scores,
        )

    def related_pred_graph(self, edge_masks, node_feat_masks):
        related_preds = []
        for i in range(len(self.list_explained_data)):
            data = self.dataset[self.list_explained_data[i]]
            data = data.to(self.device)
            data.batch = torch.zeros(data.x.shape[0], dtype=int, device=data.x.device)
            ori_prob_idx = self.model.get_prob(data).cpu().detach().numpy()[0]
            if node_feat_masks[0] is not None:
                if node_feat_masks[i].ndim == 0:
                    # if type of node feat mask is 'feature'
                    node_feat_mask = node_feat_masks[i].reshape(-1)
                else:
                    # if type of node feat mask is 'feature'
                    node_feat_mask = node_feat_masks[i]
                node_feat_mask = torch.Tensor(node_feat_mask).to(self.device)
                x_masked = data.x * node_feat_mask
                x_maskout = data.x * (1 - node_feat_mask)
            else:
                x_masked, x_maskout = data.x, data.x

            masked_data, maskout_data, masked_ext_data = (
                data.clone(),
                data.clone(),
                data.clone(),
            )
            masked_data.x, maskout_data.x, masked_ext_data.x = (
                x_masked,
                x_maskout,
                x_masked,
            )

            if (
                (edge_masks[i] is not None)
                and (hasattr(edge_masks[i], "__len__"))
                and (len(edge_masks[i]) > 0)
            ):
                edge_mask = torch.Tensor(edge_masks[i]).to(self.device)
                hard_edge_mask = (
                    torch.where(edge_mask > 0, 1, 0).to(self.device).float()
                )
                if self.mask_nature == "hard":
                    masked_data.edge_index = data.edge_index[:, edge_mask > 0].to(
                        self.device
                    )
                    masked_data.edge_attr = data.edge_attr[edge_mask > 0].to(
                        self.device
                    )
                    maskout_data.edge_index = data.edge_index[:, edge_mask <= 0].to(
                        self.device
                    )
                    maskout_data.edge_attr = data.edge_attr[edge_mask <= 0].to(
                        self.device
                    )
                    # Extended explanation: all edges (with edge attributions); important edges have weight 1; smaller weights on the unimportant edges
                    ext_edge_mask = extend_mask(hard_edge_mask, mu=self.mu)
                    masked_ext_data.edge_weight = ext_edge_mask
                elif self.mask_nature == "hard_full":
                    masked_data.edge_weight = hard_edge_mask
                    maskout_data.edge_weight = 1 - hard_edge_mask
                    # Extended explanation: same as "hard"
                    ext_edge_mask = extend_mask(hard_edge_mask, mu=self.mu)
                    masked_ext_data.edge_weight = ext_edge_mask
                elif self.mask_nature == "soft":
                    masked_data.edge_weight = edge_mask
                    maskout_data.edge_weight = 1 - edge_mask
                    # Extended explanation: add small weights tyo unimportant edges
                    ext_edge_mask = extend_mask(edge_mask, mu=self.mu)
                    masked_ext_data.edge_weight = ext_edge_mask
                else:
                    raise ValueError("Unknown mask nature: {}".format(self.mask_nature))

            masked_prob_idx = self.model.get_prob(masked_data).cpu().detach().numpy()[0]
            masked_ext_prob_idx = (
                self.model.get_prob(masked_ext_data).cpu().detach().numpy()[0]
            )
            maskout_prob_idx = (
                self.model.get_prob(maskout_data).cpu().detach().numpy()[0]
            )

            true_label = data.y.cpu().item()
            pred_label = np.argmax(ori_prob_idx)

            # assert true_label == pred_label, "The label predicted by the GCN does not match the true label."\
            related_preds.append(
                {
                    "explained_y_idx": self.list_explained_data[i],
                    "masked": masked_prob_idx,
                    "maskout": maskout_prob_idx,
                    "masked_extended": masked_ext_prob_idx,
                    "origin": ori_prob_idx,
                    "true_label": true_label,
                    "pred_label": pred_label,
                }
            )

        related_preds = list_to_dict(related_preds)
        return related_preds

    def _compute_graph(self, i):
        data = self.dataset[i].to(self.device)
        if self.focus == "phenomenon":
            target = data.y
        else:
            target = self.model(data=data).argmax(-1).item()
        start_time = time.time()
        edge_mask, node_feat_mask = self.explain_function(
            self.model, data, target, self.device, **self.explainer_params
        )
        end_time = time.time()
        duration_seconds = end_time - start_time
        return (
            edge_mask,
            node_feat_mask,
            duration_seconds,
        )

    def compute_mask(self):
        self.explain_function = eval("explain_" + self.explainer_name + self.task)
        print("Computing masks using " + self.explainer_name + " explainer.")
        if (self.save_dir is not None) and (
            Path(os.path.join(self.save_dir, self.save_name)).is_file()
        ):
            (   list_explained_data,
                edge_masks,
                node_feat_masks,
                computation_time,
            ) = self.load_mask()
        else:
            list_explained_data, edge_masks, node_feat_masks, computation_time = (
                [],
                [],
                [],
                [],
            )
            for i in range(len(self.dataset)):
                data = self.dataset[i].to(self.device)
                edge_mask, node_feat_mask, duration_seconds = eval(
                    "self._compute" + self.task
                )(i)
                if (
                    (edge_mask is not None)
                    and (hasattr(edge_mask, "__len__"))
                    and (len(edge_mask) > 0)
                ):
                    edge_masks.append(edge_mask)
                    node_feat_masks.append(node_feat_mask)
                    computation_time.append(duration_seconds)
                    list_explained_data.append(data.idx)
                self.list_explained_data = list_explained_data
            if (self.save_dir is not None) and self.save:
                self.save_mask(
                    list_explained_data, edge_masks, node_feat_masks, computation_time
                )
        return list_explained_data, edge_masks, node_feat_masks, computation_time
    

    def clean_mask(self, edge_masks, node_feat_masks):
        if edge_masks:
            if edge_masks[0] is not None:
                edge_masks = clean(edge_masks)
        if node_feat_masks:
            node_feat_masks = np.array(node_feat_masks)
            if node_feat_masks[0] is not None:
                node_feat_masks = clean(node_feat_masks)
        return edge_masks, node_feat_masks

 
    def save_mask(self, list_explained_data, edge_masks, node_feat_masks, computation_time):
        assert self.save_dir is not None, "save_dir is None. Masks are not saved"
        save_path = os.path.join(self.save_dir, self.save_name)
        with open(save_path, "wb") as f:
            pickle.dump([list_explained_data, edge_masks, node_feat_masks, computation_time], f)

    def load_mask(self):
        assert self.save_dir is not None, "save_dir is None. No mask to be loaded"
        save_path = os.path.join(self.save_dir, self.save_name)
        with open(save_path, "rb") as f:
            w_list = pickle.load(f)
        list_explained_data, edge_masks, node_feat_masks, computation_time = tuple(w_list)
        self.list_explained_data = list_explained_data
        return list_explained_data, edge_masks, node_feat_masks, computation_time


def get_mask_dir_path(args):
    mask_save_name = "mask_{}_{}_{}_{}_{}.pkl".format(
        args.dataset_name,
        args.model_name,
        args.explainer_name,
        args.focus,
        args.seed,
    )
    return mask_save_name


def explain_main(dataset, model, device, args):
    args.dataset = dataset
    mask_save_name = get_mask_dir_path(args)
    explainer = Explain(
        model=model,
        dataset=dataset,
        device=device,
        explainer_params=vars(args),
        save_dir=None
        if args.mask_save_dir == "None"
        else os.path.join(args.mask_save_dir, args.dataset_name, args.explainer_name),
        save_name=mask_save_name,
    )

    (   list_explained_data,
        edge_masks,
        node_feat_masks,
        computation_time,
    ) = explainer.compute_mask()
    edge_masks, node_feat_masks = explainer.clean_mask(edge_masks, node_feat_masks)
    return list_explained_data, edge_masks, node_feat_masks, computation_time

    