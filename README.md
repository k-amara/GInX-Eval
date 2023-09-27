# GInEx-Eval

*GInEx-Eval is an evaluation procedure that estimate edge importance in graph neural networks.*

This directory contains the code needed to implement **GInX-Eval**, Graph In-distribution eXplanation Evaluation, a procedure to evaluate how informative edges are to the GNN model.

[Paper]()

## GInX-Eval: Graph In-distribution eXplanation Evaluation

### tl;dr

- Many explainability methods for graph neural networks estimate the edge importance to the model prediction.
- Using a retraining strategy, GInX-Eval overcomes the pitfalls of faithfulness.
- the GInX score measures how informative removed edges are for the model.
- the EdgeRank score evaluates if explanatory edges are correctly ordered by their importance.

### GNN Explainability Methods


| Non-generative Explainer | Paper                                                                               |
| :----------------------- | :---------------------------------------------------------------------------------- |
| Occlusion                | Visualizing and understanding convolutional networks                                |
| SA                       | Explainability Techniques for Graph Convolutional Networks.                         |
| Grad-CAM                 | Explainability Methods for Graph Convolutional Neural Networks.                     |
| Integrated Gradients     | Axiomatic Attribution for Deep Networks                                             |
| GNNExplainer             | GNNExplainer: Generating Explanations for Graph Neural Networks                     |
| SubgraphX                | On Explainability of Graph Neural Networks via Subgraph Exploration                 |
| PGM-Explainer            | PGM-Explainer: Probabilistic Graphical Model Explanations for Graph Neural Networks |

| Generative Explainer | Paper                                                                             |
| :------------------- | :-------------------------------------------------------------------------------- |
| PGExplainer          | Parameterized Explainer for Graph Neural Network                                  |
| RCExplainer          | Reinforced Causal Explainer for Graph Neural Networks                             |
| GSAT                 | Interpretable and Generalizable Graph Learning via Stochastic Attention Mechanism |
| GraphCFE             | CLEAR: Generative Counterfactual Explanations on Graphs                           |
| DiffExplainer        | D4Explainer (not published)                                                       |

### Datasets

| Dataset        |       Name       | Description                                           |
| -------------- | :--------------: | ----------------------------------------------------- |
| BA-2motifs     |   `ba_2motifs`   | Random BA graph with house or 5-node cycle motifs.    |
| BA-HouseGrid   | `ba_house_grid`  | Random BA graph with house or grid motifs.            |
| MUTAG          |     `mutag`      | Mutagenecity Predicting the mutagenicity of molecules |
| Benzene        |     `benzene`    | Molecular dataset with or without benzene fragment    |
| BBBP           |      `bbbp`      | Blood-brain barrier penetration                       |
| MNISTbin       |     `mnist_bin`  | MNIST graph dataset with digits 0 and 1               |


## Installation

**Requirements**

- CPU or NVIDIA GPU, Linux, Python 3.7
- PyTorch >= 1.5.0, other packages

1. Load every additional packages:

```
pip install -r requirements.txt
```

2. Manual installation

Pytorch Geometric. [Official Download](https://pytorch-geometric.readthedocs.io/en/latest/notes/installation.html).

```
CUDA=cu111
TORCH=1.9.1
pip install torch-scatter -f https://pytorch-geometric.com/whl/torch-${TORCH}+${CUDA}.html
pip install torch-sparse -f https://pytorch-geometric.com/whl/torch-${TORCH}+${CUDA}.html
pip install torch-cluster -f https://pytorch-geometric.com/whl/torch-${TORCH}+${CUDA}.html
pip install torch-spline-conv -f https://pytorch-geometric.com/whl/torch-${TORCH}+${CUDA}.html
pip install torch-geometric==2.0.3
```

Other packages:

```
pip install tqdm matplotlib argparse json jupyterlab notebook pgmpy captum
```

## Getting Started: How to evaluate a GNN explainability method using GInX-Eval?

### Run code

#### GInX-Eval of Hard Explanations

A hard explanation or explanatory subgraph is a subgraph that keeps only the important edges, i.e. the positive values in the sparse explanatory edge mask. 
The hard explanation has only the nodes connected to the remaining important edges and is usually smaller than the input graphs.

```bash
python src/main_sub.py --dataset_name [dataset-name] --model_name [gnn-model] --explainer_name [explainer-name]
```

#### GInX-Eval of Soft Explanations

A soft explanation or explanatory weighted graph is the input graph weighted by the explanatory sparse explanatory mask. 
The structure of the input graph is kept in the soft explanation, the edge index and nodes are similar to the input graph but unimportant edges receive zero weights. 


```bash
python src/main.py --dataset_name [dataset-name] --model_name [gnn-model] --explainer_name [explainer-name]
```

## Parameters

- dataset-name:
  - synthetic: ba_2motifs, ba_house_grid
  - real-world: mutag, benzene, bbbp, mnist_bin
- gnn-model: gat, gin
- explainer-name: random, truth, inverse, sa, ig, gradcam, occlusion, basic_gnnexplainer, gnnexplainer, subgraphx, pgmexplainer, pgexplainer, rcexplainer, gsat, graphcfe, diffexplainer


## Citation
If you are using GInX-Eval code you may cite: (Anonymous)
```
```
For any questions about this code please file an github [issue](https://github.com/) and tag github handles sarahooker, doomie. We welcome pull requests which add additional GNN explainability methods to be evaluated or improvements to the code.
