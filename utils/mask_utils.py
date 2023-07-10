### Utils to transform edge masks
import numpy as np

def topk_edges_unique(edge_mask, edge_index, num_top_edges):
    """Return the indices of the top-k edges in the mask.

    Args:
        edge_mask (Tensor): edge mask of shape (num_edges,).
        edge_index (Tensor): edge index tensor of shape (2, num_edges)
        num_top_edges (int): number of top edges to be kept
    """
    indices = (-edge_mask).argsort()
    top = np.array([], dtype="int")
    i = 0
    list_edges = np.sort(edge_index.cpu().T, axis=1)
    while len(top) < num_top_edges:
        subset = indices[num_top_edges * i : num_top_edges * (i + 1)]
        topk_edges = list_edges[subset]
        u, idx = np.unique(topk_edges, return_index=True, axis=0)
        top = np.concatenate([top, subset[idx]])
        i += 1
    return top[:num_top_edges]


def clean(masks):
    """Clean masks by removing NaN, inf and too small values and normalizing"""
    for i in range(len(masks)):
        if (
            (masks[i] is not None)
            and (hasattr(masks[i], "__len__"))
            and (len(masks[i]) > 0)
        ):
            masks[i] = np.nan_to_num(
                masks[i], copy=True, nan=0.0, posinf=10, neginf=-10
            )
    return masks

def transform_edge_masks(edge_masks, strategy="remove", threshold=0.1):
    if strategy == "remove":
        thresh_edge_masks = []
        for edge_mask in edge_masks:
            maskout = remove(edge_mask, threshold=threshold)
            thresh_edge_masks.append(maskout)
    elif strategy == "keep":
        thresh_edge_masks = []
        for edge_mask in edge_masks:
            masked = keep(edge_mask, threshold=threshold)
            thresh_edge_masks.append(masked)
    else:
        raise ValueError("Invalid strategy")
    return thresh_edge_masks
                                     

def keep(mask, threshold=0.1):
    mask_len = len(mask)
    split_point = int((1 - threshold) * mask_len)
    unimportant_indices = (-mask).argsort()[split_point:]
    mask[unimportant_indices] = 0
    return mask


def remove(mask, threshold=0.1):
    mask_len = len(mask)
    split_point = int((1 - threshold) * mask_len)
    important_indices = (-mask).argsort()[:split_point]
    mask[important_indices] = 0
    return mask