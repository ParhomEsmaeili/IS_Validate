import torch
import numpy as np
import warnings
# from monai.auto3dseg.utils import get_label_ccp 
from typing import Any
from monai.data import MetaTensor 
from monai.transforms import ToCupy
from monai.utils import min_version, optional_import
measure_np, has_measure = optional_import("skimage.measure", "0.14.2", min_version)
cp, has_cp = optional_import("cupy")

import torch

def find(parents, x):
    while parents[x] != x:
        parents[x] = parents[parents[x]]  # Path compression
        x = parents[x]
    return x

def union(parents, ranks, x, y):
    root_x = find(parents, x)
    root_y = find(parents, y)
    if root_x != root_y:
        if ranks[root_x] > ranks[root_y]:
            parents[root_y] = root_x
        elif ranks[root_x] < ranks[root_y]:
            parents[root_x] = root_y
        else:
            parents[root_y] = root_x
            ranks[root_x] += 1

def extract_connected_components(binary_mask):
    if not isinstance(binary_mask, torch.Tensor):
        raise TypeError("Input must be a torch.Tensor.")
    if binary_mask.dim() not in (2, 3):
        raise ValueError("Only 2D and 3D masks are supported.")
    
    device = binary_mask.device
    shape = binary_mask.shape
    num_elements = binary_mask.numel()
    flat_mask = binary_mask.view(-1)
    
    # Initialize Union-Find structures
    parents = torch.arange(num_elements, device=device)
    ranks = torch.zeros(num_elements, dtype=torch.int64, device=device)
    
    # Define neighbor offsets for 2D (8-connectivity) and 3D (26-connectivity)
    if binary_mask.dim() == 2:
        offsets = [-shape[1] - 1, -shape[1], -shape[1] + 1, -1, 1, shape[1] - 1, shape[1], shape[1] + 1]
    else:
        offsets = []
        for dz in [-1, 0, 1]:
            for dy in [-1, 0, 1]:
                for dx in [-1, 0, 1]:
                    if dz == 0 and dy == 0 and dx == 0:
                        continue
                    offsets.append(dz * shape[1] * shape[2] + dy * shape[2] + dx)
    
    # Union neighboring connected components
    indices = torch.nonzero(flat_mask, as_tuple=False).flatten()
    for idx in indices:
        for offset in offsets:
            neighbor = idx + offset
            if 0 <= neighbor < num_elements and flat_mask[neighbor]:
                union(parents, ranks, idx.item(), neighbor.item())
    
    # Find unique component labels
    unique_labels = torch.unique(torch.tensor([find(parents, idx.item()) for idx in indices], device=device))
    
    # Extract component masks
    component_masks = []
    for label in unique_labels:
        mask = torch.zeros_like(flat_mask, dtype=torch.uint8, device=device)
        for idx in indices:
            if find(parents, idx.item()) == label:
                mask[idx] = 1
        component_masks.append(mask.view(shape))
    
    return component_masks

def get_label_ccp(mask_index: MetaTensor, use_gpu: bool = True) -> tuple[list[Any], int]:
    """
    Borrowed from monai.auto3dseg.utils with some slight modifications s.t. it provides us the maps and not the bbox.

    Find all connected components. Backend can be cuPy/cuCIM or Numpy
    depending on the hardware.

    Args:
        mask_index: a binary mask.
        use_gpu: a switch to use GPU/CUDA or not. If GPU is unavailable, CPU will be used
            regardless of this setting.

    returns: List of binary masks for each component, ncomponents = number of connected components.
    """
    skimage, has_cucim = optional_import("cucim.skimage")
    # shape_list = []
    if mask_index.device.type == "cuda" and has_cp and has_cucim and use_gpu:
        mask_cupy = ToCupy()(mask_index.short())
        labeled = skimage.measure.label(mask_cupy)
        vals = cp.unique(labeled[cp.nonzero(labeled)])
        # for ncomp in vals:
        #     comp_idx = cp.argwhere(labeled == ncomp)
        #     comp_idx_min = cp.min(comp_idx, axis=0).tolist()
        #     comp_idx_max = cp.max(comp_idx, axis=0).tolist()
        #     bbox_shape = [comp_idx_max[i] - comp_idx_min[i] + 1 for i in range(len(comp_idx_max))]
        #     shape_list.append(bbox_shape)
        ncomponents = len(vals)

        # del mask_cupy, labeled, vals, comp_idx, ncomp
        cp.get_default_memory_pool().free_all_blocks()

    elif has_measure:
        labeled, ncomponents = measure_np.label(mask_index.data.cpu().numpy(), background=0, return_num=True)
        
        # for ncomp in range(1, ncomponents + 1):
        #     comp_idx = np.argwhere(labeled == ncomp)
        #     comp_idx_min = np.min(comp_idx, axis=0).tolist()
        #     comp_idx_max = np.max(comp_idx, axis=0).tolist()
        #     bbox_shape = [comp_idx_max[i] - comp_idx_min[i] + 1 for i in range(len(comp_idx_max))]
        #     shape_list.append(bbox_shape)
    else:
        raise RuntimeError("Cannot find one of the following required dependencies: {cuPy+cuCIM} or {scikit-image}")

    torch_tensor = torch.from_numpy(labeled)
    masks = []
    for ncomp in range(1, ncomponents+1):
        masks.append(torch.where(torch_tensor == ncomp, 1, 0).to(dtype=torch.int64))

    return masks, ncomponents

if __name__ == '__main__':

    binary_mask_2d = torch.tensor([
        [0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 1, 1, 1, 1, 1, 1, 1, 0],
        [0, 1, 1, 1, 1, 1, 1, 1, 0],
        [0, 1, 1, 0, 0, 0, 1, 1, 0],
        [0, 1, 1, 0, 1, 0, 1, 1, 0],
        [0, 1, 1, 0, 0, 0, 1, 1, 0],
        [0, 1, 1, 1, 1, 1, 1, 1, 0],
        [0, 1, 1, 1, 1, 1, 1, 1, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0],
    ], dtype=torch.int64)

    components2d = extract_connected_components(binary_mask_2d)
    for i, comp in enumerate(components2d):
        print(f"Component {i}:\n{comp}")
    binary_mask_3d = torch.zeros((7, 7, 7))
    binary_mask_3d[1:6, 1:6, 1:6] = 1  # Solid cube
    binary_mask_3d[2:5, 2:5, 2:5] = 0  # Internal cavity (touching the boundary and a central point)
    binary_mask_3d[3,3,3] = 1 # placing back the central point
    components3d = extract_connected_components(binary_mask_3d)
    for i, comp in enumerate(components3d):
        print(f"3D component{i}:\n{comp}")


    # try :
    print(get_label_ccp(binary_mask_2d))
    print(get_label_ccp(binary_mask_3d))
    # except:
    #     print('missing import')


    print(get_label_ccp(torch.zeros(10,10)))