"""
Fast spatial extent checks for binary masks.

These functions check whether a binary mask has a minimum spatial extent
(i.e., contiguous foreground voxels spanning at least `min_length` coordinates
in each dimension). They are agnostic to the prompt type (bbox, lasso, etc.)
and serve as fast pre-checks or post-component-extraction filters.
"""
import torch
from typing import Optional, Tuple


def has_contiguous_sequence_vectorised(unique_coords, min_length=2):
    num_dims, max_coords = unique_coords.shape

    if max_coords < min_length:
        return False

    valid_mask = ~torch.isnan(unique_coords)
    valid_counts = valid_mask.sum(dim=1)
    if (valid_counts < min_length).any():
        return False

    coords_with_nan = torch.where(valid_mask, unique_coords.float(), torch.nan)

    sorted_coords, _ = torch.sort(coords_with_nan, dim=1)

    diffs = torch.diff(sorted_coords, dim=1)

    is_contiguous = (diffs == 1)

    window_size = min_length - 1
    diff_len = diffs.size(1)

    if diff_len < window_size:
        return False

    num_windows = diff_len - window_size + 1

    window_indices = torch.arange(num_windows, device=unique_coords.device).unsqueeze(1) + \
                     torch.arange(window_size, device=unique_coords.device).unsqueeze(0)

    is_contiguous_expanded = is_contiguous.unsqueeze(1).expand(num_dims, num_windows, diff_len)

    window_indices_expanded = window_indices.unsqueeze(0).expand(num_dims, -1, -1)

    windows = is_contiguous_expanded.gather(2, window_indices_expanded)

    window_sums = windows.sum(dim=-1)

    has_contiguous = ((window_sums == window_size).any(dim=1)).all()
    if has_contiguous:
        return True
    else:
        return False


def has_min_spatial_extent_2d(mask_2d: torch.Tensor, min_length: int = 2) -> bool:
    """
    Check whether a 2D binary mask has a minimum spatial extent of `min_length`
    contiguous coordinates in both the x and y dimensions.

    A mask with a single voxel (or a thin line spanning only one dimension) will
    return False when min_length >= 2.

    Args:
        mask_2d: 2D binary mask (H, W)
        min_length: Minimum number of contiguous coordinates required per dimension.

    Returns:
        True if the mask has the minimum spatial extent in all dimensions.
    """
    foreground = torch.nonzero(mask_2d, as_tuple=False)

    if foreground.shape[0] < min_length:
        return False

    unique_x = torch.unique(foreground[:, 0])
    unique_y = torch.unique(foreground[:, 1])

    max_len = max(len(unique_x), len(unique_y))

    if len(unique_x) < max_len:
        padding_x = torch.full((max_len - len(unique_x),), torch.nan,
                               device=unique_x.device)
        unique_x = torch.cat([unique_x, padding_x])

    if len(unique_y) < max_len:
        padding_y = torch.full((max_len - len(unique_y),), torch.nan,
                                 device=unique_y.device)
        unique_y = torch.cat([unique_y, padding_y])

    unique_coords = torch.stack([unique_x, unique_y])

    return has_contiguous_sequence_vectorised(unique_coords, min_length)


def has_min_spatial_extent_3d(mask_3d: torch.Tensor, min_length: int = 2) -> bool:
    """
    Check whether a 3D binary mask has a minimum spatial extent of `min_length`
    contiguous coordinates in all three dimensions.

    Args:
        mask_3d: 3D binary mask (D, H, W)
        min_length: Minimum number of contiguous coordinates required per dimension.

    Returns:
        True if the mask has the minimum spatial extent in all dimensions.
    """
    foreground = torch.nonzero(mask_3d, as_tuple=False)

    if foreground.shape[0] < min_length:
        return False

    unique_x = torch.unique(foreground[:, 0])
    unique_y = torch.unique(foreground[:, 1])
    unique_z = torch.unique(foreground[:, 2])

    max_len = max(len(unique_x), len(unique_y), len(unique_z))

    if len(unique_x) < max_len:
        padding_x = torch.full((max_len - len(unique_x),), torch.nan, device=unique_x.device)
        unique_x = torch.cat([unique_x, padding_x])

    if len(unique_y) < max_len:
        padding_y = torch.full((max_len - len(unique_y),), torch.nan, device=unique_y.device)
        unique_y = torch.cat([unique_y, padding_y])

    if len(unique_z) < max_len:
        padding_z = torch.full((max_len - len(unique_z),), torch.nan, device=unique_z.device)
        unique_z = torch.cat([unique_z, padding_z])

    unique_coords = torch.stack([unique_x, unique_y, unique_z])

    return has_contiguous_sequence_vectorised(unique_coords, min_length)
