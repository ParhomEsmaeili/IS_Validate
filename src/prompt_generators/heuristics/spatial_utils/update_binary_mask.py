import torch 
# import gc 
# import time
# from cProfile import Profile
# from pstats import Stats, SortKey
def update_binary_mask_freeform(coords: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """
    Modifies an existing binary mask by setting specified coordinates to zero.
    
    Args:
        coords (torch.Tensor): An (N x D) tensor containing coordinates, where N is the number of coordinates
                              and D is the number of spatial dimensions of the mask.
        mask (torch.Tensor): An existing binary mask.
    
    Returns:
        torch.Tensor: The modified binary mask with zeros at `coords` locations.
    
    Raises:
        ValueError: If the number of dimensions in `coords` does not match the mask's dimensions.
    """
    # Early return if coords is empty
    if coords.numel() == 0:
        return mask
    
    if coords.shape[1] != mask.dim():
        raise ValueError(f"Dimension mismatch: coords has {coords.shape[1]} spatial dimensions, but mask has {mask.dim()} dimensions.")
    
    # Ensure mask and coords are on the same device
    device = mask.device 
    coords = coords.to(device)
    
    # Apply bounds check for each coordinate, dimension by dimension
    valid_mask = torch.all((coords >= 0) & (coords < torch.tensor(mask.shape, device=device)), dim=1)
    
    # Filter valid coordinates
    valid_coords = coords[valid_mask].to(device)  # Move valid_coords to the same device as mask

    if valid_coords.numel() > 0:  # Ensure there are valid coordinates
        # indices = tuple(valid_coords.to(torch.int32).T)  
        # Convert valid coordinates to indices using int32. should be sufficient
        #for the vast majority of voxel counts. unless maybe we started working with images of size 10000 x 10000 x 10000 etc.
        #int32 should be sufficient for indexing, as it can represent indices in the range of the mask dimensions.
        
        #We move the index handling in-line to prevent unnecessary variable assignments.
        mask[tuple(valid_coords.to(torch.int32).T)] = False  # Set valid positions to False
    
    # Handling VRAM. 
    valid_coords = valid_coords.cpu()
    coords = coords.cpu()
    valid_mask = valid_mask.cpu()
    torch.cuda.empty_cache() #Not sure if this is deallocating anything? #TODO diagnose this when we have more time.
    return mask.to(dtype=torch.bool) 

def update_binary_mask_partition(prompt: list[torch.Tensor], mask: torch.Tensor) -> torch.Tensor:
    '''
    Modifies an existing binary mask by setting specified partition region to zero.
    
    For bbox prompts, each tensor in the list has shape (1, 6) with format
    [min_x, min_y, min_z, max_x, max_y, max_z] in voxel coordinates.
    The entire enclosed region is zeroed out.
    
    Args:
        prompt (list[torch.Tensor]): A list of bbox tensors, each of shape (1, 6).
        mask (torch.Tensor): An existing binary mask.
    
    Returns:
        torch.Tensor: The modified binary mask with zeros at the specified partition locations.

    '''
    if not prompt:
        return mask

    device = mask.device
    ndim = mask.dim()

    for bbox_tensor in prompt:
        bbox = bbox_tensor.to(device)
        if bbox.dim() == 2 and bbox.shape[0] == 1:
            bbox = bbox[0]
        min_coords = bbox[:ndim].long()
        max_coords = bbox[ndim:2*ndim].long()
        min_coords = torch.clamp(min_coords, min=0)
        max_coords = torch.clamp(max_coords, max=torch.tensor(mask.shape, device=device) - 1)

        slices = tuple(slice(min_c.item(), max_c.item() + 1) for min_c, max_c in zip(min_coords, max_coords))
        mask[slices] = False

    return mask.to(dtype=torch.bool) 

if __name__ == "__main__":
    # Large 3D mask: 500 x 500 x 300 #Just checking the memory usage
    mask = torch.ones((500, 500, 700), dtype=torch.uint8, device='cuda')
    # 10 random coordinates to set to zero
    coords = torch.randint(0, 500, (10, 3))
    coords[:, 2] = torch.randint(0, 700, (10,))  # Set the 3rd dimension separately for correct range
    print("Coordinates to zero out:\n", coords)
    updated_mask = update_binary_mask_freeform(coords, mask)
    # Print the values at those coordinates to confirm they are zero
    for c in coords:
        print(f"Mask at {tuple(c.tolist())}: {updated_mask[tuple(c.tolist())].item()}")