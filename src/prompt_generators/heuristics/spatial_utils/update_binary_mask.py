import torch 
import gc 

def update_binary_mask(coords: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
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
        indices = tuple(valid_coords.to(torch.int32).T)  
        # Convert valid coordinates to indices using int32. should be sufficient
        #for the vast majority of voxel counts. unless maybe we started working with images of size 10000 x 10000 x 10000 etc.
        #int32 should be sufficient for indexing, as it can represent indices in the range of the mask dimensions.
        mask[indices] = 0  # Set valid positions to 0
    
    #Obsessively emptying cuda memory where not required to prevent crashes on huge images.
    del valid_coords
    del valid_mask 
    del indices
    gc.collect()
    torch.cuda.empty_cache()

    return mask.to(dtype=torch.uint8) 

if __name__ == "__main__":
    # Large 3D mask: 500 x 500 x 300 #Just checking the memory usage
    mask = torch.ones((500, 500, 700), dtype=torch.uint8, device='cuda')
    # 10 random coordinates to set to zero
    coords = torch.randint(0, 500, (10, 3))
    coords[:, 2] = torch.randint(0, 700, (10,))  # Set the 3rd dimension separately for correct range

    print("Coordinates to zero out:\n", coords)
    updated_mask = update_binary_mask(coords, mask)
    # Print the values at those coordinates to confirm they are zero
    for c in coords:
        print(f"Mask at {tuple(c.tolist())}: {updated_mask[tuple(c.tolist())].item()}")