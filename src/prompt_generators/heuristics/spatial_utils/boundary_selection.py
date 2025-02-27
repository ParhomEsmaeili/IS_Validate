import torch
import torch.nn as nn
import torch.nn.functional as F 
import numpy as np
# import nibabel as nib
import os
import numpy as np
# from scipy.ndimage import label

def get_boundary(seg, kernel_size):
    raise NotImplementedError('Implement a check for how this works/what boundary this is extracting.')
    pad_size = int((kernel_size - 1) / 2)
    m_xy = nn.AvgPool3d((kernel_size, kernel_size, 1), stride=1, padding=(pad_size, pad_size, 0)).cuda()
    output_xy = m_xy(seg)
    edge_xy = abs(seg - output_xy)
    # edge = edge_xy[0, :]
    edge_locations = torch.multiply(edge_xy, seg)
    edge_locations[edge_locations > 0] = 1
    edge_mask = edge_locations.squeeze(0)

    return edge_mask

def find_boundary_map(seg, boundary_kernel=3, margin_kernel=7):
    raise NotImplementedError('Check what boundary this is extracting')
    boundary = get_boundary(seg, kernel_size=boundary_kernel).unsqueeze(0)
    margin = get_boundary(seg, kernel_size=margin_kernel).unsqueeze(0) - boundary
    content = seg - margin - boundary
    return boundary.squeeze(0), margin.squeeze(0), content.squeeze(0)

def extract_interior_boundary(mask: torch.Tensor, kernel_size: int = 3) -> torch.Tensor:
    """
    Extracts a one-voxel-thick interior boundary of a binary mask in 2D or 3D.

    Args:
        mask (torch.Tensor): A 2D or 3D binary mask of shape (H, W) or (H, W, D).
        kernel_size (int): Structuring element size for erosion (must be odd).

    Returns:
        torch.Tensor: Binary mask of the interior boundary.
    """
    assert mask.ndim in (2, 3), "Only 2D and 3D inputs are supported"
    assert kernel_size % 2 == 1, "Kernel size should be odd"

    # Ensure binary format
    mask = (mask > 0).float()

    # Perform true erosion: Invert mask -> max pool -> invert back
    padding = kernel_size // 2
    inverted_mask = 1 - mask  # Invert the binary mask

    if mask.ndim == 2:
        eroded_mask = 1 - F.max_pool2d(
            inverted_mask.unsqueeze(0).unsqueeze(0),  # Add batch & channel dims
            kernel_size, stride=1, padding=padding
        ).squeeze(0).squeeze(0)
    else:
        eroded_mask = 1 - F.max_pool3d(
            inverted_mask.unsqueeze(0).unsqueeze(0),  # Add batch & channel dims
            kernel_size, stride=1, padding=padding
        ).squeeze(0).squeeze(0)

    # Compute the interior boundary
    boundary = mask - eroded_mask

    return (boundary > 0).float()




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
    ], dtype=torch.float64)

    cavity_boundaries = extract_interior_boundary(binary_mask_2d)
    print(cavity_boundaries.numpy())  # Should show only the internal cavity boundaries

    binary_mask_3d = torch.zeros((7, 7, 7))
    binary_mask_3d[1:6, 1:6, 1:6] = 1  # Solid cube
    binary_mask_3d[3, 2:5, 2:5] = 0  # Internal cavity (touching the boundary and a central point)
    binary_mask_3d[3,3,3] = 1 # placing back the central point
    boundary_3d = extract_interior_boundary(binary_mask_3d)
    print(boundary_3d.nonzero())
    print(boundary_3d[3, :])
    
# if __name__ == '__main__':
    # seg_data = nib.load('./example_label_cropped.nii.gz')
    # seg = seg_data.get_fdata()
    # seg = torch.from_numpy(seg).float().cuda().unsqueeze(0).unsqueeze(0)

    # boundary, margin, content = find_boundary_map(seg)

    # points_dict = get_points_location(seg)

    # boundary = boundary.squeeze(0).squeeze(0).cpu().detach().numpy()
    # margin = margin.squeeze(0).squeeze(0).cpu().detach().numpy()
    # content = content.squeeze(0).squeeze(0).cpu().detach().numpy()

    # nib.save(nib.Nifti1Image(boundary, seg_data.affine, seg_data.header), os.path.join(os.getcwd(), 'boundary.nii.gz'))
    # nib.save(nib.Nifti1Image(margin, seg_data.affine, seg_data.header), os.path.join(os.getcwd(), 'margin.nii.gz'))
    # nib.save(nib.Nifti1Image(content, seg_data.affine, seg_data.header), os.path.join(os.getcwd(), 'content.nii.gz'))
    # print('points location: {}'.format(points_dict))

    # seg_data = nib.load('./example_label.nii.gz')
    # seg = seg_data.get_fdata()
    # seg_crop = seg[280:280+200, 200:200+200, :]
    # nib.save(nib.Nifti1Image(seg_crop, seg_data.affine, seg_data.header), os.path.join(os.getcwd(), 'example_label_cropped.nii.gz'))
    #
    # seg_data = nib.load('./example_image.nii.gz')
    # seg = seg_data.get_fdata()
    # seg_crop = seg[280:280+200, 200:200+200, :]
    # nib.save(nib.Nifti1Image(seg_crop, seg_data.affine, seg_data.header), os.path.join(os.getcwd(), 'example_image_cropped.nii.gz'))