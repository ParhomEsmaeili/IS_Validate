import torch
import unittest
import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from src.prompt_generators.heuristics.spatial_utils.component_extraction import get_label_ccp, extract_connected_components

def test_connected_components_2d():
    test_cases = [
        # Single isolated pixel
        (torch.tensor([
            [0, 0, 0],
            [0, 1, 0],
            [0, 0, 0]
        ], dtype=torch.uint8), 1),
        
        # Single large component
        (torch.tensor([
            [1, 1, 1],
            [1, 1, 1],
            [1, 1, 1]
        ], dtype=torch.uint8), 1),
        
        # Two separate components
        (torch.tensor([
            [1, 0, 1],
            [0, 0, 0],
            [1, 0, 1]
        ], dtype=torch.uint8), 4),
        
        # Cavity inside a component
        (torch.tensor([
            [1, 1, 1, 1],
            [1, 0, 0, 1],
            [1, 0, 0, 1],
            [1, 1, 1, 1]
        ], dtype=torch.uint8), 1),
        
        # Complex shape with multiple components
        (torch.tensor([
            [1, 0, 1, 0, 1],
            [0, 1, 0, 1, 0],
            [1, 0, 1, 0, 1]
        ], dtype=torch.uint8), 5),
        
        # Non-convex structure
        (torch.tensor([
            [1, 1, 0, 1, 1],
            [1, 0, 1, 0, 1],
            [0, 1, 1, 1, 0]
        ], dtype=torch.uint8), 3),
        
        # Thin bridge connecting two parts
        (torch.tensor([
            [1, 1, 0, 1, 1],
            [0, 1, 1, 1, 0],
            [1, 1, 0, 1, 1]
        ], dtype=torch.uint8), 1)
    ]
    
    for binary_mask, expected_count in test_cases:
        components_custom = extract_connected_components(binary_mask)
        components_package, ncomps = get_label_ccp(binary_mask)
        # assert len(components_custom) == expected_count, f"Expected {expected_count} components, got {len(components_custom)}"
        # assert len(components_package) == expected_count, f"Expected {expected_count} components, got {len(components_package)}"
        assert len(components_custom) == len(components_package), f"Custom had {len(components_custom)} while package had {len(components_package)}"
        
        assert all([torch.all(components_custom[i] == components_package[i]) for i in range(len(components_custom))])
    
# def test_connected_components_3d():
#     test_cases = [
#         # Single solid cube
#         (torch.ones((3, 3, 3), dtype=torch.uint8), 1),
        
#         # Two separate cubes
#         (lambda: (lambda mask: (mask[0, :, :] == 1, mask[2, :, :] == 1, mask))[0](torch.zeros((3, 3, 3), dtype=torch.uint8)), 2),
        
#         # Cube with an internal cavity
#         (lambda: (lambda mask: (mask[1, 1, 1] == 0, mask[0:3, 0:3, 0:3] == 1, mask))[1](torch.ones((3, 3, 3), dtype=torch.uint8)), 1),
        
#         # Complex multi-component structure
#         (lambda: (lambda mask: (mask[0, 0, 0] == 1, mask[2, 2, 2] == 1, mask))[1](torch.zeros((3, 3, 3), dtype=torch.uint8)), 2),
        
#         # Hollow cube with a central pillar
#         (lambda: (lambda mask: (mask[1:2, 1:2, 1:2] == 0, mask[0:3, 0:3, 0:3] == 1, mask))[1](torch.ones((3, 3, 3), dtype=torch.uint8)), 1),
        
#         # Thin connection in 3D
#         (lambda: (lambda mask: (mask[0:3, 0, 0] == 1, mask[0:3, 2, 2] == 1, mask[1, 1, 1] == 1, mask))[1](torch.zeros((3, 3, 3), dtype=torch.uint8)), 1)
#     ]
    
#     for binary_mask, expected_count in test_cases:
#         if callable(binary_mask):
#             binary_mask = binary_mask()
#         components_custom = extract_connected_components(binary_mask)
#         components_package, ncomps = get_label_ccp(binary_mask)
#         # assert len(components_custom) == expected_count, f"Expected {expected_count} components, got {len(components_custom)}"
#         # assert len(components_package) == expected_count, f"Expected {expected_count} components, got {len(components_package)}"
#         assert len(components_custom) == len(components_package), f"Custom had {len(components_custom)} while package had {len(components_package)}"
#         assert all([torch.all(components_custom[i] == components_package[i]) for i in range(len(components_custom))])
    
class TestConnectedComponents(unittest.TestCase):
    def test_2d_case(self):
        test_connected_components_2d()
    
    # def test_3d_case(self):
    #     test_connected_components_3d()
    
if __name__ == "__main__":
    unittest.main()
