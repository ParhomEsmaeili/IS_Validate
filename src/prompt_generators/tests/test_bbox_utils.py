"""
Unit tests for bounding box generation utilities.

Tests cover:
- bbox_validation.py: Fast pre-check functions for 2D/3D bbox generation
- bbox.py: Main bbox generation, jitter, and validation functions
- component_extraction.py: Connected component extraction and selection
"""

import pytest
import torch
import numpy as np
from scipy.stats import chi2
import sys
import os

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))

from src.prompt_generators.heuristics.heuristic_prompt_utils.bbox_utils.bbox_validation import (
    has_contiguous_sequence_vectorised,
    can_generate_bbox_from_slice_fast,
    can_generate_bbox_from_volume_fast,
    check_bbox_validity
)
from src.prompt_generators.heuristics.heuristic_prompt_utils.bbox import (
    bbox_extrema,
    bbox_from_binary_mask,
    extract_sampling_region,
    select_component,
)

from src.prompt_generators.heuristics.heuristic_prompt_utils.bbox_utils.bbox_augmentations import (
    generate_jitter,
    jitter_bbox,
    apply_jitter,
)

from src.prompt_generators.heuristics.spatial_utils.component_extraction import (
    convert_to_numpy,
    validate_connectivity,
    extract_connected_components,
    two_d_components_generation,
    three_d_components_generation,
    generate_components_from_mask,
    filter_valid_components,
    select_component,
)


# ============================================================================
# Test has_contiguous_sequence_vectorised
# ============================================================================

class TestHasContiguousSequenceVectorised:
    """Tests for the vectorised contiguous sequence checking function."""
    
    def test_exactly_min_length_2(self):
        """Test with exactly min_length=2 coordinates [0,1]."""
        # This is the edge case: exactly 2 contiguous coordinates
        unique_coords = torch.tensor([
            [0, 1],  # x: contiguous sequence of length 2
            [5, 6]   # y: contiguous sequence of length 2
        ])
        assert has_contiguous_sequence_vectorised(unique_coords, min_length=2) is True
        
    def test_simple_contiguous_2d(self):
        """Test with 2D coordinates that have contiguous sequences."""
        # x: [0, 1, 2], y: [5, 6, 7] - both have contiguous sequences
        unique_coords = torch.tensor([
            [0, 1, 2],
            [5, 6, 7]
        ])
        assert has_contiguous_sequence_vectorised(unique_coords, min_length=2) is True
    
    def test_non_contiguous_2d(self):
        """Test with 2D coordinates without contiguous sequences."""
        # x: [0, 2, 4], y: [5, 7, 9] - no contiguous sequences
        unique_coords = torch.tensor([
            [0, 2, 4],
            [5, 7, 9]
        ])
        assert has_contiguous_sequence_vectorised(unique_coords, min_length=2) is False
    
    def test_single_dimension_contiguous(self):
        """Test where only one dimension has contiguous sequence."""
        # x: [0, 1, 2] (contiguous), y: [5, 7, 9] (not contiguous)
        unique_coords = torch.tensor([
            [0, 1, 2],
            [5, 7, 9]
        ])
        assert has_contiguous_sequence_vectorised(unique_coords, min_length=2) is False
    
    def test_min_length_2_with_four_coordinates(self):
        """Test with min_length=2 and four coordinates where only two are contiguous."""
        # x: [0, 1, 3, 4] - has contiguous sequence of length 2 at 0,1, and for y it has a contiguous sequence at every coord.
        unique_coords = torch.tensor([
            [0, 1, 3, 4],
            [5, 6, 7, 8]
        ])
        assert has_contiguous_sequence_vectorised(unique_coords, min_length=2) is True

    def test_exactly_min_length_3(self):
        """Test with exactly min_length=3 coordinates [0,1,2]."""
        # This is the edge case: exactly 3 contiguous coordinates
        unique_coords = torch.tensor([
            [0, 1, 2],  # x: contiguous sequence of length 3
            [5, 6, 7]   # y: contiguous sequence of length 3
        ])
        assert has_contiguous_sequence_vectorised(unique_coords, min_length=3) is True

    def test_min_length_3(self):
        """Test with min_length=3."""
        # x: [0, 1, 2, 3] - has contiguous sequence of length 3
        unique_coords = torch.tensor([
            [0, 1, 2, 3],
            [5, 6, 7, 8]
        ])
        assert has_contiguous_sequence_vectorised(unique_coords, min_length=3) is True
    
    def test_min_length_3_no_contiguous(self):
        """Test with min_length=3 but no contiguous sequence."""
        unique_coords = torch.tensor([
            [0, 1, 3, 4],  # gaps at 2 and 5
            [5, 6, 8, 9]
        ])
        assert has_contiguous_sequence_vectorised(unique_coords, min_length=3) is False
    
    def test_min_length_3_one_contiguous(self):
        """ Test with min_length=3 and exactly one contiguous sequence of length 3 per dimension."""
        unique_coords = torch.tensor([
            [0, 1, 2, 4],  # contiguous sequence of length 3 at 0,1,2
            [5, 6, 7, 9]   # contiguous sequence of length 3 at 5,6,7
        ])
        assert has_contiguous_sequence_vectorised(unique_coords, min_length=3) is True

    def test_padded_arrays(self):
        """Test with padded arrays (different lengths)."""
        # x: [0, 1, 2], y: [5, 6] (padded)
        unique_coords = torch.tensor([
            [0, 1, 2],
            [5, 6, float('nan')]
        ])
        assert has_contiguous_sequence_vectorised(unique_coords, min_length=2) is True
    
    def test_min_length_larger_than_available(self):
        """Test with min_length larger than the number of available coordinates."""
        unique_coords = torch.tensor([
            [0, 1],
            [5, 6]
        ])
        assert has_contiguous_sequence_vectorised(unique_coords, min_length=3) is False
        
    def test_min_length_larger_than_available_with_padding(self):
        """Test with min_length larger than available coordinates with padding."""
        unique_coords = torch.tensor([
            [0, 1, float('nan')],
            [5, 6, float('nan')]
        ])
        assert has_contiguous_sequence_vectorised(unique_coords, min_length=3) is False


    def test_single_element(self):
        """Test with single element per dimension."""
        unique_coords = torch.tensor([
            [5],
            [10]
        ])
        assert has_contiguous_sequence_vectorised(unique_coords, min_length=2) is False
    
    def test_empty_array(self):
        """Test with empty array."""
        unique_coords = torch.tensor([
            [],
            []
        ])
        assert has_contiguous_sequence_vectorised(unique_coords, min_length=2) is False


# ============================================================================
# Test can_generate_bbox_from_slice_fast
# ============================================================================

class TestCanGenerateBboxFromSliceFast:
    """Tests for 2D slice bbox generation pre-check."""
    
    def test_valid_2d_bbox(self):
        """Test a 2D slice that can generate a valid bbox."""
        # 5x5 square - should pass
        slice_2d = torch.zeros(5, 5)
        slice_2d[1:4, 1:4] = 1
        assert can_generate_bbox_from_slice_fast(slice_2d) is True
    
    def test_single_pixel(self):
        """Test a single pixel - should fail."""
        slice_2d = torch.zeros(5, 5)
        slice_2d[2, 2] = 1
        assert can_generate_bbox_from_slice_fast(slice_2d) is False
    
    def test_line_horizontal(self):
        """Test a horizontal line - should fail (only 1 in y dimension)."""
        slice_2d = torch.zeros(5, 5)
        slice_2d[2, 1:4] = 1
        assert can_generate_bbox_from_slice_fast(slice_2d) is False
    
    def test_line_vertical(self):
        """Test a vertical line - should fail (only 1 in x dimension)."""
        slice_2d = torch.zeros(5, 5)
        slice_2d[1:4, 2] = 1
        assert can_generate_bbox_from_slice_fast(slice_2d) is False
    
    def test_l_shape(self):
        """Test L-shaped region - should pass."""
        slice_2d = torch.zeros(5, 5)
        slice_2d[1:4, 2] = 1  # vertical
        slice_2d[1, 1:4] = 1  # horizontal
        assert can_generate_bbox_from_slice_fast(slice_2d) is True
    
    def test_diagonal_line(self):
        """Test a diagonal line - should not fail (it is contiguous, and would be a valid component based on 8-connectivity)."""
        slice_2d = torch.zeros(5, 5)
        for i in range(5):
            slice_2d[i, i] = 1
        assert can_generate_bbox_from_slice_fast(slice_2d) is True

    def test_sparse_points(self):
        """Test sparse points - should fail."""
        slice_2d = torch.zeros(10, 10)
        slice_2d[2, 2] = 1
        slice_2d[5, 5] = 1
        slice_2d[8, 8] = 1
        assert can_generate_bbox_from_slice_fast(slice_2d) is False


# ============================================================================
# Test can_generate_bbox_from_volume_fast
# ============================================================================

class TestCanGenerateBboxFromVolumeFast:
    """Tests for 3D volume bbox generation pre-check."""
    
    def test_valid_3d_bbox(self):
        """Test a 3D volume that can generate a valid bbox."""
        volume_3d = torch.zeros(5, 5, 5)
        volume_3d[1:4, 1:4, 1:4] = 1
        assert can_generate_bbox_from_volume_fast(volume_3d) is True
    
    def test_single_voxel(self):
        """Test a single voxel - should fail."""
        volume_3d = torch.zeros(5, 5, 5)
        volume_3d[2, 2, 2] = 1
        assert can_generate_bbox_from_volume_fast(volume_3d) is False
    
    def test_line_3d_x(self):
        """Test a 3D line - should fail."""
        volume_3d = torch.zeros(5, 5, 5)
        volume_3d[1:4, 2, 2] = 1
        assert can_generate_bbox_from_volume_fast(volume_3d) is False
    
    def test_line_3_y(self):
        """Test a 3D line - should fail."""
        volume_3d = torch.zeros(5, 5, 5)
        volume_3d[2, 1:4, 2] = 1
        assert can_generate_bbox_from_volume_fast(volume_3d) is False

    def test_line_3d_z(self):
        """Test a 3D line - should fail."""
        volume_3d = torch.zeros(5, 5, 5)
        volume_3d[2, 2, 1:4] = 1
        assert can_generate_bbox_from_volume_fast(volume_3d) is False

    def test_face_3d_transverse(self): #We pretend that Z = the length from head to foot. and Y is back to front. X is left to right, 
        """Test a 3D face (2D plane) - should fail."""
        volume_3d = torch.zeros(5, 5, 5)
        volume_3d[1:4, 1:4, 2] = 1
        assert can_generate_bbox_from_volume_fast(volume_3d) is False

    def test_face_3d_coronal(self):
        """Test a 3D face (2D plane) - should fail."""
        volume_3d = torch.zeros(5, 5, 5)
        volume_3d[1:4, 2, 1:4] = 1
        assert can_generate_bbox_from_volume_fast(volume_3d) is False
        
    def test_face_3d_sagittal(self):
        """Test a 3D face (2D plane) - should fail."""
        volume_3d = torch.zeros(5, 5, 5)
        volume_3d[2, 1:4, 1:4] = 1
        assert can_generate_bbox_from_volume_fast(volume_3d) is False

    def test_sparse_points_3d(self):
        """Test sparse points in 3D - should fail."""
        volume_3d = torch.zeros(10, 10, 10)
        volume_3d[2, 2, 2] = 1
        volume_3d[5, 5, 5] = 1
        volume_3d[8, 8, 8] = 1
        assert can_generate_bbox_from_volume_fast(volume_3d) is False
    
    def test_inclined_plane(self):
        """Test an inclined plane in 3D - should pass (it is contiguous and has a valid bbox)."""
        volume_3d = torch.zeros(5, 5, 5)
        for i in range(1, 4):
            volume_3d[1:4, i,i] = 1 #Just a tilted face, essentially a line moving along X axis, and spread across the diagonal spanned by Y and Z.
        assert can_generate_bbox_from_volume_fast(volume_3d) is True

# ============================================================================
# Test check_bbox_validity
# ============================================================================

class TestCheckBboxValidity:
    """Tests for bounding box validity checking."""
    
    def test_valid_3d_bbox(self):
        """Test a valid 3D bbox."""
        bbox_extrema = torch.tensor([[0, 0, 0, 5, 5, 5]])
        context_config = {
            'dimensionality': {
                'expected_dimensionality': 3
            },
            'image_dimensions': torch.Size([10, 10, 10])
        }
        
        valid, invalid_indices = check_bbox_validity(bbox_extrema, context_config)
        assert valid is True
        assert len(invalid_indices) == 0
        assert len(invalid_indices) % 2 == 0, "Invalid indices should come in pairs of min and max for the same dimension, so the length of invalid indices should be even."
    
    def test_valid_2d_bbox(self):
        """Test a valid 2D bbox."""
        bbox_extrema = torch.tensor([[0, 0, 5, 5, 5, 5]])
        context_config = {
            'dimensionality': {
                'expected_dimensionality': 2,
                'collapsed_dimension': 2,
            },
            'image_dimensions': torch.Size([10, 10, 10])
        }
        
        valid, invalid_indices = check_bbox_validity(bbox_extrema, context_config)
        assert valid is True
        assert len(invalid_indices) == 0
        assert len(invalid_indices) % 2 == 0, "Invalid indices should come in pairs of min and max for the same dimension, so the length of invalid indices should be even."

    def test_invalid_min_greater_than_max(self):
        """Test bbox where min > max in some dimension."""
        bbox_extrema = torch.tensor([[5, 0, 0, 0, 5, 5]])
        context_config = {
            'dimensionality': {
                'expected_dimensionality': 3
            },
            'image_dimensions': torch.Size([10, 10, 10])
        }
        
        valid, invalid_indices = check_bbox_validity(bbox_extrema, context_config)
        assert valid is False
        assert 0 in invalid_indices  # min_x > max_x
        assert 3 in invalid_indices
    
    def test_negative_coordinates(self):
        """Test bbox with negative coordinates."""
        bbox_extrema = torch.tensor([[-1, 0, 0, 5, 5, 5]])
        context_config = {
            'dimensionality': {
                'expected_dimensionality': 3
            },
            'image_dimensions': torch.Size([10, 10, 10])
        }
        
        valid, invalid_indices = check_bbox_validity(bbox_extrema, context_config)
        assert valid is False
        assert 0 in invalid_indices  # min_x < 0
        assert 3 in invalid_indices  # max_x < 0 (since we assume original bbox was valid, we will revert both min and max values for the dimension with the negative coordinate)
        assert len(invalid_indices) % 2 == 0, "Invalid indices should come in pairs of min and max for the same dimension, so the length of invalid indices should be even."

    def test_multiple_negative_coordinates(self):
        """Test bbox with negative coordinates."""
        bbox_extrema = torch.tensor([[-1, -1, 0, 5, 5, 5]])
        context_config = {
            'dimensionality': {
                'expected_dimensionality': 3
            },
            'image_dimensions': torch.Size([10, 10, 10])
        }
        
        valid, invalid_indices = check_bbox_validity(bbox_extrema, context_config)
        assert valid is False
        assert 0 in invalid_indices  # min_x < 0
        assert 3 in invalid_indices  # max_x < 0 (since we assume original bbox was valid, we will revert both min and max values for the dimension with the negative coordinate)
        assert 1 in invalid_indices  # min_y < 0
        assert 4 in invalid_indices  # max_y < 0 (since we assume original bbox was valid, we will revert both min and max values for the dimension with the negative coordinate)
        assert len(invalid_indices) % 2 == 0, "Invalid indices should come in pairs of min and max for the same dimension, so the length of invalid indices should be even."


    def test_exceeds_image_bounds(self):
        """Test bbox exceeding image bounds."""
        bbox_extrema = torch.tensor([[0, 0, 0, 10, 5, 5]])
        context_config = {
            'dimensionality': {
                'expected_dimensionality': 3
            },
            'image_dimensions': torch.Size([10, 10, 10])
        }
        
        valid, invalid_indices = check_bbox_validity(bbox_extrema, context_config)
        assert valid is False
        assert 3 in invalid_indices  # max_x >= 10
        assert 0 in invalid_indices  # min_x >= 10 (since we assume original bbox was valid, we will revert both min and max values for the dimension that exceeds bounds)
        assert len(invalid_indices) % 2 == 0, "Invalid indices should come in pairs of min and max for the same dimension, so the length of invalid indices should be even."
    
    def test_point_bbox(self):
        """Test point bbox (degenerate)."""
        bbox_extrema = torch.tensor([[5, 5, 5, 5, 5, 5]])
        context_config = {
            'dimensionality': {
                'expected_dimensionality': 3
            },
            'image_dimensions': torch.Size([10, 10, 10])
        }
        
        valid, invalid_indices = check_bbox_validity(bbox_extrema, context_config)
        assert valid is False
        assert len(invalid_indices) == 6  # All dimensions invalid
        assert invalid_indices == [0, 1, 2, 3, 4,5]  # All dimensions have min == max, so all are invalid
        assert len(invalid_indices) % 2 == 0, "Invalid indices should come in pairs of min and max for the same dimension, so the length of invalid indices should be even."

    def test_2d_bbox_wrong_collapsed_dim(self):
        """Test 2D bbox with wrong collapsed dimension."""
        bbox_extrema = torch.tensor([[0, 5, 5, 5, 5, 6]]) #dim 1 is collapsed.
        context_config = {
            'dimensionality': {
                'expected_dimensionality': 2,
                'collapsed_dimension': 2
            },
            'image_dimensions': torch.Size([10, 10, 10])
        }
        
        valid, invalid_indices = check_bbox_validity(bbox_extrema, context_config)
        assert valid is False
        assert 1 in invalid_indices
        assert 4 in invalid_indices
        # collapsed dimension should be in dim2 but is 5,6, meanwhile in dim1 it is collapsed at 5,5. This is not valid.
        assert len(invalid_indices) % 2 == 0, "Invalid indices should come in pairs of min and max for the same dimension, so the length of invalid indices should be even."
    
    def test_3d_bbox_with_collapsed_dim(self):
        """Test 3D bbox with a collapsed dimension (invalid)."""
        bbox_extrema = torch.tensor([[0, 0, 5, 5, 5, 5]])
        context_config = {
            'dimensionality': {
                'expected_dimensionality': 3
            },
            'image_dimensions': torch.Size([10, 10, 10])
        }
        
        valid, invalid_indices = check_bbox_validity(bbox_extrema, context_config)
        assert valid is False
        assert 2 in invalid_indices
        assert 5 in invalid_indices
        assert len(invalid_indices) % 2 == 0, "Invalid indices should come in pairs of min and max for the same dimension, so the length of invalid indices should be even."

    def test_3d_provided_2d_expected(self):
        """Test 3D bbox provided when 2D was expected"""
        bbox_extrema = torch.tensor([[0, 0, 0, 5, 5, 5]])
        context_config = {
            'dimensionality': {
                'expected_dimensionality': 2,
                'collapsed_dimension': 2
            },
            'image_dimensions': torch.Size([10, 10, 10])
        }
        
        valid, invalid_indices = check_bbox_validity(bbox_extrema, context_config)
        assert valid is False
        assert 2 in invalid_indices  # collapsed dimension is 0,5
        assert 5 in invalid_indices # collapsed dimension is 0,5
        assert len(invalid_indices) % 2 == 0, "Invalid indices should come in pairs of min and max for the same dimension, so the length of invalid indices should be even."

    def test_critical_failure(self):
        """Test critical failure mode raises exception."""
        bbox_extrema = torch.tensor([[5, 0, 0, 0, 5, 5]])
        context_config = {
            'dimensionality': {
                'expected_dimensionality': 3
            },
            'image_dimensions': torch.Size([10, 10, 10])
        }
        
        with pytest.raises(ValueError, match="Invalid bbox"):
            check_bbox_validity(bbox_extrema, context_config, critical_failure=True)
    
    def test_invalid_min_greater_than_max_critical_failure(self):
        """Test bbox where min > max in some dimension with critical failure."""
        bbox_extrema = torch.tensor([[5, 0, 0, 0, 5, 5]])
        context_config = {
            'dimensionality': {
                'expected_dimensionality': 3
            },
            'image_dimensions': torch.Size([10, 10, 10])
        }
        
        with pytest.raises(ValueError, match="Invalid bbox"):
            check_bbox_validity(bbox_extrema, context_config, critical_failure=True)
    
    def test_negative_coordinates_critical_failure(self):
        """Test bbox with negative coordinates with critical failure."""
        bbox_extrema = torch.tensor([[-1, 0, 0, 5, 5, 5]])
        context_config = {
            'dimensionality': {
                'expected_dimensionality': 3
            },
            'image_dimensions': torch.Size([10, 10, 10])
        }
        
        with pytest.raises(ValueError, match="Invalid bbox"):
            check_bbox_validity(bbox_extrema, context_config, critical_failure=True)
    
    def test_exceeds_image_bounds_critical_failure(self):
        """Test bbox exceeding image bounds with critical failure."""
        bbox_extrema = torch.tensor([[0, 0, 0, 10, 5, 5]])
        context_config = {
            'dimensionality': {
                'expected_dimensionality': 3
            },
            'image_dimensions': torch.Size([10, 10, 10])
        }
        
        with pytest.raises(ValueError, match="Invalid bbox"):
            check_bbox_validity(bbox_extrema, context_config, critical_failure=True)
    
    def test_point_bbox_critical_failure(self):
        """Test point bbox (degenerate) with critical failure."""
        bbox_extrema = torch.tensor([[5, 5, 5, 5, 5, 5]])
        context_config = {
            'dimensionality': {
                'expected_dimensionality': 3
            },
            'image_dimensions': torch.Size([10, 10, 10])
        }
        
        with pytest.raises(ValueError, match="Invalid bbox"):
            check_bbox_validity(bbox_extrema, context_config, critical_failure=True)
    
    def test_2d_bbox_wrong_collapsed_dim_critical_failure(self):
        """Test 2D bbox with wrong collapsed dimension with critical failure."""
        bbox_extrema = torch.tensor([[0, 5, 5, 5, 5, 6]])
        context_config = {
            'dimensionality': {
                'expected_dimensionality': 2,
                'collapsed_dimension': 2
            },
            'image_dimensions': torch.Size([10, 10, 10])
        }
        
        with pytest.raises(ValueError, match="Invalid bbox"):
            check_bbox_validity(bbox_extrema, context_config, critical_failure=True)
    
    def test_3d_bbox_with_collapsed_dim_critical_failure(self):
        """Test 3D bbox with a collapsed dimension (invalid) with critical failure."""
        bbox_extrema = torch.tensor([[0, 0, 5, 5, 5, 5]])
        context_config = {
            'dimensionality': {
                'expected_dimensionality': 3
            },
            'image_dimensions': torch.Size([10, 10, 10])
        }
        
        with pytest.raises(ValueError, match="Invalid bbox"):
            check_bbox_validity(bbox_extrema, context_config, critical_failure=True)
    
    def test_3d_provided_2d_expected_critical_failure(self):
        """Test 3D bbox provided when 2D was expected with critical failure."""
        bbox_extrema = torch.tensor([[0, 0, 0, 5, 5, 5]])
        context_config = {
            'dimensionality': {
                'expected_dimensionality': 2,
                'collapsed_dimension': 2
            },
            'image_dimensions': torch.Size([10, 10, 10])
        }
        
        with pytest.raises(ValueError, match="Invalid bbox"):
            check_bbox_validity(bbox_extrema, context_config, critical_failure=True)

# ============================================================================
# Test bbox_extrema
# ============================================================================

class TestBboxExtrema:
    """Tests for computing bbox extrema from binary mask."""
    
    def test_basic_3d_bbox_extrema(self):
        """Test 3D bbox extrema computation."""
        binary_mask = torch.zeros(10, 10, 10)
        binary_mask[2:7, 3:8, 1:6] = 1  # bbox from (2,3,1) to (6,7,5)
        binary_mask = binary_mask.to(torch.bool)
        
        bbox_args = {
            'dimensionality': 3
            }
        bbox = bbox_extrema(binary_mask, bbox_args)
        
        assert bbox.shape == (1, 6)
        assert bbox[0, 0] == 2  # min_x
        assert bbox[0, 1] == 3  # min_y
        assert bbox[0, 2] == 1  # min_z
        assert bbox[0, 3] == 6  # max_x
        assert bbox[0, 4] == 7  # max_y
        assert bbox[0, 5] == 5  # max_z
    
    def test_basic_2d_bbox_extrema(self):
        """Test 2D bbox extrema computation."""
        binary_mask = torch.zeros(10, 10, 10)
        binary_mask[2:7, 3:8, 5] = 1
        binary_mask = binary_mask.to(torch.bool)
        bbox_args = {
            'dimensionality': 2,
            'collapsed_dim': 2,
            'collapsed_slice_idx': 5
        }
        bbox = bbox_extrema(binary_mask, bbox_args)
        
        assert bbox.shape == (1, 6)
        assert bbox[0, 0] == 2  # min_x
        assert bbox[0, 1] == 3  # min_y
        assert bbox[0, 2] == 5  # min_z (collapsed)
        assert bbox[0, 3] == 6  # max_x
        assert bbox[0, 4] == 7  # max_y
        assert bbox[0, 5] == 5  # max_z (collapsed)
    
    def test_3d_cavity_shape(self):
        """Test with irregularly shaped binary mask - cavity. 3D case."""
        binary_mask = torch.zeros(10, 15, 20)
        binary_mask[2:7, 2:8, 1:6] = 1
        binary_mask[3:5, 3:6, 2:4] = 0 #insert a cavity
        binary_mask = binary_mask.to(torch.bool)

        bbox_args = {
            'dimensionality': 3
        }
        bbox = bbox_extrema(binary_mask, bbox_args)
        
        assert bbox.shape == (1, 6)
        assert bbox[0, 0] == 2  # min_x
        assert bbox[0, 1] == 2  # min_y
        assert bbox[0, 2] == 1  # min_z
        assert bbox[0, 3] == 6  # max_x
        assert bbox[0, 4] == 7  # max_y
        assert bbox[0, 5] == 5  # max_z
    
    def test_2d_cavity_shape(self):
        """Test with irregularly shaped binary mask - cavity. 2D case."""
        binary_mask = torch.zeros(10, 15, 20)
        binary_mask[2:7, 2:8, 5] = 1
        binary_mask[3:5, 3:6, 5] = 0 #insert a cavity
        binary_mask = binary_mask.to(torch.bool)

        bbox_args = {
            'dimensionality': 2,
            'collapsed_dim': 2,
            'collapsed_slice_idx': 5
        }
        bbox = bbox_extrema(binary_mask, bbox_args)
        
        assert bbox.shape == (1, 6)
        assert bbox[0, 0] == 2  # min_x
        assert bbox[0, 1] == 2  # min_y
        assert bbox[0, 2] == 5  # min_z (collapsed)
        assert bbox[0, 3] == 6  # max_x
        assert bbox[0, 4] == 7  # max_y
        assert bbox[0, 5] == 5  # max_z (collapsed)
    

    def test_irregular_shape(self):
        """Test with irregularly shaped binary mask - chunks missing?."""
        binary_mask = torch.zeros(10, 15, 20)
        binary_mask[2:7, 3:8, 1:6] = 1
        binary_mask[2, 4:6, 1] = 0 #delete a strip along the y axis on the outer surface.
        binary_mask = binary_mask.to(torch.bool)

        bbox_args = {
            'dimensionality': 3
        }
        bbox = bbox_extrema(binary_mask, bbox_args)
        
        assert bbox.shape == (1, 6)
        assert bbox[0, 0] == 2  # min_x #We didn't delete the entire plane!, just a strip from that slice.
        assert bbox[0, 1] == 3  # min_y
        assert bbox[0, 2] == 1  # min_z #We didn't delete the entire plane!, just a strip from that slice.
        assert bbox[0, 3] == 6  # max_x 
        assert bbox[0, 4] == 7  # max_y
        assert bbox[0, 5] == 5  # max_z

    def test_3d_bbox_extrema_single_voxel(self):
        """Test 3D bbox extrema with single voxel - should raise ValueError."""
        binary_mask = torch.zeros(10, 10, 10)
        binary_mask[5, 5, 5] = 1
        binary_mask = binary_mask.to(torch.bool)

        bbox_args = {
            'dimensionality': 3
        }
        
        with pytest.raises(ValueError, match="Invalid bounding box"):
            bbox_extrema(binary_mask, bbox_args)
    
    def test_2d_bbox_extrema_single_voxel(self):
        """Test 2D bbox extrema with single voxel - should raise ValueError."""
        binary_mask = torch.zeros(10, 10, 10)
        binary_mask[5, 5, 5] = 1
        binary_mask = binary_mask.to(torch.bool)

        bbox_args = {
            'dimensionality': 2,
            'collapsed_dim': 2,
            'collapsed_slice_idx': 5
        }
        
        with pytest.raises(ValueError, match="Invalid bounding box"):
            bbox_extrema(binary_mask, bbox_args)
    
    
    
    def test_2d_bbox_extrema_incorrect_slice_idx(self):
        '''Test 2D bbox extrema with the component provided not being in the correct slice index along the collapsed dimension.'''
        binary_mask = torch.zeros(10, 10, 10)
        binary_mask[2:7, 3:8, 5] = 1
        binary_mask = binary_mask.to(torch.bool)
        bbox_args = {
            'dimensionality': 2,
            'collapsed_dim': 2,
            'collapsed_slice_idx': 4 #but the component is actually in slice index 5
        }
        with pytest.raises(ValueError, match="Invalid bounding box: for a 2D bounding box, the extrema on the collapsed dimension must match the specified slice index in bbox_args"):
            bbox_extrema(binary_mask, bbox_args)
    
    def test_2d_bbox_extrema_mismatch_collapse(self):
        """Test 2D bbox with invalid collapsed dimension, i.e. the collapsed dimension had different min and max values"""
        binary_mask = torch.zeros(10, 10, 10)
        binary_mask[2:7, 4, 4:7] = 1
        binary_mask = binary_mask.to(torch.bool)
        
        bbox_args = {
            'dimensionality': 2,
            'collapsed_dim': 2,
            'collapsed_slice_idx': 4
        }
        with pytest.raises(ValueError, match="Invalid bounding box: for a 2D bounding box, the extrema on the collapsed dimension must match, but the collapsed dim had extrema of | Invalid bounding box: for a 2D bounding box, the extrema on the non-collapsed dimensions cannot match"):
            bbox_extrema(binary_mask, bbox_args)
        # Should not work since collapsed dim extrema do not match but match on the wrong dim.
    
    def test_invalid_dimensionality(self):
        """Test with invalid dimensionality."""
        binary_mask = torch.zeros(10, 10)
        binary_mask = binary_mask.to(torch.bool)

        bbox_args = {
            'dimensionality': 5
        }
        
        with pytest.raises(ValueError, match="Bbox_args 'dimensionality' must be either 2 or 3."):
            bbox_extrema(binary_mask, bbox_args)
    
    def test_invalid_collapsed_dim(self):
        """Test with invalid collapsed_dimension index -> must be in [0, 1, 2]."""
        binary_mask = torch.zeros(10, 10)
        binary_mask = binary_mask.to(torch.bool)

        bbox_args = {
            'dimensionality': 2,
            'collapsed_dim': 5
        }
        
        with pytest.raises(ValueError, match="Bbox_args 'collapsed_dim' must be 0, 1, or 2 "):
            bbox_extrema(binary_mask, bbox_args)
    
    def test_2d_bbox_3d_blob_provided(self):
        """Test 2D bbox with a 3D blob (0 collapsed dims when 1 expected). This is a completely invalid input, especially since the slice index is provided"""
        binary_mask = torch.zeros(10, 10, 10)
        binary_mask[2:7, 3:8, 1:6] = 1
        binary_mask = binary_mask.to(torch.bool)
        
        bbox_args = {
            'dimensionality': 2,
            'collapsed_dim': 2,
            'collapsed_slice_idx': 5
        }
        
        with pytest.raises(ValueError, match="Invalid bounding box: for a 2D bounding box, the extrema on the collapsed dimension must match, but the collapsed dim had extrema"):
            bbox_extrema(binary_mask, bbox_args)

    def test_2d_bbox_1d_line_provided(self):
        """Test 2D bbox with a 1D line (2 collapsed dims when 1 expected)."""
        binary_mask = torch.zeros(10, 10, 10)
        binary_mask[5, 3:8, 5] = 1
        binary_mask = binary_mask.to(torch.bool)
        
        bbox_args = {
            'dimensionality': 2,
            'collapsed_dim': 2,
            'collapsed_slice_idx': 5
        }
        
        with pytest.raises(ValueError, match="Invalid bounding box: for a 2D bounding box, the extrema on the non-collapsed dimensions cannot match"):
            bbox_extrema(binary_mask, bbox_args)

    def test_3d_bbox_2d_face_provided(self):
        """Test 3D bbox with a 2D face (1 collapsed dim when 0 expected)."""
        binary_mask = torch.zeros(10, 10, 10)
        binary_mask[2:7, 3:8, 5] = 1
        binary_mask = binary_mask.to(torch.bool)
        
        bbox_args = {
            'dimensionality': 3
        }
        
        with pytest.raises(ValueError, match="Invalid bounding box: min and max coordinates are the same for at least one dimension, indicating a degenerate bounding box."):
            bbox_extrema(binary_mask, bbox_args)

    def test_3d_bbox_1d_line_provided(self):
        """Test 3D bbox with a 1D line (2 collapsed dims when 0 expected)."""
        binary_mask = torch.zeros(10, 10, 10)
        binary_mask[5, 3:8, 5] = 1
        binary_mask = binary_mask.to(torch.bool)
        
        bbox_args = {
            'dimensionality': 3
        }
        
        with pytest.raises(ValueError, match="Invalid bounding box: min and max coordinates are the same for at least one dimension, indicating a degenerate bounding box."):
            bbox_extrema(binary_mask, bbox_args)



############################# All the functions related to component extraction, and stored in the component_extraction.py file ######################################################################

# ============================================================================
# Test convert_to_numpy
# ============================================================================

class TestConvertToNumpy:
    """Tests for converting tensors to numpy arrays."""
    
    def test_convert_torch_tensor(self):
        """Test converting a regular torch tensor to numpy."""
        tensor = torch.tensor([[1, 2, 3], [4, 5, 6]])
        arr = convert_to_numpy(tensor)
        
        assert isinstance(arr, np.ndarray)
        assert arr.shape == (2, 3)
        assert np.array_equal(arr, tensor.numpy())
    
    def test_convert_3d_tensor(self):
        """Test converting a 3D torch tensor to numpy."""
        tensor = torch.randn(4, 5, 6)
        arr = convert_to_numpy(tensor)
        
        assert isinstance(arr, np.ndarray)
        assert arr.shape == (4, 5, 6)
    
    def test_convert_cuda_tensor(self):
        """Test converting a CUDA tensor (if available)."""
        if torch.cuda.is_available():
            tensor = torch.randn(3, 4, 5).cuda()
            arr = convert_to_numpy(tensor)
            
            assert isinstance(arr, np.ndarray)
            assert arr.shape == (3, 4, 5)
        else:
            pytest.skip("CUDA not available")
    
    def test_convert_empty_tensor(self):
        """Test converting an empty tensor."""
        tensor = torch.zeros(0, 0)
        arr = convert_to_numpy(tensor)
        
        assert isinstance(arr, np.ndarray)
        assert arr.shape == (0, 0)


# ============================================================================
# Test validate_connectivity
# ============================================================================

class TestValidateConnectivity:
    """Tests for validating and converting connectivity parameters."""
    
    def test_2d_orthogonal_hops_1(self):
        """Test 2D with 1 orthogonal hop (face-adjacent)."""
        result = validate_connectivity(1, 2)
        assert result == 1
    
    def test_2d_orthogonal_hops_2(self):
        """Test 2D with 2 orthogonal hops (face + edge-adjacent)."""
        result = validate_connectivity(2, 2)
        assert result == 2
    
    def test_3d_orthogonal_hops_1(self):
        """Test 3D with 1 orthogonal hop (face-adjacent)."""
        result = validate_connectivity(1, 3)
        assert result == 1
    
    def test_3d_orthogonal_hops_2(self):
        """Test 3D with 2 orthogonal hops (face + edge-adjacent)."""
        result = validate_connectivity(2, 3)
        assert result == 2
    
    def test_3d_orthogonal_hops_3(self):
        """Test 3D with 3 orthogonal hops (all neighbors)."""
        result = validate_connectivity(3, 3)
        assert result == 3
    
    def test_orthogonal_hops_less_than_1(self):
        """Test with orthogonal hops < 1."""
        with pytest.raises(ValueError, match="at least 1"):
            validate_connectivity(0, 2)
    
    def test_orthogonal_hops_greater_than_ndim(self):
        """Test with orthogonal hops > ndim."""
        with pytest.raises(ValueError, match="not valid for 2D"):
            validate_connectivity(3, 2)


# ============================================================================
# Test generate_jitter
# ============================================================================

class TestGenerateJitter:
    """Tests for jitter parameter generation."""
    
    def test_absolute_jitter_3d_symmetric(self):
        """Test absolute jitter for 3D bbox."""
        sampling_config = {
            'dimensionality': {
                'expected_dimensionality': 3
            },
            'jitter_config': {
                'type': 'absolute',
                'sampling_mechanism': 'uniform_integer',
                'jitter_symmetric': True
            },
            'jitter_parameterisation': [2, 3, 4]
        }
        context_config = {
            'sampling_dimensions': torch.Size([10, 10, 10]),
            'bbox_extrema': torch.tensor([[0, 0, 0, 5, 5, 5]]),
            'collapsed_dim': None
        }
        
        jitter = generate_jitter(sampling_config, context_config)
        assert jitter.shape == (1, 6)
        # Check that jitter values are within expected range
        assert torch.all(jitter.abs() <= torch.tensor([2, 3, 4, 2, 3, 4]))
        assert torch.all(jitter[0, :3] == jitter[0, 3:]), "Jitter should be symmetric for min and max"
    
    def test_relative_box_jitter_3d_symmetric(self):
        """Test relative to box jitter for 3D bbox."""
        sampling_config = {
            'dimensionality': {
                'expected_dimensionality': 3
            },
            'jitter_config': {
                'type': 'relative_box',
                'sampling_mechanism': 'uniform_integer',
                'jitter_symmetric': True
            },
            'jitter_parameterisation': [0.1, 0.2, 0.3]
        }
        context_config = {
            'sampling_dimensions': torch.Size([10, 10, 10]),
            'bbox_extrema': torch.tensor([[0, 0, 0, 10, 10, 10]]),
            'collapsed_dim': None
        }
        
        jitter = generate_jitter(sampling_config, context_config)
        assert jitter.shape == (1, 6)
        # Box size is 10, so relative 0.1 = 1 voxel jitter
        assert torch.all(jitter.abs() <= torch.tensor([1, 2, 3, 1, 2, 3]))
    
    def test_relative_array_jitter_3d_symmetric(self):
        """Test relative_array jitter for 3D bbox with symmetric sampling."""
        sampling_config = {
            'dimensionality': {
                'expected_dimensionality': 3
            },
            'jitter_config': {
                'type': 'relative_array',
                'sampling_mechanism': 'uniform_integer',
                'jitter_symmetric': True
            },
            'jitter_parameterisation': [0.05, 0.1, 0.15]
        }
        context_config = {
            'sampling_dimensions': torch.Size([20, 20, 20]),
            'bbox_extrema': torch.tensor([[0, 0, 0, 5, 5, 5]]),
            'collapsed_dim': None
        }

        jitter = generate_jitter(sampling_config, context_config)
        assert jitter.shape == (1, 6)
        # Array size is 20, so relative 0.05*20 = 1, 0.1*20 = 2, 0.15*20 = 3
        thresholds = torch.tensor([1, 2, 3, 1, 2, 3])
        assert torch.all(jitter.abs() <= thresholds), f"Jitter {jitter} exceeds thresholds {thresholds}"
        assert torch.all(jitter[0, :3] == jitter[0, 3:]), \
            "Symmetric jitter should produce equal values for min and max"


    def test_absolute_jitter_2d_symmetric(self):
        """Test absolute jitter for 2D bbox."""
        sampling_config = {
            'dimensionality': {
                'expected_dimensionality': 2
            },
            'jitter_config': {
                'type': 'absolute',
                'sampling_mechanism': 'uniform_integer',
                'jitter_symmetric': True
            },
            'jitter_parameterisation': [2, 3]
        }
        context_config = {
            'sampling_dimensions': torch.Size([10, 10, 0]),  # collapsed dim = 2
            'bbox_extrema': torch.tensor([[0, 0, 5, 5, 5, 5]]),
            'collapsed_dim': 2
        }
        
        jitter = generate_jitter(sampling_config, context_config)
        assert jitter.shape == (1, 6)
        # Collapsed dimension should have 0 jitter
        assert jitter[0, 2] == 0
        assert jitter[0, 5] == 0
        # Non-collapsed dimensions should be within absolute threshold
        thresholds = torch.tensor([2, 3, 0, 2, 3, 0])
        assert torch.all(jitter.abs() <= thresholds), f"Jitter {jitter} exceeds thresholds {thresholds}"
        # Symmetric: min and max jitter should match
        assert torch.all(jitter[0, :3] == jitter[0, 3:])
    
    def test_relative_box_jitter_2d_symmetric(self):
        """Test relative_box jitter for 2D bbox with symmetric sampling."""
        sampling_config = {
            'dimensionality': {
                'expected_dimensionality': 2
            },
            'jitter_config': {
                'type': 'relative_box',
                'sampling_mechanism': 'uniform_integer',
                'jitter_symmetric': True
            },
            'jitter_parameterisation': [0.1, 0.2]
        }
        context_config = {
            'sampling_dimensions': torch.Size([10, 10, 0]),  # collapsed dim = 2
            'bbox_extrema': torch.tensor([[0, 0, 5, 10, 10, 5]]),
            'collapsed_dim': 2
        }

        jitter = generate_jitter(sampling_config, context_config)
        assert jitter.shape == (1, 6)
        # Box size is 10 in x and y, so relative 0.1*10 = 1, 0.2*10 = 2
        thresholds = torch.tensor([1, 2, 0, 1, 2, 0])
        assert torch.all(jitter.abs() <= thresholds), f"Jitter {jitter} exceeds thresholds {thresholds}"
        # Collapsed dimension should have 0 jitter
        assert jitter[0, 2] == 0 and jitter[0, 5] == 0
        # Symmetric: min and max jitter should match
        assert torch.all(jitter[0, :2] == jitter[0, 3:5])
        assert jitter[0, 2] == jitter[0, 5]

    def test_relative_array_jitter_2d_symmetric(self):
        """Test relative_array jitter for 2D bbox with symmetric sampling."""
        sampling_config = {
            'dimensionality': {
                'expected_dimensionality': 2
            },
            'jitter_config': {
                'type': 'relative_array',
                'sampling_mechanism': 'uniform_integer',
                'jitter_symmetric': True
            },
            'jitter_parameterisation': [0.1, 0.2]
        }
        context_config = {
            'sampling_dimensions': torch.Size([10, 10, 0]),  # collapsed dim = 2
            'bbox_extrema': torch.tensor([[0, 0, 5, 5, 5, 5]]),
            'collapsed_dim': 2
        }

        jitter = generate_jitter(sampling_config, context_config)
        assert jitter.shape == (1, 6)
        # Array size is 10 in x and y, so relative 0.1*10 = 1, 0.2*10 = 2
        thresholds = torch.tensor([1, 2, 0, 1, 2, 0])
        assert torch.all(jitter.abs() <= thresholds), f"Jitter {jitter} exceeds thresholds {thresholds}"
        # Collapsed dimension should have 0 jitter
        assert jitter[0, 2] == 0 and jitter[0, 5] == 0
        # Symmetric: min and max jitter should match
        assert torch.all(jitter[0, :2] == jitter[0, 3:5])

    #Now we will do the asymmetric jitter tests.
    def test_absolute_jitter_3d_asymmetric(self):
        """Test non-symmetric jitter for absolute 3D."""
        sampling_config = {
            'dimensionality': {
                'expected_dimensionality': 3
            },
            'jitter_config': {
                'type': 'absolute',
                'sampling_mechanism': 'uniform_integer',
                'jitter_symmetric': False
            },
            'jitter_parameterisation': [2, 3, 4]
        }
        context_config = {
            'sampling_dimensions': torch.Size([10, 10, 10]),
            'bbox_extrema': torch.tensor([[0, 0, 0, 5, 5, 5]]),
            'collapsed_dim': None
        }
        
        jitter = generate_jitter(sampling_config, context_config)
        assert jitter.shape == (1, 6)
        thresholds = torch.tensor([2, 3, 4, 2, 3, 4])
        assert torch.all(jitter.abs() <= thresholds), f"Jitter {jitter} exceeds thresholds {thresholds}"
        # Min and max jitter are independently sampled, so they should differ with high probability
        if torch.all(jitter[0, :3] == jitter[0, 3:]):
            for _ in range(10):
                jitter = generate_jitter(sampling_config, context_config)
                if not torch.all(jitter[0, :3] == jitter[0, 3:]):
                    break
            else:
                pytest.fail("Asymmetric jitter produced identical min/max across 10 trials")

    def test_relative_box_jitter_3d_asymmetric(self):
        """Test relative_box jitter for 3D bbox with asymmetric sampling."""
        sampling_config = {
            'dimensionality': {
                'expected_dimensionality': 3
            },
            'jitter_config': {
                'type': 'relative_box',
                'sampling_mechanism': 'uniform_integer',
                'jitter_symmetric': False
            },
            'jitter_parameterisation': [0.1, 0.2, 0.5]
        }
        context_config = {
            'sampling_dimensions': torch.Size([10, 10, 10]),
            'bbox_extrema': torch.tensor([[0, 0, 0, 10, 10, 10]]),
            'collapsed_dim': None
        }

        jitter = generate_jitter(sampling_config, context_config)
        assert jitter.shape == (1, 6)
        # Box size is 10, so relative 0.1*10 = 1, 0.2*10 = 2, 0.5*10 = 5
        thresholds = torch.tensor([1, 2, 5, 1, 2, 5])
        assert torch.all(jitter.abs() <= thresholds), f"Jitter {jitter} exceeds thresholds {thresholds}"
        if torch.all(jitter[0, :3] == jitter[0, 3:]):
            for _ in range(10):
                jitter = generate_jitter(sampling_config, context_config)
                if not torch.all(jitter[0, :3] == jitter[0, 3:]):
                    break
            else:
                pytest.fail("Asymmetric jitter produced identical min/max across 10 trials")

    def test_relative_array_jitter_3d_asymmetric(self):
        """Test relative_array jitter for 3D bbox with asymmetric sampling."""
        sampling_config = {
            'dimensionality': {
                'expected_dimensionality': 3
            },
            'jitter_config': {
                'type': 'relative_array',
                'sampling_mechanism': 'uniform_integer',
                'jitter_symmetric': False
            },
            'jitter_parameterisation': [0.05, 0.1, 0.35]
        }
        context_config = {
            'sampling_dimensions': torch.Size([20, 20, 20]),
            'bbox_extrema': torch.tensor([[0, 0, 0, 5, 5, 5]]),
            'collapsed_dim': None
        }

        jitter = generate_jitter(sampling_config, context_config)
        assert jitter.shape == (1, 6)
        # Array size is 20, so relative 0.05*20 = 1, 0.1*20 = 2, 0.35*20 = 7
        thresholds = torch.tensor([1, 2, 7, 1, 2, 7])
        assert torch.all(jitter.abs() <= thresholds), f"Jitter {jitter} exceeds thresholds {thresholds}"
        if torch.all(jitter[0, :3] == jitter[0, 3:]):
            for _ in range(10):
                jitter = generate_jitter(sampling_config, context_config)
                if not torch.all(jitter[0, :3] == jitter[0, 3:]):
                    break
            else:
                pytest.fail("Asymmetric jitter produced identical min/max across 10 trials")

    def test_absolute_jitter_2d_asymmetric(self):
        """Test absolute jitter for 2D bbox with asymmetric sampling."""
        sampling_config = {
            'dimensionality': {
                'expected_dimensionality': 2
            },
            'jitter_config': {
                'type': 'absolute',
                'sampling_mechanism': 'uniform_integer',
                'jitter_symmetric': False
            },
            'jitter_parameterisation': [2, 3]
        }
        context_config = {
            'sampling_dimensions': torch.Size([10, 0, 10]),  # collapsed dim = 1
            'bbox_extrema': torch.tensor([[0, 5, 5, 5, 5, 8]]),
            'collapsed_dim': 1
        }

        jitter = generate_jitter(sampling_config, context_config)
        assert jitter.shape == (1, 6)
        # Collapsed dimension should have 0 jitter
        assert jitter[0, 1] == 0
        assert jitter[0, 4] == 0
        # Non-collapsed dimensions should be within absolute threshold
        thresholds = torch.tensor([2, 0, 3, 2, 0, 3])
        assert torch.all(jitter.abs() <= thresholds), f"Jitter {jitter} exceeds thresholds {thresholds}"
        # Min and max jitter differ with high probability for non-collapsed dims
        if torch.all(jitter[0, :3] == jitter[0, 3:]):
            for _ in range(10):
                jitter = generate_jitter(sampling_config, context_config)
                if not torch.all(jitter[0, :3] == jitter[0, 3:]):
                    break
            else:
                pytest.fail("Asymmetric jitter produced identical min/max across 10 trials")

    def test_relative_box_jitter_2d_asymmetric(self):
        """Test relative_box jitter for 2D bbox with asymmetric sampling."""
        sampling_config = {
            'dimensionality': {
                'expected_dimensionality': 2
            },
            'jitter_config': {
                'type': 'relative_box',
                'sampling_mechanism': 'uniform_integer',
                'jitter_symmetric': False
            },
            'jitter_parameterisation': [0.1, 0.2]
        }
        context_config = {
            'sampling_dimensions': torch.Size([10, 10, 0]),  # collapsed dim = 2
            'bbox_extrema': torch.tensor([[0, 0, 5, 10, 10, 5]]),
            'collapsed_dim': 2
        }

        jitter = generate_jitter(sampling_config, context_config)
        assert jitter.shape == (1, 6)
        # Box size is 10 in x and y, so relative 0.1*10 = 1, 0.2*10 = 2
        thresholds = torch.tensor([1, 2, 0, 1, 2, 0])
        assert torch.all(jitter.abs() <= thresholds), f"Jitter {jitter} exceeds thresholds {thresholds}"
        # Collapsed dimension should have 0 jitter
        assert jitter[0, 2] == 0 and jitter[0, 5] == 0
        # Min and max jitter differ with high probability for non-collapsed dims
        if torch.all(jitter[0, :2] == jitter[0, 3:5]):
            for _ in range(10):
                jitter = generate_jitter(sampling_config, context_config)
                if not torch.all(jitter[0, :2] == jitter[0, 3:5]):
                    break
            else:
                pytest.fail("Asymmetric jitter produced identical min/max across 10 trials")

    def test_relative_array_jitter_2d_asymmetric(self):
        """Test relative_array jitter for 2D bbox with asymmetric sampling."""
        sampling_config = {
            'dimensionality': {
                'expected_dimensionality': 2
            },
            'jitter_config': {
                'type': 'relative_array',
                'sampling_mechanism': 'uniform_integer',
                'jitter_symmetric': False
            },
            'jitter_parameterisation': [0.1, 0.2]
        }
        context_config = {
            'sampling_dimensions': torch.Size([10, 10, 0]),  # collapsed dim = 2
            'bbox_extrema': torch.tensor([[0, 0, 5, 5, 5, 5]]),
            'collapsed_dim': 2
        }

        jitter = generate_jitter(sampling_config, context_config)
        assert jitter.shape == (1, 6)
        # Array size is 10 in x and y, so relative 0.1*10 = 1, 0.2*10 = 2
        thresholds = torch.tensor([1, 2, 0, 1, 2, 0])
        assert torch.all(jitter.abs() <= thresholds), f"Jitter {jitter} exceeds thresholds {thresholds}"
        # Collapsed dimension should have 0 jitter
        assert jitter[0, 2] == 0 and jitter[0, 5] == 0
        # Asymmetric jitter should produce different min/max values; retry if coincidental match occurs
        if torch.all(jitter[0, :2] == jitter[0, 3:5]):
            for _ in range(10):
                jitter = generate_jitter(sampling_config, context_config)
                if not torch.all(jitter[0, :2] == jitter[0, 3:5]):
                    break
            else:
                pytest.fail("Asymmetric jitter produced identical min/max across 10 trials")
    
    #Now we will do some tests to check for config validity.
    def test_relative_box_parameterisation_exceeds_one(self):
        """Test relative_box with parameterisation > 1 raises error."""
        sampling_config = {
            'dimensionality': {
                'expected_dimensionality': 3
            },
            'jitter_config': {
                'type': 'relative_box',
                'sampling_mechanism': 'uniform_integer',
                'jitter_symmetric': True
            },
            'jitter_parameterisation': [0.5, 1.5, 0.3]
        }
        context_config = {
            'sampling_dimensions': torch.Size([10, 10, 10]),
            'bbox_extrema': torch.tensor([[0, 0, 0, 5, 5, 5]]),
            'collapsed_dim': None
        }

        with pytest.raises(ValueError, match="in \\[0, 1\\]"):
            generate_jitter(sampling_config, context_config)

    def test_relative_array_parameterisation_exceeds_one(self):
        """Test relative_array with parameterisation > 1 raises error."""
        sampling_config = {
            'dimensionality': {
                'expected_dimensionality': 3
            },
            'jitter_config': {
                'type': 'relative_array',
                'sampling_mechanism': 'uniform_integer',
                'jitter_symmetric': True
            },
            'jitter_parameterisation': [1.2, 0.5, 0.3]
        }
        context_config = {
            'sampling_dimensions': torch.Size([10, 10, 10]),
            'bbox_extrema': torch.tensor([[0, 0, 0, 5, 5, 5]]),
            'collapsed_dim': None
        }

        with pytest.raises(ValueError, match="in \\[0, 1\\]"):
            generate_jitter(sampling_config, context_config)

    def test_missing_dimensionality(self):
        """Test with missing dimensionality in sampling_config."""
        sampling_config = {
            'jitter_config': {
                'type': 'absolute',
                'sampling_mechanism': 'uniform_integer',
                'jitter_symmetric': True
            },
            'jitter_parameterisation': [2, 3, 4]
        }
        context_config = {
            'sampling_dimensions': torch.Size([10, 10, 10]),
            'bbox_extrema': torch.tensor([[0, 0, 0, 5, 5, 5]]),
            'collapsed_dim': None
        }
        
        with pytest.raises(KeyError, match="Sampling config must contain the key 'dimensionality'"):
            generate_jitter(sampling_config, context_config)
    
    def test_missing_jitter_config(self):
        """Test with missing jitter_config."""
        sampling_config = {
            'dimensionality': {
                'expected_dimensionality': 3
            },
            'jitter_parameterisation': [2, 3, 4]
        }
        context_config = {
            'sampling_dimensions': torch.Size([10, 10, 10]),
            'bbox_extrema': torch.tensor([[0, 0, 0, 5, 5, 5]]),
            'collapsed_dim': None
        }
        
        with pytest.raises(KeyError, match="Sampling config must contain the key 'jitter_config'"):
            generate_jitter(sampling_config, context_config)
    
    def test_missing_jitter_symmetric(self):
        """Test with missing jitter_symmetric in jitter_config."""
        sampling_config = {
            'dimensionality': {
                'expected_dimensionality': 3
            },
            'jitter_config': {
                'type': 'absolute',
                'sampling_mechanism': 'uniform_integer'
            },
            'jitter_parameterisation': [2, 3, 4]
        }
        context_config = {
            'sampling_dimensions': torch.Size([10, 10, 10]),
            'bbox_extrema': torch.tensor([[0, 0, 0, 5, 5, 5]]),
            'collapsed_dim': None
        }

        with pytest.raises(KeyError, match="Sampling config must contain the key 'jitter_symmetric'"):
            generate_jitter(sampling_config, context_config)

    def test_missing_jitter_type(self):
        """Test with missing type in jitter_config."""
        sampling_config = {
            'dimensionality': {
                'expected_dimensionality': 3
            },
            'jitter_config': {
                'sampling_mechanism': 'uniform_integer',
                'jitter_symmetric': True
            },
            'jitter_parameterisation': [2, 3, 4]
        }
        context_config = {
            'sampling_dimensions': torch.Size([10, 10, 10]),
            'bbox_extrema': torch.tensor([[0, 0, 0, 5, 5, 5]]),
            'collapsed_dim': None
        }

        with pytest.raises(KeyError, match="Sampling config must contain the key 'type' within 'jitter_config'"):
            generate_jitter(sampling_config, context_config)

    def test_missing_sampling_mechanism(self):
        """Test with missing sampling_mechanism in jitter_config."""
        sampling_config = {
            'dimensionality': {
                'expected_dimensionality': 3
            },
            'jitter_config': {
                'type': 'absolute',
                'jitter_symmetric': True
            },
            'jitter_parameterisation': [2, 3, 4]
        }
        context_config = {
            'sampling_dimensions': torch.Size([10, 10, 10]),
            'bbox_extrema': torch.tensor([[0, 0, 0, 5, 5, 5]]),
            'collapsed_dim': None
        }

        with pytest.raises(KeyError, match="Sampling config must contain the key 'sampling_mechanism' within 'jitter_config'"):
            generate_jitter(sampling_config, context_config)

    def test_invalid_expected_dimensionality(self):
        """Test with expected_dimensionality not 2 or 3."""
        sampling_config = {
            'dimensionality': {
                'expected_dimensionality': 4
            },
            'jitter_config': {
                'type': 'absolute',
                'sampling_mechanism': 'uniform_integer',
                'jitter_symmetric': True
            },
            'jitter_parameterisation': [2, 3, 4, 5]
        }
        context_config = {
            'sampling_dimensions': torch.Size([10, 10, 10]),
            'bbox_extrema': torch.tensor([[0, 0, 0, 5, 5, 5]]),
            'collapsed_dim': None
        }

        with pytest.raises(ValueError, match="must be either 2 or 3"):
            generate_jitter(sampling_config, context_config)

    def test_unsupported_jitter_type(self):
        """Test with unsupported jitter config type."""
        sampling_config = {
            'dimensionality': {
                'expected_dimensionality': 3
            },
            'jitter_config': {
                'type': 'invalid_type',
                'sampling_mechanism': 'uniform_integer',
                'jitter_symmetric': True
            },
            'jitter_parameterisation': [2, 3, 4]
        }
        context_config = {
            'sampling_dimensions': torch.Size([10, 10, 10]),
            'bbox_extrema': torch.tensor([[0, 0, 0, 5, 5, 5]]),
            'collapsed_dim': None
        }

        with pytest.raises(ValueError, match="Unsupported jitter config type"):
            generate_jitter(sampling_config, context_config)

    def test_invalid_collapsed_dim_2d(self):
        """Test with invalid collapsed_dim for 2D bbox."""
        sampling_config = {
            'dimensionality': {
                'expected_dimensionality': 2
            },
            'jitter_config': {
                'type': 'absolute',
                'sampling_mechanism': 'uniform_integer',
                'jitter_symmetric': True
            },
            'jitter_parameterisation': [2, 3]
        }
        context_config = {
            'sampling_dimensions': torch.Size([10, 10, 0]),
            'bbox_extrema': torch.tensor([[0, 0, 5, 5, 5, 5]]),
            'collapsed_dim': 5
        }

        with pytest.raises(ValueError, match="must be 0, 1, or 2"):
            generate_jitter(sampling_config, context_config)

    def test_missing_collapsed_dim_2d(self):
        """Test with missing collapsed_dim for 2D bbox."""
        sampling_config = {
            'dimensionality': {
                'expected_dimensionality': 2
            },
            'jitter_config': {
                'type': 'absolute',
                'sampling_mechanism': 'uniform_integer',
                'jitter_symmetric': True
            },
            'jitter_parameterisation': [2, 3]
        }
        context_config = {
            'sampling_dimensions': torch.Size([10, 10, 0]),
            'bbox_extrema': torch.tensor([[0, 0, 5, 5, 5, 5]])
        }

        with pytest.raises(KeyError, match="collapsed_dim"):
            generate_jitter(sampling_config, context_config)

    def test_2d_collapsed_dim_not_actually_collapsed(self):
        """Test 2D with the collapsed dim having different min and max values."""
        sampling_config = {
            'dimensionality': {
                'expected_dimensionality': 2
            },
            'jitter_config': {
                'type': 'absolute',
                'sampling_mechanism': 'uniform_integer',
                'jitter_symmetric': True
            },
            'jitter_parameterisation': [2, 3]
        }
        context_config = {
            'sampling_dimensions': torch.Size([10, 10, 0]),
            'bbox_extrema': torch.tensor([[0, 0, 4, 5, 5, 6]]),  # z dim has 4 and 6, not collapsed
            'collapsed_dim': 2
        }

        with pytest.raises(ValueError, match="extrema for the collapsed dimension must be the same"):
            generate_jitter(sampling_config, context_config)

    def test_3d_bbox_with_collapsed_dimension(self):
        """Test 3D jitter on bbox extrema that are actually 2D (collapsed dimension)."""
        sampling_config = {
            'dimensionality': {
                'expected_dimensionality': 3
            },
            'jitter_config': {
                'type': 'absolute',
                'sampling_mechanism': 'uniform_integer',
                'jitter_symmetric': True
            },
            'jitter_parameterisation': [2, 3, 4]
        }
        context_config = {
            'sampling_dimensions': torch.Size([10, 10, 10]),
            'bbox_extrema': torch.tensor([[0, 0, 5, 5, 5, 5]]),  # z dim is collapsed (5==5)
            'collapsed_dim': None
        }

        with pytest.raises(ValueError, match="must not have collapsed dimensions"):
            generate_jitter(sampling_config, context_config)

    def test_3d_bbox_with_multiple_collapsed_dimensions(self):
        """Test 3D jitter on bbox extrema with multiple collapsed dimensions."""
        sampling_config = {
            'dimensionality': {
                'expected_dimensionality': 3
            },
            'jitter_config': {
                'type': 'absolute',
                'sampling_mechanism': 'uniform_integer',
                'jitter_symmetric': True
            },
            'jitter_parameterisation': [2, 3, 4]
        }
        context_config = {
            'sampling_dimensions': torch.Size([10, 10, 10]),
            'bbox_extrema': torch.tensor([[0, 5, 5, 5, 5, 5]]),  # y and z both collapsed
            'collapsed_dim': None
        }

        with pytest.raises(ValueError, match="must not have collapsed dimensions"):
            generate_jitter(sampling_config, context_config)

    def test_2d_sampling_dim_not_zero(self):
        """Test 2D with sampling dimension not 0 along the collapsed dim."""
        sampling_config = {
            'dimensionality': {
                'expected_dimensionality': 2
            },
            'jitter_config': {
                'type': 'absolute',
                'sampling_mechanism': 'uniform_integer',
                'jitter_symmetric': True
            },
            'jitter_parameterisation': [2, 3]
        }
        context_config = {
            'sampling_dimensions': torch.Size([10, 10, 10]),  # collapsed dim should be 0
            'bbox_extrema': torch.tensor([[0, 0, 5, 5, 5, 5]]),
            'collapsed_dim': 2
        }

        with pytest.raises(ValueError, match="size of the collapsed dimension in the sampling dimensions must be 0"):
            generate_jitter(sampling_config, context_config)

    def test_2d_sampling_dim_zero_wrong_dim(self):
        """Test 2D with sampling dimension 0 in a non-collapsed dim (should fail)."""
        sampling_config = {
            'dimensionality': {
                'expected_dimensionality': 2
            },
            'jitter_config': {
                'type': 'absolute',
                'sampling_mechanism': 'uniform_integer',
                'jitter_symmetric': True
            },
            'jitter_parameterisation': [2, 3]
        }
        context_config = {
            'sampling_dimensions': torch.Size([0, 10, 0]),  # dim 0 is 0 even though collapsed_dim=2
            'bbox_extrema': torch.tensor([[5, 0, 5, 10, 5, 5]]),
            'collapsed_dim': 2
        }
        with pytest.raises(ValueError, match="The size of the non-collapsed dimensions in the sampling dimensions cannot be 0"):
            generate_jitter(sampling_config, context_config)

    def test_negative_jitter_parameterisation(self):
        """Test with negative jitter parameterisation."""
        sampling_config = {
            'dimensionality': {
                'expected_dimensionality': 3
            },
            'jitter_config': {
                'type': 'absolute',
                'sampling_mechanism': 'uniform_integer',
                'jitter_symmetric': True
            },
            'jitter_parameterisation': [-2, 3, 4]
        }
        context_config = {
            'sampling_dimensions': torch.Size([10, 10, 10]),
            'bbox_extrema': torch.tensor([[0, 0, 0, 5, 5, 5]]),
            'collapsed_dim': None
        }
        
        with pytest.raises(ValueError, match="non-negative"):
            generate_jitter(sampling_config, context_config)

    # --- Distributional tests for jitter randomness ---

    def _compute_expected_frequencies(self, threshold, n_samples):
        """Compute expected frequencies for round(U(-T,T)) distribution.
        
        Interior values |k| < T have P(k) = 1/(2T), endpoints |k| = T have P(k) = 1/(4T), due to 
        half the mass in the outer half-intervals being rounded to the endpoint -> i.e. only [-T, -T+0.5) maps to
        -T, and same symmetry wise for the upper endpoint. the remaining have full integer length intervals.
        """
        possible_vals = torch.arange(-threshold, threshold + 1)
        #Expected, since it is uniform, would be a uniform distribution across the bins.
        expected = torch.full_like(possible_vals, n_samples / (2 * threshold), dtype=torch.float)
        expected[0] = n_samples / (4 * threshold)   # k = -threshold
        expected[-1] = n_samples / (4 * threshold)  # k = threshold
        return possible_vals, expected

    def _run_distributional_test(self, sampling_config, context_config, dim,
                                  threshold=None, n_samples=30000):
        """Draw many jitter samples and run chi-squared test on one dimension's distribution.

        Args:
            threshold: Expected absolute jitter bound for this dimension.
                       If None, inferred from sampling_config (for 'absolute' type).
        """
        if threshold is None:
            raw_dim = dim if dim < 3 else dim - 3
            raw_val = sampling_config['jitter_parameterisation'][raw_dim]
            threshold = int(raw_val.item() if hasattr(raw_val, 'item') else raw_val)
        if threshold == 0:
            samples = torch.cat([generate_jitter(sampling_config, context_config)[:, dim:dim+1]
                                for _ in range(n_samples)])
            assert (samples == 0).all(), f"Dim {dim}: expected zero jitter but got nonzero values"
            return

        samples = torch.cat([generate_jitter(sampling_config, context_config)[:, dim:dim+1]
                            for _ in range(n_samples)])

        observed_counts = torch.zeros(2 * threshold + 1)
        for i in range(-threshold, threshold + 1):
            observed_counts[i + threshold] = (samples == i).sum().item()

        expected_vals, expected_counts = self._compute_expected_frequencies(threshold, n_samples)

        chi_sq = ((observed_counts - expected_counts).pow(2) / expected_counts).sum().item()
        df = len(expected_vals) - 1
        critical = chi2.ppf(0.999, df)
        assert chi_sq < critical, (
            f"Dim {dim}: chi-squared={chi_sq:.2f} (df={df}) exceeds critical={critical:.2f}. "
            f"Threshold={threshold}, distribution may deviate from expected."
        )

    def test_symmetric_jitter_3d_distribution(self):
        """Distributional test for symmetric 3D jitter across all dimensions."""
        torch.manual_seed(42)
        sampling_config = {
            'dimensionality': {'expected_dimensionality': 3},
            'jitter_config': {
                'type': 'absolute',
                'sampling_mechanism': 'uniform_integer',
                'jitter_symmetric': True
            },
            'jitter_parameterisation': [2, 3, 4]
        }
        context_config = {
            'sampling_dimensions': torch.Size([10, 10, 10]),
            'bbox_extrema': torch.tensor([[0, 0, 0, 5, 5, 5]]),
            'collapsed_dim': None
        }
        for dim in range(3):
            self._run_distributional_test(sampling_config, context_config, dim)

    def test_asymmetric_jitter_3d_distribution(self):
        """Distributional test for asymmetric 3D jitter across all dimensions."""
        torch.manual_seed(42)
        sampling_config = {
            'dimensionality': {'expected_dimensionality': 3},
            'jitter_config': {
                'type': 'absolute',
                'sampling_mechanism': 'uniform_integer',
                'jitter_symmetric': False
            },
            'jitter_parameterisation': [2, 3, 4]
        }
        context_config = {
            'sampling_dimensions': torch.Size([10, 10, 10]),
            'bbox_extrema': torch.tensor([[0, 0, 0, 5, 5, 5]]),
            'collapsed_dim': None
        }
        # For asymmetric jitter, test both min (dim) and max (dim+3) independently
        thresholds = [2, 3, 4]
        for raw_dim in range(3):
            for offset in [0, 3]:
                dim = raw_dim + offset
                self._run_distributional_test(
                    sampling_config, context_config, dim,
                    threshold=thresholds[raw_dim]
                )

    def test_relative_box_jitter_distribution(self):
        """Distributional test for relative_box jitter (thresholds derived from box size)."""
        torch.manual_seed(42)
        sampling_config = {
            'dimensionality': {'expected_dimensionality': 3},
            'jitter_config': {
                'type': 'relative_box',
                'sampling_mechanism': 'uniform_integer',
                'jitter_symmetric': True
            },
            'jitter_parameterisation': [0.1, 0.2, 0.3]
        }
        # Box is 10x10x10, so thresholds = [1, 2, 3]
        context_config = {
            'sampling_dimensions': torch.Size([10, 10, 10]),
            'bbox_extrema': torch.tensor([[0, 0, 0, 10, 10, 10]]),
            'collapsed_dim': None
        }
        thresholds = [1, 2, 3]
        for dim in range(3):
            self._run_distributional_test(
                sampling_config, context_config, dim,
                threshold=thresholds[dim]
            )

    def test_relative_array_jitter_distribution(self):
        """Distributional test for relative_array jitter (thresholds derived from array size)."""
        torch.manual_seed(42)
        sampling_config = {
            'dimensionality': {'expected_dimensionality': 3},
            'jitter_config': {
                'type': 'relative_array',
                'sampling_mechanism': 'uniform_integer',
                'jitter_symmetric': True
            },
            'jitter_parameterisation': [0.05, 0.1, 0.15]
        }
        # Array is 20x20x20, so thresholds = [1, 2, 3]
        context_config = {
            'sampling_dimensions': torch.Size([20, 20, 20]),
            'bbox_extrema': torch.tensor([[0, 0, 0, 5, 5, 5]]),
            'collapsed_dim': None
        }
        thresholds = [1, 2, 3]
        for dim in range(3):
            self._run_distributional_test(
                sampling_config, context_config, dim,
                threshold=thresholds[dim]
            )


# ============================================================================
# Test apply_jitter
# ============================================================================

class TestApplyJitter:
    """Tests for applying jitter to bounding boxes.

    For each of 2D and 3D, tests cover:
      - Non-problematic jitter (bbox remains valid)
      - Each failure mode from check_bbox_validity:
          min > max, negative coordinates, exceeds image bounds,
          point bbox (all dims degenerate), and dimensionality mismatches
          (collapsed dim in 3D; collapsed/uncollapsed dim in 2D).
      - missing config keys 
    """

    # ============================================================================
    # 3D bbox tests
    # ============================================================================

    def test_3d_non_problematic_jitter(self):
        """3D: Non-problematic jitter keeps bbox valid."""
        bbox = torch.tensor([[0, 0, 0, 10, 10, 10]])
        jitter = torch.tensor([[1, 1, 1, 1, 1, 1]])
        context_config = {
            'dimensionality': {'expected_dimensionality': 3},
            'image_dimensions': torch.Size([20, 20, 20])
        }
        result = apply_jitter(bbox, jitter, context_config)
        expected = torch.tensor([[1, 1, 1, 11, 11, 11]])
        assert torch.equal(result, expected)

    def test_3d_jitter_min_greater_than_max_fallback(self):
        """3D: min > max triggers fallback for that dimension only."""
        bbox = torch.tensor([[5, 1, 0, 10, 5, 5]])
        jitter = torch.tensor([[10, 1, 0, 0, 1, 0]])
        context_config = {
            'dimensionality': {'expected_dimensionality': 3},
            'image_dimensions': torch.Size([20, 20, 20])
        }
        result = apply_jitter(bbox, jitter, context_config)
        expected = torch.tensor([[5, 2, 0, 10, 6, 5]])
        assert torch.equal(result, expected)

    def test_3d_jitter_negative_coordinates_fallback(self):
        """3D: Negative coordinate triggers fallback for that dimension only."""
        bbox = torch.tensor([[1, 0, 0, 5, 5, 5]])
        jitter = torch.tensor([[-5, 1, 1, 1, 1, 1]]) #We have a valid jitter on x_max, but not x_min.
        #We will still assert that both values match their previous values to ensure we can definitely revert to
        # a valid bbox.
        context_config = {
            'dimensionality': {'expected_dimensionality': 3},
            'image_dimensions': torch.Size([10, 10, 10])
        }
        result = apply_jitter(bbox, jitter, context_config)
        expected = torch.tensor([[1, 1, 1, 5, 6, 6]])
        assert torch.equal(result, expected)

    def test_3d_jitter_exceeds_bounds_fallback(self):
        """3D: Exceeding image bounds triggers fallback for that dimension only."""
        bbox = torch.tensor([[0, 0, 0, 8, 8, 8]])
        jitter = torch.tensor([[1, 1, 1, 5, 1, 1]])
        context_config = {
            'dimensionality': {'expected_dimensionality': 3},
            'image_dimensions': torch.Size([10, 10, 10])
        }
        result = apply_jitter(bbox, jitter, context_config)
        expected = torch.tensor([[0, 1, 1, 8, 9, 9]])
        assert torch.equal(result, expected)

    def test_3d_jitter_point_bbox_fallback(self):
        """3D: Point bbox triggers full fallback (all dimensions)."""
        bbox = torch.tensor([[5, 5, 5, 6, 6, 6]])
        jitter = torch.tensor([[0, 0, 0, -1, -1, -1]])
        context_config = {
            'dimensionality': {'expected_dimensionality': 3},
            'image_dimensions': torch.Size([10, 10, 10])
        }
        result = apply_jitter(bbox, jitter, context_config)
        assert torch.equal(result, bbox)

    def test_3d_jitter_collapsed_dim_fallback(self):
        """3D: Collapsed dimension (3D expected) triggers fallback for that dim."""
        bbox = torch.tensor([[0, 0, 0, 5, 5, 3]])
        jitter = torch.tensor([[0, 0, 3, 0, 0, 0]])
        context_config = {
            'dimensionality': {'expected_dimensionality': 3},
            'image_dimensions': torch.Size([10, 10, 10])
        }
        result = apply_jitter(bbox, jitter, context_config)
        assert torch.equal(result, bbox)

    # ============================================================================
    # 2D bbox tests
    # ============================================================================

    def test_2d_non_problematic_jitter(self):
        """2D: Non-problematic jitter (collapsed dim unchanged) keeps bbox valid."""
        bbox = torch.tensor([[0, 0, 5, 5, 5, 5]])
        jitter = torch.tensor([[1, 1, 0, 1, 1, 0]])
        context_config = {
            'dimensionality': {
                'expected_dimensionality': 2,
                'collapsed_dimension': 2
            },
            'image_dimensions': torch.Size([10, 10, 10])
        }
        result = apply_jitter(bbox, jitter, context_config)
        expected = torch.tensor([[1, 1, 5, 6, 6, 5]])
        assert torch.equal(result, expected)

    def test_2d_jitter_min_greater_than_max_fallback(self):
        """2D: min > max triggers fallback for that dimension only."""
        bbox = torch.tensor([[5, 0, 5, 10, 5, 5]])
        jitter = torch.tensor([[10, 1, 0, 0, 1, 0]])
        context_config = {
            'dimensionality': {
                'expected_dimensionality': 2,
                'collapsed_dimension': 2
            },
            'image_dimensions': torch.Size([10, 10, 10])
        }
        result = apply_jitter(bbox, jitter, context_config)
        expected = torch.tensor([[5, 1, 5, 10, 6, 5]])
        assert torch.equal(result, expected)

    def test_2d_jitter_negative_coordinates_fallback(self):
        """2D: Negative coordinate triggers fallback for that dimension only."""
        bbox = torch.tensor([[1, 0, 5, 5, 5, 5]])
        jitter = torch.tensor([[-5, 0, 0, 0, 0, 0]])
        context_config = {
            'dimensionality': {
                'expected_dimensionality': 2,
                'collapsed_dimension': 2
            },
            'image_dimensions': torch.Size([10, 10, 10])
        }
        result = apply_jitter(bbox, jitter, context_config)
        expected = torch.tensor([[1, 0, 5, 5, 5, 5]])
        assert torch.equal(result, expected)

    def test_2d_jitter_exceeds_bounds_fallback(self):
        """2D: Exceeding image bounds triggers fallback for that dimension only."""
        bbox = torch.tensor([[0, 0, 5, 8, 8, 5]])
        jitter = torch.tensor([[1, 0, 0, 5, 0, 0]])
        context_config = {
            'dimensionality': {
                'expected_dimensionality': 2,
                'collapsed_dimension': 2
            },
            'image_dimensions': torch.Size([10, 10, 10])
        }
        result = apply_jitter(bbox, jitter, context_config)
        expected = torch.tensor([[0, 0, 5, 8, 8, 5]])
        assert torch.equal(result, expected)

    def test_2d_jitter_non_collapsed_dim_collapses_fallback(self):
        """2D: Non-collapsed dim collapsing (i.e. a collapse in an unexpected axis) 
        triggers fallback for that dim."""
        bbox = torch.tensor([[0, 0, 5, 5, 5, 5]])
        jitter = torch.tensor([[1, 5, 0, 0, 0, 0]])
        context_config = {
            'dimensionality': {
                'expected_dimensionality': 2,
                'collapsed_dimension': 2
            },
            'image_dimensions': torch.Size([10, 10, 10])
        }
        result = apply_jitter(bbox, jitter, context_config)
        expected = torch.tensor([[1, 0, 5, 5, 5, 5]])
        assert torch.equal(result, expected)

    def test_2d_jitter_collapsed_dim_uncollapses_fallback(self):
        """2D: Collapsed dim uncollapsing triggers fallback; other dims keep jitter."""
        bbox = torch.tensor([[0, 0, 5, 5, 5, 5]])
        jitter = torch.tensor([[1, 1, 1, 1, 1, 2]])
        context_config = {
            'dimensionality': {
                'expected_dimensionality': 2,
                'collapsed_dimension': 2
            },
            'image_dimensions': torch.Size([10, 10, 10])
        }
        result = apply_jitter(bbox, jitter, context_config)
        expected = torch.tensor([[1, 1, 5, 6, 6, 5]])
        assert torch.equal(result, expected)

    def test_2d_jitter_point_bbox_fallback(self):
        """2D: Point bbox triggers full fallback (all dimensions)."""
        bbox = torch.tensor([[5, 5, 5, 6, 6, 5]])
        jitter = torch.tensor([[0, 0, 0, -1, -1, 0]])
        context_config = {
            'dimensionality': {
                'expected_dimensionality': 2,
                'collapsed_dimension': 2
            },
            'image_dimensions': torch.Size([10, 10, 10])
        }
        result = apply_jitter(bbox, jitter, context_config)
        assert torch.equal(result, bbox)

    # ============================================================================
    # Multi-failure tests (different failure modes in different dims)
    # ============================================================================

    def test_3d_jitter_multiple_failures_different_axes(self):
        """3D: Negative coord in x and min>max in y; only those dims fall back."""
        bbox = torch.tensor([[1, 5, 0, 5, 10, 5]])
        jitter = torch.tensor([[-5, 10, 1, 0, 3, 1]])
        context_config = {
            'dimensionality': {'expected_dimensionality': 3},
            'image_dimensions': torch.Size([20, 20, 20])
        }
        result = apply_jitter(bbox, jitter, context_config)
        expected = torch.tensor([[1, 5, 1, 5, 10, 6]])
        assert torch.equal(result, expected)

    def test_3d_jitter_negative_and_exceeds_bounds(self):
        """3D: Negative coord in x and exceeds bounds in y; only those dims fall back."""
        bbox = torch.tensor([[1, 0, 0, 5, 8, 5]])
        jitter = torch.tensor([[-5, 3, 1, 0, 5, 1]])
        context_config = {
            'dimensionality': {'expected_dimensionality': 3},
            'image_dimensions': torch.Size([10, 10, 10])
        }
        result = apply_jitter(bbox, jitter, context_config)
        expected = torch.tensor([[1, 0, 1, 5, 8, 6]])
        assert torch.equal(result, expected)

    def test_2d_jitter_multiple_failures_different_axes(self):
        """2D: Negative coord in x and min>max in y; collapsed dim unchanged."""
        bbox = torch.tensor([[1, 5, 5, 5, 10, 5]])
        jitter = torch.tensor([[-5, 10, 0, 0, 0, 0]])
        context_config = {
            'dimensionality': {
                'expected_dimensionality': 2,
                'collapsed_dimension': 2
            },
            'image_dimensions': torch.Size([20, 20, 20])
        }
        result = apply_jitter(bbox, jitter, context_config)
        expected = torch.tensor([[1, 5, 5, 5, 10, 5]])
        assert torch.equal(result, expected)

    def test_3d_jitter_multiple_failures_same_axis(self):
        """3D: exceeds bounds and min>max both in x; only x falls back."""
        bbox = torch.tensor([[1, 5, 0, 5, 10, 5]])
        jitter = torch.tensor([[20, 10, 1, 0, 0, 0]]) #x_min has both negative coord and min>max issues
        context_config = {
            'dimensionality': {'expected_dimensionality': 3},
            'image_dimensions': torch.Size([20, 20, 20])
        }
        result = apply_jitter(bbox, jitter, context_config)
        expected = torch.tensor([[1, 5, 1, 5, 10, 5]])
        assert torch.equal(result, expected)
    # ============================================================================
    # Config validation tests
    # ============================================================================

    def test_missing_dimensionality_key(self):
        """Should error when context_config lacks 'dimensionality'."""
        bbox = torch.tensor([[0, 0, 0, 5, 5, 5]])
        jitter = torch.tensor([[1, 1, 1, 1, 1, 1]])
        context_config = {'image_dimensions': torch.Size([10, 10, 10])}
        with pytest.raises(AssertionError, match="Context config must contain the key 'dimensionality'"):
            apply_jitter(bbox, jitter, context_config)

    def test_missing_image_dimensions(self):
        """Should error when context_config lacks 'image_dimensions'."""
        bbox = torch.tensor([[0, 0, 0, 5, 5, 5]])
        jitter = torch.tensor([[1, 1, 1, 1, 1, 1]])
        context_config = {'dimensionality': {'expected_dimensionality': 3}}
        with pytest.raises(AssertionError, match= "Context config must contain the key 'image_dimensions' to specify the dimensions of the image volume which the bbox is situated in."):
            apply_jitter(bbox, jitter, context_config)

    def test_invalid_expected_dimensionality(self):
        """Should error when expected_dimensionality is not 2 or 3."""
        bbox = torch.tensor([[0, 0, 0, 5, 5, 5]])
        jitter = torch.tensor([[1, 1, 1, 1, 1, 1]])
        context_config = {
            'dimensionality': {'expected_dimensionality': 4},
            'image_dimensions': torch.Size([10, 10, 10])
        }
        with pytest.raises(AssertionError, match="Context config 'expected_dimensionality' must be either 2 or 3."):
            apply_jitter(bbox, jitter, context_config)

    def test_2d_missing_collapsed_dimension(self):
        """2D: Should error when collapsed_dimension is not provided."""
        bbox = torch.tensor([[0, 0, 5, 5, 5, 5]])
        jitter = torch.tensor([[1, 1, 0, 1, 1, 0]])
        context_config = {
            'dimensionality': {'expected_dimensionality': 2},
            'image_dimensions': torch.Size([10, 10, 10])
        }
        with pytest.raises(AssertionError, match="Context config must contain the key 'collapsed_dimension' to specify which dimension is collapsed for 2D bounding box generation."):
            apply_jitter(bbox, jitter, context_config)

    def test_2d_invalid_collapsed_dimension(self):
        """2D: Should error when collapsed_dimension is not 0, 1, or 2."""
        bbox = torch.tensor([[0, 0, 5, 5, 5, 5]])
        jitter = torch.tensor([[1, 1, 0, 1, 1, 0]])
        context_config = {
            'dimensionality': {
                'expected_dimensionality': 2,
                'collapsed_dimension': 5
            },
            'image_dimensions': torch.Size([10, 10, 10])
        }
        with pytest.raises(AssertionError):
            apply_jitter(bbox, jitter, context_config)


# ============================================================================
# Test jitter_bbox
# ============================================================================

class TestJitterBbox:
    """Tests for the wrapper which generates and applies jitter to bounding boxes."""
    
    def test_jitter_bbox_3d(self):
        """Test jittering a 3D bbox."""
        bbox = torch.tensor([[0, 0, 0, 5, 5, 5]])
        jitter_config = {
            'dimensionality': {
                'expected_dimensionality': 3
            },
            'jitter_config': {
                'type': 'absolute',
                'sampling_mechanism': 'uniform_integer',
                'jitter_symmetric': True,
            },
            'jitter_parameterisation': [2, 2, 2]
        }
        context_config = {
            'expected_dimensionality': 3,
            'sampling_dimensions': torch.Size([10, 10, 10]),
            'bbox_extrema': bbox,
            'image_dimensions': torch.Size([10, 10, 10])
        }
        
        result = jitter_bbox(bbox, jitter_config, context_config)
        assert result.shape == (1, 6)
        valid, _ = check_bbox_validity(result, {
            'dimensionality': {'expected_dimensionality': 3},
            'image_dimensions': torch.Size([10, 10, 10])
        })
        assert valid
    
    def test_jitter_bbox_2d(self):
        """Test jittering a 2D bbox."""
        bbox = torch.tensor([[0, 0, 5, 5, 5, 5]])
        jitter_config = {
            'dimensionality': {
                'expected_dimensionality': 2
            },
            'jitter_config': {
                'type': 'absolute',
                'sampling_mechanism': 'uniform_integer',
                'jitter_symmetric': True,
            },
            'jitter_parameterisation': [2, 2]
        }
        context_config = {
            'expected_dimensionality': 2,
            'sampling_dimensions': torch.Size([10, 10, 0]),
            'bbox_extrema': bbox,
            'collapsed_dim': 2,
            'image_dimensions': torch.Size([10, 10, 10])
        }
        
        result = jitter_bbox(bbox, jitter_config, context_config)
        assert result.shape == (1, 6)
        valid, _ = check_bbox_validity(result, {
            'dimensionality': {
                'expected_dimensionality': 2,
                'collapsed_dimension': 2
            },
            'image_dimensions': torch.Size([10, 10, 10])
        })
        assert valid
        assert result[0, 2] == result[0, 5]
    
    def test_jitter_bbox_2d_missing_collapsed_dim(self):
        """2D: Should error when collapsed_dim is missing from context_config."""
        bbox = torch.tensor([[0, 0, 5, 5, 5, 5]])
        jitter_config = {
            'dimensionality': {'expected_dimensionality': 2},
            'jitter_config': {
                'type': 'absolute',
                'sampling_mechanism': 'uniform_integer',
                'jitter_symmetric': True,
            },
            'jitter_parameterisation': [2, 2]
        }
        context_config = {
            'expected_dimensionality': 2,
            'sampling_dimensions': torch.Size([10, 10, 0]),
            'bbox_extrema': bbox,
            'image_dimensions': torch.Size([10, 10, 10])
        }
        with pytest.raises(AssertionError, match="Context config must contain 'collapsed_dim' key"):
            jitter_bbox(bbox, jitter_config, context_config)

    def test_jitter_bbox_missing_expected_dimensionality(self):
        """Should error when expected_dimensionality is missing from context_config."""
        bbox = torch.tensor([[0, 0, 0, 5, 5, 5]])
        jitter_config = {
            'dimensionality': {'expected_dimensionality': 3},
            'jitter_config': {
                'type': 'absolute',
                'sampling_mechanism': 'uniform_integer',
                'jitter_symmetric': True,
            },
            'jitter_parameterisation': [2, 2, 2]
        }
        context_config = {
            'sampling_dimensions': torch.Size([10, 10, 10]),
            'bbox_extrema': bbox,
            'image_dimensions': torch.Size([10, 10, 10])
        }
        with pytest.raises(KeyError):
            jitter_bbox(bbox, jitter_config, context_config)

    def test_jitter_bbox_missing_sampling_dimensions(self):
        """Should error when sampling_dimensions is missing from context_config."""
        bbox = torch.tensor([[0, 0, 0, 5, 5, 5]])
        jitter_config = {
            'dimensionality': {'expected_dimensionality': 3},
            'jitter_config': {
                'type': 'absolute',
                'sampling_mechanism': 'uniform_integer',
                'jitter_symmetric': True,
            },
            'jitter_parameterisation': [2, 2, 2]
        }
        context_config = {
            'expected_dimensionality': 3,
            'bbox_extrema': bbox,
            'image_dimensions': torch.Size([10, 10, 10])
        }
        with pytest.raises(KeyError):
            jitter_bbox(bbox, jitter_config, context_config)

    def test_jitter_bbox_missing_bbox_extrema(self):
        """Should error when bbox_extrema is missing from context_config."""
        bbox = torch.tensor([[0, 0, 0, 5, 5, 5]])
        jitter_config = {
            'dimensionality': {'expected_dimensionality': 3},
            'jitter_config': {
                'type': 'absolute',
                'sampling_mechanism': 'uniform_integer',
                'jitter_symmetric': True,
            },
            'jitter_parameterisation': [2, 2, 2]
        }
        context_config = {
            'expected_dimensionality': 3,
            'sampling_dimensions': torch.Size([10, 10, 10]),
            'image_dimensions': torch.Size([10, 10, 10])
        }
        with pytest.raises(KeyError):
            jitter_bbox(bbox, jitter_config, context_config)

    def test_jitter_bbox_missing_image_dimensions(self):
        """Should error when image_dimensions is missing from context_config."""
        bbox = torch.tensor([[0, 0, 0, 5, 5, 5]])
        jitter_config = {
            'dimensionality': {'expected_dimensionality': 3},
            'jitter_config': {
                'type': 'absolute',
                'sampling_mechanism': 'uniform_integer',
                'jitter_symmetric': True,
            },
            'jitter_parameterisation': [2, 2, 2]
        }
        context_config = {
            'expected_dimensionality': 3,
            'sampling_dimensions': torch.Size([10, 10, 10]),
            'bbox_extrema': bbox,
        }
        with pytest.raises(KeyError):
            jitter_bbox(bbox, jitter_config, context_config)

    def test_jitter_bbox_missing_dimensionality_in_jitter_config(self):
        """Should error when dimensionality is missing from jitter_config."""
        bbox = torch.tensor([[0, 0, 0, 5, 5, 5]])
        jitter_config = {
            'jitter_config': {
                'type': 'absolute',
                'sampling_mechanism': 'uniform_integer',
                'jitter_symmetric': True,
            },
            'jitter_parameterisation': [2, 2, 2]
        }
        context_config = {
            'expected_dimensionality': 3,
            'sampling_dimensions': torch.Size([10, 10, 10]),
            'bbox_extrema': bbox,
            'image_dimensions': torch.Size([10, 10, 10])
        }
        with pytest.raises(KeyError, match="Sampling config must contain the key 'dimensionality'"):
            jitter_bbox(bbox, jitter_config, context_config)

    def test_jitter_bbox_missing_jitter_config_key(self):
        """Should error when jitter_config key is missing from jitter_config."""
        bbox = torch.tensor([[0, 0, 0, 5, 5, 5]])
        jitter_config = {
            'dimensionality': {'expected_dimensionality': 3},
            'jitter_parameterisation': [2, 2, 2]
        }
        context_config = {
            'expected_dimensionality': 3,
            'sampling_dimensions': torch.Size([10, 10, 10]),
            'bbox_extrema': bbox,
            'image_dimensions': torch.Size([10, 10, 10])
        }
        with pytest.raises(KeyError, match="Sampling config must contain the key 'jitter_config'"):
            jitter_bbox(bbox, jitter_config, context_config)

    def test_jitter_bbox_missing_jitter_parameterisation(self):
        """Should error when jitter_parameterisation is missing from jitter_config."""
        bbox = torch.tensor([[0, 0, 0, 5, 5, 5]])
        jitter_config = {
            'dimensionality': {'expected_dimensionality': 3},
            'jitter_config': {
                'type': 'absolute',
                'sampling_mechanism': 'uniform_integer',
                'jitter_symmetric': True,
            },
        }
        context_config = {
            'expected_dimensionality': 3,
            'sampling_dimensions': torch.Size([10, 10, 10]),
            'bbox_extrema': bbox,
            'image_dimensions': torch.Size([10, 10, 10])
        }
        with pytest.raises(KeyError, match="Sampling config must contain the key 'jitter_parameterisation'"):
            jitter_bbox(bbox, jitter_config, context_config)


# ============================================================================
# Test extract_connected_components
# ============================================================================
class TestExtractConnectedComponents:
    """Tests for connected component extraction."""
    
    def test_extract_2d_components_1_hop(self):
        """Test 2D connected component extraction."""
        mask = torch.zeros(10, 10)
        mask[0:3, 0:3] = 1
        mask[5:8, 5:8] = 1  # Two separate components in 1 hop.
        mask[8,8] = 1 #Third component when we use 1 hop connectivity, but not when we use 2 hop connectivity.
        
        labeled = extract_connected_components(mask, orthogonal_hops=1)
        
        assert labeled.shape == mask.shape
        assert labeled.max() == 3  # Three components
        assert 0 in labeled  # Background
    
    def test_extract_2d_components_2_hop(self):
        """Test 2D connected component extraction."""
        mask = torch.zeros(10, 10)
        mask[0:3, 0:3] = 1
        mask[5:8, 5:8] = 1  # Two separate components in 1 hop.
        mask[8,8] = 1 #Second component when we use 2 hop connectivity.
        
        labeled = extract_connected_components(mask, orthogonal_hops=2)
        
        assert labeled.shape == mask.shape
        assert labeled.max() == 2  # Two components
        assert 0 in labeled  # Background

    def test_extract_3d_components_1_hop(self):
        """Test 3D connected component extraction."""
        mask = torch.zeros(10, 10, 10)
        mask[0:3, 0:3, 0:3] = 1
        mask[5:8, 5:8, 5:8] = 1  # Two separate components
        mask[3,3,2] = 1 #Third component when we use 1 hop connectivity, would merge with the first component
        #in 2 hop connectivity. 
        mask[8,8,8] = 1 #Fourth component when we use 1 hop connectivity, would merge with second component
        #in 3 hop connectivity.
        labeled = extract_connected_components(mask, orthogonal_hops=1)
        
        assert labeled.shape == mask.shape
        assert labeled.max() == 4 # Four components
    
    def test_extract_3d_components_2_hop(self):
        """Test 3D connected component extraction."""
        mask = torch.zeros(10, 10, 10)
        mask[0:3, 0:3, 0:3] = 1
        mask[5:8, 5:8, 5:8] = 1  # Two separate components
        mask[3,3,2] = 1 #Merges with 1st component when we use 2 hop connectivity.
        mask[8,8,8] = 1 #Third component when we use 1 & 2 hop connectivity, would merge with second component
        #in 3 hop connectivity.
        labeled = extract_connected_components(mask, orthogonal_hops=2)
        
        assert labeled.shape == mask.shape
        assert labeled.max() == 3 # Three components
    
    def test_extract_3d_components_3_hop(self):
        """Test 3D connected component extraction."""
        mask = torch.zeros(10, 10, 10)
        mask[0:3, 0:3, 0:3] = 1
        mask[5:8, 5:8, 5:8] = 1  # Two separate components
        mask[3,3,2] = 1 #Merged with 1st component when we use 2 hop connectivity.
        mask[8,8,8] = 1 #Merged with 2nd component when we use 3 hop connectivity
        labeled = extract_connected_components(mask, orthogonal_hops=3)
        
        assert labeled.shape == mask.shape
        assert labeled.max() == 2 # Two components

    def test_extract_invalid_connectivity(self):
        """Test with invalid connectivity for 2D."""
        mask = torch.zeros(10, 10)
        mask[0:3, 0:3] = 1
        
        with pytest.raises(ValueError, match="Orthogonal hops"):
            extract_connected_components(mask, orthogonal_hops=3)  # Max is 2 for 2D
    
    def test_extract_invalid_connectivity_3d(self):
        """Test with invalid connectivity for 3D."""
        mask = torch.zeros(10, 10, 10)
        mask[0:3, 0:3, 0:3] = 1
        
        with pytest.raises(ValueError, match="Orthogonal hops"):
            extract_connected_components(mask, orthogonal_hops=4)  # Max is 3 for 3D

    def test_extract_hops_zero_2d(self):
        """2D: Should error when orthogonal_hops is 0."""
        mask = torch.zeros(10, 10)
        mask[0:3, 0:3] = 1
        with pytest.raises(ValueError, match="Orthogonal hops must be at least 1"):
            extract_connected_components(mask, orthogonal_hops=0)

    def test_extract_hops_negative_2d(self):
        """2D: Should error when orthogonal_hops is negative."""
        mask = torch.zeros(10, 10)
        mask[0:3, 0:3] = 1
        with pytest.raises(ValueError, match="Orthogonal hops must be at least 1"):
            extract_connected_components(mask, orthogonal_hops=-1)

    def test_extract_hops_zero_3d(self):
        """3D: Should error when orthogonal_hops is 0."""
        mask = torch.zeros(10, 10, 10)
        mask[0:3, 0:3, 0:3] = 1
        with pytest.raises(ValueError, match="Orthogonal hops must be at least 1"):
            extract_connected_components(mask, orthogonal_hops=0)

    def test_extract_hops_negative_3d(self):
        """3D: Should error when orthogonal_hops is negative."""
        mask = torch.zeros(10, 10, 10)
        mask[0:3, 0:3, 0:3] = 1
        with pytest.raises(ValueError, match="Orthogonal hops must be at least 1"):
            extract_connected_components(mask, orthogonal_hops=-1)

    def test_extract_invalid_mask_dim_1d(self):
        """Should error when mask is 1D."""
        mask = torch.zeros(10)
        with pytest.raises(ValueError, match="Only 2D and 3D masks"):
            extract_connected_components(mask, orthogonal_hops=1)

    def test_extract_invalid_mask_dim_4d(self):
        """Should error when mask is 4D."""
        mask = torch.zeros(3, 3, 3, 3)
        with pytest.raises(ValueError, match="Only 2D and 3D masks"):
            extract_connected_components(mask, orthogonal_hops=1)

    def test_extract_empty_mask_2d(self):
        """2D: Should error on empty mask."""
        mask = torch.zeros(10, 10)
        with pytest.raises(ValueError, match="No connected components"):
            extract_connected_components(mask, orthogonal_hops=1)

    def test_extract_2d_single_row(self):
        """2D: Mask with a single row (dim size 1)."""
        mask = torch.zeros(1, 10)
        mask[0, 2:5] = 1
        labeled = extract_connected_components(mask, orthogonal_hops=1)
        assert labeled.shape == mask.shape
        assert labeled.max() == 1
        assert torch.all(labeled[0, 2:5] == 1)

    def test_extract_2d_single_col(self):
        """2D: Mask with a single column (dim size 1)."""
        mask = torch.zeros(10, 1)
        mask[2:5, 0] = 1
        labeled = extract_connected_components(mask, orthogonal_hops=1)
        assert labeled.shape == mask.shape
        assert labeled.max() == 1

    def test_extract_2d_single_voxel(self):
        """2D: Mask of shape (1, 1) with foreground."""
        mask = torch.ones(1, 1)
        labeled = extract_connected_components(mask, orthogonal_hops=1)
        assert labeled.shape == (1, 1)
        assert labeled.max() == 1
        assert labeled[0, 0] == 1

    def test_extract_3d_single_slice(self):
        """3D: Mask with a single slice along first dim (size 1)."""
        mask = torch.zeros(1, 10, 10)
        mask[0, 2:5, 2:5] = 1
        labeled = extract_connected_components(mask, orthogonal_hops=1)
        assert labeled.shape == mask.shape
        assert labeled.max() == 1

    def test_extract_3d_thin_volume(self):
        """3D: Mask with dim of size 1 in middle dimension."""
        mask = torch.zeros(10, 1, 10)
        mask[2:5, 0, 2:5] = 1
        labeled = extract_connected_components(mask, orthogonal_hops=1)
        assert labeled.shape == mask.shape
        assert labeled.max() == 1

    def test_extract_3d_single_voxel_volume(self):
        """3D: Mask of shape (1, 1, 1) with foreground."""
        mask = torch.ones(1, 1, 1)
        labeled = extract_connected_components(mask, orthogonal_hops=1)
        assert labeled.shape == (1, 1, 1)
        assert labeled.max() == 1

    def test_extract_no_components(self):
        """Test with empty mask."""
        mask = torch.zeros(10, 10, 10)
        
        with pytest.raises(ValueError, match="No connected components"):
            extract_connected_components(mask, orthogonal_hops=3)

    def test_extract_2d_symmetric_cavity(self):
        """2D max-hop: shell with a symmetric cavity is one component."""
        mask = torch.ones(7, 7)
        mask[2:5, 2:5] = 0
        labeled = extract_connected_components(mask, orthogonal_hops=2)
        assert labeled.shape == mask.shape
        assert labeled.max() == 1
        assert torch.all(labeled[2:5, 2:5] == 0)

    def test_extract_2d_asymmetric_cavity(self):
        """2D max-hop: shell with an asymmetric cavity is one component."""
        mask = torch.ones(9, 9)
        mask[3:6, 2:7] = 0
        labeled = extract_connected_components(mask, orthogonal_hops=2)
        assert labeled.shape == mask.shape
        assert labeled.max() == 1
        assert torch.all(labeled[3:6, 2:7] == 0)

    def test_extract_2d_cavity_with_inner_component(self):
        """2D max-hop: foreground inside cavity is a separate component."""
        mask = torch.ones(9, 9)
        mask[2:8, 2:8] = 0
        mask[5, 5] = 1
        labeled = extract_connected_components(mask, orthogonal_hops=2)
        assert labeled.shape == mask.shape
        assert labeled.max() == 2
        assert labeled[5, 5] != labeled[2,2]
        inner_cavity_mask = labeled[2:8, 2:8]
        #We are going to invert the cavity, and treat it like a boolean.
        expected_cavity = torch.zeros(6, 6)
        expected_cavity[3,3] = 1
        expected_cavity = expected_cavity.bool()
        assert torch.all(inner_cavity_mask.bool() == expected_cavity)

    def test_extract_2d_pretzel_1_hop(self):
        """2D 1-hop: diagonal bridge ."""
        mask = torch.ones(7, 7)
        mask[1:6, 1:6] = 0 #We empty the middle.
        mask[list(range(7)), list(range(7))] = 1 #We add a diagonal bridge.
        labeled = extract_connected_components(mask, orthogonal_hops=1)
        assert labeled.shape == mask.shape
        assert labeled.max() == mask.shape[0] -3 #Each diagonal element is a separate component in 1 hop connectivity
        #except for the first and last connected to the corners. Hence, shape[0] - 2 * 2 (diagonal elements) + 1 for the border.
        assert torch.all(torch.unique(labeled[list(range(2,5)), list(range(2,5))]) == torch.tensor(list(range(2,5))))
        #We need to assert that the diagonal bridge is all unique labels, (excluding the ones touching the border)

    def test_extract_2d_pretzel_max_hop(self):
        """2D max-hop: diagonal bridge connected """
        mask = torch.ones(7, 7)
        mask[1:7, 1:7] = 0 #We empty the middle.
        mask[list(range(7)), list(range(7))] = 1 #We add a diagonal bridge
        labeled = extract_connected_components(mask, orthogonal_hops=2)
        assert labeled.shape == mask.shape
        assert labeled.max() == 1
        assert torch.all(labeled.bool() == mask.bool()) #The middle should be connected to the border, 
        #as we are using max connectivity.


    def test_extract_3d_symmetric_cavity(self):
        """3D max-hop: shell with a symmetric cavity is one component."""
        mask = torch.ones(7, 7, 7)
        mask[2:5, 2:5, 2:5] = 0
        labeled = extract_connected_components(mask, orthogonal_hops=3)
        assert labeled.shape == mask.shape
        assert labeled.max() == 1
        assert torch.all(labeled.bool() == mask.bool())

    def test_extract_3d_asymmetric_cavity(self):
        """3D max-hop: shell with an asymmetric cavity is one component."""
        mask = torch.ones(9, 9, 9)
        mask[3:6, 2:7, 3:6] = 0
        labeled = extract_connected_components(mask, orthogonal_hops=3)
        assert labeled.shape == mask.shape
        assert labeled.max() == 1
        assert torch.all(labeled.bool() == mask.bool())

    def test_extract_3d_cavity_with_inner_component(self):
        """3D max-hop: foreground inside cavity is a separate component."""
        mask = torch.ones(9, 9, 9)
        mask[2:8, 2:8, 2:8] = 0
        mask[5, 5, 5] = 1
        labeled = extract_connected_components(mask, orthogonal_hops=3)
        assert labeled.shape == mask.shape
        assert labeled.max() == 2
        assert labeled[0, 0, 0] != labeled[5, 5, 5]
        assert torch.all(labeled.bool() == mask.bool())

    def test_extract_3d_pretzel_1_hop(self):
        """3D 1-hop: diagonal bridge does not connect blocks."""
        mask = torch.ones(9,9,9)
        mask[1:8, 1:8, 1:8] = 0 #We empty the middle.
        #Now insert a diagonal.
        mask[list(range(9)), list(range(9)), list(range(9))] = 1
        labeled = extract_connected_components(mask, orthogonal_hops=1)
        assert labeled.shape == mask.shape
        assert labeled.max() == mask.shape[0] - 3
        #Same rationale as before, except now in 3d. -2 * 2 on each end of the diagonal interior to the bordre,
        # and + 1 for the border.
        assert torch.all(torch.unique(labeled[list(range(2,7)), list(range(2,7)), list(range(2,7))]) == torch.tensor(list(range(2,7))))
        
    def test_extract_3d_pretzel_2_hop(self):
        """3D 2-hop: corner bridge (3 hops) still does not connect blocks."""
        mask = torch.ones(9,9,9)
        mask[1:8, 1:8, 1:8] = 0 #We empty the middle.
        #Now insert a diagonal.
        mask[list(range(9)), list(range(9)), list(range(9))] = 1
        labeled = extract_connected_components(mask, orthogonal_hops=2)
        assert labeled.shape == mask.shape
        assert labeled.max() == mask.shape[0] - 3
        assert torch.all(torch.unique(labeled[list(range(2,7)), list(range(2,7)), list(range(2,7))]) == torch.tensor(list(range(2,7))))
        #Still not connected! 
        
    def test_extract_3d_pretzel_max_hop(self):
        """3D max-hop: corner bridge connects all three blobs into one."""
        mask = torch.ones(9,9,9)
        mask[1:8, 1:8, 1:8] = 0 #We empty the middle.
        #Now insert a diagonal.
        mask[list(range(9)), list(range(9)), list(range(9))] = 1
        labeled = extract_connected_components(mask, orthogonal_hops=3)
        assert labeled.shape == mask.shape
        assert labeled.max() == 1
        #Should all be connected now, as we are using max connectivity.
        assert torch.all(labeled.bool() == mask.bool())

# ============================================================================
# Test filter_valid_components
# ============================================================================

class TestFilterValidComponents:
    """Tests for filtering components by size."""
    
    def test_filter_valid_components_2d(self):
        """Test filtering valid 2D components."""
        # Create components with different sizes
        components = torch.zeros(10, 10, dtype=torch.int32)
        components[0:3, 0:3] = 1  # 9 voxels - valid
        components[5:6, 5:6] = 2  # 1 voxel - invalid (too small)
        
        filtered = filter_valid_components(components, 2)
        
        # Component 1 should be kept, component 2 should be filtered
        assert 1 in torch.unique(filtered)
        # Component 2 should be filtered out (become 0)
        assert 2 not in torch.unique(filtered)
    
    def test_filter_valid_components_3d(self):
        """Test filtering valid 3D components."""
        components = torch.zeros(10, 10, 10, dtype=torch.int32)
        components[2:8, 2:8, 2:8] = 1  # 216 voxels - valid
        components[9, 9, 9] = 2  # 1 voxel - invalid
        
        filtered = filter_valid_components(components, 3)
        
        assert 1 in torch.unique(filtered)
        assert 2 not in torch.unique(filtered)
    
    def test_filter_2d_horizontal_line(self):
        """2D: Horizontal line (1 row) — fails, only 1 unique y-coord."""
        components = torch.zeros(10, 10, dtype=torch.int32)
        components[3:8, 5] = 1  # 5x1 horizontal line
        filtered = filter_valid_components(components, 2)
        assert torch.all(filtered == 0)

    def test_filter_2d_vertical_line(self):
        """2D: Vertical line (1 col) — fails, only 1 unique x-coord."""
        components = torch.zeros(10, 10, dtype=torch.int32)
        components[5, 3:8] = 1  # 1x5 vertical line
        filtered = filter_valid_components(components, 2)
        assert torch.all(filtered == 0)

    def test_filter_2d_mixed_block_and_line(self):
        """2D: Valid block kept, horizontal line filtered."""
        components = torch.zeros(10, 10, dtype=torch.int32)
        components[0:3, 0:3] = 1  # 3x3 block — valid
        components[3:8, 5] = 2   # 5x1 line — invalid
        filtered = filter_valid_components(components, 2)
        assert 1 in torch.unique(filtered)
        assert 2 not in torch.unique(filtered)

    def test_filter_3d_plane(self):
        """3D: 2D plane (single z-slice) — fails, only 1 unique z-coord."""
        components = torch.zeros(10, 10, 10, dtype=torch.int32)
        components[2:7, 2:7, 5] = 1  # 5x5x1 plane
        filtered = filter_valid_components(components, 3)
        assert torch.all(filtered == 0)

    def test_filter_3d_line(self):
        """3D: 1D line (single row, single z-slice) — fails, < 2 unique coords in y and z."""
        components = torch.zeros(10, 10, 10, dtype=torch.int32)
        components[2:7, 5, 5] = 1  # 5x1x1 line along x
        filtered = filter_valid_components(components, 3)
        assert torch.all(filtered == 0)

    def test_filter_3d_mixed_block_and_plane(self):
        """3D: Valid block kept, 2D plane filtered."""
        components = torch.zeros(10, 10, 10, dtype=torch.int32)
        components[0:4, 0:4, 0:4] = 1  # 4x4x4 block — valid
        components[2:7, 2:7, 5] = 2    # 5x5x1 plane — invalid
        filtered = filter_valid_components(components, 3)
        assert 1 in torch.unique(filtered)
        assert 2 not in torch.unique(filtered)

    def test_diagonal_line_2d(self):
        '''
        Diagonal line should not fail, as it has multiple unique x and y coordinates.
        '''
        components = torch.zeros(10, 10, dtype=torch.int32)
        components[list(range(5)), list(range(5))] = 1  # Diagonal line from (0,0) to (4,4)
        filtered = filter_valid_components(components, 2)
        assert 1 in torch.unique(filtered)
    
    def test_diagonal_line_3d(self):
        '''
        Diagonal line in 3D should not fail, as it has multiple unique x, y, and z coordinates.
        '''
        components = torch.zeros(10, 10, 10, dtype=torch.int32)
        components[list(range(5)), list(range(5)), list(range(5))] = 1  # Diagonal line from (0,0,0) to (4,4,4)
        filtered = filter_valid_components(components, 3)
        assert 1 in torch.unique(filtered)

    def test_filter_all_invalid_2d(self):
        """Test when all components are invalid."""
        components = torch.zeros(10, 10, dtype=torch.int32)
        components[5, 5] = 1  # Single voxel - invalid
        components[6, 6] = 2  # Single voxel - invalid
        
        filtered = filter_valid_components(components, 2)
        
        # All should be filtered out
        assert torch.all(filtered == 0)

    def test_filter_all_invalid_3d(self):
        """Test when all components are invalid in 3D."""
        components = torch.zeros(10, 10, 10, dtype=torch.int32)
        components[5, 5, 5] = 1  # Single voxel - invalid
        components[6, 6, 6] = 2  # Single voxel - invalid

        filtered = filter_valid_components(components, 3)

        # All should be filtered out
        assert torch.all(filtered == 0)

    def test_filter_all_invalid_nontrivial_2d(self):
        """Test when all components are invalid, but non-trivial (e.g. lines)."""
        components = torch.zeros(10, 10, dtype=torch.int32)
        components[2:7, 5] = 1  # 5x1 horizontal line - invalid
        components[5, 2:7] = 2  # 1x5 vertical line - invalid
        
        filtered = filter_valid_components(components, 2)
        
        assert torch.all(filtered == 0)

    def test_filter_all_invalid_nontrivial_3d(self):
        """Test when all components are invalid, but non-trivial (e.g. lines)."""
        components = torch.zeros(10, 10, 10, dtype=torch.int32)
        components[2:7, 5, 5] = 1  # 5x1x1 line along x - invalid
        components[5, 2:7, 5] = 2  # 1x5x1 line along y - invalid

        filtered = filter_valid_components(components, 3)

        assert torch.all(filtered == 0)


# ============================================================================
# Test generate_components_from_mask
# ============================================================================

class TestGenerateComponentsFromMask:
    """Tests for combined component generation with pre-check.

    Pipeline: fast pre-check → CC extraction → filter_valid_components.
    Each stage can return (False, None).
    """

    # =========================================================================
    # Existing valid/invalid tests (keep as-is)
    # =========================================================================

    def test_generate_components_valid_2d(self):
        """2D: Valid 6x6 block with conn=1."""
        mask = torch.zeros(10, 10)
        mask[2:8, 2:8] = 1
        is_compatible, components = generate_components_from_mask(mask, 2, 1)
        assert is_compatible is True
        assert components is not None
        assert components.max() == 1


    def test_generate_components_valid_3d(self):
        """3D: Valid 6x6x6 block with conn=1."""
        mask = torch.zeros(10, 10, 10)
        mask[2:8, 2:8, 2:8] = 1
        is_compatible, components = generate_components_from_mask(mask, 3, 1)
        assert is_compatible is True
        assert components is not None
        assert components.max() == 1


    # =========================================================================
    # Stage 1: Fast pre-check failures
    # =========================================================================

    def test_generate_components_invalid_2d(self):
        """2D: Horizontal line fails fast check (only 1 y-coord)."""
        mask = torch.zeros(10, 10)
        mask[5, 2:8] = 1
        is_compatible, components = generate_components_from_mask(mask, 2, 1)
        assert is_compatible is False
        assert components is None

    def test_generate_components_invalid_3d_line(self):
        """3D: Line fails fast check (only 1 y- and z-coord)."""
        mask = torch.zeros(10, 10, 10)
        mask[2:8, 5, 5] = 1
        is_compatible, components = generate_components_from_mask(mask, 3, 1)
        assert is_compatible is False
        assert components is None
    
    def test_generate_components_invalid_3d(self):
        """3D: Plane fails fast check (only 1 z-coord)."""
        mask = torch.zeros(10, 10, 10)
        mask[5, 2:8, 2:8] = 1
        is_compatible, components = generate_components_from_mask(mask, 3, 1)
        assert is_compatible is False
        assert components is None

    def test_generate_components_empty_2d(self):
        """2D: Empty mask, fast check fails (< 2 voxels)."""
        mask = torch.zeros(10, 10)
        is_compatible, components = generate_components_from_mask(mask, 2, 1)
        assert is_compatible is False
        assert components is None

    def test_generate_components_empty_3d(self):
        """3D: Empty mask, fast check fails (< 2 voxels)."""
        mask = torch.zeros(10, 10, 10)
        is_compatible, components = generate_components_from_mask(mask, 3, 1)
        assert is_compatible is False
        assert components is None

    def test_generate_components_single_voxel_2d(self):
        """2D: Single voxel fails fast check (< 2 voxels)."""
        mask = torch.zeros(10, 10)
        mask[5, 5] = 1
        is_compatible, components = generate_components_from_mask(mask, 2, 1)
        assert is_compatible is False
        assert components is None

    def test_generate_components_single_voxel_3d(self):
        """3D: Single voxel fails fast check (< 2 voxels)."""
        mask = torch.zeros(10, 10, 10)
        mask[5, 5, 5] = 1
        is_compatible, components = generate_components_from_mask(mask, 3, 1)
        assert is_compatible is False
        assert components is None

    # =========================================================================
    # Stage 2: Fast check passes, but all components filtered
    #          (diagonal of isolated single-voxel components)
    # =========================================================================

    def test_generate_components_diagonal_2d_conn_1(self):
        """2D diagonal conn=1: isolated voxels, each filtered."""
        mask = torch.zeros(4, 4)
        for i in range(4):
            mask[i, i] = 1
        is_compatible, components = generate_components_from_mask(mask, 2, 1)
        assert is_compatible is False
        assert components is None

    def test_generate_components_diagonal_3d_conn_1(self):
        """3D diagonal conn=1: isolated voxels, each filtered."""
        mask = torch.zeros(4, 4, 4)
        for i in range(4):
            mask[i, i, i] = 1
        is_compatible, components = generate_components_from_mask(mask, 3, 1)
        assert is_compatible is False
        assert components is None

    def test_generate_components_diagonal_3d_conn_2(self):
        """3D diagonal conn=2: corner (1,1,1) is 3 hops, not in 18-conn."""
        mask = torch.zeros(4, 4, 4)
        for i in range(4):
            mask[i, i, i] = 1
        is_compatible, components = generate_components_from_mask(mask, 3, 2)
        assert is_compatible is False
        assert components is None

    # =========================================================================
    # Stage 3: Mixed — valid blob + separate line (filtered)
    #          Blob ensures fast pre-check passes; the line is its own
    #          component at extraction time and gets filtered out.
    # =========================================================================

    def test_generate_components_blob_and_line_2d(self):
        """2D: Blob valid, misaligned line is separate and filtered."""
        mask = torch.zeros(10, 10)
        mask[2:5, 2:5] = 1       # 3x3 valid blob
        mask[7:10, 7] = 1        # vertical line, only 1 x-coord
        is_compatible, components = generate_components_from_mask(mask, 2, 1)
        assert is_compatible is True
        assert components is not None
        assert components.max() == 1
        assert torch.all(components[7:10, 7] == 0)  # Line should be filtered out

    def test_generate_components_blob_and_line_3d(self):
        """3D: Blob valid, misaligned line is separate and filtered."""
        mask = torch.zeros(10, 10, 10)
        mask[2:5, 2:5, 2:5] = 1  # 3x3x3 valid blob
        mask[6:10, 7, 7] = 1     # line along x, only 1 y and 1 z
        is_compatible, components = generate_components_from_mask(mask, 3, 1)
        assert is_compatible is True
        assert components is not None
        assert components.max() == 1
        assert torch.all(components[6:10, 7, 7] == 0)  # Line should be filtered out

    # =========================================================================
    # Cavity / shell tests
    # =========================================================================

    def test_generate_components_shell_2d(self):
        """2D: Ring (hollow centre) is one valid component."""
        mask = torch.ones(7, 7)
        mask[2:5, 2:5] = 0
        is_compatible, components = generate_components_from_mask(mask, 2, 1)
        assert is_compatible is True
        assert components is not None
        assert components.max() == 1
        assert torch.all(components[2:5, 2:5] == 0)

    def test_generate_components_shell_3d(self):
        """3D: Shell (hollow centre) is one valid component."""
        mask = torch.ones(7, 7, 7)
        mask[2:5, 2:5, 2:5] = 0
        is_compatible, components = generate_components_from_mask(mask, 3, 1)
        assert is_compatible is True
        assert components is not None
        assert components.max() == 1
        assert torch.all(components[2:5, 2:5, 2:5] == 0)

    def test_generate_components_shell_with_inner_blob_2d(self):
        """2D: Shell valid, inner single-voxel filtered out."""
        mask = torch.ones(7, 7)
        mask[2:5, 2:5] = 0
        mask[3, 3] = 1
        is_compatible, components = generate_components_from_mask(mask, 2, 1)
        assert is_compatible is True
        assert components is not None
        assert components.max() == 1
        assert components[3, 3] == 0

    def test_generate_components_shell_with_inner_blob_3d(self):
        """3D: Shell valid, inner single-voxel filtered out."""
        mask = torch.ones(7, 7, 7)
        mask[2:5, 2:5, 2:5] = 0
        mask[3, 3, 3] = 1
        is_compatible, components = generate_components_from_mask(mask, 3, 1)
        assert is_compatible is True
        assert components is not None
        assert components.max() == 1
        assert components[3, 3, 3] == 0

    # =========================================================================
    # Pretzel tests (shell + diagonal bridge)
    # =========================================================================

    def test_generate_components_pretzel_2d_conn_1(self):
        """2D pretzel conn=1: interior diagonal voxels filtered, shell remains."""
        mask = torch.ones(7, 7)
        mask[1:6, 1:6] = 0
        mask[range(7), range(7)] = 1
        is_compatible, components = generate_components_from_mask(mask, 2, 1)
        assert is_compatible is True
        assert components is not None
        assert components.max() == 1
        assert torch.all(components[2:5, 2:5] == 0)
        # Diagonal bridge elements at (2,2), (3,3), (4,4) should be filtered out
        assert components[2, 2] == 0
        assert components[3, 3] == 0
        assert components[4, 4] == 0 #Making it explicit but was not necessary.

    def test_generate_components_pretzel_2d_conn_2(self):
        """2D pretzel conn=2: diagonal bridges via 8-conn, one component."""
        mask = torch.ones(7, 7)
        mask[1:6, 1:6] = 0
        mask[range(7), range(7)] = 1
        is_compatible, components = generate_components_from_mask(mask, 2, 2)
        assert is_compatible is True
        assert components is not None
        assert components.max() == 1
        assert torch.all(components.bool() == mask.bool())

    def test_generate_components_pretzel_3d_conn_1(self):
        """3D pretzel conn=1: interior diagonal voxels filtered, shell remains."""
        mask = torch.ones(9, 9, 9)
        mask[1:8, 1:8, 1:8] = 0
        mask[range(9), range(9), range(9)] = 1
        is_compatible, components = generate_components_from_mask(mask, 3, 1)
        assert is_compatible is True
        assert components is not None
        assert components.max() == 1
        assert torch.all(components[2:7, 2:7, 2:7] == 0)
        # Diagonal bridge elements at (2,2,2), (3,3,3), (4,4,4), (5,5,5), (6,6,6) should be filtered.
        assert components[2, 2, 2] == 0
        assert components[3, 3, 3] == 0
        assert components[4, 4, 4] == 0
        assert components[5, 5, 5] == 0
        assert components[6, 6, 6] == 0
        assert components[1,1,1] == 1
        assert components[7,7,7] == 1
        #diagonals attached to the shell should be connected so they should be 
        #labeled as 1, not 0.
        assert torch.all(components[:,:,0] == 1)
        assert torch.all(components[:,0,:] == 1)
        assert torch.all(components[0,:,:] == 1)
        assert torch.all(components[:,:,8] == 1)
        assert torch.all(components[:,8,:] == 1)
        assert torch.all(components[8,:,:] == 1)

    def test_generate_components_pretzel_3d_conn_2(self):
        """3D pretzel conn=2: interior diagonal voxels filtered, shell remains."""
        mask = torch.ones(9, 9, 9)
        mask[1:8, 1:8, 1:8] = 0
        mask[range(9), range(9), range(9)] = 1
        is_compatible, components = generate_components_from_mask(mask, 3, 2)
        assert is_compatible is True
        assert components is not None
        assert components.max() == 1
        assert torch.all(components[2:7, 2:7, 2:7] == 0)
        # Diagonal bridge elements at (2,2,2), (3,3,3), (4,4,4), (5,5,5), (6,6,6) should be filtered.
        assert components[2, 2, 2] == 0
        assert components[3, 3, 3] == 0
        assert components[4, 4, 4] == 0
        assert components[5, 5, 5] == 0
        assert components[6, 6, 6] == 0
        assert components[1,1,1] == 1
        assert components[7,7,7] == 1
        #diagonals attached to the shell should be connected so they should be 
        #labeled as 1, not 0.
        assert torch.all(components[:,:,0] == 1)
        assert torch.all(components[:,0,:] == 1)
        assert torch.all(components[0,:,:] == 1)
        assert torch.all(components[:,:,8] == 1)
        assert torch.all(components[:,8,:] == 1)
        assert torch.all(components[8,:,:] == 1)

    def test_generate_components_pretzel_planar_3d_conn_2(self):
        '''
        Not a diagonal, but rather a plane through the diagonal. Bijects the shell into two, but is one component!
        '''
        mask = torch.ones(9, 9, 9)
        mask[1:8, 1:8, 1:8] = 0
        for i in range(9):
            mask[i, i, :] = 1
        is_compatible, components = generate_components_from_mask(mask, 3, 2)
        assert is_compatible is True
        assert components is not None
        assert components.max() == 1
        assert torch.all(components.bool() == mask.bool())

    def test_generate_components_pretzel_3d_conn_3(self):
        """3D pretzel conn=3: 26-conn bridges, one component."""
        mask = torch.ones(9, 9, 9)
        mask[1:8, 1:8, 1:8] = 0
        mask[range(9), range(9), range(9)] = 1
        is_compatible, components = generate_components_from_mask(mask, 3, 3)
        assert is_compatible is True
        assert components is not None
        assert components.max() == 1
        assert torch.all(components.bool() == mask.bool())

    # =========================================================================
    # Config validation
    # =========================================================================

    def test_generate_components_invalid_dimensionality_value_error(self):
        """Should error when dimensionality is not 2 or 3 (ValueError branch)."""
        mask = torch.zeros(3, 3, 3, 3)
        with pytest.raises(ValueError, match=f"Unsupported dimensionality: 4. Must be 2 or 3."):
            generate_components_from_mask(mask, 4, 1)

# ============================================================================
# Test two_d_components_generation
# ============================================================================

class TestTwoDComponentsGeneration:
    """Tests for 2D component generation from slices."""
    
    def test_two_d_components_center_slice(self):
        """Test 2D component generation with center slice selection."""
        binary_mask = torch.zeros(10, 10, 10)
        binary_mask[2:7, 2:7, 4] = 1  # Slice 4
        binary_mask[3:6, 3:6, 5] = 1  # Slice 5 (has more coverage)
        binary_mask[2:7, 2:7, 6] = 1  # Slice 6
        
        slice_selection_config = {
            'slice_selection_strategy': 'center',
            'collapsed_dim': 2
        }
        
        components, slice_idx, is_compatible = two_d_components_generation(
            binary_mask, slice_selection_config, 1
        )
        
        assert is_compatible is True
        assert slice_idx == 5  # Center slice
        assert components.sum() > 0
        # Verify components are only non-zero at the selected slice
        assert components.ndim == 2
    
    def test_two_d_components_top_slice(self):
        """Test 2D component generation with top slice selection."""
        binary_mask = torch.zeros(10, 10, 10)
        binary_mask[2:7, 2:7, 1:8] = 1  # Top slice

        slice_selection_config = {
            'slice_selection_strategy': 'top',
            'collapsed_dim': 2
        }
        
        components, slice_idx, is_compatible = two_d_components_generation(
            binary_mask, slice_selection_config, 1
        )
        
        assert is_compatible is True
        assert slice_idx == 1  # Top slice
        assert components.sum() > 0
        assert components.ndim == 2
        
    def test_two_d_components_bottom_slice(self):

        """Test 2D component generation with bottom slice selection."""
        binary_mask = torch.zeros(10, 10, 10)
        binary_mask[2:7, 2:7, 1:8] = 1
        
        slice_selection_config = {
            'slice_selection_strategy': 'bottom',
            'collapsed_dim': 2
        }
        
        components, slice_idx, is_compatible = two_d_components_generation(
            binary_mask, slice_selection_config, 1
        )
        
        assert is_compatible is True
        assert slice_idx == 7  # Bottom slice
        assert components.sum() > 0
        assert components.ndim == 2
    
    def test_two_d_components_random_slice(self):
        """Test 2D component generation with random slice selection."""
        binary_mask = torch.zeros(10, 10, 10)
        binary_mask[2:7, 2:7, 3] = 1
        binary_mask[2:7, 2:7, 7] = 1
        
        slice_selection_config = {
            'slice_selection_strategy': 'random',
            'collapsed_dim': 2
        }
        
        components, slice_idx, is_compatible = two_d_components_generation(
            binary_mask, slice_selection_config, 1
        )
        
        assert is_compatible is True
        assert slice_idx in [3, 7]  # Only valid slices
        assert components.ndim == 2

    def test_two_d_components_skip_center_retry(self):
        """2D: Center strategy retries when the middle slice fails."""
        binary_mask = torch.zeros(10, 10, 10)
        binary_mask[5, 5, 2] = 1         # Slice 2: 1 voxel (fails)
        binary_mask[5, 5, 5] = 1         # Slice 5: 1 voxel (fails) — this is the first center
        binary_mask[2:7, 2:7, 7] = 1     # Slice 7: 5x5 block (succeeds)
        #Slice 5 is the first center, then it wold be slice 2 (floor). Both should fail and then
        #Slice 7 will be the next center and succeed.
        slice_selection_config = {
            'slice_selection_strategy': 'center',
            'collapsed_dim': 2
        }
        components, slice_idx, is_compatible = two_d_components_generation(
            binary_mask, slice_selection_config, 1
        )
        assert is_compatible is True
        assert slice_idx == 7


    def test_two_d_components_skip_bottom_retry(self):
        """2D: Bottom strategy retries when the last slice fails."""
        binary_mask = torch.zeros(10, 10, 10)
        binary_mask[2:7, 2:7, 5] = 1     # Slice 5: 5x5 block (succeeds)
        binary_mask[5, 5, 9] = 1         # Slice 9: 1 voxel (fails) — bottom picks this first
        slice_selection_config = {
            'slice_selection_strategy': 'bottom',
            'collapsed_dim': 2
        }
        components, slice_idx, is_compatible = two_d_components_generation(
            binary_mask, slice_selection_config, 1
        )
        assert is_compatible is True
        assert slice_idx == 5

    def test_two_d_components_skip_top_retry(self):
        """2D: Top strategy retries when the last slice fails."""
        binary_mask = torch.zeros(10, 10, 10)
        binary_mask[2:7, 2:7, 5] = 1     # Slice 5: 5x5 block (succeeds)
        binary_mask[5, 5, 3] = 1         # Slice 3: 1 voxel (fails) — top picks this first
        slice_selection_config = {
            'slice_selection_strategy': 'top',
            'collapsed_dim': 2
        }
        components, slice_idx, is_compatible = two_d_components_generation(
            binary_mask, slice_selection_config, 1
        )
        assert is_compatible is True
        assert slice_idx == 5

    def test_two_d_components_skip_center_filter_removes_all(self):
        """2D: Center slice passes fast check but all components fail filter_valid_components 
        — retries next slice."""
        binary_mask = torch.zeros(10, 10, 10)
        binary_mask[2:7, 2:7, 3] = 1       # Slice 3: valid 5x5 block
        binary_mask[0, 0, 5] = 1           # Slice 5: pixel at (0,0), component 1
        binary_mask[1, 1, 5] = 1           # Slice 5: pixel at (1,1), component 2 (diagonal, not 4-connected)
        binary_mask[2:7, 2:7, 7] = 1       # Slice 7: valid 5x5 block
        # Slice 5: x=[0,1] contiguous len 2, y=[0,1] contiguous len 2 → passes fast check
        # But each is a single voxel → fails filter_valid_components

        slice_selection_config = {
            'slice_selection_strategy': 'center',
            'collapsed_dim': 2
        }
        #1 Connectivity is the reason it fails, with 2 connectivity the diagonal would have passed!
        components, slice_idx, is_compatible = two_d_components_generation(
            binary_mask, slice_selection_config, 1
        )
        assert is_compatible is True
        assert slice_idx == 7  # Center (5) fails filter, retries, picks 7

    def test_two_d_components_skip_top_filter_removes_all(self):
        """2D: Top slice passes fast check but all components fail filter_valid_components — retries next slice."""
        binary_mask = torch.zeros(10, 10, 10)
        binary_mask[0, 0, 3] = 1           # Slice 3: pixel at (0,0), component 1
        binary_mask[1, 1, 3] = 1           # Slice 3: pixel at (1,1), component 2 (diagonal, not 4-connected)
        binary_mask[2:7, 2:7, 5] = 1       # Slice 5: valid 5x5 block
        # Slice 3: x=[0,1] contiguous len 2, y=[0,1] contiguous len 2 → passes fast check
        # But not connected in 1-connectivity → fails filter_valid_components

        slice_selection_config = {
            'slice_selection_strategy': 'top',
            'collapsed_dim': 2
        }

        #1 connectivity is the reason it fails, with 2 connectivity the diagonal would have passed!
        components, slice_idx, is_compatible = two_d_components_generation(
            binary_mask, slice_selection_config, 1
        )
        assert is_compatible is True
        assert slice_idx == 5  # Top (3) fails filter, retries, picks 5

    def test_two_d_components_skip_bottom_filter_removes_all(self):
        """2D: Bottom slice passes fast check but all components fail filter_valid_components — retries next slice."""
        binary_mask = torch.zeros(10, 10, 10)
        binary_mask[2:7, 2:7, 3] = 1       # Slice 3: valid 5x5 block
        binary_mask[0, 0, 7] = 1           # Slice 7: pixel at (0,0), component 1
        binary_mask[1, 1, 7] = 1           # Slice 7: pixel at (1,1), component 2 (diagonal, not 4-connected)
        # Slice 7: x=[0,1] contiguous len 2, y=[0,1] contiguous len 2 → passes fast check
        # But in 1-connectivity, they are separate single-voxel components → fails filter_valid_components

        slice_selection_config = {
            'slice_selection_strategy': 'bottom',
            'collapsed_dim': 2
        }
        components, slice_idx, is_compatible = two_d_components_generation(
            binary_mask, slice_selection_config, 1
        )
        assert is_compatible is True
        assert slice_idx == 3  # Bottom (7) fails filter, retries, picks 3

    def test_two_d_components_collapsed_dim_0(self):
        """2D: Collapsed dim 0 extracts slices along dim 0."""
        binary_mask = torch.zeros(10, 10, 10)
        binary_mask[5, 2:7, 2:7] = 1
        slice_selection_config = {
            'slice_selection_strategy': 'center',
            'collapsed_dim': 0
        }
        components, slice_idx, is_compatible = two_d_components_generation(
            binary_mask, slice_selection_config, 1
        )
        assert is_compatible is True
        assert slice_idx == 5


    def test_two_d_components_collapsed_dim_1(self):
        """2D: Collapsed dim 1 extracts slices along dim 1."""
        binary_mask = torch.zeros(10, 10, 10)
        binary_mask[2:7, 5, 2:7] = 1
        slice_selection_config = {
            'slice_selection_strategy': 'center',
            'collapsed_dim': 1
        }
        components, slice_idx, is_compatible = two_d_components_generation(
            binary_mask, slice_selection_config, 1
        )
        assert is_compatible is True
        assert slice_idx == 5

    ################################### Failure Cases ##############################################
    def test_two_d_components_no_valid_slice(self):
        """Test 2D component generation with no valid slices."""
        binary_mask = torch.zeros(10, 10, 10)
        binary_mask[5, 5, 5] = 1  # Single voxel - won't pass fast check
        
        slice_selection_config = {
            'slice_selection_strategy': 'center',
            'collapsed_dim': 2
        }
        
        components, slice_idx, is_compatible = two_d_components_generation(
            binary_mask, slice_selection_config, 1
        )
        
        assert is_compatible is False
        assert slice_idx is None
        assert components.sum() == 0

    def test_two_d_components_invalid_mask_dim(self):
        """2D: Should error when mask is not 3D."""
        mask = torch.zeros(10, 10)
        with pytest.raises(AssertionError):
            two_d_components_generation(mask, {'slice_selection_strategy': 'center', 'collapsed_dim': 2}, 1)

    def test_two_d_components_zero_size_dim(self):
        """2D: Should error when mask has a zero-size dimension."""
        mask = torch.zeros(10, 0, 10)
        slice_selection_config = {
            'slice_selection_strategy': 'center',
            'collapsed_dim': 2
        }
        with pytest.raises(ValueError, match="non-zero size"):
            two_d_components_generation(mask, slice_selection_config, 1)

    def test_two_d_components_invalid_connectivity(self):
        """2D: Should error when connectivity > 2."""
        mask = torch.zeros(10, 10, 10)
        mask[2:7, 2:7, 5] = 1
        slice_selection_config = {
            'slice_selection_strategy': 'center',
            'collapsed_dim': 2
        }
        with pytest.raises(ValueError, match="Connectivity must be 1 or 2"):
            two_d_components_generation(mask, slice_selection_config, 3)

    def test_two_d_components_no_nonzero_slices(self):
        """2D: Should error when no foreground along collapsed dim."""
        mask = torch.zeros(10, 10, 10)
        slice_selection_config = {
            'slice_selection_strategy': 'center',
            'collapsed_dim': 2
        }
        with pytest.raises(ValueError, match="No non-zero slices"):
            two_d_components_generation(mask, slice_selection_config, 1)

    def test_two_d_components_missing_collapsed_dim(self):
        """2D: Should error when collapsed_dim is missing from config."""
        mask = torch.zeros(10, 10, 10)
        mask[2:7, 2:7, 5] = 1
        with pytest.raises(KeyError):
            two_d_components_generation(mask, {'slice_selection_strategy': 'center'}, 1)

    def test_two_d_components_missing_strategy(self):
        """2D: Should error when slice_selection_strategy is missing."""
        mask = torch.zeros(10, 10, 10)
        mask[2:7, 2:7, 5] = 1
        with pytest.raises(KeyError):
            two_d_components_generation(mask, {'collapsed_dim': 2}, 1)


    def test_two_d_components_all_invalid_after_filter(self):
        """2D: Components exist but all fail filter_valid_components."""
        binary_mask = torch.zeros(10, 10, 10)
        binary_mask[0, 0, 5] = 1
        binary_mask[1, 1, 5] = 1
        binary_mask[2, 2, 5] = 1
        binary_mask[3, 3, 5] = 1
        slice_selection_config = {
            'slice_selection_strategy': 'center',
            'collapsed_dim': 2
        }
        components, slice_idx, is_compatible = two_d_components_generation(
            binary_mask, slice_selection_config, 1
        )
        assert is_compatible is False
        assert slice_idx is None
        assert components.sum() == 0

class TestThreeDComponentsGeneration:
    """Tests for 3D component generation.
    
    three_d_components_generation is a thin wrapper around 
    generate_components_from_mask (dimensionality=3) that validates 
    inputs and delegates. Unlike the 2D version, it does not perform 
    slice selection — it operates directly on the full 3D volume.
    """

    # =========================================================================
    # Input validation — wrapper-specific checks
    # =========================================================================

    def test_three_d_components_invalid_mask_dim(self):
        """3D: Should error when mask is not 3D."""
        mask = torch.zeros(10, 10)
        with pytest.raises(AssertionError):
            three_d_components_generation(mask, 1)

    def test_three_d_components_zero_size_dim(self):
        """3D: Should error when mask has a zero-size dimension."""
        mask = torch.zeros(10, 0, 10)
        with pytest.raises(ValueError, match="non-zero size"):
            three_d_components_generation(mask, 1)

    def test_three_d_components_empty_raises(self):
        """3D: Should raise ValueError for empty mask."""
        binary_mask = torch.zeros(10, 10, 10)

        with pytest.raises(ValueError, match="non-zero values"):
            three_d_components_generation(binary_mask, 1)

    def test_three_d_components_dim_assertion_1d(self):
        """3D: Should raise AssertionError for 1D mask."""
        mask = torch.zeros(10)
        with pytest.raises(AssertionError, match="Input binary mask must be 3D."):
            three_d_components_generation(mask, 1)

    def test_three_d_components_dim_assertion_4d(self):
        """3D: Should raise AssertionError for 4D mask."""
        mask = torch.zeros(3, 3, 3, 3)
        with pytest.raises(AssertionError, match="Input binary mask must be 3D."):
            three_d_components_generation(mask, 1)

    # =========================================================================
    # Connectivity config validation
    # =========================================================================

    def test_three_d_components_invalid_connectivity_above_3(self):
        """3D: Should error when connectivity > 3."""
        mask = torch.zeros(10, 10, 10)
        mask[2:8, 2:8, 2:8] = 1
        with pytest.raises(ValueError, match="Connectivity must be 1, 2, or 3"):
            three_d_components_generation(mask, 4)

    def test_three_d_components_zero_connectivity_raises(self):
        """3D: Should raise ValueError for connectivity 0."""
        mask = torch.zeros(10, 10, 10)
        mask[2:8, 2:8, 2:8] = 1
        with pytest.raises(ValueError, match="Connectivity must be 1, 2, or 3"):
            three_d_components_generation(mask, 0)

    def test_three_d_components_negative_connectivity_raises(self):
        """3D: Should raise ValueError for negative connectivity."""
        mask = torch.zeros(10, 10, 10)
        mask[2:8, 2:8, 2:8] = 1
        with pytest.raises(ValueError, match="Connectivity must be 1, 2, or 3"):
            three_d_components_generation(mask, -1)

    # =========================================================================
    # Fast pre-check failures — mask lacks 3D volumetric extent
    # =========================================================================

    def test_three_d_components_single_voxel_fails(self):
        """3D: Single foreground voxel fails the fast check."""
        binary_mask = torch.zeros(10, 10, 10)
        binary_mask[5, 5, 5] = 1

        components, is_compatible = three_d_components_generation(
            binary_mask, 1
        )

        assert is_compatible is False
        assert components is None

    def test_three_d_components_line_fails(self):
        """3D: 1D line fails fast pre-check."""
        mask = torch.zeros(10, 10, 10)
        mask[2:8, 5, 5] = 1
        components, is_compatible = three_d_components_generation(mask, 1)
        assert is_compatible is False
        assert components is None

    def test_three_d_components_plane_fails(self):
        """3D: 2D plane fails fast pre-check."""
        mask = torch.zeros(10, 10, 10)
        mask[5, 2:8, 2:8] = 1
        components, is_compatible = three_d_components_generation(mask, 1)
        assert is_compatible is False
        assert components is None

    def test_three_d_components_single_slice(self):
        """3D: Mask with single-slice dim (1, H, W) — fails, only 1 unique z-coord."""
        mask = torch.zeros(1, 10, 10)
        mask[0, 2:6, 2:6] = 1
        with pytest.raises(ValueError, match="Input binary mask must have non-zero size and at least 2 in all dimensions."):
            three_d_components_generation(mask, 1)
        # components, is_compatible = three_d_components_generation(mask, 1)
        

    def test_three_d_components_thin_volume(self):
        """3D: Mask with middle dim of size 1 (H, 1, W) — fails, only 1 unique y-coord."""
        mask = torch.zeros(10, 1, 10)
        mask[2:6, 0, 2:6] = 1
        with pytest.raises(ValueError, match="Input binary mask must have non-zero size and at least 2 in all dimensions."):
            three_d_components_generation(mask, 1)
        # components, is_compatible = three_d_components_generation(mask, 1)

    # =========================================================================
    # Valid extractions
    # =========================================================================

    def test_three_d_components_valid(self):
        """Test 3D component generation with valid mask."""
        binary_mask = torch.zeros(10, 10, 10)
        binary_mask[2:8, 2:8, 2:8] = 1

        components, is_compatible = three_d_components_generation(
            binary_mask, 1
        )

        assert is_compatible is True
        assert components.sum() > 0
        assert components.max() >= 1

    def test_three_d_components_connectivity_2(self):
        """3D: Valid mask with 18-conn connectivity."""
        binary_mask = torch.zeros(10, 10, 10)
        binary_mask[2:8, 2:8, 2:8] = 1

        components, is_compatible = three_d_components_generation(
            binary_mask, 2
        )

        assert is_compatible is True
        assert components.sum() > 0
        assert components.max() >= 1

    def test_three_d_components_connectivity_3(self):
        """3D: Valid mask with 26-conn connectivity."""
        binary_mask = torch.zeros(10, 10, 10)
        binary_mask[2:8, 2:8, 2:8] = 1

        components, is_compatible = three_d_components_generation(
            binary_mask, 3
        )

        assert is_compatible is True
        assert components.sum() > 0
        assert components.max() >= 1

    # =========================================================================
    # All components filtered — pass fast check but fail filter_valid_components
    # =========================================================================

    def test_three_d_components_all_invalid_after_filter(self):
        """3D: Components exist but all fail filter_valid_components."""
        binary_mask = torch.zeros(5, 5, 5)
        binary_mask[0, 0, 0] = 1
        binary_mask[1, 1, 1] = 1
        binary_mask[2, 2, 2] = 1
        binary_mask[3, 3, 3] = 1
        components, is_compatible = three_d_components_generation(
            binary_mask, 1
        )
        assert is_compatible is False
        assert components is None

    def test_three_d_components_mixed_valid_invalid(self):
        """3D: Mixed components — valid kept, invalid filtered."""
        binary_mask = torch.zeros(10, 10, 10)
        binary_mask[0:2, 0:2, 0:2] = 1   # valid 2x2x2 block
        binary_mask[9, 9, 9] = 1          # invalid single voxel
        components, is_compatible = three_d_components_generation(
            binary_mask, 1
        )
        assert is_compatible is True
        assert components.max() == 1


    # =========================================================================
    # Cavity / Shell tests
    # =========================================================================

    def test_three_d_components_symmetric_cavity(self):
        """3D: Shell with symmetric cavity is one valid component."""
        mask = torch.ones(7, 7, 7)
        mask[2:5, 2:5, 2:5] = 0
        components, is_compatible = three_d_components_generation(mask, 1)
        assert is_compatible is True
        assert components is not None
        assert components.max() == 1
        assert torch.all(components[2:5, 2:5, 2:5] == 0)

    def test_three_d_components_asymmetric_cavity(self):
        """3D: Shell with asymmetric cavity is one valid component."""
        mask = torch.ones(9, 9, 9)
        mask[3:6, 2:7, 3:6] = 0
        components, is_compatible = three_d_components_generation(mask, 1)
        assert is_compatible is True
        assert components is not None
        assert components.max() == 1
        assert torch.all(components[3:6, 2:7, 3:6] == 0)

    def test_three_d_components_cavity_with_inner_voxel(self):
        """3D: Shell valid, inner single-voxel filtered out."""
        mask = torch.ones(9, 9, 9)
        mask[2:8, 2:8, 2:8] = 0
        mask[5, 5, 5] = 1
        components, is_compatible = three_d_components_generation(mask, 1)
        assert is_compatible is True
        assert components is not None
        assert components.max() == 1
        assert components[5, 5, 5] == 0

    # =========================================================================
    # Diagonal / Connectivity variants  — isolated voxels at various conn
    # =========================================================================

    def test_three_d_components_diagonal_conn_1(self):
        """3D diagonal conn=1: isolated single-voxel components, all filtered."""
        mask = torch.zeros(4, 4, 4)
        for i in range(4):
            mask[i, i, i] = 1
        components, is_compatible = three_d_components_generation(mask, 1)
        assert is_compatible is False
        assert components is None

    def test_three_d_components_diagonal_conn_2(self):
        """3D diagonal conn=2: corner diagonal still 3 hops apart, all filtered."""
        mask = torch.zeros(4, 4, 4)
        for i in range(4):
            mask[i, i, i] = 1
        components, is_compatible = three_d_components_generation(mask, 2)
        assert is_compatible is False
        assert components is None

    def test_three_d_components_diagonal_conn_3(self):
        """3D diagonal conn=3: 26-conn bridges diagonal, one valid component."""
        mask = torch.ones(4, 4, 4)
        for i in range(4):
            mask[i, i, i] = 1
        components, is_compatible = three_d_components_generation(mask, 3)
        assert is_compatible is True
        assert components is not None
        assert components.max() >= 1

    # =========================================================================
    # Mixed valid/invalid — valid blob + separate line (filtered)
    # =========================================================================

    def test_three_d_components_blob_and_line(self):
        """3D: Valid blob kept, separate line filtered out."""
        mask = torch.zeros(10, 10, 10)
        mask[2:5, 2:5, 2:5] = 1  # 3x3x3 valid blob
        mask[6:10, 7, 7] = 1     # line along x, only 1 y and 1 z
        components, is_compatible = three_d_components_generation(mask, 1)
        assert is_compatible is True
        assert components is not None
        assert components.max() == 1
        assert torch.all(components[6:10, 7, 7] == 0)

    # =========================================================================
    # Pretzel tests (shell + diagonal bridge)
    # =========================================================================

    def test_three_d_components_pretzel_conn_1(self):
        """3D pretzel conn=1: interior diagonal voxels filtered, shell remains."""
        mask = torch.ones(9, 9, 9)
        mask[1:8, 1:8, 1:8] = 0
        mask[range(9), range(9), range(9)] = 1
        components, is_compatible = three_d_components_generation(mask, 1)
        assert is_compatible is True
        assert components is not None
        assert components.max() == 1
        assert torch.all(components[2:7, 2:7, 2:7] == 0)
        assert components[2, 2, 2] == 0
        assert components[3, 3, 3] == 0
        assert components[4, 4, 4] == 0
        assert components[5, 5, 5] == 0
        assert components[6, 6, 6] == 0

    def test_three_d_components_pretzel_conn_2(self):
        """3D pretzel conn=2: interior diagonal voxels filtered, shell remains."""
        mask = torch.ones(9, 9, 9)
        mask[1:8, 1:8, 1:8] = 0
        mask[range(9), range(9), range(9)] = 1
        components, is_compatible = three_d_components_generation(mask, 2)
        assert is_compatible is True
        assert components is not None
        assert components.max() == 1
        assert torch.all(components[2:7, 2:7, 2:7] == 0)
        assert components[2, 2, 2] == 0
        assert components[3, 3, 3] == 0
        assert components[4, 4, 4] == 0
        assert components[5, 5, 5] == 0
        assert components[6, 6, 6] == 0

    def test_three_d_components_pretzel_conn_3(self):
        """3D pretzel conn=3: 26-conn bridges diagonal into the shell."""
        mask = torch.ones(9, 9, 9)
        mask[1:8, 1:8, 1:8] = 0
        mask[range(9), range(9), range(9)] = 1
        components, is_compatible = three_d_components_generation(mask, 3)
        assert is_compatible is True
        assert components is not None
        assert components.max() == 1
        assert torch.all(components.bool() == mask.bool())

    def test_three_d_components_pretzel_planar_conn_2(self):
        """
        3D pretzel planar conn=2: plane through the diagonal bijects the 
        shell but remains one component at 18-connectivity.
        """
        mask = torch.ones(9, 9, 9)
        mask[1:8, 1:8, 1:8] = 0
        for i in range(9):
            mask[i, i, :] = 1
        components, is_compatible = three_d_components_generation(mask, 2)
        assert is_compatible is True
        assert components is not None
        assert components.max() == 1
        assert torch.all(components.bool() == mask.bool())

# ============================================================================
# Test select_component
# ============================================================================

class TestSelectComponent:
    """Tests for component selection."""
    
    def test_select_top_k_components(self):
        """Test selecting top-k components."""
        # Create a mask with 3 components of different sizes
        components = torch.zeros(10, 10)
        components[0:3, 0:3] = 1  # Component 1: 9 voxels
        components[4:7, 4:7] = 3  # Component 3: 9 voxels
        components[8:9, 8:9] = 2  # Component 2: 1 voxel
        
        component_selection_config = {
            'component_selection_process': 'top-k',
            'top-k': 2
        }
        
        selected = select_component(components, component_selection_config)
        
        # Should have 2 components (IDs 1 and 2)
        unique_vals = torch.unique(selected)
        assert len(unique_vals) == 2 # 0 for background + 1 foreground
        #Apply mask to original components to check that we kept the correct ones
        masked_components = components * selected.bool()
        assert torch.all(masked_components[0:3, 0:3] == 1)  # Component 1 should be kept
        assert torch.all(masked_components[4:7, 4:7] == 3)  # Component 3 should be kept
        assert torch.all(masked_components[8:9, 8:9] == 0)  # Component 2 should be removed

    def test_select_top_k_exact_count(self):
        """Test selecting top-k when exactly k components exist."""
        components = torch.zeros(10, 10)
        components[0:3, 0:3] = 1  # Component 1: 9 voxels
        components[4:7, 4:7] = 2  # Component 2: 9 voxels

        component_selection_config = {
            'component_selection_process': 'top-k',
            'top-k': 2
        }

        selected = select_component(components, component_selection_config)

        # All foreground should be selected (binary mask of all 1s)
        fg_components = torch.where(components > 0, 1, 0)
        assert torch.all(selected == fg_components)  # All foreground should be kept


    def test_select_top_k_fewer_than_k(self):
        """Test selecting top-k when fewer components exist."""
        components = torch.zeros(10, 10)
        components[0:3, 0:3] = 1  # Only 1 component
        
        component_selection_config = {
            'component_selection_process': 'top-k',
            'top-k': 5
        }
        
        selected = select_component(components, component_selection_config)
        
        # Should still have the 1 component
        unique_vals = torch.unique(selected)
        assert len(unique_vals) == 2  # 0, 1
        assert torch.all(selected == components.float())  # The existing component should be kept
    
    def test_select_top_k_invalid_config(self):
        """Test with invalid component_selection_process."""
        components = torch.zeros(10, 10)
        components[0:3, 0:3] = 1
        
        component_selection_config = {
            'component_selection_process': 'invalid-process',
            'top-k': 2
        }
        
        with pytest.raises(ValueError, match="Invalid component_selection_process"):
            select_component(components, component_selection_config)
    
    def test_select_top_k_missing_parameter(self):
        """Test with missing top_k parameter."""
        components = torch.zeros(10, 10)
        components[0:3, 0:3] = 1
        
        component_selection_config = {
            'component_selection_process': 'top-k'
        }
        
        with pytest.raises(KeyError, match="top-k"):
            select_component(components, component_selection_config)
    
    def test_select_empty_components(self):
        """Test with empty components tensor."""
        components = torch.zeros(10, 10)
        
        component_selection_config = {
            'component_selection_process': 'top-k',
            'top-k': 2
        }
        
        with pytest.raises(ValueError, match="no non-zero values"):
            select_component(components, component_selection_config)


# ============================================================================
# Test extract_sampling_region
# ============================================================================

class TestExtractSamplingRegion:
    """Tests for extracting sampling regions from binary masks."""
    
    def test_extract_sampling_region_3d(self):
        """Test 3D sampling region extraction."""
        binary_mask = torch.zeros(10, 10, 10)
        binary_mask[2:7, 2:7, 2:7] = 1
        
        sampling_config = {
            'dimensionality': 3,
            'region_extraction_config': {
                'connectivity': 1,
                'component_selection_process': 'top-k',
                'top-k': 1
            }
        }
        
        is_compatible, component_mask, slice_idx = extract_sampling_region(
            binary_mask, sampling_config
        )
        
        assert is_compatible is True
        assert component_mask.sum() > 0
        assert slice_idx is None  # 3D has no slice index
    
    def test_extract_sampling_region_2d(self):
        """Test 2D sampling region extraction."""
        binary_mask = torch.zeros(10, 10, 10)
        binary_mask[2:7, 2:7, 5] = 1  # Slice 5
        
        sampling_config = {
            'dimensionality': 2,
            'collapsed_dim': 2,
            'region_extraction_config': {
                'connectivity': 1,
                'component_selection_process': 'top-k',
                'top-k': 1,
                'slice_selection': 'center'
            }
        }
        
        is_compatible, component_mask, slice_idx = extract_sampling_region(
            binary_mask, sampling_config
        )
        
        assert is_compatible is True
        assert component_mask.sum() > 0
        assert slice_idx == 5
    
    def test_extract_sampling_region_empty_mask(self):
        """Test with empty binary mask."""
        binary_mask = torch.zeros(10, 10, 10)
        
        sampling_config = {
            'dimensionality': 3,
            'region_extraction_config': {
                'connectivity': 1,
                'component_selection_process': 'top-k',
                'top-k': 1
            }
        }
        
        is_compatible, component_mask, slice_idx = extract_sampling_region(
            binary_mask, sampling_config
        )
        
        assert is_compatible is False
        assert component_mask.sum() == 0
        assert slice_idx is None
    
    def test_extract_sampling_region_missing_region_extraction_config(self):
        """Test with missing required config."""
        binary_mask = torch.zeros(10, 10, 10)
        
        sampling_config = {
            'dimensionality': 3
            # Missing region_extraction_config
        }
        
        with pytest.raises(KeyError, match="region_extraction_config"):
            extract_sampling_region(binary_mask, sampling_config)
    
    def test_extract_sampling_region_2d_missing_collapsed_dim(self):
        """Test 2D with missing collapsed_dim."""
        binary_mask = torch.zeros(10, 10, 10)
        
        sampling_config = {
            'dimensionality': 2,
            'region_extraction_config': {
                'connectivity': 1,
                'component_selection_process': 'top-k',
                'top-k': 1,
                'slice_selection': 'center'
            }
        }
        with pytest.raises(KeyError, match="collapsed_dim"):
            extract_sampling_region(binary_mask, sampling_config)

    # ==================== Config Validation ====================

    def test_extract_sampling_region_missing_dimensionality(self):
        """Missing dimensionality key raises KeyError."""
        binary_mask = torch.zeros(5, 5, 5)
        sampling_config = {
            'region_extraction_config': {
                'connectivity': 1,
                'component_selection_process': 'top-k',
                'top-k': 1
            }
        }
        with pytest.raises(KeyError, match="dimensionality"):
            extract_sampling_region(binary_mask, sampling_config)

    def test_extract_sampling_region_invalid_dimensionality(self):
        """Invalid dimensionality raises ValueError."""
        binary_mask = torch.zeros(5, 5, 5)
        sampling_config = {
            'dimensionality': 4,
            'region_extraction_config': {
                'connectivity': 1,
                'component_selection_process': 'top-k',
                'top-k': 1
            }
        }
        with pytest.raises(ValueError, match="dimensionality"):
            extract_sampling_region(binary_mask, sampling_config)

    def test_extract_sampling_region_missing_connectivity(self):
        """Missing connectivity in region_extraction_config raises KeyError."""
        binary_mask = torch.zeros(5, 5, 5)
        sampling_config = {
            'dimensionality': 3,
            'region_extraction_config': {
                'component_selection_process': 'top-k',
                'top-k': 1
            }
        }
        with pytest.raises(KeyError, match="connectivity"):
            extract_sampling_region(binary_mask, sampling_config)

    def test_extract_sampling_region_missing_component_selection_process(self):
        """Missing component_selection_process raises KeyError."""
        binary_mask = torch.zeros(5, 5, 5)
        sampling_config = {
            'dimensionality': 3,
            'region_extraction_config': {
                'connectivity': 1
            }
        }
        with pytest.raises(KeyError, match="component_selection_process"):
            extract_sampling_region(binary_mask, sampling_config)

    def test_extract_sampling_region_invalid_slice_selection(self):
        """Invalid slice_selection value for 2D raises ValueError."""
        binary_mask = torch.zeros(10, 10, 10)
        sampling_config = {
            'dimensionality': 2,
            'collapsed_dim': 2,
            'region_extraction_config': {
                'connectivity': 1,
                'component_selection_process': 'top-k',
                'top-k': 1,
                'slice_selection': 'invalid_strategy'
            }
        }
        with pytest.raises(ValueError, match="slice_selection"):
            extract_sampling_region(binary_mask, sampling_config)

    def test_extract_sampling_region_slice_selection_not_none_3d(self):
        """slice_selection not None for 3D raises ValueError."""
        binary_mask = torch.zeros(10, 10, 10)
        sampling_config = {
            'dimensionality': 3,
            'region_extraction_config': {
                'connectivity': 1,
                'component_selection_process': 'top-k',
                'top-k': 1,
                'slice_selection': 'center'
            }
        }
        with pytest.raises(ValueError, match="slice_selection"):
            extract_sampling_region(binary_mask, sampling_config)

    def test_extract_sampling_region_invalid_collapsed_dim(self):
        """Invalid collapsed_dim value raises ValueError."""
        binary_mask = torch.zeros(10, 10, 10)
        sampling_config = {
            'dimensionality': 2,
            'collapsed_dim': 5,
            'region_extraction_config': {
                'connectivity': 1,
                'component_selection_process': 'top-k',
                'top-k': 1,
                'slice_selection': 'center'
            }
        }
        with pytest.raises(ValueError, match="collapsed_dim"):
            extract_sampling_region(binary_mask, sampling_config)

    def test_extract_sampling_region_2d_missing_slice_selection(self):
        """2D missing slice_selection key raises KeyError."""
        binary_mask = torch.zeros(10, 10, 10)
        sampling_config = {
            'dimensionality': 2,
            'collapsed_dim': 2,
            'region_extraction_config': {
                'connectivity': 1,
                'component_selection_process': 'top-k',
                'top-k': 1
            }
        }
        with pytest.raises(KeyError, match="slice_selection"):
            extract_sampling_region(binary_mask, sampling_config)

    # ==================== 2D Slice Selection Strategies ====================

    def test_extract_sampling_region_2d_slice_top(self):
        """2D with 'top' slice selection selects the first non-zero slice."""
        binary_mask = torch.zeros(10, 10, 10)
        binary_mask[2:7, 2:7, 2] = 1
        binary_mask[2:7, 2:7, 8] = 1

        sampling_config = {
            'dimensionality': 2,
            'collapsed_dim': 2,
            'region_extraction_config': {
                'connectivity': 1,
                'component_selection_process': 'top-k',
                'top-k': 1,
                'slice_selection': 'top'
            }
        }

        is_compatible, component_mask, slice_idx = extract_sampling_region(
            binary_mask, sampling_config
        )

        assert is_compatible is True
        assert component_mask.sum() > 0
        assert slice_idx == 2

    def test_extract_sampling_region_2d_slice_bottom(self):
        """2D with 'bottom' slice selection selects the last non-zero slice."""
        binary_mask = torch.zeros(10, 10, 10)
        binary_mask[2:7, 2:7, 2] = 1
        binary_mask[2:7, 2:7, 8] = 1

        sampling_config = {
            'dimensionality': 2,
            'collapsed_dim': 2,
            'region_extraction_config': {
                'connectivity': 1,
                'component_selection_process': 'top-k',
                'top-k': 1,
                'slice_selection': 'bottom'
            }
        }

        is_compatible, component_mask, slice_idx = extract_sampling_region(
            binary_mask, sampling_config
        )

        assert is_compatible is True
        assert component_mask.sum() > 0
        assert slice_idx == 8

    def test_extract_sampling_region_2d_slice_random(self):
        """2D with 'random' slice selection returns a valid result."""
        binary_mask = torch.zeros(10, 10, 10)
        binary_mask[2:7, 2:7, 5] = 1

        sampling_config = {
            'dimensionality': 2,
            'collapsed_dim': 2,
            'region_extraction_config': {
                'connectivity': 1,
                'component_selection_process': 'top-k',
                'top-k': 1,
                'slice_selection': 'random'
            }
        }

        is_compatible, component_mask, slice_idx = extract_sampling_region(
            binary_mask, sampling_config
        )

        assert is_compatible is True
        assert component_mask.sum() > 0
        assert slice_idx == 5

    def test_extract_sampling_region_2d_collapsed_dim_0(self):
        """2D with collapsed_dim=0 extracts the correct slice."""
        binary_mask = torch.zeros(10, 10, 10)
        binary_mask[4, 2:7, 2:7] = 1

        sampling_config = {
            'dimensionality': 2,
            'collapsed_dim': 0,
            'region_extraction_config': {
                'connectivity': 1,
                'component_selection_process': 'top-k',
                'top-k': 1,
                'slice_selection': 'center'
            }
        }

        is_compatible, component_mask, slice_idx = extract_sampling_region(
            binary_mask, sampling_config
        )

        assert is_compatible is True
        assert component_mask.sum() > 0
        assert slice_idx == 4

    def test_extract_sampling_region_2d_collapsed_dim_1(self):
        """2D with collapsed_dim=1 extracts the correct slice."""
        binary_mask = torch.zeros(10, 10, 10)
        binary_mask[2:7, 4, 2:7] = 1

        sampling_config = {
            'dimensionality': 2,
            'collapsed_dim': 1,
            'region_extraction_config': {
                'connectivity': 1,
                'component_selection_process': 'top-k',
                'top-k': 1,
                'slice_selection': 'center'
            }
        }

        is_compatible, component_mask, slice_idx = extract_sampling_region(
            binary_mask, sampling_config
        )

        assert is_compatible is True
        assert component_mask.sum() > 0
        assert slice_idx == 4

    # ==================== Connectivity Variants ====================

    def test_extract_sampling_region_2d_connectivity_2(self):
        """2D with connectivity=2."""
        binary_mask = torch.zeros(10, 10, 10)
        slice_2d = torch.zeros(10, 10)
        slice_2d[0:3, 0:2] = 1  # 6 voxels
        slice_2d[4:6, 4:6] = 1  # 4 voxels, separated by a gap so 8-conn doesn't merge
        binary_mask[:, :, 5] = slice_2d

        sampling_config = {
            'dimensionality': 2,
            'collapsed_dim': 2,
            'region_extraction_config': {
                'connectivity': 2,
                'component_selection_process': 'top-k',
                'top-k': 1,
                'slice_selection': 'center'
            }
        }

        is_compatible, component_mask, slice_idx = extract_sampling_region(
            binary_mask, sampling_config
        )

        assert is_compatible is True
        assert component_mask.sum() > 0
        assert slice_idx == 5
        assert component_mask[0:3, 0:2, 5].sum() > 0  # Larger component kept
        assert component_mask[4:6, 4:6, 5].sum() == 0  # Smaller component filtered

    def test_extract_sampling_region_3d_connectivity_2(self):
        """3D with connectivity=2."""
        binary_mask = torch.zeros(10, 10, 10)
        binary_mask[2:4, 2:4, 2:4] = 1
        binary_mask[5:8, 5:8, 5:8] = 1

        sampling_config = {
            'dimensionality': 3,
            'region_extraction_config': {
                'connectivity': 2,
                'component_selection_process': 'top-k',
                'top-k': 1
            }
        }

        is_compatible, component_mask, slice_idx = extract_sampling_region(
            binary_mask, sampling_config
        )

        assert is_compatible is True
        assert component_mask.sum() > 0
        assert slice_idx is None
        assert component_mask[2:4, 2:4, 2:4].sum() == 0  # Smaller component filtered
        assert component_mask[5:8, 5:8, 5:8].sum() > 0  # Larger component kept

    def test_extract_sampling_region_3d_connectivity_3_top_k_2(self):
        """3D with connectivity=3."""
        binary_mask = torch.zeros(10, 10, 10)
        binary_mask[list(range(10)), list(range(10)), list(range(10))] = 1
        #Diagonal.

        #Now lets add a standard blob disconnected.
        binary_mask[5:8, 5:8, 0:3] = 1

        sampling_config = {
            'dimensionality': 3,
            'region_extraction_config': {
                'connectivity': 3,
                'component_selection_process': 'top-k',
                'top-k': 2
            }
        }

        is_compatible, component_mask, slice_idx = extract_sampling_region(
            binary_mask, sampling_config
        )

        assert is_compatible is True
        assert component_mask.sum() > 0
        assert slice_idx is None
        # select_component returns a binary mask, so both components are kept as 1s
        assert component_mask[5:8, 5:8, 0:3].sum() > 0  # Blob kept
        assert component_mask[list(range(10)), list(range(10)), list(range(10))].sum() > 0  # Diagonal kept

    def test_extract_sampling_region_3d_connectivity_3_top_k_1(self):
        """3D with connectivity=3."""
        binary_mask = torch.zeros(10, 10, 10)
        binary_mask[list(range(10)), list(range(10)), list(range(10))] = 1
        #Diagonal.

        #Now lets add a standard blob disconnected.
        binary_mask[5:8, 5:8, 0:3] = 1

        sampling_config = {
            'dimensionality': 3,
            'region_extraction_config': {
                'connectivity': 3,
                'component_selection_process': 'top-k',
                'top-k': 1
            }
        }

        is_compatible, component_mask, slice_idx = extract_sampling_region(
            binary_mask, sampling_config
        )

        assert is_compatible is True
        assert component_mask.sum() > 0
        assert slice_idx is None
        assert component_mask[5:8, 5:8, 0:3].sum() > 0  # Blob (larger, 27 voxels) kept
        assert component_mask[list(range(10)), list(range(10)), list(range(10))].sum() == 0  # Diagonal (10 voxels) filtered

    # ==================== Flow-through Edge Cases ====================

    def test_extract_sampling_region_no_compatible_slice_2d(self):
        """2D where no slice generates a valid component returns is_compatible=False."""
        binary_mask = torch.zeros(10, 10, 10)
        binary_mask[5, 5, 3] = 1
        binary_mask[5, 5, 7] = 1

        sampling_config = {
            'dimensionality': 2,
            'collapsed_dim': 2,
            'region_extraction_config': {
                'connectivity': 1,
                'component_selection_process': 'top-k',
                'top-k': 1,
                'slice_selection': 'center'
            }
        }

        is_compatible, component_mask, slice_idx = extract_sampling_region(
            binary_mask, sampling_config
        )

        assert is_compatible is False
        assert component_mask.sum() == 0
        assert slice_idx is None

    def test_extract_sampling_region_no_compatible_slice_2d_nontrivial(self):
        """2D where no slice generates a valid component returns 
        is_compatible=False, but the shape was not trivially detected by
        a contiguity check, fails due to connectivity=1."""
        binary_mask = torch.zeros(10, 10, 10)
        binary_mask[list(range(10)), list(range(10)), [3] * len(list(range(10)))] = 1

        sampling_config = {
            'dimensionality': 2,
            'collapsed_dim': 2,
            'region_extraction_config': {
                'connectivity': 1,
                'component_selection_process': 'top-k',
                'top-k': 1,
                'slice_selection': 'center'
            }
        }

        is_compatible, component_mask, slice_idx = extract_sampling_region(
            binary_mask, sampling_config
        )

        assert is_compatible is False
        assert component_mask.sum() == 0
        assert slice_idx is None

    def test_extract_sampling_region_top_k_more_than_components(self):
        """top-k > available components returns all existing components."""
        binary_mask = torch.zeros(10, 10, 10)
        binary_mask[0:3, 0:3, 0:3] = 1
        binary_mask[5:7, 5:7, 5:7] = 1

        sampling_config = {
            'dimensionality': 3,
            'region_extraction_config': {
                'connectivity': 1,
                'component_selection_process': 'top-k',
                'top-k': 5
            }
        }

        is_compatible, component_mask, slice_idx = extract_sampling_region(
            binary_mask, sampling_config
        )

        assert is_compatible is True
        assert component_mask.sum() > 0
        assert component_mask[0:3, 0:3, 0:3].sum() > 0
        assert component_mask[5:7, 5:7, 5:7].sum() > 0
        assert slice_idx is None

    def test_extract_sampling_region_top_k_1_multiple_components(self):
        """top-k=1 with multiple components returns only the largest component."""
        binary_mask = torch.zeros(10, 10, 10)
        binary_mask[0:3, 0:3, 0:3] = 1
        binary_mask[5:7, 5:7, 5:7] = 1

        sampling_config = {
            'dimensionality': 3,
            'region_extraction_config': {
                'connectivity': 1,
                'component_selection_process': 'top-k',
                'top-k': 1
            }
        }

        is_compatible, component_mask, slice_idx = extract_sampling_region(
            binary_mask, sampling_config
        )

        assert is_compatible is True
        assert component_mask.sum() > 0
        assert component_mask[0:3, 0:3, 0:3].sum() > 0
        assert component_mask[5:7, 5:7, 5:7].sum() == 0
        assert slice_idx is None

    def test_extract_sampling_region_top_k_2_exact_2_components(self):
        """top-k=2 with exactly 2 components returns both."""
        binary_mask = torch.zeros(10, 10, 10)
        binary_mask[0:3, 0:3, 0:3] = 1
        binary_mask[5:8, 5:8, 5:8] = 1

        sampling_config = {
            'dimensionality': 3,
            'region_extraction_config': {
                'connectivity': 1,
                'component_selection_process': 'top-k',
                'top-k': 2
            }
        }

        is_compatible, component_mask, slice_idx = extract_sampling_region(
            binary_mask, sampling_config
        )

        assert is_compatible is True
        assert component_mask[0:3, 0:3, 0:3].sum() > 0
        assert component_mask[5:8, 5:8, 5:8].sum() > 0
        assert slice_idx is None

    def test_extract_sampling_region_2d_top_k_more_than_components(self):
        """2D: top-k > available components in a single slice keeps all."""
        binary_mask = torch.zeros(10, 10, 10)
        binary_mask[0:2, 0:3, 5] = 1   # Component 1: 6 voxels
        binary_mask[7:9, 7:9, 5] = 1   # Component 2: 4 voxels

        sampling_config = {
            'dimensionality': 2,
            'collapsed_dim': 2,
            'region_extraction_config': {
                'connectivity': 1,
                'component_selection_process': 'top-k',
                'top-k': 5,
                'slice_selection': 'center'
            }
        }

        is_compatible, component_mask, slice_idx = extract_sampling_region(
            binary_mask, sampling_config
        )

        assert is_compatible is True
        assert component_mask.sum() > 0
        assert component_mask[0:2, 0:3, 5].sum() > 0
        assert component_mask[7:9, 7:9, 5].sum() > 0
        assert slice_idx == 5

    def test_extract_sampling_region_2d_top_k_1_multiple_components(self):
        """2D: top-k=1 with multiple components in a single slice picks the largest."""
        binary_mask = torch.zeros(10, 10, 10)
        binary_mask[0:3, 0:3, 5] = 1   # Component 1: 9 voxels (larger)
        binary_mask[7:9, 7:9, 5] = 1   # Component 2: 4 voxels

        sampling_config = {
            'dimensionality': 2,
            'collapsed_dim': 2,
            'region_extraction_config': {
                'connectivity': 1,
                'component_selection_process': 'top-k',
                'top-k': 1,
                'slice_selection': 'center'
            }
        }

        is_compatible, component_mask, slice_idx = extract_sampling_region(
            binary_mask, sampling_config
        )

        assert is_compatible is True
        assert component_mask.sum() > 0
        assert component_mask[0:3, 0:3, 5].sum() > 0
        assert component_mask[7:9, 7:9, 5].sum() == 0
        assert slice_idx == 5

    def test_extract_sampling_region_2d_top_k_2_exact_2_components(self):
        """2D: top-k=2 with exactly 2 components in a single slice keeps both."""
        binary_mask = torch.zeros(10, 10, 10)
        binary_mask[0:3, 0:3, 5] = 1   # Component 1: 9 voxels
        binary_mask[7:9, 7:9, 5] = 1   # Component 2: 4 voxels

        sampling_config = {
            'dimensionality': 2,
            'collapsed_dim': 2,
            'region_extraction_config': {
                'connectivity': 1,
                'component_selection_process': 'top-k',
                'top-k': 2,
                'slice_selection': 'center'
            }
        }

        is_compatible, component_mask, slice_idx = extract_sampling_region(
            binary_mask, sampling_config
        )

        assert is_compatible is True
        assert component_mask[0:3, 0:3, 5].sum() > 0
        assert component_mask[7:9, 7:9, 5].sum() > 0
        assert slice_idx == 5

    # ==================== Shape/Volume Edge Cases ====================

    def test_extract_sampling_region_non_cube_shape_3d(self):
        """3D extraction with non-cube mask shape."""
        binary_mask = torch.zeros(5, 10, 20)
        binary_mask[1:4, 2:8, 3:15] = 1

        sampling_config = {
            'dimensionality': 3,
            'region_extraction_config': {
                'connectivity': 1,
                'component_selection_process': 'top-k',
                'top-k': 1
            }
        }

        is_compatible, component_mask, slice_idx = extract_sampling_region(
            binary_mask, sampling_config
        )

        assert is_compatible is True
        assert component_mask.sum() > 0
        assert component_mask.shape == binary_mask.shape
        assert slice_idx is None

    def test_extract_sampling_region_non_cube_shape_2d(self):
        """2D extraction with non-cube mask shape (collapsed_dim=0)."""
        binary_mask = torch.zeros(5, 10, 20)
        binary_mask[2, 2:8, 3:15] = 1

        sampling_config = {
            'dimensionality': 2,
            'collapsed_dim': 0,
            'region_extraction_config': {
                'connectivity': 1,
                'component_selection_process': 'top-k',
                'top-k': 1,
                'slice_selection': 'center'
            }
        }

        is_compatible, component_mask, slice_idx = extract_sampling_region(
            binary_mask, sampling_config
        )

        assert is_compatible is True
        assert component_mask.sum() > 0
        assert slice_idx == 2

    def test_extract_sampling_region_3d_multi_component(self):
        """3D mask with multiple disconnected components."""
        binary_mask = torch.zeros(12, 12, 12)
        binary_mask[0:4, 0:4, 0:4] = 1
        binary_mask[8:12, 8:12, 8:12] = 1

        sampling_config = {
            'dimensionality': 3,
            'region_extraction_config': {
                'connectivity': 1,
                'component_selection_process': 'top-k',
                'top-k': 1
            }
        }

        is_compatible, component_mask, slice_idx = extract_sampling_region(
            binary_mask, sampling_config
        )

        assert is_compatible is True
        assert component_mask.sum() > 0
        assert slice_idx is None

    def test_extract_sampling_region_mask_touching_boundary(self):
        """Mask with foreground touching volume boundary."""
        binary_mask = torch.zeros(10, 10, 10)
        binary_mask[0:3, 0:3, 0:3] = 1

        sampling_config = {
            'dimensionality': 3,
            'region_extraction_config': {
                'connectivity': 1,
                'component_selection_process': 'top-k',
                'top-k': 1
            }
        }

        is_compatible, component_mask, slice_idx = extract_sampling_region(
            binary_mask, sampling_config
        )

        assert is_compatible is True
        assert component_mask.sum() > 0
        assert component_mask[0:3, 0:3, 0:3].sum() > 0
        assert slice_idx is None


# ============================================================================
# Test bbox_from_binary_mask
# ============================================================================

class TestBboxFromBinaryMask:
    """Tests for the main bbox generation function."""
    
    def test_bbox_from_binary_mask_3d(self):
        """Test 3D bbox generation."""
        binary_mask = torch.zeros(10, 10, 10)
        binary_mask[2:7, 3:8, 1:6] = 1
        
        args = {
            'dimensionality': 3,
            'component_sampling_config': {
                'dimensionality': 3,
                'region_extraction_config': {
                    'connectivity': 1,
                    'component_selection_process': 'top-k',
                    'top-k': 1
                }
            }
        }
        
        bbox, is_generated = bbox_from_binary_mask(binary_mask, args)
        
        assert is_generated is True
        assert bbox is not None
        assert bbox.shape == (1, 6)
        assert bbox[0, 0] == 2  # min_x
        assert bbox[0, 3] == 6  # max_x
    
    def test_bbox_from_binary_mask_2d(self):
        """Test 2D bbox generation."""
        binary_mask = torch.zeros(10, 10, 10)
        binary_mask[2:7, 3:8, 5] = 1
        
        args = {
            'dimensionality': 2,
            'collapsed_dim': 2,
            'component_sampling_config': {
                'dimensionality': 2,
                'collapsed_dim': 2,
                'region_extraction_config': {
                    'connectivity': 1,
                    'component_selection_process': 'top-k',
                    'top-k': 1,
                    'slice_selection': 'center'
                }
            }
        }
        
        bbox, is_generated = bbox_from_binary_mask(binary_mask, args)
        
        assert is_generated is True
        assert bbox is not None
        assert bbox.shape == (1, 6)
        assert bbox[0, 2] == 5  # collapsed dim
        assert bbox[0, 5] == 5  # collapsed dim
    
    def test_bbox_from_binary_mask_empty_mask(self):
        """Test bbox generation with empty mask."""
        binary_mask = torch.zeros(10, 10, 10)
        
        args = {
            'dimensionality': 3,
            'component_sampling_config': {
                'dimensionality': 3,
                'region_extraction_config': {
                    'connectivity': 1,
                    'component_selection_process': 'top-k',
                    'top-k': 1
                }
            }
        }
        
        bbox, is_generated = bbox_from_binary_mask(binary_mask, args)
        
        assert is_generated is False
        assert bbox is None
    
    def test_bbox_from_binary_mask_with_jitter(self):
        """Test bbox generation with jitter augmentation."""
        binary_mask = torch.zeros(10, 10, 10)
        binary_mask[2:7, 3:8, 1:6] = 1
        
        args = {
            'dimensionality': 3,
            'component_sampling_config': {
                'dimensionality': 3,
                'region_extraction_config': {
                    'connectivity': 1,
                    'component_selection_process': 'top-k',
                    'top-k': 1
                }
            },
            'augmentation_config': {
                'jitter': {
                    'dimensionality': {'expected_dimensionality': 3},
                    'jitter_config': {
                        'type': 'absolute',
                        'sampling_mechanism': 'uniform_integer',
                        'jitter_symmetric': True,
                    },
                    'jitter_parameterisation': torch.tensor([1, 1, 1]),
                    'context_parameters': ['image_dimensions', 'sampling_dimensions', 'bbox_extrema', 'collapsed_dim', 'expected_dimensionality']
                }
            }
        }
        
        bbox, is_generated = bbox_from_binary_mask(binary_mask, args)
        
        assert is_generated is True
        assert bbox is not None
        # Bbox should be slightly jittered
        assert bbox[0, 0] <= 3  # min_x was 2, can jitter by 1
        assert bbox[0, 3] >= 5  # max_x was 6, can jitter by 1

    # ==================== Config Validation ====================

    def test_bbox_from_binary_mask_args_none(self):
        """args=None raises ValueError."""
        binary_mask = torch.zeros(10, 10, 10)
        binary_mask[2:7, 3:8, 1:6] = 1
        with pytest.raises(ValueError, match="args cannot be None"):
            bbox_from_binary_mask(binary_mask, None)

    def test_bbox_from_binary_mask_missing_component_sampling_config(self):
        """Missing component_sampling_config raises ValueError."""
        binary_mask = torch.zeros(10, 10, 10)
        binary_mask[2:7, 3:8, 1:6] = 1
        args = {'dimensionality': 3}
        with pytest.raises(ValueError, match="component_sampling_config"):
            bbox_from_binary_mask(binary_mask, args)

    def test_bbox_from_binary_mask_missing_dimensionality_in_args(self):
        """Missing dimensionality in args raises KeyError."""
        binary_mask = torch.zeros(10, 10, 10)
        binary_mask[2:7, 3:8, 1:6] = 1
        args = {
            'component_sampling_config': {
                'dimensionality': 3,
                'region_extraction_config': {
                    'connectivity': 1,
                    'component_selection_process': 'top-k',
                    'top-k': 1
                }
            }
        }
        with pytest.raises(KeyError, match="dimensionality"):
            bbox_from_binary_mask(binary_mask, args)

    def test_bbox_from_binary_mask_dimensionality_mismatch(self):
        """Mismatch between args and component_sampling_config dimensionality raises AssertionError."""
        binary_mask = torch.zeros(10, 10, 10)
        binary_mask[2:7, 3:8, 1:6] = 1
        args = {
            'dimensionality': 2,
            'component_sampling_config': {
                'dimensionality': 3,
                'region_extraction_config': {
                    'connectivity': 1,
                    'component_selection_process': 'top-k',
                    'top-k': 1
                }
            }
        }
        with pytest.raises(AssertionError, match="Dimensionality specified in args"):
            bbox_from_binary_mask(binary_mask, args)

    def test_bbox_from_binary_mask_2d_collapsed_dim_mismatch(self):
        """Mismatch in collapsed_dim between args and component_sampling_config raises ValueError."""
        binary_mask = torch.zeros(10, 10, 10)
        binary_mask[2:7, 3:8, 5] = 1
        args = {
            'dimensionality': 2,
            'collapsed_dim': 2,
            'component_sampling_config': {
                'dimensionality': 2,
                'collapsed_dim': 0,
                'region_extraction_config': {
                    'connectivity': 1,
                    'component_selection_process': 'top-k',
                    'top-k': 1,
                    'slice_selection': 'center'
                }
            }
        }
        with pytest.raises(ValueError, match="collapsed dimension"):
            bbox_from_binary_mask(binary_mask, args)

    def test_bbox_from_binary_mask_invalid_connectivity(self):
        """Invalid connectivity value raises ValueError."""
        binary_mask = torch.zeros(10, 10, 10)
        binary_mask[2:7, 3:8, 1:6] = 1
        args = {
            'dimensionality': 3,
            'component_sampling_config': {
                'dimensionality': 3,
                'region_extraction_config': {
                    'connectivity': 4,
                    'component_selection_process': 'top-k',
                    'top-k': 1
                }
            }
        }
        with pytest.raises(ValueError, match="connectivity"):
            bbox_from_binary_mask(binary_mask, args)

    def test_bbox_from_binary_mask_missing_region_extraction_config(self):
        """Missing region_extraction_config raises KeyError."""
        binary_mask = torch.zeros(10, 10, 10)
        binary_mask[2:7, 3:8, 1:6] = 1
        args = {
            'dimensionality': 3,
            'component_sampling_config': {
                'dimensionality': 3
            }
        }
        with pytest.raises(KeyError, match="region_extraction_config"):
            bbox_from_binary_mask(binary_mask, args)

    def test_bbox_from_binary_mask_missing_component_selection_key(self):
        """component_selection_process set but parameter key missing raises KeyError."""
        binary_mask = torch.zeros(10, 10, 10)
        binary_mask[2:7, 3:8, 1:6] = 1
        args = {
            'dimensionality': 3,
            'component_sampling_config': {
                'dimensionality': 3,
                'region_extraction_config': {
                    'connectivity': 1,
                    'component_selection_process': 'top-k'
                }
            }
        }
        with pytest.raises(KeyError, match="top-k"):
            bbox_from_binary_mask(binary_mask, args)

    # ==================== Incompatibility Flag Tests ====================

    def test_bbox_from_binary_mask_incompatible_2d_nontrivial(self):
        """2D nontrivial incompatible mask (passes contiguity, fails CC) returns (None, False)."""
        binary_mask = torch.zeros(10, 10, 10)
        binary_mask[list(range(10)), list(range(10)), [3] * len(list(range(10)))] = 1

        args = {
            'dimensionality': 2,
            'collapsed_dim': 2,
            'component_sampling_config': {
                'dimensionality': 2,
                'collapsed_dim': 2,
                'region_extraction_config': {
                    'connectivity': 1,
                    'component_selection_process': 'top-k',
                    'top-k': 1,
                    'slice_selection': 'center'
                }
            }
        }

        bbox, is_generated = bbox_from_binary_mask(binary_mask, args)

        assert is_generated is False
        assert bbox is None

    # ==================== Augmentation Edge Cases ====================

    def test_bbox_from_binary_mask_unsupported_augmentation(self):
        """Unsupported augmentation name raises ValueError."""
        binary_mask = torch.zeros(10, 10, 10)
        binary_mask[2:7, 3:8, 1:6] = 1

        args = {
            'dimensionality': 3,
            'component_sampling_config': {
                'dimensionality': 3,
                'region_extraction_config': {
                    'connectivity': 1,
                    'component_selection_process': 'top-k',
                    'top-k': 1
                }
            },
            'augmentation_config': {
                'unsupported_aug': {
                    'context_parameters': ['image_dimensions']
                }
            }
        }

        with pytest.raises(ValueError, match="augmentation"):
            bbox_from_binary_mask(binary_mask, args)


if __name__ == '__main__':
    pytest.main([__file__, '-v'])