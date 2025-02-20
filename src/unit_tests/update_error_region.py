import unittest
import torch
# from your_module import create_binary_mask  # Make sure to import the function from your module

class TestModifyBinaryMask(unittest.TestCase):

    # Test 1: 1D Mask
    def test_1d_valid(self):
        coords_1d_valid = torch.tensor([[0], [2]], dtype=torch.int32)
        mask_1d = torch.tensor([1, 1, 1], dtype=torch.float32)
        expected = torch.tensor([0, 1, 0], dtype=torch.float32)
        result = create_binary_mask(coords_1d_valid, mask_1d)
        torch.testing.assert_allclose(result, expected)

    def test_1d_invalid(self):
        coords_1d_invalid = torch.tensor([[0], [3]], dtype=torch.int32)
        mask_1d_invalid = torch.tensor([1, 1, 1], dtype=torch.float32)
        expected = torch.tensor([1, 1, 1], dtype=torch.float32)
        result = create_binary_mask(coords_1d_invalid, mask_1d_invalid)
        torch.testing.assert_allclose(result, expected)

    def test_1d_empty(self):
        coords_1d_empty = torch.tensor([], dtype=torch.int32).reshape(0, 1)
        mask_1d_empty = torch.tensor([1, 1, 1], dtype=torch.float32)
        expected = torch.tensor([1, 1, 1], dtype=torch.float32)
        result = create_binary_mask(coords_1d_empty, mask_1d_empty)
        torch.testing.assert_allclose(result, expected)

    # Test 2: 2D Mask
    def test_2d_valid(self):
        coords_2d_valid = torch.tensor([[0, 0], [1, 1]], dtype=torch.int32)
        mask_2d = torch.tensor([[1, 1], [1, 1]], dtype=torch.float32)
        expected = torch.tensor([[0, 1], [1, 0]], dtype=torch.float32)
        result = create_binary_mask(coords_2d_valid, mask_2d)
        torch.testing.assert_allclose(result, expected)

    def test_2d_invalid(self):
        coords_2d_invalid = torch.tensor([[0, 0], [2, 2]], dtype=torch.int32)
        mask_2d_invalid = torch.tensor([[1, 1], [1, 1]], dtype=torch.float32)
        expected = torch.tensor([[1, 1], [1, 1]], dtype=torch.float32)
        result = create_binary_mask(coords_2d_invalid, mask_2d_invalid)
        torch.testing.assert_allclose(result, expected)

    def test_2d_empty(self):
        coords_2d_empty = torch.tensor([], dtype=torch.int32).reshape(0, 2)
        mask_2d_empty = torch.tensor([[1, 1], [1, 1]], dtype=torch.float32)
        expected = torch.tensor([[1, 1], [1, 1]], dtype=torch.float32)
        result = create_binary_mask(coords_2d_empty, mask_2d_empty)
        torch.testing.assert_allclose(result, expected)

    # Test 3: 3D Mask
    def test_3d_valid(self):
        coords_3d_valid = torch.tensor([[0, 0, 0], [1, 1, 1]], dtype=torch.int32)
        mask_3d = torch.ones((2, 2, 2), dtype=torch.float32)
        expected = torch.tensor([[[0, 1], [1, 1]], [[1, 1], [1, 0]]], dtype=torch.float32)
        result = create_binary_mask(coords_3d_valid, mask_3d)
        torch.testing.assert_allclose(result, expected)

    def test_3d_invalid(self):
        coords_3d_invalid = torch.tensor([[0, 0, 0], [2, 2, 2]], dtype=torch.int32)
        mask_3d_invalid = torch.ones((2, 2, 2), dtype=torch.float32)
        expected = torch.ones((2, 2, 2), dtype=torch.float32)
        result = create_binary_mask(coords_3d_invalid, mask_3d_invalid)
        torch.testing.assert_allclose(result, expected)

    def test_3d_empty(self):
        coords_3d_empty = torch.tensor([], dtype=torch.int32).reshape(0, 3)
        mask_3d_empty = torch.ones((2, 2, 2), dtype=torch.float32)
        expected = torch.ones((2, 2, 2), dtype=torch.float32)
        result = create_binary_mask(coords_3d_empty, mask_3d_empty)
        torch.testing.assert_allclose(result, expected)

if __name__ == "__main__":
    unittest.main()
