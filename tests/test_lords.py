"""Unit tests for LoRDS standalone module."""

import torch
import pytest
import sys
import os
import importlib.util

# Direct import to avoid llmc __init__ chain pulling in all dependencies
_lords_path = os.path.join(
    os.path.dirname(__file__), '..', 'llmc', 'compression', 'quantization', 'lords.py'
)
_spec = importlib.util.spec_from_file_location('lords', _lords_path)
_lords = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_lords)

get_int4_lut = _lords.get_int4_lut
calculate_equivalent_rank = _lords.calculate_equivalent_rank
lords_init_BA = _lords.lords_init_BA
lords_find_best_Q = _lords.lords_find_best_Q
quantize_lords = _lords.quantize_lords


class TestGetInt4Lut:
    def test_shape_and_values(self):
        lut = get_int4_lut(device='cpu')
        assert lut.shape == (16,)
        assert lut[0].item() == -8
        assert lut[-1].item() == 7
        assert lut.dtype == torch.float32


class TestCalculateEquivalentRank:
    def test_square_matrix(self):
        # (4096 * 4096) // (128 * 8192) = 16
        assert calculate_equivalent_rank(4096, 4096, 128) == 16

    def test_rectangular_matrix(self):
        # (4096 * 11008) // (128 * 15104) = 23
        r = calculate_equivalent_rank(4096, 11008, 128)
        assert r > 0
        # Verify parameter budget: m*r + r*n ≈ m*n / block_size
        m, n, bs = 4096, 11008, 128
        lr_params = m * r + r * n
        bw_params = (m * n) // bs
        assert abs(lr_params - bw_params) / bw_params < 0.1  # within 10%


class TestLordsInitBA:
    def test_init_W_shapes(self):
        W = torch.randn(256, 512)
        r = 8
        B, A = lords_init_BA(W, r, init='W')
        assert B.shape == (256, 8)
        assert A.shape == (8, 512)
        assert B.dtype == torch.float32
        assert A.dtype == torch.float32

    def test_init_scale_shapes(self):
        W = torch.randn(256, 512)
        r = 4
        B, A = lords_init_BA(W, r, init='scale', block_size=128)
        assert B.shape == (256, 4)
        assert A.shape == (4, 512)

    def test_init_scale_requires_divisible(self):
        # total elements must be divisible by block_size
        W = torch.randn(256, 256)
        B, A = lords_init_BA(W, 4, init='scale', block_size=64)
        assert B.shape == (256, 4)


class TestLordsFindBestQ:
    def test_output_shape(self):
        W = torch.randn(32, 64)
        S = torch.abs(torch.randn(32, 64)) + 0.1
        lut = get_int4_lut(device='cpu')
        Q = lords_find_best_Q(W, S, lut)
        assert Q.shape == (32, 64)

    def test_values_in_lut(self):
        W = torch.randn(16, 16)
        S = torch.abs(torch.randn(16, 16)) + 0.1
        lut = get_int4_lut(device='cpu')
        Q = lords_find_best_Q(W, S, lut)
        # All values in Q should be from the LUT
        unique_vals = Q.unique()
        for v in unique_vals:
            assert v.item() in [i for i in range(-8, 8)]


class TestQuantizeLords:
    def test_output_shapes(self):
        W = torch.randn(256, 256)
        lut = get_int4_lut(device='cpu')
        W_hat, B, A = quantize_lords(W, rank=8, lut=lut, steps=5, lr=0.01)
        assert W_hat.shape == W.shape
        assert B.shape == (256, 8)
        assert A.shape == (8, 256)

    def test_loss_decreases(self):
        torch.manual_seed(42)
        W = torch.randn(256, 256)
        lut = get_int4_lut(device='cpu')

        # Run with 0 steps (just initialization)
        W_hat_0, _, _ = quantize_lords(W, rank=8, lut=lut, steps=0, lr=0.01)
        loss_0 = torch.mean((W_hat_0 - W) ** 2).item()

        # Run with 50 steps
        torch.manual_seed(42)
        W_hat_50, _, _ = quantize_lords(W, rank=8, lut=lut, steps=50, lr=0.01)
        loss_50 = torch.mean((W_hat_50 - W) ** 2).item()

        assert loss_50 < loss_0, f'Loss did not decrease: {loss_50} >= {loss_0}'

    def test_init_scale(self):
        W = torch.randn(256, 256)
        lut = get_int4_lut(device='cpu')
        W_hat, B, A = quantize_lords(
            W, rank=8, lut=lut, steps=5, lr=0.01,
            init='scale', block_size=64
        )
        assert W_hat.shape == W.shape


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
