"""
OmniQuant + LoRDS: Combines OmniQuant's LET (Learnable Equivalent Transformation)
with LoRDS (Low-Rank Decomposed Scaling) for improved low-bit quantization.

Pipeline:
  1. Full OmniQuant (LET + LWC joint training) per block — standard block_transform.
     w_qdq uses block-wise INT4 during training and block_forward.
  2. At deploy time (EffcientFakeQuantLinear.new calls w_qdq for each weight),
     LoRDS replaces block-wise INT4 with low-rank per-element scaling.
"""

import torch
import torch.nn as nn
from loguru import logger

from llmc.utils.registry_factory import ALGO_REGISTRY

from .lords import calculate_equivalent_rank, get_int4_lut, quantize_lords
from .omniq import OmniQuant


@ALGO_REGISTRY
class OmniQuantLoRDS(OmniQuant):
    def __init__(self, model, quant_config, input, padding_mask, config):
        super().__init__(model, quant_config, input, padding_mask, config)
        self._lords_deploy = False

    def add_quant_config(self):
        super().add_quant_config()

        # LoRDS-specific config
        config = self.quant_config['special']
        self.lords_rank = config.get('lords_rank', 'auto')
        self.lords_steps = int(config.get('lords_steps', 50))
        self.lords_lr = float(config.get('lords_lr', 0.05))
        self.lords_init = config.get('lords_init', 'W')
        self.lords_block_size = int(config.get('lords_block_size', 128))
        self.lords_patience = int(config.get('lords_patience', 20))
        self.lords_min_improvement = float(config.get('lords_min_improvement', 1e-6))
        self.lords_log_interval = int(config.get('lords_log_interval', 10))

        # Build INT4 LUT
        self.lords_lut = get_int4_lut(device='cuda')

    def deploy(self, quant_format, keep_device=False):
        # Enable LoRDS before deploy.
        # deploy() calls replace_module_all → EffcientFakeQuantLinear.new → w_qdq
        # for each weight matrix. This is where LoRDS actually runs.
        self._lords_deploy = True
        super().deploy(quant_format, keep_device=keep_device)

    def _compute_lwc_scale_matrix(self, module, W):
        """
        Compute the LWC-optimized per-element scale matrix from learned
        clipping bounds (upbound_factor, lowbound_factor).
        """
        group_size = self.lords_block_size
        m, n = W.shape
        sigmoid = nn.Sigmoid()

        # Reshape weight into groups
        W_grouped = W.reshape(-1, group_size)

        # Per-group min/max
        xmin = W_grouped.amin(dim=-1, keepdim=True)
        xmax = W_grouped.amax(dim=-1, keepdim=True)

        # Apply learned clipping bounds
        upbound = getattr(module, 'buf_upbound_factor', None)
        lowbound = getattr(module, 'buf_lowbound_factor', None)
        if upbound is not None and lowbound is not None:
            xmax = sigmoid(upbound) * xmax
            xmin = sigmoid(lowbound) * xmin

        # Symmetric scale
        abs_max = torch.max(xmax.abs(), xmin.abs())
        n_bits = self.quant_config['weight']['bit']
        scale = abs_max / (2 ** (n_bits - 1) - 1)
        scale = scale.clamp(min=1e-5)

        # Expand to full matrix
        scale_matrix = scale.repeat(1, group_size).reshape(m, n)
        return scale_matrix

    def w_qdq(self, module, wquantizer):
        """
        Weight quantize-dequantize override.

        Called in three contexts:
        1. During LET+LWC training (omni_train → FakeQuantLinear.forward):
           → standard OmniQuant block-wise INT4.
        2. During block_forward after replace_module_block:
           → standard OmniQuant block-wise INT4.
        3. During deploy (EffcientFakeQuantLinear.new):
           → LoRDS low-rank scaling.
        """
        if not self._lords_deploy:
            return super().w_qdq(module, wquantizer)

        # LoRDS: replace block-wise scaling with low-rank per-element scaling
        W = module.weight.data.float()
        m, n = W.shape

        if self.lords_rank == 'auto':
            rank = calculate_equivalent_rank(m, n, self.lords_block_size)
            rank = max(rank, 1)
        else:
            rank = int(self.lords_rank)

        # Compute OmniQuant baseline error for comparison
        W_omniq = super().w_qdq(module, wquantizer).float()
        omniq_mse = torch.mean((W_omniq - W) ** 2).item()

        logger.info(
            f'  LoRDS: shape=({m},{n}), rank={rank}, init={self.lords_init}, '
            f'omniq_mse={omniq_mse:.6e}'
        )

        self.lords_lut = self.lords_lut.to(W.device)

        # For init="lwc", compute scale matrix from LWC's learned clipping bounds
        scale_matrix = None
        if self.lords_init == 'lwc':
            scale_matrix = self._compute_lwc_scale_matrix(module, W)

        W_hat, B, A = quantize_lords(
            W,
            rank=rank,
            lut=self.lords_lut,
            steps=self.lords_steps,
            lr=self.lords_lr,
            init=self.lords_init,
            block_size=self.lords_block_size,
            patience=self.lords_patience,
            min_improvement=self.lords_min_improvement,
            log_interval=self.lords_log_interval,
            scale_matrix=scale_matrix,
        )

        lords_mse = torch.mean((W_hat - W) ** 2).item()
        improvement = (omniq_mse - lords_mse) / max(omniq_mse, 1e-10) * 100

        logger.info(
            f'  LoRDS result: '
            f'omniq_mse={omniq_mse:.6e} → lords_mse={lords_mse:.6e} '
            f'({improvement:+.1f}%)'
        )

        return W_hat.to(module.weight.dtype)
