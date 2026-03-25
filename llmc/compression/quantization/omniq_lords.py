"""
OmniQuant + LoRDS: Combines OmniQuant's LET (Learnable Equivalent Transformation)
with LoRDS (Low-Rank Decomposed Scaling) for improved low-bit quantization.

Pipeline:
  1. LET transforms weights to be more quantization-friendly (from OmniQuant)
  2. LoRDS replaces block-wise INT4 scaling with low-rank per-element scaling
"""

import torch
from loguru import logger

from llmc.utils.registry_factory import ALGO_REGISTRY

from .lords import calculate_equivalent_rank, get_int4_lut, quantize_lords
from .module_utils import FakeQuantLinear
from .omniq import OmniQuant


@ALGO_REGISTRY
class OmniQuantLoRDS(OmniQuant):
    def __init__(self, model, quant_config, input, padding_mask, config):
        super().__init__(model, quant_config, input, padding_mask, config)

    def add_quant_config(self):
        super().add_quant_config()

        # Keep LWC enabled during LET+LWC joint training.
        # LWC provides better quantization noise simulation, helping LET
        # learn better transforms. LWC bounds are discarded after training;
        # LoRDS replaces block-wise quantization in the final step.

        # LoRDS-specific config
        config = self.quant_config['special']
        self.lords_rank = config.get('lords_rank', 'auto')
        self.lords_steps = config.get('lords_steps', 50)
        self.lords_lr = config.get('lords_lr', 0.05)
        self.lords_init = config.get('lords_init', 'W')
        self.lords_block_size = config.get('lords_block_size', 128)
        self.lords_patience = config.get('lords_patience', 20)
        self.lords_min_improvement = config.get('lords_min_improvement', 1e-6)
        self.lords_log_interval = config.get('lords_log_interval', 10)

        # Build INT4 LUT
        self.lords_lut = get_int4_lut(device='cuda')

        # Allow epochs=0 with let=False (LoRDS-only mode, no LET training)
        # Override parent's assertion: assert self.lwc or self.let
        # (parent checks this only when epochs > 0)

    def block_transform(self, block, input_feat, block_kwargs):
        logger.info(f'Start transform the {self.block_idx}-th block')

        # Phase 1: Full OmniQuant (LET + LWC joint training)
        # LWC provides better quantization noise → LET learns better transforms.
        super().block_transform(block, input_feat, block_kwargs)

        # Phase 2: LoRDS refinement on each linear layer
        # At this point, LET transforms have been permanently applied to weights.
        # LoRDS replaces block-wise INT4 quantization with low-rank scaling.
        self.lords_refine_block(block)

        logger.info(f'End transform the {self.block_idx}-th block')

    def lords_refine_block(self, block):
        """Apply LoRDS quantization to each linear layer in the block."""
        self.lords_lut = self.lords_lut.to(next(block.parameters()).device)

        for name, module in block.named_modules():
            if not self._is_linear(module):
                continue

            W = module.weight.data.float()
            m, n = W.shape

            # Determine rank
            if self.lords_rank == 'auto':
                rank = calculate_equivalent_rank(m, n, self.lords_block_size)
                rank = max(rank, 1)
            else:
                rank = int(self.lords_rank)

            logger.info(
                f'  Block {self.block_idx} layer {name}: '
                f'shape=({m},{n}), rank={rank}'
            )

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
            )

            # Replace weight with LoRDS result
            module.weight.data = W_hat.to(module.weight.dtype)
            module.lords_quantized = True

    def _is_linear(self, module):
        """Check if module is a linear layer (original or FakeQuant wrapped)."""
        if isinstance(module, FakeQuantLinear):
            return True
        if isinstance(module, torch.nn.Linear):
            return True
        return False

    def w_qdq(self, module, wquantizer):
        """
        Weight quantize-dequantize override.

        If the module has been LoRDS-quantized, return the weight as-is
        (it already contains the fake-quantized result W_hat = (B@A)*Q).
        Otherwise, fall back to standard OmniQuant w_qdq (used during LET training).
        """
        if getattr(module, 'lords_quantized', False):
            return module.weight
        return super().w_qdq(module, wquantizer)
