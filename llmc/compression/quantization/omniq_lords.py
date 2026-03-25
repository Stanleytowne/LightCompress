"""
OmniQuant + LoRDS: Combines OmniQuant's LET (Learnable Equivalent Transformation)
with LoRDS (Low-Rank Decomposed Scaling) for improved low-bit quantization.

Pipeline per block:
  1. Full OmniQuant (LET + LWC joint training) — block_transform via super()
     During training, w_qdq uses standard block-wise INT4 (_lords_deploy=False).
  2. After training, replace_module_block calls w_qdq for each weight matrix.
     At this point _lords_deploy=True, so w_qdq runs LoRDS instead of block-wise INT4.
"""

import torch
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

    def block_transform(self, block, input_feat, block_kwargs):
        # Disable LoRDS during LET+LWC training.
        # w_qdq is called repeatedly during omni_train and must use
        # standard block-wise INT4 quantization.
        self._lords_deploy = False

        # Phase 1: Full OmniQuant (LET + LWC joint training)
        super().block_transform(block, input_feat, block_kwargs)

        # Enable LoRDS for the deployment step that follows.
        # run() will call replace_module_block → w_qdq for each weight matrix,
        # which is where LoRDS actually runs.
        self._lords_deploy = True

    def w_qdq(self, module, wquantizer):
        """
        Weight quantize-dequantize override.

        Called in two contexts:
        1. During LET+LWC training (_lords_deploy=False):
           → standard OmniQuant block-wise INT4 quantization.
        2. During deployment (_lords_deploy=True):
           → LoRDS low-rank scaling, called once per weight matrix.
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

        logger.info(
            f'  Block {self.block_idx} LoRDS: '
            f'shape=({m},{n}), rank={rank}'
        )

        self.lords_lut = self.lords_lut.to(W.device)

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

        return W_hat.to(module.weight.dtype)
