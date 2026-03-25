"""
OmniQuant + LoRDS: Combines OmniQuant's LET (Learnable Equivalent Transformation)
with LoRDS (Low-Rank Decomposed Scaling) for improved low-bit quantization.

Pipeline:
  1. Full OmniQuant (LET + LWC joint training) per block — standard block_transform.
     w_qdq uses block-wise INT4 during training and block_forward.
  2. At deploy time (EffcientFakeQuantLinear.new calls w_qdq for each weight),
     LoRDS replaces block-wise INT4 with low-rank per-element scaling.
  3. Both OmniQuant baseline and LoRDS models are saved in one run.
"""

import os

import torch
import torch.nn as nn
from loguru import logger

from llmc.utils.registry_factory import ALGO_REGISTRY

from itertools import chain

from .lords import calculate_equivalent_rank, get_int4_lut, quantize_lords
from .module_utils import EffcientFakeQuantLinear
from .omniq import OmniQuant


def _get_block_device(block):
    """Get device from block's parameters or buffers (fallback)."""
    try:
        return next(block.parameters()).device
    except StopIteration:
        return next(block.buffers()).device


@ALGO_REGISTRY
class OmniQuantLoRDS(OmniQuant):
    def __init__(self, model, quant_config, input, padding_mask, config):
        super().__init__(model, quant_config, input, padding_mask, config)
        self._lords_deploy = False
        self._omniq_weights = []

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

    def block_forward(self, block, input_data=None):
        """Override to fix StopIteration when block has no nn.Parameter
        (all weights are buffers after replace_module_block)."""
        output = []
        if input_data is None:
            input_data = self.input['data']

        device = _get_block_device(block)

        for i in range(len(input_data)):
            input_data[i] = input_data[i].to(device=device)
            if (
                'attention_mask' in self.input['kwargs'][i]
                and self.input['kwargs'][i]['attention_mask'] is not None
            ):
                self.input['kwargs'][i]['attention_mask'] = self.input['kwargs'][i][
                    'attention_mask'
                ].cuda()
            with torch.no_grad():
                with torch.cuda.amp.autocast():
                    out = block(input_data[i], **self.input['kwargs'][i])[0]
                    output.append(out)
        return output

    def deploy(self, quant_format, keep_device=False):
        # Enable LoRDS and collect OmniQuant baseline weights during deploy.
        self._lords_deploy = True
        self._omniq_weights = []
        super().deploy(quant_format, keep_device=keep_device)

        # Save OmniQuant baseline model (swap weights, save, swap back)
        self._save_omniq_baseline()

    def _save_omniq_baseline(self):
        """Save OmniQuant baseline fake quant model by swapping weights."""
        save_config = self.config.get('save', {})
        save_path = save_config.get('save_path', None)
        if not save_path or not self._omniq_weights:
            return

        omniq_path = os.path.join(save_path, 'omniq_baseline_model')
        os.makedirs(omniq_path, exist_ok=True)

        # Collect all EffcientFakeQuantLinear modules in block order
        efql_modules = []
        for block in self.blocks:
            block.cuda()
            for name, m in block.named_modules():
                if isinstance(m, EffcientFakeQuantLinear):
                    efql_modules.append(m)
            block.cpu()

        if len(efql_modules) != len(self._omniq_weights):
            logger.warning(
                f'Module count mismatch: {len(efql_modules)} modules vs '
                f'{len(self._omniq_weights)} saved weights. Skip saving baseline.'
            )
            return

        # Swap to OmniQuant weights
        lords_weights = []
        for i, m in enumerate(efql_modules):
            lords_weights.append(m.weight.data.clone())
            m.weight.data = self._omniq_weights[i]

        # Save OmniQuant baseline
        self.save_model(omniq_path)
        logger.info(f'Saved OmniQuant baseline model to {omniq_path}')

        # Swap back to LoRDS weights
        for i, m in enumerate(efql_modules):
            m.weight.data = lords_weights[i]

        # Free memory
        self._omniq_weights = []

    def _compute_lwc_scale_matrix(self, module, W):
        """
        Compute the LWC-optimized per-element scale matrix from learned
        clipping bounds (upbound_factor, lowbound_factor).
        """
        group_size = self.lords_block_size
        m, n = W.shape
        sigmoid = nn.Sigmoid()

        W_grouped = W.reshape(-1, group_size)
        xmin = W_grouped.amin(dim=-1, keepdim=True)
        xmax = W_grouped.amax(dim=-1, keepdim=True)

        upbound = getattr(module, 'buf_upbound_factor', None)
        lowbound = getattr(module, 'buf_lowbound_factor', None)
        if upbound is not None and lowbound is not None:
            xmax = sigmoid(upbound) * xmax
            xmin = sigmoid(lowbound) * xmin

        abs_max = torch.max(xmax.abs(), xmin.abs())
        n_bits = self.quant_config['weight']['bit']
        scale = abs_max / (2 ** (n_bits - 1) - 1)
        scale = scale.clamp(min=1e-5)

        scale_matrix = scale.repeat(1, group_size).reshape(m, n)
        return scale_matrix

    def w_qdq(self, module, wquantizer):
        """
        Weight quantize-dequantize override.

        During training/block_forward: standard OmniQuant block-wise INT4.
        During deploy: computes both OmniQuant and LoRDS, saves OmniQuant
        for later baseline export, returns LoRDS result.
        """
        if not self._lords_deploy:
            return super().w_qdq(module, wquantizer)

        W = module.weight.data.float()
        m, n = W.shape

        if self.lords_rank == 'auto':
            rank = calculate_equivalent_rank(m, n, self.lords_block_size)
            rank = max(rank, 1)
        else:
            rank = int(self.lords_rank)

        # Compute OmniQuant baseline and save for later export
        W_omniq = super().w_qdq(module, wquantizer)
        self._omniq_weights.append(W_omniq.clone().cpu())
        omniq_mse = torch.mean((W_omniq.float() - W) ** 2).item()

        logger.info(
            f'  LoRDS: shape=({m},{n}), rank={rank}, init={self.lords_init}, '
            f'omniq_mse={omniq_mse:.6e}'
        )

        self.lords_lut = self.lords_lut.to(W.device)

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
            f'omniq_mse={omniq_mse:.6e} -> lords_mse={lords_mse:.6e} '
            f'({improvement:+.1f}%)'
        )

        return W_hat.to(module.weight.dtype)
