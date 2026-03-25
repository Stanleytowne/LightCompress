"""
LoRDS: Low-Rank Decomposed Scaling for LLM Quantization.

Standalone module with no llmc framework dependencies.
Replaces block-wise quantization scaling with low-rank per-element scaling:
    W ≈ (B @ A) ⊙ Q
where B (m, r), A (r, n) parameterize the scaling matrix S = B @ A,
and Q (m, n) contains INT4 quantized values from a lookup table.
"""

from typing import Literal, Tuple

import torch

try:
    from loguru import logger
except ImportError:
    import logging
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)


def get_int4_lut(device: str = 'cuda') -> torch.Tensor:
    """Return INT4 signed lookup table: [-8, -7, ..., 7], shape (16,)."""
    return torch.arange(-8, 8, dtype=torch.float32, device=device)


def calculate_equivalent_rank(m: int, n: int, block_size: int) -> int:
    """
    Compute rank r such that low-rank params (m*r + r*n) equals
    block-wise scale params (m*n / block_size).

    r = (m * n) // (block_size * (m + n))
    """
    return (m * n) // (block_size * (m + n))


def lords_init_BA(
    W: torch.Tensor,
    r: int,
    init: Literal['W', 'scale'] = 'W',
    block_size: int = 128,
    scale_matrix: torch.Tensor = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Initialize B (m, r) and A (r, n) via truncated SVD.

    Args:
        W: Weight matrix, shape (m, n).
        r: Target rank.
        init: Initialization strategy.
            - "W": SVD of |W|.
            - "scale": Compute per-group absmax scales, expand to (m, n), then SVD.
            - "lwc": Use pre-computed LWC-optimized scale matrix (pass via scale_matrix).
        block_size: Group size for "scale" initialization.
        scale_matrix: Pre-computed per-element scale matrix, shape (m, n).
            Required when init="lwc".

    Returns:
        (B, A) as float32 tensors.
    """
    m, n = W.shape
    device = W.device

    if init == 'W':
        target = torch.abs(W).float()
    elif init == 'scale':
        # Compute per-group absmax without bitsandbytes
        W_flat = W.float().reshape(-1, block_size)
        absmax = W_flat.abs().amax(dim=-1, keepdim=True)  # (num_groups, 1)
        scale = absmax.repeat(1, block_size).reshape(m, n)  # (m, n)
        target = scale
    elif init == 'lwc':
        assert scale_matrix is not None, "scale_matrix required for init='lwc'"
        target = scale_matrix.float()
    else:
        raise ValueError(f"Unknown init method: {init}")

    U, S, Vh = torch.linalg.svd(target, full_matrices=False)
    U_r = U[:, :r]
    S_r = torch.diag(S[:r])
    Vh_r = Vh[:r, :]
    sqrt_S = torch.sqrt(S_r)

    B = (U_r @ sqrt_S).to(torch.float32)
    A = (sqrt_S @ Vh_r).to(torch.float32)
    return B.to(device), A.to(device)


def lords_find_best_Q(
    W: torch.Tensor,
    S: torch.Tensor,
    lut: torch.Tensor,
) -> torch.Tensor:
    """
    Given scaling matrix S, find optimal Q for each element.

    Q_ij = argmin_q (S_ij * q - W_ij)^2

    Args:
        W: Original weight matrix, shape (m, n).
        S: Scaling matrix (B @ A), shape (m, n).
        lut: Lookup table, shape (num_levels,).

    Returns:
        Q: Quantized values from lut, shape (m, n).
    """
    W_exp = W.unsqueeze(-1)           # (m, n, 1)
    S_exp = S.unsqueeze(-1)           # (m, n, 1)
    lut_exp = lut.view(1, 1, -1)      # (1, 1, num_levels)

    recons_cands = S_exp * lut_exp    # (m, n, num_levels)
    dists = (recons_cands - W_exp).pow(2)
    min_indices = torch.argmin(dists, dim=-1)
    return lut[min_indices]


@torch.no_grad()
def quantize_lords(
    W: torch.Tensor,
    rank: int,
    lut: torch.Tensor,
    steps: int = 50,
    lr: float = 1e-3,
    init: Literal['W', 'scale', 'lwc'] = 'W',
    block_size: int = 128,
    patience: int = 20,
    min_improvement: float = 1e-6,
    log_interval: int = 10,
    scale_matrix: torch.Tensor = None,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    LoRDS alternating optimization: W ≈ (B @ A) ⊙ Q.

    Args:
        W: Weight matrix, shape (m, n), any dtype.
        rank: Low-rank dimension r.
        lut: INT4 lookup table, shape (16,).
        steps: Maximum number of alternating optimization steps.
        lr: Learning rate for B, A optimization.
        init: Initialization method ("W", "scale", or "lwc").
        block_size: Group size for "scale" init and equivalent rank calculation.
        patience: Early stopping patience. Stop if no improvement for this many steps.
        min_improvement: Minimum relative improvement to reset patience counter.
        log_interval: Log every N steps.
        scale_matrix: Pre-computed per-element scale matrix for init="lwc".

    Returns:
        (W_hat, B, A) where W_hat = (B @ A) * Q is the fake-quantized weight.
    """
    W = W.float()
    m, n = W.shape
    device = W.device

    # Initialize B, A
    B, A = lords_init_BA(W, rank, init=init, block_size=block_size,
                         scale_matrix=scale_matrix)
    B = B.clone().detach().requires_grad_(True)
    A = A.clone().detach().requires_grad_(True)

    optimizer = torch.optim.AdamW([B, A], lr=lr)

    # Initial Q
    with torch.no_grad():
        S = B @ A
        Q = lords_find_best_Q(W, S, lut)

    # Early stopping state
    best_loss = float('inf')
    no_improve_count = 0
    actual_steps = 0

    with torch.enable_grad():
        for step in range(steps):
            # Step A: Fix B, A → update Q
            with torch.no_grad():
                S = B @ A
                Q = lords_find_best_Q(W, S, lut)

            # Step B: Fix Q → optimize B, A
            optimizer.zero_grad()
            S = B @ A
            W_hat = S * Q
            loss = torch.mean((W_hat - W) ** 2)
            loss.backward()
            optimizer.step()

            current_loss = loss.item()
            actual_steps = step + 1

            # Logging
            if step % log_interval == 0 or step == steps - 1:
                logger.info(
                    f'  LoRDS step {step:>4d}/{steps}, '
                    f'loss={current_loss:.6e}, '
                    f'best={best_loss:.6e}, '
                    f'no_improve={no_improve_count}/{patience}'
                )

            # Early stopping check
            relative_improvement = (best_loss - current_loss) / max(best_loss, 1e-10)
            if current_loss < best_loss and relative_improvement > min_improvement:
                best_loss = current_loss
                no_improve_count = 0
            else:
                no_improve_count += 1

            if no_improve_count >= patience:
                logger.info(
                    f'  LoRDS early stop at step {step}, '
                    f'no improvement for {patience} steps'
                )
                break

    # Final reconstruction
    with torch.no_grad():
        S = B @ A
        Q = lords_find_best_Q(W, S, lut)
        W_hat = S * Q
        final_loss = torch.mean((W_hat - W) ** 2).item()

    logger.info(
        f'  LoRDS done: shape=({m},{n}), rank={rank}, '
        f'steps={actual_steps}/{steps}, '
        f'final_loss={final_loss:.6e}'
    )

    return W_hat, B.detach(), A.detach()
