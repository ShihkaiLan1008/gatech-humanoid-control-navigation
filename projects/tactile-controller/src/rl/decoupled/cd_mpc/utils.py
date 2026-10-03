from __future__ import annotations
import torch
from torch import Tensor


def torch_to_jax(t: Tensor):
    """Torch → JAX, zero-copy via dlpack when possible."""
    import jax
    t = t.contiguous()
    try:
        return jax.dlpack.from_dlpack(t)
    except Exception:
        from torch.utils.dlpack import to_dlpack
        return jax.dlpack.from_dlpack(to_dlpack(t))

def jax_to_torch(arr) -> Tensor:
    """JAX → Torch, zero-copy via dlpack when possible."""
    try:
        return torch.from_dlpack(arr)
    except Exception:
        import jax
        return torch.from_dlpack(jax.dlpack.to_dlpack(arr))

def skew(v: Tensor) -> Tensor:
    """Batch skew-symmetric matrix [v]_x, shape (..., 3) → (..., 3, 3)."""
    z = torch.zeros_like(v[..., 0])
    return torch.stack([
        torch.stack([z, -v[..., 2], v[..., 1]], dim=-1),
        torch.stack([v[..., 2], z, -v[..., 0]], dim=-1),
        torch.stack([-v[..., 1], v[..., 0], z], dim=-1),
    ], dim=-2)

def make_T6(R: Tensor) -> Tensor:
    """Build block-diagonal rotation T6 = block_diag(R^T, R^T), shape (B, 6, 6)."""
    B = R.shape[0]
    RT = R.transpose(-1, -2)
    T  = torch.zeros(B, 6, 6, device=R.device, dtype=R.dtype)
    T[:, :3, :3] = RT
    T[:, 3:, 3:] = RT
    return T
