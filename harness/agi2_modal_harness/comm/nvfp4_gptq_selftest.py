"""Self-test of the NVFP4 GPTQ (nvfp4_gptq.gptq) on synthetic expert-shaped data, before it touches the real build.
Random W [E, 640, 2560] ~ N(0, 0.02); inputs with correlated channels and a few large-magnitude outlier channels;
H = X^T X. Checks: every chosen block scale is a legal E4M3 value; codes decode in range; output-error proxy
tr(dW H dW^T) orders GPTQ < QS (exact best scales) < RTN (max scales); prints the ratios and the time.

  python -m agi2_modal_harness.comm.nvfp4_gptq_selftest
"""
import time

import torch

from agi2_modal_harness.comm.nvfp4_gptq import gptq, proxy
from agi2_modal_harness.comm.nvfp4_reround import E4M3_POS, dequant, rtn_codes, search_scales


def main() -> int:
    torch.manual_seed(0)
    torch.backends.cuda.matmul.allow_tf32 = False
    dev = torch.device("cuda")
    E, R, C, N = 8, 640, 2560, 6000
    W = torch.randn(E, R, C, device=dev) * 0.02
    mix = torch.randn(C, C, device=dev) / C ** 0.5 * 0.3 + torch.eye(C, device=dev)
    X = torch.randn(E, N, C, device=dev) @ mix
    X[:, :, :8] *= 20.0  # outlier channels
    H = X.transpose(1, 2) @ X
    g = (W.abs().amax((1, 2)) / (6 * 448)).float()
    base = (W.view(E, R, C // 16, 16).abs().amax(-1) / 6 / g.view(E, 1, 1))
    table = E4M3_POS.to(dev)
    idx = torch.bucketize(base, table).clamp(0, len(table) - 1)
    lo = table[(idx - 1).clamp(min=0)]
    hi = table[idx]
    base = torch.where((base - lo).abs() <= (hi - base).abs(), lo, hi)  # nearest E4M3, as ModelOpt's max recipe
    t = time.time()
    c_rtn = rtn_codes(W, g, base)
    s_qs, c_qs = search_scales(W, g, base)
    c_g, s_g = gptq(W, H, g)
    torch.cuda.synchronize()
    el = time.time() - t
    legal = bool(torch.isin(s_g.flatten(), table).all()) and bool(torch.isin(s_qs.flatten(), table).all())
    p_rtn = proxy(W - dequant(c_rtn, base, g), H)
    p_qs = proxy(W - dequant(c_qs, s_qs, g), H)
    p_g = proxy(W - dequant(c_g, s_g, g), H)
    w_rtn = ((W - dequant(c_rtn, base, g)) ** 2).sum() / (W ** 2).sum()
    w_qs = ((W - dequant(c_qs, s_qs, g)) ** 2).sum() / (W ** 2).sum()
    w_g = ((W - dequant(c_g, s_g, g)) ** 2).sum() / (W ** 2).sum()
    print(f"legal E4M3 scales: {legal}; codes in range: {int(c_g.max()) <= 15}")
    print(f"weight rel err: rtn {w_rtn:.5f} qs {w_qs:.5f} gptq {w_g:.5f}")
    print(f"output proxy vs rtn: qs {float((p_qs / p_rtn).mean()):.3f} gptq {float((p_g / p_rtn).mean()):.3f}")
    print(f"gptq + qs + rtn for {E} experts of 640x2560 in {el:.1f} s")
    ok = legal and bool((p_g < p_qs).all()) and bool((p_qs < p_rtn).all())
    print("SELFTEST", "PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
