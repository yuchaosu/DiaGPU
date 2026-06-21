#!/usr/bin/env python3.11
"""Headline figure: Pauli-basis operator-build vs assembled-DIA SpMSpM.
Both build U = sum_k c_k H^k (heis, K=6, dt=1.2e-3, H100). Measured 2026-06-15.
"""
import os
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

q   = [10, 12, 14]
asm_nnz   = [1_027_200, 15_494_272, 219_716_736]   # nnz of U (= H^6) dense-DIA
pau_terms = [550, 846, 1206]                        # Pauli terms in U (thr=1e-8)
asm_ms    = [0.75, 1.95, 15.75]                     # build time (SpGEMM+accum)
pau_ms    = [1.32, 1.34, 1.56]

fig, ax = plt.subplots(1, 2, figsize=(13, 5.2))

# Panel 1: storage / operator size (log)
ax[0].plot(q, asm_nnz,  "o-", color="#d62728", lw=2, ms=8, label="assembled DIA  (U nonzeros)")
ax[0].plot(q, pau_terms,"s-", color="#1f77b4", lw=2, ms=8, label="Pauli basis  (U terms)")
ax[0].set_yscale("log"); ax[0].set_xticks(q)
ax[0].set_xlabel("qubits  (n = 2^q)"); ax[0].set_ylabel("operator U size (log) — lower better")
ax[0].set_title("Operator storage: assembled explodes ~exp(q), Pauli ~linear(q)")
ax[0].annotate("180,000x\nsmaller", xy=(14, 1206), xytext=(12.3, 5e4),
               arrowprops=dict(arrowstyle="->"), fontsize=10, ha="center", color="#1f77b4")
ax[0].axvspan(14.2, 14.8, color="gray", alpha=0.15)
ax[0].text(14.5, 3e6, "assembled\nwalls q>14\n(grid.y / OOM)", fontsize=8, ha="center", rotation=0)
ax[0].legend(fontsize=10); ax[0].grid(alpha=.3, which="both")

# Panel 2: build time
ax[1].plot(q, asm_ms, "o-", color="#d62728", lw=2, ms=8, label="assembled DIA SpMSpM")
ax[1].plot(q, pau_ms, "s-", color="#1f77b4", lw=2, ms=8, label="Pauli basis")
ax[1].set_xticks(q)
ax[1].set_xlabel("qubits"); ax[1].set_ylabel("operator-build time (ms) — lower better")
ax[1].set_title("Build time: 10x faster at q14, gap widens (crossover ~q12)")
for xi, a, p in zip(q, asm_ms, pau_ms):
    ax[1].annotate(f"{a:.1f}", (xi, a), textcoords="offset points", xytext=(0,8), fontsize=8, ha="center", color="#d62728")
    ax[1].annotate(f"{p:.2f}", (xi, p), textcoords="offset points", xytext=(0,-14), fontsize=8, ha="center", color="#1f77b4")
ax[1].legend(fontsize=10); ax[1].grid(alpha=.3)

fig.suptitle("Pauli-basis vs assembled-DIA operator build  U = $\\Sigma_k c_k H^k$  (heis, K=6, H100)\n"
             "same operator, ~1e-10 fidelity vs exact $e^{-iHt}$; Pauli wins compactness, speed, and scalability",
             fontsize=12)
fig.tight_layout(rect=[0,0,1,0.94])
out = os.path.join(os.path.dirname(os.path.abspath(__file__)), "pauli_vs_assembled.png")
fig.savefig(out, dpi=140); print("wrote", out)
