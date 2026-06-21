#!/usr/bin/env python3.11
"""End-to-end evolution in the Pauli basis, with truncation.

Build U(t) = sum_k (-i t)^k/k! H^k by Pauli algebra (the operator-build), truncating
each chain power H^k at thr/|c_k| so we keep exactly the terms contributing > thr to U.
Compare the evolved state U|psi> against the EXACT propagator expm(-i H t)|psi> (built
from qiskit's H), and against the untruncated ('original') Pauli Taylor.

Reports, per truncation threshold: #terms in U, and fidelity |<psi_exact|psi_pauli>|^2.
Small q only (needs the dense/sparse exact propagator).
"""
import sys, numpy as np
from math import factorial
from scipy.sparse.linalg import expm_multiply
from qiskit.quantum_info import SparsePauliOp
from export_pauli import load_op

def build_U(H, t, K, thr):
    nq = H.num_qubits
    I = SparsePauliOp("I"*nq, [1.0+0j])
    acc = I                                  # k=0 term
    Pk = I
    for k in range(1, K+1):
        ck = (-1j*t)**k / factorial(k)
        # truncate the chain power at thr/|ck| -> keeps terms contributing > thr to U
        atol_chain = (thr/abs(ck)) if thr > 0 else 0.0
        Pk = (Pk.dot(H)).simplify(atol=atol_chain)
        acc = (acc + ck*Pk).simplify(atol=(thr if thr > 0 else 1e-15))
    return acc

def run(family, q, t=1.2, num_steps=500, K=6):
    # faithful scheme: per-step Taylor propagator U_step (dt = t/num_steps), applied
    # num_steps times. dt is small so ||H||*dt << 1 and K~5-6 converges per step.
    H = load_op(family, q); nq = H.num_qubits; n = 1 << nq
    dt = t/num_steps
    Hs = H.to_matrix(sparse=True)
    rng = np.random.default_rng(0)
    psi = rng.standard_normal(n) + 1j*rng.standard_normal(n); psi /= np.linalg.norm(psi)
    exact = expm_multiply(-1j*Hs*t, psi)                         # qiskit-H exact propagator
    print(f"\n=== {family} q{q}: H {len(H)} terms, t={t}, {num_steps} steps (dt={dt:.2e}), K={K} ===")
    print(f"   {'thr':>7} {'Ustep terms':>11} {'fidelity':>12} {'infidelity':>11}")
    for thr in [0.0, 1e-10, 1e-8, 1e-6, 1e-4, 1e-3, 1e-2]:
        U = build_U(H, dt, K, thr)
        Um = U.to_matrix(sparse=True)
        p = psi.copy()
        for _ in range(num_steps): p = Um @ p
        fid = abs(np.vdot(exact, p))**2 / (np.vdot(p,p).real)
        tag = "  <- original (untruncated per-step Taylor)" if thr == 0.0 else ""
        print(f"   {thr:>7.0e} {len(U):>11} {fid:>12.8f} {1-fid:>11.2e}{tag}")

for fam, q in [("heis",10), ("tfim",10), ("BeH",8)]:
    run(fam, q)
