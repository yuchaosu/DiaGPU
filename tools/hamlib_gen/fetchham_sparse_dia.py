"""
Sparse diagonal build of a HamLib Hamiltonian.

Motivation
----------
fetchham_dia.py builds the DENSE 2^q x 2^q matrix via SparsePauliOp.to_matrix()
just to read off a few (~2q) nonzero diagonals. That caps us at ~14 qubits
(16384^2 complex128 ~ 4 GB). But a Pauli string is a monomial matrix (one nonzero
per row): the column of row r is r XOR x_mask, and an X/Y on qubit i lands on the
two constant diagonals +/-2^i. So the nonzero diagonal offsets are *predetermined*
by the X/Y supports of the terms, and we never need the dense matrix.

This script builds the matrix with qiskit's SPARSE path (to_matrix(sparse=True)),
which constructs each Pauli as a CSR (O(n) nonzeros) and sums them -- memory is
O(#offsets * n), not O(n^2). It then reads the diagonals straight off the CSR.

Using qiskit's own sparse builder (rather than re-deriving Pauli phase algebra)
guarantees the result matches fetchham_dia.py's dense path by construction; the
--validate path checks this to float epsilon at small q.

Output format is byte-identical to fetchham_dia.save_dia:
  Line 1: N <n> D <num_diagonals>
  Per diagonal: <offset>: v0 v1 v2 ...   (length n-|offset|, position p=min(row,col))
"""
import sys
import os
import re
import time
import h5py
import numpy as np
import scipy.sparse as sp
from qiskit.quantum_info import SparsePauliOp

if len(sys.argv) < 3:
    print(f"Usage: python {sys.argv[0]} <HDF5_FILE> <HDF5_KEY> "
          f"[FINAL_TIME] [NUM_TIMESTEPS] [OUTPUT_DIR] [--validate]")
    sys.exit(1)

HDF5_FILE = sys.argv[1]
HDF5_KEY = sys.argv[2]
FINAL_TIME = float(sys.argv[3]) if len(sys.argv) > 3 and not sys.argv[3].startswith('--') else 1.2
NUM_TIMESTEPS = int(sys.argv[4]) if len(sys.argv) > 4 and not sys.argv[4].startswith('--') else 1000
OUTPUT_DIR = sys.argv[5] if len(sys.argv) > 5 and not sys.argv[5].startswith('--') else "."
VALIDATE = '--validate' in sys.argv

if not os.path.exists(HDF5_FILE):
    print(f"File does not exist: {HDF5_FILE}")
    sys.exit(1)


def read_qiskit_hdf5_new(fname_hdf5: str, key: str):
    """Identical parsing to fetchham_dia.py so the two paths read the same H."""
    def _generate_string(term):
        indices = [(m.group(1), int(m.group(2))) for m in re.finditer(r'([A-Z])(\d+)', term)]
        return ''.join([next((char for char, idx in indices if idx == i), 'I')
                        for i in range(max(idx for _, idx in indices) + 1)])
    def _append_ids(pstrings):
        return [p + 'I' * (max(map(len, pstrings)) - len(p)) for p in pstrings]
    with h5py.File(fname_hdf5, 'r') as f:
        pattern = r'\(?([+-]?\d+(?:\.\d*)?(?:[eE][+-]?\d+)?(?:[+-]\d+(?:\.\d*)?(?:[eE][+-]?\d+)?j)?)\)? \[([^\]]+)\]'  # e-notation-safe (2026-08-01: old class mangled 1e-05 -> -5)
        matches = re.findall(pattern, f[key][()].decode("utf-8"))
        labels = [_generate_string(m[1]) for m in matches]
        # Keep FULL complex coefficients: a genuinely complex coeff (non-Hermitian
        # input) must reach the matrix so the imaginary-data checker can see it,
        # rather than being silently realified here. For a real Hermitian H the
        # coeffs are real and this is identical to the old .real path.
        coeffs = [complex(m[0]) for m in matches]
        return SparsePauliOp(_append_ids(labels), coeffs)


def taylor_convergence_iterations(h1, dt, tol=1e-8, max_iter=100):
    """Smallest K such that the 1-norm Taylor remainder bound is below tol:
         ||exp(-iH dt) - sum_{k=0}^K (-iH dt)^k/k!||  <~  (dt*||H||_1)^{K+1}/(K+1)!
    Uses the induced operator 1-norm ||H||_1 = max column abs-sum, which
    upper-bounds the spectral norm -> a rigorous, conservative truncation order
    (the old diagonal-only estimate ignored the off-diagonal coupling)."""
    x = dt * h1
    term = 1.0
    for k in range(1, max_iter + 1):
        term = term * x / k
        if term < tol:
            return k
    return max_iter


def extract_dia_from_csr(M):
    """
    Read non-zero diagonals straight off a sparse matrix (no dense buffer).
    Matches np.diag(M, k) indexing: diagonal of offset `off` is stored at
    position p = row (off>=0) or p = col (off<0), i.e. p = min(row, col),
    with length n - |off|.

    Returns (real_diagonals, imag_diagonals, max_abs_imag):
      real_diagonals : list of (offset, real-part values), zero diags dropped
      imag_diagonals : list of (offset, imag-part values), zero diags dropped
      max_abs_imag   : largest |Im(entry)| anywhere (the checker's metric)
    Both lists are sorted by offset.
    """
    n = M.shape[0]
    coo = M.tocoo()
    rows, cols, vals = coo.row, coo.col, coo.data
    offs = cols.astype(np.int64) - rows.astype(np.int64)
    pos = np.where(offs >= 0, rows, cols)          # p = min(row, col)
    real_diagonals, imag_diagonals = [], []
    max_abs_imag = 0.0
    for off in np.unique(offs):
        m = offs == off
        vr = vals[m].real
        vi = vals[m].imag
        if vi.size:
            max_abs_imag = max(max_abs_imag, float(np.max(np.abs(vi))))
        L = n - abs(int(off))
        if np.any(vr != 0):
            arr = np.zeros(L, dtype=np.float64); arr[pos[m]] = vr
            real_diagonals.append((int(off), arr))
        if np.any(vi != 0):
            arr = np.zeros(L, dtype=np.float64); arr[pos[m]] = vi
            imag_diagonals.append((int(off), arr))
    return real_diagonals, imag_diagonals, max_abs_imag


def save_dia(diagonals, n, output_path):
    with open(output_path, 'w') as f:
        f.write(f"N {n} D {len(diagonals)}\n")
        for offset, values in diagonals:
            vals_str = " ".join(f"{v:.15g}" for v in values)
            f.write(f"{offset}: {vals_str}\n")


# --- build H sparsely ---
# Time the full Pauli-string -> diagonal CONSTRUCTION (this is the "matrix
# construction" cost the #0 dominance profile measures against the kernel).
t_build0 = time.perf_counter()
sp_op = read_qiskit_hdf5_new(HDF5_FILE, HDF5_KEY)
num_qubits = sp_op.num_qubits
n = 2 ** num_qubits
M = sp_op.to_matrix(sparse=True)                   # csr, O(#offsets * n) memory
M = sp.csr_matrix(M)
real_diags, imag_diags, max_abs_imag = extract_dia_from_csr(M)   # included in construction time
t_build = time.perf_counter() - t_build0
print(f"num_qubits: {num_qubits}")
print(f"matrix size: {n} x {n}")
print(f"num_pauli_terms: {len(sp_op)}")
print(f"sparse nnz: {M.nnz}  ({M.nnz / (n*n):.2e} fill)")
print(f"construction_time_s: {t_build:.4f}")

# Taylor convergence order — dynamically from the operator 1-norm ||H||_1.
dt = FINAL_TIME / NUM_TIMESTEPS
h1 = float(np.abs(M).sum(axis=0).max())          # ||H||_1 = max column abs-sum
n_taylor = taylor_convergence_iterations(h1, dt)
print(f"final_time: {FINAL_TIME}  num_timesteps: {NUM_TIMESTEPS}  dt: {dt}")
print(f"||H||_1: {h1:.6g}  dt*||H||_1: {dt*h1:.6g}")
print(f"taylor_convergence_iterations (1-norm): {n_taylor}")

# Drop matrices whose Taylor order is too large to be a useful evolution at this dt
# (e.g. vibrational/high-penalty Hamiltonians with ||H||_1 ~ 1e6 -> K pins at the cap).
KMAX = int(os.environ.get("HAMLIB_KMAX", "30"))
if n_taylor > KMAX:
    print(f"SKIP: K={n_taylor} > KMAX={KMAX} (non-convergent at dt={dt:.3g}); not writing this matrix.")
    sys.exit(7)   # distinct skip code so the SLURM driver records DROPPED (not FAILED)

diagonals = real_diags                             # real part = the stored DIA matrix
print(f"non-zero diagonals: {len(diagonals)} / {2*n-1}")
print(f"diagonal offsets: {[d[0] for d in diagonals]}")

# --- imaginary-data checker ---
# A Hamiltonian is Hermitian: H[i][j] = conj(H[j][i]). The real part is symmetric
# (Re(-d)=Re(+d)); the imaginary part is anti-symmetric (Im(-d)=-Im(+d)). So if any
# imaginary content is present it is NEEDED (the operator is complex-Hermitian, not
# real-symmetric) and is recorded to a sidecar .imag DIA file. If it is negligible
# the matrix is genuinely real and we drop it (the old behavior).
IMAG_TOL = 1e-10
has_imag = max_abs_imag >= IMAG_TOL
print(f"max|Im(H)| = {max_abs_imag:.3e}  (tol {IMAG_TOL:.0e})  ->  "
      f"{'COMPLEX-HERMITIAN: imaginary part RECORDED' if has_imag else 'real: imaginary dropped'}")
herm_err = None
if has_imag:
    # Hermiticity sanity check: Im(-d) must equal -Im(+d) (anti-symmetric).
    imd = {o: v for o, v in imag_diags}
    herm_err = 0.0
    for o, v in imag_diags:
        mo = imd.get(-o)
        if mo is None:
            herm_err = float('inf'); break
        L = min(v.size, mo.size)
        herm_err = max(herm_err, float(np.max(np.abs(v[:L] + mo[:L]))) if L else 0.0)
    print(f"  [hermitian check] max|Im(-d)+Im(+d)| = {herm_err:.3e}  "
          f"{'(Hermitian: conjugate-symmetric)' if herm_err < 1e-6 else '(NOT Hermitian -- check input!)'}")

if VALIDATE:
    # Rebuild dense the way fetchham_dia.py does and diff, to prove correctness.
    Hd = sp_op.to_matrix().real
    dense_diags = {}
    for off in range(-n + 1, n):
        d = np.diag(Hd, k=off)
        if np.any(d != 0):
            dense_diags[off] = d
    sparse_diags = {o: v for o, v in diagonals}
    keys = set(dense_diags) | set(sparse_diags)
    max_err = 0.0
    for k in keys:
        a = dense_diags.get(k, np.zeros(n - abs(k)))
        b = sparse_diags.get(k, np.zeros(n - abs(k)))
        max_err = max(max_err, float(np.max(np.abs(a - b))) if a.size else 0.0)
    print(f"[validate] offset-set match: {set(dense_diags)==set(sparse_diags)}  "
          f"max abs diff vs dense: {max_err:.3e}")

os.makedirs(OUTPUT_DIR, exist_ok=True)
# Output base name: HAMLIB_TAG (set by the portal array driver, unique per
# sub-family so basenames don't collide) else the HDF5 basename.
ham_name = os.environ.get("HAMLIB_TAG") or os.path.splitext(os.path.basename(HDF5_FILE))[0]
output_path = os.path.join(OUTPUT_DIR, f"{ham_name}_{num_qubits}_{n_taylor}.txt")
save_dia(diagonals, n, output_path)
print(f"saved to: {output_path}")

# Record the imaginary part ONLY when it is needed (complex-Hermitian H). Same DIA
# format, parallel file `<name>.imag.txt`; offsets/positions align with the real file
# so a loader reconstructs H[+d] = real[+d] + i*imag[+d], H[-d] = conj (real mirror,
# imag negated). Negligible-imaginary (real) matrices write no sidecar (old behavior).
imag_path = None
if has_imag:
    imag_path = output_path[:-4] + ".imag.txt" if output_path.endswith(".txt") else output_path + ".imag"
    save_dia(imag_diags, n, imag_path)
    print(f"saved IMAGINARY part to: {imag_path}  ({len(imag_diags)} diagonals)")

# CSR sidecar (real part) — same matrix in CSR for the cuSPARSE baseline / external
# tools. M is already CSR in memory, so this is just a real-cast + save.
csr_path = (output_path[:-4] if output_path.endswith(".txt") else output_path) + ".csr.npz"
sp.save_npz(csr_path, sp.csr_matrix(M.real))
print(f"saved CSR (real) to: {csr_path}  (nnz={int(M.nnz)})")

# Sidecar consumed by sim/dominance_profile.cu (FULL-denominator construction cost).
with open(output_path + ".meta", "w") as mf:
    mf.write(f"source_hdf5 {HDF5_FILE}\n")
    mf.write(f"source_key {HDF5_KEY}\n")
    mf.write(f"build_time_s {t_build:.6f}\n")
    mf.write(f"n {n}\n")
    mf.write(f"num_qubits {num_qubits}\n")
    mf.write(f"num_pauli_terms {len(sp_op)}\n")
    mf.write(f"num_diagonals {len(diagonals)}\n")
    mf.write(f"nnz {int(M.nnz)}\n")
    mf.write(f"n_taylor {n_taylor}\n")
    mf.write(f"h1_norm {h1:.6e}\n")
    mf.write(f"final_time {FINAL_TIME}\n")
    mf.write(f"num_timesteps {NUM_TIMESTEPS}\n")
    mf.write(f"dt {dt}\n")
    mf.write(f"max_abs_imag {max_abs_imag:.6e}\n")
    mf.write(f"has_imag {1 if has_imag else 0}\n")
    if has_imag:
        mf.write(f"num_imag_diagonals {len(imag_diags)}\n")
        mf.write(f"hermitian {1 if (herm_err is not None and herm_err < 1e-6) else 0}\n")
        mf.write(f"imag_file {os.path.basename(imag_path)}\n")
print(f"meta: {output_path}.meta  (construction_time_s={t_build:.4f})")
