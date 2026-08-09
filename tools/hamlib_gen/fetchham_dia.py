import sys
import os
import re
import h5py
import numpy as np
from qiskit.quantum_info import SparsePauliOp

if len(sys.argv) < 3:
	print(f"Usage: python {sys.argv[0]} <HDF5_FILE> <HDF5_KEY> [FINAL_TIME] [NUM_TIMESTEPS] [OUTPUT_DIR]")
	print(f"Example: python {sys.argv[0]} /path/to/file.hdf5 key 1.2 1000 ./output")
	sys.exit(1)

HDF5_FILE = sys.argv[1]
HDF5_KEY = sys.argv[2]
FINAL_TIME = float(sys.argv[3]) if len(sys.argv) > 3 else 1.2
NUM_TIMESTEPS = int(sys.argv[4]) if len(sys.argv) > 4 else 1000
OUTPUT_DIR = sys.argv[5] if len(sys.argv) > 5 else "."

if not os.path.exists(HDF5_FILE):
	print(f"File does not exist: {HDF5_FILE}")
	sys.exit(1)

def read_qiskit_hdf5_new(fname_hdf5: str, key: str):
	def _generate_string(term):
		indices = [(m.group(1), int(m.group(2))) for m in re.finditer(r'([A-Z])(\d+)', term)]
		return ''.join([next((char for char, idx in indices if idx == i), 'I') for i in range(max(idx for _, idx in indices) + 1)])
	def _append_ids(pstrings):
		return [p + 'I' * (max(map(len, pstrings)) - len(p)) for p in pstrings]
	with h5py.File(fname_hdf5, 'r') as f:
		pattern = r'\(?([+-]?\d+(?:\.\d*)?(?:[eE][+-]?\d+)?(?:[+-]\d+(?:\.\d*)?(?:[eE][+-]?\d+)?j)?)\)? \[([^\]]+)\]'  # e-notation-safe (2026-08-01: old class mangled 1e-05 -> -5)
		matches = re.findall(pattern, f[key][()].decode("utf-8"))
		labels = [_generate_string(m[1]) for m in matches]
		coeffs = [complex(m[0]).real for m in matches]
		return SparsePauliOp(_append_ids(labels), coeffs)

def taylor_convergence_iterations(H_diag, dt, tol=1e-8, max_iter=100):
	"""
	Estimate Taylor convergence iteration count using the diagonal of H.
	Computes how many terms of exp(-i * diag(H) * dt) are needed to converge.
	"""
	X = -1j * H_diag * dt
	term = np.ones_like(X, dtype=np.complex128)
	for k in range(1, max_iter + 1):
		term = term * X / k
		if np.max(np.abs(term)) < tol:
			return k
	return max_iter

def extract_dia_no_padding(matrix):
	"""
	Extract non-zero diagonals from a matrix.
	Returns a list of (offset, values) where values has no padding zeros —
	only the actual diagonal entries (length = N - |offset|).
	"""
	n = matrix.shape[0]
	diagonals = []
	for offset in range(-n + 1, n):
		diag = np.diag(matrix, k=offset)
		if np.any(diag != 0):
			diagonals.append((offset, diag))
	return diagonals

def save_dia(diagonals, n, output_path):
	"""
	Save diagonal format to a text file:
	  Line 1: N <matrix_size> D <num_diagonals>
	  Per diagonal: offset: val1 val2 val3 ...
	"""
	with open(output_path, 'w') as f:
		f.write(f"N {n} D {len(diagonals)}\n")
		for offset, values in diagonals:
			vals_str = " ".join(f"{v:.15g}" for v in values)
			f.write(f"{offset}: {vals_str}\n")

# Read Hamiltonian
sp_op = read_qiskit_hdf5_new(HDF5_FILE, HDF5_KEY)
H = sp_op.to_matrix().real  # real part only, matching fetchham2 behavior
n = H.shape[0]
num_qubits = sp_op.num_qubits

print(f"num_qubits: {num_qubits}")
print(f"matrix size: {n} x {n}")

# Taylor convergence
dt = FINAL_TIME / NUM_TIMESTEPS
H_diag = np.diag(sp_op.to_matrix())  # complex diagonal for convergence estimate
n_taylor = taylor_convergence_iterations(H_diag, dt)
print(f"final_time: {FINAL_TIME}")
print(f"num_timesteps: {NUM_TIMESTEPS}")
print(f"dt: {dt}")
print(f"taylor_convergence_iterations: {n_taylor}")

# Extract diagonals
diagonals = extract_dia_no_padding(H)
num_diags = len(diagonals)
max_diags = 2 * n - 1
print(f"non-zero diagonals: {num_diags} / {max_diags}")
print(f"diagonal offsets: {[d[0] for d in diagonals]}")

# Sparsity
total_entries = n * n
nnz = int(np.count_nonzero(H))
sparsity = 1.0 - nnz / total_entries
dia_stored = sum(len(values) for _, values in diagonals)
dia_sparsity = 1.0 - dia_stored / total_entries
print(f"non-zero entries: {nnz} / {total_entries}")
print(f"matrix sparsity: {sparsity:.6f}")
print(f"DIA stored entries: {dia_stored} / {total_entries}")
print(f"DIA sparsity: {dia_sparsity:.6f}")

# Save — filename: <ham_name>_<num_qubits>_<n_taylor>.txt  e.g. heis_10_7.txt
os.makedirs(OUTPUT_DIR, exist_ok=True)
ham_name = os.path.splitext(os.path.basename(HDF5_FILE))[0]
output_path = os.path.join(OUTPUT_DIR, f"{ham_name}_{num_qubits}_{n_taylor}.txt")
save_dia(diagonals, n, output_path)
print(f"saved to: {output_path}")
