import sys
import os
import re
import h5py
import numpy as np
import scipy as sc
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from qiskit.quantum_info import Statevector
from qiskit.quantum_info import SparsePauliOp
from time import perf_counter

# Check number of arguments
# if len(sys.argv) < 6:
# 	print("Usage: python app2_numpy_trotterized.py <HDF5_FILE> <HDF5_KEY> <FINAL_TIME> <NUM_TIMESTEPS> <ITRS>")
# 	print("Example: python app2_numpy_trotterized.py file.hdf5 key 1.8 200 5")
# 	sys.exit(1)

# Parse arguments

folder = "/mnt/beegfs/ysu34/hamlib/"
HDF5_FILE = folder + sys.argv[1]
parent_folder = os.path.dirname(HDF5_FILE)
print("Parent folder:", parent_folder)
# Extract HDF5 filename components
HDF5_BASENAME = os.path.basename(HDF5_FILE)
HDF5_NAME, HDF5_EXT = os.path.splitext(HDF5_BASENAME)
HDF5_KEY = sys.argv[2]
FINAL_TIME = 1.2
NUM_TIMESTEPS = 1000
ITRS = 5
HAM_NAME = sys.argv[1].split(".")[1].split("/")[2]
# print("HAM_NAME:", HAM_NAME)


# Check file existence
if not os.path.exists(HDF5_FILE):
	print("File does not exist:", HDF5_FILE)
	sys.exit(1)
if FINAL_TIME <= 0:
	print("Final time must be positive.")
	sys.exit(1)
if NUM_TIMESTEPS < 1:
	print("Number of timesteps must be at least 1.")
	sys.exit(1)
if ITRS < 1:
	print("Number of iterations must be at least 1.")
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

def run_numpy_trotter(H_array, initial_state, delta_t, num_steps):
	U = sc.linalg.expm(-1j * H_array * delta_t)
	state = initial_state.data
	for _ in range(num_steps):
		state = np.dot(U, state)
	return state
def expm_negative_taylor_iterations(A_diag_real, A_diag_imag, t, tol=1e-8, max_iter=100):
    """
    Simulates Taylor expansion of e^{-i A t} for a diagonal matrix A,
    only tracks how many iterations it takes to converge.

    A_diag_real: np.array of real parts of diagonal elements
    A_diag_imag: np.array of imag parts of diagonal elements
    t: time scalar
    tol: convergence threshold
    max_iter: max number of iterations
    """

    # Construct X = -i * A * t elementwise
    diag_complex = A_diag_real + 1j * A_diag_imag
    X_diag = -1j * diag_complex * t

    # Initialize: result = I, term = I
    result_diag = np.ones_like(X_diag, dtype=np.complex128)
    term_diag = np.ones_like(X_diag, dtype=np.complex128)

    for i in range(1, max_iter + 1):
        # term = term * X / i
        term_diag = term_diag * X_diag / i
        result_diag += term_diag

        norm = np.max(np.abs(term_diag))  # max norm to check convergence
        if norm < tol:
            return i

    return max_iter



def extract_diagonal_components(H_array):
    """
    Given a dense complex matrix H_array, extract the diagonal
    and return its real and imaginary parts separately.

    Returns:
        A_diag_real: np.array of real parts of diagonal
        A_diag_imag: np.array of imaginary parts of diagonal
    """
    if not np.iscomplexobj(H_array):
        raise ValueError("Input matrix must be complex.")

    diag = np.diag(H_array)
    A_diag_real = np.real(diag)
    A_diag_imag = np.imag(diag)
    return A_diag_real, A_diag_imag

#count NNZ and non-zero diagonals




# Load and initialize
sp_op = read_qiskit_hdf5_new(HDF5_FILE, HDF5_KEY)
num_qubits = sp_op.num_qubits
initial_state = Statevector.from_label('10' + '0' * (num_qubits - 2))
H_array = sp_op.to_matrix()
delta_t = FINAL_TIME / NUM_TIMESTEPS

# Extract diagonal components
A_diag_real, A_diag_imag = extract_diagonal_components(H_array)
# Calculate number of iterations for convergence
num_iterations = expm_negative_taylor_iterations(A_diag_real, A_diag_imag, delta_t)

print()
print(f"\tnum_qubits: {num_qubits}")
print(f"\tkey: {HDF5_KEY}")
print(f"\tFinal time: {FINAL_TIME}")
print(f"\tNum timesteps: {NUM_TIMESTEPS}")
print(f"\tDelta t: {delta_t}")
print(f"\tIterations: {ITRS}")
print()

# Benchmark Trotterized NumPy
# for i in range(ITRS):
# 	t_start = perf_counter()
# 	final_state = run_numpy_trotter(H_array, initial_state, delta_t, NUM_TIMESTEPS)
# 	t_elapsed = perf_counter() - t_start
	# print(f"numpy-trotter,{num_qubits},{HDF5_KEY},{t_elapsed}", flush=True)


H_real = H_array.real
H_imag = H_array.imag
imag_counter = 0
def analyze_real_matrix(mat: np.ndarray):
    """
    Analyze a real dense matrix and return sparsity metrics.

    Returns: (nnz, nonzero_diagonal_count, nonzero_diagonal_offsets, sparsity)
    """
    if mat.ndim != 2 or mat.shape[0] != mat.shape[1]:
        raise ValueError("Input must be a square 2D array")
    n = mat.shape[0]
    nnz = int(np.count_nonzero(mat))
    nonzero_diagonal_offsets = [
        offset for offset in range(-n + 1, n) if np.any(np.diag(mat, k=offset) != 0)
    ]
    nonzero_diagonal_count = len(nonzero_diagonal_offsets)
    sparsity = 1.0 - (nnz / float(n * n))
    return nnz, nonzero_diagonal_count, nonzero_diagonal_offsets, sparsity


# Save dense real matrix as text
out_dir = "/mnt/beegfs/ysu34/hamlib/hamMatrix"
if not os.path.exists(out_dir):
    os.makedirs(out_dir)

def write_coo_combined(matrix: np.ndarray, path: str, one_based: bool = False):
    """
    Write matrix in combined COO format:
      First line: M N NNZ
      Following lines: row col value

    Indices are 0-based by default; set `one_based=True` to write 1-based indices.
    """
    rows, cols = matrix.shape
    r_idx, c_idx = np.nonzero(matrix)
    data = matrix[r_idx, c_idx]
    nnz = len(data)

    with open(path, "w") as f:
        f.write(f"{rows} {cols} {nnz}\n")
        for r, c, v in zip(r_idx, c_idx, data):
            if one_based:
                f.write(f"{r+1} {c+1} {v}\n")
            else:
                f.write(f"{r} {c} {v}\n")

# out_path = os.path.join(out_dir, HDF5_NAME + str(pow(2, num_qubits)) + ".txt")
# print(f"Saving to {out_path}...")
# np.savetxt(out_path, H_real, fmt="%.6f", delimiter=" ")

# Save H_real in combined COO format (M N NNZ\nrow col value)
def save_matrix_power_iterations(matrix: np.ndarray, out_dir: str, base_name: str, iterations: int, one_based: bool = False):
    """
    Save the initial `matrix` and its repeated-squared intermediates up to `iterations`.

    Each saved file uses combined COO format and is named `base_name_step{i}.coo` where
    i==0 is the original matrix and 1..iterations are the successive A <- A @ A powers.
    """
    os.makedirs(out_dir, exist_ok=True)
    # Save step 0
    path0 = os.path.join(out_dir, f"{base_name}_step0.coo")
    write_coo_combined(matrix, path0, one_based=one_based)
    print(f"Wrote COO file: {path0}")

    # A = matrix.copy()
    # for i in range(1, iterations + 1):
    #     A = A @ A
    #     p = os.path.join(out_dir, f"{base_name}_step{i}.coo")
    #     write_coo_combined(A, p, one_based=one_based)
    #     print(f"Wrote COO file: {p}")


# Save H_real and its powers limited by Taylor convergence iterations
base_name = "coo"
out_dir = "/mnt/beegfs/ysu34/hamlib/" + str(HAM_NAME) + "/data/" + str(num_qubits)
print(f"Saving matrix and powers to {out_dir}...")
save_matrix_power_iterations(H_real, out_dir, base_name, num_iterations)

# Analyze matrix
# nnz, nonzero_diagonal_count, nonzero_diagonal_offsets, sparsity = analyze_real_matrix(H_real)
# imag_counter = int(np.count_nonzero(H_imag))

# print(f"Matrix sparsity analysis for {HDF5_NAME + str(pow(2, num_qubits))}:")
# print(f"\tNumber of non-zero elements (NNZ): {nnz}")
# print(f"\tNumber of non-zero diagonals: {nonzero_diagonal_count}")
# print(f"\tDiagonal Sparsity: {(1 - nonzero_diagonal_count / (H_array.shape[0] * 2 - 1)):.4f} (1.0 means all diagonals are non-zero)")
# print(f"\tSparsity: {sparsity:.6f} (0.0 means dense, 1.0 means empty)")
# print(f"\tNum iterations:", num_iterations)
# print(f"\tNumber of non-zero imaginary elements: {imag_counter}")


