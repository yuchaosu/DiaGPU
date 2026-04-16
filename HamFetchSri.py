import sys
import os
import re

# check number of arguments
if len(sys.argv) < 5:
	print(f"Usage: python {sys.argv[0]} <PERCENT_DIAGONALS> <HDF5_FILE> <HDF5_KEY> <METHOD> <FINAL_TIME> [ITRS] [TEST_FIDELITY] [optional: number of timesteps]")
	print(f"Example: python {sys.argv[0]} 30 /path/to/file.hdf5 key numpy 1.8 5 True 6057")
	sys.exit(1)

PERCENT_DIAGONALS = int(sys.argv[1]) 									# e.g., 30 for 30% of the maximum possible diagonals, which is 2N-1 for an N x N matrix. This is used for setting the diagonal-budget.
HDF5_FILE = sys.argv[2]													# path to the HDF5 file containing the Hamiltonian data in qiskit SparsePauliOp format.
HDF5_KEY = sys.argv[3]													# key within the HDF5 file to read the Hamiltonian data from.
METHOD = sys.argv[4] 													# numpy, diaq, csr, csc, dia, qiskit-aer
FINAL_TIME = float(sys.argv[5]) if len(sys.argv) > 5 else 1.8			# default 1.8
ITRS = int(sys.argv[6]) if len(sys.argv) > 6 else 5 					# default 5
TEST_FIDELITY = sys.argv[7] == 'True' if len(sys.argv) > 7 else False 	# default is False
NUM_TIMESTEPS = int(sys.argv[8]) if len(sys.argv) > 8 else None 		# default is None, meaning auto-calculate using the technique below.

# sanity checks
if not os.path.exists(HDF5_FILE):
	print(f"File does not exist: {HDF5_FILE}", flush=True)
	sys.exit(1)
valid_methods = [
	"numpy",
	# "diaq",		# will not work without diaq library
	# "diaq-gpu",	# will not work without diaq library
	"csr",
	"csc",
	"dia",
	"qiskit-aer",
	"qiskit-aer-gpu", # requires qiskit-aer installation with GPU support
	"expm_multiply"
]
if METHOD not in valid_methods:
	print(f"Method not supported: {METHOD}", flush=True)
	print(f"Valid methods: {valid_methods}", flush=True)
	sys.exit(1)
if FINAL_TIME < 0:
	print(f"Final time must be at least 0: {FINAL_TIME}", flush=True)
	sys.exit(1)
if ITRS < 1:
	print(f"Number of iterations must be at least 1: {ITRS}", flush=True)
	sys.exit(1)

import h5py
import numpy as np
import scipy as sc

import warnings
warnings.filterwarnings('ignore')

from pprint import pprint

# if "gpu" in METHOD:
# 	import diaq_gpu as dq
# 	pprint(dq.install_info())
# else:
# 	import diaq as dq
# 	pprint(dq.install_info())

from qiskit.quantum_info import state_fidelity, Statevector
from qiskit_aer import AerSimulator
from qiskit import transpile, QuantumCircuit
from qiskit.circuit.library import PauliEvolutionGate
from qiskit.quantum_info import SparsePauliOp
from time import perf_counter, process_time
import copy

# sys.path.insert(0, '/mnt/beegfs/ysu34/hamlib/')
# sys.path.insert(0, '/ccs/home/schundu3/qsparse/ham_benchmarks')
from hamlib_functions.hamlib.hamlib_snippets import *

if METHOD == "qiskit-aer-gpu":
	statevector_simulator = AerSimulator(method='statevector', device='GPU')
else:
	statevector_simulator = AerSimulator(method='statevector')

def read_qiskit_hdf5_new(fname_hdf5: str, key: str):
	"""
	Read the operator object from HDF5 at specified key to qiskit SparsePauliOp
	format.
	"""
	def _generate_string(term):
		# change X0 Z3 to XIIZ
		indices = [
			(m.group(1), int(m.group(2)))
			for m in re.finditer(r'([A-Z])(\d+)', term)
		]
		return ''.join(
			[next((char for char, idx in indices if idx == i), 'I')
			 for i in range(max(idx for _, idx in indices) + 1)]
		)

	def _append_ids(pstrings):
		# append Ids to strings
		return [p + 'I' * (max(map(len, pstrings)) - len(p)) for p in pstrings]

	with h5py.File(fname_hdf5, 'r', libver='latest') as f:
		pattern = r'([\d.]+) \[([^\]]+)\]' # OLD REGEX
		pattern = r'\(?([\d.-]+(?:[+-][\d.]+j)?)\)? \[([^\]]+)\]'
		matches = re.findall(pattern, f[key][()].decode("utf-8"))
		labels = [_generate_string(m[1]) for m in matches]
		coeffs = [complex(match[0]).real for match in matches]
		op = SparsePauliOp(_append_ids(labels), coeffs)
	return op

def run_numpy(H_array, initial_state, exact_times, final_time, trott=False):
	if trott:
		# trotterization
		init_state = initial_state.data
		U = sc.linalg.expm(-1j * (H_array * exact_times[0]))
		for i, t in enumerate(exact_times):
			init_state = np.dot(U, init_state)
		return init_state
	else:
		# no trotterization!!
		dense_U = sc.linalg.expm(-1j * final_time * H_array)
		initial_state_data = initial_state.data
		final_state = np.dot(dense_U, initial_state_data)
		return final_state

# def run_diaq(dq_H, diaq_initial_state, exact_times, final_time):
# 	dq_U = dq.expm_negative(dq_H, exact_times[0])
# 	diaq_final_state = dq.vector(diaq_initial_state.numRows)
# 	# old way to do this
# 	# for i, t in enumerate(exact_times):
# 	# 	dq.spMV(dq_U, diaq_initial_state, diaq_final_state)
# 	# 	diaq_initial_state = diaq_final_state
# 	# improved way to do this, using spMV_ntimes kernel
# 	dq.spMV_ntimes(dq_U, diaq_initial_state, diaq_final_state, len(exact_times))
# 	return diaq_final_state
# 	# diaq_final_state = dq.hamiltonian_simulation_cuda(dq_H, diaq_initial_state, final_time, len(exact_times))
# 	# return diaq_final_state

def run_sparse(sc_H, sc_initial_state, exact_times, final_time):
	was_dia = False
	if isinstance(sc_H, sc.sparse.dia_matrix):
		# convert to CSR format for expm
		was_dia = True
		sc_H = sc_H.tocsr()
	# compute exp(-i H dt) only once
	sc_U = sc.sparse.linalg.expm(-1j * exact_times[0] * sc_H)
	if was_dia:
		sc_H = sc_H.todia()
	for i, t in enumerate(exact_times):
		sc_initial_state = sc_U.dot(sc_initial_state)
	return sc_initial_state

def run_expm_multiply_csr(csr_H, csr_initial_state, exact_times):
	for i, t in enumerate(exact_times):
		csr_initial_state = sc.sparse.linalg.expm_multiply(-1j * t * csr_H, csr_initial_state)
	return csr_initial_state

def run_qiskit(sp_op, initial_state, exact_times, final_time):
	evolution_gate = PauliEvolutionGate(sp_op, time=exact_times[0])
	qc = QuantumCircuit(sp_op.num_qubits)
	qc.initialize(initial_state.data, range(sp_op.num_qubits))

	# Apply the evolution gate multiple times
	for _ in range(len(exact_times)):
		qc.append(evolution_gate, range(sp_op.num_qubits))

	qc.save_statevector()
	transpiled_qc = transpile(qc, statevector_simulator)
	result = statevector_simulator.run(transpiled_qc).result()
	final_statevector = result.get_statevector()
	return final_statevector

sp_op = read_qiskit_hdf5_new(HDF5_FILE, HDF5_KEY)
num_qubits = sp_op.num_qubits
key = HDF5_KEY

print()
print("\tnum_qubits: ", num_qubits, flush=True)
print("\tChosen key: ", key, flush=True)
print("\tMethod: ", METHOD, flush=True)
print("\tFinal time: ", FINAL_TIME, flush=True)
print("\tNumber of iterations: ", ITRS, flush=True)

which_node_this_run = os.uname().nodename
print("\tRunning on node: ", which_node_this_run, flush=True)

initial_state = Statevector.from_label('10' + '0' * (num_qubits - 2))

# # H_array = sp_op.to_matrix()
# # dq_H = dq.from_numpy(H_array)
# # instead of using dq.from_numpy, we use the custom function to convert SparsePauliOp to diaq format
# DQ_PAULI_MATRICES = {
# 	'I': dq.identityMatrix(2),
# 	'X': dq.from_numpy(np.array([[0, 1], [1, 0]], dtype=np.complex128)),
# 	'Y': dq.from_numpy(np.array([[0, -1j], [1j, 0]], dtype=np.complex128)),
# 	'Z': dq.from_numpy(np.array([[1, 0], [0, -1]], dtype=np.complex128))
# }
# def from_qiskit_sp_op(sp_op):
# 	num_qubits = sp_op.num_qubits
# 	# sparse_pauli_list = sp_op.to_sparse_list()
# 	pauli_list = sp_op.to_list()
# 	dq_H = None
# 	for (full_ops, coeff) in pauli_list:
# 		result = None
# 		for op in full_ops:
# 			mat = DQ_PAULI_MATRICES[op]
# 			result = dq.kroneckerProduct(result, mat) if result else mat
# 		if dq_H is None:
# 			dq_H = result
# 			dq_H.multiplyScalar(coeff.real, coeff.imag)
# 		else:
# 			dq_H.addMatrixScaled(result, coeff.real, coeff.imag)
# 	dq_H.pruneMatrix()
# 	return dq_H

# # making dq_H always
# dq_H = from_qiskit_sp_op(sp_op) # convert SparsePauliOp to diaq format
# print()
# dq_H.print_meta()
# print()

H_array = sp_op.to_matrix()
N = H_array.shape[0]

def estimate_min_timesteps_diaq(
    # dq_H,
	H_array,
    final_time,
    max_allowed_diags,
    min_timesteps: int | None = None,
    max_timesteps: int = 1000_000,   		# let this be generous; we’ll stop early anyway
    delta_t_hint: float = 0.01,
    verbose: bool = True,
):
    """
    Find the *least* number of time steps T such that
        num_diags( exp(-i * H * (final_time/T)) ) <= max_allowed_diags.
    """
    if verbose:
        print(f"[DEBUG] final_time={final_time}, max_allowed_diags={max_allowed_diags}, "
              f"min_timesteps={min_timesteps}, max_timesteps={max_timesteps}, "
              f"delta_t_hint={delta_t_hint}", flush=True)

    # ds = dq_H.get_diagonals()
    H_sparse = sc.sparse.csr_matrix(H_array)
    H_coo = H_sparse.tocoo()
    ds = np.unique(H_coo.col - H_coo.row)
    if verbose:
        print(f"[DEBUG] H has {len(ds)} diagonals; dIndex of first is {ds[0]}", flush=True)
    if len(ds) == 1 and ds[0] == 0:
        print("[DEBUG] H is diagonal; returning 1 timestep.", flush=True)
        return 1

    # Cache results so we never recompute expm for the same T
    _cache: dict[int, int] = {}

    def diag_count_for_steps(time_steps: int) -> int:
        if time_steps in _cache:
            return _cache[time_steps]

        dt = final_time / time_steps

        # dq_U = dq.expm_negative(dq_H, dt) # this is the expensive call we want to optimize!!
		# cnt = dq_U.num_diags()
        U_sparse = sc.sparse.linalg.expm(-1j * dt * H_sparse)
        U_coo = U_sparse.tocoo()
        offsets = np.unique(U_coo.col - U_coo.row)
        cnt = len(offsets)

        _cache[time_steps] = cnt

        if verbose:
            print(f"[DEBUG] T={time_steps:>7}  dt={dt:.6g} -> diags={cnt}", flush=True)

        return cnt

    def feasible(T: int) -> bool:
        return diag_count_for_steps(T) <= max_allowed_diags

    # Choose a starting lower bound:
    # If hint is dt≈delta_t_hint, then T≈final_time/delta_t_hint
    if min_timesteps is None:
        guess = max(1, int(final_time / max(delta_t_hint, 1e-12)))
        # keep guess within [1, max_timesteps]
        min_timesteps = max(1, min(guess, max_timesteps))

    # If min_timesteps is already feasible, binary-search down to the minimum.
    low = max(1, min_timesteps)
    if feasible(low):
        # search the minimal feasible in [1, low]
        L, R = 1, low
        best = low
        while L <= R:
            mid = (L + R) // 2
            if feasible(mid):
                best = mid
                R = mid - 1
            else:
                L = mid + 1
        return best

    # Exponential search to find the first feasible high
    high = low
    while high < max_timesteps and not feasible(high):
        # Double, but guard against overflow and cap at max_timesteps
        nxt = high * 2
        high = max(high + 1, min(nxt, max_timesteps))
        if high == max_timesteps:
            break

    if not feasible(high):
        # Could not satisfy constraint within limit
        raise RuntimeError(
            f"Could not meet max_allowed_diags={max_allowed_diags} even at "
            f"max_timesteps={max_timesteps} (diags={diag_count_for_steps(high)})."
        )

    # Now we know: low is infeasible, high is feasible → find minimal feasible in (low, high]
    best = high
    L, R = low + 1, high
    while L <= R:
        mid = (L + R) // 2
        if feasible(mid):
            best = mid
            R = mid - 1
        else:
            L = mid + 1

    return best

############################################################################################################################################
allowed_percentage_of_diags = PERCENT_DIAGONALS
############################################################################################################################################
max_d = allowed_percentage_of_diags * (2 * N - 1) // 100
max_d = max(1500, max_d) # just so it does not drop too low!!

if num_qubits <= 10:
	min_steps = 1
elif num_qubits <= 14:
	min_steps = 100
else:
	min_steps = 10000

if NUM_TIMESTEPS is not None:
	print("\tUsing user-provided number of timesteps: ", NUM_TIMESTEPS, flush=True)
	print("\tCalculating number of diagonals in exp(-i H dt) for dt = FINAL_TIME / NUM_TIMESTEPS", flush=True)
	dt = FINAL_TIME / NUM_TIMESTEPS
	# dq_U = dq.expm_negative(dq_H, dt)
	# num_diags = dq.num_diags()
	H_sparse = sc.sparse.csr_matrix(H_array)
	U_sparse = sc.sparse.linalg.expm(-1j * dt * H_sparse)
	U_coo = U_sparse.tocoo()
	offsets = np.unique(U_coo.col - U_coo.row)
	num_diags = len(offsets)
	print("\tNumber of diagonals in exp(-i H dt): ", num_diags, flush=True)
	if num_diags > max_d:
		print("\tWarning: number of diagonals exceeds max allowed diagonals (", max_d, ")", flush=True)
else:
	NUM_TIMESTEPS = estimate_min_timesteps_diaq(
		H_array, FINAL_TIME, max_allowed_diags=max_d, min_timesteps=min_steps,
		verbose=True
	)

print("\tNumber of timesteps: ", NUM_TIMESTEPS, flush=True)
delta_t = FINAL_TIME / NUM_TIMESTEPS
exact_times = [delta_t] * NUM_TIMESTEPS
print() # all config is printed!

skip_numpy = False
skip_csr = False
skip_dia = False
if TEST_FIDELITY is False:
	skip_numpy = True
	skip_csr = True
if num_qubits >= 16:
	print("Skipping fidelity test for num_qubits >= 16 to avoid memory issues.", flush=True)
	TEST_FIDELITY = False
	skip_numpy = True
	skip_csr = True
	skip_dia = True

#print all run config
print(f"\tSkip numpy: {skip_numpy}", flush=True)
print(f"\tSkip csr: {skip_csr}", flush=True)
print(f"\tSkip dia: {skip_dia}", flush=True)
print(f"\tTest fidelity: {TEST_FIDELITY}", flush=True)
print()

print(f"\tFor Yuchao: t={FINAL_TIME}, dt={delta_t}, r={NUM_TIMESTEPS}", flush=True)
print()

# if TEST_FIDELITY:
# 	try:
# 		# using numpy
# 		time_before = perf_counter()
# 		correct_final_state = run_numpy(H_array, initial_state, exact_times, FINAL_TIME)
# 		print(f"correct_final_state, {num_qubits}, {perf_counter() - time_before}", flush=True)
# 	except Exception as e:
# 		print(f"Error calculating correct final state: {e}", flush=True)
# 		correct_final_state = None
# 		# try the trotterized version using numpy
# 		try:
# 			time_before = perf_counter()
# 			correct_final_state = run_numpy(H_array, initial_state, exact_times, FINAL_TIME, trott=True)
# 			print(f"(trotterized numpy) correct_final_state, {num_qubits}, {perf_counter() - time_before}", flush=True)
# 		except Exception as e:
# 			print(f"Error calculating correct final state: {e}", flush=True)
# 			correct_final_state = None
# else:
# 	correct_final_state = None

# if not skip_numpy:
# 	if METHOD == "numpy":
# 		try:
# 			time = perf_counter()
# 			numpy_vector = run_numpy(H_array, initial_state, exact_times, FINAL_TIME)
# 			print(f"numpy,{num_qubits},{key},{perf_counter() - time}", flush=True)
# 			# print("fidelity,numpy,{},{}".format(num_qubits, state_fidelity(correct_final_state, numpy_vector)), flush=True)
# 			for _ in range(ITRS-1):
# 				time = perf_counter()
# 				run_numpy(H_array, initial_state, exact_times, FINAL_TIME)
# 				print(f"numpy,{num_qubits},{key},{perf_counter() - time}", flush=True)
# 		except Exception as e:
# 			print(f"Error calculating numpy final state: {e}", flush=True)
# 			numpy_vector = None

# # if METHOD == "diaq" or METHOD == "diaq-gpu":
# # 	try:
# # 		diaq_initial_state = dq.from_numpy_vector(initial_state.data)
# # 		st_time = perf_counter()
# # 		diaq_vector = run_diaq(dq_H, diaq_initial_state, exact_times, FINAL_TIME)
# # 		tt = perf_counter() - st_time
# # 		print(f"{METHOD},{num_qubits},{key},{tt}", flush=True)
# # 		if correct_final_state is not None:
# # 			print(f"fidelity,{METHOD},{num_qubits},{state_fidelity(Statevector(correct_final_state), Statevector(dq.to_numpy_vector(diaq_vector)), validate=False)}", flush=True)
# # 		for _ in range(ITRS-1):
# # 			st_time = perf_counter()
# # 			run_diaq(dq_H, diaq_initial_state, exact_times, FINAL_TIME)
# # 			tt = perf_counter() - st_time
# # 			print(f"{METHOD},{num_qubits},{key},{tt}", flush=True)
# # 	except Exception as e:
# # 		print(f"Error calculating diaq final state: {e}", flush=True)
# # 		diaq_vector = None

# if METHOD in ["csr", "csc", "dia"]:
# 	initial_state = initial_state.data
# 	if METHOD == "csr":
# 		sc_H = sc.sparse.csr_matrix(H_array)
# 	elif METHOD == "csc":
# 		sc_H = sc.sparse.csc_matrix(H_array)
# 	else: # dia
# 		sc_H = sc.sparse.dia_matrix(H_array)
# 	time = perf_counter()
# 	_vector = run_sparse(sc_H, initial_state, exact_times, FINAL_TIME)
# 	print(f"{METHOD},{num_qubits},{key},{perf_counter() - time}", flush=True)
# 	if correct_final_state is not None:
# 		print(f"fidelity,{METHOD},{num_qubits},{state_fidelity(Statevector(correct_final_state), Statevector(_vector), validate=False)}", flush=True)
# 	for _ in range(ITRS-1):
# 		time = perf_counter()
# 		run_sparse(sc_H, initial_state, exact_times, FINAL_TIME)
# 		print(f"{METHOD},{num_qubits},{key},{perf_counter() - time}", flush=True)

# elif METHOD == "expm_multiply":
# 	if isinstance(initial_state, Statevector):
# 		initial_state = initial_state.data
# 	sc_H = sc.sparse.csr_matrix(H_array)
# 	time = perf_counter()
# 	csr_vector = run_expm_multiply_csr(sc_H, initial_state, exact_times)
# 	print(f"expm_multiply,{num_qubits},{key},{perf_counter() - time}", flush=True)
# 	if correct_final_state is not None:
# 		print(f"fidelity,expm_multiply,{num_qubits},{state_fidelity(Statevector(correct_final_state), Statevector(csr_vector), validate=False)}", flush=True)
# 	for _ in range(ITRS-1):
# 		time = perf_counter()
# 		run_expm_multiply_csr(sc_H, initial_state, exact_times)
# 		print(f"expm_multiply,{num_qubits},{key},{perf_counter() - time}", flush=True)

# elif METHOD == "qiskit-aer":
# 	time = perf_counter()
# 	qiskit_vector = run_qiskit(sp_op, initial_state, exact_times, FINAL_TIME)
# 	print(f"qiskit-aer,{num_qubits},{key},{perf_counter() - time}", flush=True)
# 	if correct_final_state is not None:
# 		print(f"fidelity,qiskit-aer,{num_qubits},{state_fidelity(Statevector(correct_final_state), qiskit_vector, validate=False)}", flush=True)
# 	for _ in range(ITRS-1):
# 		time = perf_counter()
# 		run_qiskit(sp_op, initial_state, exact_times, FINAL_TIME)
# 		print(f"qiskit-aer,{num_qubits},{key},{perf_counter() - time}", flush=True)

# elif METHOD == "qiskit-aer-gpu":
# 	time = perf_counter()
# 	qiskit_vector = run_qiskit(sp_op, initial_state, exact_times, FINAL_TIME)
# 	print(f"qiskit-aer-gpu,{num_qubits},{key},{perf_counter() - time}", flush=True)
# 	if correct_final_state is not None:
# 		print(f"fidelity,qiskit-aer-gpu,{num_qubits},{state_fidelity(Statevector(correct_final_state), qiskit_vector, validate=False)}", flush=True)
# 	for _ in range(ITRS-1):
# 		time = perf_counter()
# 		run_qiskit(sp_op, initial_state, exact_times, FINAL_TIME)
# 		print(f"qiskit-aer-gpu,{num_qubits},{key},{perf_counter() - time}", flush=True)
