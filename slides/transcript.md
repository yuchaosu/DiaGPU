
## Slide 1: Title

Good afternoon everyone. My name is Yuchao Su, and this is joint work with my advisor Frank Mueller at NC State. The title of our talk is "Diagonal-Aware SpMV and SpMSpM Kernels for GPU-Accelerated Hamiltonian Simulation." 

## Slide 2: Agenda

Here's how I'll structure the talk. First, the **motivation**: why two sparse kernels, and why the structure of these Hamiltonians matters. Second, the **kernel design**: our diagonal-aware SpMV and SpMSpM kernels. Third, the **results**. And finally a **summary and what's next**.

## Slide 3: Evaluating e^(−iHΔt), the choice is reuse

Let's start from the physics. With h-bar set to one, simulating a quantum system means repeatedly advancing the state by the time-evolution operator. H is an N-by-N complex Hamiltonian, psi is the state vector.

The key thing to notice: both ways of evaluating that exponential compute the *same* object, a polynomial in H applied to the state. The only thing that differs is how often you reuse the operator, and that single question decides which sparse kernel you need.

*[Left card.]* If you're evolving a single trajectory, with no reuse, you stay matrix-free. You apply H to the state vector, then apply H again, and again. Each of those steps is one sparse-matrix-times-dense-vector product, an SpMV operation. With nothing to amortize, this is the right choice for one, or a few, states, and it's exactly what Krylov-subspace methods do.

*[Right card.]* But if you're going to reuse the operator, across many states, many time steps, or repeated analyses, it pays to build the propagator once, explicitly, as a matrix U, for example a truncated Taylor series. Now forming the powers of H really does mean multiplying matrices together, sparse-times-sparse, that's the SpMSpM kernel. It's more expensive to build, but once you have U, every application is a single cheap operation, so it wins as soon as the build cost amortizes over enough uses.

## Slide 4: The problem: general libraries pay for generality

So why isn't this already solved by an off-the-shelf library? The dimension is 2-to-the-q. At 24 qubits that's a 16.7-million by 16.7-million operator, so memory bandwidth is everything.

A general sparse library, like cuSPARSE with CSR, treats H as an arbitrary sparse matrix, and it pays for that generality: it stores a column index for every nonzero, it does gather and indirection, and the memory access is scattered. Concretely, that's 8 bytes per nonzero, a 4-byte value plus a 4-byte index, with random access. And at 21 qubits and beyond it even has to fall back to 64-bit indices. We're paying, in bandwidth, for flexibility we don't need.

## Slide 5: Key observation: they're banded

Here's the structural fact we exploit. A local Hamiltonian is a sum of Pauli strings, H equals a sum of coefficients times Pauli operators, each a tensor product of I, X, Y, and Z.

The key point: each Pauli string maps to exactly **one fixed diagonal**. I and Z keep you on the diagonal; every X or Y on qubit k shifts the column by plus-or-minus 2-to-the-k, and crucially, it's the *same* offset for every row. So a few-body Hamiltonian, which only has a handful of strings, puts all of its nonzeros on a small, fixed set of diagonals.

Here is an example Hamiltonian matrix. On the left is the full matrix, at N equals 65,536, the band of plus-or-minus 64 is about 0.2% of N, so it collapses to a single line. Zoom into the top-left corner and you see the non zeros only lie on diagonal with offset plus-minus 1, 2, 4, 8, 16, 32, 64.

So we store **by diagonal**: one offset and one value array per diagonal. There's no per-nonzero column index, that's 4 bytes per diagonal versus CSR's 8 per nonzero, and the access becomes a regular, coalesced stream with no gather. That's the whole idea: the Pauli structure *guarantees* a fixed banded set of diagonals, and we build GPU kernels that stream those diagonals.

## Slide 6: Diagonal tensor-core SpMV

Now the first kernel: SpMV.

Because each diagonal is a constant offset, row i simply pairs with x at i-plus-offset, a contiguous, coalesced stream, no gather. We lift the band into a dense K-by-N "reconstruction" matrix, one zero-padded diagonal per row, and then, per 16-row tile, we feed the reconstruction slice as the A operand and the shifted x as the B operand into a raw tensor-core instruction.

It's register-direct PTX: the operands stream from global memory straight into registers, and the 16 outputs are read directly off the accumulator registers. Compared to CSR, the scattered gathers become contiguous coalesced loads, and the tensor-core MMA absorbs the multiply-accumulate.

## Slide 7: TF32 fidelity

After knowing the algorithm details. A fair question appear: the tensor-core path runs in TF32, so do we lose accuracy? The answer is essentially no.

Spin chains and QUBO problems, like TSP, max-cut, the transverse-field Ising model, Heisenberg, match the FP64 reference to 100% . Only the wider molecular bands drift, and only in the sixth decimal: the worst case is beryllium hydride at 12 qubits, that's about four parts in a million. 
## Slide 8: Diagonal-format matrix multiply

Now the harder kernel: sparse-times-sparse. 

Stored by diagonal, A and B are each just a handful of offset/value-array pairs, the rest of the N-by-N matrix is structurally zero.  

So the full N-by-N-by-N dense multiply collapses to a few diagonal-by-diagonal products: for each pair of input offsets, one shifted vector multiply-add, and no index search anywhere. Every output diagonal stays inside the band; there's never any work off the diagonals that the sums of input offsets can reach.

## Slide 9: Current diagonal SpMSpM algorithm

But there's a catch in the naive way of doing this, and this is what the prior state of the art, the Haque algorithm, runs into.

If every offset-pair is an independent thread that adds its product into C, then many different pairs cross to the *same* output C-diagonal. When that happens, multiple writers hit one cell at the same time. Those collisions force an atomicAdd, which serializes the writes, and that's the bottleneck of the atomic-scatter baseline.

## Slide 10: SpMSpM Kernel

Our kernel removes the atomics. Three steps.

First, the dense-to-diagonal mapping: an entry H at i,j lives on diagonal d equals j-minus-i, at position p equals the minimum of i and j, and each diagonal is one contiguous array.

Second, and this is the core, **one block owns one C-diagonal group**. The block loads only the related A and B diagonals into shared memory, runs the schedule there, and the entire reduction stays on-chip. Because each output diagonal is owned by exactly one block, there are **no atomics**.

Third, going back to dense: the accumulated C-diagonals scatter back out to C at i,j.

## Slide 11: Per-block C-group ownership *(interactive)*

Let me walk through the data movement for a single block. *[Click to step through the four phases.]*

This block owns C-group G1, the diagonals C-minus-one, C-zero, C-plus-one. Phase one: it loads just this group's A and B diagonals from global memory into the shared-memory tiles, note these are multiple diagonals per tile, the rows of A-tile and B-tile. Phase two: it runs the pair schedule, each scheduled product is a shifted fused-multiply-add that accumulates into the shared C-tile. All of this accumulation happens in shared memory, with no atomics. Phase three: it writes the finished C-tile back out to the group's diagonals in global C, coalesced.

## Slide 12: Tensor-core SpMV results

Now the numbers. For SpMV, we outperform cuSPARSE on all four HamLib matrices, by 3.9 to 5.6 times, a 4.7-times average speedup. The best case is c2h-16 qubits at 5.6 times, where the diagonals are long but few and map cleanly onto balanced tiles; even the lowest case, b2-10 qubits, is still 3.9 times. The driver is that local reconstruction converts most of the diagonal work into dense tensor-core tiles, with only limited boundary padding.

## Slide 13: Diagonal SpMSpM results

For the sparse-times-sparse kernel, we get 5.8 to 7.2 times over cuSPARSE, a 6.4-times average, and we're consistently ahead of the Haque atomic baseline by 1.62 to 1.99 times, about 1.8× on average. The largest relative gain is again on c2h-16 qubits, and the high-diagonal-count b2 cases also improve clearly. The win comes from moving output discovery off the GPU onto a host-side planner: the diagonal interactions are precomputed, which removes the irregular accumulation and avoids Haque's atomic-update overhead.

## Slide 14: Summary & what's next

To summarize: local Hamiltonians are banded, so we store by diagonal and stream diagonals on the GPU instead of paying for generality. Our tensor-core SpMV reconstructs the band into dense TF32 tiles for 3.9 to 5.6× over cuSPARSE, averaging 4.7×, at FP64-level fidelity. And our diagonal SpMSpM uses a host-planned pair schedule with per-block C-groups and no atomics, for 5.8 to 7.2× over cuSPARSE, averaging 6.4×, and 1.8× over Haque.

What's next *[the right column]*: first, scaling to larger problems, we expect the speedup to grow with size, because once the band no longer fits in L2 the atomic-scatter baseline bottlenecks on its atomics. Second, sweeping a broader range of Hamiltonian families. And third, measuring end-to-end performance, the whole-simulation speedup over the full Krylov and Taylor pipelines, not just the isolated kernels.

## Slide 15: Acknowledgments

This work was supported in part by the National Science Foundation under these five grants. Thank you for listening. 

## Appendix: Pauli matrices and diagonal offsets

The single-qubit Paulis. I and Z are diagonal (offset 0); X and Y are anti-diagonal (offset ±1 = ±2⁰ on one qubit):

$$
I=\begin{bmatrix}1&0\\0&1\end{bmatrix},\qquad
Z=\begin{bmatrix}1&0\\0&-1\end{bmatrix},\qquad
X=\begin{bmatrix}0&1\\1&0\end{bmatrix},\qquad
Y=\begin{bmatrix}0&-i\\i&0\end{bmatrix}
$$

Tensoring lifts the offset to the qubit's weight. For a 2-qubit system (qubits 1,0; an X or Y on qubit k gives offset ±2ᵏ), take the strings ZI, XI, YI = (Z, X, Y) on qubit 1 ⊗ I on qubit 0:

$$
ZI=Z\otimes I=\begin{bmatrix}1&0&0&0\\0&1&0&0\\0&0&-1&0\\0&0&0&-1\end{bmatrix},\quad
XI=X\otimes I=\begin{bmatrix}0&0&1&0\\0&0&0&1\\1&0&0&0\\0&1&0&0\end{bmatrix},\quad
YI=Y\otimes I=\begin{bmatrix}0&0&-i&0\\0&0&0&-i\\i&0&0&0\\0&i&0&0\end{bmatrix}
$$

ZI stays on the main diagonal (offset 0), while XI and YI sit entirely on the offset ±2 = ±2¹ diagonals, exactly the qubit-1 weight, and the *same* offset in every row.