#!/usr/bin/env python3.11
"""Trace the sparse pattern back to the Pauli strings.

A Pauli term P = (X/Y on a 'flip mask' xmask) gives, for each row r, exactly one
nonzero at col = r XOR xmask. So:
   offset(r) = (r XOR xmask) - r = xmask - 2*(r AND xmask)
A term lands on 2^popcount(xmask) diagonals (one per value of r AND xmask). The rows
feeding a given diagonal are {r : r&xmask = b} = a 'sub-cube' of the bit-hypercube
(masked bits fixed = b, free bits arbitrary). We test what shape that set takes:
contiguous run? single stride? or genuine multi-stride sub-cube?
"""
import sys, re, glob
import numpy as np, h5py
from qiskit.quantum_info import SparsePauliOp

HAM = "/mnt/beegfs/ysu34/hamlib"
def find_key(family, q):
    import csv
    for row in csv.reader(open(f"{HAM}/dia_e2e/generated_matrices.csv")):
        if len(row)>3 and row[0]==family and row[3]==str(q):
            return f"{HAM}/{row[1]}", row[2]
    # fallback: guess heis key
    return f"{HAM}/{family}/{family}.hdf5", f"graph-1D-grid-nonpbc-qubitnodes_Lx-{q}_h-1"

def load_op(family,q):
    fn,key=find_key(family,q)
    def gen(term):
        idx=[(m.group(1),int(m.group(2))) for m in re.finditer(r'([A-Z])(\d+)',term)]
        return ''.join(next((c for c,i in idx if i==k),'I') for k in range(max(i for _,i in idx)+1))
    with h5py.File(fn,'r') as f:
        pat=r'\(?([\d.-]+(?:[+-][\d.]+j)?)\)? \[([^\]]+)\]'
        ms=re.findall(pat,f[key][()].decode())
        labels=[gen(m[1]) for m in ms]; co=[complex(m[0]).real for m in ms]
        L=max(map(len,labels)); labels=[p+'I'*(L-len(p)) for p in labels]
    return SparsePauliOp(labels,co), L

def runs_shape(rows):
    rows=np.sort(rows)
    if len(rows)==1: return "single"
    g=np.diff(rows)
    if (g==1).all(): return "contiguous"          # one interval -> (start,end)
    if len(np.unique(g))==1: return "stride"       # one arithmetic progression -> (start,stride,count)
    return "subcube"                                # multi-stride bit sub-cube

for family,q in [("heis",10),("tfim",10),("BeH",8)]:
    try: op,nq = load_op(family,q)
    except Exception as e: print(f"{family} q{q}: {e}"); continue
    n=1<<nq
    shapes={"single":0,"contiguous":0,"stride":0,"subcube":0}
    subdiags=0; nnz_total=0; offsets=set()
    for p in op.paulis:
        # qiskit: bit q is qubit q; x=True where X or Y
        xmask=0
        for qb in range(nq):
            if p.x[qb]: xmask|=(1<<qb)
        k=bin(xmask).count("1")
        # enumerate the 2^k masked-bit values b that are subsets of xmask
        mbits=[qb for qb in range(nq) if (xmask>>qb)&1]
        for sub in range(1<<k):
            b=0
            for t,qb in enumerate(mbits):
                if (sub>>t)&1: b|=(1<<qb)
            off=xmask-2*b
            offsets.add(off)
            # rows with r&xmask==b : free bits = non-mask bits
            free=[qb for qb in range(nq) if not ((xmask>>qb)&1)]
            # build the row set (only feasible for small q)
            rows=[]
            for fs in range(1<<len(free)):
                r=b
                for t,qb in enumerate(free):
                    if (fs>>t)&1: r|=(1<<qb)
                # valid diagonal entry: 0<=r<n and 0<=r+off<n
                if 0<=r<n and 0<=r+off<n: rows.append(r)
            if not rows: continue
            subdiags+=1; nnz_total+=len(rows)
            shapes[runs_shape(np.array(rows))]+=1
    print(f"\n=== {family} q{q}: {len(op.paulis)} Pauli terms, {len(offsets)} distinct offsets, ~{nnz_total} nnz ===")
    print(f"   per (term,sub-diagonal) position shape: {shapes}")
    print(f"   descriptor size: {len(op.paulis)} terms  vs  {nnz_total} nonzeros  ->  {nnz_total/max(len(op.paulis),1):.0f}x")
