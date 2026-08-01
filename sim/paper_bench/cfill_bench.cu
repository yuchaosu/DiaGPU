/* ============================================================
 * cfill_bench.cu — measure f_C: the TRUE-nonzero fill of C = H*H inside
 * its stored dense diagonals.  This is the boundary variable of the
 * large-q SpMSpM inversion: ours materializes C_stored slots, cuSPARSE
 * SpGEMM only C_true — the crossover follows f_C, not q.
 *
 * Runs gather_flat ONCE (no timing), counts nonzeros on the DEVICE
 * (no multi-GB copy back).  Matrices whose C exceeds device memory are
 * reported as CFILL,...,OOM (their f_C is unobtainable by this route).
 *
 * out: CFILL,file,n,Cd,C_stored,C_true,f_C
 * usage: cfill_bench <dia.txt>
 * ============================================================ */
#include "../dia_io.hpp"
#include "common.cuh"
#include "../../spmspm/gather_flat.cuh"
#include <unordered_map>

struct Cstruct { std::vector<int> offsets, lengths; std::vector<size_t> starts; size_t nnz; };
static Cstruct make_c(const DiaHost& A, const DiaHost& B){
    int n = A.n; std::vector<char> present(2*(size_t)n-1, 0);
    for (int da : A.offsets) for (int db : B.offsets){ int dc = da+db; if (dc > -n && dc < n) present[dc+n-1] = 1; }
    Cstruct C; size_t off = 0;
    for (int d = -(n-1); d <= n-1; ++d){
        if (!present[d+n-1]) continue;
        int len = n - std::abs(d);
        C.offsets.push_back(d); C.starts.push_back(off); C.lengths.push_back(len); off += len;
    }
    C.nnz = off; return C;
}

__global__ void count_nz_kernel(const float* __restrict__ v, size_t nnz,
                                unsigned long long* __restrict__ cnt)
{
    size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    unsigned long long local = 0;
    for (; i < nnz; i += (size_t)gridDim.x * blockDim.x)
        if (v[i] != 0.f) ++local;
#pragma unroll
    for (int o = 16; o > 0; o >>= 1)
        local += __shfl_down_sync(0xffffffffu, local, o);
    if ((threadIdx.x & 31) == 0) atomicAdd(cnt, local);
}

int main(int argc, char** argv){
    if (argc < 2){ fprintf(stderr, "usage: %s dia.txt\n", argv[0]); return 1; }
    DiaHost H = load_dia(argv[1]);
    const int n = H.n, nd = (int)H.offsets.size();
    Cstruct C = make_c(H, H);
    const int Cn = (int)C.offsets.size();

    float* dCv = nullptr;
    if (cudaMalloc(&dCv, C.nnz * 4) != cudaSuccess) {
        printf("CFILL,%s,%d,%d,%zu,-1,OOM\n", argv[1], n, Cn, C.nnz);
        return 0;
    }
    /* pair plan (same as prep_driver ours_flat) */
    std::unordered_map<int,int> bIdx; for (int i = 0; i < nd; ++i) bIdx[H.offsets[i]] = i;
    std::vector<int> pairPtr(Cn+1, 0); std::vector<GPair> pairs;
    for (int k = 0; k < Cn; ++k){
        int dc = C.offsets[k], minc = dc < 0 ? dc : 0;
        pairPtr[k] = (int)pairs.size();
        for (int ai = 0; ai < nd; ++ai){
            int da = H.offsets[ai], db = dc - da; if (db <= -n || db >= n) continue;
            auto it = bIdx.find(db); if (it == bIdx.end()) continue;
            GPair g; g.ab = H.starts[ai]; g.bb = H.starts[it->second];
            g.ash = (da < 0 ? da : 0) - minc;
            g.bsh = da + (db < 0 ? db : 0) - minc;
            g.al = H.lengths[ai]; g.bl = H.lengths[it->second];
            pairs.push_back(g);
        }
    }
    pairPtr[Cn] = (int)pairs.size();
    const int TILE = 256, ILP = n >= 65536 ? 4 : 1, POS = TILE * ILP;
    std::vector<int2> tiles;
    for (int k = 0; k < Cn; ++k) for (int ts = 0; ts < C.lengths[k]; ts += POS) tiles.push_back(make_int2(k, ts));

    float* dHv = dupload(H.values);
    GPair* dP = dupload(pairs); int* dPp = dupload(pairPtr); int2* dT = dupload(tiles);
    size_t* dCs = dupload(C.starts); int* dCl = dupload(std::vector<int>(C.lengths));
    int nT = (int)tiles.size();
    if (ILP == 4) gather_flat_kernel<64,4><<<nT,TILE>>>(dHv,dHv,dT,dPp,dP,dCv,dCs,dCl);
    else          gather_flat_kernel<64,1><<<nT,TILE>>>(dHv,dHv,dT,dPp,dP,dCv,dCs,dCl);
    CUDA_CHECK(cudaGetLastError()); CUDA_CHECK(cudaDeviceSynchronize());

    unsigned long long *dcnt, hcnt = 0;
    CUDA_CHECK(cudaMalloc(&dcnt, 8)); CUDA_CHECK(cudaMemset(dcnt, 0, 8));
    count_nz_kernel<<<1024, 256>>>(dCv, C.nnz, dcnt);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaMemcpy(&hcnt, dcnt, 8, cudaMemcpyDeviceToHost));
    printf("CFILL,%s,%d,%d,%zu,%llu,%.4f\n", argv[1], n, Cn, C.nnz, hcnt,
           (double)hcnt / (double)C.nnz);
    return 0;
}
