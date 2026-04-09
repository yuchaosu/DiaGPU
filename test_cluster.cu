/* test_cluster.cu — correctness + timing for diag_cluster_kernel. */
#include "diag_cluster_kernel.cuh"
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <numeric>
#include <vector>

static constexpr int N_WARMUP  = 3;
static constexpr int N_MEASURE = 10;

#define CUDA_CHECK(call)                                                \
    do {                                                                \
        cudaError_t _e = (call);                                        \
        if (_e != cudaSuccess) {                                        \
            fprintf(stderr, "CUDA error %s:%d  %s\n",                  \
                    __FILE__, __LINE__, cudaGetErrorString(_e));        \
            exit(EXIT_FAILURE);                                         \
        }                                                               \
    } while (0)

struct GpuTimer {
    cudaEvent_t ev_start, ev_stop;
    GpuTimer()  { CUDA_CHECK(cudaEventCreate(&ev_start)); CUDA_CHECK(cudaEventCreate(&ev_stop)); }
    ~GpuTimer() { cudaEventDestroy(ev_start); cudaEventDestroy(ev_stop); }
    void  start(cudaStream_t s = 0) { CUDA_CHECK(cudaEventRecord(ev_start, s)); }
    float stop (cudaStream_t s = 0) {
        CUDA_CHECK(cudaEventRecord(ev_stop, s));
        CUDA_CHECK(cudaEventSynchronize(ev_stop));
        float ms = 0.f; CUDA_CHECK(cudaEventElapsedTime(&ms, ev_start, ev_stop));
        return ms;
    }
};
struct TimingResult { float mean_ms = 0, min_ms = 0, max_ms = 0; };

template <typename Fn>
static TimingResult measure_gpu(Fn&& fn) {
    for (int i = 0; i < N_WARMUP; ++i) { fn(); CUDA_CHECK(cudaDeviceSynchronize()); }
    GpuTimer t; std::vector<float> s(N_MEASURE);
    for (int i = 0; i < N_MEASURE; ++i) { t.start(); fn(); s[i] = t.stop(); }
    TimingResult r;
    r.mean_ms = std::accumulate(s.begin(), s.end(), 0.f) / N_MEASURE;
    r.min_ms  = *std::min_element(s.begin(), s.end());
    r.max_ms  = *std::max_element(s.begin(), s.end());
    return r;
}

template <typename T>
static T* upload(const std::vector<T>& h) {
    if (h.empty()) return nullptr;
    T* d = nullptr;
    CUDA_CHECK(cudaMalloc(&d, h.size() * sizeof(T)));
    CUDA_CHECK(cudaMemcpy(d, h.data(), h.size() * sizeof(T), cudaMemcpyHostToDevice));
    return d;
}
template <typename T>
static std::vector<T> download(const T* d, size_t n) {
    std::vector<T> h(n);
    CUDA_CHECK(cudaMemcpy(h.data(), d, n * sizeof(T), cudaMemcpyDeviceToHost));
    return h;
}

static float cpu_reference(const DiagMatrix& A, const DiagMatrix& B,
                            int M, int /*K*/, int N, std::vector<float>& C_dense) {
    C_dense.assign(static_cast<size_t>(M) * N, 0.f);
    auto t0 = std::chrono::high_resolution_clock::now();
    for (int ai = 0; ai < A.num_diags; ++ai) {
        int d_a = A.offsets[ai], a_sr = std::max(0,-d_a), a_sc = std::max(0,d_a);
        int a_len = A.diag_lengths[ai];
        for (int bi = 0; bi < B.num_diags; ++bi) {
            int d_b = B.offsets[bi], b_sr = std::max(0,-d_b), b_sc = std::max(0,d_b);
            int b_len = B.diag_lengths[bi];
            for (int pa = 0; pa < a_len; ++pa) {
                int row = a_sr + pa, k = a_sc + pa;
                int pb = k - b_sr;
                if (pb < 0 || pb >= b_len) continue;
                int col = b_sc + pb;
                if (row >= M || col >= N) continue;
                C_dense[static_cast<size_t>(row)*N + col] +=
                    A.values[A.diag_starts[ai] + pa] * B.values[B.diag_starts[bi] + pb];
            }
        }
    }
    auto t1 = std::chrono::high_resolution_clock::now();
    return std::chrono::duration<float, std::milli>(t1 - t0).count();
}

static DiagMatrix make_diag_matrix(int rows, int cols,
                                   const std::vector<int>& offsets, float seed) {
    DiagMatrix M; M.rows = rows; M.cols = cols; M.num_diags = (int)offsets.size();
    M.offsets = offsets; M.diag_starts.resize(M.num_diags); M.diag_lengths.resize(M.num_diags);
    int base = 0;
    for (int i = 0; i < M.num_diags; ++i) {
        int len = DiagMatrix::diag_length(rows, cols, offsets[i]);
        M.diag_starts[i] = base; M.diag_lengths[i] = len;
        int d = offsets[i];
        for (int p = 0; p < len; ++p)
            M.values.push_back(1.f + seed * (float)((std::abs(d)*13+p)%97)*0.01f);
        base += len;
    }
    return M;
}

static FILE* g_out = nullptr;
static void md_row(const char* l, float mn, float mi, float mx) {
    fprintf(g_out, "| %s | %.3f | %.3f | %.3f |\n", l, mn, mi, mx); }
static void md_header() {
    fprintf(g_out, "| Phase | mean (ms) | min (ms) | max (ms) |\n");
    fprintf(g_out, "|:------|----------:|---------:|---------:|\n"); }

static float compare_result(const std::vector<float>& C_gpu,
                             const ClusterPlan& plan,
                             const std::vector<float>& C_ref, int M, int N) {
    std::vector<float> C_dense(static_cast<size_t>(M)*N, 0.f);
    for (const auto& cd : plan.c_diags) {
        int c_sc = std::max(0, cd.c_offset);
        for (int k = 0; k < cd.length; ++k) {
            int row = cd.c_sr + k, col = c_sc + k;
            if (row < M && col < N)
                C_dense[static_cast<size_t>(row)*N + col] = C_gpu[cd.values_start + k];
        }
    }
    float err = 0.f;
    for (size_t i = 0; i < C_dense.size(); ++i)
        err = std::max(err, fabsf(C_dense[i] - C_ref[i]));
    return err;
}

static bool run_test(const char* name, DiagMatrix A, DiagMatrix B,
                     int M, int K, int N, float tol = 1e-3f,
                     bool skip_cpu = false) {
    printf("[%s] M=%d K=%d N=%d ...\n", name, M, K, N);
    fprintf(g_out, "\n---\n\n## %s\n\n", name);
    fprintf(g_out, "**Config:** M=%d  K=%d  N=%d  |  A\\_diags=%d  B\\_diags=%d\n\n",
            M, K, N, A.num_diags, B.num_diags);

    sort_diag_matrix_by_offset(A);

    std::vector<float> C_ref;
    float cpu_ms = 0.f;
    if (!skip_cpu) cpu_ms = cpu_reference(A, B, M, K, N, C_ref);

    ClusterPlan plan = build_cluster_plan(A, B, M, K, N);
    fprintf(g_out, "**Plan:** clusters=%zu  tasks=%zu  a\\_contrib=%zu"
                   "  b\\_contrib=%zu  part\\_b\\_meta=%zu  max\\_smem=%d bytes\n\n",
            plan.cluster_meta.size(), plan.tasks.size(),
            plan.a_contrib.size(), plan.b_contrib.size(),
            plan.part_b_meta.size(), plan.max_smem);

    /* Upload. */
    float* d_Av  = upload(A.values);
    int*   d_Ao  = upload(A.offsets);
    int*   d_As  = upload(A.diag_starts);
    int*   d_Al  = upload(A.diag_lengths);
    float* d_Bv  = upload(B.values);
    int*   d_Bo  = upload(B.offsets);
    int*   d_Bs  = upload(B.diag_starts);
    int*   d_Bl  = upload(B.diag_lengths);

    ClusterMeta* d_cm  = upload(plan.cluster_meta);
    ClusterTask* d_t   = upload(plan.tasks);
    HybridCDiag* d_cd  = upload(plan.c_diags);
    int*         d_ac  = upload(plan.a_contrib);
    /* Guard against empty b_contrib: kernel never dereferences when c_count==0. */
    int*         d_bc  = upload(plan.b_contrib.empty()
                               ? std::vector<int>{0} : plan.b_contrib);
    PartBMeta*   d_pbm = upload(plan.part_b_meta);

    float* d_Cv = nullptr;
    CUDA_CHECK(cudaMalloc(&d_Cv, (size_t)plan.total_c_values * sizeof(float)));

    ClusterKernelArgs kargs = {};
    kargs.cluster_meta = d_cm;
    kargs.n_clusters   = (int)plan.cluster_meta.size();
    kargs.max_smem     = plan.max_smem;
    kargs.tasks        = d_t;
    kargs.c_diags      = d_cd;  kargs.n_c_diags   = (int)plan.c_diags.size();
    kargs.a_contrib    = d_ac;  kargs.b_contrib    = d_bc;
    kargs.part_b_meta  = d_pbm;
    kargs.A_vals = d_Av; kargs.A_offsets = d_Ao;
    kargs.A_starts = d_As; kargs.A_lengths = d_Al; kargs.A_num_diags = A.num_diags;
    kargs.B_vals = d_Bv; kargs.B_offsets = d_Bo;
    kargs.B_starts = d_Bs; kargs.B_lengths = d_Bl;
    kargs.C_vals = d_Cv;

    /* Timing. */
    TimingResult t_total = measure_gpu([&]{ launch_cluster(kargs); });

    /* Timing table. */
    fprintf(g_out, "### Cluster kernel  (warmup=%d, runs=%d)\n\n", N_WARMUP, N_MEASURE);
    md_header();
    if (!skip_cpu) fprintf(g_out, "| CPU reference | %.3f | — | — |\n", cpu_ms);
    else           fprintf(g_out, "| CPU reference | skipped | — | — |\n");
    md_row("**cluster unified**", t_total.mean_ms, t_total.min_ms, t_total.max_ms);
    fprintf(g_out, "\n");

    /* Correctness. */
    bool ok = true;
    float err = -1.f;
    if (!skip_cpu) {
        CUDA_CHECK(cudaMemset(d_Cv, 0, (size_t)plan.total_c_values * sizeof(float)));
        launch_cluster(kargs);
        CUDA_CHECK(cudaDeviceSynchronize());
        auto C_gpu = download(d_Cv, (size_t)plan.total_c_values);
        err = compare_result(C_gpu, plan, C_ref, M, N);
        ok = (err < tol);
        fprintf(g_out, "### Correctness\n\n| max\\_err | result |\n|----------:|:------|\n");
        fprintf(g_out, "| %.2e | %s |\n\n", (double)err, ok ? "PASS" : "**FAIL**");
    } else {
        fprintf(g_out, "### Correctness\n\n_Skipped for large test._\n\n");
    }
    fflush(g_out);

    /* Cleanup. */
    cudaFree(d_Av); cudaFree(d_Ao); cudaFree(d_As); cudaFree(d_Al);
    cudaFree(d_Bv); cudaFree(d_Bo); cudaFree(d_Bs); cudaFree(d_Bl);
    cudaFree(d_cm); cudaFree(d_t);  cudaFree(d_cd);
    cudaFree(d_ac); cudaFree(d_bc); cudaFree(d_pbm);
    cudaFree(d_Cv);
    return ok;
}

int main() {
    int dev; CUDA_CHECK(cudaGetDevice(&dev));
    cudaDeviceProp prop; CUDA_CHECK(cudaGetDeviceProperties(&prop, dev));

    g_out = fopen("results_cluster.md", "w");
    if (!g_out) { fprintf(stderr, "cannot open results_cluster.md\n"); return 1; }

    fprintf(g_out, "# DiagSpMM Cluster Kernel Benchmark\n\n");
    fprintf(g_out, "| | |\n|:---|:---|\n");
    fprintf(g_out, "| **Device** | %s |\n| **SM** | %d.%d (%d SMs) |\n",
            prop.name, prop.major, prop.minor, prop.multiProcessorCount);
    fprintf(g_out, "| **Cluster size** | %d CTAs |\n", CLUSTER_SIZE);
    fprintf(g_out, "| **Warmup** | %d |\n| **Timed runs** | %d |\n\n", N_WARMUP, N_MEASURE);

    printf("Device : %s\nOutput : results_cluster.md\n\n", prop.name);

    bool all_pass = true;

    /* Test 1 — small square, few diagonals */
    {
        auto A = make_diag_matrix(256, 256, {-2,-1,0,1,2}, 1.0f);
        auto B = make_diag_matrix(256, 256, {-2,-1,0,1,2}, 2.0f);
        all_pass &= run_test("small_square_5diag", A, B, 256, 256, 256);
    }

    /* Test 2 — medium, more diagonals */
    {
        std::vector<int> offs;
        for (int d = -16; d <= 16; ++d) offs.push_back(d);
        auto A = make_diag_matrix(1024, 1024, offs, 1.0f);
        auto B = make_diag_matrix(1024, 1024, offs, 2.0f);
        all_pass &= run_test("medium_33diag", A, B, 1024, 1024, 1024);
    }

    /* Test 3 — large, skip CPU check */
    {
        std::vector<int> offs;
        for (int d = -50; d <= 50; ++d) offs.push_back(d);
        auto A = make_diag_matrix(4096, 4096, offs, 1.0f);
        auto B = make_diag_matrix(4096, 4096, offs, 2.0f);
        all_pass &= run_test("large_101diag", A, B, 4096, 4096, 4096,
                              1e-3f, /*skip_cpu=*/true);
    }

    printf("\nAll tests: %s\n", all_pass ? "PASS" : "FAIL");
    fclose(g_out);
    return all_pass ? 0 : 1;
}
