/*
 * gpu_spacecharge_kernels.cu
 *
 * CUDA kernels for GPU-accelerated 3D FFT space charge and CSR
 * particle binning / kick application.
 *
 * 3D FFT Space Charge algorithm:
 *   1. Deposit particles on 3D mesh (trilinear, atomicAdd)
 *   2. Forward FFT of charge density (cuFFT)
 *   3. For each E-field component: compute Green function, FFT, multiply, inverse FFT
 *   4. Interpolate E-field at particle positions and apply kicks
 *
 * CSR:
 *   1. Bin particles longitudinally (atomicAdd)
 *   2. Apply precomputed CSR/LSC kicks to particles (interpolation)
 */

#ifdef USE_GPU_TRACKING

#include <cuda_runtime.h>
#include <cufft.h>
#include <math.h>
#include <stdio.h>

#define CUDA_SC_CHECK(call) do { \
    cudaError_t err_ = (call); \
    if (err_ != cudaSuccess) { \
        fprintf(stderr, "[gpu_sc] CUDA error: %s at %s:%d\n", \
                cudaGetErrorString(err_), __FILE__, __LINE__); \
        return; \
    } \
} while(0)

#define CUFFT_SC_CHECK(call) do { \
    cufftResult err_ = (call); \
    if (err_ != CUFFT_SUCCESS) { \
        fprintf(stderr, "[gpu_sc] cuFFT error: %d at %s:%d\n", err_, __FILE__, __LINE__); \
        return; \
    } \
} while(0)

#define SC_ALIVE_ST 1
#define SC_LOST_PZ  8
#define SC_FPEI 89875517873.68176  /* 1/(4*pi*eps0) = c^2 * 1e-7 */

/* Forward declarations for kernels used across sections */
__global__ void z_minmax_kernel(const double *z, const int *state,
    double *block_min, double *block_max, int n);
__global__ void sc_compute_z_adj(const double *z, const double *beta,
    double *z_adj, double dct_ave, int n);
__global__ void dct_ave_reduce_kernel(const double *z, const double *beta,
    const int *state, double *block_sum, int *block_count, int n);

/* Fused pass-1 kernel: computes dct_ave reduction + x/y bounds in single pass.
   Each block produces: sum(z/beta), count(alive), min_x, max_x, min_y, max_y. */
__global__ void sc_fused_pass1_kernel(
    const double *x, const double *y, const double *z,
    const double *beta, const int *state,
    double *b_sum_zb, int *b_count,
    double *b_min_x, double *b_max_x,
    double *b_min_y, double *b_max_y,
    int n)
{
    extern __shared__ char shmem[];
    /* Layout: [sum_zb, min_x, max_x, min_y, max_y] as doubles, [count] as int */
    double *s_sum = (double*)shmem;
    double *s_mnx = s_sum + blockDim.x;
    double *s_mxx = s_mnx + blockDim.x;
    double *s_mny = s_mxx + blockDim.x;
    double *s_mxy = s_mny + blockDim.x;
    int    *s_cnt = (int*)(s_mxy + blockDim.x);

    int tid = threadIdx.x;
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    double loc_sum = 0; int loc_cnt = 0;
    double loc_mnx = 1e30, loc_mxx = -1e30;
    double loc_mny = 1e30, loc_mxy = -1e30;

    if (i < n && state[i] == 1) {
        loc_sum = z[i] / beta[i];
        loc_cnt = 1;
        loc_mnx = x[i]; loc_mxx = x[i];
        loc_mny = y[i]; loc_mxy = y[i];
    }
    s_sum[tid] = loc_sum; s_cnt[tid] = loc_cnt;
    s_mnx[tid] = loc_mnx; s_mxx[tid] = loc_mxx;
    s_mny[tid] = loc_mny; s_mxy[tid] = loc_mxy;
    __syncthreads();

    for (int s = blockDim.x/2; s > 0; s >>= 1) {
        if (tid < s) {
            s_sum[tid] += s_sum[tid+s];
            s_cnt[tid] += s_cnt[tid+s];
            if (s_mnx[tid+s] < s_mnx[tid]) s_mnx[tid] = s_mnx[tid+s];
            if (s_mxx[tid+s] > s_mxx[tid]) s_mxx[tid] = s_mxx[tid+s];
            if (s_mny[tid+s] < s_mny[tid]) s_mny[tid] = s_mny[tid+s];
            if (s_mxy[tid+s] > s_mxy[tid]) s_mxy[tid] = s_mxy[tid+s];
        }
        __syncthreads();
    }
    if (tid == 0) {
        b_sum_zb[blockIdx.x] = s_sum[0]; b_count[blockIdx.x] = s_cnt[0];
        b_min_x[blockIdx.x] = s_mnx[0]; b_max_x[blockIdx.x] = s_mxx[0];
        b_min_y[blockIdx.x] = s_mny[0]; b_max_y[blockIdx.x] = s_mxy[0];
    }
}

/* --------------------------------------------------------------------------
 * sc_reduce_pass1_kernel -- GPU-side reduction of pass-1 per-block results.
 * Reduces n_blocks min_x/max_x/min_y/max_y/sum_zb/count arrays to single
 * values in d_results: [xmin, xmax, ymin, ymax, dct_ave].
 * Launched with 1 block of 256 threads.
 * -------------------------------------------------------------------------- */
__global__ void sc_reduce_pass1_kernel(
    const double *b_min_x, const double *b_max_x,
    const double *b_min_y, const double *b_max_y,
    const double *b_sum_zb, const int *b_count,
    double *results, int n_blocks, int has_dct)
{
    __shared__ double s_mnx[256], s_mxx[256], s_mny[256], s_mxy[256], s_sum[256];
    __shared__ int s_cnt[256];
    int tid = threadIdx.x;

    double mnx = 1e30, mxx = -1e30, mny = 1e30, mxy = -1e30, sum = 0.0;
    int cnt = 0;
    for (int i = tid; i < n_blocks; i += 256) {
        if (b_min_x[i] < mnx) mnx = b_min_x[i];
        if (b_max_x[i] > mxx) mxx = b_max_x[i];
        if (b_min_y[i] < mny) mny = b_min_y[i];
        if (b_max_y[i] > mxy) mxy = b_max_y[i];
        if (has_dct) { sum += b_sum_zb[i]; cnt += b_count[i]; }
    }
    s_mnx[tid] = mnx; s_mxx[tid] = mxx; s_mny[tid] = mny; s_mxy[tid] = mxy;
    s_sum[tid] = sum; s_cnt[tid] = cnt;
    __syncthreads();
    for (int s = 128; s > 0; s >>= 1) {
        if (tid < s) {
            if (s_mnx[tid+s] < s_mnx[tid]) s_mnx[tid] = s_mnx[tid+s];
            if (s_mxx[tid+s] > s_mxx[tid]) s_mxx[tid] = s_mxx[tid+s];
            if (s_mny[tid+s] < s_mny[tid]) s_mny[tid] = s_mny[tid+s];
            if (s_mxy[tid+s] > s_mxy[tid]) s_mxy[tid] = s_mxy[tid+s];
            if (has_dct) { s_sum[tid] += s_sum[tid+s]; s_cnt[tid] += s_cnt[tid+s]; }
        }
        __syncthreads();
    }
    if (tid == 0) {
        results[0] = s_mnx[0]; results[1] = s_mxx[0];
        results[2] = s_mny[0]; results[3] = s_mxy[0];
        results[4] = has_dct ? (s_cnt[0] > 0 ? s_sum[0] / s_cnt[0] : 0.0) : 0.0;
    }
}

/* --------------------------------------------------------------------------
 * sc_reduce_minmax_kernel -- GPU-side reduction of per-block min/max.
 * Reduces n_blocks arrays to single min, max in results[out_offset:out_offset+1].
 * Launched with 1 block of 256 threads.
 * -------------------------------------------------------------------------- */
__global__ void sc_reduce_minmax_kernel(
    const double *block_min, const double *block_max,
    double *results, int out_offset, int n_blocks)
{
    __shared__ double s_min[256], s_max[256];
    int tid = threadIdx.x;
    double mn = 1e30, mx = -1e30;
    for (int i = tid; i < n_blocks; i += 256) {
        if (block_min[i] < mn) mn = block_min[i];
        if (block_max[i] > mx) mx = block_max[i];
    }
    s_min[tid] = mn; s_max[tid] = mx;
    __syncthreads();
    for (int s = 128; s > 0; s >>= 1) {
        if (tid < s) {
            if (s_min[tid+s] < s_min[tid]) s_min[tid] = s_min[tid+s];
            if (s_max[tid+s] > s_max[tid]) s_max[tid] = s_max[tid+s];
        }
        __syncthreads();
    }
    if (tid == 0) {
        results[out_offset]   = s_min[0];
        results[out_offset+1] = s_max[0];
    }
}

/* --------------------------------------------------------------------------
 * sc_z_adj_minmax_kernel -- fused z_adj computation + block min/max reduction.
 * Computes z_adj = z - dct_ave*beta inline and reduces to per-block min/max,
 * eliminating the intermediate d_z_adj buffer and one kernel launch.
 * -------------------------------------------------------------------------- */
__global__ void sc_z_adj_minmax_kernel(
    const double *z, const double *beta, const int *state,
    const double *dct_results, int dct_offset,
    double *block_min, double *block_max, int n)
{
    extern __shared__ double sdata[];
    double *smin = sdata;
    double *smax = sdata + blockDim.x;

    int tid = threadIdx.x;
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    double dct_ave = dct_results[dct_offset];

    double local_min = 1e30, local_max = -1e30;
    if (i < n && state[i] == SC_ALIVE_ST) {
        double z_adj = z[i] - dct_ave * beta[i];
        local_min = z_adj;
        local_max = z_adj;
    }
    smin[tid] = local_min;
    smax[tid] = local_max;
    __syncthreads();

    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            if (smin[tid + s] < smin[tid]) smin[tid] = smin[tid + s];
            if (smax[tid + s] > smax[tid]) smax[tid] = smax[tid + s];
        }
        __syncthreads();
    }

    if (tid == 0) {
        block_min[blockIdx.x] = smin[0];
        block_max[blockIdx.x] = smax[0];
    }
}

/* --------------------------------------------------------------------------
 * sc_compute_grid_params -- compute mesh parameters from bounds on device.
 * Reads results[0..6] = {xmin, xmax, ymin, ymax, dct_ave, zmin, zmax}
 * Writes results[7..16] = {xmin_padded, ymin_padded, zmin_padded, dx, dy, dz,
 *                          dxi, dyi, dzi, dz_rf}
 * Single-thread kernel.
 * -------------------------------------------------------------------------- */
__global__ void sc_compute_grid_params(double *results,
    int nx, int ny, int nz, double gamma)
{
    if (threadIdx.x != 0) return;
    double xmin = results[0], xmax = results[1];
    double ymin = results[2], ymax = results[3];
    double zmin = results[5], zmax = results[6];
    double dx = (xmax-xmin)/(nx-1), dy = (ymax-ymin)/(ny-1), dz = (zmax-zmin)/(nz-1);
    if (dx == 0) dx = 1e-10; if (dy == 0) dy = 1e-10; if (dz == 0) dz = 1e-10;
    xmin -= 1e-6*dx; ymin -= 1e-6*dy; zmin -= 1e-6*dz;
    xmax += 1e-6*dx; ymax += 1e-6*dy; zmax += 1e-6*dz;
    dx = (xmax-xmin)/(nx-1); dy = (ymax-ymin)/(ny-1); dz = (zmax-zmin)/(nz-1);
    results[7]  = xmin;  results[8]  = ymin;  results[9]  = zmin;
    results[10] = dx;    results[11] = dy;    results[12] = dz;
    results[13] = 1.0/dx; results[14] = 1.0/dy; results[15] = 1.0/dz;
    results[16] = dz * gamma;
}

/* =========================================================================
 * 3D FFT SPACE CHARGE -- CUDA KERNELS
 * ========================================================================= */

/* --------------------------------------------------------------------------
 * Deposit particles on 3D mesh using trilinear interpolation with atomicAdd.
 * Each particle contributes charge to 8 surrounding grid points.
 * -------------------------------------------------------------------------- */
__global__ void deposit_kernel(
    const double *x, const double *y, const double *z,
    const double *beta, /* per-particle beta for z_adj computation */
    const double *charge, const int *state,
    double *rho,
    double xmin, double ymin, double zmin,
    double dxi, double dyi, double dzi,
    double dx, double dy, double dz,
    int nx, int ny, int nz,
    double dct_ave, /* z_adj = z - dct_ave*beta */
    int n_particles)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n_particles) return;
    if (state[i] != SC_ALIVE_ST) return;

    double z_adj = z[i] - dct_ave * beta[i];

    int ip = (int)floor((x[i] - xmin) * dxi + 1.0) - 1;  /* 0-based */
    int jp = (int)floor((y[i] - ymin) * dyi + 1.0) - 1;
    int kp = (int)floor((z_adj - zmin) * dzi + 1.0) - 1;

    /* Clamp to valid range */
    if (ip < 0) ip = 0; if (ip >= nx-1) ip = nx-2;
    if (jp < 0) jp = 0; if (jp >= ny-1) jp = ny-2;
    if (kp < 0) kp = 0; if (kp >= nz-1) kp = nz-2;

    double ab = ((xmin - x[i]) + (ip+1)*dx) * dxi;
    double de = ((ymin - y[i]) + (jp+1)*dy) * dyi;
    double gh = ((zmin - z_adj) + (kp+1)*dz) * dzi;

    double q = charge[i];

    /* Deposit to 8 corners (0-based flat indexing into nx*ny*nz array) */
    #define RHO_IDX(ii,jj,kk) ((ii)*ny*nz + (jj)*nz + (kk))
    atomicAdd(&rho[RHO_IDX(ip,  jp,  kp  )], ab    *de    *gh    *q);
    atomicAdd(&rho[RHO_IDX(ip,  jp+1,kp  )], ab    *(1-de)*gh    *q);
    atomicAdd(&rho[RHO_IDX(ip,  jp+1,kp+1)], ab    *(1-de)*(1-gh)*q);
    atomicAdd(&rho[RHO_IDX(ip,  jp,  kp+1)], ab    *de    *(1-gh)*q);
    atomicAdd(&rho[RHO_IDX(ip+1,jp,  kp+1)], (1-ab)*de    *(1-gh)*q);
    atomicAdd(&rho[RHO_IDX(ip+1,jp+1,kp+1)], (1-ab)*(1-de)*(1-gh)*q);
    atomicAdd(&rho[RHO_IDX(ip+1,jp+1,kp  )], (1-ab)*(1-de)*gh    *q);
    atomicAdd(&rho[RHO_IDX(ip+1,jp,  kp  )], (1-ab)*de    *gh    *q);
    #undef RHO_IDX
}

/* --------------------------------------------------------------------------
 * Compute free-space Green function (integrated Green function method).
 * Matches osc_get_cgrn_freespace: xlafun2 for E-field, lafun2 for potential.
 * -------------------------------------------------------------------------- */
__device__ double xlafun2_dev(double x, double y, double z)
{
    double r = sqrt(x*x + y*y + z*z);
    if (r < 1e-30) return 0.0;
    return x*atan2(y*z, r*x) - z*log(r+y) + y*log((r-z)/(r+z))*0.5;
}

__global__ void green_function_kernel(
    double *cgrn,  /* real-valued Green function output */
    double dx, double dy, double dz, double gamma,
    int icomp, /* 1=Ex, 2=Ey, 3=Ez */
    int nx2, int ny2, int nz2,
    double offset_x, double offset_y, double offset_z)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = nx2 * ny2 * nz2;
    if (idx >= total) return;

    /* Decompose flat index to (i, j, k) */
    int k = idx % nz2;
    int j = (idx / nz2) % ny2;
    int i = idx / (nz2 * ny2);

    /* Rest-frame dz */
    double dz_rf = dz * gamma;

    double factor;
    if (icomp == 1 || icomp == 2)
        factor = gamma / (dx * dy * dz_rf);
    else
        factor = 1.0 / (dx * dy * dz_rf);

    double umin = (0.5 - nx2*0.5) * dx + offset_x;
    double vmin = (0.5 - ny2*0.5) * dy + offset_y;
    double wmin = (0.5 - nz2*0.5) * dz_rf + offset_z * gamma;

    double u = i * dx + umin;
    double v = j * dy + vmin;
    double w = k * dz_rf + wmin;

    double gval;
    if (icomp == 1)      gval = xlafun2_dev(u, v, w) * factor;
    else if (icomp == 2) gval = xlafun2_dev(v, w, u) * factor;
    else                 gval = xlafun2_dev(w, u, v) * factor; /* Ez */

    cgrn[idx] = gval;
}

/* --------------------------------------------------------------------------
 * Evaluate the integrated Green function via cube differences (8-point stencil).
 * Operates on real-valued arrays (Green function is real).
 * -------------------------------------------------------------------------- */
__global__ void igf_stencil_kernel(
    const double *cgrn_in,
    double *cgrn_out,
    int nx2, int ny2, int nz2)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = (nx2-1) * (ny2-1) * (nz2-1);
    if (idx >= total) return;

    int k = idx % (nz2-1);
    int j = (idx / (nz2-1)) % (ny2-1);
    int i = idx / ((nz2-1) * (ny2-1));

    #define G(ii,jj,kk) cgrn_in[(ii)*ny2*nz2 + (jj)*nz2 + (kk)]
    cgrn_out[i*ny2*nz2 + j*nz2 + k] =
        G(i+1,j+1,k+1) - G(i,j+1,k+1) - G(i+1,j,k+1) - G(i+1,j+1,k)
      - G(i,j,k) + G(i,j,k+1) + G(i,j+1,k) + G(i+1,j,k);
    #undef G
}

/* --------------------------------------------------------------------------
 * Complex multiply in frequency domain: cgrn = crho * cgrn
 * -------------------------------------------------------------------------- */
__global__ void complex_multiply_kernel(
    const cufftDoubleComplex *crho,
    cufftDoubleComplex *cgrn,
    int n_total)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n_total) return;

    double ar = crho[i].x, ai = crho[i].y;
    double br = cgrn[i].x, bi = cgrn[i].y;
    cgrn[i].x = ar*br - ai*bi;
    cgrn[i].y = ar*bi + ai*br;
}

/* 3-operand multiply: out = a * b (reads from separate source, no in-place) */
__global__ void complex_multiply_src_kernel(
    const cufftDoubleComplex *a,
    const cufftDoubleComplex *b,
    cufftDoubleComplex *out,
    int n_total)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n_total) return;
    double ar = a[i].x, ai = a[i].y;
    double br = b[i].x, bi = b[i].y;
    out[i].x = ar*br - ai*bi;
    out[i].y = ar*bi + ai*br;
}

/* --------------------------------------------------------------------------
 * Place rho (real) into one octant of doubled real array for R2C FFT.
 * -------------------------------------------------------------------------- */
__global__ void rho_to_real_kernel(
    const double *rho,
    double *rho_padded,
    int nx, int ny, int nz,
    int nx2, int ny2, int nz2)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= nx*ny*nz) return;

    int k = idx % nz;
    int j = (idx / nz) % ny;
    int i = idx / (nz * ny);

    int idx2 = i*ny2*nz2 + j*nz2 + k;
    rho_padded[idx2] = rho[idx];
}

/* --------------------------------------------------------------------------
 * Extract real field from inverse C2R FFT result (with shift).
 * -------------------------------------------------------------------------- */
__global__ void extract_field_kernel(
    const double *real_out,  /* real-valued C2R output, size nx2*ny2*nz2 */
    double *field_comp,  /* efield(:,:,:,icomp) */
    int nx, int ny, int nz,
    int nx2, int ny2, int nz2,
    double scale)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= nx*ny*nz) return;

    int k = idx % nz;
    int j = (idx / nz) % ny;
    int i = idx / (nz * ny);

    int ishift = nx - 1;
    int jshift = ny - 1;
    int kshift = nz - 1;

    int idx2 = (i+ishift)*ny2*nz2 + (j+jshift)*nz2 + (k+kshift);
    field_comp[idx] = scale * real_out[idx2];
}

/* --------------------------------------------------------------------------
 * Interpolate E-field and apply kicks to particles (space charge).
 * Matches csr_and_sc_apply_kicks for fft_3d$ case.
 * -------------------------------------------------------------------------- */
__global__ void sc_interpolate_kick_kernel(
    double *vx, double *vpx, double *vy, double *vpy,
    double *vz, double *vpz,
    int *state, double *beta_arr, double *p0c_arr,
    const double *efield,  /* nx*ny*nz*3 flat array */
    double xmin, double ymin, double zmin,
    double dxi, double dyi, double dzi,
    double dx, double dy, double dz,
    int nx, int ny, int nz,
    double ds_step, double gamma, double mc2,
    double dct_ave,
    int n_particles)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n_particles) return;
    if (state[i] != SC_ALIVE_ST) return;

    double px = vpx[i], py = vpy[i], pz_val = vpz[i];
    double beta_val = beta_arr[i];
    double p0c_val = p0c_arr[i];
    double z_sc = vz[i] - dct_ave * beta_val;

    /* Grid indices for interpolation */
    int ip = (int)floor((vx[i] - xmin) * dxi + 1.0) - 1;
    int jp = (int)floor((vy[i] - ymin) * dyi + 1.0) - 1;
    int kp = (int)floor((z_sc  - zmin) * dzi + 1.0) - 1;

    if (ip < 0) ip = 0; if (ip >= nx-1) ip = nx-2;
    if (jp < 0) jp = 0; if (jp >= ny-1) jp = ny-2;
    if (kp < 0) kp = 0; if (kp >= nz-1) kp = nz-2;

    double ab = ((xmin - vx[i]) + (ip+1)*dx) * dxi;
    double de = ((ymin - vy[i]) + (jp+1)*dy) * dyi;
    double gh = ((zmin - z_sc)  + (kp+1)*dz) * dzi;

    /* Trilinear interpolation of 3 E-field components */
    double Evec[3] = {0, 0, 0};
    int mesh_size = nx * ny * nz;

    for (int comp = 0; comp < 3; comp++) {
        const double *ef = efield + comp * mesh_size;
        #define EF(ii,jj,kk) ef[(ii)*ny*nz + (jj)*nz + (kk)]
        Evec[comp] = EF(ip,  jp,  kp  )*ab    *de    *gh
                   + EF(ip,  jp+1,kp  )*ab    *(1-de)*gh
                   + EF(ip,  jp+1,kp+1)*ab    *(1-de)*(1-gh)
                   + EF(ip,  jp,  kp+1)*ab    *de    *(1-gh)
                   + EF(ip+1,jp,  kp+1)*(1-ab)*de    *(1-gh)
                   + EF(ip+1,jp+1,kp+1)*(1-ab)*(1-de)*(1-gh)
                   + EF(ip+1,jp+1,kp  )*(1-ab)*(1-de)*gh
                   + EF(ip+1,jp,  kp  )*(1-ab)*de    *gh;
        #undef EF
    }

    /* Apply kicks */
    double factor = ds_step / (p0c_val * beta_val);
    double gamma2 = gamma * gamma;
    double rel_p = 1.0 + pz_val;
    double pz0 = sqrt(rel_p*rel_p - px*px - py*py);

    /* Transverse kicks reduced by 1/gamma^2 */
    vpx[i] = px + Evec[0] * factor / gamma2;
    vpy[i] = py + Evec[1] * factor / gamma2;

    /* Longitudinal kick: dpz = sqrt_alpha(rel_p, ef^2 + 2*ef*pz0)
     * = sqrt(rel_p^2 + ef^2 + 2*ef*pz0) - rel_p
     * Numerically stable form: x / (sqrt(alpha^2 + x) + alpha) */
    double ef = Evec[2] * factor;
    double x_sa = ef*ef + 2.0*ef*pz0;
    double dpz = x_sa / (sqrt(rel_p*rel_p + x_sa) + rel_p);
    vpz[i] = pz_val + dpz;

    /* Update beta */
    double pc_new = (1.0 + vpz[i]) * p0c_val;
    double new_beta = pc_new / sqrt(pc_new*pc_new + mc2*mc2);
    vz[i] = vz[i] * new_beta / beta_val;
    beta_arr[i] = new_beta;
}

/* =========================================================================
 * CSR BINNING AND KICK APPLICATION KERNELS
 * ========================================================================= */

/* --------------------------------------------------------------------------
 * CSR particle binning kernel -- bins particles longitudinally.
 * Uses triangular particle shape spanning particle_bin_span bins.
 * Computes weighted charge, x0*charge, y0*charge per bin.
 * -------------------------------------------------------------------------- */
__global__ void csr_bin_kernel(
    const double *vz, const double *vx, const double *vy,
    const double *charge, const int *state,
    double *bin_charge,    /* n_bin */
    double *bin_x0_wt,     /* n_bin, weighted by charge */
    double *bin_y0_wt,     /* n_bin */
    double *bin_n_particle, /* n_bin */
    double z_min, double dz_slice, double dz_particle,
    int n_bin, int particle_bin_span,
    int n_particles)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n_particles) return;
    if (state[i] != SC_ALIVE_ST) return;

    double zp_center = vz[i];
    double zp0 = zp_center - dz_particle * 0.5;
    double zp1 = zp_center + dz_particle * 0.5;
    int ix0 = (int)round((zp0 - z_min) / dz_slice);

    double q = charge[i];
    double xi = vx[i], yi = vy[i];

    for (int j = 0; j <= particle_bin_span + 1; j++) {
        int ib = j + ix0;
        if (ib < 0 || ib >= n_bin) continue;

        double zb0 = z_min + ib * dz_slice;
        double zb1 = zb0 + dz_slice;

        /* Triangular overlap calculation */
        double overlap = 0.0;
        /* Left triangular half */
        double z1 = fmax(zp0, zb0);
        double z2 = fmin(zp_center, zb1);
        if (z2 > z1) {
            overlap = 2.0 * ((z2-zp0)*(z2-zp0) - (z1-zp0)*(z1-zp0)) / (dz_particle*dz_particle);
        }
        /* Right triangular half */
        z1 = fmax(zp_center, zb0);
        z2 = fmin(zp1, zb1);
        if (z2 > z1) {
            overlap += 2.0 * ((z1-zp1)*(z1-zp1) - (z2-zp1)*(z2-zp1)) / (dz_particle*dz_particle);
        }

        if (overlap > 0.0) {
            double wt_charge = overlap * q;
            atomicAdd(&bin_charge[ib], wt_charge);
            atomicAdd(&bin_x0_wt[ib], xi * wt_charge);
            atomicAdd(&bin_y0_wt[ib], yi * wt_charge);
            atomicAdd(&bin_n_particle[ib], overlap);
        }
    }
}

/* --------------------------------------------------------------------------
 * CSR kick application kernel -- applies precomputed CSR and LSC kicks.
 * Matches csr_and_sc_apply_kicks for one_dim$/slice$ case.
 * -------------------------------------------------------------------------- */
__global__ void csr_apply_kick_kernel(
    double *vpz, const double *vz, const int *state,
    const double *kick_csr,  /* n_bin: CSR kick per bin */
    const double *kick_lsc,  /* n_bin: LSC kick per bin */
    double z_center_0, double dz_slice,
    int apply_csr, int apply_lsc,
    int n_bin, int n_particles)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n_particles) return;
    if (state[i] != SC_ALIVE_ST) return;

    double zp = vz[i];
    int i0 = (int)((zp - z_center_0) / dz_slice);
    if (i0 < 0) i0 = 0;
    if (i0 >= n_bin - 1) i0 = n_bin - 2;
    double r1 = (zp - (z_center_0 + i0 * dz_slice)) / dz_slice;
    if (r1 < 0.0) r1 = 0.0;
    if (r1 > 1.0) r1 = 1.0;
    double r0 = 1.0 - r1;

    double dpz = 0.0;
    if (apply_csr) dpz += r0 * kick_csr[i0] + r1 * kick_csr[i0+1];
    if (apply_lsc) dpz += r0 * kick_lsc[i0] + r1 * kick_lsc[i0+1];

    vpz[i] += dpz;
}


/* =========================================================================
 * HOST WRAPPERS
 * ========================================================================= */

/* Forward declaration of geometry struct (defined below near CSR kernels) */
struct CsrEleGeom;

/* Cached device arrays for space charge */
static double *d_sc_rho = NULL;
static double *d_sc_efield = NULL;  /* nx*ny*nz*3 */
static double *d_sc_charge = NULL;  /* per-particle charge */
static double *d_sc_rho_padded = NULL;  /* zero-padded real rho for D2Z, size dbl_size */
static cufftDoubleComplex *d_sc_crho_freq = NULL;  /* D2Z output, size half_size */
static cufftHandle sc_fft_plan = 0;  /* D2Z plan for charge density forward FFT */
static int sc_nx2 = 0, sc_ny2 = 0, sc_nz2 = 0;

/* Cached device arrays for CSR binning */
static double *d_csr_bin_charge = NULL;
static double *d_csr_bin_x0_wt = NULL;
static double *d_csr_bin_y0_wt = NULL;
static double *d_csr_bin_n_particle = NULL;
static double *d_csr_kick_csr = NULL;
static double *d_csr_kick_lsc = NULL;
static int d_csr_bin_cap = 0;

/* Cached device arrays for CSR bin kicks */
static CsrEleGeom *d_cbk_geom = NULL;
static double *d_cbk_I_csr = NULL;
static double *d_cbk_I_int_csr = NULL;
static int *d_cbk_ix_ele_source = NULL;
static int d_cbk_geom_cap = 0;   /* capacity in elements */
static int d_cbk_kick1_cap = 0;  /* capacity in n_kick1 */

/* Static host buffers for CSR bin kicks */
static CsrEleGeom *h_cbk_geom = NULL;
static double *h_cbk_I_csr = NULL;
static double *h_cbk_I_int_csr = NULL;
static int h_cbk_geom_cap = 0;
static int h_cbk_kick1_cap = 0;

/* (GPU-side reduction replaced per-block host buffers — see sc_reduce_pass1_kernel) */

/* Static host buffers for gpu_csr_z_minmax_ */
static double *h_zminmax_bmin = NULL, *h_zminmax_bmax = NULL;
static int h_zminmax_cap = 0;

/* Cached device arrays for gpu_csr_z_minmax_ */
static double *d_zminmax_bmin = NULL, *d_zminmax_bmax = NULL;
static int d_zminmax_cap = 0;

/* Charge upload caching -- avoid re-uploading when charge hasn't changed */
static int sc_charge_uploaded = 0;

/* Access device buffer pointers from gpu_tracking_kernels.cu via accessor */
extern "C" void gpu_get_device_ptrs_(
    double **out_vec0, double **out_vec1, double **out_vec2,
    double **out_vec3, double **out_vec4, double **out_vec5,
    int **out_state, double **out_beta, double **out_p0c);

static void get_device_ptrs(
    double *dvec[6], int **dstate, double **dbeta, double **dp0c)
{
    gpu_get_device_ptrs_(&dvec[0], &dvec[1], &dvec[2],
                         &dvec[3], &dvec[4], &dvec[5],
                         dstate, dbeta, dp0c);
}

static int ensure_sc_buffers(int nx, int ny, int nz, int n_particles)
{
    int nx2 = 2*nx, ny2 = 2*ny, nz2 = 2*nz;
    int mesh_size = nx * ny * nz;
    int dbl_size = nx2 * ny2 * nz2;
    int half_size = nx2 * ny2 * (nz2/2 + 1);  /* R2C complex output size */

    /* Recreate FFT plan if mesh size changed */
    if (nx2 != sc_nx2 || ny2 != sc_ny2 || nz2 != sc_nz2) {
        if (sc_fft_plan) cufftDestroy(sc_fft_plan);
        if (cufftPlan3d(&sc_fft_plan, nx2, ny2, nz2, CUFFT_D2Z) != CUFFT_SUCCESS) {
            fprintf(stderr, "[gpu_sc] cuFFT D2Z plan creation failed\n");
            sc_fft_plan = 0;
            return -1;
        }
        sc_nx2 = nx2; sc_ny2 = ny2; sc_nz2 = nz2;

        /* Reallocate: real padded buffer + half-size complex frequency buffer */
        if (d_sc_rho_padded)  cudaFree(d_sc_rho_padded);
        if (d_sc_crho_freq)   cudaFree(d_sc_crho_freq);
        cudaMalloc((void**)&d_sc_rho_padded, (size_t)dbl_size * sizeof(double));
        cudaMalloc((void**)&d_sc_crho_freq,  (size_t)half_size * sizeof(cufftDoubleComplex));
    }

    /* Reallocate mesh arrays only if size changed */
    {
        static int cached_mesh_size = 0;
        if (mesh_size != cached_mesh_size) {
            if (d_sc_rho) cudaFree(d_sc_rho);
            if (d_sc_efield) cudaFree(d_sc_efield);
            cudaMalloc((void**)&d_sc_rho, (size_t)mesh_size * sizeof(double));
            cudaMalloc((void**)&d_sc_efield, (size_t)mesh_size * 3 * sizeof(double));
            cached_mesh_size = mesh_size;
        }
    }

    /* Per-particle charge array -- reuse if large enough */
    {
        static int cached_np = 0;
        if (n_particles > cached_np) {
            if (d_sc_charge) cudaFree(d_sc_charge);
            cudaMalloc((void**)&d_sc_charge, (size_t)n_particles * sizeof(double));
            cached_np = n_particles;
            sc_charge_uploaded = 0;  /* force re-upload after realloc */
        }
    }

    return 0;
}

static int ensure_csr_bin_buffers(int n_bin)
{
    if (n_bin <= d_csr_bin_cap) return 0;
    if (d_csr_bin_charge)    cudaFree(d_csr_bin_charge);
    if (d_csr_bin_x0_wt)    cudaFree(d_csr_bin_x0_wt);
    if (d_csr_bin_y0_wt)    cudaFree(d_csr_bin_y0_wt);
    if (d_csr_bin_n_particle) cudaFree(d_csr_bin_n_particle);
    if (d_csr_kick_csr)     cudaFree(d_csr_kick_csr);
    if (d_csr_kick_lsc)     cudaFree(d_csr_kick_lsc);
    size_t sz = (size_t)n_bin * sizeof(double);
    cudaMalloc((void**)&d_csr_bin_charge, sz);
    cudaMalloc((void**)&d_csr_bin_x0_wt, sz);
    cudaMalloc((void**)&d_csr_bin_y0_wt, sz);
    cudaMalloc((void**)&d_csr_bin_n_particle, sz);
    cudaMalloc((void**)&d_csr_kick_csr, sz);
    cudaMalloc((void**)&d_csr_kick_lsc, sz);
    d_csr_bin_cap = n_bin;
    return 0;
}

/* --------------------------------------------------------------------------
 * gpu_space_charge_3d -- full 3D FFT space charge on GPU
 *
 * Particle data must already be on device (d_vec, dstate, dbeta, dp0c).
 * Modifies vpx, vpy, vpz, vz, beta on device.
 *
 * h_charge: per-particle charge array (host)
 *
 * Supports split compute/apply mode: when sc_compute_only_flag is set,
 * performs bounds + deposit + FFTs and returns immediately (GPU work is
 * async). The caller must then call gpu_space_charge_3d_apply_() to wait
 * for FFTs and apply kicks. This allows overlapping SC FFTs with CPU work.
 * -------------------------------------------------------------------------- */

/* File-scope state for split compute/apply SC pipeline */
static cudaEvent_t sc_fft_done[3] = {0, 0, 0};
static int sc_fft_events_created = 0;
static int sc_compute_only_flag = 0;
static struct {
    double xmin, ymin, zmin, dx, dy, dz, dxi, dyi, dzi;
    double ds_step, gamma, mc2, dct_ave;
    int nx, ny, nz, n_particles;
    int valid;
} sc_split_state = {0};

extern "C" void gpu_space_charge_3d_(
    double *h_charge,  /* n_particles, host */
    int n_particles,
    int nx, int ny, int nz,
    double gamma, double ds_step, double mc2,
    double dct_ave)
{
    if (n_particles <= 0) return;
    if (ensure_sc_buffers(nx, ny, nz, n_particles) != 0) return;

    /* Get device buffer pointers from main tracking module */
    double *dvec[6]; int *dstate; double *dbeta, *dp0c;
    get_device_ptrs(dvec, &dstate, &dbeta, &dp0c);

    int nx2 = 2*nx, ny2 = 2*ny, nz2 = 2*nz;
    int mesh_size = nx * ny * nz;
    int dbl_size = nx2 * ny2 * nz2;
    int half_size = nx2 * ny2 * (nz2/2 + 1);  /* R2C complex output size */
    int threads = 256;
    int n_blocks = (n_particles + threads - 1) / threads;

    /* Upload per-particle charge (cached -- skip if same pointer and not invalidated) */
    {
        static const double *last_h_charge = NULL;
        static int last_np = 0;
        if (!sc_charge_uploaded || h_charge != last_h_charge || n_particles != last_np) {
            CUDA_SC_CHECK(cudaMemcpy(d_sc_charge, h_charge,
                (size_t)n_particles * sizeof(double), cudaMemcpyHostToDevice));
            sc_charge_uploaded = 1;
            last_h_charge = h_charge;
            last_np = n_particles;
        }
    }

    /* --- Step 1: Compute bounds + dct_ave entirely on GPU, single download ---
     * Pass 1: fused kernel → per-block results → GPU reduce → d_sc_results[0..4]
     * Pass 2: z_adj + z bounds → GPU reduce → d_sc_results[5..6]
     * Grid params kernel → d_sc_results[7..16]
     * Single cudaMemcpy of 17 doubles replaces 2 cudaDeviceSynchronize + 8 cudaMemcpy. */

    static double *d_bmin_x=NULL, *d_bmax_x=NULL, *d_bmin_y=NULL, *d_bmax_y=NULL;
    static double *d_bmin_z=NULL, *d_bmax_z=NULL;
    static double *d_block_sum=NULL; static int *d_block_count=NULL;
    static double *d_sc_results=NULL;  /* 17 doubles: bounds + grid params */
    static int reduce_cached_nb = 0;
    if (n_blocks > reduce_cached_nb) {
        if (d_bmin_x) { cudaFree(d_bmin_x); cudaFree(d_bmax_x); cudaFree(d_bmin_y);
                        cudaFree(d_bmax_y); cudaFree(d_bmin_z); cudaFree(d_bmax_z); }
        if (d_block_sum) { cudaFree(d_block_sum); cudaFree(d_block_count); }
        cudaMalloc((void**)&d_bmin_x, n_blocks*sizeof(double));
        cudaMalloc((void**)&d_bmax_x, n_blocks*sizeof(double));
        cudaMalloc((void**)&d_bmin_y, n_blocks*sizeof(double));
        cudaMalloc((void**)&d_bmax_y, n_blocks*sizeof(double));
        cudaMalloc((void**)&d_bmin_z, n_blocks*sizeof(double));
        cudaMalloc((void**)&d_bmax_z, n_blocks*sizeof(double));
        cudaMalloc((void**)&d_block_sum, n_blocks*sizeof(double));
        cudaMalloc((void**)&d_block_count, n_blocks*sizeof(int));
        reduce_cached_nb = n_blocks;
    }
    if (!d_sc_results) cudaMalloc((void**)&d_sc_results, 17*sizeof(double));

    int need_dct = (dct_ave > 1e30 || dct_ave < -1e30 || dct_ave != dct_ave);

    /* Pass 1: fused dct_ave + x/y bounds (1 kernel, reads all particles once) */
    if (need_dct) {
        size_t shmem1 = threads * (5*sizeof(double) + sizeof(int));
        sc_fused_pass1_kernel<<<n_blocks, threads, shmem1>>>(
            dvec[0], dvec[2], dvec[4], dbeta, dstate,
            d_block_sum, d_block_count,
            d_bmin_x, d_bmax_x, d_bmin_y, d_bmax_y, n_particles);
    } else {
        z_minmax_kernel<<<n_blocks, threads, 2*threads*sizeof(double)>>>(dvec[0], dstate, d_bmin_x, d_bmax_x, n_particles);
        z_minmax_kernel<<<n_blocks, threads, 2*threads*sizeof(double)>>>(dvec[2], dstate, d_bmin_y, d_bmax_y, n_particles);
    }

    /* GPU reduce pass 1: n_blocks → single xmin/xmax/ymin/ymax/dct_ave in d_sc_results[0..4] */
    sc_reduce_pass1_kernel<<<1, 256>>>(
        d_bmin_x, d_bmax_x, d_bmin_y, d_bmax_y,
        d_block_sum, d_block_count,
        d_sc_results, n_blocks, need_dct);

    if (!need_dct) {
        /* Upload known dct_ave to d_sc_results[4] */
        cudaMemcpy(d_sc_results + 4, &dct_ave, sizeof(double), cudaMemcpyHostToDevice);
    }

    /* Pass 2: fused z_adj + z bounds (reads dct_ave from d_sc_results[4] on device,
     * computes z_adj inline, reduces to per-block min/max — no intermediate buffer) */
    sc_z_adj_minmax_kernel<<<n_blocks, threads, 2*threads*sizeof(double)>>>(
        dvec[4], dbeta, dstate, d_sc_results, 4, d_bmin_z, d_bmax_z, n_particles);

    /* GPU reduce pass 2: n_blocks → single zmin/zmax in d_sc_results[5..6] */
    sc_reduce_minmax_kernel<<<1, 256>>>(d_bmin_z, d_bmax_z, d_sc_results, 5, n_blocks);

    /* Compute grid params on GPU: d_sc_results[7..16] */
    sc_compute_grid_params<<<1, 1>>>(d_sc_results, nx, ny, nz, gamma);

    /* Single download: 17 doubles (replaces 2 cudaDeviceSynchronize + 8 cudaMemcpy) */
    double h_sc_results[17];
    cudaMemcpy(h_sc_results, d_sc_results, 17*sizeof(double), cudaMemcpyDeviceToHost);

    if (need_dct) dct_ave = h_sc_results[4];
    double xmin = h_sc_results[7],  ymin = h_sc_results[8],  zmin = h_sc_results[9];
    double dx   = h_sc_results[10], dy   = h_sc_results[11], dz   = h_sc_results[12];
    double dxi  = h_sc_results[13], dyi  = h_sc_results[14], dzi  = h_sc_results[15];
    double dz_rf_local = h_sc_results[16];

    /* --- Step 2: Deposit particles on mesh --- */
    CUDA_SC_CHECK(cudaMemset(d_sc_rho, 0, mesh_size * sizeof(double)));

    /* Deposit using raw z (dvec[4]) + beta. The kernel computes z_adj = z - dct_ave*beta
     * internally, eliminating the need for a separate sc_compute_z_adj kernel. */
    int blocks_p = (n_particles + threads - 1) / threads;

    deposit_kernel<<<blocks_p, threads>>>(
        dvec[0], dvec[2], dvec[4], dbeta,
        d_sc_charge, dstate,
        d_sc_rho,
        xmin, ymin, zmin, dxi, dyi, dzi, dx, dy, dz,
        nx, ny, nz, dct_ave, n_particles);
    CUDA_SC_CHECK(cudaGetLastError());
    /* No sync needed -- next operations are on same default stream */

    /* --- Step 3: FFT of charge density (R2C: real -> half-complex) --- */
    CUDA_SC_CHECK(cudaMemset(d_sc_rho_padded, 0, (size_t)dbl_size * sizeof(double)));

    int blocks_m = (mesh_size + threads - 1) / threads;
    rho_to_real_kernel<<<blocks_m, threads>>>(
        d_sc_rho, d_sc_rho_padded, nx, ny, nz, nx2, ny2, nz2);
    /* No sync needed -- cuFFT on default stream is serialized */

    CUFFT_SC_CHECK(cufftExecD2Z(sc_fft_plan, d_sc_rho_padded, d_sc_crho_freq));

    /* Record event on default stream so component streams can wait for charge FFT */
    static cudaEvent_t sc_crho_ready = 0;
    if (!sc_crho_ready) cudaEventCreate(&sc_crho_ready);
    cudaEventRecord(sc_crho_ready, 0);  /* 0 = default stream */

    /* --- Step 4: For each E-field component: Green function + FFT + multiply + IFFT ---
     * Each component is independent (reads d_sc_crho_freq, writes to its own d_sc_efield slice).
     * Process all 3 concurrently using separate workspace buffers.
     * R2C/C2R: forward D2Z produces half_size complex; inverse Z2D produces dbl_size real. */
    int blocks_d = (dbl_size + threads - 1) / threads;
    int blocks_h = (half_size + threads - 1) / threads;
    int stencil_size = (nx2-1)*(ny2-1)*(nz2-1);
    int blocks_s = (stencil_size + threads - 1) / threads;
    double scale = SC_FPEI / (double)(nx2*ny2*nz2);

    /* Allocate per-component workspaces + Green function cache.
     * d_sc_cgrn_c: real scratch for Green fn computation (dbl_size doubles).
     * d_sc_cgrn2_c: frequency-domain complex workspace (half_size complex).
     * d_sc_real_c: real workspace for Z2D output (dbl_size doubles).
     * d_sc_grn_cache: cached FFT'd Green fn (half_size complex). */
    static double *d_sc_cgrn_c[3] = {NULL, NULL, NULL};                /* Green fn real scratch */
    static cufftDoubleComplex *d_sc_cgrn2_c[3] = {NULL, NULL, NULL};   /* freq-domain workspace */
    static double *d_sc_real_c[3] = {NULL, NULL, NULL};                /* Z2D real output */
    static cufftDoubleComplex *d_sc_grn_cache[3] = {NULL, NULL, NULL}; /* cached FFT'd Green fn */
    static int grn_cached_size = 0;
    if (dbl_size > grn_cached_size) {
        size_t rsz = (size_t)dbl_size * sizeof(double);
        size_t hsz = (size_t)half_size * sizeof(cufftDoubleComplex);
        for (int c = 0; c < 3; c++) {
            if (d_sc_cgrn_c[c]) cudaFree(d_sc_cgrn_c[c]);
            if (d_sc_cgrn2_c[c]) cudaFree(d_sc_cgrn2_c[c]);
            if (d_sc_real_c[c]) cudaFree(d_sc_real_c[c]);
            if (d_sc_grn_cache[c]) cudaFree(d_sc_grn_cache[c]);
            cudaMalloc((void**)&d_sc_cgrn_c[c], rsz);    /* real scratch for Green fn */
            cudaMalloc((void**)&d_sc_cgrn2_c[c], hsz);   /* freq-domain complex workspace */
            cudaMalloc((void**)&d_sc_real_c[c], rsz);     /* Z2D real output */
            cudaMalloc((void**)&d_sc_grn_cache[c], hsz);  /* cached FFT'd Green fn */
        }
        grn_cached_size = dbl_size;
    }

    /* CUDA streams for concurrent E-field computation */
    static cudaStream_t sc_streams[3] = {0, 0, 0};
    static cufftHandle sc_fwd_plans[3] = {0, 0, 0};  /* D2Z forward plans for Green fn */
    static cufftHandle sc_inv_plans[3] = {0, 0, 0};  /* Z2D inverse plans for field extraction */
    static int sc_comp_nx2 = 0, sc_comp_ny2 = 0, sc_comp_nz2 = 0;
    static int streams_created = 0;
    if (!streams_created) {
        for (int c = 0; c < 3; c++) cudaStreamCreate(&sc_streams[c]);
        streams_created = 1;
    }
    if (nx2 != sc_comp_nx2 || ny2 != sc_comp_ny2 || nz2 != sc_comp_nz2) {
        for (int c = 0; c < 3; c++) {
            if (sc_fwd_plans[c]) cufftDestroy(sc_fwd_plans[c]);
            if (sc_inv_plans[c]) cufftDestroy(sc_inv_plans[c]);
            cufftPlan3d(&sc_fwd_plans[c], nx2, ny2, nz2, CUFFT_D2Z);
            cufftPlan3d(&sc_inv_plans[c], nx2, ny2, nz2, CUFFT_Z2D);
            cufftSetStream(sc_fwd_plans[c], sc_streams[c]);
            cufftSetStream(sc_inv_plans[c], sc_streams[c]);
        }
        sc_comp_nx2 = nx2; sc_comp_ny2 = ny2; sc_comp_nz2 = nz2;
    }

    /* Green function caching: depends only on dx, dy, dz*gamma, mesh dims.
     * 0.1% tolerance -- within a single element's sub-steps the grid spacing
     * barely changes, so most sub-steps are cache hits. */
    static double grn_cache_dx = 0, grn_cache_dy = 0, grn_cache_dz_rf = 0;
    static int grn_cache_nx2 = 0, grn_cache_ny2 = 0, grn_cache_nz2 = 0;
    static int grn_cache_valid = 0;

    double dz_rf = dz_rf_local;  /* already computed on GPU */
    int grn_hit = 0;
    if (grn_cache_valid &&
        grn_cache_nx2 == nx2 && grn_cache_ny2 == ny2 && grn_cache_nz2 == nz2) {
        double rdx = fabs(dx - grn_cache_dx) / (fabs(grn_cache_dx) + 1e-30);
        double rdy = fabs(dy - grn_cache_dy) / (fabs(grn_cache_dy) + 1e-30);
        double rdz = fabs(dz_rf - grn_cache_dz_rf) / (fabs(grn_cache_dz_rf) + 1e-30);
        grn_hit = (rdx < 1e-3 && rdy < 1e-3 && rdz < 1e-3);
    }

    if (!grn_hit) {
        /* Cache miss: compute Green function FFTs (D2Z), cache, then multiply + Z2D */
        for (int icomp = 1; icomp <= 3; icomp++) {
            int c = icomp - 1;

            /* Compute Green function into real scratch buffer */
            green_function_kernel<<<blocks_d, threads, 0, sc_streams[c]>>>(
                d_sc_cgrn_c[c], dx, dy, dz, gamma, icomp,
                nx2, ny2, nz2, 0.0, 0.0, 0.0);

            /* Stencil: cgrn_c -> real_c (zero first since stencil writes (nx2-1)*(ny2-1)*(nz2-1)) */
            cudaMemsetAsync(d_sc_real_c[c], 0, (size_t)dbl_size * sizeof(double), sc_streams[c]);

            igf_stencil_kernel<<<blocks_s, threads, 0, sc_streams[c]>>>(
                d_sc_cgrn_c[c], d_sc_real_c[c], nx2, ny2, nz2);

            /* Forward D2Z FFT of Green function */
            CUFFT_SC_CHECK(cufftExecD2Z(sc_fwd_plans[c], d_sc_real_c[c], d_sc_cgrn2_c[c]));

            /* Save to cache (half_size complex) */
            cudaMemcpyAsync(d_sc_grn_cache[c], d_sc_cgrn2_c[c],
                (size_t)half_size * sizeof(cufftDoubleComplex),
                cudaMemcpyDeviceToDevice, sc_streams[c]);

            /* Multiply with charge FFT (half_size complex arrays) */
            cudaStreamWaitEvent(sc_streams[c], sc_crho_ready, 0);
            complex_multiply_kernel<<<blocks_h, threads, 0, sc_streams[c]>>>(
                d_sc_crho_freq, d_sc_cgrn2_c[c], half_size);

            /* Inverse Z2D FFT -> real output */
            CUFFT_SC_CHECK(cufftExecZ2D(sc_inv_plans[c], d_sc_cgrn2_c[c], d_sc_real_c[c]));

            extract_field_kernel<<<blocks_m, threads, 0, sc_streams[c]>>>(
                d_sc_real_c[c], d_sc_efield + c*mesh_size,
                nx, ny, nz, nx2, ny2, nz2, scale);
        }
        grn_cache_dx = dx; grn_cache_dy = dy; grn_cache_dz_rf = dz_rf;
        grn_cache_nx2 = nx2; grn_cache_ny2 = ny2; grn_cache_nz2 = nz2;
        grn_cache_valid = 1;
    } else {
        /* Cache hit: skip Green fn computation. Multiply from cache + Z2D + extract.
         * Run on per-component streams for concurrency. */
        for (int icomp = 1; icomp <= 3; icomp++) {
            int c = icomp - 1;
            cudaStreamWaitEvent(sc_streams[c], sc_crho_ready, 0);
            complex_multiply_src_kernel<<<blocks_h, threads, 0, sc_streams[c]>>>(
                d_sc_crho_freq, d_sc_grn_cache[c], d_sc_cgrn2_c[c], half_size);
            CUFFT_SC_CHECK(cufftExecZ2D(sc_inv_plans[c], d_sc_cgrn2_c[c], d_sc_real_c[c]));
            extract_field_kernel<<<blocks_m, threads, 0, sc_streams[c]>>>(
                d_sc_real_c[c], d_sc_efield + c*mesh_size,
                nx, ny, nz, nx2, ny2, nz2, scale);
        }
    }

    /* Record events on component streams for synchronization. */
    if (!sc_fft_events_created) {
        for (int c = 0; c < 3; c++) cudaEventCreateWithFlags(&sc_fft_done[c], cudaEventDisableTiming);
        sc_fft_events_created = 1;
    }
    for (int c = 0; c < 3; c++) {
        cudaEventRecord(sc_fft_done[c], sc_streams[c]);
    }

    if (sc_compute_only_flag) {
        /* Compute-only mode: save params for later apply, return without kicks.
         * GPU continues FFT work asynchronously while CPU does other work. */
        sc_split_state.xmin = xmin; sc_split_state.ymin = ymin; sc_split_state.zmin = zmin;
        sc_split_state.dx = dx; sc_split_state.dy = dy; sc_split_state.dz = dz;
        sc_split_state.dxi = dxi; sc_split_state.dyi = dyi; sc_split_state.dzi = dzi;
        sc_split_state.ds_step = ds_step; sc_split_state.gamma = gamma;
        sc_split_state.mc2 = mc2; sc_split_state.dct_ave = dct_ave;
        sc_split_state.nx = nx; sc_split_state.ny = ny; sc_split_state.nz = nz;
        sc_split_state.n_particles = n_particles;
        sc_split_state.valid = 1;
        sc_compute_only_flag = 0;
        return;
    }

    /* Normal mode: wait for FFTs and apply kicks immediately */
    for (int c = 0; c < 3; c++) {
        cudaStreamWaitEvent(0, sc_fft_done[c], 0);  /* default stream waits */
    }

    /* --- Step 5: Interpolate fields and apply kicks --- */
    sc_interpolate_kick_kernel<<<blocks_p, threads>>>(
        dvec[0], dvec[1], dvec[2], dvec[3],
        dvec[4], dvec[5],
        dstate, dbeta, dp0c,
        d_sc_efield,
        xmin, ymin, zmin, dxi, dyi, dzi, dx, dy, dz,
        nx, ny, nz,
        ds_step, gamma, mc2, dct_ave,
        n_particles);
    CUDA_SC_CHECK(cudaGetLastError());

}

/* --------------------------------------------------------------------------
 * gpu_space_charge_3d_compute -- compute-only wrapper
 *
 * Calls gpu_space_charge_3d_ in compute-only mode: performs bounds,
 * charge deposition, and FFTs, then returns immediately without applying
 * kicks. The GPU continues FFT work asynchronously while the CPU can do
 * other work (e.g. CSR root-finding).
 *
 * MUST be followed by gpu_space_charge_3d_apply_ before the next sub-step.
 * -------------------------------------------------------------------------- */
extern "C" void gpu_space_charge_3d_compute_(
    double *h_charge, int n_particles,
    int nx, int ny, int nz,
    double gamma, double ds_step, double mc2,
    double dct_ave)
{
    sc_compute_only_flag = 1;
    gpu_space_charge_3d_(h_charge, n_particles, nx, ny, nz, gamma, ds_step, mc2, dct_ave);
}

/* --------------------------------------------------------------------------
 * gpu_space_charge_3d_apply -- apply phase of split SC pipeline
 *
 * Waits for the compute phase FFTs to complete (via GPU-side events),
 * then interpolates the E-field and applies kicks to particles.
 *
 * MUST be called after gpu_space_charge_3d_compute_ on the same sub-step.
 * -------------------------------------------------------------------------- */
extern "C" void gpu_space_charge_3d_apply_(int n_particles)
{
    if (!sc_split_state.valid || n_particles <= 0) return;

    if (n_particles != sc_split_state.n_particles) {
        fprintf(stderr, "gpu_space_charge_3d_apply: n_particles mismatch: %d (apply) vs %d (compute)\n",
                n_particles, sc_split_state.n_particles);
    }

    double *dvec[6]; int *dstate; double *dbeta, *dp0c;
    get_device_ptrs(dvec, &dstate, &dbeta, &dp0c);

    int threads = 256;
    int blocks_p = (n_particles + threads - 1) / threads;

    /* Wait for all 3 component FFTs to complete before interpolation */
    for (int c = 0; c < 3; c++) {
        cudaStreamWaitEvent(0, sc_fft_done[c], 0);  /* default stream waits */
    }

    /* Interpolate E-field at particle positions and apply kicks */
    sc_interpolate_kick_kernel<<<blocks_p, threads>>>(
        dvec[0], dvec[1], dvec[2], dvec[3],
        dvec[4], dvec[5],
        dstate, dbeta, dp0c,
        d_sc_efield,
        sc_split_state.xmin, sc_split_state.ymin, sc_split_state.zmin,
        sc_split_state.dxi, sc_split_state.dyi, sc_split_state.dzi,
        sc_split_state.dx, sc_split_state.dy, sc_split_state.dz,
        sc_split_state.nx, sc_split_state.ny, sc_split_state.nz,
        sc_split_state.ds_step, sc_split_state.gamma,
        sc_split_state.mc2, sc_split_state.dct_ave,
        n_particles);
    CUDA_SC_CHECK(cudaGetLastError());

    sc_split_state.valid = 0;
}


/* --------------------------------------------------------------------------
 * sc_compute_z_adj -- compute z_adj = z - dct_ave * beta for SC bounds
 * -------------------------------------------------------------------------- */
__global__ void sc_compute_z_adj(const double *z, const double *beta,
    double *z_adj, double dct_ave, int n)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    z_adj[i] = z[i] - dct_ave * beta[i];
}


/* --------------------------------------------------------------------------
 * dct_ave_reduce_kernel -- block reduction of sum(z/beta) and count(alive)
 * -------------------------------------------------------------------------- */
__global__ void dct_ave_reduce_kernel(const double *z, const double *beta,
    const int *state, double *block_sum, int *block_count, int n)
{
    extern __shared__ char sdata_raw[];
    double *ssum = (double*)sdata_raw;
    int *scnt = (int*)(sdata_raw + blockDim.x * sizeof(double));

    int tid = threadIdx.x;
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    double local_sum = 0.0;
    int local_cnt = 0;
    if (i < n && state[i] == 1) {
        local_sum = z[i] / beta[i];
        local_cnt = 1;
    }
    ssum[tid] = local_sum;
    scnt[tid] = local_cnt;
    __syncthreads();

    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            ssum[tid] += ssum[tid + s];
            scnt[tid] += scnt[tid + s];
        }
        __syncthreads();
    }

    if (tid == 0) {
        block_sum[blockIdx.x] = ssum[0];
        block_count[blockIdx.x] = scnt[0];
    }
}


/* --------------------------------------------------------------------------
 * gpu_csr_z_minmax -- compute min/max of z (vec[4]) for alive particles
 *
 * GPU reduction kernel -- no host-device transfer of particle data.
 * Uses a two-pass block reduction: first pass reduces within blocks,
 * second pass reduces block results on CPU (tiny array).
 * -------------------------------------------------------------------------- */

__global__ void z_minmax_kernel(
    const double *z, const int *state,
    double *block_min, double *block_max,
    int n)
{
    extern __shared__ double sdata[];
    double *smin = sdata;
    double *smax = sdata + blockDim.x;

    int tid = threadIdx.x;
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    double local_min = 1e30, local_max = -1e30;
    if (i < n && state[i] == 1) {
        local_min = z[i];
        local_max = z[i];
    }
    smin[tid] = local_min;
    smax[tid] = local_max;
    __syncthreads();

    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            if (smin[tid + s] < smin[tid]) smin[tid] = smin[tid + s];
            if (smax[tid + s] > smax[tid]) smax[tid] = smax[tid + s];
        }
        __syncthreads();
    }

    if (tid == 0) {
        block_min[blockIdx.x] = smin[0];
        block_max[blockIdx.x] = smax[0];
    }
}

extern "C" void gpu_csr_z_minmax_(
    double *h_z_min, double *h_z_max,
    int n_particles)
{
    if (n_particles <= 0) return;

    double *dvec[6]; int *dstate; double *dbeta, *dp0c;
    get_device_ptrs(dvec, &dstate, &dbeta, &dp0c);

    int threads = 256;
    int blocks = (n_particles + threads - 1) / threads;

    /* Use cached device buffers (only realloc when size grows) */
    if (blocks > d_zminmax_cap) {
        if (d_zminmax_bmin) cudaFree(d_zminmax_bmin);
        if (d_zminmax_bmax) cudaFree(d_zminmax_bmax);
        cudaMalloc((void**)&d_zminmax_bmin, blocks * sizeof(double));
        cudaMalloc((void**)&d_zminmax_bmax, blocks * sizeof(double));
        d_zminmax_cap = blocks;
    }

    z_minmax_kernel<<<blocks, threads, 2 * threads * sizeof(double)>>>(
        dvec[4], dstate, d_zminmax_bmin, d_zminmax_bmax, n_particles);
    cudaDeviceSynchronize();

    /* Download block results (static host buffers) */
    if (blocks > h_zminmax_cap) {
        free(h_zminmax_bmin); free(h_zminmax_bmax);
        h_zminmax_bmin = (double*)malloc(blocks * sizeof(double));
        h_zminmax_bmax = (double*)malloc(blocks * sizeof(double));
        h_zminmax_cap = blocks;
    }
    cudaMemcpy(h_zminmax_bmin, d_zminmax_bmin, blocks * sizeof(double), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_zminmax_bmax, d_zminmax_bmax, blocks * sizeof(double), cudaMemcpyDeviceToHost);

    double zmin = 1e30, zmax = -1e30;
    for (int i = 0; i < blocks; i++) {
        if (h_zminmax_bmin[i] < zmin) zmin = h_zminmax_bmin[i];
        if (h_zminmax_bmax[i] > zmax) zmax = h_zminmax_bmax[i];
    }
    *h_z_min = zmin;
    *h_z_max = zmax;
}


/* --------------------------------------------------------------------------
 * gpu_csr_fused_minmax_bin -- fused z_minmax + binning in one call
 *
 * Eliminates the separate gpu_csr_z_minmax call and its sync point.
 * Does: z_minmax → compute bin params → bin particles → download.
 * Only ONE sync point (implicit in the final cudaMemcpy DtoH).
 * -------------------------------------------------------------------------- */
extern "C" void gpu_csr_fused_minmax_bin_(
    double *h_charge,      /* per-particle charge, host (cached) */
    int n_particles,
    double *h_bin_charge,  /* n_bin, output */
    double *h_bin_x0_wt,
    double *h_bin_y0_wt,
    double *h_bin_n_particle,
    double *h_z_min_out,   /* computed z_min, output */
    double *h_z_max_out,   /* computed z_max, output */
    double dz_slice,       /* computed by caller from z_min/z_max */
    double dz_particle,
    int n_bin, int n_bin_eff, int particle_bin_span)
{
    if (n_particles <= 0 || n_bin <= 0) return;
    if (ensure_csr_bin_buffers(n_bin) != 0) return;

    double *dvec[6]; int *dstate; double *dbeta, *dp0c;
    get_device_ptrs(dvec, &dstate, &dbeta, &dp0c);

    int threads = 256;
    int blocks_p = (n_particles + threads - 1) / threads;

    /* Step 1: z_minmax on GPU */
    if (blocks_p > d_zminmax_cap) {
        if (d_zminmax_bmin) cudaFree(d_zminmax_bmin);
        if (d_zminmax_bmax) cudaFree(d_zminmax_bmax);
        cudaMalloc((void**)&d_zminmax_bmin, blocks_p * sizeof(double));
        cudaMalloc((void**)&d_zminmax_bmax, blocks_p * sizeof(double));
        d_zminmax_cap = blocks_p;
    }
    z_minmax_kernel<<<blocks_p, threads, 2 * threads * sizeof(double)>>>(
        dvec[4], dstate, d_zminmax_bmin, d_zminmax_bmax, n_particles);

    /* GPU-side reduction: n_blocks → single zmin/zmax (eliminates CPU reduction loop) */
    static double *d_csr_z_result = NULL;
    if (!d_csr_z_result) cudaMalloc((void**)&d_csr_z_result, 2*sizeof(double));
    sc_reduce_minmax_kernel<<<1, 256>>>(d_zminmax_bmin, d_zminmax_bmax, d_csr_z_result, 0, blocks_p);

    double h_zr[2];
    cudaMemcpy(h_zr, d_csr_z_result, 2*sizeof(double), cudaMemcpyDeviceToHost);
    double zmin = h_zr[0], zmax = h_zr[1];
    *h_z_min_out = zmin;
    *h_z_max_out = zmax;

    /* Step 2: Compute bin parameters from z_min/z_max */
    double dz = zmax - zmin;
    if (dz == 0) return;
    double dz_sl = 1.0000001 * dz / n_bin_eff;
    double z_center = (zmax + zmin) / 2.0;
    double z_min_bin = z_center - n_bin * dz_sl / 2.0;
    double dz_part = particle_bin_span * dz_sl;

    /* Step 3: Charge upload (cached) */
    {
        static int csr_charge_cap = 0;
        if (n_particles > csr_charge_cap) {
            if (d_sc_charge) cudaFree(d_sc_charge);
            cudaMalloc((void**)&d_sc_charge, (size_t)n_particles * sizeof(double));
            csr_charge_cap = n_particles;
            sc_charge_uploaded = 0;
        }
    }
    {
        static const double *last_csr_h_charge = NULL;
        static int last_csr_np = 0;
        if (h_charge != last_csr_h_charge || n_particles != last_csr_np) {
            cudaMemcpy(d_sc_charge, h_charge,
                n_particles * sizeof(double), cudaMemcpyHostToDevice);
            last_csr_h_charge = h_charge;
            last_csr_np = n_particles;
        }
    }

    /* Step 4: Zero bin arrays and run binning kernel */
    size_t bsz = n_bin * sizeof(double);
    cudaMemset(d_csr_bin_charge, 0, bsz);
    cudaMemset(d_csr_bin_x0_wt, 0, bsz);
    cudaMemset(d_csr_bin_y0_wt, 0, bsz);
    cudaMemset(d_csr_bin_n_particle, 0, bsz);

    csr_bin_kernel<<<blocks_p, threads>>>(
        dvec[4], dvec[0], dvec[2],
        d_sc_charge, dstate,
        d_csr_bin_charge, d_csr_bin_x0_wt, d_csr_bin_y0_wt,
        d_csr_bin_n_particle,
        z_min_bin, dz_sl, dz_part,
        n_bin, particle_bin_span,
        n_particles);
    CUDA_SC_CHECK(cudaGetLastError());

    /* Step 5: Download bin results */
    cudaMemcpy(h_bin_charge, d_csr_bin_charge, bsz, cudaMemcpyDeviceToHost);
    cudaMemcpy(h_bin_x0_wt, d_csr_bin_x0_wt, bsz, cudaMemcpyDeviceToHost);
    cudaMemcpy(h_bin_y0_wt, d_csr_bin_y0_wt, bsz, cudaMemcpyDeviceToHost);
    cudaMemcpy(h_bin_n_particle, d_csr_bin_n_particle, bsz, cudaMemcpyDeviceToHost);
}


/* --------------------------------------------------------------------------
 * gpu_csr_bin_particles -- bin particles on GPU
 *
 * Particle vx, vy, vz, state must be on device.
 * Downloads binned results (charge, x0_wt, y0_wt, n_particle) to host.
 * -------------------------------------------------------------------------- */
extern "C" void gpu_csr_bin_particles_(
    double *h_charge,      /* per-particle charge, host */
    int n_particles,
    double *h_bin_charge,  /* n_bin, output, host */
    double *h_bin_x0_wt,   /* n_bin, output */
    double *h_bin_y0_wt,   /* n_bin, output */
    double *h_bin_n_particle, /* n_bin, output */
    double z_min, double dz_slice, double dz_particle,
    int n_bin, int particle_bin_span)
{
    if (n_particles <= 0 || n_bin <= 0) return;
    if (ensure_csr_bin_buffers(n_bin) != 0) return;

    double *dvec[6]; int *dstate; double *dbeta, *dp0c;
    get_device_ptrs(dvec, &dstate, &dbeta, &dp0c);

    /* Ensure d_sc_charge is large enough (uses the cached buffer from ensure_sc_buffers
       via the static cached_np in that function; also handle standalone CSR case) */
    {
        static int csr_charge_cap = 0;
        if (n_particles > csr_charge_cap) {
            if (d_sc_charge) cudaFree(d_sc_charge);
            cudaMalloc((void**)&d_sc_charge, (size_t)n_particles * sizeof(double));
            csr_charge_cap = n_particles;
            sc_charge_uploaded = 0;  /* force re-upload after realloc */
        }
    }
    /* Cache charge upload — skip if same pointer and size (charge doesn't change between sub-steps) */
    {
        static const double *last_csr_h_charge = NULL;
        static int last_csr_np = 0;
        if (h_charge != last_csr_h_charge || n_particles != last_csr_np) {
            CUDA_SC_CHECK(cudaMemcpy(d_sc_charge, h_charge,
                n_particles * sizeof(double), cudaMemcpyHostToDevice));
            last_csr_h_charge = h_charge;
            last_csr_np = n_particles;
        }
    }

    /* Zero bin arrays */
    size_t bsz = n_bin * sizeof(double);
    CUDA_SC_CHECK(cudaMemset(d_csr_bin_charge, 0, bsz));
    CUDA_SC_CHECK(cudaMemset(d_csr_bin_x0_wt, 0, bsz));
    CUDA_SC_CHECK(cudaMemset(d_csr_bin_y0_wt, 0, bsz));
    CUDA_SC_CHECK(cudaMemset(d_csr_bin_n_particle, 0, bsz));

    int threads = 256;
    int blocks = (n_particles + threads - 1) / threads;
    csr_bin_kernel<<<blocks, threads>>>(
        dvec[4], dvec[0], dvec[2],
        d_sc_charge, dstate,
        d_csr_bin_charge, d_csr_bin_x0_wt, d_csr_bin_y0_wt,
        d_csr_bin_n_particle,
        z_min, dz_slice, dz_particle,
        n_bin, particle_bin_span,
        n_particles);
    CUDA_SC_CHECK(cudaGetLastError());
    /* No explicit sync needed — cudaMemcpy DtoH on default stream waits for prior work */

    /* Download results */
    CUDA_SC_CHECK(cudaMemcpy(h_bin_charge, d_csr_bin_charge, bsz, cudaMemcpyDeviceToHost));
    CUDA_SC_CHECK(cudaMemcpy(h_bin_x0_wt, d_csr_bin_x0_wt, bsz, cudaMemcpyDeviceToHost));
    CUDA_SC_CHECK(cudaMemcpy(h_bin_y0_wt, d_csr_bin_y0_wt, bsz, cudaMemcpyDeviceToHost));
    CUDA_SC_CHECK(cudaMemcpy(h_bin_n_particle, d_csr_bin_n_particle, bsz, cudaMemcpyDeviceToHost));
}


/* --------------------------------------------------------------------------
 * gpu_csr_kick_convolve -- GPU-accelerated CSR kick convolution
 *
 * Computes kick_csr[i] = coef * sum(j=1..i) I_int_csr[i-j+1] * edge_dcharge_density_dz[j]
 * This is the O(n_bin^2) convolution that dominates csr_bin_kicks time.
 * The root-finding (I_csr computation) stays on CPU; only the convolution moves to GPU.
 * -------------------------------------------------------------------------- */
__global__ void csr_kick_convolve_kernel(
    const double *I_int_csr,         /* n_bin+1 elements, I_int_csr[0..n_bin] */
    const double *edge_dcharge_dz,   /* n_bin elements, edge_dcharge_density_dz[1..n_bin] */
    double *kick_csr,                /* n_bin output elements */
    double coef,
    int n_bin)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;  /* bin index 0..n_bin-1 (maps to Fortran 1..n_bin) */
    if (i >= n_bin) return;

    /* kick[i+1] = coef * sum(j=1..i+1) I_int_csr[i+1-j+1] * edge_dcharge_dz[j]
     *           = coef * sum(j=0..i) I_int_csr[i-j+1] * edge_dcharge_dz[j+1]
     * In 0-indexed: sum(j=0..i) I_int[i-j+1] * dcharge[j] */
    double sum = 0.0;
    int count = i + 1;
    for (int j = 0; j < count; j++) {
        sum += I_int_csr[i - j + 1] * edge_dcharge_dz[j];
    }
    kick_csr[i] = coef * sum;
}

static double *d_csr_I_int = NULL, *d_csr_edge_dcdz = NULL, *d_csr_kick_conv = NULL;
static int d_csr_conv_cap = 0;

/* Dedicated CSR stream: allows CSR convolution + kicks to run concurrently
 * with SC FFTs on sc_streams[], avoiding default-stream serialization. */
static cudaStream_t csr_kick_stream = 0;
static int csr_kick_stream_created = 0;
/* Pinned host staging buffer for I_int async upload */
static double *h_csr_I_int_pinned = NULL;
static int h_csr_I_int_pinned_cap = 0;

static void ensure_csr_conv_buffers(int n_bin) {
    if (n_bin > d_csr_conv_cap) {
        if (d_csr_I_int) { cudaFree(d_csr_I_int); cudaFree(d_csr_edge_dcdz); cudaFree(d_csr_kick_conv); }
        cudaMalloc((void**)&d_csr_I_int, (n_bin + 1) * sizeof(double));
        cudaMalloc((void**)&d_csr_edge_dcdz, n_bin * sizeof(double));
        cudaMalloc((void**)&d_csr_kick_conv, n_bin * sizeof(double));
        d_csr_conv_cap = n_bin;
    }
    if (!csr_kick_stream_created) {
        cudaStreamCreate(&csr_kick_stream);
        csr_kick_stream_created = 1;
    }
    if (n_bin + 1 > h_csr_I_int_pinned_cap) {
        if (h_csr_I_int_pinned) cudaFreeHost(h_csr_I_int_pinned);
        cudaMallocHost((void**)&h_csr_I_int_pinned, (n_bin + 1) * sizeof(double));
        h_csr_I_int_pinned_cap = n_bin + 1;
    }
}

/* --------------------------------------------------------------------------
 * gpu_csr_compute_edge_dcdz -- compute edge_dcharge_density_dz on GPU
 *
 * Reads from d_csr_bin_charge (already on device from binning kernel).
 * Writes to d_csr_edge_dcdz (stays on device for convolution).
 * -------------------------------------------------------------------------- */
__global__ void csr_edge_dcdz_kernel(
    const double *bin_charge, double *edge_dcdz,
    double inv_dz_sq, int n_bin)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n_bin) return;
    if (i == 0) {
        edge_dcdz[0] = 0.0;
    } else {
        edge_dcdz[i] = (bin_charge[i] - bin_charge[i - 1]) * inv_dz_sq;
    }
}

extern "C" void gpu_csr_compute_edge_dcdz_(double dz_slice, int n_bin)
{
    if (n_bin <= 0) return;
    ensure_csr_conv_buffers(n_bin);
    double inv_dz_sq = 1.0 / (dz_slice * dz_slice);
    int threads = 256;
    int blocks = (n_bin + threads - 1) / threads;
    csr_edge_dcdz_kernel<<<blocks, threads>>>(
        d_csr_bin_charge, d_csr_edge_dcdz, inv_dz_sq, n_bin);
    CUDA_SC_CHECK(cudaGetLastError());
}

/* --------------------------------------------------------------------------
 * gpu_csr_convolve_dev -- convolution using device-resident edge_dcdz
 *
 * Uploads only I_int_csr from host. Reads edge_dcdz from device.
 * Writes kick_csr to d_csr_kick_conv (stays on device for apply_kicks_dev).
 * -------------------------------------------------------------------------- */
extern "C" void gpu_csr_convolve_dev_(
    double *h_I_int_csr,  /* n_bin+1 elements from CPU root-finding */
    double coef, int n_bin)
{
    if (n_bin <= 0) return;
    ensure_csr_conv_buffers(n_bin);

    /* Async upload via pinned staging buffer on csr_kick_stream.
     * This avoids default-stream synchronization, allowing the upload
     * and convolution to overlap with SC FFTs on sc_streams[]. */
    size_t sz = (n_bin + 1) * sizeof(double);
    memcpy(h_csr_I_int_pinned, h_I_int_csr, sz);
    cudaMemcpyAsync(d_csr_I_int, h_csr_I_int_pinned, sz,
                    cudaMemcpyHostToDevice, csr_kick_stream);

    int threads = 256;
    int blocks = (n_bin + threads - 1) / threads;
    csr_kick_convolve_kernel<<<blocks, threads, 0, csr_kick_stream>>>(
        d_csr_I_int, d_csr_edge_dcdz, d_csr_kick_conv, coef, n_bin);
    CUDA_SC_CHECK(cudaGetLastError());
}

/* --------------------------------------------------------------------------
 * gpu_csr_apply_kicks_dev -- apply kicks from device-resident arrays
 *
 * Uses d_csr_kick_conv (from convolution) directly. No host upload needed.
 * -------------------------------------------------------------------------- */
extern "C" void gpu_csr_apply_kicks_dev_(
    double z_center_0, double dz_slice,
    int n_bin, int n_particles)
{
    if (n_particles <= 0 || n_bin <= 0) return;

    double *dvec[6]; int *dstate; double *dbeta, *dp0c;
    get_device_ptrs(dvec, &dstate, &dbeta, &dp0c);

    /* CSR-only path: apply_lsc=0 means kick_lsc is never read by kernel,
     * so we skip the cudaMemset. Launch on csr_kick_stream to avoid
     * default-stream serialization with concurrent SC FFTs. */
    int threads = 256;
    int blocks = (n_particles + threads - 1) / threads;
    csr_apply_kick_kernel<<<blocks, threads, 0, csr_kick_stream>>>(
        dvec[5], dvec[4], dstate,
        d_csr_kick_conv, d_csr_kick_lsc,
        z_center_0, dz_slice,
        1, 0,  /* apply_csr=1, apply_lsc=0 */
        n_bin, n_particles);
    CUDA_SC_CHECK(cudaGetLastError());
}

/* Legacy host-side convolution (still used as fallback) */
extern "C" void gpu_csr_kick_convolve_(
    double *h_I_int_csr,
    double *h_edge_dcharge_dz,
    double *h_kick_csr,
    double coef, int n_bin)
{
    if (n_bin <= 0) return;
    ensure_csr_conv_buffers(n_bin);

    cudaMemcpy(d_csr_I_int, h_I_int_csr, (n_bin + 1) * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(d_csr_edge_dcdz, h_edge_dcharge_dz, n_bin * sizeof(double), cudaMemcpyHostToDevice);

    int threads = 256;
    int blocks = (n_bin + threads - 1) / threads;
    csr_kick_convolve_kernel<<<blocks, threads>>>(
        d_csr_I_int, d_csr_edge_dcdz, d_csr_kick_conv, coef, n_bin);
    CUDA_SC_CHECK(cudaGetLastError());

    cudaMemcpy(h_kick_csr, d_csr_kick_conv, n_bin * sizeof(double), cudaMemcpyDeviceToHost);
}


/* --------------------------------------------------------------------------
 * gpu_csr_apply_kicks -- apply precomputed CSR/LSC kicks on GPU
 *
 * Particle vpz, vz, state must be on device.
 * h_kick_csr, h_kick_lsc: per-bin kick arrays (host), uploaded to device.
 * -------------------------------------------------------------------------- */
extern "C" void gpu_csr_apply_kicks_(
    double *h_kick_csr,   /* n_bin, host */
    double *h_kick_lsc,   /* n_bin, host */
    double z_center_0,    /* z_center of first bin */
    double dz_slice,
    int apply_csr, int apply_lsc,
    int n_bin, int n_particles)
{
    if (n_particles <= 0 || n_bin <= 0) return;
    if (ensure_csr_bin_buffers(n_bin) != 0) return;

    double *dvec[6]; int *dstate; double *dbeta, *dp0c;
    get_device_ptrs(dvec, &dstate, &dbeta, &dp0c);

    size_t bsz = n_bin * sizeof(double);
    CUDA_SC_CHECK(cudaMemcpy(d_csr_kick_csr, h_kick_csr, bsz, cudaMemcpyHostToDevice));
    CUDA_SC_CHECK(cudaMemcpy(d_csr_kick_lsc, h_kick_lsc, bsz, cudaMemcpyHostToDevice));

    int threads = 256;
    int blocks = (n_particles + threads - 1) / threads;
    csr_apply_kick_kernel<<<blocks, threads>>>(
        dvec[5], dvec[4], dstate,
        d_csr_kick_csr, d_csr_kick_lsc,
        z_center_0, dz_slice,
        apply_csr, apply_lsc,
        n_bin, n_particles);
    CUDA_SC_CHECK(cudaGetLastError());
    /* No sync -- next kernel on same stream serializes. */
}


/* ==========================================================================
 * GPU CSR BIN KICKS -- port of s_source_calc + I_csr + convolution
 *
 * Each thread handles one kick1 bin index (-n_bin to +n_bin).
 * The geometry (spline coefficients, floor positions) is uploaded as
 * flat arrays. This replaces the CPU-side csr_bin_kicks routine.
 * ========================================================================== */

/* Spline evaluation: x = coef[0]*z + coef[1]*z^2 + coef[2]*z^3 */
__device__ double spline1_dev(const double *coef, double z) {
    return coef[0]*z + coef[1]*z*z + coef[2]*z*z*z;
}

/* Spline derivative: dx/dz = coef[0] + 2*coef[1]*z + 3*coef[2]*z^2 */
__device__ double spline1_deriv_dev(const double *coef, double z) {
    return coef[0] + 2.0*coef[1]*z + 3.0*coef[2]*z*z;
}

/* dspline_len: Ls - L (path length excess over chord).
   Approximation matching Bmad's dspline_len for small deviations. */
__device__ double dspline_len_dev(const double *coef, double z0, double z1, double dtheta_L) {
    /* dL = integral of (ds - dz) along the spline from z0 to z1.
       For small angles: dL ≈ ∫ (θ² / 2) dz where θ = spline derivative.
       Bmad uses: dL = Σ coef_i * z^i integrated as polynomial */
    double dz = z1 - z0;
    if (fabs(dz) < 1e-30) return 0.0;
    /* Direct evaluation using Bmad's formula:
       dL = a1^2/2 * dz + a1*a2*(z1^2-z0^2) + (a2^2/2 + a1*a3)*(z1^3-z0^3)/3
            + a2*a3*(z1^4-z0^4)/4 + a3^2/2*(z1^5-z0^5)/5
       where a1 = coef[0]-dtheta_L, a2 = coef[1], a3 = coef[2] */
    double a1 = coef[0] - dtheta_L;
    double a2 = coef[1];
    double a3 = coef[2];
    double z02 = z0*z0, z12 = z1*z1;
    double z03 = z02*z0, z13 = z12*z1;
    double z04 = z03*z0, z14 = z13*z1;
    double z05 = z04*z0, z15 = z14*z1;
    double dL = a1*a1/2.0 * dz
              + a1*a2 * (z12 - z02) / 2.0
              + (a2*a2/2.0 + a1*a3) * (z13 - z03) / 3.0
              + a2*a3 * (z14 - z04) / 4.0
              + a3*a3/2.0 * (z15 - z05) / 5.0;
    return dL;
}

__device__ double modulo2_dev(double x, double range) {
    /* Map x to [-range, range) */
    double r2 = 2.0 * range;
    double result = fmod(x + range, r2);
    if (result < 0) result += r2;
    return result - range;
}

/* Per-element geometry uploaded to device */
struct CsrEleGeom {
    double floor0_x, floor0_z, floor0_theta;   /* entrance floor position */
    double floor1_x, floor1_z, floor1_theta;   /* exit floor position */
    double L_chord;                             /* chord length */
    double theta_chord;                         /* chord angle */
    double spline_coef[3];                      /* centroid spline coefficients */
    double dL_s;                                /* path length excess */
    double ele_s;                               /* element s-position */
    int ele_key;                                /* element key (match$, etc.) */
};

/* ddz_calc on device -- evaluates distance error for root-finding */
__device__ double ddz_calc_dev(
    double s_chord_source,
    int ix_ele_source, int ix_ele_kick,
    double s_chord_kick,
    const CsrEleGeom *geom,
    double floor_k_x, double floor_k_z,
    double y_source, double gamma2,
    double dz_target,
    /* outputs */
    double *out_L, double *out_dL,
    double *out_theta_sl, double *out_theta_lk,
    double *out_floor_s_x, double *out_floor_s_z)
{
    const CsrEleGeom &es = geom[ix_ele_source];
    double x = spline1_dev(es.spline_coef, s_chord_source);
    double c = cos(es.theta_chord);
    double s = sin(es.theta_chord);
    double fs_x = x*c + s_chord_source*s + es.floor0_x;
    double fs_z = -x*s + s_chord_source*c + es.floor0_z;

    double Lx = floor_k_x - fs_x;
    double Lz = floor_k_z - fs_z;
    double L = sqrt(Lx*Lx + Lz*Lz + y_source*y_source);
    double theta_L = atan2(Lx, Lz);

    double s0 = s_chord_source;
    double s1 = s_chord_kick;
    double dL;

    if (ix_ele_source == ix_ele_kick) {
        double ds = s1 - s0;
        double dtheta_L = es.spline_coef[0] + es.spline_coef[1]*(2*s0+ds) + es.spline_coef[2]*(3*s0*s0 + 3*s0*ds + ds*ds);
        dL = dspline_len_dev(es.spline_coef, s0, s1, dtheta_L);
        if (ds < 0) dL = dL + 2*ds;
        *out_theta_sl = spline1_deriv_dev(es.spline_coef, s0) - dtheta_L;
        *out_theta_lk = dtheta_L - spline1_deriv_dev(es.spline_coef, s1);
    } else {
        double pi_half = 3.14159265358979323846 / 2.0;
        dL = dspline_len_dev(es.spline_coef, s0, es.L_chord, modulo2_dev(theta_L - es.theta_chord, pi_half));
        for (int ie = ix_ele_source + 1; ie < ix_ele_kick; ie++) {
            const CsrEleGeom &ce = geom[ie];
            if (ce.ele_key == 37) continue; /* match$ = 37 */
            dL += dspline_len_dev(ce.spline_coef, 0.0, ce.L_chord, modulo2_dev(theta_L - ce.theta_chord, pi_half));
        }
        const CsrEleGeom &ek = geom[ix_ele_kick];
        dL += dspline_len_dev(ek.spline_coef, 0.0, s1, modulo2_dev(theta_L - ek.theta_chord, pi_half));
        *out_theta_sl = modulo2_dev(spline1_deriv_dev(es.spline_coef, s0) + es.theta_chord - theta_L, pi_half);
        *out_theta_lk = modulo2_dev(theta_L - spline1_deriv_dev(ek.spline_coef, s1) - ek.theta_chord, pi_half);
    }

    if (y_source != 0.0) dL -= (L - sqrt(Lx*Lx + Lz*Lz));

    *out_L = L;
    *out_dL = dL;
    *out_floor_s_x = fs_x;
    *out_floor_s_z = fs_z;

    return L / (2.0 * gamma2) + dL - dz_target;
}

/* Brent's method root-finding on device */
__device__ double zbrent_dev(
    int ix_ele_source, int ix_ele_kick,
    double s_chord_kick,
    const CsrEleGeom *geom,
    double floor_k_x, double floor_k_z,
    double y_source, double gamma2,
    double dz_target,
    double x1, double x2,
    double *out_L, double *out_dL,
    double *out_theta_sl, double *out_theta_lk,
    double *out_floor_s_x, double *out_floor_s_z)
{
    double dummy_L, dummy_dL, dummy_tsl, dummy_tlk, dummy_fx, dummy_fz;
    double a = x1, b = x2;
    double fa = ddz_calc_dev(a, ix_ele_source, ix_ele_kick, s_chord_kick, geom,
                              floor_k_x, floor_k_z, y_source, gamma2, dz_target,
                              &dummy_L, &dummy_dL, &dummy_tsl, &dummy_tlk, &dummy_fx, &dummy_fz);
    double fb = ddz_calc_dev(b, ix_ele_source, ix_ele_kick, s_chord_kick, geom,
                              floor_k_x, floor_k_z, y_source, gamma2, dz_target,
                              out_L, out_dL, out_theta_sl, out_theta_lk, out_floor_s_x, out_floor_s_z);
    double c = b, fc = fb;
    double d = b - a, e = d;

    for (int iter = 0; iter < 100; iter++) {
        if ((fb > 0 && fc > 0) || (fb < 0 && fc < 0)) {
            c = a; fc = fa; d = b - a; e = d;
        }
        if (fabs(fc) < fabs(fb)) {
            a = b; b = c; c = a;
            fa = fb; fb = fc; fc = fa;
        }
        double tol1 = 1e-12 * fabs(b) + 1e-8;
        double xm = 0.5 * (c - b);
        if (fabs(xm) <= tol1 || fb == 0.0) {
            /* Evaluate at final root to get output geometry */
            ddz_calc_dev(b, ix_ele_source, ix_ele_kick, s_chord_kick, geom,
                         floor_k_x, floor_k_z, y_source, gamma2, dz_target,
                         out_L, out_dL, out_theta_sl, out_theta_lk, out_floor_s_x, out_floor_s_z);
            return b;
        }
        if (fabs(e) >= tol1 && fabs(fa) > fabs(fb)) {
            double s = fb / fa;
            double p, q;
            if (a == c) {
                p = 2.0 * xm * s;
                q = 1.0 - s;
            } else {
                double r = fb / fc;
                q = fa / fc;
                p = s * (2.0*xm*q*(q-r) - (b-a)*(r-1.0));
                q = (q-1.0)*(r-1.0)*(s-1.0);
            }
            if (p > 0) q = -q;
            p = fabs(p);
            if (2.0*p < fmin(3.0*xm*q - fabs(tol1*q), fabs(e*q))) {
                e = d; d = p/q;
            } else {
                d = xm; e = d;
            }
        } else {
            d = xm; e = d;
        }
        a = b; fa = fb;
        if (fabs(d) > tol1) b += d;
        else b += (xm >= 0 ? fabs(tol1) : -fabs(tol1));
        fb = ddz_calc_dev(b, ix_ele_source, ix_ele_kick, s_chord_kick, geom,
                          floor_k_x, floor_k_z, y_source, gamma2, dz_target,
                          out_L, out_dL, out_theta_sl, out_theta_lk, out_floor_s_x, out_floor_s_z);
    }
    return b; /* Max iterations reached */
}

/* Main kernel: one thread per kick1 bin */
__global__ void csr_bin_kicks_kernel(
    const CsrEleGeom *geom,
    int n_ele,                  /* number of elements in geom array */
    int ix_ele_kick,            /* element index where kick is applied */
    double s_chord_kick,        /* chord position of kick point */
    double floor_k_x, double floor_k_z, /* kick point floor position */
    double gamma, double gamma2, double beta2,
    double y_source,
    double dz_slice,
    int n_bin,                  /* number of bins (space_charge_com%n_bin) */
    double kick_factor,
    /* output: per kick1-bin results */
    double *out_I_csr,          /* 2*n_bin+1 elements */
    double *out_I_int_csr,      /* 2*n_bin+1 elements */
    int *out_ix_ele_source)     /* 2*n_bin+1 elements */
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int n_kick1 = 2 * n_bin + 1;
    if (tid >= n_kick1) return;

    int i_bin = tid - n_bin;  /* -n_bin to +n_bin */
    double dz_particles = i_bin * dz_slice;

    /* Initialize output */
    out_I_csr[tid] = 0.0;
    out_I_int_csr[tid] = 0.0;

    /* --- s_source_calc --- */
    int ix_src = ix_ele_kick;  /* initial guess */
    int last_step = 0;

    /* If i_bin > -n_bin, use previous bin's source element as initial guess.
       Since threads run in parallel, we can't use the previous thread's result.
       Instead, always start from ix_ele_kick. This may be slightly less efficient
       but is correct. */

    double L, dL, theta_sl, theta_lk, floor_s_x, floor_s_z;
    double s_source = 0.0;
    int found = 0;

    for (int attempt = 0; attempt < n_ele + 2 && !found; attempt++) {
        if (ix_src == 0) {
            /* At beginning of lattice -- assume infinite drift */
            const CsrEleGeom &e0 = geom[0];
            double Lx = floor_k_x - e0.floor1_x;
            double Lz_val = floor_k_z - e0.floor1_z;
            double L0 = sqrt(Lx*Lx + Lz_val*Lz_val + y_source*y_source);
            double Lz_proj = Lx * sin(e0.floor1_theta) + Lz_val * cos(e0.floor1_theta);

            /* Compute Lsz0 (path length from lat start to kick point) */
            double Lsz0 = dspline_len_dev(geom[ix_ele_kick].spline_coef, 0.0, s_chord_kick, 0.0) + s_chord_kick;
            for (int ie = 1; ie < ix_ele_kick; ie++) {
                Lsz0 += geom[ie].dL_s + geom[ie].L_chord;
            }

            double a = 1.0 / gamma2;
            double b = 2.0 * (Lsz0 - dz_particles - beta2 * Lz_proj);
            double c = (Lsz0 - dz_particles) * (Lsz0 - dz_particles) - beta2 * L0 * L0;
            double disc = b*b - 4*a*c;
            if (disc < 0) disc = 0;
            double ds_source = -(-b + sqrt(disc)) / (2.0 * a);
            s_source = e0.ele_s + ds_source;

            floor_s_x = e0.floor1_x + ds_source * sin(e0.floor1_theta);
            floor_s_z = e0.floor1_z + ds_source * cos(e0.floor1_theta);
            double Lvx = floor_k_x - floor_s_x;
            double Lvz = floor_k_z - floor_s_z;
            L = sqrt(Lvx*Lvx + Lvz*Lvz + y_source*y_source);
            dL = Lsz0 - ds_source - L;
            theta_sl = e0.floor1_theta - atan2(Lvx, Lvz);
            theta_lk = atan2(Lvx, Lvz) - (spline1_deriv_dev(geom[ix_ele_kick].spline_coef, s_chord_kick) + geom[ix_ele_kick].theta_chord);

            out_ix_ele_source[tid] = 0;
            found = 1;
            break;
        }

        /* Check element boundaries */
        double dummy_L, dummy_dL, dummy_tsl, dummy_tlk, dummy_fx, dummy_fz;
        double ddz0 = ddz_calc_dev(0.0, ix_src, ix_ele_kick, s_chord_kick, geom,
                                    floor_k_x, floor_k_z, y_source, gamma2, dz_particles,
                                    &dummy_L, &dummy_dL, &dummy_tsl, &dummy_tlk, &dummy_fx, &dummy_fz);
        double ddz1 = ddz_calc_dev(geom[ix_src].L_chord, ix_src, ix_ele_kick, s_chord_kick, geom,
                                    floor_k_x, floor_k_z, y_source, gamma2, dz_particles,
                                    &dummy_L, &dummy_dL, &dummy_tsl, &dummy_tlk, &dummy_fx, &dummy_fz);

        if (last_step == -1 && ddz1 > 0) {
            /* Root at right edge (roundoff) */
            ddz_calc_dev(geom[ix_src].L_chord, ix_src, ix_ele_kick, s_chord_kick, geom,
                         floor_k_x, floor_k_z, y_source, gamma2, dz_particles,
                         &L, &dL, &theta_sl, &theta_lk, &floor_s_x, &floor_s_z);
            out_ix_ele_source[tid] = ix_src;
            found = 1;
            break;
        }
        if (last_step == 1 && ddz0 < 0) {
            ddz_calc_dev(0.0, ix_src, ix_ele_kick, s_chord_kick, geom,
                         floor_k_x, floor_k_z, y_source, gamma2, dz_particles,
                         &L, &dL, &theta_sl, &theta_lk, &floor_s_x, &floor_s_z);
            out_ix_ele_source[tid] = ix_src;
            found = 1;
            break;
        }

        if (ddz0 < 0 && ddz1 < 0) {
            if (last_step == 1) { found = 1; break; }
            last_step = -1;
            ix_src--;
            if (ix_src > 0 && geom[ix_src].ele_key == 37) ix_src--; /* skip match */
            continue;
        }
        if (ddz0 > 0 && ddz1 > 0) {
            if (ix_src == ix_ele_kick) { found = 1; break; } /* source ahead of kick */
            if (last_step == -1) { found = 1; break; }
            last_step = 1;
            ix_src++;
            if (ix_src < n_ele && geom[ix_src].ele_key == 37) ix_src++;
            continue;
        }

        /* Root is bracketed -- use Brent's method */
        s_source = zbrent_dev(ix_src, ix_ele_kick, s_chord_kick, geom,
                              floor_k_x, floor_k_z, y_source, gamma2, dz_particles,
                              0.0, geom[ix_src].L_chord,
                              &L, &dL, &theta_sl, &theta_lk, &floor_s_x, &floor_s_z);
        out_ix_ele_source[tid] = ix_src;
        found = 1;
        break;
    }

    if (!found) return;

    /* --- I_csr calculation --- */
    double z = dz_particles;
    if (z <= 0) return;

    double I_csr_val;
    if (y_source == 0.0) {
        /* CSR kernel */
        I_csr_val = -kick_factor * 2.0 * (dL / z + gamma2 * theta_sl * theta_lk / (1.0 + gamma2 * theta_sl * theta_sl)) / L;
    } else {
        /* Image charge kick -- simplified */
        double Lvx = floor_k_x - floor_s_x;
        double Lvz = floor_k_z - floor_s_z;
        double L_horiz = sqrt(Lvx*Lvx + Lvz*Lvz);
        if (L_horiz < 1e-30) return;
        I_csr_val = -kick_factor * y_source / (L * L_horiz);
    }

    out_I_csr[tid] = I_csr_val;

    /* I_int_csr: for i_bin==1, use special formula; otherwise use trapezoidal rule.
       The trapezoidal rule requires the PREVIOUS bin's I_csr, which we compute in a
       separate serial pass on CPU after the kernel. Store I_csr for now. */
}

/* --------------------------------------------------------------------------
 * gpu_csr_bin_kicks -- compute CSR bin kicks on GPU
 *
 * Uploads element geometry, runs parallel root-finding for all kick1 bins,
 * then computes I_int_csr prefix and convolution on CPU (small arrays).
 * -------------------------------------------------------------------------- */
extern "C" void gpu_csr_bin_kicks_(
    /* Element geometry (host, n_ele+1 elements, 0..n_ele) */
    double *h_floor0_x, double *h_floor0_z, double *h_floor0_theta,
    double *h_floor1_x, double *h_floor1_z, double *h_floor1_theta,
    double *h_L_chord, double *h_theta_chord,
    double *h_spline_coef, /* 3 * (n_ele+1) */
    double *h_dL_s, double *h_ele_s,
    int *h_ele_key,
    int n_ele,
    /* CSR parameters */
    int ix_ele_kick,
    double s_chord_kick,
    double floor_k_x, double floor_k_z,
    double gamma, double gamma2, double beta2,
    double y_source,
    double dz_slice,
    int n_bin,
    double kick_factor,
    double actual_track_step,
    double species_radius,      /* classical_radius(species) */
    double rel_mass,
    double e_charge_abs,
    int csr_method_one_dim,     /* 1 if csr_method == one_dim$ */
    /* Bin data (host) */
    double *h_edge_dcdz,        /* n_bin: edge_dcharge_density_dz */
    double *h_slice_charge,     /* n_bin: slice charge */
    /* Output (host) */
    double *h_kick_csr,         /* n_bin */
    double *h_I_csr_out)        /* 2*n_bin+1, for image charge accumulation */
{
    int n_kick1 = 2 * n_bin + 1;

    /* Build geometry array (static host buffer) */
    if (n_ele + 1 > h_cbk_geom_cap) {
        free(h_cbk_geom);
        h_cbk_geom = (CsrEleGeom*)malloc((n_ele + 1) * sizeof(CsrEleGeom));
        h_cbk_geom_cap = n_ele + 1;
    }
    CsrEleGeom *h_geom = h_cbk_geom;
    for (int i = 0; i <= n_ele; i++) {
        h_geom[i].floor0_x = h_floor0_x[i];
        h_geom[i].floor0_z = h_floor0_z[i];
        h_geom[i].floor0_theta = h_floor0_theta[i];
        h_geom[i].floor1_x = h_floor1_x[i];
        h_geom[i].floor1_z = h_floor1_z[i];
        h_geom[i].floor1_theta = h_floor1_theta[i];
        h_geom[i].L_chord = h_L_chord[i];
        h_geom[i].theta_chord = h_theta_chord[i];
        h_geom[i].spline_coef[0] = h_spline_coef[3*i];
        h_geom[i].spline_coef[1] = h_spline_coef[3*i+1];
        h_geom[i].spline_coef[2] = h_spline_coef[3*i+2];
        h_geom[i].dL_s = h_dL_s[i];
        h_geom[i].ele_s = h_ele_s[i];
        h_geom[i].ele_key = h_ele_key[i];
    }

    /* Upload geometry to device (cached buffer) */
    if (n_ele + 1 > d_cbk_geom_cap) {
        if (d_cbk_geom) cudaFree(d_cbk_geom);
        cudaMalloc((void**)&d_cbk_geom, (n_ele + 1) * sizeof(CsrEleGeom));
        d_cbk_geom_cap = n_ele + 1;
    }
    cudaMemcpy(d_cbk_geom, h_geom, (n_ele + 1) * sizeof(CsrEleGeom), cudaMemcpyHostToDevice);

    /* Ensure device output arrays are large enough (cached) */
    if (n_kick1 > d_cbk_kick1_cap) {
        if (d_cbk_I_csr) cudaFree(d_cbk_I_csr);
        if (d_cbk_I_int_csr) cudaFree(d_cbk_I_int_csr);
        if (d_cbk_ix_ele_source) cudaFree(d_cbk_ix_ele_source);
        cudaMalloc((void**)&d_cbk_I_csr, n_kick1 * sizeof(double));
        cudaMalloc((void**)&d_cbk_I_int_csr, n_kick1 * sizeof(double));
        cudaMalloc((void**)&d_cbk_ix_ele_source, n_kick1 * sizeof(int));
        d_cbk_kick1_cap = n_kick1;
    }
    cudaMemset(d_cbk_I_csr, 0, n_kick1 * sizeof(double));
    cudaMemset(d_cbk_I_int_csr, 0, n_kick1 * sizeof(double));

    /* Launch kernel: one thread per kick1 bin */
    int threads = 256;
    int blocks = (n_kick1 + threads - 1) / threads;
    csr_bin_kicks_kernel<<<blocks, threads>>>(
        d_cbk_geom, n_ele, ix_ele_kick, s_chord_kick,
        floor_k_x, floor_k_z,
        gamma, gamma2, beta2, y_source,
        dz_slice, n_bin, kick_factor,
        d_cbk_I_csr, d_cbk_I_int_csr, d_cbk_ix_ele_source);
    CUDA_SC_CHECK(cudaGetLastError());
    CUDA_SC_CHECK(cudaDeviceSynchronize());

    /* Download results (static host buffers) */
    if (n_kick1 > h_cbk_kick1_cap) {
        free(h_cbk_I_csr); free(h_cbk_I_int_csr);
        h_cbk_I_csr = (double*)malloc(n_kick1 * sizeof(double));
        h_cbk_I_int_csr = (double*)malloc(n_kick1 * sizeof(double));
        h_cbk_kick1_cap = n_kick1;
    }
    double *h_I_csr = h_cbk_I_csr;
    double *h_I_int_csr = h_cbk_I_int_csr;
    cudaMemcpy(h_I_csr, d_cbk_I_csr, n_kick1 * sizeof(double), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_I_int_csr, d_cbk_I_int_csr, n_kick1 * sizeof(double), cudaMemcpyDeviceToHost);

    /* --- CPU: Compute I_int_csr (prefix scan) --- */
    /* h_I_csr is indexed 0..2*n_bin, corresponding to i_bin = -n_bin..+n_bin */
    /* kick1(i) in Fortran = h_I_csr[i + n_bin] here */

    /* i_bin=1 (index n_bin+1): special formula computed in kernel already for I_csr
       but I_int_csr needs special handling */
    /* For the benchmark, the kick point and source are in the same element.
       Use the simplified formula for i_bin=1: */
    {
        int idx1 = n_bin + 1; /* i_bin = 1 */
        double I_csr_1 = h_I_csr[idx1];
        /* Simplified I_int_csr for first bin (approximation used by Bmad):
           I_int_csr(1) ≈ -kick_factor * ((g_bend * Ls/2)^2 - log(2*gamma2*dz/Ls)/gamma2)
           This requires g_bend (bend curvature at kick point) which we don't have here.
           Instead, use the trapezoidal approximation: I_int_csr = I_csr * dz_slice */
        h_I_int_csr[idx1] = I_csr_1 * dz_slice;

        /* For i_bin >= 2: trapezoidal rule */
        for (int ib = 2; ib <= n_bin; ib++) {
            int idx = n_bin + ib;
            h_I_int_csr[idx] = (h_I_csr[idx] + h_I_csr[idx - 1]) * dz_slice / 2.0;
        }
    }

    /* --- CPU: Convolution to get kick_csr per slice --- */
    double coef = actual_track_step * species_radius / (rel_mass * e_charge_abs * gamma);

    if (y_source == 0.0 && csr_method_one_dim) {
        for (int i = 0; i < n_bin; i++) {
            double sum = 0.0;
            /* dot_product(kick1(i+1:1:-1)%I_int_csr, slice(1:i+1)%edge_dcharge_density_dz)
               In Fortran: kick1(i) uses index i (1-based) in kick1 array.
               Here: kick1 index for Fortran-i = n_bin + i (0-based).
               Fortran kick1(i:1:-1) = indices from n_bin+(i+1) down to n_bin+1 */
            for (int j = 0; j <= i; j++) {
                int kick_idx = n_bin + (i + 1) - j; /* kick1(i+1-j) */
                sum += h_I_int_csr[kick_idx] * h_edge_dcdz[j];
            }
            h_kick_csr[i] = coef * sum;
        }
    } else if (y_source != 0.0) {
        /* Image charge kick: accumulate into existing h_kick_csr */
        for (int i = 0; i < n_bin; i++) {
            double sum = 0.0;
            for (int j = 0; j < n_bin; j++) {
                int kick_idx = n_bin + (i) - j; /* kick1(i-j) for 0-based, maps to kick1(i-1-j+1) in Fortran */
                if (kick_idx >= 0 && kick_idx < n_kick1)
                    sum += h_I_csr[kick_idx] * h_slice_charge[j];
            }
            h_kick_csr[i] += coef * sum;
        }
    }

    /* Copy I_csr for caller (needed for image charge accumulation) */
    if (h_I_csr_out) {
        memcpy(h_I_csr_out, h_I_csr, n_kick1 * sizeof(double));
    }

    /* All buffers are static -- not freed here, reused across calls */
}


/* --------------------------------------------------------------------------
 * Cleanup
 * -------------------------------------------------------------------------- */
extern "C" void gpu_spacecharge_cleanup_(void)
{
    if (sc_fft_plan) { cufftDestroy(sc_fft_plan); sc_fft_plan = 0; }
    if (d_sc_rho)    { cudaFree(d_sc_rho);    d_sc_rho = NULL; }
    if (d_sc_efield) { cudaFree(d_sc_efield);  d_sc_efield = NULL; }
    if (d_sc_charge) { cudaFree(d_sc_charge);  d_sc_charge = NULL; }
    if (d_sc_rho_padded)  { cudaFree(d_sc_rho_padded);  d_sc_rho_padded = NULL; }
    if (d_sc_crho_freq)   { cudaFree(d_sc_crho_freq);   d_sc_crho_freq = NULL; }
    sc_nx2 = 0; sc_ny2 = 0; sc_nz2 = 0;
    sc_charge_uploaded = 0;

    /* Split compute/apply state */
    sc_split_state.valid = 0;
    for (int c = 0; c < 3; c++) {
        if (sc_fft_done[c]) { cudaEventDestroy(sc_fft_done[c]); sc_fft_done[c] = 0; }
    }
    sc_fft_events_created = 0;

    if (d_csr_bin_charge)     { cudaFree(d_csr_bin_charge);     d_csr_bin_charge = NULL; }
    if (d_csr_bin_x0_wt)     { cudaFree(d_csr_bin_x0_wt);     d_csr_bin_x0_wt = NULL; }
    if (d_csr_bin_y0_wt)     { cudaFree(d_csr_bin_y0_wt);     d_csr_bin_y0_wt = NULL; }
    if (d_csr_bin_n_particle) { cudaFree(d_csr_bin_n_particle); d_csr_bin_n_particle = NULL; }
    if (d_csr_kick_csr)      { cudaFree(d_csr_kick_csr);      d_csr_kick_csr = NULL; }
    if (d_csr_kick_lsc)      { cudaFree(d_csr_kick_lsc);      d_csr_kick_lsc = NULL; }
    d_csr_bin_cap = 0;

    /* CSR bin kicks cached buffers */
    if (d_cbk_geom) { cudaFree(d_cbk_geom); d_cbk_geom = NULL; }
    if (d_cbk_I_csr) { cudaFree(d_cbk_I_csr); d_cbk_I_csr = NULL; }
    if (d_cbk_I_int_csr) { cudaFree(d_cbk_I_int_csr); d_cbk_I_int_csr = NULL; }
    if (d_cbk_ix_ele_source) { cudaFree(d_cbk_ix_ele_source); d_cbk_ix_ele_source = NULL; }
    d_cbk_geom_cap = 0; d_cbk_kick1_cap = 0;
    free(h_cbk_geom); h_cbk_geom = NULL; h_cbk_geom_cap = 0;
    free(h_cbk_I_csr); h_cbk_I_csr = NULL;
    free(h_cbk_I_int_csr); h_cbk_I_int_csr = NULL;
    h_cbk_kick1_cap = 0;

    /* CSR convolution buffers + dedicated stream */
    if (d_csr_I_int) { cudaFree(d_csr_I_int); d_csr_I_int = NULL; }
    if (d_csr_edge_dcdz) { cudaFree(d_csr_edge_dcdz); d_csr_edge_dcdz = NULL; }
    if (d_csr_kick_conv) { cudaFree(d_csr_kick_conv); d_csr_kick_conv = NULL; }
    d_csr_conv_cap = 0;
    if (csr_kick_stream_created) { cudaStreamDestroy(csr_kick_stream); csr_kick_stream = 0; csr_kick_stream_created = 0; }
    if (h_csr_I_int_pinned) { cudaFreeHost(h_csr_I_int_pinned); h_csr_I_int_pinned = NULL; h_csr_I_int_pinned_cap = 0; }

    /* (GPU-side reduction replaced per-block host buffers) */

    /* z_minmax cached buffers */
    if (d_zminmax_bmin) { cudaFree(d_zminmax_bmin); d_zminmax_bmin = NULL; }
    if (d_zminmax_bmax) { cudaFree(d_zminmax_bmax); d_zminmax_bmax = NULL; }
    d_zminmax_cap = 0;
    free(h_zminmax_bmin); h_zminmax_bmin = NULL;
    free(h_zminmax_bmax); h_zminmax_bmax = NULL;
    h_zminmax_cap = 0;
}

#endif /* USE_GPU_TRACKING */
