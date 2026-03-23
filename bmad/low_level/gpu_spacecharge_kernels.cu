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
 * Float deposit: deposit charge directly into the zero-padded float buffer
 * used by R2C FFT, using float atomicAdd (hardware-native, ~5x faster than
 * double atomicAdd).  Eliminates the intermediate d_sc_rho double buffer
 * and the rho_to_padded_f conversion kernel.
 * Particle positions [0,nx)×[0,ny)×[0,nz) map to the first octant of the
 * padded [0,nx2)×[0,ny2)×[0,nz2) buffer; padding positions stay zero.
 * -------------------------------------------------------------------------- */
__global__ void deposit_kernel_f(
    const double *x, const double *y, const double *z,
    const double *beta,
    const double *charge, const int *state,
    float *rho_padded,
    double xmin, double ymin, double zmin,
    double dxi, double dyi, double dzi,
    double dx, double dy, double dz,
    int nx, int ny, int nz,
    int ny2, int nz2,
    double dct_ave,
    int n_particles)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n_particles) return;
    if (state[i] != SC_ALIVE_ST) return;

    double z_adj = z[i] - dct_ave * beta[i];

    int ip = (int)floor((x[i] - xmin) * dxi + 1.0) - 1;
    int jp = (int)floor((y[i] - ymin) * dyi + 1.0) - 1;
    int kp = (int)floor((z_adj - zmin) * dzi + 1.0) - 1;

    if (ip < 0) ip = 0; if (ip >= nx-1) ip = nx-2;
    if (jp < 0) jp = 0; if (jp >= ny-1) jp = ny-2;
    if (kp < 0) kp = 0; if (kp >= nz-1) kp = nz-2;

    double ab = ((xmin - x[i]) + (ip+1)*dx) * dxi;
    double de = ((ymin - y[i]) + (jp+1)*dy) * dyi;
    double gh = ((zmin - z_adj) + (kp+1)*dz) * dzi;

    double q = charge[i];

    #define PAD_IDX(ii,jj,kk) ((ii)*ny2*nz2 + (jj)*nz2 + (kk))
    atomicAdd(&rho_padded[PAD_IDX(ip,  jp,  kp  )], (float)(ab    *de    *gh    *q));
    atomicAdd(&rho_padded[PAD_IDX(ip,  jp+1,kp  )], (float)(ab    *(1-de)*gh    *q));
    atomicAdd(&rho_padded[PAD_IDX(ip,  jp+1,kp+1)], (float)(ab    *(1-de)*(1-gh)*q));
    atomicAdd(&rho_padded[PAD_IDX(ip,  jp,  kp+1)], (float)(ab    *de    *(1-gh)*q));
    atomicAdd(&rho_padded[PAD_IDX(ip+1,jp,  kp+1)], (float)((1-ab)*de    *(1-gh)*q));
    atomicAdd(&rho_padded[PAD_IDX(ip+1,jp+1,kp+1)], (float)((1-ab)*(1-de)*(1-gh)*q));
    atomicAdd(&rho_padded[PAD_IDX(ip+1,jp+1,kp  )], (float)((1-ab)*(1-de)*gh    *q));
    atomicAdd(&rho_padded[PAD_IDX(ip+1,jp,  kp  )], (float)((1-ab)*de    *gh    *q));
    #undef PAD_IDX
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
 * Mixed-precision FFT helper kernels.
 * The 3D SC FFTs are bandwidth-limited; using float halves data movement.
 * Deposit uses float atomicAdd directly into the padded R2C buffer.
 * Kick interpolation stays in double for accuracy.
 * -------------------------------------------------------------------------- */

/* Convert double complex → float complex (for Green fn cache conversion). */
__global__ void dc_to_fc_kernel(
    const cufftDoubleComplex *in, cufftComplex *out, int n)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) { out[i].x = (float)in[i].x; out[i].y = (float)in[i].y; }
}

/* Float complex multiply: out = a * b (3-operand). */
__global__ void complex_multiply_src_kernel_f(
    const cufftComplex *a, const cufftComplex *b, cufftComplex *out, int n)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    float ar = a[i].x, ai = a[i].y;
    float br = b[i].x, bi = b[i].y;
    out[i].x = ar*br - ai*bi;
    out[i].y = ar*bi + ai*br;
}

/* --------------------------------------------------------------------------
 * Interpolate E-field and apply kicks to particles (space charge).
 * Reads directly from padded C2R float output buffers, eliminating the
 * intermediate extract_field step and d_sc_efield buffer.
 * -------------------------------------------------------------------------- */
__global__ void sc_interpolate_kick_kernel(
    double *vx, double *vpx, double *vy, double *vpy,
    double *vz, double *vpz,
    int *state, double *beta_arr, double *p0c_arr,
    const float *field_x, const float *field_y, const float *field_z,
    double xmin, double ymin, double zmin,
    double dxi, double dyi, double dzi,
    double dx, double dy, double dz,
    int nx, int ny, int nz,
    int nx2, int ny2, int nz2,
    double scale,
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

    /* Trilinear interpolation from padded C2R output (float → double with scale).
     * Padded index: (ii + nx-1)*ny2*nz2 + (jj + ny-1)*nz2 + (kk + nz-1) */
    int xshift = nx - 1, yshift = ny - 1, zshift = nz - 1;
    const float *fields[3] = {field_x, field_y, field_z};
    double Evec[3];

    for (int comp = 0; comp < 3; comp++) {
        const float *ef = fields[comp];
        #define EF(ii,jj,kk) ((double)ef[((ii)+xshift)*ny2*nz2 + ((jj)+yshift)*nz2 + ((kk)+zshift)] * scale)
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

/* Cached device arrays for space charge */
static double *d_sc_charge = NULL;  /* per-particle charge */
static float *d_sc_rho_padded_f = NULL;         /* float padded rho for R2C */
static cufftComplex *d_sc_crho_freq_f = NULL;   /* R2C output (float complex) */
static cufftHandle sc_fft_plan_f = 0;           /* R2C plan for float forward FFT */
static int sc_nx2 = 0, sc_ny2 = 0, sc_nz2 = 0;
static float *d_sc_real_c_f[3] = {NULL, NULL, NULL};  /* C2R output per component (read by interp kernel) */

/* Cached device arrays for CSR binning */
static double *d_csr_bin_charge = NULL;
static double *d_csr_bin_x0_wt = NULL;
static double *d_csr_bin_y0_wt = NULL;
static double *d_csr_bin_n_particle = NULL;
static double *d_csr_kick_csr = NULL;
static double *d_csr_kick_lsc = NULL;
static int d_csr_bin_cap = 0;

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
    int dbl_size = nx2 * ny2 * nz2;
    int half_size = nx2 * ny2 * (nz2/2 + 1);  /* R2C complex output size */

    /* Recreate FFT plans if mesh size changed */
    if (nx2 != sc_nx2 || ny2 != sc_ny2 || nz2 != sc_nz2) {
        /* R2C float plan for charge density forward FFT */
        if (sc_fft_plan_f) cufftDestroy(sc_fft_plan_f);
        if (cufftPlan3d(&sc_fft_plan_f, nx2, ny2, nz2, CUFFT_R2C) != CUFFT_SUCCESS) {
            fprintf(stderr, "[gpu_sc] cuFFT R2C plan creation failed\n");
            sc_fft_plan_f = 0;
            return -1;
        }
        sc_nx2 = nx2; sc_ny2 = ny2; sc_nz2 = nz2;

        /* Float padded rho (deposit target + R2C input) + float freq output */
        if (d_sc_rho_padded_f) cudaFree(d_sc_rho_padded_f);
        if (d_sc_crho_freq_f)  cudaFree(d_sc_crho_freq_f);
        cudaMalloc((void**)&d_sc_rho_padded_f, (size_t)dbl_size * sizeof(float));
        cudaMalloc((void**)&d_sc_crho_freq_f,  (size_t)half_size * sizeof(cufftComplex));
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
    double ds_step, gamma, mc2, dct_ave, scale;
    int nx, ny, nz, nx2, ny2, nz2, n_particles;
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

    /* --- Step 2+3: Deposit directly into float padded buffer, then R2C FFT ---
     * Float atomicAdd is hardware-native (~5x faster than double).
     * Eliminates the intermediate double rho buffer and rho_to_padded conversion. */
    int blocks_p = (n_particles + threads - 1) / threads;
    int blocks_d = (dbl_size + threads - 1) / threads;
    int blocks_h = (half_size + threads - 1) / threads;

    CUDA_SC_CHECK(cudaMemset(d_sc_rho_padded_f, 0, (size_t)dbl_size * sizeof(float)));

    deposit_kernel_f<<<blocks_p, threads>>>(
        dvec[0], dvec[2], dvec[4], dbeta,
        d_sc_charge, dstate,
        d_sc_rho_padded_f,
        xmin, ymin, zmin, dxi, dyi, dzi, dx, dy, dz,
        nx, ny, nz, ny2, nz2, dct_ave, n_particles);
    CUDA_SC_CHECK(cudaGetLastError());

    CUFFT_SC_CHECK(cufftExecR2C(sc_fft_plan_f, d_sc_rho_padded_f, d_sc_crho_freq_f));

    /* Record event on default stream so component streams can wait for charge FFT */
    static cudaEvent_t sc_crho_ready = 0;
    if (!sc_crho_ready) cudaEventCreate(&sc_crho_ready);
    cudaEventRecord(sc_crho_ready, 0);  /* 0 = default stream */

    /* --- Step 4: For each E-field component: Green function + FFT + multiply + IFFT ---
     * Mixed precision: Green fn cache + multiply + inverse FFT all use float.
     * Green fn forward FFT uses double D2Z (cache miss only), then converts to float cache.
     * Charge freq is float from R2C above. Inverse C2R produces float in padded layout;
     * the interpolation kernel reads directly from the padded output (no extract step). */
    int stencil_size = (nx2-1)*(ny2-1)*(nz2-1);
    int blocks_s = (stencil_size + threads - 1) / threads;
    double scale = SC_FPEI / (double)(nx2*ny2*nz2);

    /* Per-component workspaces: double (for Green fn D2Z on cache miss) + float (hot path).
     * d_sc_cgrn_c: real scratch for Green fn (double, dbl_size).
     * d_sc_cgrn2_c: freq workspace for Green fn D2Z (double complex, half_size).
     * d_sc_real_c: real workspace for Green fn stencil+D2Z (double, dbl_size).
     * d_sc_grn_cache_f: cached FFT'd Green fn in float (float complex, half_size).
     * d_sc_cgrn2_c_f: multiply output + C2R input (float complex, half_size).
     * d_sc_real_c_f: C2R output (float, dbl_size). */
    static double *d_sc_cgrn_c[3] = {NULL, NULL, NULL};
    static cufftDoubleComplex *d_sc_cgrn2_c[3] = {NULL, NULL, NULL};
    static double *d_sc_real_c[3] = {NULL, NULL, NULL};
    static cufftComplex *d_sc_grn_cache_f[3] = {NULL, NULL, NULL};
    static cufftComplex *d_sc_cgrn2_c_f[3] = {NULL, NULL, NULL};
    static int grn_cached_size = 0;
    if (dbl_size > grn_cached_size) {
        size_t rsz = (size_t)dbl_size * sizeof(double);
        size_t hsz_d = (size_t)half_size * sizeof(cufftDoubleComplex);
        size_t hsz_f = (size_t)half_size * sizeof(cufftComplex);
        size_t rsz_f = (size_t)dbl_size * sizeof(float);
        for (int c = 0; c < 3; c++) {
            if (d_sc_cgrn_c[c]) cudaFree(d_sc_cgrn_c[c]);
            if (d_sc_cgrn2_c[c]) cudaFree(d_sc_cgrn2_c[c]);
            if (d_sc_real_c[c]) cudaFree(d_sc_real_c[c]);
            if (d_sc_grn_cache_f[c]) cudaFree(d_sc_grn_cache_f[c]);
            if (d_sc_cgrn2_c_f[c]) cudaFree(d_sc_cgrn2_c_f[c]);
            if (d_sc_real_c_f[c]) cudaFree(d_sc_real_c_f[c]);
            cudaMalloc((void**)&d_sc_cgrn_c[c], rsz);       /* Green fn scratch (double) */
            cudaMalloc((void**)&d_sc_cgrn2_c[c], hsz_d);    /* Green fn D2Z output (double) */
            cudaMalloc((void**)&d_sc_real_c[c], rsz);        /* Green fn stencil+D2Z (double) */
            cudaMalloc((void**)&d_sc_grn_cache_f[c], hsz_f); /* cached Green fn (float) */
            cudaMalloc((void**)&d_sc_cgrn2_c_f[c], hsz_f);  /* multiply output (float) */
            cudaMalloc((void**)&d_sc_real_c_f[c], rsz_f);    /* C2R output (float) */
        }
        grn_cached_size = dbl_size;
    }

    /* CUDA streams for concurrent E-field computation */
    static cudaStream_t sc_streams[3] = {0, 0, 0};
    static cufftHandle sc_fwd_plans[3] = {0, 0, 0};  /* D2Z forward plans for Green fn */
    static cufftHandle sc_inv_plans_f[3] = {0, 0, 0}; /* C2R float inverse plans */
    static int sc_comp_nx2 = 0, sc_comp_ny2 = 0, sc_comp_nz2 = 0;
    static int streams_created = 0;
    if (!streams_created) {
        for (int c = 0; c < 3; c++) cudaStreamCreate(&sc_streams[c]);
        streams_created = 1;
    }
    if (nx2 != sc_comp_nx2 || ny2 != sc_comp_ny2 || nz2 != sc_comp_nz2) {
        for (int c = 0; c < 3; c++) {
            if (sc_fwd_plans[c]) cufftDestroy(sc_fwd_plans[c]);
            if (sc_inv_plans_f[c]) cufftDestroy(sc_inv_plans_f[c]);
            cufftPlan3d(&sc_fwd_plans[c], nx2, ny2, nz2, CUFFT_D2Z);
            cufftPlan3d(&sc_inv_plans_f[c], nx2, ny2, nz2, CUFFT_C2R);
            cufftSetStream(sc_fwd_plans[c], sc_streams[c]);
            cufftSetStream(sc_inv_plans_f[c], sc_streams[c]);
        }
        sc_comp_nx2 = nx2; sc_comp_ny2 = ny2; sc_comp_nz2 = nz2;
    }

    /* Green function caching: depends only on dx, dy, dz*gamma, mesh dims.
     * Default 15% tolerance -- the Green function varies smoothly with grid
     * spacing, so reusing a cached Green function when the grid has drifted
     * by up to ~15% introduces only a small error in the SC kicks (the SC
     * force is itself a small perturbation on the tracking).
     * Override: set SC_GRN_TOL to a fractional value (e.g. 0.001 for 0.1%). */
    static double grn_cache_dx = 0, grn_cache_dy = 0, grn_cache_dz_rf = 0;
    static int grn_cache_nx2 = 0, grn_cache_ny2 = 0, grn_cache_nz2 = 0;
    static int grn_cache_valid = 0;
    static double grn_tol = -1;
    if (grn_tol < 0) {
        grn_tol = 0.15;
        const char *env = getenv("SC_GRN_TOL");
        if (env) { double v = atof(env); if (v > 0) grn_tol = v; }
    }

    double dz_rf = dz_rf_local;  /* already computed on GPU */
    int grn_hit = 0;
    if (grn_cache_valid &&
        grn_cache_nx2 == nx2 && grn_cache_ny2 == ny2 && grn_cache_nz2 == nz2) {
        double rdx = fabs(dx - grn_cache_dx) / (fabs(grn_cache_dx) + 1e-30);
        double rdy = fabs(dy - grn_cache_dy) / (fabs(grn_cache_dy) + 1e-30);
        double rdz = fabs(dz_rf - grn_cache_dz_rf) / (fabs(grn_cache_dz_rf) + 1e-30);
        grn_hit = (rdx < grn_tol && rdy < grn_tol && rdz < grn_tol);
    }

    if (!grn_hit) {
        /* Cache miss: Green fn D2Z in double, convert cache to float, then float pipeline */
        for (int icomp = 1; icomp <= 3; icomp++) {
            int c = icomp - 1;

            /* Compute Green function into real scratch buffer (double) */
            green_function_kernel<<<blocks_d, threads, 0, sc_streams[c]>>>(
                d_sc_cgrn_c[c], dx, dy, dz, gamma, icomp,
                nx2, ny2, nz2, 0.0, 0.0, 0.0);

            /* Stencil: cgrn_c -> real_c (zero first) */
            cudaMemsetAsync(d_sc_real_c[c], 0, (size_t)dbl_size * sizeof(double), sc_streams[c]);

            igf_stencil_kernel<<<blocks_s, threads, 0, sc_streams[c]>>>(
                d_sc_cgrn_c[c], d_sc_real_c[c], nx2, ny2, nz2);

            /* Forward D2Z FFT of Green function (double) */
            CUFFT_SC_CHECK(cufftExecD2Z(sc_fwd_plans[c], d_sc_real_c[c], d_sc_cgrn2_c[c]));

            /* Convert to float cache */
            dc_to_fc_kernel<<<blocks_h, threads, 0, sc_streams[c]>>>(
                d_sc_cgrn2_c[c], d_sc_grn_cache_f[c], half_size);

            /* Float multiply: crho_freq_f * grn_cache_f -> cgrn2_c_f */
            cudaStreamWaitEvent(sc_streams[c], sc_crho_ready, 0);
            complex_multiply_src_kernel_f<<<blocks_h, threads, 0, sc_streams[c]>>>(
                d_sc_crho_freq_f, d_sc_grn_cache_f[c], d_sc_cgrn2_c_f[c], half_size);

            /* Inverse C2R FFT (float) — interpolation reads directly from padded output */
            CUFFT_SC_CHECK(cufftExecC2R(sc_inv_plans_f[c], d_sc_cgrn2_c_f[c], d_sc_real_c_f[c]));
        }
        grn_cache_dx = dx; grn_cache_dy = dy; grn_cache_dz_rf = dz_rf;
        grn_cache_nx2 = nx2; grn_cache_ny2 = ny2; grn_cache_nz2 = nz2;
        grn_cache_valid = 1;
    } else {
        /* Cache hit: float multiply + C2R (hot path).
         * Interpolation reads directly from padded C2R output. */
        for (int icomp = 1; icomp <= 3; icomp++) {
            int c = icomp - 1;
            cudaStreamWaitEvent(sc_streams[c], sc_crho_ready, 0);
            complex_multiply_src_kernel_f<<<blocks_h, threads, 0, sc_streams[c]>>>(
                d_sc_crho_freq_f, d_sc_grn_cache_f[c], d_sc_cgrn2_c_f[c], half_size);
            CUFFT_SC_CHECK(cufftExecC2R(sc_inv_plans_f[c], d_sc_cgrn2_c_f[c], d_sc_real_c_f[c]));
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
        sc_split_state.mc2 = mc2; sc_split_state.dct_ave = dct_ave; sc_split_state.scale = scale;
        sc_split_state.nx = nx; sc_split_state.ny = ny; sc_split_state.nz = nz;
        sc_split_state.nx2 = nx2; sc_split_state.ny2 = ny2; sc_split_state.nz2 = nz2;
        sc_split_state.n_particles = n_particles;
        sc_split_state.valid = 1;
        sc_compute_only_flag = 0;
        return;
    }

    /* Normal mode: wait for FFTs and apply kicks immediately */
    for (int c = 0; c < 3; c++) {
        cudaStreamWaitEvent(0, sc_fft_done[c], 0);  /* default stream waits */
    }

    /* --- Step 5: Interpolate directly from padded C2R output and apply kicks --- */
    sc_interpolate_kick_kernel<<<blocks_p, threads>>>(
        dvec[0], dvec[1], dvec[2], dvec[3],
        dvec[4], dvec[5],
        dstate, dbeta, dp0c,
        d_sc_real_c_f[0], d_sc_real_c_f[1], d_sc_real_c_f[2],
        xmin, ymin, zmin, dxi, dyi, dzi, dx, dy, dz,
        nx, ny, nz, nx2, ny2, nz2, scale,
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

    /* Interpolate directly from padded C2R output and apply kicks */
    sc_interpolate_kick_kernel<<<blocks_p, threads>>>(
        dvec[0], dvec[1], dvec[2], dvec[3],
        dvec[4], dvec[5],
        dstate, dbeta, dp0c,
        d_sc_real_c_f[0], d_sc_real_c_f[1], d_sc_real_c_f[2],
        sc_split_state.xmin, sc_split_state.ymin, sc_split_state.zmin,
        sc_split_state.dxi, sc_split_state.dyi, sc_split_state.dzi,
        sc_split_state.dx, sc_split_state.dy, sc_split_state.dz,
        sc_split_state.nx, sc_split_state.ny, sc_split_state.nz,
        sc_split_state.nx2, sc_split_state.ny2, sc_split_state.nz2,
        sc_split_state.scale,
        sc_split_state.ds_step, sc_split_state.gamma,
        sc_split_state.mc2, sc_split_state.dct_ave,
        n_particles);
    CUDA_SC_CHECK(cudaGetLastError());

    sc_split_state.valid = 0;
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


/* --------------------------------------------------------------------------
 * Cleanup
 * -------------------------------------------------------------------------- */
extern "C" void gpu_spacecharge_cleanup_(void)
{
    if (d_sc_charge) { cudaFree(d_sc_charge);  d_sc_charge = NULL; }
    if (d_sc_rho_padded_f) { cudaFree(d_sc_rho_padded_f); d_sc_rho_padded_f = NULL; }
    if (d_sc_crho_freq_f)  { cudaFree(d_sc_crho_freq_f);  d_sc_crho_freq_f = NULL; }
    for (int c = 0; c < 3; c++) {
        if (d_sc_real_c_f[c]) { cudaFree(d_sc_real_c_f[c]); d_sc_real_c_f[c] = NULL; }
    }
    if (sc_fft_plan_f) { cufftDestroy(sc_fft_plan_f); sc_fft_plan_f = 0; }
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
