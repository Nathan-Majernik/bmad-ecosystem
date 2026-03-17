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

/* =========================================================================
 * 3D FFT SPACE CHARGE — CUDA KERNELS
 * ========================================================================= */

/* --------------------------------------------------------------------------
 * Deposit particles on 3D mesh using trilinear interpolation with atomicAdd.
 * Each particle contributes charge to 8 surrounding grid points.
 * -------------------------------------------------------------------------- */
__global__ void deposit_kernel(
    const double *x, const double *y, const double *z,
    const double *charge, const int *state,
    double *rho,
    double xmin, double ymin, double zmin,
    double dxi, double dyi, double dzi,
    double dx, double dy, double dz,
    int nx, int ny, int nz,
    int n_particles)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n_particles) return;
    if (state[i] != SC_ALIVE_ST) return;

    int ip = (int)floor((x[i] - xmin) * dxi + 1.0) - 1;  /* 0-based */
    int jp = (int)floor((y[i] - ymin) * dyi + 1.0) - 1;
    int kp = (int)floor((z[i] - zmin) * dzi + 1.0) - 1;

    /* Clamp to valid range */
    if (ip < 0) ip = 0; if (ip >= nx-1) ip = nx-2;
    if (jp < 0) jp = 0; if (jp >= ny-1) jp = ny-2;
    if (kp < 0) kp = 0; if (kp >= nz-1) kp = nz-2;

    double ab = ((xmin - x[i]) + (ip+1)*dx) * dxi;
    double de = ((ymin - y[i]) + (jp+1)*dy) * dyi;
    double gh = ((zmin - z[i]) + (kp+1)*dz) * dzi;

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
    cufftDoubleComplex *cgrn,
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

    cgrn[idx].x = gval;
    cgrn[idx].y = 0.0;
}

/* --------------------------------------------------------------------------
 * Evaluate the integrated Green function via cube differences (8-point stencil).
 * Operates in-place on the cgrn array.
 * -------------------------------------------------------------------------- */
__global__ void igf_stencil_kernel(
    const cufftDoubleComplex *cgrn_in,
    cufftDoubleComplex *cgrn_out,
    int nx2, int ny2, int nz2)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = (nx2-1) * (ny2-1) * (nz2-1);
    if (idx >= total) return;

    int k = idx % (nz2-1);
    int j = (idx / (nz2-1)) % (ny2-1);
    int i = idx / ((nz2-1) * (ny2-1));

    #define G(ii,jj,kk) cgrn_in[(ii)*ny2*nz2 + (jj)*nz2 + (kk)]
    cufftDoubleComplex val;
    val.x = G(i+1,j+1,k+1).x - G(i,j+1,k+1).x - G(i+1,j,k+1).x - G(i+1,j+1,k).x
           - G(i,j,k).x + G(i,j,k+1).x + G(i,j+1,k).x + G(i+1,j,k).x;
    val.y = 0.0;
    #undef G

    cgrn_out[i*ny2*nz2 + j*nz2 + k] = val;
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

/* --------------------------------------------------------------------------
 * Place rho (real) into one octant of doubled complex array.
 * -------------------------------------------------------------------------- */
__global__ void rho_to_complex_kernel(
    const double *rho,
    cufftDoubleComplex *crho,
    int nx, int ny, int nz,
    int nx2, int ny2, int nz2)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= nx*ny*nz) return;

    int k = idx % nz;
    int j = (idx / nz) % ny;
    int i = idx / (nz * ny);

    int idx2 = i*ny2*nz2 + j*nz2 + k;
    crho[idx2].x = rho[idx];
    crho[idx2].y = 0.0;
}

/* --------------------------------------------------------------------------
 * Extract real field from inverse FFT result (with shift).
 * -------------------------------------------------------------------------- */
__global__ void extract_field_kernel(
    const cufftDoubleComplex *cgrn,
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
    field_comp[idx] = scale * cgrn[idx2].x;
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

    /* Longitudinal kick (energy conserving) */
    double ef = Evec[2] * factor;
    double dpz = sqrt(ef*ef + 2.0*ef*pz0 + rel_p*rel_p - px*px - py*py) - rel_p;
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
 * CSR particle binning kernel — bins particles longitudinally.
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
 * CSR kick application kernel — applies precomputed CSR and LSC kicks.
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
static double *d_sc_rho = NULL;
static double *d_sc_efield = NULL;  /* nx*ny*nz*3 */
static double *d_sc_charge = NULL;  /* per-particle charge */
static cufftDoubleComplex *d_sc_crho = NULL;
static cufftDoubleComplex *d_sc_cgrn = NULL;
static cufftDoubleComplex *d_sc_cgrn2 = NULL;
static cufftHandle sc_fft_plan = 0;
static int sc_nx2 = 0, sc_ny2 = 0, sc_nz2 = 0;

/* Cached device arrays for CSR binning */
static double *d_csr_bin_charge = NULL;
static double *d_csr_bin_x0_wt = NULL;
static double *d_csr_bin_y0_wt = NULL;
static double *d_csr_bin_n_particle = NULL;
static double *d_csr_kick_csr = NULL;
static double *d_csr_kick_lsc = NULL;
static int d_csr_bin_cap = 0;

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

    /* Recreate FFT plan if mesh size changed */
    if (nx2 != sc_nx2 || ny2 != sc_ny2 || nz2 != sc_nz2) {
        if (sc_fft_plan) cufftDestroy(sc_fft_plan);
        if (cufftPlan3d(&sc_fft_plan, nx2, ny2, nz2, CUFFT_Z2Z) != CUFFT_SUCCESS) {
            fprintf(stderr, "[gpu_sc] cuFFT plan creation failed\n");
            sc_fft_plan = 0;
            return -1;
        }
        sc_nx2 = nx2; sc_ny2 = ny2; sc_nz2 = nz2;

        /* Reallocate complex arrays */
        if (d_sc_crho)  cudaFree(d_sc_crho);
        if (d_sc_cgrn)  cudaFree(d_sc_cgrn);
        if (d_sc_cgrn2) cudaFree(d_sc_cgrn2);
        size_t csz = (size_t)dbl_size * sizeof(cufftDoubleComplex);
        cudaMalloc((void**)&d_sc_crho,  csz);
        cudaMalloc((void**)&d_sc_cgrn,  csz);
        cudaMalloc((void**)&d_sc_cgrn2, csz);
    }

    /* Reallocate mesh arrays if needed */
    if (d_sc_rho) cudaFree(d_sc_rho);
    if (d_sc_efield) cudaFree(d_sc_efield);
    cudaMalloc((void**)&d_sc_rho, (size_t)mesh_size * sizeof(double));
    cudaMalloc((void**)&d_sc_efield, (size_t)mesh_size * 3 * sizeof(double));

    /* Per-particle charge array */
    if (d_sc_charge) cudaFree(d_sc_charge);
    cudaMalloc((void**)&d_sc_charge, (size_t)n_particles * sizeof(double));

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
 * gpu_space_charge_3d — full 3D FFT space charge on GPU
 *
 * Particle data must already be on device (d_vec, dstate, dbeta, dp0c).
 * Modifies vpx, vpy, vpz, vz, beta on device.
 *
 * h_charge: per-particle charge array (host)
 * -------------------------------------------------------------------------- */
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
    int threads = 256;

    /* Upload per-particle charge */
    CUDA_SC_CHECK(cudaMemcpy(d_sc_charge, h_charge,
        (size_t)n_particles * sizeof(double), cudaMemcpyHostToDevice));

    /* --- Step 1: Compute mesh bounds from particle positions --- */
    /* We need min/max of x, y, z-dct_ave*beta on the device.
     * For simplicity, download coordinates, compute bounds on CPU,
     * then do the deposition on GPU. The bounds computation is O(n)
     * but minimal compared to the FFT. */
    double *h_x = (double*)malloc(n_particles * sizeof(double));
    double *h_y = (double*)malloc(n_particles * sizeof(double));
    double *h_z = (double*)malloc(n_particles * sizeof(double));
    int *h_state = (int*)malloc(n_particles * sizeof(int));
    double *h_beta = (double*)malloc(n_particles * sizeof(double));

    CUDA_SC_CHECK(cudaMemcpy(h_x, dvec[0], n_particles*sizeof(double), cudaMemcpyDeviceToHost));
    CUDA_SC_CHECK(cudaMemcpy(h_y, dvec[2], n_particles*sizeof(double), cudaMemcpyDeviceToHost));
    CUDA_SC_CHECK(cudaMemcpy(h_z, dvec[4], n_particles*sizeof(double), cudaMemcpyDeviceToHost));
    CUDA_SC_CHECK(cudaMemcpy(h_state, dstate, n_particles*sizeof(int), cudaMemcpyDeviceToHost));
    CUDA_SC_CHECK(cudaMemcpy(h_beta, dbeta, n_particles*sizeof(double), cudaMemcpyDeviceToHost));

    double xmin=1e30, xmax=-1e30, ymin=1e30, ymax=-1e30, zmin=1e30, zmax=-1e30;
    for (int i = 0; i < n_particles; i++) {
        if (h_state[i] != SC_ALIVE_ST) continue;
        double zz = h_z[i] - dct_ave * h_beta[i];
        if (h_x[i] < xmin) xmin = h_x[i]; if (h_x[i] > xmax) xmax = h_x[i];
        if (h_y[i] < ymin) ymin = h_y[i]; if (h_y[i] > ymax) ymax = h_y[i];
        if (zz < zmin) zmin = zz;          if (zz > zmax) zmax = zz;
    }

    double dx = (xmax-xmin)/(nx-1), dy = (ymax-ymin)/(ny-1), dz = (zmax-zmin)/(nz-1);
    if (dx == 0) dx = 1e-10; if (dy == 0) dy = 1e-10; if (dz == 0) dz = 1e-10;
    /* Small padding */
    xmin -= 1e-6*dx; xmax += 1e-6*dx;
    ymin -= 1e-6*dy; ymax += 1e-6*dy;
    zmin -= 1e-6*dz; zmax += 1e-6*dz;
    dx = (xmax-xmin)/(nx-1); dy = (ymax-ymin)/(ny-1); dz = (zmax-zmin)/(nz-1);

    free(h_x); free(h_y); free(h_z); free(h_state); free(h_beta);

    double dxi = 1.0/dx, dyi = 1.0/dy, dzi = 1.0/dz;

    /* --- Step 2: Deposit particles on mesh --- */
    CUDA_SC_CHECK(cudaMemset(d_sc_rho, 0, mesh_size * sizeof(double)));

    /* We need z_sc = vz - dct_ave*beta on device for deposition.
     * Compute it in the deposit kernel by passing dct_ave and using dbeta. */
    /* Actually, the deposit kernel takes z directly. We need a z_sc array.
     * For simplicity, let's create a temporary z_sc array on device. */
    double *d_z_sc = NULL;
    CUDA_SC_CHECK(cudaMalloc((void**)&d_z_sc, n_particles * sizeof(double)));

    /* Quick kernel to compute z_sc = vz - dct_ave * beta */
    {
        /* Lambda-like inline: just use a simple kernel */
        /* We'll reuse the deposit kernel with z_sc computed on host... no, let's keep on GPU */
        /* Actually, let's just do it with a simple transform kernel */
    }

    /* Compute z_sc on device via a simple kernel */
    /* For now, use a small inline approach */
    int blocks_p = (n_particles + threads - 1) / threads;

    /* We'll pass vz and beta arrays and dct_ave to deposit_kernel modified version,
     * but our deposit_kernel takes a z array. Let's compute z_sc array separately. */
    /* Simple kernel to compute z_sc = vz[i] - dct_ave * beta[i] */
    /* Use thrust or a simple custom kernel. Let's add a helper kernel. */
    /* For simplicity, just download vz and beta, compute on CPU, upload z_sc */
    {
        double *h_vz2 = (double*)malloc(n_particles * sizeof(double));
        double *h_beta2 = (double*)malloc(n_particles * sizeof(double));
        double *h_z_sc = (double*)malloc(n_particles * sizeof(double));
        CUDA_SC_CHECK(cudaMemcpy(h_vz2, dvec[4], n_particles*sizeof(double), cudaMemcpyDeviceToHost));
        CUDA_SC_CHECK(cudaMemcpy(h_beta2, dbeta, n_particles*sizeof(double), cudaMemcpyDeviceToHost));
        for (int i = 0; i < n_particles; i++) h_z_sc[i] = h_vz2[i] - dct_ave * h_beta2[i];
        CUDA_SC_CHECK(cudaMemcpy(d_z_sc, h_z_sc, n_particles*sizeof(double), cudaMemcpyHostToDevice));
        free(h_vz2); free(h_beta2); free(h_z_sc);
    }

    deposit_kernel<<<blocks_p, threads>>>(
        dvec[0], dvec[2], d_z_sc,
        d_sc_charge, dstate,
        d_sc_rho,
        xmin, ymin, zmin, dxi, dyi, dzi, dx, dy, dz,
        nx, ny, nz, n_particles);
    CUDA_SC_CHECK(cudaGetLastError());
    CUDA_SC_CHECK(cudaDeviceSynchronize());

    /* --- Step 3: FFT of charge density --- */
    CUDA_SC_CHECK(cudaMemset(d_sc_crho, 0, dbl_size * sizeof(cufftDoubleComplex)));

    int blocks_m = (mesh_size + threads - 1) / threads;
    rho_to_complex_kernel<<<blocks_m, threads>>>(
        d_sc_rho, d_sc_crho, nx, ny, nz, nx2, ny2, nz2);
    CUDA_SC_CHECK(cudaDeviceSynchronize());

    CUFFT_SC_CHECK(cufftExecZ2Z(sc_fft_plan, d_sc_crho, d_sc_crho, CUFFT_FORWARD));
    CUDA_SC_CHECK(cudaDeviceSynchronize());

    /* --- Step 4: For each E-field component: Green function + FFT + multiply + IFFT --- */
    int blocks_d = (dbl_size + threads - 1) / threads;
    double delta[3] = {dx, dy, dz};

    for (int icomp = 1; icomp <= 3; icomp++) {
        /* Compute Green function on device */
        CUDA_SC_CHECK(cudaMemset(d_sc_cgrn, 0, dbl_size * sizeof(cufftDoubleComplex)));

        green_function_kernel<<<blocks_d, threads>>>(
            d_sc_cgrn, dx, dy, dz, gamma, icomp,
            nx2, ny2, nz2, 0.0, 0.0, 0.0);
        CUDA_SC_CHECK(cudaDeviceSynchronize());

        /* Apply IGF stencil */
        int stencil_size = (nx2-1)*(ny2-1)*(nz2-1);
        int blocks_s = (stencil_size + threads - 1) / threads;
        CUDA_SC_CHECK(cudaMemcpy(d_sc_cgrn2, d_sc_cgrn,
            dbl_size * sizeof(cufftDoubleComplex), cudaMemcpyDeviceToDevice));
        igf_stencil_kernel<<<blocks_s, threads>>>(
            d_sc_cgrn2, d_sc_cgrn, nx2, ny2, nz2);
        CUDA_SC_CHECK(cudaDeviceSynchronize());

        /* FFT of Green function */
        CUFFT_SC_CHECK(cufftExecZ2Z(sc_fft_plan, d_sc_cgrn, d_sc_cgrn, CUFFT_FORWARD));
        CUDA_SC_CHECK(cudaDeviceSynchronize());

        /* Multiply in frequency domain */
        complex_multiply_kernel<<<blocks_d, threads>>>(
            d_sc_crho, d_sc_cgrn, dbl_size);
        CUDA_SC_CHECK(cudaDeviceSynchronize());

        /* Inverse FFT */
        CUFFT_SC_CHECK(cufftExecZ2Z(sc_fft_plan, d_sc_cgrn, d_sc_cgrn, CUFFT_INVERSE));
        CUDA_SC_CHECK(cudaDeviceSynchronize());

        /* Extract field component */
        double scale = SC_FPEI / (double)(nx2*ny2*nz2);
        extract_field_kernel<<<blocks_m, threads>>>(
            d_sc_cgrn, d_sc_efield + (icomp-1)*mesh_size,
            nx, ny, nz, nx2, ny2, nz2, scale);
        CUDA_SC_CHECK(cudaDeviceSynchronize());
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
    CUDA_SC_CHECK(cudaDeviceSynchronize());

    cudaFree(d_z_sc);
}


/* --------------------------------------------------------------------------
 * gpu_csr_bin_particles — bin particles on GPU
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

    /* Upload per-particle charge */
    if (d_sc_charge) cudaFree(d_sc_charge);
    cudaMalloc((void**)&d_sc_charge, (size_t)n_particles * sizeof(double));
    CUDA_SC_CHECK(cudaMemcpy(d_sc_charge, h_charge,
        n_particles * sizeof(double), cudaMemcpyHostToDevice));

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
    CUDA_SC_CHECK(cudaDeviceSynchronize());

    /* Download results */
    CUDA_SC_CHECK(cudaMemcpy(h_bin_charge, d_csr_bin_charge, bsz, cudaMemcpyDeviceToHost));
    CUDA_SC_CHECK(cudaMemcpy(h_bin_x0_wt, d_csr_bin_x0_wt, bsz, cudaMemcpyDeviceToHost));
    CUDA_SC_CHECK(cudaMemcpy(h_bin_y0_wt, d_csr_bin_y0_wt, bsz, cudaMemcpyDeviceToHost));
    CUDA_SC_CHECK(cudaMemcpy(h_bin_n_particle, d_csr_bin_n_particle, bsz, cudaMemcpyDeviceToHost));
}


/* --------------------------------------------------------------------------
 * gpu_csr_apply_kicks — apply precomputed CSR/LSC kicks on GPU
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
    CUDA_SC_CHECK(cudaDeviceSynchronize());
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
    if (d_sc_crho)   { cudaFree(d_sc_crho);   d_sc_crho = NULL; }
    if (d_sc_cgrn)   { cudaFree(d_sc_cgrn);   d_sc_cgrn = NULL; }
    if (d_sc_cgrn2)  { cudaFree(d_sc_cgrn2);  d_sc_cgrn2 = NULL; }
    sc_nx2 = 0; sc_ny2 = 0; sc_nz2 = 0;

    if (d_csr_bin_charge)     { cudaFree(d_csr_bin_charge);     d_csr_bin_charge = NULL; }
    if (d_csr_bin_x0_wt)     { cudaFree(d_csr_bin_x0_wt);     d_csr_bin_x0_wt = NULL; }
    if (d_csr_bin_y0_wt)     { cudaFree(d_csr_bin_y0_wt);     d_csr_bin_y0_wt = NULL; }
    if (d_csr_bin_n_particle) { cudaFree(d_csr_bin_n_particle); d_csr_bin_n_particle = NULL; }
    if (d_csr_kick_csr)      { cudaFree(d_csr_kick_csr);      d_csr_kick_csr = NULL; }
    if (d_csr_kick_lsc)      { cudaFree(d_csr_kick_lsc);      d_csr_kick_lsc = NULL; }
    d_csr_bin_cap = 0;
}

#endif /* USE_GPU_TRACKING */
