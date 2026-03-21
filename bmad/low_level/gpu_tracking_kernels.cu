/*
 * gpu_tracking_kernels.cu
 *
 * CUDA kernels for GPU-accelerated particle tracking through supported
 * elements (drift, quadrupole, sbend, lcavity, pipe).  Called from Fortran
 * via iso_c_binding wrappers defined in gpu_tracking_mod.f90.
 * Pipe elements reuse the quad kernel with b1=0 (no quad gradient).
 *
 * Build requirements:
 *   - CUDA Toolkit (cuda_runtime.h)
 *   - Compile with nvcc: -DUSE_GPU_TRACKING
 *   - Link with: -lcudart
 *
 * The wrapper caches device memory allocations to avoid re-allocation
 * on repeated calls.
 */

#ifdef USE_GPU_TRACKING

#include <cuda_runtime.h>
#include <curand_kernel.h>
#include <math.h>
#include <stdio.h>
#include <string.h>

/* --------------------------------------------------------------------------
 * CUDA error checking macro.  Returns -1 from the calling function on error.
 * Host wrappers (void) check the return value and issue a bare "return;".
 * -------------------------------------------------------------------------- */
#define CUDA_CHECK(call) do { \
    cudaError_t err_ = (call); \
    if (err_ != cudaSuccess) { \
        fprintf(stderr, "[gpu_tracking] CUDA error: %s at %s:%d\n", \
                cudaGetErrorString(err_), __FILE__, __LINE__); \
        return -1; \
    } \
} while(0)

/* Same as CUDA_CHECK but for void functions (host wrappers) */
#define CUDA_CHECK_VOID(call) do { \
    cudaError_t err_ = (call); \
    if (err_ != cudaSuccess) { \
        fprintf(stderr, "[gpu_tracking] CUDA error: %s at %s:%d\n", \
                cudaGetErrorString(err_), __FILE__, __LINE__); \
        return; \
    } \
} while(0)

/* Physical constants (must match Bmad values exactly) */
#define C_LIGHT   2.99792458e8
#define ALIVE_ST  1   /* alive$ */
#define LOST_NEG_X 3  /* lost_neg_x$ */
#define LOST_POS_X 4  /* lost_pos_x$ */
#define LOST_NEG_Y 5  /* lost_neg_y$ */
#define LOST_POS_Y 6  /* lost_pos_y$ */
#define LOST_PZ   8   /* lost_pz$ */

/* --------------------------------------------------------------------------
 * Cached device buffers
 * -------------------------------------------------------------------------- */
static double *d_vec[6]  = {NULL,NULL,NULL,NULL,NULL,NULL};
static int    *d_state   = NULL;
static double *d_beta    = NULL;
static double *d_p0c     = NULL;
static double *d_s       = NULL;
static double *d_t       = NULL;
static int     d_cap     = 0;          /* allocated capacity (particles) */

/* Cached device buffers for multipole data */
static double *d_a2  = NULL;
static double *d_b2  = NULL;
static double *d_ea2 = NULL;
static double *d_eb2 = NULL;
static double *d_cm  = NULL;

/* --------------------------------------------------------------------------
 * ensure_buffers -- (re-)allocate device arrays when size changes
 * -------------------------------------------------------------------------- */
static int ensure_buffers(int n)
{
    if (n <= 0) return (n == 0) ? 0 : -1;
    if (n <= d_cap) return 0;

    /* Free old */
    for (int k = 0; k < 6; k++) { if (d_vec[k]) cudaFree(d_vec[k]); d_vec[k] = NULL; }
    if (d_state) cudaFree(d_state); d_state = NULL;
    if (d_beta)  cudaFree(d_beta);  d_beta  = NULL;
    if (d_p0c)   cudaFree(d_p0c);   d_p0c   = NULL;
    if (d_s)     cudaFree(d_s);     d_s     = NULL;
    if (d_t)     cudaFree(d_t);     d_t     = NULL;

    size_t db = (size_t)n * sizeof(double);
    size_t ib = (size_t)n * sizeof(int);

    for (int k = 0; k < 6; k++) {
        if (cudaMalloc((void**)&d_vec[k], db) != cudaSuccess) goto fail;
    }
    if (cudaMalloc((void**)&d_state, ib) != cudaSuccess) goto fail;
    if (cudaMalloc((void**)&d_beta,  db) != cudaSuccess) goto fail;
    if (cudaMalloc((void**)&d_p0c,   db) != cudaSuccess) goto fail;
    if (cudaMalloc((void**)&d_s,     db) != cudaSuccess) goto fail;
    if (cudaMalloc((void**)&d_t,     db) != cudaSuccess) goto fail;

    d_cap = n;
    return 0;

fail:
    fprintf(stderr, "[gpu_tracking] cudaMalloc failed for %d particles\n", n);
    /* Clean up any partially allocated buffers */
    for (int k = 0; k < 6; k++) { if (d_vec[k]) cudaFree(d_vec[k]); d_vec[k] = NULL; }
    if (d_state) cudaFree(d_state); d_state = NULL;
    if (d_beta)  cudaFree(d_beta);  d_beta  = NULL;
    if (d_p0c)   cudaFree(d_p0c);   d_p0c   = NULL;
    if (d_s)     cudaFree(d_s);     d_s     = NULL;
    if (d_t)     cudaFree(d_t);     d_t     = NULL;
    d_cap = 0;
    return -1;
}

/* --------------------------------------------------------------------------
 * Accessor functions for device buffer pointers -- used by
 * gpu_spacecharge_kernels.cu to access the cached particle buffers
 * without breaking static linkage.
 * -------------------------------------------------------------------------- */
extern "C" void gpu_get_device_ptrs_(
    double **out_vec0, double **out_vec1, double **out_vec2,
    double **out_vec3, double **out_vec4, double **out_vec5,
    int **out_state, double **out_beta, double **out_p0c)
{
    *out_vec0 = d_vec[0]; *out_vec1 = d_vec[1]; *out_vec2 = d_vec[2];
    *out_vec3 = d_vec[3]; *out_vec4 = d_vec[4]; *out_vec5 = d_vec[5];
    *out_state = d_state; *out_beta = d_beta; *out_p0c = d_p0c;
}

/* --------------------------------------------------------------------------
 * gpu_tracking_available -- query whether a CUDA GPU is present
 * -------------------------------------------------------------------------- */
extern "C" int gpu_tracking_available_(void)
{
    int count = 0;
    cudaError_t err = cudaGetDeviceCount(&count);
    return (err == cudaSuccess && count > 0) ? 1 : 0;
}

/* --------------------------------------------------------------------------
 * Numerically stable sqrt(1+x) - 1
 * -------------------------------------------------------------------------- */
__device__ __forceinline__ double sqrt_one_dev(double x)
{
    double sq = sqrt(1.0 + x);
    return x / (sq + 1.0);
}

/* --------------------------------------------------------------------------
 * Low energy z correction (matches low_energy_z_correction.f90).
 * Returns the z increment for one integration step.
 * -------------------------------------------------------------------------- */
__device__ __forceinline__ double low_energy_z_correction_dev(
    double pz_val, double step_len, double beta_val, double beta_ref,
    double mc2, double e_tot_ele)
{
    if (mc2 * (beta_ref * pz_val) * (beta_ref * pz_val) < 3e-7 * e_tot_ele) {
        /* Taylor expansion for small pz -- avoids precision loss */
        double mr = mc2 / e_tot_ele;
        double b02 = beta_ref * beta_ref;
        double f_tay = b02 * (2.0 * b02 - mr * mr * 0.5);
        return step_len * pz_val * (1.0 - 1.5 * pz_val * b02 + pz_val * pz_val * f_tay) * mr * mr;
    } else {
        return step_len * (beta_val - beta_ref) / beta_ref;
    }
}

/* --------------------------------------------------------------------------
 * drift_body_dev -- core drift physics for a single particle
 *
 * Updates position (x, y), longitudinal (z), and time (t) for a drift of
 * length ds.  Returns 0 on success, 1 if particle is lost (pxy2 >= 1).
 * Does NOT update s_pos -- callers handle that themselves.
 * -------------------------------------------------------------------------- */
/* Returns 0 on success, or a LOST_xxx code on failure.
 * Matches CPU orbit_too_large: px^2+py^2 > (1+pz)^2 → lost,
 * with direction from the dominant momentum component. */
__device__ int drift_body_dev(
    double *x, double *px, double *y, double *py, double *z, double *pz,
    double *beta, double *t, double mc2, double p0c, double ds)
{
    double delta  = *pz;
    double rel_pc = 1.0 + delta;

    if (rel_pc < 0.0) return LOST_PZ;
    if (*beta <= 0.0) return LOST_PZ;

    double px_rel = *px / rel_pc;
    double py_rel = *py / rel_pc;
    double pxy2   = px_rel * px_rel + py_rel * py_rel;

    if (pxy2 >= 1.0) {
        /* Match CPU orbit_too_large: use un-normalized comparison for exact
         * boundary agreement, and set direction from dominant component. */
        double f_unstable = (*px)*(*px) + (*py)*(*py) - rel_pc*rel_pc;
        if (f_unstable > 0.0) {
            if (fabs(*px) > fabs(*py))
                return (*px > 0.0) ? LOST_POS_X : LOST_NEG_X;
            else
                return (*py > 0.0) ? LOST_POS_Y : LOST_NEG_Y;
        }
        /* pxy2 >= 1 but f_unstable <= 0: rounding difference, treat as alive */
    }

    double ps_rel = sqrt(1.0 - pxy2);

    *x += ds * px_rel / ps_rel;
    *y += ds * py_rel / ps_rel;

    double p_tot = p0c * rel_pc;
    double A = (mc2 * mc2 * (2.0 * delta + delta * delta)) /
               (p_tot * p_tot + mc2 * mc2);
    *z += ds * (sqrt_one_dev(A) + sqrt_one_dev(-pxy2) / ps_rel);

    *t += ds / (*beta * ps_rel * C_LIGHT);

    return 0;
}

/* =========================================================================
 * DRIFT KERNEL
 * Replicates the physics of track_a_drift (forward, include_ref_motion=true)
 * ========================================================================= */
__global__ void drift_kernel(
    double *vx, double *vpx, double *vy, double *vpy, double *vz, double *vpz,
    int *state, double *beta, double *p0c, double *s_pos, double *t_time,
    double mc2, double length, int n)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    if (state[i] != ALIVE_ST) return;

    {
        int rc = drift_body_dev(&vx[i], &vpx[i], &vy[i], &vpy[i], &vz[i], &vpz[i],
                                &beta[i], &t_time[i], mc2, p0c[i], length);
        if (rc) { state[i] = rc; return; }
    }

    /* s update (direction = +1) */
    s_pos[i] += length;
}

/* --------------------------------------------------------------------------
 * Device helper: compute x^p for small non-negative integer p
 * -------------------------------------------------------------------------- */
__device__ __forceinline__ double ipow(double x, int p)
{
    if (p == 0) return 1.0;
    double r = 1.0;
    for (int k = 0; k < p; k++) r *= x;
    return r;
}

/* --------------------------------------------------------------------------
 * Device helper: apply magnetic multipole kicks (ab_multipole_kick equivalent)
 *
 * Precomputed c_multi coefficients are passed in cm[N_MULTI*N_MULTI].
 * Scaled a2[n], b2[n] arrays include charge/orientation/scale factors.
 * -------------------------------------------------------------------------- */
/* Must match Bmad's n_pole_maxx + 1 (defined in bmad_struct.f90) */
#define N_MULTI 22

__device__ void multipole_kick_dev(
    const double *a2, const double *b2, int ix_max,
    const double *cm,
    double x, double y, double *kx_out, double *ky_out)
{
    double kx = 0.0, ky = 0.0;
    for (int nn = 0; nn <= ix_max; nn++) {
        if (a2[nn] == 0.0 && b2[nn] == 0.0) continue;
        /* even m -- cm is Fortran column-major: cm(nn, m) at offset m*N_MULTI+nn */
        for (int m = 0; m <= nn; m += 2) {
            double f = cm[m * N_MULTI + nn] * ipow(x, nn - m) * ipow(y, m);
            kx += b2[nn] * f;
            ky -= a2[nn] * f;
        }
        /* odd m */
        for (int m = 1; m <= nn; m += 2) {
            double f = cm[m * N_MULTI + nn] * ipow(x, nn - m) * ipow(y, m);
            kx += a2[nn] * f;
            ky += b2[nn] * f;
        }
    }
    *kx_out = kx;
    *ky_out = ky;
}

/* --------------------------------------------------------------------------
 * apply_electric_kick_dev -- apply a scaled electric multipole kick and
 * update pz, beta, and z.  Caller pre-computes the scaled kick (kx_s, ky_s)
 * including any element-specific factors (1/beta, (1+g*x)/ps, etc.).
 * Returns 1 if particle is lost (alpha < -1), 0 on success.
 * -------------------------------------------------------------------------- */
__device__ int apply_electric_kick_dev(
    double kx_s, double ky_s,
    double *px, double *py, double *pz, double *z,
    double *beta_val, double *beta_arr_i,
    double mc2, double p0c_val, int *state_i)
{
    double px_old = *px, py_old = *py, pz_old = *pz;
    *px += kx_s;
    *py += ky_s;
    double alpha = (kx_s * (2.0*px_old + kx_s) + ky_s * (2.0*py_old + ky_s))
                   / ((1.0 + pz_old) * (1.0 + pz_old));
    if (alpha < -1.0) { *state_i = LOST_PZ; return 1; }
    *pz = pz_old + (1.0 + pz_old) * sqrt_one_dev(alpha);
    double new_beta = (1.0 + *pz) / sqrt((1.0 + *pz) * (1.0 + *pz)
                      + (mc2 / p0c_val) * (mc2 / p0c_val));
    *z = *z * new_beta / *beta_val;
    *beta_val = new_beta;
    *beta_arr_i = new_beta;
    return 0;
}

/* --------------------------------------------------------------------------
 * quad_mat2_calc_dev -- compute 2x2 transfer matrix and z-correction terms
 *
 * Given focusing strength k_val and step length, computes:
 *   c, s: cosine-like and sine-like matrix elements
 *   zc1, zc2, zc3: z-correction coefficients for the plane
 * -------------------------------------------------------------------------- */
__device__ void quad_mat2_calc_dev(
    double k_val, double step_len, double rel_p,
    double *c_out, double *s_out,
    double *zc1, double *zc2, double *zc3)
{
    double abs_k = fabs(k_val);
    double sqrt_k = sqrt(abs_k);
    double sk_l = sqrt_k * step_len;

    if (fabs(sk_l) < 1e-10) {
        double kl2 = k_val * step_len * step_len;
        *c_out = 1.0 + kl2 * 0.5;
        *s_out = (1.0 + kl2 / 6.0) * step_len;
    } else if (k_val < 0.0) {
        *c_out = cos(sk_l);
        *s_out = sin(sk_l) / sqrt_k;
    } else {
        *c_out = cosh(sk_l);
        *s_out = sinh(sk_l) / sqrt_k;
    }

    double c = *c_out, s = *s_out;
    *zc1 = k_val * (-c * s + step_len) / 4.0;
    *zc2 = -k_val * s * s / (2.0 * rel_p);
    *zc3 = -(c * s + step_len) / (4.0 * rel_p * rel_p);
}

/* =========================================================================
 * QUADRUPOLE KERNEL
 * Replicates the full body of track_a_quadrupole including:
 *   - Split-step integration with n_step steps
 *   - Magnetic and electric multipole kicks (interleaved)
 *   - low_energy_z_correction
 *   - Forward tracking only (direction=1, time_dir=1)
 *
 * When ix_mag_max < 0, ix_elec_max < 0, and n_step == 1, reduces to simple case.
 * ========================================================================= */
__global__ void quad_kernel(
    double *vx, double *vpx, double *vy, double *vpy, double *vz, double *vpz,
    int *state, double *beta_arr, double *p0c_arr, double *t_arr,
    double mc2, double b1, double ele_length,
    double delta_ref_time, double e_tot_ele, double charge_dir,
    int n_particles,
    /* Multipole parameters (may be NULL/unused if ix < 0) */
    const double *d_a2, const double *d_b2, const double *d_cm,
    int ix_mag_max, int n_step,
    /* Electric multipole parameters */
    const double *d_ea2, const double *d_eb2, int ix_elec_max)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n_particles) return;
    if (state[i] != ALIVE_ST) return;

    int has_mag = (ix_mag_max >= 0);
    int has_elec = (ix_elec_max >= 0);
    double step_len = ele_length / (double)n_step;
    double z_start = vz[i];
    double t_start = t_arr[i];
    double beta_val = beta_arr[i];
    double p0c_val = p0c_arr[i];
    double beta_ref = p0c_val / e_tot_ele;

    /* Entrance half magnetic multipole kick (scale = r_step/2, built into d_a2/d_b2) */
    if (has_mag) {
        double kx, ky;
        multipole_kick_dev(d_a2, d_b2, ix_mag_max, d_cm,
                           vx[i], vy[i], &kx, &ky);
        vpx[i] += 0.5 * kx;
        vpy[i] += 0.5 * ky;
    }

    /* Entrance half electric multipole kick */
    if (has_elec) {
        double kx, ky;
        multipole_kick_dev(d_ea2, d_eb2, ix_elec_max, d_cm,
                           vx[i], vy[i], &kx, &ky);
        if (apply_electric_kick_dev(0.5 * kx / beta_val, 0.5 * ky / beta_val,
                &vpx[i], &vpy[i], &vpz[i], &vz[i],
                &beta_val, &beta_arr[i], mc2, p0c_val, &state[i])) return;
    }

    /* Body: n_step integration steps */
    for (int istep = 1; istep <= n_step; istep++) {

        double rel_p = 1.0 + vpz[i];
        double k1 = charge_dir * b1 / (ele_length * rel_p);

        /* quad_mat2_calc for x plane (k_x = -k1) and y plane (k_y = +k1) */
        double cx, sx, zc_x1, zc_x2, zc_x3;
        double cy, sy, zc_y1, zc_y2, zc_y3;
        quad_mat2_calc_dev(-k1, step_len, rel_p, &cx, &sx, &zc_x1, &zc_x2, &zc_x3);
        quad_mat2_calc_dev( k1, step_len, rel_p, &cy, &sy, &zc_y1, &zc_y2, &zc_y3);

        /* Save pre-matrix coords for z update */
        double x0 = vx[i], px0 = vpx[i];
        double y0 = vy[i], py0 = vpy[i];

        /* z update from quad focusing */
        vz[i] += zc_x1*x0*x0 + zc_x2*x0*px0 + zc_x3*px0*px0 +
                 zc_y1*y0*y0 + zc_y2*y0*py0 + zc_y3*py0*py0;

        /* Apply 2x2 matrices */
        double k1_x = -k1, k1_y = k1;
        vx[i]  = cx * x0 + (sx / rel_p) * px0;
        vpx[i] = (k1_x * sx * rel_p) * x0 + cx * px0;
        vy[i]  = cy * y0 + (sy / rel_p) * py0;
        vpy[i] = (k1_y * sy * rel_p) * y0 + cy * py0;

        /* Low energy z correction */
        vz[i] += low_energy_z_correction_dev(vpz[i], step_len, beta_val, beta_ref, mc2, e_tot_ele);

        /* Magnetic multipole kick (half at last step, full otherwise) */
        if (has_mag) {
            double kx, ky;
            multipole_kick_dev(d_a2, d_b2, ix_mag_max, d_cm,
                               vx[i], vy[i], &kx, &ky);
            double scl = (istep == n_step) ? 0.5 : 1.0;
            vpx[i] += scl * kx;
            vpy[i] += scl * ky;
        }

        /* Electric multipole kick (half at last step, full otherwise) */
        if (has_elec) {
            double kx, ky;
            multipole_kick_dev(d_ea2, d_eb2, ix_elec_max, d_cm,
                               vx[i], vy[i], &kx, &ky);
            double scl = (istep == n_step) ? 0.5 : 1.0;
            if (apply_electric_kick_dev(scl * kx / beta_val, scl * ky / beta_val,
                    &vpx[i], &vpy[i], &vpz[i], &vz[i],
                    &beta_val, &beta_arr[i], mc2, p0c_val, &state[i])) return;
        }
    }

    /* Time update */
    t_arr[i] = t_start + delta_ref_time + (z_start - vz[i]) / (beta_val * C_LIGHT);
}

/* ==========================================================================
 * SEXTUPOLE KERNEL -- drift-kick-drift split-step integrator
 *
 * Tracks through a thick sextupole (or any thick multipole element) using:
 *   1. Half multipole kick at entrance
 *   2. n_step sub-steps of [drift + full kick] (half kick at last step)
 *   3. Time update from z displacement
 *
 * Unlike the quad kernel, there is no linear focusing matrix -- all
 * transverse impulse comes from nonlinear multipole kicks via drift sub-steps.
 * ========================================================================== */

__global__ void sextupole_kernel(
    double *vx, double *vpx, double *vy, double *vpy, double *vz, double *vpz,
    int *state, double *beta_arr, double *p0c_arr, double *t_arr,
    double mc2, double ele_length,
    double delta_ref_time, double e_tot_ele, double charge_dir,
    int n_particles,
    const double *d_a2, const double *d_b2, const double *d_cm,
    int ix_mag_max, int n_step,
    const double *d_ea2, const double *d_eb2, int ix_elec_max)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n_particles) return;
    if (state[i] != ALIVE_ST) return;

    int has_mag = (ix_mag_max >= 0);
    int has_elec = (ix_elec_max >= 0);
    double step_len = ele_length / (double)n_step;
    double z_start = vz[i];
    double t_start = t_arr[i];
    double beta_val = beta_arr[i];
    double p0c_val = p0c_arr[i];
    double beta_ref = p0c_val / e_tot_ele;

    /* Entrance half magnetic multipole kick */
    if (has_mag) {
        double kx, ky;
        multipole_kick_dev(d_a2, d_b2, ix_mag_max, d_cm,
                           vx[i], vy[i], &kx, &ky);
        vpx[i] += 0.5 * kx;
        vpy[i] += 0.5 * ky;
    }

    /* Entrance half electric multipole kick */
    if (has_elec) {
        double kx, ky;
        multipole_kick_dev(d_ea2, d_eb2, ix_elec_max, d_cm,
                           vx[i], vy[i], &kx, &ky);
        if (apply_electric_kick_dev(0.5 * kx / beta_val, 0.5 * ky / beta_val,
                &vpx[i], &vpy[i], &vpz[i], &vz[i],
                &beta_val, &beta_arr[i], mc2, p0c_val, &state[i])) return;
    }

    /* Body: n_step drift-kick sub-steps */
    for (int istep = 1; istep <= n_step; istep++) {

        /* Drift through one sub-step */
        {
            int rc = drift_body_dev(&vx[i], &vpx[i], &vy[i], &vpy[i], &vz[i], &vpz[i],
                                    &beta_val, &t_arr[i], mc2, p0c_val, step_len);
            if (rc) { state[i] = rc; return; }
        }
        /* Time handled at end (not from drift_body_dev) -- reset t */
        t_arr[i] = t_start;

        /* No low_energy_z_correction here: drift_body_dev already handles the
           full z-update via the exact formula. The correction is only needed for
           the quad kernel where z is updated via the focusing matrix (not drift). */

        /* Magnetic multipole kick (half at last step, full otherwise) */
        if (has_mag) {
            double kx, ky;
            multipole_kick_dev(d_a2, d_b2, ix_mag_max, d_cm,
                               vx[i], vy[i], &kx, &ky);
            double scl = (istep == n_step) ? 0.5 : 1.0;
            vpx[i] += scl * kx;
            vpy[i] += scl * ky;
        }

        /* Electric multipole kick (half at last step, full otherwise) */
        if (has_elec) {
            double kx, ky;
            multipole_kick_dev(d_ea2, d_eb2, ix_elec_max, d_cm,
                               vx[i], vy[i], &kx, &ky);
            double scl = (istep == n_step) ? 0.5 : 1.0;
            if (apply_electric_kick_dev(scl * kx / beta_val, scl * ky / beta_val,
                    &vpx[i], &vpy[i], &vpz[i], &vz[i],
                    &beta_val, &beta_arr[i], mc2, p0c_val, &state[i])) return;
        }
    }

    /* Time update */
    t_arr[i] = t_start + delta_ref_time + (z_start - vz[i]) / (beta_val * C_LIGHT);
}


/* --------------------------------------------------------------------------
 * upload_particle_data -- H->D transfer of core particle arrays
 * -------------------------------------------------------------------------- */
static int upload_particle_data(int n,
    double *h_vx, double *h_vpx, double *h_vy, double *h_vpy,
    double *h_vz, double *h_vpz,
    int *h_state, double *h_beta, double *h_p0c, double *h_t)
{
    size_t db = (size_t)n * sizeof(double);
    size_t ib = (size_t)n * sizeof(int);
    CUDA_CHECK(cudaMemcpy(d_vec[0], h_vx,    db, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_vec[1], h_vpx,   db, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_vec[2], h_vy,    db, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_vec[3], h_vpy,   db, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_vec[4], h_vz,    db, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_vec[5], h_vpz,   db, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_state,  h_state,  ib, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_beta,   h_beta,   db, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_p0c,    h_p0c,    db, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_t,      h_t,      db, cudaMemcpyHostToDevice));
    return 0;
}

/* --------------------------------------------------------------------------
 * download_particle_data -- D→H transfer of core particle arrays
 *
 * copy_beta/copy_p0c: set to 1 to also download beta/p0c arrays.
 * Drift: both 0.  Quad/bend: copy_beta only when electric multipoles.
 * Lcavity: both 1.
 * -------------------------------------------------------------------------- */
static int download_particle_data(int n,
    double *h_vx, double *h_vpx, double *h_vy, double *h_vpy,
    double *h_vz, double *h_vpz,
    int *h_state, double *h_beta, double *h_p0c, double *h_t,
    int copy_beta, int copy_p0c)
{
    size_t db = (size_t)n * sizeof(double);
    size_t ib = (size_t)n * sizeof(int);
    CUDA_CHECK(cudaMemcpy(h_vx,    d_vec[0], db, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h_vpx,   d_vec[1], db, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h_vy,    d_vec[2], db, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h_vpy,   d_vec[3], db, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h_vz,    d_vec[4], db, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h_vpz,   d_vec[5], db, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h_state,  d_state,  ib, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h_t,      d_t,      db, cudaMemcpyDeviceToHost));
    if (copy_beta) { CUDA_CHECK(cudaMemcpy(h_beta, d_beta, db, cudaMemcpyDeviceToHost)); }
    if (copy_p0c)  { CUDA_CHECK(cudaMemcpy(h_p0c, d_p0c,  db, cudaMemcpyDeviceToHost)); }
    return 0;
}

/* --------------------------------------------------------------------------
 * upload_multipole_data -- H→D transfer of multipole coefficient arrays
 * -------------------------------------------------------------------------- */
static int upload_multipole_data(
    double *h_a2, double *h_b2, double *h_cm,
    double *h_ea2, double *h_eb2,
    int ix_mag_max, int ix_elec_max)
{
    size_t multi_sz = N_MULTI * sizeof(double);
    size_t cm_sz    = N_MULTI * N_MULTI * sizeof(double);

    if (ix_mag_max >= 0 || ix_elec_max >= 0) {
        if (!d_cm) { CUDA_CHECK(cudaMalloc((void**)&d_cm, cm_sz)); }
        CUDA_CHECK(cudaMemcpy(d_cm, h_cm, cm_sz, cudaMemcpyHostToDevice));
    }
    if (ix_mag_max >= 0) {
        if (!d_a2) { CUDA_CHECK(cudaMalloc((void**)&d_a2, multi_sz)); }
        if (!d_b2) { CUDA_CHECK(cudaMalloc((void**)&d_b2, multi_sz)); }
        CUDA_CHECK(cudaMemcpy(d_a2, h_a2, multi_sz, cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_b2, h_b2, multi_sz, cudaMemcpyHostToDevice));
    }
    if (ix_elec_max >= 0) {
        if (!d_ea2) { CUDA_CHECK(cudaMalloc((void**)&d_ea2, multi_sz)); }
        if (!d_eb2) { CUDA_CHECK(cudaMalloc((void**)&d_eb2, multi_sz)); }
        CUDA_CHECK(cudaMemcpy(d_ea2, h_ea2, multi_sz, cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_eb2, h_eb2, multi_sz, cudaMemcpyHostToDevice));
    }
    return 0;
}

/* =========================================================================
 * HOST WRAPPER: gpu_track_drift
 *
 * Fortran signature (iso_c_binding):
 *   subroutine gpu_track_drift(vec_x, vec_px, vec_y, vec_py, vec_z, vec_pz,
 *              state, beta, p0c, s_pos, t_time, mc2, length, n) bind(C)
 * ========================================================================= */
extern "C" void gpu_track_drift_(
    double *h_vx, double *h_vpx, double *h_vy, double *h_vpy,
    double *h_vz, double *h_vpz,
    int *h_state, double *h_beta, double *h_p0c,
    double *h_s, double *h_t,
    double mc2, double length, int n)
{
    if (ensure_buffers(n) != 0) return;

    size_t db = (size_t)n * sizeof(double);

    /* Host -> Device */
    if (upload_particle_data(n, h_vx, h_vpx, h_vy, h_vpy, h_vz, h_vpz,
                             h_state, h_beta, h_p0c, h_t) != 0) return;
    CUDA_CHECK_VOID(cudaMemcpy(d_s, h_s, db, cudaMemcpyHostToDevice));

    /* Launch kernel */
    int threads = 256;
    int blocks  = (n + threads - 1) / threads;
    drift_kernel<<<blocks, threads>>>(
        d_vec[0], d_vec[1], d_vec[2], d_vec[3], d_vec[4], d_vec[5],
        d_state, d_beta, d_p0c, d_s, d_t, mc2, length, n);
    CUDA_CHECK_VOID(cudaGetLastError());
    CUDA_CHECK_VOID(cudaDeviceSynchronize());

    /* Device -> Host */
    if (download_particle_data(n, h_vx, h_vpx, h_vy, h_vpy, h_vz, h_vpz,
                               h_state, h_beta, h_p0c, h_t, 0, 0) != 0) return;
    CUDA_CHECK_VOID(cudaMemcpy(h_s, d_s, db, cudaMemcpyDeviceToHost));
}

/* =========================================================================
 * HOST WRAPPER: gpu_track_quad
 * ========================================================================= */
extern "C" void gpu_track_quad_(
    double *h_vx, double *h_vpx, double *h_vy, double *h_vpy,
    double *h_vz, double *h_vpz,
    int *h_state, double *h_beta, double *h_p0c, double *h_t,
    double mc2, double b1, double ele_length,
    double delta_ref_time, double e_tot_ele, double charge_dir,
    int n_particles,
    double *h_a2, double *h_b2, double *h_cm,
    int ix_mag_max, int n_step,
    double *h_ea2, double *h_eb2, int ix_elec_max)
{
    if (ensure_buffers(n_particles) != 0) return;

    /* Host -> Device */
    if (upload_particle_data(n_particles, h_vx, h_vpx, h_vy, h_vpy, h_vz, h_vpz,
                             h_state, h_beta, h_p0c, h_t) != 0) return;
    if (upload_multipole_data(h_a2, h_b2, h_cm, h_ea2, h_eb2,
                              ix_mag_max, ix_elec_max) != 0) return;

    /* Launch kernel */
    int threads = 256;
    int blocks  = (n_particles + threads - 1) / threads;
    quad_kernel<<<blocks, threads>>>(
        d_vec[0], d_vec[1], d_vec[2], d_vec[3], d_vec[4], d_vec[5],
        d_state, d_beta, d_p0c, d_t,
        mc2, b1, ele_length, delta_ref_time, e_tot_ele, charge_dir,
        n_particles, d_a2, d_b2, d_cm, ix_mag_max, n_step,
        d_ea2, d_eb2, ix_elec_max);
    CUDA_CHECK_VOID(cudaGetLastError());
    CUDA_CHECK_VOID(cudaDeviceSynchronize());

    /* Device -> Host (electric kicks can modify beta) */
    if (download_particle_data(n_particles, h_vx, h_vpx, h_vy, h_vpy, h_vz, h_vpz,
                               h_state, h_beta, h_p0c, h_t,
                               (ix_elec_max >= 0), 0) != 0) return;
}

/* =========================================================================
 * BEND KERNEL
 * Replicates the body of track_a_bend including:
 *   - General nonlinear bend map (g != 0)
 *   - sbend_body_with_k1_map (b1 != 0, quad component in bend)
 *   - Pure drift fallback (g=0, dg=0)
 *   - Linear approximation for near-axis particles
 *   - Split-step magnetic/electric multipole kicks
 *   - Low energy z correction (for k1 map)
 *   - Forward tracking only (direction=1, time_dir=1)
 * ========================================================================= */

/* Device helper: sinc(x) = sin(x)/x, numerically stable for small x */
__device__ __forceinline__ double sinc_dev(double x)
{
    if (fabs(x) < 1e-8) return 1.0;
    return sin(x) / x;
}

/* Device helper: cosc(x) = (1-cos(x))/x^2, numerically stable */
__device__ __forceinline__ double cosc_dev(double x)
{
    if (fabs(x) < 1e-8) return 0.5;
    double h = x * 0.5;
    double s = sinc_dev(h);
    return 0.5 * s * s;
}

/* Device helper: sincc(x) = (x - sin(x))/x^3, numerically stable */
__device__ __forceinline__ double sincc_dev(double x)
{
    double x2 = x * x;
    if (fabs(x) < 0.1) {
        return (1.0/6.0) + x2 * ((-1.0/120.0) + x2 * ((1.0/5040.0) + x2 * (-1.0/362880.0)));
    }
    return (x - sin(x)) / (x * x2);
}

__global__ void bend_kernel(
    double *vx, double *vpx, double *vy, double *vpy, double *vz, double *vpz,
    int *state, double *beta_arr, double *p0c_arr, double *t_arr,
    double mc2, double g, double g_tot, double dg, double b1,
    double ele_length, double delta_ref_time, double e_tot_ele,
    double rel_charge_dir,
    double p0c_ele,
    int n_particles,
    /* Multipole parameters */
    const double *d_a2, const double *d_b2, const double *d_cm,
    int ix_mag_max, int n_step,
    /* Electric multipole parameters */
    const double *d_ea2, const double *d_eb2, int ix_elec_max)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n_particles) return;
    if (state[i] != ALIVE_ST) return;

    int has_mag = (ix_mag_max >= 0);
    int has_elec = (ix_elec_max >= 0);
    double step_len = ele_length / (double)n_step;
    double angle = g * step_len;
    double z_start = vz[i];
    double t_start = t_arr[i];
    double beta_val = beta_arr[i];
    double p0c_val = p0c_arr[i];
    double beta_ref = p0c_ele / e_tot_ele;

    /* Entrance half magnetic multipole kick */
    if (has_mag) {
        double kx, ky;
        multipole_kick_dev(d_a2, d_b2, ix_mag_max, d_cm,
                           vx[i], vy[i], &kx, &ky);
        /* For bends, magnetic kick includes (1+g*x) factor and uses field formulation.
         * The precomputed a2/b2 arrays include all element-level scaling.
         * The (1+g*x) factor is applied here per-particle. */
        double f_gx = 1.0 + g * vx[i];
        vpx[i] += 0.5 * kx * f_gx;
        vpy[i] += 0.5 * ky * f_gx;
    }

    /* Entrance half electric multipole kick */
    if (has_elec) {
        double kx, ky;
        multipole_kick_dev(d_ea2, d_eb2, ix_elec_max, d_cm,
                           vx[i], vy[i], &kx, &ky);
        double f_gx = 1.0 + g * vx[i];
        double rel_p0 = 1.0 + vpz[i];
        double ps = sqrt(rel_p0*rel_p0 - vpx[i]*vpx[i] - vpy[i]*vpy[i]) / rel_p0;
        double f_scale = 0.5 * f_gx / (ps * beta_val);
        if (apply_electric_kick_dev(kx * f_scale, ky * f_scale,
                &vpx[i], &vpy[i], &vpz[i], &vz[i],
                &beta_val, &beta_arr[i], mc2, p0c_val, &state[i])) return;
    }

    /* Body: n_step integration steps */
    for (int istep = 1; istep <= n_step; istep++) {

        double pz = vpz[i];
        double rel_p = 1.0 + pz;

        /* ---- Branch 1: b1 != 0 → sbend_body_with_k1_map ---- */
        if (b1 != 0.0) {
            double k1 = b1 / ele_length;
            double k_x = k1 + g * g_tot;
            double x_c = (g * rel_p - g_tot) / k_x;
            double om_x = sqrt(fabs(k_x) / rel_p);
            double om_y = sqrt(fabs(k1) / rel_p);
            double tau_x = (k_x > 0) ? -1.0 : 1.0;
            double tau_y = (k1 > 0) ? 1.0 : -1.0;
            /* Note: k_x > 0 means focusing in x → sin/cos; k_x < 0 → sinh/cosh */
            /* But tau_x = -sign(1, k_x), so tau_x < 0 when k_x > 0 */

            double arg_x = om_x * step_len;
            double s_x, c_x, z2;
            if (arg_x < 1e-6) {
                s_x = (1.0 + tau_x * arg_x * arg_x / 6.0) * step_len;
                c_x = 1.0 + tau_x * arg_x * arg_x / 2.0;
                z2 = g * step_len * step_len / (2.0 * rel_p);
            } else if (k_x > 0) {
                s_x = sin(arg_x) / om_x;
                c_x = cos(arg_x);
                z2 = tau_x * g * (1.0 - c_x) / (rel_p * om_x * om_x);
            } else {
                s_x = sinh(arg_x) / om_x;
                c_x = cosh(arg_x);
                z2 = tau_x * g * (1.0 - c_x) / (rel_p * om_x * om_x);
            }

            double arg_y = om_y * step_len;
            double s_y, c_y;
            if (arg_y < 1e-6) {
                s_y = (1.0 + tau_y * arg_y * arg_y / 6.0) * step_len;
                c_y = 1.0 + tau_y * arg_y * arg_y / 2.0;
            } else if (k1 < 0) {
                s_y = sin(om_y * step_len) / om_y;
                c_y = cos(om_y * step_len);
            } else {
                s_y = sinh(om_y * step_len) / om_y;
                c_y = cosh(om_y * step_len);
            }

            double r1 = vx[i] - x_c;
            double r2 = vpx[i];
            double r3 = vy[i];
            double r4 = vpy[i];

            double z0  = -g * x_c * step_len;
            double z1  = -g * s_x;
            double z11 = tau_x * om_x*om_x * (step_len - c_x*s_x) / 4.0;
            double z12 = -tau_x * om_x*om_x * s_x*s_x / (2.0 * rel_p);
            double z22 = -(step_len + c_x*s_x) / (4.0 * rel_p * rel_p);
            double z33 = tau_y * om_y*om_y * (step_len - c_y*s_y) / 4.0;
            double z34 = -tau_y * om_y*om_y * s_y*s_y / (2.0 * rel_p);
            double z44 = -(step_len + c_y*s_y) / (4.0 * rel_p * rel_p);

            vx[i]  = c_x * r1 + s_x * r2 / rel_p + x_c;
            vpx[i] = tau_x * om_x*om_x * rel_p * s_x * r1 + c_x * r2;
            vy[i]  = c_y * r3 + s_y * r4 / rel_p;
            vpy[i] = tau_y * om_y*om_y * rel_p * s_y * r3 + c_y * r4;
            /* orientation*direction = 1 for forward tracking */
            vz[i] += z0 + z1*r1 + z2*r2 +
                     z11*r1*r1 + z12*r1*r2 + z22*r2*r2 +
                     z33*r3*r3 + z34*r3*r4 + z44*r4*r4;

            /* Low energy z correction for k1 map */
            vz[i] += low_energy_z_correction_dev(vpz[i], step_len, beta_val, beta_ref, mc2, e_tot_ele);

        /* ---- Branch 2: g=0 and dg=0 → pure drift ---- */
        } else if ((g == 0.0 && dg == 0.0) || step_len == 0.0) {
            double t_dummy = t_arr[i];
            {
                int rc = drift_body_dev(&vx[i], &vpx[i], &vy[i], &vpy[i], &vz[i], &vpz[i],
                                        &beta_val, &t_dummy, mc2, p0c_val, step_len);
                if (rc) { state[i] = rc; return; }
            }
            /* time handled at end (not from drift_body_dev) */

        /* ---- Branch 3: General bend (g != 0, b1 = 0) ---- */
        } else {
            double x  = vx[i];
            double px = vpx[i];
            double y  = vy[i];
            double py = vpy[i];
            double z  = vz[i];
            double rel_p2 = rel_p * rel_p;

            /* Linear approximation for near-axis particles */
            if (dg == 0.0 && fabs(x*g) < 1e-9 && fabs(px) < 1e-9 &&
                fabs(py) < 1e-9 && fabs(pz) < 1e-9) {
                double ll = step_len;
                double cos_a = cos(angle);
                double sin_a = sin(angle);
                double sinc_a = sinc_dev(angle);
                double cosc_a = cosc_dev(angle);
                double gam2 = mc2*mc2 / (rel_p2 * p0c_val*p0c_val + mc2*mc2);
                double m56 = ll * (gam2 - (g*ll)*(g*ll) * sincc_dev(angle));

                vx[i]  = cos_a * x      + ll*sinc_a * px + g*ll*ll*cosc_a * pz;
                vpx[i] = -g*sin_a * x   + cos_a * px     + g*ll*sinc_a * pz;
                vy[i]  = y + ll * py;
                vz[i]  = -g*ll*sinc_a*x - g*ll*ll*cosc_a*px + z + m56*pz;

            /* General nonlinear case */
            } else {
                double sinc_a = sinc_dev(angle);
                double pt = sqrt(rel_p2 - py*py);
                if (fabs(px) > pt) { state[i] = LOST_PZ; return; }
                double g_p = g_tot / pt;
                double phi_1 = asin(px / pt);
                double cos_a = cos(angle);
                double sin_a = sin(angle);
                double cosc_a = cosc_dev(angle);
                double cos_plus = cos(angle + phi_1);
                double sin_plus = sin(angle + phi_1);
                double alpha_b = 2.0*(1.0+g*x)*sin_plus*step_len*sinc_a -
                                 g_p*((1.0+g*x)*step_len*sinc_a)*((1.0+g*x)*step_len*sinc_a);
                double r_val = cos_plus*cos_plus + g_p*alpha_b;

                if (r_val < 0.0 || (fabs(g_p) < 1e-5 && fabs(cos_plus) < 1e-5)) {
                    state[i] = LOST_PZ;
                    return;
                }

                double rad = sqrt(r_val);
                double xi;
                if (cos_plus > 0.0) {
                    double denom = rad + cos_plus;
                    xi = alpha_b / denom;
                } else {
                    if (fabs(g_p) < 1e-30) { state[i] = LOST_PZ; return; }
                    xi = (rad - cos_plus) / g_p;
                }
                vx[i] = x*cos_a - step_len*step_len*g*cosc_a + xi;

                /* Check aperture limit */
                if (fabs(vx[i]) > 1.0) {
                    state[i] = LOST_PZ;
                    return;
                }

                double L_u = xi;
                double L_v = -(step_len*sinc_a + x*sin_a);  /* time_dir=1 */
                double L_c = sqrt(L_v*L_v + L_u*L_u);
                double angle_p = 2.0*(angle + phi_1 - atan2(L_u, -L_v));  /* time_dir=1 */
                double L_p = L_c / sinc_dev(angle_p*0.5);  /* time_dir=1 */
                vpx[i] = pt * sin(phi_1 + angle - angle_p);
                vy[i]  = y + py * L_p / pt;
                vz[i]  = z + beta_val * step_len / beta_ref - rel_p * L_p / pt;
            }
        }

        /* Multipole kick after each step */
        if (has_mag) {
            double kx, ky;
            multipole_kick_dev(d_a2, d_b2, ix_mag_max, d_cm,
                               vx[i], vy[i], &kx, &ky);
            double scl = (istep == n_step) ? 0.5 : 1.0;
            double f_gx = 1.0 + g * vx[i];
            vpx[i] += scl * kx * f_gx;
            vpy[i] += scl * ky * f_gx;
        }

        if (has_elec) {
            double kx, ky;
            multipole_kick_dev(d_ea2, d_eb2, ix_elec_max, d_cm,
                               vx[i], vy[i], &kx, &ky);
            double f_gx = 1.0 + g * vx[i];
            double rel_p0 = 1.0 + vpz[i];
            double ps = sqrt(rel_p0*rel_p0 - vpx[i]*vpx[i] - vpy[i]*vpy[i]) / rel_p0;
            double scl = (istep == n_step) ? 0.5 : 1.0;
            double f_scale = scl * f_gx / (ps * beta_val);
            if (apply_electric_kick_dev(kx * f_scale, ky * f_scale,
                    &vpx[i], &vpy[i], &vpz[i], &vz[i],
                    &beta_val, &beta_arr[i], mc2, p0c_val, &state[i])) return;
        }
    }

    /* Time update */
    t_arr[i] = t_start + delta_ref_time + (z_start - vz[i]) / (beta_val * C_LIGHT);
}

/* =========================================================================
 * HOST WRAPPER: gpu_track_bend
 * ========================================================================= */
extern "C" void gpu_track_bend_(
    double *h_vx, double *h_vpx, double *h_vy, double *h_vpy,
    double *h_vz, double *h_vpz,
    int *h_state, double *h_beta, double *h_p0c, double *h_t,
    double mc2, double g, double g_tot, double dg, double b1,
    double ele_length, double delta_ref_time, double e_tot_ele,
    double rel_charge_dir,
    double p0c_ele,
    int n_particles,
    double *h_a2, double *h_b2, double *h_cm,
    int ix_mag_max, int n_step,
    double *h_ea2, double *h_eb2, int ix_elec_max)
{
    if (ensure_buffers(n_particles) != 0) return;

    /* Host -> Device */
    if (upload_particle_data(n_particles, h_vx, h_vpx, h_vy, h_vpy, h_vz, h_vpz,
                             h_state, h_beta, h_p0c, h_t) != 0) return;
    if (upload_multipole_data(h_a2, h_b2, h_cm, h_ea2, h_eb2,
                              ix_mag_max, ix_elec_max) != 0) return;

    /* Launch kernel */
    int threads = 256;
    int blocks  = (n_particles + threads - 1) / threads;
    bend_kernel<<<blocks, threads>>>(
        d_vec[0], d_vec[1], d_vec[2], d_vec[3], d_vec[4], d_vec[5],
        d_state, d_beta, d_p0c, d_t,
        mc2, g, g_tot, dg, b1, ele_length, delta_ref_time, e_tot_ele,
        rel_charge_dir, p0c_ele,
        n_particles, d_a2, d_b2, d_cm, ix_mag_max, n_step,
        d_ea2, d_eb2, ix_elec_max);
    CUDA_CHECK_VOID(cudaGetLastError());
    CUDA_CHECK_VOID(cudaDeviceSynchronize());

    /* Device -> Host (electric kicks can modify beta) */
    if (download_particle_data(n_particles, h_vx, h_vpx, h_vy, h_vpy, h_vz, h_vpz,
                               h_state, h_beta, h_p0c, h_t,
                               (ix_elec_max >= 0), 0) != 0) return;
}

/* =========================================================================
 * LCAVITY KERNEL
 * Replicates track_a_lcavity stair-step RF approximation.
 * Each particle independently loops through drift + energy kick steps.
 * Handles ponderomotive transverse kicks for standing-wave cavities.
 * Forward tracking only, relative time tracking.
 * ========================================================================= */

#define TWOPI 6.283185307179586476925286766559
#define STANDING_WAVE 1
#define TRAVELING_WAVE 2

/* Device helper: dpc_given_dE -- momentum change from energy change.
 * Uses rationalization to avoid catastrophic cancellation for small dE. */
__device__ __forceinline__ double dpc_given_dE_dev(double pc_old, double mc2, double dE)
{
    double del2 = dE * dE + 2.0 * sqrt(pc_old * pc_old + mc2 * mc2) * dE;
    double pc_new = sqrt(pc_old * pc_old + del2);
    /* Rationalize: pc_new - pc_old = del2 / (pc_new + pc_old) to avoid cancellation */
    return del2 / (pc_new + pc_old);
}

/* Device helper: lcavity fringe kick (entrance or exit)
 * Matches track_a_lcavity.f90 fringe_kick subroutine (lines 232-301)
 * for GPU case: body_dir=1, time_dir=1.
 * edge: +1 for entrance, -1 for exit.
 */
__device__ void lcavity_fringe_kick_dev(
    double &x, double &px, double &y, double &py, double &z, double &pz,
    double &beta_val, double p0c, double mc2, double t,
    int edge, double gradient_tot, double charge_ratio,
    double rf_frequency, double phi0_total,
    int abs_time, double phi0_no_multi, double ref_time_start,
    double step_time_val)
{
    double particle_time, phase;
    if (abs_time) {
        double half_period = 0.5 / rf_frequency;
        double period = 1.0 / rf_frequency;
        double t_shifted = t - ref_time_start;
        double mod_val = fmod(t_shifted, period);
        if (mod_val < 0.0) mod_val += period;
        particle_time = mod_val;
        if (particle_time >= half_period) particle_time -= period;
        particle_time -= step_time_val;
        phase = TWOPI * (phi0_no_multi + particle_time * rf_frequency);
    } else {
        particle_time = -z / (beta_val * C_LIGHT);
        phase = TWOPI * (phi0_total + particle_time * rf_frequency);
    }

    double ez_field = gradient_tot * cos(phase);
    double rf_omega = TWOPI * rf_frequency / C_LIGHT;
    double dez_dz_field = gradient_tot * sin(phase) * rf_omega;

    double ff = edge * charge_ratio;
    double f = ff / p0c;

    double dE = -ff * 0.5 * dez_dz_field * (x * x + y * y);

    double pc = p0c * (1.0 + pz);
    double pz_end = pz + dpc_given_dE_dev(pc, mc2, dE) / p0c;

    /* to_energy_coords */
    z = z / beta_val;
    pz = (1.0 + pz) / beta_val;

    /* kicks in energy coords */
    px = px - f * ez_field * x;
    py = py - f * ez_field * y;
    pz = pz + dE / p0c;

    /* to_momentum_coords */
    double pc_new = (1.0 + pz_end) * p0c;
    double beta_new = pc_new / (p0c * pz);
    z = z * beta_new;
    pz = pz_end;
    beta_val = beta_new;
}

/* --------------------------------------------------------------------------
 * ponderomotive_kick_dev -- standing-wave ponderomotive transverse kick
 *
 * Applied symmetrically before and after each RF energy kick step.
 * -------------------------------------------------------------------------- */
__device__ void ponderomotive_kick_dev(
    double *px, double *py, double *z, double *t,
    double x, double y, double pz, double beta_val,
    double grad, double l_active, double p0c, int n_rf_steps)
{
    double rel_p = 1.0 + pz;
    double coef = grad * grad * l_active / (16.0 * p0c * p0c * rel_p * n_rf_steps);
    *px -= coef * x;
    *py -= coef * y;
    double dzp = -0.5 * coef * (x * x + y * y) / rel_p;
    *z += dzp;
    *t -= dzp / (C_LIGHT * beta_val);
}

__global__ void lcavity_kernel(
    double *vx, double *vpx, double *vy, double *vpy, double *vz, double *vpz,
    int *state, double *beta_arr, double *p0c_arr, double *t_arr,
    double mc2,
    /* Step data arrays: n_steps_total entries (indices 0..n_rf_steps+1) */
    const double *step_s0, const double *step_s,
    const double *step_p0c, const double *step_p1c,
    const double *step_scale, const double *step_time,
    int n_rf_steps,
    /* Element parameters */
    double voltage, double voltage_err, double field_autoscale,
    double rf_frequency,
    double phi0_total,      /* phi0 + phi0_err + phi0_multipass (precomputed) */
    double voltage_tot, double l_active,
    int cavity_type,        /* 1=standing_wave, 2=traveling_wave */
    int fringe_at,          /* 0=none, 1=entrance, 2=exit, 3=both */
    double charge_ratio,    /* charge_of(species) / (2 * charge_of(ref_species)) */
    int n_particles,
    int abs_time,           /* 1 if absolute_time_tracking */
    double phi0_no_multi,   /* phi0 + phi0_err (without phi0_multipass) */
    double ref_time_start)  /* lord%value(ref_time_start$) for abs time ref shift */
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n_particles) return;
    if (state[i] != ALIVE_ST) return;

    double x = vx[i], px = vpx[i], y = vy[i], py = vpy[i], z = vz[i], pz = vpz[i];
    double beta_val = beta_arr[i], p0c = p0c_arr[i], t = t_arr[i];

    double s_now = step_s0[0];
    int ix_step_end = n_rf_steps + 1;  /* phantom step index */

    /* Ponderomotive kicks only for standing wave + forward tracking */
    int do_ponderomotive = (cavity_type == STANDING_WAVE) && (l_active > 0.0);

    for (int ix = 0; ix <= ix_step_end; ix++) {
        /* ---- Drift to step boundary ---- */
        double ds = step_s[ix] - s_now;
        s_now = step_s[ix];

        if (ds != 0.0) {
            {
                int rc = drift_body_dev(&x, &px, &y, &py, &z, &pz,
                                        &beta_val, &t, mc2, p0c, ds);
                if (rc) { state[i] = rc; return; }
            }
        }

        /* ---- Entrance fringe: step 0, after drift, before energy kick ---- */
        if (ix == 0 && (fringe_at & 1) && l_active > 0.0) {
            double grad_tot = voltage_tot * field_autoscale / l_active;
            lcavity_fringe_kick_dev(x, px, y, py, z, pz, beta_val, p0c, mc2, t,
                +1, grad_tot, charge_ratio, rf_frequency, phi0_total,
                abs_time, phi0_no_multi, ref_time_start, step_time[0]);
        }

        /* ---- Stair-step kick (not at phantom step) ---- */
        if (ix != ix_step_end) {
            /* Upstream ponderomotive kick (skip at step 0) */
            if (do_ponderomotive && ix > 0) {
                double grad = field_autoscale * voltage_tot / l_active;
                ponderomotive_kick_dev(&px, &py, &z, &t, x, y, pz, beta_val,
                                       grad, l_active, p0c, n_rf_steps);
            }

            /* ---- Energy kick ---- */
            /* Compute RF phase */
            double particle_time, phase;
            if (abs_time) {
                /* Absolute time tracking: use particle t, no phi0_multipass.
                 * Must subtract ref_time_start (absolute_time_ref_shift).
                 * modulo2(t, half_period) maps t into [-half_period, half_period] */
                double half_period = 0.5 / rf_frequency;
                double period = 1.0 / rf_frequency;
                /* Fortran modulo2(x, amp) = modulo(x, 2*amp) - 2*amp if >= amp
                 * Fortran modulo(x, p) always returns [0, p) for p>0 */
                double t_shifted = t - ref_time_start;
                double mod_val = fmod(t_shifted, period);
                if (mod_val < 0.0) mod_val += period;  /* Fortran modulo: always [0, period) */
                particle_time = mod_val;
                if (particle_time >= half_period) particle_time -= period;
                /* Now particle_time is in [-half_period, half_period) */
                particle_time -= step_time[ix];
                phase = TWOPI * (phi0_no_multi + particle_time * rf_frequency);
            } else {
                /* Relative time tracking */
                particle_time = -z / (beta_val * C_LIGHT);
                phase = TWOPI * (phi0_total + particle_time * rf_frequency);
            }

            /* Energy change */
            double dE_amp = (voltage + voltage_err) * step_scale[ix] * field_autoscale;
            double dE = dE_amp * cos(phase);

            /* Compute pz_end before coordinate conversion (avoid round-off) */
            double rel_p = 1.0 + pz;
            double pc = rel_p * p0c;
            double pz_end = pz + dpc_given_dE_dev(pc, mc2, dE) / p0c;

            /* to_energy_coords: (z, pz) -> (c*(t0-t), E/p0c) */
            z = z / beta_val;
            pz = (1.0 + pz) / beta_val;  /* now pz = E/p0c in energy coords */

            /* Apply energy kick in energy coords */
            pz = pz + dE / p0c;

            /* to_momentum_coords: convert back */
            double pc_new = (1.0 + pz_end) * p0c;
            double beta_new = pc_new / (p0c * pz);  /* pz here is E/p0c */
            z = z * beta_new;
            pz = pz_end;
            beta_val = beta_new;

            /* orbit_reference_energy_correction */
            double p1c = step_p1c[ix];
            double p_rel = p0c / p1c;
            px = px * p_rel;
            py = py * p_rel;
            pz = (pz * p0c - (p1c - p0c)) / p1c;
            p0c = p1c;

            /* Update beta after reference energy change */
            pc_new = (1.0 + pz) * p0c;
            beta_val = pc_new / sqrt(pc_new * pc_new + mc2 * mc2);

            /* Downstream ponderomotive kick (skip at step n_rf_steps) */
            if (do_ponderomotive && ix < n_rf_steps) {
                double grad = field_autoscale * voltage_tot / l_active;
                ponderomotive_kick_dev(&px, &py, &z, &t, x, y, pz, beta_val,
                                       grad, l_active, p0c, n_rf_steps);
            }
        }

        /* ---- Exit fringe: step n_rf_steps, after energy kick ---- */
        if (ix == n_rf_steps && (fringe_at & 2) && l_active > 0.0) {
            double grad_tot = voltage_tot * field_autoscale / l_active;
            lcavity_fringe_kick_dev(x, px, y, py, z, pz, beta_val, p0c, mc2, t,
                -1, grad_tot, charge_ratio, rf_frequency, phi0_total,
                abs_time, phi0_no_multi, ref_time_start, step_time[n_rf_steps]);
        }
    }

    /* Write back */
    vx[i] = x;   vpx[i] = px;  vy[i] = y;   vpy[i] = py;
    vz[i] = z;   vpz[i] = pz;
    beta_arr[i] = beta_val;
    p0c_arr[i]  = p0c;
    t_arr[i]    = t;
}

/* =========================================================================
 * HOST WRAPPER: gpu_track_lcavity
 * ========================================================================= */

/* Step data device buffers */
static double *d_step_s0   = NULL;
static double *d_step_s    = NULL;
static double *d_step_p0c  = NULL;
static double *d_step_p1c  = NULL;
static double *d_step_scl  = NULL;
static double *d_step_time = NULL;
static int     d_step_cap  = 0;

static int ensure_step_buffers(int n_steps_total)
{
    if (n_steps_total <= d_step_cap) return 0;
    if (d_step_s0)   cudaFree(d_step_s0);
    if (d_step_s)    cudaFree(d_step_s);
    if (d_step_p0c)  cudaFree(d_step_p0c);
    if (d_step_p1c)  cudaFree(d_step_p1c);
    if (d_step_scl)  cudaFree(d_step_scl);
    if (d_step_time) cudaFree(d_step_time);
    d_step_s0 = d_step_s = d_step_p0c = d_step_p1c = d_step_scl = d_step_time = NULL;

    size_t sz = (size_t)n_steps_total * sizeof(double);
    if (cudaMalloc((void**)&d_step_s0,   sz) != cudaSuccess) goto sfail;
    if (cudaMalloc((void**)&d_step_s,    sz) != cudaSuccess) goto sfail;
    if (cudaMalloc((void**)&d_step_p0c,  sz) != cudaSuccess) goto sfail;
    if (cudaMalloc((void**)&d_step_p1c,  sz) != cudaSuccess) goto sfail;
    if (cudaMalloc((void**)&d_step_scl,  sz) != cudaSuccess) goto sfail;
    if (cudaMalloc((void**)&d_step_time, sz) != cudaSuccess) goto sfail;
    d_step_cap = n_steps_total;
    return 0;
sfail:
    fprintf(stderr, "[gpu_tracking] cudaMalloc failed for %d step buffers\n", n_steps_total);
    /* Clean up any partially allocated step buffers */
    if (d_step_s0)   cudaFree(d_step_s0);   d_step_s0   = NULL;
    if (d_step_s)    cudaFree(d_step_s);    d_step_s    = NULL;
    if (d_step_p0c)  cudaFree(d_step_p0c);  d_step_p0c  = NULL;
    if (d_step_p1c)  cudaFree(d_step_p1c);  d_step_p1c  = NULL;
    if (d_step_scl)  cudaFree(d_step_scl);  d_step_scl  = NULL;
    if (d_step_time) cudaFree(d_step_time); d_step_time = NULL;
    d_step_cap = 0;
    return -1;
}

extern "C" void gpu_track_lcavity_(
    double *h_vx, double *h_vpx, double *h_vy, double *h_vpy,
    double *h_vz, double *h_vpz,
    int *h_state, double *h_beta, double *h_p0c, double *h_t,
    double mc2,
    double *h_step_s0, double *h_step_s,
    double *h_step_p0c, double *h_step_p1c,
    double *h_step_scale, double *h_step_time,
    int n_rf_steps,
    double voltage, double voltage_err, double field_autoscale,
    double rf_frequency, double phi0_total,
    double voltage_tot, double l_active,
    int cavity_type,
    int fringe_at, double charge_ratio,
    int n_particles,
    int abs_time, double phi0_no_multi,
    double ref_time_start)
{
    if (ensure_buffers(n_particles) != 0) return;

    int n_steps_total = n_rf_steps + 2;  /* indices 0..n_rf_steps+1 */
    if (ensure_step_buffers(n_steps_total) != 0) return;

    size_t sb = (size_t)n_steps_total * sizeof(double);

    /* Host -> Device: particle data */
    if (upload_particle_data(n_particles, h_vx, h_vpx, h_vy, h_vpy, h_vz, h_vpz,
                             h_state, h_beta, h_p0c, h_t) != 0) return;

    /* Host -> Device: step data */
    CUDA_CHECK_VOID(cudaMemcpy(d_step_s0,   h_step_s0,    sb, cudaMemcpyHostToDevice));
    CUDA_CHECK_VOID(cudaMemcpy(d_step_s,    h_step_s,     sb, cudaMemcpyHostToDevice));
    CUDA_CHECK_VOID(cudaMemcpy(d_step_p0c,  h_step_p0c,   sb, cudaMemcpyHostToDevice));
    CUDA_CHECK_VOID(cudaMemcpy(d_step_p1c,  h_step_p1c,   sb, cudaMemcpyHostToDevice));
    CUDA_CHECK_VOID(cudaMemcpy(d_step_scl,  h_step_scale, sb, cudaMemcpyHostToDevice));
    CUDA_CHECK_VOID(cudaMemcpy(d_step_time, h_step_time,  sb, cudaMemcpyHostToDevice));

    /* Launch kernel */
    int threads = 256;
    int blocks  = (n_particles + threads - 1) / threads;
    lcavity_kernel<<<blocks, threads>>>(
        d_vec[0], d_vec[1], d_vec[2], d_vec[3], d_vec[4], d_vec[5],
        d_state, d_beta, d_p0c, d_t, mc2,
        d_step_s0, d_step_s, d_step_p0c, d_step_p1c, d_step_scl, d_step_time,
        n_rf_steps,
        voltage, voltage_err, field_autoscale,
        rf_frequency, phi0_total, voltage_tot, l_active,
        cavity_type,
        fringe_at, charge_ratio,
        n_particles,
        abs_time, phi0_no_multi, ref_time_start);
    CUDA_CHECK_VOID(cudaGetLastError());
    CUDA_CHECK_VOID(cudaDeviceSynchronize());

    /* Device -> Host (lcavity changes beta and p0c) */
    if (download_particle_data(n_particles, h_vx, h_vpx, h_vy, h_vpy, h_vz, h_vpz,
                               h_state, h_beta, h_p0c, h_t, 1, 1) != 0) return;
}

/* ==========================================================================
 * CURAND RNG STATE MANAGEMENT
 * ========================================================================== */

static curandState *d_rng_states = NULL;
static int d_rng_cap = 0;

__global__ void init_rng_kernel(curandState *states, unsigned long long seed, int n)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    curand_init(seed, (unsigned long long)i, 0, &states[i]);
}

static int ensure_rng(int n)
{
    if (n <= d_rng_cap) return 0;
    if (d_rng_states) cudaFree(d_rng_states);
    d_rng_states = NULL;
    if (cudaMalloc((void**)&d_rng_states, (size_t)n * sizeof(curandState)) != cudaSuccess) {
        fprintf(stderr, "[gpu_tracking] cudaMalloc failed for %d curandStates\n", n);
        d_rng_cap = 0;
        return -1;
    }
    d_rng_cap = n;
    /* Initialize RNG states with a fixed base seed */
    int threads = 256;
    int blocks = (n + threads - 1) / threads;
    init_rng_kernel<<<blocks, threads>>>(d_rng_states, 123456789ULL, n);
    cudaDeviceSynchronize();
    return 0;
}

/* ==========================================================================
 * RADIATION KICK KERNEL
 *
 * Applies radiation damping and/or stochastic fluctuation kicks to all
 * alive particles.  Called once for entrance (rm0) and once for exit (rm1).
 *
 * Fluctuation kick: vec += scale * stoc_mat * ran6
 * Damping kick:     vec += scale * (damp_dmat * (vec - ref_orb) + xfer_damp_vec)
 *
 * stoc_mat and damp_dmat are 6x6 column-major (Fortran order).
 * ========================================================================== */

__global__ void rad_kick_kernel(
    double *vx, double *vpx, double *vy, double *vpy, double *vz, double *vpz,
    int *state,
    const double *stoc_mat,       /* 6x6, device, column-major */
    const double *damp_dmat,      /* 6x6, device, column-major */
    const double *xfer_damp_vec,  /* 6, device */
    const double *ref_orb,        /* 6, device */
    double synch_rad_scale,
    int apply_damp, int apply_fluct, int zero_average,
    curandState *rng_states,
    int n)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    if (state[i] != ALIVE_ST) return;

    double vec[6] = {vx[i], vpx[i], vy[i], vpy[i], vz[i], vpz[i]};

    /* Damping: vec += rr * damp_dmat * (vec - ref_orb) [+ rr * xfer_damp_vec] */
    if (apply_damp) {
        double dv[6], kick[6];
        for (int k = 0; k < 6; k++) dv[k] = vec[k] - ref_orb[k];
        for (int k = 0; k < 6; k++) {
            double s = 0.0;
            for (int j = 0; j < 6; j++) s += damp_dmat[j * 6 + k] * dv[j];
            kick[k] = s;
        }
        double rr = synch_rad_scale;  /* time_dir = 1 for GPU tracking */
        for (int k = 0; k < 6; k++) vec[k] += rr * kick[k];
        if (!zero_average) {
            for (int k = 0; k < 6; k++) vec[k] += rr * xfer_damp_vec[k];
        }
    }

    /* Fluctuations: vec += scale * stoc_mat * ran6 */
    if (apply_fluct) {
        double ran[6];
        curandState local_state = rng_states[i];
        for (int k = 0; k < 6; k++)
            ran[k] = curand_normal_double(&local_state);
        rng_states[i] = local_state;

        for (int k = 0; k < 6; k++) {
            double s = 0.0;
            for (int j = 0; j < 6; j++) s += stoc_mat[j * 6 + k] * ran[j];
            vec[k] += synch_rad_scale * s;
        }
    }

    /* Write back */
    vx[i] = vec[0]; vpx[i] = vec[1]; vy[i] = vec[2];
    vpy[i] = vec[3]; vz[i] = vec[4]; vpz[i] = vec[5];

    /* Check for pz < -1 */
    if (vpz[i] < -1.0) state[i] = LOST_PZ;
}

/* ==========================================================================
 * SPLIT UPLOAD/DOWNLOAD WRAPPERS
 *
 * These allow the Fortran code to upload particle data once, run multiple
 * kernels (radiation + body), then download once -- avoiding redundant
 * host-device transfers.
 * ========================================================================== */

extern "C" void gpu_upload_particles_(
    double *h_vx, double *h_vpx, double *h_vy, double *h_vpy,
    double *h_vz, double *h_vpz,
    int *h_state, double *h_beta, double *h_p0c, double *h_t, int n)
{
    if (ensure_buffers(n) != 0) return;
    upload_particle_data(n, h_vx, h_vpx, h_vy, h_vpy, h_vz, h_vpz,
                         h_state, h_beta, h_p0c, h_t);
}

extern "C" void gpu_download_particles_(
    double *h_vx, double *h_vpx, double *h_vy, double *h_vpy,
    double *h_vz, double *h_vpz,
    int *h_state, double *h_beta, double *h_p0c, double *h_t,
    int n, int copy_beta, int copy_p0c)
{
    download_particle_data(n, h_vx, h_vpx, h_vy, h_vpy, h_vz, h_vpz,
                           h_state, h_beta, h_p0c, h_t, copy_beta, copy_p0c);
}

/* --------------------------------------------------------------------------
 * Radiation data device buffers (small, allocated once)
 * -------------------------------------------------------------------------- */
static double *d_stoc_mat = NULL;       /* 36 doubles */
static double *d_damp_dmat = NULL;      /* 36 doubles */
static double *d_xfer_damp_vec = NULL;  /* 6 doubles */
static double *d_ref_orb = NULL;        /* 6 doubles */

static int ensure_rad_buffers(void)
{
    size_t m6 = 36 * sizeof(double);
    size_t v6 = 6 * sizeof(double);
    if (!d_stoc_mat)       { if (cudaMalloc((void**)&d_stoc_mat, m6) != cudaSuccess) return -1; }
    if (!d_damp_dmat)      { if (cudaMalloc((void**)&d_damp_dmat, m6) != cudaSuccess) return -1; }
    if (!d_xfer_damp_vec)  { if (cudaMalloc((void**)&d_xfer_damp_vec, v6) != cudaSuccess) return -1; }
    if (!d_ref_orb)        { if (cudaMalloc((void**)&d_ref_orb, v6) != cudaSuccess) return -1; }
    return 0;
}

/* --------------------------------------------------------------------------
 * gpu_rad_kick -- apply radiation kick on device data (already uploaded)
 * -------------------------------------------------------------------------- */
extern "C" void gpu_rad_kick_(
    int n,
    double *h_stoc_mat,       /* 36 doubles (host) */
    double *h_damp_dmat,      /* 36 doubles (host) */
    double *h_xfer_damp_vec,  /* 6 doubles (host) */
    double *h_ref_orb,        /* 6 doubles (host) */
    double synch_rad_scale,
    int apply_damp, int apply_fluct, int zero_average)
{
    if (n <= 0) return;
    if (ensure_rng(n) != 0) return;
    if (ensure_rad_buffers() != 0) return;

    /* Upload radiation data */
    CUDA_CHECK_VOID(cudaMemcpy(d_stoc_mat, h_stoc_mat, 36*sizeof(double), cudaMemcpyHostToDevice));
    CUDA_CHECK_VOID(cudaMemcpy(d_damp_dmat, h_damp_dmat, 36*sizeof(double), cudaMemcpyHostToDevice));
    CUDA_CHECK_VOID(cudaMemcpy(d_xfer_damp_vec, h_xfer_damp_vec, 6*sizeof(double), cudaMemcpyHostToDevice));
    CUDA_CHECK_VOID(cudaMemcpy(d_ref_orb, h_ref_orb, 6*sizeof(double), cudaMemcpyHostToDevice));

    int threads = 256;
    int blocks = (n + threads - 1) / threads;
    rad_kick_kernel<<<blocks, threads>>>(
        d_vec[0], d_vec[1], d_vec[2], d_vec[3], d_vec[4], d_vec[5],
        d_state,
        d_stoc_mat, d_damp_dmat, d_xfer_damp_vec, d_ref_orb,
        synch_rad_scale,
        apply_damp, apply_fluct, zero_average,
        d_rng_states, n);
    CUDA_CHECK_VOID(cudaGetLastError());
    CUDA_CHECK_VOID(cudaDeviceSynchronize());
}

/* ==========================================================================
 * BODY-ONLY KERNEL WRAPPERS (data already on device)
 * ========================================================================== */

/* Drift body-only: also uploads/downloads s_pos array */
extern "C" void gpu_track_drift_dev_(
    double *h_s, double mc2, double length, int n)
{
    size_t db = (size_t)n * sizeof(double);
    CUDA_CHECK_VOID(cudaMemcpy(d_s, h_s, db, cudaMemcpyHostToDevice));

    int threads = 256;
    int blocks = (n + threads - 1) / threads;
    drift_kernel<<<blocks, threads>>>(
        d_vec[0], d_vec[1], d_vec[2], d_vec[3], d_vec[4], d_vec[5],
        d_state, d_beta, d_p0c, d_s, d_t, mc2, length, n);
    CUDA_CHECK_VOID(cudaGetLastError());
    CUDA_CHECK_VOID(cudaDeviceSynchronize());

    CUDA_CHECK_VOID(cudaMemcpy(h_s, d_s, db, cudaMemcpyDeviceToHost));
}

/* Drift body-only WITHOUT s-array transfer (for persistent/CSR path where
 * gpu_s_update is called separately). Saves 16 MB of PCIe per sub-step. */
extern "C" void gpu_track_drift_dev_no_s_(double mc2, double length, int n)
{
    int threads = 256;
    int blocks = (n + threads - 1) / threads;
    drift_kernel<<<blocks, threads>>>(
        d_vec[0], d_vec[1], d_vec[2], d_vec[3], d_vec[4], d_vec[5],
        d_state, d_beta, d_p0c, d_s, d_t, mc2, length, n);
    CUDA_CHECK_VOID(cudaGetLastError());
}

/* Quad body-only: uploads multipoles, runs kernel */
extern "C" void gpu_track_quad_dev_(
    double mc2, double b1, double ele_length,
    double delta_ref_time, double e_tot_ele, double charge_dir,
    int n_particles,
    double *h_a2, double *h_b2, double *h_cm,
    int ix_mag_max, int n_step,
    double *h_ea2, double *h_eb2, int ix_elec_max)
{
    if (upload_multipole_data(h_a2, h_b2, h_cm, h_ea2, h_eb2,
                              ix_mag_max, ix_elec_max) != 0) return;

    int threads = 256;
    int blocks = (n_particles + threads - 1) / threads;
    quad_kernel<<<blocks, threads>>>(
        d_vec[0], d_vec[1], d_vec[2], d_vec[3], d_vec[4], d_vec[5],
        d_state, d_beta, d_p0c, d_t,
        mc2, b1, ele_length, delta_ref_time, e_tot_ele, charge_dir,
        n_particles, d_a2, d_b2, d_cm, ix_mag_max, n_step,
        d_ea2, d_eb2, ix_elec_max);
    CUDA_CHECK_VOID(cudaGetLastError());
    /* sync removed -- persistent path launches next kernel on same stream */
}

/* Sextupole body-only: uploads multipoles, runs kernel */
extern "C" void gpu_track_sextupole_dev_(
    double mc2, double ele_length,
    double delta_ref_time, double e_tot_ele, double charge_dir,
    int n_particles,
    double *h_a2, double *h_b2, double *h_cm,
    int ix_mag_max, int n_step,
    double *h_ea2, double *h_eb2, int ix_elec_max)
{
    if (upload_multipole_data(h_a2, h_b2, h_cm, h_ea2, h_eb2,
                              ix_mag_max, ix_elec_max) != 0) return;

    int threads = 256;
    int blocks = (n_particles + threads - 1) / threads;
    sextupole_kernel<<<blocks, threads>>>(
        d_vec[0], d_vec[1], d_vec[2], d_vec[3], d_vec[4], d_vec[5],
        d_state, d_beta, d_p0c, d_t,
        mc2, ele_length, delta_ref_time, e_tot_ele, charge_dir,
        n_particles, d_a2, d_b2, d_cm, ix_mag_max, n_step,
        d_ea2, d_eb2, ix_elec_max);
    CUDA_CHECK_VOID(cudaGetLastError());
    /* sync removed -- persistent path launches next kernel on same stream */
}

/* Combined sextupole: upload, kernel, download */
extern "C" void gpu_track_sextupole_(
    double *h_vx, double *h_vpx, double *h_vy, double *h_vpy,
    double *h_vz, double *h_vpz,
    int *h_state, double *h_beta, double *h_p0c, double *h_t,
    double mc2, double ele_length,
    double delta_ref_time, double e_tot_ele, double charge_dir,
    int n_particles,
    double *h_a2, double *h_b2, double *h_cm,
    int ix_mag_max, int n_step,
    double *h_ea2, double *h_eb2, int ix_elec_max)
{
    if (ensure_buffers(n_particles) != 0) return;

    if (upload_particle_data(n_particles, h_vx, h_vpx, h_vy, h_vpy, h_vz, h_vpz,
                             h_state, h_beta, h_p0c, h_t) != 0) return;
    if (upload_multipole_data(h_a2, h_b2, h_cm, h_ea2, h_eb2,
                              ix_mag_max, ix_elec_max) != 0) return;

    int threads = 256;
    int blocks = (n_particles + threads - 1) / threads;
    sextupole_kernel<<<blocks, threads>>>(
        d_vec[0], d_vec[1], d_vec[2], d_vec[3], d_vec[4], d_vec[5],
        d_state, d_beta, d_p0c, d_t,
        mc2, ele_length, delta_ref_time, e_tot_ele, charge_dir,
        n_particles, d_a2, d_b2, d_cm, ix_mag_max, n_step,
        d_ea2, d_eb2, ix_elec_max);
    CUDA_CHECK_VOID(cudaGetLastError());
    /* sync removed -- persistent path launches next kernel on same stream */

    download_particle_data(n_particles, h_vx, h_vpx, h_vy, h_vpy, h_vz, h_vpz,
                           h_state, h_beta, h_p0c, h_t, 1, 1);
}

/* Bend body-only: uploads multipoles, runs kernel */
extern "C" void gpu_track_bend_dev_(
    double mc2, double g, double g_tot, double dg, double b1,
    double ele_length, double delta_ref_time, double e_tot_ele,
    double rel_charge_dir, double p0c_ele,
    int n_particles,
    double *h_a2, double *h_b2, double *h_cm,
    int ix_mag_max, int n_step,
    double *h_ea2, double *h_eb2, int ix_elec_max)
{
    if (upload_multipole_data(h_a2, h_b2, h_cm, h_ea2, h_eb2,
                              ix_mag_max, ix_elec_max) != 0) return;

    int threads = 256;
    int blocks = (n_particles + threads - 1) / threads;
    bend_kernel<<<blocks, threads>>>(
        d_vec[0], d_vec[1], d_vec[2], d_vec[3], d_vec[4], d_vec[5],
        d_state, d_beta, d_p0c, d_t,
        mc2, g, g_tot, dg, b1, ele_length, delta_ref_time, e_tot_ele,
        rel_charge_dir, p0c_ele,
        n_particles, d_a2, d_b2, d_cm, ix_mag_max, n_step,
        d_ea2, d_eb2, ix_elec_max);
    CUDA_CHECK_VOID(cudaGetLastError());
    /* sync removed -- persistent path launches next kernel on same stream */
}

/* Lcavity body-only: uploads step data, runs kernel */
extern "C" void gpu_track_lcavity_dev_(
    double mc2,
    double *h_step_s0, double *h_step_s,
    double *h_step_p0c, double *h_step_p1c,
    double *h_step_scale, double *h_step_time,
    int n_rf_steps,
    double voltage, double voltage_err, double field_autoscale,
    double rf_frequency, double phi0_total,
    double voltage_tot, double l_active,
    int cavity_type,
    int fringe_at, double charge_ratio,
    int n_particles,
    int abs_time, double phi0_no_multi,
    double ref_time_start)
{
    int n_steps_total = n_rf_steps + 2;
    if (ensure_step_buffers(n_steps_total) != 0) return;

    size_t sb = (size_t)n_steps_total * sizeof(double);
    CUDA_CHECK_VOID(cudaMemcpy(d_step_s0,   h_step_s0,    sb, cudaMemcpyHostToDevice));
    CUDA_CHECK_VOID(cudaMemcpy(d_step_s,    h_step_s,     sb, cudaMemcpyHostToDevice));
    CUDA_CHECK_VOID(cudaMemcpy(d_step_p0c,  h_step_p0c,   sb, cudaMemcpyHostToDevice));
    CUDA_CHECK_VOID(cudaMemcpy(d_step_p1c,  h_step_p1c,   sb, cudaMemcpyHostToDevice));
    CUDA_CHECK_VOID(cudaMemcpy(d_step_scl,  h_step_scale, sb, cudaMemcpyHostToDevice));
    CUDA_CHECK_VOID(cudaMemcpy(d_step_time, h_step_time,  sb, cudaMemcpyHostToDevice));

    int threads = 256;
    int blocks = (n_particles + threads - 1) / threads;
    lcavity_kernel<<<blocks, threads>>>(
        d_vec[0], d_vec[1], d_vec[2], d_vec[3], d_vec[4], d_vec[5],
        d_state, d_beta, d_p0c, d_t, mc2,
        d_step_s0, d_step_s, d_step_p0c, d_step_p1c, d_step_scl, d_step_time,
        n_rf_steps,
        voltage, voltage_err, field_autoscale,
        rf_frequency, phi0_total, voltage_tot, l_active,
        cavity_type,
        fringe_at, charge_ratio,
        n_particles,
        abs_time, phi0_no_multi, ref_time_start);
    CUDA_CHECK_VOID(cudaGetLastError());
    CUDA_CHECK_VOID(cudaDeviceSynchronize());
}

/* ==========================================================================
 * QUADRUPOLE FRINGE KERNELS
 *
 * Implements soft_quadrupole_edge_kick and hard_multipole_edge_kick
 * for quadrupoles directly on the GPU.
 *
 * Fringe types: none$=0, soft_edge_only$=2, hard_edge_only$=3, full$=4
 * Edge: first_track_edge$=1, second_track_edge$=2
 * ========================================================================== */

#define FRINGE_NONE       0
#define FRINGE_SOFT_ONLY  2
#define FRINGE_HARD_ONLY  3
#define FRINGE_FULL       4
#define FIRST_EDGE  11  /* first_track_edge$ */
#define SECOND_EDGE 12  /* second_track_edge$ */

/* Soft quadrupole edge kick (SAD linear model) */
__device__ void soft_quad_edge_kick_dev(
    double *x, double *px, double *y, double *py, double *z,
    double rel_p, double k1_rel, double fq1, double fq2,
    int particle_at, int time_dir)
{
    double f1 = k1_rel * fq1;
    double f2 = k1_rel * fq2;

    if (particle_at == SECOND_EDGE) f1 = -f1;
    if (time_dir == -1) f2 = -f2;

    double ef1 = exp(f1);
    double vx = *px / rel_p;
    double vy = *py / rel_p;

    *z += -(f1 * (*x) + f2 * (1.0 + f1*0.5) * vx / ef1) * vx
          +(f1 * (*y) + f2 * (1.0 - f1*0.5) * vy * ef1) * vy;

    *x = (*x) * ef1 + vx * f2;
    *y = (*y) / ef1 - vy * f2;
    *px = (*px) / ef1;
    *py = (*py) * ef1;
}

/* Hard multipole edge kick for quadrupole (n_max=1 specialization).
 * Uses complex arithmetic with (real, imag) pairs. */
__device__ void hard_quad_edge_kick_dev(
    double *x, double *px, double *y, double *py, double *z,
    double rel_p, double k1, double charge_dir, int particle_at)
{
    double cab = charge_dir * k1 / (12.0 * rel_p);
    if (particle_at == FIRST_EDGE) cab = -cab;

    double xv = *x, yv = *y;

    /* poly = (x+iy)^2 = (x^2-y^2, 2xy) */
    double poly_r = xv*xv - yv*yv, poly_i = 2.0*xv*yv;
    /* dpoly/dx = 2*(x+iy) */
    double dpx_r = 2.0*xv, dpx_i = 2.0*yv;
    /* dpoly/dy = i * dpoly/dx = (-2y, 2x) */
    double dpy_r = -2.0*yv, dpy_i = 2.0*xv;
    /* d2poly/dxx = 2, d2poly/dxy = 2i, d2poly/dyy = -2 */
    double cn = 2.0;

    /* fx: xny = (x, -2y), poly*xny, cab*real(...) */
    double xny_r = xv, xny_i = -cn*yv;
    double fx = cab * (poly_r*xny_r - poly_i*xny_i);

    /* dfx/dx: cab*real(dpoly_dx*xny + poly) */
    double dfx_dx = cab * (dpx_r*xny_r - dpx_i*xny_i + poly_r);

    /* dfx/dy: cab*real(dpoly_dy*xny + poly*(0,-cn)) */
    double dfx_dy = cab * (dpy_r*xny_r - dpy_i*xny_i + cn*poly_i);

    /* d2fx/dxx: cab*real(2*xny + 2*dpoly_dx) */
    double d2fx_dxx = cab * (2.0*xny_r + 2.0*dpx_r);
    /* d2fx/dxy: cab*real(2i*xny + dpoly_dx*(0,-cn) + dpoly_dy) */
    double d2fx_dxy = cab * (-2.0*xny_i + cn*dpx_i + dpy_r);
    /* d2fx/dyy: cab*real(-2*xny + 2*dpoly_dy*(0,-cn)) */
    double d2fx_dyy = cab * (-2.0*xny_r + 2.0*cn*dpy_i);

    /* fy: xny2 = (y, cn*x), dxny2_dx = (0, cn) */
    double xny2_r = yv, xny2_i = cn*xv;
    double fy = cab * (poly_r*xny2_r - poly_i*xny2_i);

    /* dfy/dx: cab*real(dpoly_dx*xny2 + poly*(0,cn)) */
    double dfy_dx = cab * (dpx_r*xny2_r - dpx_i*xny2_i - cn*poly_i);
    /* dfy/dy: cab*real(dpoly_dy*xny2 + poly) */
    double dfy_dy = cab * (dpy_r*xny2_r - dpy_i*xny2_i + poly_r);

    /* d2fy/dxy: cab*real(2i*xny2 + dpoly_dx + dpoly_dy*(0,cn)) */
    double d2fy_dxy = cab * (-2.0*xny2_i + dpx_r - cn*dpy_i);

    /* Denominator */
    double denom = (1.0 - dfx_dx) * (1.0 - dfy_dy) - dfx_dy * dfy_dx;

    double pxv = *px, pyv = *py;

    *x = xv - fx;
    *y = yv - fy;
    *px = pxv + ((1.0 - dfy_dy - denom) * pxv + dfy_dx * pyv) / denom;
    *py = pyv + (dfx_dy * pxv + (1.0 - dfx_dx - denom) * pyv) / denom;
    *z = *z + ((*px) * fx + (*py) * fy) / rel_p;
}

/* Combined quad fringe kernel */
__global__ void quad_fringe_kernel(
    double *vx, double *vpx, double *vy, double *vpy, double *vz, double *vpz,
    int *state,
    double k1, double fq1, double fq2, double charge_dir,
    int fringe_type, int edge, int time_dir, int n)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    if (state[i] != ALIVE_ST) return;

    double rel_p = 1.0 + vpz[i];
    double k1_rel = charge_dir * k1 / rel_p;

    int do_hard = (fringe_type == FRINGE_HARD_ONLY || fringe_type == FRINGE_FULL);
    int do_soft = (fringe_type == FRINGE_SOFT_ONLY || fringe_type == FRINGE_FULL);

    if (edge == FIRST_EDGE) {
        if (do_hard)
            hard_quad_edge_kick_dev(&vx[i], &vpx[i], &vy[i], &vpy[i], &vz[i],
                                    rel_p, k1, charge_dir, edge);
        if (do_soft)
            soft_quad_edge_kick_dev(&vx[i], &vpx[i], &vy[i], &vpy[i], &vz[i],
                                    rel_p, k1_rel, fq1, fq2, edge, time_dir);
    } else {
        if (do_soft)
            soft_quad_edge_kick_dev(&vx[i], &vpx[i], &vy[i], &vpy[i], &vz[i],
                                    rel_p, k1_rel, fq1, fq2, edge, time_dir);
        if (do_hard)
            hard_quad_edge_kick_dev(&vx[i], &vpx[i], &vy[i], &vpy[i], &vz[i],
                                    rel_p, k1, charge_dir, edge);
    }
}

extern "C" void gpu_quad_fringe_(
    double k1, double fq1, double fq2, double charge_dir,
    int fringe_type, int edge, int time_dir, int n)
{
    if (n <= 0 || fringe_type == FRINGE_NONE) return;
    int threads = 256;
    int blocks = (n + threads - 1) / threads;
    quad_fringe_kernel<<<blocks, threads>>>(
        d_vec[0], d_vec[1], d_vec[2], d_vec[3], d_vec[4], d_vec[5],
        d_state,
        k1, fq1, fq2, charge_dir,
        fringe_type, edge, time_dir, n);
    CUDA_CHECK_VOID(cudaGetLastError());
    CUDA_CHECK_VOID(cudaDeviceSynchronize());
}

/* ==========================================================================
 * MISALIGNMENT KERNEL -- applies offset_particle on GPU
 *
 * Handles x_offset, y_offset, and tilt for non-bend elements.
 * set_flag: 1 = set (lab→body), -1 = unset (body→lab)
 * For set: subtract offsets, rotate by -tilt
 * For unset: rotate by +tilt, add offsets
 * ========================================================================== */

__global__ void misalign_kernel(
    double *vx, double *vpx, double *vy, double *vpy,
    int *state,
    double x_off, double y_off, double tilt,
    int set_flag, int n)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    if (state[i] != ALIVE_ST) return;

    double x = vx[i], px = vpx[i], y = vy[i], py = vpy[i];

    if (set_flag == 1) {
        /* set: lab → body: subtract offset, then rotate by -tilt */
        x -= x_off;
        y -= y_off;
        if (tilt != 0.0) {
            double ct = cos(-tilt), st = sin(-tilt);
            double xn  = ct * x  - st * y;
            double yn  = st * x  + ct * y;
            double pxn = ct * px - st * py;
            double pyn = st * px + ct * py;
            x = xn; y = yn; px = pxn; py = pyn;
        }
    } else {
        /* unset: body → lab: rotate by +tilt, then add offset */
        if (tilt != 0.0) {
            double ct = cos(tilt), st = sin(tilt);
            double xn  = ct * x  - st * y;
            double yn  = st * x  + ct * y;
            double pxn = ct * px - st * py;
            double pyn = st * px + ct * py;
            x = xn; y = yn; px = pxn; py = pyn;
        }
        x += x_off;
        y += y_off;
    }

    vx[i] = x; vpx[i] = px; vy[i] = y; vpy[i] = py;
}

/* ==========================================================================
 * BEND FRINGE (Hwang) -- GPU port of hwang_bend_edge_kick
 *
 * Per-particle coordinate transform at bend entrance/exit edge.
 * Parameters: g_tot, e_angle (e1 or e2), fint_gap, k1, time_dir.
 * entering=1 for entrance, 0 for exit.
 * ========================================================================== */

__global__ void bend_fringe_kernel(
    double *vx, double *vpx, double *vy, double *vpy, double *vz,
    const double *vpz, int *state,
    double g_tot, double e_angle, double fint_gap, double k1,
    int entering, int time_dir, int n)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    if (state[i] != ALIVE_ST) return;

    double td = (double)time_dir;
    double e_factor = 1.0 / (1.0 + vpz[i]);
    double cos_e = cos(e_angle), sin_e = sin(e_angle);
    double tan_e = sin_e / cos_e, sec_e = 1.0 / cos_e;
    double gt = g_tot * tan_e;
    double gt2 = g_tot * tan_e * tan_e;
    double gs2 = g_tot * sec_e * sec_e;
    double k1_tane = k1 * tan_e;
    double fg_factor = 2.0 * fint_gap * gs2 * g_tot * sec_e * (1.0 + sin_e * sin_e);

    double v0 = vx[i], v1 = vpx[i], v2 = vy[i], v3 = vpy[i], v4 = vz[i];
    double dx, dpx, dy, dpy, dz;
    double w0 = v0, w2 = v2;

    if (entering) {
        dx  = (-gt2 * v0*v0 + gs2 * v2*v2) * e_factor / 2.0;
        dpx = (gt * g_tot * (1.0 + 2.0*tan_e*tan_e) * v2*v2 / 2.0 + gt2 * (v0*v1 - v2*v3) + k1_tane * (v0*v0 - v2*v2)) * e_factor;
        dy  = gt2 * v0 * v2 * e_factor;
        dpy = (fg_factor * v2 - gt2 * v0 * v3 - (g_tot + gt2) * v1 * v2 - 2.0 * k1_tane * v0 * v2) * e_factor;
        dz  = e_factor * e_factor * 0.5 * (v2*v2 * fg_factor
              + v0*v0*v0 * (4.0*k1_tane - gt*gt2) / 6.0 + 0.5*v0*v2*v2 * (-4.0*k1_tane + gt*gs2)
              + (v0*v0*v1 - 2.0*v0*v2*v3) * gt2 - v1*v2*v2 * gs2);

        if (td == -1.0) {
            w0 = v0 + td * dx;
            double w1 = v1 + td * (dpx + gt * v0);
            w2 = v2 + td * dy;
            double w3 = v3 + td * (dpy - gt * v2);
            dx  = (-gt2 * w0*w0 + gs2 * w2*w2) * e_factor / 2.0;
            dpx = (gt * g_tot * (1.0 + 2.0*tan_e*tan_e) * w2*w2 / 2.0 + gt2 * (w0*w1 - w2*w3) + k1_tane * (w0*w0 - w2*w2)) * e_factor;
            dy  = gt2 * w0 * w2 * e_factor;
            dpy = (fg_factor * w2 - gt2 * w0 * w3 - (g_tot + gt2) * w1 * w2 - 2.0 * k1_tane * w0 * w2) * e_factor;
            dz  = e_factor * e_factor * 0.5 * (w2*w2 * fg_factor
                  + w0*w0*w0 * (4.0*k1_tane - gt*gt2) / 6.0 + 0.5*w0*w2*w2 * (-4.0*k1_tane + gt*gs2)
                  + (w0*w0*w1 - 2.0*w0*w2*w3) * gt2 - w1*w2*w2 * gs2);
        }

        vx[i]  = v0 + td * dx;
        vpx[i] = v1 + td * (dpx + gt * w0);
        vy[i]  = v2 + td * dy;
        vpy[i] = v3 + td * (dpy - gt * w2);
        vz[i]  = v4 + td * dz;
    } else {
        dx  = (gt2 * v0*v0 - gs2 * v2*v2) * e_factor / 2.0;
        dpx = (gt2 * (v2*v3 - v0*v1) + k1_tane * (v0*v0 - v2*v2) - gt * gt2 * (v0*v0 + v2*v2) / 2.0) * e_factor;
        dy  = -gt2 * v0 * v2 * e_factor;
        dpy = (fg_factor * v2 + gt2 * v0 * v3 + (g_tot + gt2) * v1 * v2 + (gt * gs2 - 2.0 * k1_tane) * v0 * v2) * e_factor;
        dz  = e_factor * e_factor * 0.5 * (v2*v2 * fg_factor
              + v0*v0*v0 * (4.0*k1_tane - gt*gt2) / 6.0 + 0.5*v0*v2*v2 * (-4.0*k1_tane + gt*gs2)
              - (v0*v0*v1 - 2.0*v0*v2*v3) * gt2 + v1*v2*v2 * gs2);

        if (td == -1.0) {
            w0 = v0 + td * dx;
            double w1 = v1 + td * (dpx + gt * w0);
            w2 = v2 + td * dy;
            double w3 = v3 + td * (dpy - gt * w2);
            dx  = (gt2 * w0*w0 - gs2 * w2*w2) * e_factor / 2.0;
            dpx = (gt2 * (w2*w3 - w0*w1) + k1_tane * (w0*w0 - w2*w2) - gt * gt2 * (w0*w0 + w2*w2) / 2.0) * e_factor;
            dy  = -gt2 * w0 * w2 * e_factor;
            dpy = (fg_factor * w2 + gt2 * w0 * w3 + (g_tot + gt2) * w1 * w2 + (gt * gs2 - 2.0 * k1_tane) * w0 * w2) * e_factor;
            dz  = e_factor * e_factor * 0.5 * (w2*w2 * fg_factor
                  + w0*w0*w0 * (4.0*k1_tane - gt*gt2) / 6.0 + 0.5*w0*w2*w2 * (-4.0*k1_tane + gt*gs2)
                  - (w0*w0*w1 - 2.0*w0*w2*w3) * gt2 + w1*w2*w2 * gs2);
        }

        vx[i]  = v0 + td * dx;
        vpx[i] = v1 + td * (dpx + gt * w0);
        vy[i]  = v2 + td * dy;
        vpy[i] = v3 + td * (dpy - gt * w2);
        vz[i]  = v4 + td * dz;
    }
}

extern "C" void gpu_bend_fringe_(
    double g_tot, double e_angle, double fint_gap, double k1,
    int entering, int time_dir, int n)
{
    if (n <= 0) return;
    int threads = 256;
    int blocks = (n + threads - 1) / threads;
    bend_fringe_kernel<<<blocks, threads>>>(
        d_vec[0], d_vec[1], d_vec[2], d_vec[3], d_vec[4],
        d_vec[5], d_state,
        g_tot, e_angle, fint_gap, k1,
        entering, time_dir, n);
    CUDA_CHECK_VOID(cudaGetLastError());
}


/* ==========================================================================
 * EXACT BEND FRINGE (PTC-style) -- GPU port of exact_bend_edge_kick
 * Tracking only (no transfer matrix).
 * ========================================================================== */

__device__ void vec_bmad_to_ptc_dev(double *v, double beta0) {
    double pz = v[5], z = v[4];
    double ib = 1.0/beta0;
    double ptc5 = (pz*pz + 2*pz) / (ib + sqrt(ib*ib + pz*pz + 2*pz));
    v[4] = ptc5;
    v[5] = -z * (ib + ptc5) / (1.0 + pz);
}

__device__ int vec_ptc_to_bmad_dev(double *v, double beta0) {
    double ptc5 = v[4], ptc6 = v[5];
    double pz = sqrt(1.0 + ptc5*(2.0/beta0 + ptc5)) - 1.0;
    double z = -ptc6 * (1.0 + pz) / (1.0/beta0 + ptc5);
    v[4] = z;
    v[5] = pz;
    return 0;
}

__device__ int ptc_rot_xz_dev(double a, double *X, double beta0) {
    double ib = 1.0/beta0;
    double arg = 1.0 + 2.0*X[4]*ib + X[4]*X[4] - X[1]*X[1] - X[3]*X[3];
    if (arg < 0) return 1;
    double pz = sqrt(arg);
    double pt = 1.0 - X[1]*tan(a)/pz;
    double x1 = X[0];
    X[0] = x1 / (cos(a) * pt);
    X[1] = X[1]*cos(a) + sin(a)*pz;
    X[2] = X[2] + X[3]*x1*tan(a)/(pz*pt);
    X[5] = X[5] + x1*tan(a)/(pz*pt) * (ib + X[4]);
    return 0;
}

__device__ int ptc_wedger_dev(double a, double g_tot, double beta0, double *X) {
    if (g_tot == 0.0) return ptc_rot_xz_dev(a, X, beta0);
    double ib = 1.0/beta0;
    double b1 = g_tot;
    double fac = 1.0 + 2.0*X[4]*ib + X[4]*X[4] - X[1]*X[1] - X[3]*X[3];
    if (fac < 0) return 1;
    double radix = 1.0 + 2.0*X[4]*ib + X[4]*X[4] - X[3]*X[3];
    if (radix < 1e-10) return 1;
    double pz = sqrt(fac), pt = sqrt(radix);
    double Xn2 = X[1]*cos(a) + (pz - b1*X[0])*sin(a);
    radix = 1.0 + 2.0*X[4]*ib + X[4]*X[4] - Xn2*Xn2 - X[3]*X[3];
    if (radix < 1e-10) return 1;
    double pzs = sqrt(radix);
    double denom = pzs + pz*cos(a) - X[1]*sin(a);
    double Xn1 = X[0]*cos(a) + (X[0]*X[1]*sin(2.0*a) +
        sin(a)*sin(a)*(2.0*X[0]*pz - b1*X[0]*X[0])) / denom;
    double Xn3_delta = (a + asin(X[1]/pt) - asin(Xn2/pt)) / b1;
    X[5] = X[5] + Xn3_delta*(ib + X[4]);
    X[2] = X[2] + X[3]*Xn3_delta;
    X[0] = Xn1; X[1] = Xn2;
    return 0;
}

__device__ int ptc_fringe_dipoler_dev(double *X, double g_tot, double beta0,
    double fint_signed, double hgap, int is_exit) {
    double B = is_exit ? -g_tot : g_tot;
    double ib = 1.0/beta0;
    double fac = 1.0 + 2.0*X[4]*ib + X[4]*X[4] - X[1]*X[1] - X[3]*X[3];
    if (fac <= 0) return 1;
    double pz = sqrt(fac);
    double xp = X[1]/pz, yp = X[3]/pz;
    double d12 = xp*yp/pz, d22 = (1.0+yp*yp)/pz, d32 = -yp;
    double d11 = (1.0+xp*xp)/pz, d21 = xp*yp/pz, d31 = -xp;
    double time_fac = ib + X[4];
    double d13 = -time_fac*xp/(pz*pz), d23 = -time_fac*yp/(pz*pz), d33 = time_fac/pz;
    double fi0 = atan(xp/(1.0+yp*yp)) - B*fint_signed*hgap*2.0*(1.0+xp*xp*(2.0+yp*yp))*pz;
    double co2 = B / (cos(fi0)*cos(fi0));
    double co1 = co2 / (1.0 + (xp/(1.0+yp*yp))*(xp/(1.0+yp*yp)));
    double fi1 = co1/(1.0+yp*yp) - co2*B*fint_signed*hgap*2.0*(2.0*xp*(2.0+yp*yp)*pz);
    double fi2 = -co1*2.0*xp*yp/((1.0+yp*yp)*(1.0+yp*yp)) - co2*B*fint_signed*hgap*2.0*(2.0*xp*xp*yp)*pz;
    double fi3 = -co2*B*fint_signed*hgap*2.0*(1.0+xp*xp*(2.0+yp*yp));
    double fi0t = B*tan(fi0);
    double bb = fi1*d12 + fi2*d22 + fi3*d32;
    fac = 1.0 - 2.0*bb*X[2];
    if (fac < 0) return 1;
    X[2] = 2.0*X[2] / (1.0 + sqrt(fac));
    X[3] = X[3] - fi0t*X[2];
    bb = fi1*d11 + fi2*d21 + fi3*d31;
    X[0] = X[0] + 0.5*bb*X[2]*X[2];
    bb = fi1*d13 + fi2*d23 + fi3*d33;
    X[5] = X[5] - 0.5*bb*X[2]*X[2];
    return 0;
}

__global__ void exact_bend_fringe_kernel(
    double *vx, double *vpx, double *vy, double *vpy, double *vz, double *vpz,
    int *state,
    double g_tot, double beta0,
    double edge_angle, double fint_signed, double hgap,
    int is_exit, int n)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    if (state[i] != ALIVE_ST) return;
    double X[6] = {vx[i], vpx[i], vy[i], vpy[i], vz[i], vpz[i]};
    vec_bmad_to_ptc_dev(X, beta0);
    if (!is_exit) {
        if (ptc_wedger_dev(edge_angle, 0.0, beta0, X)) { state[i] = LOST_PZ; return; }
        if (ptc_fringe_dipoler_dev(X, g_tot, beta0, fint_signed, hgap, 0)) { state[i] = LOST_PZ; return; }
        if (ptc_wedger_dev(-edge_angle, g_tot, beta0, X)) { state[i] = LOST_PZ; return; }
    } else {
        if (ptc_wedger_dev(-edge_angle, g_tot, beta0, X)) { state[i] = LOST_PZ; return; }
        if (ptc_fringe_dipoler_dev(X, g_tot, beta0, fint_signed, hgap, 1)) { state[i] = LOST_PZ; return; }
        if (ptc_wedger_dev(edge_angle, 0.0, beta0, X)) { state[i] = LOST_PZ; return; }
    }
    if (vec_ptc_to_bmad_dev(X, beta0)) { state[i] = LOST_PZ; return; }
    vx[i] = X[0]; vpx[i] = X[1]; vy[i] = X[2]; vpy[i] = X[3];
    vz[i] = X[4]; vpz[i] = X[5];
}

/* Hard multipole edge kick for bends (n_max=1, k1 only).
 * Port of hard_multipole_edge_kick for sbend$ with fringe_type=full$.
 * Uses complex polynomial: poly = (x+iy)^2, cab = charge_dir*k1/(12*rel_p) */
__global__ void hard_bend_edge_kernel(
    double *vx, double *vpx, double *vy, double *vpy, double *vz, double *vpz,
    int *state, double k1, double charge_dir, int is_entrance, int n)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    if (state[i] != ALIVE_ST) return;

    double x = vx[i], y = vy[i], px = vpx[i], py = vpy[i];
    double rel_p = 1.0 + vpz[i];
    if (rel_p <= 0) return;

    /* cab = charge_dir * k1 / (4 * (n+2) * rel_p) with n=1 -> / (12 * rel_p) */
    double cab = charge_dir * k1 / (12.0 * rel_p);
    if (is_entrance) cab = -cab;

    /* poly = (x+iy)^2 = (x^2 - y^2) + i(2xy) */
    double pr = x*x - y*y, pi_val = 2.0*x*y;

    /* cn = (n+3)/(n+1) = 4/2 = 2 for n=1 */
    double cn = 2.0;

    /* fx = real(cab * poly * (x - cn*i*y)) */
    /* (x - cn*i*y) = x - 2iy */
    double xny_r = x, xny_i = -cn*y;
    /* poly * xny = (pr + i*pi) * (xny_r + i*xny_i) */
    double prod_r = pr*xny_r - pi_val*xny_i;
    /* double prod_i = pr*xny_i + pi_val*xny_r;  // not needed for fx */
    double fx = cab * prod_r;

    /* fy = real(cab * poly * (y + cn*i*x)) */
    double xny2_r = y, xny2_i = cn*x;
    double prod2_r = pr*xny2_r - pi_val*xny2_i;
    double fy = cab * prod2_r;

    /* Derivatives for the symplectic kick */
    /* dpoly_dx = 2*(x+iy) = 2x + 2iy */
    double dpr_dx = 2*x, dpi_dx = 2*y;
    double dpr_dy = -2*y, dpi_dy = 2*x;

    /* dfx_dx = real(cab * (dpoly_dx * xny + poly)) */
    double dfx_dx = cab * (dpr_dx*xny_r - dpi_dx*xny_i + pr);
    /* dfx_dy = real(cab * (dpoly_dy * xny + poly * dxny_dy)) where dxny_dy = -cn*i */
    double dfx_dy = cab * (dpr_dy*xny_r - dpi_dy*xny_i + pr*0.0 - pi_val*(-cn));
    /* dfy_dx = real(cab * (dpoly_dx * xny2 + poly * dxny2_dx)) where dxny2_dx = cn*i */
    double dfy_dx = cab * (dpr_dx*xny2_r - dpi_dx*xny2_i + pr*0.0 - pi_val*(cn));
    /* dfy_dy = real(cab * (dpoly_dy * xny2 + poly)) */
    double dfy_dy = cab * (dpr_dy*xny2_r - dpi_dy*xny2_i + pr);

    double denom = (1.0 - dfx_dx) * (1.0 - dfy_dy) - dfx_dy * dfy_dx;
    if (fabs(denom) < 1e-30) return;

    vx[i] = x - fx;
    vpx[i] = px + ((1.0 - dfy_dy - denom) * px + dfy_dx * py) / denom;
    vy[i] = y - fy;
    vpy[i] = py + (dfx_dy * px + (1.0 - dfx_dx - denom) * py) / denom;
    vz[i] = vz[i] + (vpx[i] * fx + vpy[i] * fy) / rel_p;
}

extern "C" void gpu_hard_bend_edge_(double k1, double charge_dir, int is_entrance, int n)
{
    if (n <= 0) return;
    int threads = 256;
    int blocks = (n + threads - 1) / threads;
    hard_bend_edge_kernel<<<blocks, threads>>>(
        d_vec[0], d_vec[1], d_vec[2], d_vec[3], d_vec[4], d_vec[5],
        d_state, k1, charge_dir, is_entrance, n);
    CUDA_CHECK_VOID(cudaGetLastError());
}


extern "C" void gpu_exact_bend_fringe_(
    double g_tot, double beta0,
    double edge_angle, double fint_signed, double hgap,
    int is_exit, int n)
{
    if (n <= 0) return;
    int threads = 256;
    int blocks = (n + threads - 1) / threads;
    exact_bend_fringe_kernel<<<blocks, threads>>>(
        d_vec[0], d_vec[1], d_vec[2], d_vec[3], d_vec[4], d_vec[5], d_state,
        g_tot, beta0, edge_angle, fint_signed, hgap, is_exit, n);
    CUDA_CHECK_VOID(cudaGetLastError());
}


extern "C" void gpu_misalign_(
    double x_off, double y_off, double tilt,
    int set_flag, int n)
{
    if (n <= 0) return;
    int threads = 256;
    int blocks = (n + threads - 1) / threads;
    misalign_kernel<<<blocks, threads>>>(
        d_vec[0], d_vec[1], d_vec[2], d_vec[3],
        d_state,
        x_off, y_off, tilt, set_flag, n);
    CUDA_CHECK_VOID(cudaGetLastError());
    CUDA_CHECK_VOID(cudaDeviceSynchronize());
}

/* ==========================================================================
 * 3D MISALIGNMENT KERNEL -- general affine coordinate transform
 *
 * Applies a precomputed 3x3 rotation W and offset L to transform
 * particle (x, px, y, py) between lab and body frames.
 * Handles bends (curvature), pitches, z_offset -- any misalignment.
 *
 * set_flag=1: lab→body:  r_body = W · (r_lab - L), p_body = W · p_lab
 * set_flag=-1: body→lab: r_lab = W^T · r_body + L, p_lab = W^T · p_body
 *
 * W is column-major (Fortran): W(i,j) at W[j*3+i]
 * ========================================================================== */

__global__ void misalign_3d_kernel(
    double *vx, double *vpx, double *vy, double *vpy,
    double *vz, double *vpz,
    int *state,
    const double *W,  /* 9 doubles, 3x3 column-major */
    double Lx, double Ly, double Lz,
    int set_flag, int n)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    if (state[i] != ALIVE_ST) return;

    double x = vx[i], px = vpx[i], y = vy[i], py = vpy[i];
    double rel_p = 1.0 + vpz[i];
    double pz_sign = 1.0;  /* forward tracking */
    double pz_sq = rel_p*rel_p - px*px - py*py;
    if (pz_sq <= 0.0) { state[i] = LOST_PZ; return; }
    double pz_val = pz_sign * sqrt(pz_sq);

    if (set_flag == 1) {
        /* lab → body: r_body = W · (r_lab - L) */
        double rx = x - Lx, ry = y - Ly, rz = 0.0 - Lz;
        double bx = W[0]*rx + W[3]*ry + W[6]*rz;
        double by = W[1]*rx + W[4]*ry + W[7]*rz;
        /* bz = W[2]*rx + W[5]*ry + W[8]*rz; (used for drift correction) */

        /* p_body = W · p_lab */
        double pbx = W[0]*px + W[3]*py + W[6]*pz_val;
        double pby = W[1]*px + W[4]*py + W[7]*pz_val;

        vx[i] = bx; vpx[i] = pbx;
        vy[i] = by; vpy[i] = pby;
    } else {
        /* body → lab: r_lab = W^T · r_body + L */
        double rx = x, ry = y, rz = 0.0;
        double lx = W[0]*rx + W[1]*ry + W[2]*rz + Lx;
        double ly = W[3]*rx + W[4]*ry + W[5]*rz + Ly;

        /* p_lab = W^T · p_body */
        double plx = W[0]*px + W[1]*py + W[2]*pz_val;
        double ply = W[3]*px + W[4]*py + W[5]*pz_val;

        vx[i] = lx; vpx[i] = plx;
        vy[i] = ly; vpy[i] = ply;
    }
}

/* Device buffer for W matrix */
static double *d_misalign_W = NULL;

extern "C" void gpu_misalign_3d_(
    double *h_W,  /* 9 doubles, 3x3 column-major */
    double Lx, double Ly, double Lz,
    int set_flag, int n)
{
    if (n <= 0) return;
    if (!d_misalign_W) {
        cudaMalloc((void**)&d_misalign_W, 9*sizeof(double));
    }
    CUDA_CHECK_VOID(cudaMemcpy(d_misalign_W, h_W, 9*sizeof(double), cudaMemcpyHostToDevice));

    int threads = 256;
    int blocks = (n + threads - 1) / threads;
    misalign_3d_kernel<<<blocks, threads>>>(
        d_vec[0], d_vec[1], d_vec[2], d_vec[3],
        d_vec[4], d_vec[5], d_state,
        d_misalign_W, Lx, Ly, Lz, set_flag, n);
    CUDA_CHECK_VOID(cudaGetLastError());
    CUDA_CHECK_VOID(cudaDeviceSynchronize());
}

/* ==========================================================================
 * RECTANGULAR APERTURE CHECK KERNEL
 *
 * Checks x1_limit, x2_limit, y1_limit, y2_limit.
 * Limits of 0 mean no limit in that direction.
 * Lost particles have their state set to lost constants.
 * ========================================================================== */

/* LOST_NEG_X etc. defined at top of file */

__global__ void aperture_rect_kernel(
    const double *vx, const double *vy, int *state,
    double x1_lim, double x2_lim, double y1_lim, double y2_lim,
    int n)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    if (state[i] != ALIVE_ST) return;

    double x = vx[i], y = vy[i];
    /* Fast path: most particles are within aperture */
    int viol_x1 = (x1_lim > 0.0 && x < -x1_lim);
    int viol_x2 = (x2_lim > 0.0 && x >  x2_lim);
    int viol_y1 = (y1_lim > 0.0 && y < -y1_lim);
    int viol_y2 = (y2_lim > 0.0 && y >  y2_lim);
    if (!(viol_x1 | viol_x2 | viol_y1 | viol_y2)) return;

    /* Multiple violations: pick largest fractional violation to match CPU */
    double f_max = 0.0;
    int    lost  = LOST_NEG_X;
    double denom_x = x2_lim + x1_lim;
    double denom_y = y2_lim + y1_lim;
    if (viol_x1) { double f = (denom_x > 0) ? fabs((x+x1_lim)/denom_x) : 1.0; if (f > f_max) { f_max = f; lost = LOST_NEG_X; } }
    if (viol_x2) { double f = (denom_x > 0) ? fabs((x-x2_lim)/denom_x) : 1.0; if (f > f_max) { f_max = f; lost = LOST_POS_X; } }
    if (viol_y1) { double f = (denom_y > 0) ? fabs((y+y1_lim)/denom_y) : 1.0; if (f > f_max) { f_max = f; lost = LOST_NEG_Y; } }
    if (viol_y2) { double f = (denom_y > 0) ? fabs((y-y2_lim)/denom_y) : 1.0; if (f > f_max) { f_max = f; lost = LOST_POS_Y; } }
    state[i] = lost;
}

extern "C" void gpu_check_aperture_rect_(
    double x1_lim, double x2_lim, double y1_lim, double y2_lim, int n)
{
    if (n <= 0) return;
    int threads = 256;
    int blocks = (n + threads - 1) / threads;
    aperture_rect_kernel<<<blocks, threads>>>(
        d_vec[0], d_vec[2], d_state,
        x1_lim, x2_lim, y1_lim, y2_lim, n);
    CUDA_CHECK_VOID(cudaGetLastError());
    CUDA_CHECK_VOID(cudaDeviceSynchronize());
}

/* --------------------------------------------------------------------------
 * Elliptical aperture check kernel.
 * Particle is lost if (x/x_width)^2 + (y/y_width)^2 > 1.
 * x_width = (x1_lim + x2_lim)/2, y_width = (y1_lim + y2_lim)/2.
 * Lost direction determined by max deviation.
 * -------------------------------------------------------------------------- */
__global__ void aperture_ellipse_kernel(
    const double *vx, const double *vy, int *state,
    double x1_lim, double x2_lim, double y1_lim, double y2_lim,
    int n)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    if (state[i] != ALIVE_ST) return;

    double x = vx[i], y = vy[i];

    /* Match CPU elliptical_params_setup:
     * x1_lim/x2_lim are positive element values.
     * CPU negates x1_limit: lim1 = -x1_limit, lim2 = x2_limit
     * width2 = (lim2 - lim1)/2 = (x1+x2)/2
     * center = (lim2 + lim1)/2 = (x2-x1)/2
     * pos = x - center */
    double x_width2 = (x1_lim + x2_lim) * 0.5;
    double y_width2 = (y1_lim + y2_lim) * 0.5;
    double x_center = (x2_lim - x1_lim) * 0.5;
    double y_center = (y2_lim - y1_lim) * 0.5;

    if (x_width2 <= 0.0 || y_width2 <= 0.0) return;

    double xp = x - x_center;
    double yp = y - y_center;
    double rx = xp / x_width2;
    double ry = yp / y_width2;
    double r2 = rx*rx + ry*ry;

    if (r2 <= 1.0) return;

    /* Match CPU: direction from abs(x/xw) vs abs(y/yw) */
    if (fabs(rx) > fabs(ry)) {
        state[i] = (xp > 0.0) ? LOST_POS_X : LOST_NEG_X;
    } else {
        state[i] = (yp > 0.0) ? LOST_POS_Y : LOST_NEG_Y;
    }
}

extern "C" void gpu_check_aperture_ellipse_(
    double x1_lim, double x2_lim, double y1_lim, double y2_lim, int n)
{
    if (n <= 0) return;
    int threads = 256;
    int blocks = (n + threads - 1) / threads;
    aperture_ellipse_kernel<<<blocks, threads>>>(
        d_vec[0], d_vec[2], d_state,
        x1_lim, x2_lim, y1_lim, y2_lim, n);
    CUDA_CHECK_VOID(cudaGetLastError());
    CUDA_CHECK_VOID(cudaDeviceSynchronize());
}

/* ==========================================================================
 * S-POSITION UPDATE KERNEL -- update s for all alive particles
 * ========================================================================== */

__global__ void s_update_kernel(double *s_pos, const int *state, double s_val, int n)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    if (state[i] != ALIVE_ST) return;
    s_pos[i] = s_val;
}

extern "C" void gpu_s_update_(double s_val, int n)
{
    if (n <= 0) return;
    int threads = 256;
    int blocks = (n + threads - 1) / threads;
    s_update_kernel<<<blocks, threads>>>(d_s, d_state, s_val, n);
    CUDA_CHECK_VOID(cudaGetLastError());
    CUDA_CHECK_VOID(cudaDeviceSynchronize());
}

/* ==========================================================================
 * ORBIT-TOO-LARGE CHECK -- matches CPU orbit_too_large logic
 *
 * CPU checks: |x| or |y| > max_aperture_limit (default 1000 m),
 *             px^2 + py^2 > (1+pz)^2 for charged particles.
 * CPU does NOT check |z| or |pz| directly.
 * ========================================================================== */

__global__ void orbit_check_kernel(
    const double *vx, const double *vpx, const double *vy, const double *vpy,
    const double *vz, const double *vpz,
    int *state, int n)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    if (state[i] != ALIVE_ST) return;

    /* bmad_com%max_aperture_limit default = 1000.0 m */
    const double max_aper = 1000.0;

    /* Transverse position check */
    if (fabs(vx[i]) > max_aper) {
        state[i] = (vx[i] > 0) ? LOST_POS_X : LOST_NEG_X;
        return;
    }
    if (fabs(vy[i]) > max_aper) {
        state[i] = (vy[i] > 0) ? LOST_POS_Y : LOST_NEG_Y;
        return;
    }

    /* Momentum check: px^2 + py^2 must not exceed (1+pz)^2 */
    double rel_p = 1.0 + vpz[i];
    if (rel_p < 0.0) {
        state[i] = LOST_PZ;
        return;
    }
    if (vpx[i]*vpx[i] + vpy[i]*vpy[i] > rel_p*rel_p) {
        /* Match CPU: assign directional code based on dominant momentum */
        if (fabs(vpx[i]) > fabs(vpy[i])) {
            state[i] = (vpx[i] > 0) ? LOST_POS_X : LOST_NEG_X;
        } else {
            state[i] = (vpy[i] > 0) ? LOST_POS_Y : LOST_NEG_Y;
        }
    }
}

extern "C" void gpu_orbit_check_(int n)
{
    if (n <= 0) return;
    int threads = 256;
    int blocks = (n + threads - 1) / threads;
    orbit_check_kernel<<<blocks, threads>>>(
        d_vec[0], d_vec[1], d_vec[2], d_vec[3], d_vec[4], d_vec[5],
        d_state, n);
    CUDA_CHECK_VOID(cudaGetLastError());
    CUDA_CHECK_VOID(cudaDeviceSynchronize());
}

/* --------------------------------------------------------------------------
 * gpu_tracking_cleanup -- release cached device buffers
 * -------------------------------------------------------------------------- */
extern "C" void gpu_tracking_cleanup_(void)
{
    for (int k = 0; k < 6; k++) { if (d_vec[k]) cudaFree(d_vec[k]); d_vec[k] = NULL; }
    if (d_state) cudaFree(d_state); d_state = NULL;
    if (d_beta)  cudaFree(d_beta);  d_beta  = NULL;
    if (d_p0c)   cudaFree(d_p0c);   d_p0c   = NULL;
    if (d_s)     cudaFree(d_s);     d_s     = NULL;
    if (d_t)     cudaFree(d_t);     d_t     = NULL;
    if (d_a2)    cudaFree(d_a2);    d_a2    = NULL;
    if (d_b2)    cudaFree(d_b2);    d_b2    = NULL;
    if (d_ea2)   cudaFree(d_ea2);   d_ea2   = NULL;
    if (d_eb2)   cudaFree(d_eb2);   d_eb2   = NULL;
    if (d_cm)    cudaFree(d_cm);    d_cm    = NULL;
    if (d_step_s0)   cudaFree(d_step_s0);   d_step_s0   = NULL;
    if (d_step_s)    cudaFree(d_step_s);    d_step_s    = NULL;
    if (d_step_p0c)  cudaFree(d_step_p0c);  d_step_p0c  = NULL;
    if (d_step_p1c)  cudaFree(d_step_p1c);  d_step_p1c  = NULL;
    if (d_step_scl)  cudaFree(d_step_scl);  d_step_scl  = NULL;
    if (d_step_time) cudaFree(d_step_time); d_step_time = NULL;
    if (d_rng_states)    cudaFree(d_rng_states);    d_rng_states    = NULL;
    if (d_stoc_mat)      cudaFree(d_stoc_mat);      d_stoc_mat      = NULL;
    if (d_damp_dmat)     cudaFree(d_damp_dmat);     d_damp_dmat     = NULL;
    if (d_xfer_damp_vec) cudaFree(d_xfer_damp_vec); d_xfer_damp_vec = NULL;
    if (d_ref_orb)       cudaFree(d_ref_orb);       d_ref_orb       = NULL;
    d_rng_cap = 0;
    d_step_cap = 0;
    d_cap = 0;
}

#endif /* USE_GPU_TRACKING */
