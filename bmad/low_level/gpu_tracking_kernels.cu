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

/* ==========================================================================
 * BEND EXACT MULTIPOLE FIELD COMPUTATION
 *
 * Ported from bmad/code/bend_exact_multipole_field.f90.
 * Uses the F_coef table (Pade approximant / exact log formulas) to compute
 * the magnetic field in a curved reference frame (bend).  The curvature of
 * the reference orbit modifies Maxwell's equations, so the standard flat-
 * geometry multipole kick is not correct for bends.
 *
 * The F_coef table has 23 entries (orders 0..22).  Each entry contains
 * coefficients for F_value (the radial function) and F_derivative (used
 * for Bx).  Near x*g = 0, a Pade approximant is used for numerical
 * stability; outside the cutoff region, the exact log formula is used.
 * ========================================================================== */

/* Packed structure for one F_coef entry -- all arrays laid out flat */
struct FCoefEntry {
    int    order;
    double cutoff_minus, cutoff_plus;
    int    n_exact_non;
    double exact_non_coef[12];  /* 0..11 */
    int    n_exact_log;
    double exact_log_coef[11];  /* 0..10 */
    int    n_pade_numer;
    double pade_numer_coef[8];  /* 0..7 */
    int    n_pade_denom;
    double pade_denom_coef[9];  /* 0..8 */
};

/* F_coef table -- 23 entries, orders 0..22.
 * Transcribed from bend_exact_multipole_field.f90 F_coef(0:22). */
__constant__ struct FCoefEntry d_F_coef[23] = {
    /* order 0 */
    {0, 0, 0,   0, {0,0,0,0,0,0,0,0,0,0,0,0},  0, {0,0,0,0,0,0,0,0,0,0,0},
     0, {0,0,0,0,0,0,0,0},  0, {0,0,0,0,0,0,0,0,0}},
    /* order 1 */
    {1, -0.01, 0.01,
     -1, {0,0,0,0,0,0,0,0,0,0,0,0},
     0, {1.0,0,0,0,0,0,0,0,0,0,0},
     3, {1.0, 0.83333333333333, 0.066666666666667, -0.0055555555555556, 0,0,0,0},
     2, {1.0, 1.3333333333333, 0.4, 0,0,0,0,0,0}},
    /* order 2 */
    {2, -0.0224, 0.0316,
     1, {-0.5, 0.5, 0,0,0,0,0,0,0,0,0,0},
     0, {-1.0, 0,0,0,0,0,0,0,0,0,0},
     4, {1.0, 1.1666666666667, 0.28571428571429, -0.0035714285714286, 0.0005952380952381, 0,0,0},
     2, {1.0, 1.5, 0.53571428571429, 0,0,0,0,0,0}},
    /* order 3 */
    {3, -0.0794, 0.1,
     1, {1.5, -1.5, 0,0,0,0,0,0,0,0,0,0},
     1, {1.5, 1.5, 0,0,0,0,0,0,0,0,0},
     4, {1.0, 1.3834196891192, 0.48255613126079, 0.022366148531952, -0.0010270170244264, 0,0,0},
     3, {1.0, 1.8834196891192, 1.0742659758204, 0.17530224525043, 0,0,0,0,0}},
    /* order 4 */
    {4, -0.15, 0.178,
     2, {-1.875, 1.5, 0.375, 0,0,0,0,0,0,0,0,0},
     1, {-1.5, -3.0, 0,0,0,0,0,0,0,0,0},
     4, {1.0, 1.9565979090138, 1.267864573968, 0.30992013543582, 0.023174683703788, 0,0,0},
     4, {1.0, 2.3565979090138, 1.9105037375735, 0.60999940061822, 0.060982814868093, 0,0,0,0}},
    /* order 5 */
    {5, -0.219, 0.251,
     2, {2.8125, 0.0, -2.8125, 0,0,0,0,0,0,0,0,0},
     2, {1.875, 7.5, 1.875, 0,0,0,0,0,0,0,0},
     4, {1.0, 2.0, 1.3133394383394, 0.31333943833944, 0.019614644614645, 0,0,0},
     5, {1.0, 2.5, 2.2061965811966, 0.80929487179487, 0.10954901579902, 0.0032253626003626, 0,0,0}},
    /* order 6 */
    {6, -0.282, 0.316,
     3, {-3.125, -2.8125, 5.625, 0.3125, 0,0,0,0,0,0,0,0},
     2, {-1.875, -11.25, -5.625, 0,0,0,0,0,0,0,0},
     5, {1.0, 2.4187964980921, 2.1040382083956, 0.79642117021259, 0.1255086337457, 0.0062963159736057, 0,0},
     5, {1.0, 2.8473679266635, 3.0029101769656, 1.4300620315322, 0.29569616640861, 0.019880608705292, 0,0,0}},
    /* order 7 */
    {7, -0.338, 0.373,
     3, {4.0104166666667, 9.84375, -9.84375, -4.0104166666667, 0,0,0,0,0,0,0,0},
     3, {2.1875, 19.6875, 19.6875, 2.1875, 0,0,0,0,0,0,0},
     5, {1.0, 2.5, 2.2514276625317, 0.87714149379762, 0.13753081472819, 0.0059084917311593, 0,0},
     6, {1.0, 3.0, 3.3903165514206, 1.7806331028413, 0.43046032864087, 0.040143777220235, 0.00082665505743365, 0,0}},
    /* order 8 */
    {8, -0.387, 0.422,
     4, {-4.2838541666667, -17.5, 9.84375, 11.666666666667, 0.2734375, 0,0,0,0,0,0,0},
     3, {-2.1875, -26.25, -39.375, -8.75, 0,0,0,0,0,0,0},
     5, {1.0, 2.5610637208739, 2.3919099978286, 0.98937170028964, 0.17357822943969, 0.0096995892929036, 0,0},
     6, {1.0, 3.0055081653184, 3.3943580713034, 1.7688665051567, 0.41438661668697, 0.033650020420609, 0.000050495737648433, 0,0}},
    /* order 9 */
    {9, -0.43, 0.464,
     4, {5.126953125, 32.8125, 0.0, -32.8125, -5.126953125, 0,0,0,0,0,0,0},
     4, {2.4609375, 39.375, 88.59375, 39.375, 2.4609375, 0,0,0,0,0,0},
     6, {1.0, 2.733043364401, 2.7718291902232, 1.2772433471491, 0.25997188487676, 0.018326394926351, 0.00013278743887644, 0},
     6, {1.0, 3.233043364401, 4.0247145087873, 2.4094030144879, 0.70373582351232, 0.088893885031519, 0.0033132610003509, 0,0}},
    /* order 10 */
    {10, -0.468, 0.501,
     5, {-5.373046875, -47.16796875, -24.609375, 57.421875, 19.482421875, 0.24609375, 0,0,0,0,0,0},
     4, {-2.4609375, -49.21875, -147.65625, -98.4375, -12.3046875, 0,0,0,0,0,0},
     6, {1.0, 2.8752528324977, 3.1372522022764, 1.6195427416905, 0.40170620422812, 0.04303954725897, 0.0014669363563495, 0},
     6, {1.0, 3.3297982870431, 4.309887787296, 2.7231444177806, 0.86124668679305, 0.12371395040216, 0.0058577600405354, 0,0}},
    /* order 11 */
    {11, -0.501, 0.534,
     5, {6.1810546875, 73.3154296875, 90.234375, -90.234375, -73.3154296875, -6.1810546875, 0,0,0,0,0,0},
     5, {2.70703125, 67.67578125, 270.703125, 270.703125, 67.67578125, 2.70703125, 0,0,0,0,0},
     6, {1.0, 2.7333931468221, 2.7730902668328, 1.2786959487213, 0.26066077963333, 0.018454839938474, 0.00013857880170088, 0},
     6, {1.0, 3.2333931468221, 4.0244022248592, 2.407541872889, 0.70200770652817, 0.088320424888959, 0.0032543708103201, 0,0}},
    /* order 12 */
    {12, -0.531, 0.562,
     6, {-6.406640625, -96.099609375, -186.1083984375, 90.234375, 169.189453125, 28.965234375, 0.2255859375, 0,0,0,0,0},
     5, {-2.70703125, -81.2109375, -406.0546875, -541.40625, -203.02734375, -16.2421875, 0,0,0,0,0},
     6, {1.0, 3.0448349228732, 3.5652325648319, 2.0107624226303, 0.5581463905672, 0.068912599775289, 0.0027503351125044, 0},
     7, {1.0, 3.5063733844117, 4.8374048960989, 3.3142815877642, 1.1661040072182, 0.19412177384366, 0.011475769024855, 0.000014175308484462, 0}},
    /* order 13 */
    {13, -0.557, 0.588,
     6, {7.184912109375, 135.4869140625, 384.90600585937, 0.0, -384.90600585937, -135.4869140625, -7.184912109375, 0,0,0,0,0},
     6, {2.9326171875, 105.57421875, 659.8388671875, 1173.046875, 659.8388671875, 105.57421875, 2.9326171875, 0,0,0,0},
     6, {1.0, 3.0, 3.4398700386372, 1.8797400772743, 0.49529660107961, 0.055426562442451, 0.0017537166619788, 0},
     7, {1.0, 3.5, 4.8232033719705, 3.3080084299262, 1.1727046776928, 0.20104858661291, 0.013546152700783, 0.00019859291244789, 0}},
    /* order 14 */
    {14, -0.518, 0.611,
     7, {-7.394384765625, -168.3322265625, -631.24584960937, -256.60400390625, 641.51000976562, 381.8267578125, 40.030224609375, 0.20947265625, 0,0,0,0},
     6, {-2.9326171875, -123.169921875, -923.7744140625, -2052.83203125, -1539.6240234375, -369.509765625, -20.5283203125, 0,0,0,0},
     6, {1.0, 3.034629705342, 3.5369070660493, 1.9813638048949, 0.54414752125926, 0.065935628026009, 0.0025331166641352, 0},
     7, {1.0, 3.5012963720087, 4.8208453729867, 3.2938732095366, 1.154415151606, 0.19109330453762, 0.011210062927602, 0.000016581712694799, 0}},
    /* order 15 */
    {15, -0.541, 0.631,
     7, {8.1469900948661, 223.24548339844, 1085.4349365234, 962.26501464844, -962.26501464844, -1085.4349365234, -223.24548339844, -8.1469900948661, 0,0,0,0},
     7, {3.14208984375, 153.96240234375, 1385.6616210938, 3849.0600585937, 3849.0600585937, 1385.6616210938, 153.96240234375, 3.14208984375, 0,0,0},
     6, {1.0, 3.0, 3.4400585392107, 1.8801170784215, 0.49556813038953, 0.055509591178792, 0.0017626354878635, 0},
     7, {1.0, 3.5, 4.8224114803872, 3.306028700968, 1.1709157214433, 0.20034488119687, 0.013433848489043, 0.00019373407731112, 0}},
    /* order 16 */
    {16, -0.562, 0.649,
     8, {-8.3433707101004, -267.7060546875, -1601.208984375, -2155.4736328125, 962.26501464844, 2278.6435546875, 739.01953125, 52.607561383929, 0.19638061523438, 0,0,0},
     7, {-3.14208984375, -175.95703125, -1847.548828125, -6158.49609375, -7698.1201171875, -3695.09765625, -615.849609375, -25.13671875, 0,0,0},
     6, {1.0, 3.0267474784445, 3.5150700003594, 1.9587569664169, 0.53342251019288, 0.063667813336172, 0.0023691677567435, 0},
     7, {1.0, 3.4973357137386, 4.8079338656481, 3.2779819701702, 1.1453369246616, 0.18875391331486, 0.011008376720253, 0.000018907401960289, 0}},
    /* order 17 */
    {17, -0.582, 0.666,
     8, {9.0734857831682, 340.33321707589, 2486.4927978516, 4711.2495117188, 0.0, -4711.2495117188, -2486.4927978516, -340.33321707589, -9.0734857831682, 0,0,0},
     8, {3.3384704589844, 213.662109375, 2617.3608398437, 10469.443359375, 16358.505249023, 10469.443359375, 2617.3608398437, 213.662109375, 3.3384704589844, 0,0},
     6, {1.0, 3.2157192672142, 4.0246058913212, 2.4656382544766, 0.7600290519396, 0.10732582666566, 0.0050545105774104, 0},
     8, {1.0, 3.7157192672142, 5.5140444722967, 4.1563428658619, 1.6699262094031, 0.34103513472824, 0.02989625911893, 0.00066188011699055, -3.8321173171098e-6}},
    /* order 18 */
    {18, -0.599, 0.681,
     9, {-9.2589563642229, -397.89798627581, -3437.2891845703, -8375.5546875, -2944.5309448242, 7655.780456543, 6150.7979736328, 1291.1296037946, 66.638254983085, 0.18547058105469, 0,0},
     8, {-3.3384704589844, -240.36987304687, -3365.1782226562, -15704.165039063, -29445.309448242, -23556.247558594, -7852.0825195312, -961.4794921875, -30.046234130859, 0,0},
     7, {1.0, 3.5306836929127, 4.9787705805848, 3.5748288749893, 1.3787221678681, 0.27591307847442, 0.025151905086668, 0.00074062043812306},
     8, {1.0, 4.004367903439, 6.5203132716875, 5.5340270154316, 2.6041300484626, 0.66504710948496, 0.082751975852322, 0.0037091269405145, 4.1162181059434e-6}},
    /* order 19 */
    {19, -0.616, 0.695,
     9, {9.9691173311264, 490.34381021772, 4991.1087210519, 15333.372253418, 11189.217590332, -11189.217590332, -15333.372253418, -4991.1087210519, -490.34381021772, -9.9691173311264, 0,0},
     9, {3.5239410400391, 285.43922424316, 4567.0275878906, 24864.927978516, 55946.08795166, 55946.08795166, 24864.927978516, 4567.0275878906, 285.43922424316, 3.5239410400391, 0},
     7, {1.0, 3.5, 4.8775459718386, 3.4438649295966, 1.294825988365, 0.24837405295099, 0.020877112471247, 0.00050504506367791},
     8, {1.0, 4.0, 6.508498352791, 5.525495058373, 2.6074910509159, 0.67249033787685, 0.086419935430204, 0.0044239428872841, 0.000049085206240032}},
    /* order 20 */
    {20, -0.631, 0.708,
     10, {-10.145314383128, -562.44616099766, -6595.6849316188, -24442.055053711, -26418.985977173, 11189.217590332, 30563.140640259, 14099.791521345, 2094.9200207847, 82.071468111068, 0.17619705200195, 0},
     9, {-3.5239410400391, -317.15469360352, -5708.7844848633, -35521.325683594, -93243.479919434, -111892.17590332, -62162.319946289, -15223.425292969, -1427.1961212158, -35.239410400391, 0},
     7, {1.0, 3.5251396690978, 4.9605987099841, 3.5514756331536, 1.3638765739909, 0.27108740107008, 0.024413032466054, 0.00070074454442973},
     8, {1.0, 4.0013301452883, 6.5088511601214, 5.5169131419853, 2.5913408457078, 0.66008942597404, 0.081834003934574, 0.0036501401808332, 4.5632350175799e-6}},
    /* order 21 */
    {21, -0.645, 0.72,
     10, {10.837587006887, 676.74351056417, 9125.1352000237, 40468.938903809, 59831.232948303, 0.0, -59831.232948303, -40468.938903809, -9125.1352000237, -676.74351056417, -10.837587006887, 0},
     10, {3.700138092041, 370.0138092041, 7492.7796363831, 53281.988525391, 163176.08985901, 234973.56939697, 163176.08985901, 53281.988525391, 7492.7796363831, 370.0138092041, 3.700138092041},
     7, {1.0, 3.5, 4.877620852413, 3.4440521310325, 1.2950082042224, 0.24846017530121, 0.020896613402104, 0.00050668185193194},
     8, {1.0, 4.0, 6.5080556350217, 5.524166905065, 2.6059624003339, 0.67164662555942, 0.0861932030478, 0.004397707778928, 0.000048223327017921}},
    /* order 22 */
    {22, -0.658, 0.731,
     11, {-11.00577510198, -764.76862112681, -11661.712009907, -60223.711881638, -112336.19247437, -35898.739768982, 95729.972717285, 92973.898429871, 28879.908177853, 3213.3203204473, 98.862697569529, 0.16818809509277},
     10, {-3.700138092041, -407.01519012451, -9157.8417778015, -73262.734222412, -256419.56977844, -430784.87722778, -358987.39768982, -146525.46844482, -27473.525333405, -2035.0759506226, -40.701519012451},
     7, {1.0, 3.5205293526005, 4.9455011102732, 3.5320978036904, 1.3515804842329, 0.26710119359406, 0.023805221453277, 0.00066816326407054},
     8, {1.0, 3.9987902221657, 6.4992703469612, 5.5026175638515, 2.5806701431507, 0.65596147985212, 0.081072889433702, 0.0036018948868731, 4.9980796385363e-6}}
};

/* Factorial table for orders 0..22 */
__constant__ double d_factorial[23] = {
    1.0, 1.0, 2.0, 6.0, 24.0, 120.0, 720.0, 5040.0, 40320.0, 362880.0,
    3628800.0, 39916800.0, 479001600.0, 6227020800.0, 87178291200.0,
    1307674368000.0, 20922789888000.0, 355687428096000.0,
    6402373705728000.0, 121645100408832000.0, 2432902008176640000.0,
    51090942171709440000.0, 1124000727777607680000.0
};

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

/* Cached device buffers for exact bend multipole arrays */
static double *d_exact_an = NULL;
static double *d_exact_bn = NULL;

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
 * F_value_dev -- evaluate the radial F-function at xg for a given order j.
 * Matches F_value in bend_exact_multipole_field.f90.
 * -------------------------------------------------------------------------- */
__device__ double F_value_dev(int j, double xg)
{
    const struct FCoefEntry *c = &d_F_coef[j];
    if (c->order == 0) return 1.0;

    double value;
    if (xg < c->cutoff_minus || xg > c->cutoff_plus) {
        /* Exact log formula */
        double r = 1.0 + xg;
        value = 0.0;
        for (int i = 0; i <= c->n_exact_log; i++)
            value += c->exact_log_coef[i] * ipow(r, 2*i);
        value *= log(r);
        for (int i = 0; i <= c->n_exact_non; i++)
            value += c->exact_non_coef[i] * ipow(r, 2*i);
    } else {
        /* Pade approximant */
        double numer = 0.0;
        for (int i = 0; i <= c->n_pade_numer; i++)
            numer += c->pade_numer_coef[i] * ipow(xg, i + c->order);
        double denom = 0.0;
        for (int i = 0; i <= c->n_pade_denom; i++)
            denom += c->pade_denom_coef[i] * ipow(xg, i);
        value = numer / denom;
    }
    return value;
}

/* --------------------------------------------------------------------------
 * F_derivative_dev -- evaluate dF/d(xg) for a given order j.
 * Matches F_derivative in bend_exact_multipole_field.f90.
 * -------------------------------------------------------------------------- */
__device__ double F_derivative_dev(int j, double xg)
{
    const struct FCoefEntry *c = &d_F_coef[j];
    if (c->order == 0) return 0.0;

    double value;
    if (xg < c->cutoff_minus || xg > c->cutoff_plus) {
        double r = 1.0 + xg;
        double v0 = 0.0;
        value = 0.0;
        for (int i = 0; i <= c->n_exact_log; i++) {
            double f = c->exact_log_coef[i] * ipow(r, 2*i - 1);
            v0    += f;
            value += 2.0 * i * f;
        }
        value = value * log(r) + v0;
        for (int i = 1; i <= c->n_exact_non; i++)
            value += 2.0 * i * c->exact_non_coef[i] * ipow(r, 2*i - 1);
    } else {
        double numer = 0.0, d_numer = 0.0;
        for (int i = 0; i <= c->n_pade_numer; i++) {
            numer += c->pade_numer_coef[i] * ipow(xg, i + c->order);
            if (i + c->order >= 1)
                d_numer += (i + c->order) * c->pade_numer_coef[i] * ipow(xg, i - 1 + c->order);
        }
        double denom = 0.0, d_denom = 0.0;
        for (int i = 0; i <= c->n_pade_denom; i++) {
            denom += c->pade_denom_coef[i] * ipow(xg, i);
            if (i > 0)
                d_denom += i * c->pade_denom_coef[i] * ipow(xg, i - 1);
        }
        value = (denom * d_numer - numer * d_denom) / (denom * denom);
    }
    return value;
}

/* --------------------------------------------------------------------------
 * bend_exact_multipole_field_dev -- compute exact Bx, By for a bend.
 *
 * Ported from bend_exact_multipole_field.f90.  The raw multipole arrays
 * (a_pole, b_pole) must already be in the vertically_pure basis (converted
 * on CPU if the element uses horizontally_pure).
 *
 * Input:
 *   a_pole, b_pole: raw multipole arrays (0..ix_mag_max)
 *   ix_mag_max: maximum multipole order
 *   g:   bend curvature (1/rho)
 *   rho: bend radius (1/g)
 *   x, y: particle position
 *   f_scale: p0c / (c_light * charge * L) -- element-level scaling
 *
 * Output:
 *   Bx, By: field components (scaled by f_scale)
 * -------------------------------------------------------------------------- */
__device__ void bend_exact_multipole_field_dev(
    const double *a_pole, const double *b_pole, int ix_mag_max,
    double g, double rho, double x, double y, double f_scale,
    double *Bx_out, double *By_out)
{
    double xg = x * g;
    double yg = y * g;

    /* Precompute yg^n for n = 0..ix_mag_max+1 */
    double yg_n[24];  /* enough for ix_mag_max up to 22 */
    yg_n[0] = 1.0;
    for (int n = 0; n <= ix_mag_max; n++)
        yg_n[n+1] = yg_n[n] * yg;

    double Bx = 0.0, By = 0.0;
    double rho_n = 1.0;

    for (int n = 0; n <= ix_mag_max; n++) {
        if (n > 0) rho_n *= rho;
        double fact_n = d_factorial[n];

        /* Process b_pole[n] (non-skew): j goes from n down to 0 in steps of 2 */
        if (b_pole[n] != 0.0) {
            int sgn = 1;
            for (int j = n; j >= 0; j -= 2) {
                double crs = b_pole[n] * fact_n * rho_n * sgn;
                double pc_val = crs * yg_n[n+1-j] / (d_factorial[j] * d_factorial[n+1-j]);
                double pc_der = crs * yg_n[n-j] / (d_factorial[j] * d_factorial[n-j]);

                if (pc_val != 0.0) Bx += F_derivative_dev(j, xg) * pc_val;
                if (pc_der != 0.0) By += F_value_dev(j, xg) * pc_der;

                sgn = -sgn;
            }
        }

        /* Process a_pole[n] (skew): j goes from n+1 down to 0 in steps of 2 */
        if (a_pole[n] != 0.0) {
            int sgn = 1;
            for (int j = n+1; j >= 0; j -= 2) {
                double crs = a_pole[n] * fact_n * rho_n * sgn;
                double pc_val = crs * yg_n[n+1-j] / (d_factorial[j] * d_factorial[n+1-j]);
                double pc_der = (n-j >= 0) ?
                    crs * yg_n[n-j] / (d_factorial[j] * d_factorial[n-j]) : 0.0;

                if (pc_val != 0.0) Bx += F_derivative_dev(j, xg) * pc_val;
                if (pc_der != 0.0) By += F_value_dev(j, xg) * pc_der;

                sgn = -sgn;
            }
        }
    }

    *Bx_out = Bx * f_scale;
    *By_out = By * f_scale;
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

/* --------------------------------------------------------------------------
 * upload_exact_multipole_data -- H→D transfer of exact bend multipole arrays
 * -------------------------------------------------------------------------- */
static int upload_exact_multipole_data(
    double *h_exact_an, double *h_exact_bn, int ix_exact_mag_max)
{
    if (ix_exact_mag_max < 0) return 0;
    size_t multi_sz = N_MULTI * sizeof(double);
    if (!d_exact_an) { CUDA_CHECK(cudaMalloc((void**)&d_exact_an, multi_sz)); }
    if (!d_exact_bn) { CUDA_CHECK(cudaMalloc((void**)&d_exact_bn, multi_sz)); }
    CUDA_CHECK(cudaMemcpy(d_exact_an, h_exact_an, multi_sz, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_exact_bn, h_exact_bn, multi_sz, cudaMemcpyHostToDevice));
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
    const double *d_ea2, const double *d_eb2, int ix_elec_max,
    /* Exact multipoles parameters (when is_exact != 0) */
    int is_exact,
    const double *d_exact_an, const double *d_exact_bn,
    int ix_exact_mag_max,
    double rho, double c_dir, double exact_f_scale)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n_particles) return;
    if (state[i] != ALIVE_ST) return;

    int has_mag = (ix_mag_max >= 0);
    int has_elec = (ix_elec_max >= 0);
    int has_exact_mag = (is_exact && ix_exact_mag_max >= 0);
    double step_len = ele_length / (double)n_step;
    double angle = g * step_len;
    double z_start = vz[i];
    double t_start = t_arr[i];
    double beta_val = beta_arr[i];
    double p0c_val = p0c_arr[i];
    double beta_ref = p0c_ele / e_tot_ele;

    /* Entrance half magnetic multipole kick */
    if (has_exact_mag) {
        /* Exact multipole field in curved geometry */
        double Bx, By;
        bend_exact_multipole_field_dev(d_exact_an, d_exact_bn, ix_exact_mag_max,
                                       g, rho, vx[i], vy[i], exact_f_scale, &Bx, &By);
        double f_coef = 0.5 * c_dir * (1.0 + g * vx[i]) * C_LIGHT / p0c_arr[i];
        f_coef *= step_len;
        vpx[i] -= f_coef * By;
        vpy[i] += f_coef * Bx;
    } else if (has_mag) {
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
        if (has_exact_mag) {
            double Bx, By;
            bend_exact_multipole_field_dev(d_exact_an, d_exact_bn, ix_exact_mag_max,
                                           g, rho, vx[i], vy[i], exact_f_scale, &Bx, &By);
            double scl = (istep == n_step) ? 0.5 : 1.0;
            double f_coef = scl * c_dir * (1.0 + g * vx[i]) * C_LIGHT / p0c_arr[i];
            f_coef *= step_len;
            vpx[i] -= f_coef * By;
            vpy[i] += f_coef * Bx;
        } else if (has_mag) {
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
    double *h_ea2, double *h_eb2, int ix_elec_max,
    int is_exact,
    double *h_exact_an, double *h_exact_bn,
    int ix_exact_mag_max,
    double rho, double c_dir, double exact_f_scale)
{
    if (ensure_buffers(n_particles) != 0) return;

    /* Host -> Device */
    if (upload_particle_data(n_particles, h_vx, h_vpx, h_vy, h_vpy, h_vz, h_vpz,
                             h_state, h_beta, h_p0c, h_t) != 0) return;
    if (upload_multipole_data(h_a2, h_b2, h_cm, h_ea2, h_eb2,
                              ix_mag_max, ix_elec_max) != 0) return;
    if (is_exact) {
        if (upload_exact_multipole_data(h_exact_an, h_exact_bn, ix_exact_mag_max) != 0) return;
    }

    /* Launch kernel */
    int threads = 256;
    int blocks  = (n_particles + threads - 1) / threads;
    bend_kernel<<<blocks, threads>>>(
        d_vec[0], d_vec[1], d_vec[2], d_vec[3], d_vec[4], d_vec[5],
        d_state, d_beta, d_p0c, d_t,
        mc2, g, g_tot, dg, b1, ele_length, delta_ref_time, e_tot_ele,
        rel_charge_dir, p0c_ele,
        n_particles, d_a2, d_b2, d_cm, ix_mag_max, n_step,
        d_ea2, d_eb2, ix_elec_max,
        is_exact, d_exact_an, d_exact_bn, ix_exact_mag_max,
        rho, c_dir, exact_f_scale);
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

/* =========================================================================
 * SOLENOID KERNEL
 * Replicates the body of solenoid_track_and_mat (tracking only, no matrix).
 *
 * The solenoid body applies a 4x4 rotation in (x, px, y, py) space plus
 * longitudinal (z) and time updates.  For a pure solenoid (b1=0) or when
 * ele%key == solenoid$, the CPU code calls solenoid_track_and_mat.
 *
 * Parameters:
 *   ks0     -- solenoid strength: rel_tracking_charge * bs_field * charge * c_light / p0c
 *              (pre-computed on host, same for all particles at given p0c)
 *   length  -- step length (signed, includes time_dir)
 *   ref_beta -- reference beta = p0c / e_tot
 *
 * This kernel handles the n_step loop and multipole kicks, matching
 * track_a_sol_quad for the solenoid$ branch.
 * ========================================================================= */

__global__ void solenoid_kernel(
    double *vx, double *vpx, double *vy, double *vpy, double *vz, double *vpz,
    int *state, double *beta_arr, double *p0c_arr, double *t_arr,
    double mc2, double ks0, double ele_length,
    double delta_ref_time, double e_tot_ele,
    int n_particles, int n_step,
    /* Multipole parameters */
    const double *d_a2, const double *d_b2, const double *d_cm,
    int ix_mag_max,
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

    /* Body: n_step solenoid sub-steps */
    for (int istep = 1; istep <= n_step; istep++) {
        double vec0_x  = vx[i];
        double vec0_px = vpx[i];
        double vec0_y  = vy[i];
        double vec0_py = vpy[i];

        double rel_p = 1.0 + vpz[i];
        double kss0 = ks0 / 2.0;

        double xp = vec0_px + kss0 * vec0_y;
        double yp = vec0_py - kss0 * vec0_x;
        double ff = rel_p * rel_p - xp * xp - yp * yp;
        if (ff <= 0.0) {
            state[i] = LOST_PZ;
            return;
        }
        double pz = sqrt(ff);

        /* z update */
        double dir_beta_ratio = beta_val / beta_ref;
        vz[i] += step_len * (dir_beta_ratio - rel_p / pz);

        double ks_rel = ks0 / pz;
        double kss = ks_rel / 2.0;
        double kssl = kss * step_len;

        double c, s, c2, s2, cs;
        double mat4_11, mat4_12, mat4_13, mat4_14;
        double mat4_21, mat4_22, mat4_23, mat4_24;
        double mat4_31, mat4_32, mat4_33, mat4_34;
        double mat4_41, mat4_42, mat4_43, mat4_44;

        if (fabs(step_len * kss) < 1e-10) {
            double ll = step_len;
            double kssl2 = kssl * kssl;

            mat4_11 = 1.0;          mat4_12 = ll / pz;
            mat4_13 = kssl;         mat4_14 = kssl * ll / pz;
            mat4_21 = -kssl * kss0; mat4_22 = 1.0;
            mat4_23 = -kssl2 * kss0; mat4_24 = kssl;
            mat4_31 = -kssl;        mat4_32 = -kssl * ll / pz;
            mat4_33 = 1.0;          mat4_34 = ll / pz;
            mat4_41 = kssl2 * kss0; mat4_42 = -kssl;
            mat4_43 = -kssl * kss0; mat4_44 = 1.0;
        } else {
            c = cos(kssl);
            s = sin(kssl);
            c2 = c * c;
            s2 = s * s;
            cs = c * s;

            mat4_11 = c2;           mat4_12 = cs / kss0;
            mat4_13 = cs;           mat4_14 = s2 / kss0;
            mat4_21 = -kss0 * cs;   mat4_22 = c2;
            mat4_23 = -kss0 * s2;   mat4_24 = cs;
            mat4_31 = -cs;          mat4_32 = -s2 / kss0;
            mat4_33 = c2;           mat4_34 = cs / kss0;
            mat4_41 = kss0 * s2;    mat4_42 = -cs;
            mat4_43 = -kss0 * cs;   mat4_44 = c2;
        }

        /* Apply 4x4 rotation matrix */
        vx[i]  = mat4_11 * vec0_x + mat4_12 * vec0_px + mat4_13 * vec0_y + mat4_14 * vec0_py;
        vpx[i] = mat4_21 * vec0_x + mat4_22 * vec0_px + mat4_23 * vec0_y + mat4_24 * vec0_py;
        vy[i]  = mat4_31 * vec0_x + mat4_32 * vec0_px + mat4_33 * vec0_y + mat4_34 * vec0_py;
        vpy[i] = mat4_41 * vec0_x + mat4_42 * vec0_px + mat4_43 * vec0_y + mat4_44 * vec0_py;

        /* Time update for this sub-step */
        t_arr[i] += step_len * rel_p / (pz * beta_val * C_LIGHT);

        /* s update within sub-step */
        /* (s is updated at the end by the caller, not per sub-step) */

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

    /* Final time: override sub-step accumulation with the standard formula
     * to exactly match CPU:  t = t_start + dir*time_dir*delta_ref_time + (z_start - z) / (beta * c_light)
     * Since we only do forward tracking (dir=1, time_dir=1), this is: */
    t_arr[i] = t_start + delta_ref_time + (z_start - vz[i]) / (beta_val * C_LIGHT);
}

/* =========================================================================
 * SOL_QUAD KERNEL
 * Replicates the body of sol_quad_mat6_calc (tracking only, no matrix).
 *
 * For a combined solenoid + quadrupole element.  The CPU routine computes
 * a 4x4 transfer matrix in (x, x', y, y') coordinates using eigenvalue
 * decomposition involving alpha, beta, trig/hyp functions of alpha*L and
 * beta*L, then converts back to (x, px, y, py) coordinates.
 *
 * Parameters:
 *   ks_in  -- solenoid strength: rel_tracking_charge * ele%value(ks$)
 *   k1_in  -- quad strength: charge_dir * b1 / ele%value(l$)
 *   length -- step length
 * ========================================================================= */

__global__ void sol_quad_kernel(
    double *vx, double *vpx, double *vy, double *vpy, double *vz, double *vpz,
    int *state, double *beta_arr, double *p0c_arr, double *t_arr,
    double mc2, double ks_in, double k1_in, double ele_length,
    double delta_ref_time, double e_tot_ele,
    int n_particles, int n_step,
    /* Multipole parameters */
    const double *d_a2, const double *d_b2, const double *d_cm,
    int ix_mag_max,
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

    /* Body: n_step sol_quad sub-steps */
    for (int istep = 1; istep <= n_step; istep++) {

        double rel_p = 1.0 + vpz[i];

        /* Convert (x, px, y, py) to (x, x', y, y') for the matrix */
        double orb_x  = vx[i];
        double orb_xp = vpx[i] / rel_p;
        double orb_y  = vy[i];
        double orb_yp = vpy[i] / rel_p;

        double k1  = k1_in / rel_p;
        double ks  = ks_in / rel_p;
        double k1_2 = k1 * k1;
        double ks2 = ks * ks;
        double ks3 = ks2 * ks;
        double ks4 = ks2 * ks2;
        double f = sqrt(ks4 + 4.0 * k1_2);
        double ug = 1.0 / (4.0 * f);
        double alpha2 = (f + ks2) / 2.0;
        double alpha = sqrt(alpha2);

        double beta2, beta_sq;
        if (fabs(k1) < 1e-2 * f) {
            double rk = (k1 / ks2) * (k1 / ks2);
            beta2 = ks2 * (rk - rk * rk + 2.0 * rk * rk * rk);
        } else {
            beta2 = (f - ks2) / 2.0;
        }
        beta_sq = sqrt(beta2);

        double S   = sin(alpha * step_len);
        double C   = cos(alpha * step_len);
        double Snh = sinh(beta_sq * step_len);
        double Csh = cosh(beta_sq * step_len);
        double q   = 2.0 * beta2  + 2.0 * k1;
        double r   = 2.0 * alpha2 - 2.0 * k1;
        double a   = 2.0 * alpha2 + 2.0 * k1;
        double b   = 2.0 * beta2  - 2.0 * k1;
        double fp  = f + 2.0 * k1;
        double fm  = f - 2.0 * k1;

        double S1 = S * alpha;
        double S2 = S / alpha;
        double Snh1 = Snh * beta_sq;
        double Snh2 = (fabs(beta_sq) < 1e-10) ? step_len : Snh / beta_sq;

        double coef1 = ks2 * r + 4.0 * k1 * a;
        double coef2 = ks2 * q + 4.0 * k1 * b;

        /* m0 is the transfer matrix in (x, x', y, y') space */
        double m0_11 = 2.0 * ug * (fp * C + fm * Csh);
        double m0_12 = (2.0 * ug / k1) * (q * S1 - r * Snh1);
        double m0_13 = (ks * ug / k1) * (-b * S1 + a * Snh1);
        double m0_14 = 4.0 * ug * ks * (-C + Csh);

        double m0_21 = -(ug / 2.0) * (coef1 * S2 + coef2 * Snh2);
        double m0_22 = m0_11;
        double m0_23 = ug * ks3 * (C - Csh);
        double m0_24 = ug * ks * (a * S2 + b * Snh2);

        double m0_31 = -m0_24;
        double m0_32 = -m0_14;
        double m0_33 = 2.0 * ug * (fm * C + fp * Csh);
        double m0_34 = 2.0 * ug * (r * S2 + q * Snh2);

        double m0_41 = -m0_23;
        double m0_42 = -m0_13;
        double m0_43 = (ug / (2.0 * k1)) * (-coef2 * S1 + coef1 * Snh1);
        double m0_44 = m0_33;

        /* Compute t4: derivative of m0 w.r.t. energy, dm/dE at pz=0.
         * This is needed for the z-correction bilinear form.
         * ts is built from t4, NOT from m0. */
        double df = -2.0 * (ks4 + 2.0 * k1_2) / f;
        double dalpha2 = df / 2.0 - ks2;
        double dalpha = (df / 2.0 - ks2) / (2.0 * alpha);
        double dbeta_v;
        if (k1_2 < 1e-5 * ks4) {
            dbeta_v = fabs(k1 * k1 * k1 / (ks3 * ks2)) * (-1.0 + 3.5 * k1_2 / ks4);
        } else {
            dbeta_v = (ks2 + df / 2.0) / (2.0 * beta_sq);
        }
        double dbeta2v = 2.0 * beta_sq * dbeta_v;
        double darg  = step_len * dalpha;
        double darg1 = step_len * dbeta_v;
        double dC   = -darg * S;
        double dCsh =  darg1 * Snh;
        double dS   =  darg * C;
        double dSnh =  darg1 * Csh;
        double dq   = -2.0 * k1 + 2.0 * dbeta2v;
        double dr   =  2.0 * k1 + 2.0 * dalpha2;
        double da   = -2.0 * k1 + 2.0 * dalpha2;
        double db   =  2.0 * k1 + 2.0 * dbeta2v;
        double dfp  = df - 2.0 * k1;
        double dfm  = df + 2.0 * k1;
        double df_f = -df / f;

        double dS1   = dS * alpha + S * dalpha;
        double dS2   = dS / alpha - S * dalpha / alpha2;
        double dSnh1 = dSnh * beta_sq + Snh * dbeta_v;
        double dSnh2;
        if (k1_2 < 1e-5 * ks4) {
            double L3 = step_len * step_len * step_len;
            dSnh2 = k1_2 * k1_2 * L3 * (-1.0/3.0 + (40.0 - ks2 * L3 / step_len) * k1_2 / (30.0 * ks4)) / (ks3 * ks3);
        } else {
            dSnh2 = dSnh / beta_sq - Snh * dbeta_v / beta2;
        }

        double dcoef1 = -2.0*ks2*r + ks2*dr - 4.0*k1*a + 4.0*k1*da;
        double dcoef2 = -2.0*ks2*q + ks2*dq - 4.0*k1*b + 4.0*k1*db;

        /* t4 matrix (derivative of m0 w.r.t. energy) */
        double t4_11 = m0_11*df_f + 2.0*ug*(fp*dC + C*dfp + fm*dCsh + Csh*dfm);
        double t4_12 = m0_12*df_f + (2.0*ug/k1) * (dq*S1 + q*dS1 - dr*Snh1 - r*dSnh1);
        double t4_13 = m0_13*df_f + (ks*ug/k1)*(-db*S1 - b*dS1 + da*Snh1 + a*dSnh1);
        double t4_14 = m0_14*(df_f - 2.0) + 4.0*ks*ug*(-dC + dCsh);

        double t4_21 = m0_21*(df_f + 1.0) - (ug/2.0)*(dcoef1*S2 + coef1*dS2 + dcoef2*Snh2 + coef2*dSnh2);
        double t4_22 = t4_11;
        double t4_23 = m0_23*(df_f - 2.0) + ks3*ug*(dC - dCsh);
        double t4_24 = m0_24*(df_f - 1.0) + ug*ks*(da*S2 + a*dS2 + db*Snh2 + b*dSnh2);

        double t4_31 = -t4_24;
        double t4_32 = -t4_14;
        double t4_33 = m0_33*df_f + 2.0*ug*(fm*dC + C*dfm + fp*dCsh + Csh*dfp);
        double t4_34 = m0_34*(df_f - 1.0) + 2.0*ug*(dr*S2 + r*dS2 + dq*Snh2 + q*dSnh2);

        double t4_41 = -t4_23;
        double t4_42 = -t4_13;
        double t4_43 = m0_43*(df_f + 2.0) + (ug/(2.0*k1))*(-dcoef2*S1 - coef2*dS1 + dcoef1*Snh1 + coef1*dSnh1);
        double t4_44 = t4_33;

        /* ts is constructed from t4 (NOT m0):
         * ts(1:4,1) = -t4(2,1:4)   ->  ts row1 = -t4_21, -t4_22, -t4_23, -t4_24
         * ts(1:4,2) =  t4(1,1:4)   ->  ts row2 =  t4_11,  t4_12,  t4_13,  t4_14
         * ts(1:4,3) = -t4(4,1:4)   ->  ts row3 = -t4_41, -t4_42, -t4_43, -t4_44
         * ts(1:4,4) =  t4(3,1:4)   ->  ts row4 =  t4_31,  t4_32,  t4_33,  t4_34
         *
         * Note: Fortran ts(i,j) with column-major => ts(1:4,1) sets column 1.
         * ts(1,1) = -t4(2,1), ts(2,1) = -t4(2,2), ts(3,1) = -t4(2,3), ts(4,1) = -t4(2,4)
         * i.e. ts column 1 = -t4 row 2 transposed as a column.
         *
         * In row-major (C) terms:
         * ts_row_1 = [ts(1,1), ts(1,2), ts(1,3), ts(1,4)] = [-t4_21, t4_11, -t4_41, t4_31]
         * ts_row_2 = [ts(2,1), ts(2,2), ts(2,3), ts(2,4)] = [-t4_22, t4_12, -t4_42, t4_32]
         * ts_row_3 = [ts(3,1), ts(3,2), ts(3,3), ts(3,4)] = [-t4_23, t4_13, -t4_43, t4_33]
         * ts_row_4 = [ts(4,1), ts(4,2), ts(4,3), ts(4,4)] = [-t4_24, t4_14, -t4_44, t4_34]
         */
        double ts_11 = -t4_21, ts_12 = t4_11, ts_13 = -t4_41, ts_14 = t4_31;
        double ts_21 = -t4_22, ts_22 = t4_12, ts_23 = -t4_42, ts_24 = t4_32;
        double ts_31 = -t4_23, ts_32 = t4_13, ts_33 = -t4_43, ts_34 = t4_33;
        double ts_41 = -t4_24, ts_42 = t4_14, ts_43 = -t4_44, ts_44 = t4_34;

        /* Compute tsm = ts * m0 */
        double tsm_11 = ts_11*m0_11 + ts_12*m0_21 + ts_13*m0_31 + ts_14*m0_41;
        double tsm_12 = ts_11*m0_12 + ts_12*m0_22 + ts_13*m0_32 + ts_14*m0_42;
        double tsm_13 = ts_11*m0_13 + ts_12*m0_23 + ts_13*m0_33 + ts_14*m0_43;
        double tsm_14 = ts_11*m0_14 + ts_12*m0_24 + ts_13*m0_34 + ts_14*m0_44;

        double tsm_21 = ts_21*m0_11 + ts_22*m0_21 + ts_23*m0_31 + ts_24*m0_41;
        double tsm_22 = ts_21*m0_12 + ts_22*m0_22 + ts_23*m0_32 + ts_24*m0_42;
        double tsm_23 = ts_21*m0_13 + ts_22*m0_23 + ts_23*m0_33 + ts_24*m0_43;
        double tsm_24 = ts_21*m0_14 + ts_22*m0_24 + ts_23*m0_34 + ts_24*m0_44;

        double tsm_31 = ts_31*m0_11 + ts_32*m0_21 + ts_33*m0_31 + ts_34*m0_41;
        double tsm_32 = ts_31*m0_12 + ts_32*m0_22 + ts_33*m0_32 + ts_34*m0_42;
        double tsm_33 = ts_31*m0_13 + ts_32*m0_23 + ts_33*m0_33 + ts_34*m0_43;
        double tsm_34 = ts_31*m0_14 + ts_32*m0_24 + ts_33*m0_34 + ts_34*m0_44;

        double tsm_41 = ts_41*m0_11 + ts_42*m0_21 + ts_43*m0_31 + ts_44*m0_41;
        double tsm_42 = ts_41*m0_12 + ts_42*m0_22 + ts_43*m0_32 + ts_44*m0_42;
        double tsm_43 = ts_41*m0_13 + ts_42*m0_23 + ts_43*m0_33 + ts_44*m0_43;
        double tsm_44 = ts_41*m0_14 + ts_42*m0_24 + ts_43*m0_34 + ts_44*m0_44;

        /* Compute dz = sum_ij dz_coef(i,j) * orbit(i) * orbit(j)
         * where orbit = (x, px, y, py) in the original momentum coordinates.
         * dz_coef = (ts * m0) / 2, with rel_p corrections on rows/cols 2,4 (1-indexed). */
        double rp2 = rel_p * rel_p;
        double dz_x = vx[i], dz_px = vpx[i], dz_y = vy[i], dz_py = vpy[i];

        double dz_val = 0.5 * (
            tsm_11 * dz_x * dz_x +
            tsm_13 * dz_x * dz_y +
            tsm_31 * dz_y * dz_x +
            tsm_33 * dz_y * dz_y +
            (tsm_12 * dz_x * dz_px +
             tsm_21 * dz_px * dz_x +
             tsm_14 * dz_x * dz_py +
             tsm_41 * dz_py * dz_x +
             tsm_32 * dz_y * dz_px +
             tsm_23 * dz_px * dz_y +
             tsm_34 * dz_y * dz_py +
             tsm_43 * dz_py * dz_y) / rel_p +
            (tsm_22 * dz_px * dz_px +
             tsm_24 * dz_px * dz_py +
             tsm_42 * dz_py * dz_px +
             tsm_44 * dz_py * dz_py) / rp2
        );

        /* low_energy_z_correction */
        dz_val += low_energy_z_correction_dev(vpz[i], step_len, beta_val, beta_ref, mc2, e_tot_ele);

        vz[i] += dz_val;

        /* Convert m0 to kmat (px/py coordinates): multiply/divide appropriate rows/cols by rel_p */
        /* kmat(1,2) = m0(1,2)/rel_p, kmat(1,4) = m0(1,4)/rel_p
         * kmat(2,1) = m0(2,1)*rel_p, kmat(2,3) = m0(2,3)*rel_p
         * kmat(3,2) = m0(3,2)/rel_p, kmat(3,4) = m0(3,4)/rel_p
         * kmat(4,1) = m0(4,1)*rel_p, kmat(4,3) = m0(4,3)*rel_p
         * kmat(i,j) = m0(i,j) otherwise for i,j in {1,3} or {2,4} same row/col */
        double km_11 = m0_11;          double km_12 = m0_12 / rel_p;
        double km_13 = m0_13;          double km_14 = m0_14 / rel_p;
        double km_21 = m0_21 * rel_p;  double km_22 = m0_22;
        double km_23 = m0_23 * rel_p;  double km_24 = m0_24;
        double km_31 = m0_31;          double km_32 = m0_32 / rel_p;
        double km_33 = m0_33;          double km_34 = m0_34 / rel_p;
        double km_41 = m0_41 * rel_p;  double km_42 = m0_42;
        double km_43 = m0_43 * rel_p;  double km_44 = m0_44;

        /* Apply kmat to orbit (x, px, y, py) -- note input is in px/py coords */
        double ox = vx[i], opx = vpx[i], oy = vy[i], opy = vpy[i];
        vx[i]  = km_11*ox + km_12*opx + km_13*oy + km_14*opy;
        vpx[i] = km_21*ox + km_22*opx + km_23*oy + km_24*opy;
        vy[i]  = km_31*ox + km_32*opx + km_33*oy + km_34*opy;
        vpy[i] = km_41*ox + km_42*opx + km_43*oy + km_44*opy;

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

    /* Time update: match CPU formula exactly */
    t_arr[i] = t_start + delta_ref_time + (z_start - vz[i]) / (beta_val * C_LIGHT);
}

/* =========================================================================
 * HOST WRAPPER: gpu_track_solenoid  (combined upload + body + download)
 * ========================================================================= */
extern "C" void gpu_track_solenoid_(
    double *h_vx, double *h_vpx, double *h_vy, double *h_vpy,
    double *h_vz, double *h_vpz,
    int *h_state, double *h_beta, double *h_p0c, double *h_t,
    double mc2, double ks0, double ele_length,
    double delta_ref_time, double e_tot_ele,
    int n_particles, int n_step,
    double *h_a2, double *h_b2, double *h_cm,
    int ix_mag_max,
    double *h_ea2, double *h_eb2, int ix_elec_max)
{
    if (ensure_buffers(n_particles) != 0) return;

    if (upload_particle_data(n_particles, h_vx, h_vpx, h_vy, h_vpy, h_vz, h_vpz,
                             h_state, h_beta, h_p0c, h_t) != 0) return;
    if (upload_multipole_data(h_a2, h_b2, h_cm, h_ea2, h_eb2,
                              ix_mag_max, ix_elec_max) != 0) return;

    int threads = 256;
    int blocks = (n_particles + threads - 1) / threads;
    solenoid_kernel<<<blocks, threads>>>(
        d_vec[0], d_vec[1], d_vec[2], d_vec[3], d_vec[4], d_vec[5],
        d_state, d_beta, d_p0c, d_t,
        mc2, ks0, ele_length, delta_ref_time, e_tot_ele,
        n_particles, n_step,
        d_a2, d_b2, d_cm, ix_mag_max,
        d_ea2, d_eb2, ix_elec_max);
    CUDA_CHECK_VOID(cudaGetLastError());
    CUDA_CHECK_VOID(cudaDeviceSynchronize());

    if (download_particle_data(n_particles, h_vx, h_vpx, h_vy, h_vpy, h_vz, h_vpz,
                               h_state, h_beta, h_p0c, h_t,
                               (ix_elec_max >= 0), 0) != 0) return;
}

/* Solenoid body-only: data already on device */
extern "C" void gpu_track_solenoid_dev_(
    double mc2, double ks0, double ele_length,
    double delta_ref_time, double e_tot_ele,
    int n_particles, int n_step,
    double *h_a2, double *h_b2, double *h_cm,
    int ix_mag_max,
    double *h_ea2, double *h_eb2, int ix_elec_max)
{
    if (upload_multipole_data(h_a2, h_b2, h_cm, h_ea2, h_eb2,
                              ix_mag_max, ix_elec_max) != 0) return;

    int threads = 256;
    int blocks = (n_particles + threads - 1) / threads;
    solenoid_kernel<<<blocks, threads>>>(
        d_vec[0], d_vec[1], d_vec[2], d_vec[3], d_vec[4], d_vec[5],
        d_state, d_beta, d_p0c, d_t,
        mc2, ks0, ele_length, delta_ref_time, e_tot_ele,
        n_particles, n_step,
        d_a2, d_b2, d_cm, ix_mag_max,
        d_ea2, d_eb2, ix_elec_max);
    CUDA_CHECK_VOID(cudaGetLastError());
}

/* =========================================================================
 * HOST WRAPPER: gpu_track_sol_quad  (combined upload + body + download)
 * ========================================================================= */
extern "C" void gpu_track_sol_quad_(
    double *h_vx, double *h_vpx, double *h_vy, double *h_vpy,
    double *h_vz, double *h_vpz,
    int *h_state, double *h_beta, double *h_p0c, double *h_t,
    double mc2, double ks_in, double k1_in, double ele_length,
    double delta_ref_time, double e_tot_ele,
    int n_particles, int n_step,
    double *h_a2, double *h_b2, double *h_cm,
    int ix_mag_max,
    double *h_ea2, double *h_eb2, int ix_elec_max)
{
    if (ensure_buffers(n_particles) != 0) return;

    if (upload_particle_data(n_particles, h_vx, h_vpx, h_vy, h_vpy, h_vz, h_vpz,
                             h_state, h_beta, h_p0c, h_t) != 0) return;
    if (upload_multipole_data(h_a2, h_b2, h_cm, h_ea2, h_eb2,
                              ix_mag_max, ix_elec_max) != 0) return;

    int threads = 256;
    int blocks = (n_particles + threads - 1) / threads;
    sol_quad_kernel<<<blocks, threads>>>(
        d_vec[0], d_vec[1], d_vec[2], d_vec[3], d_vec[4], d_vec[5],
        d_state, d_beta, d_p0c, d_t,
        mc2, ks_in, k1_in, ele_length, delta_ref_time, e_tot_ele,
        n_particles, n_step,
        d_a2, d_b2, d_cm, ix_mag_max,
        d_ea2, d_eb2, ix_elec_max);
    CUDA_CHECK_VOID(cudaGetLastError());
    CUDA_CHECK_VOID(cudaDeviceSynchronize());

    if (download_particle_data(n_particles, h_vx, h_vpx, h_vy, h_vpy, h_vz, h_vpz,
                               h_state, h_beta, h_p0c, h_t,
                               (ix_elec_max >= 0), 0) != 0) return;
}

/* Sol_quad body-only: data already on device */
extern "C" void gpu_track_sol_quad_dev_(
    double mc2, double ks_in, double k1_in, double ele_length,
    double delta_ref_time, double e_tot_ele,
    int n_particles, int n_step,
    double *h_a2, double *h_b2, double *h_cm,
    int ix_mag_max,
    double *h_ea2, double *h_eb2, int ix_elec_max)
{
    if (upload_multipole_data(h_a2, h_b2, h_cm, h_ea2, h_eb2,
                              ix_mag_max, ix_elec_max) != 0) return;

    int threads = 256;
    int blocks = (n_particles + threads - 1) / threads;
    sol_quad_kernel<<<blocks, threads>>>(
        d_vec[0], d_vec[1], d_vec[2], d_vec[3], d_vec[4], d_vec[5],
        d_state, d_beta, d_p0c, d_t,
        mc2, ks_in, k1_in, ele_length, delta_ref_time, e_tot_ele,
        n_particles, n_step,
        d_a2, d_b2, d_cm, ix_mag_max,
        d_ea2, d_eb2, ix_elec_max);
    CUDA_CHECK_VOID(cudaGetLastError());
}

/* =========================================================================
 * WIGGLER / UNDULATOR KERNEL
 *
 * Replicates track_a_wiggler for the averaged-field model (planar or helical).
 * The tracking uses:
 *   1. Averaged quadrupole focusing in each plane (k1x, k1y)
 *   2. Octupole kicks at entrance and exit of each sub-step
 *   3. quad_mat2_calc for the 2x2 body transfer in each plane
 *   4. low_energy_z_correction
 *   5. Final time correction for the undulating path length
 *
 * Parameters:
 *   k1x, k1y   -- averaged focusing strengths (pre-computed on host)
 *   kz          -- 2*pi / l_period
 *   is_helical  -- 1 for helical_model$, 0 for planar_model$
 *   osc_amp     -- osc_amplitude$ value
 *   p0c_ele     -- ele%value(p0c$)
 * ========================================================================= */

__global__ void wiggler_kernel(
    double *vx, double *vpx, double *vy, double *vpy, double *vz, double *vpz,
    int *state, double *beta_arr, double *p0c_arr, double *t_arr,
    double mc2, double ele_length,
    double delta_ref_time, double e_tot_ele, double p0c_ele,
    double k1x, double k1y, double kz, int is_helical,
    double osc_amp,
    int n_particles, int n_step,
    /* Multipole parameters */
    const double *d_a2, const double *d_b2, const double *d_cm,
    int ix_mag_max,
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
    double kz2 = kz * kz;

    /* Entrance half magnetic multipole kick */
    if (has_mag) {
        double kx_m, ky_m;
        multipole_kick_dev(d_a2, d_b2, ix_mag_max, d_cm,
                           vx[i], vy[i], &kx_m, &ky_m);
        vpx[i] += 0.5 * kx_m;
        vpy[i] += 0.5 * ky_m;
    }

    /* Entrance half electric multipole kick */
    if (has_elec) {
        double kx_e, ky_e;
        multipole_kick_dev(d_ea2, d_eb2, ix_elec_max, d_cm,
                           vx[i], vy[i], &kx_e, &ky_e);
        if (apply_electric_kick_dev(0.5 * kx_e / beta_val, 0.5 * ky_e / beta_val,
                &vpx[i], &vpy[i], &vpz[i], &vz[i],
                &beta_val, &beta_arr[i], mc2, p0c_val, &state[i])) return;
    }

    /* Body: n_step sub-steps */
    for (int istep = 1; istep <= n_step; istep++) {

        double rel_p = 1.0 + vpz[i];
        double k1yy = k1y / (rel_p * rel_p);

        /* Entrance half octupole kick */
        double k3l = 2.0 * step_len * k1yy;
        if (istep == 1) k3l *= 0.5;

        vpy[i] += k3l * rel_p * kz2 * vy[i] * vy[i] * vy[i] / 3.0;
        if (is_helical) {
            vpx[i] += k3l * rel_p * kz2 * vx[i] * vx[i] * vx[i] / 3.0;
        }

        /* quad_mat2_calc for each plane */
        double cx_x, sx_x, zc_x1, zc_x2, zc_x3;
        double cx_y, sx_y, zc_y1, zc_y2, zc_y3;

        if (is_helical) {
            quad_mat2_calc_dev(k1yy, step_len, rel_p, &cx_x, &sx_x, &zc_x1, &zc_x2, &zc_x3);
        } else {
            quad_mat2_calc_dev(k1x / (rel_p * rel_p), step_len, rel_p, &cx_x, &sx_x, &zc_x1, &zc_x2, &zc_x3);
        }
        quad_mat2_calc_dev(k1yy, step_len, rel_p, &cx_y, &sx_y, &zc_y1, &zc_y2, &zc_y3);

        /* Save pre-matrix coords for z update */
        double x0 = vx[i], px0 = vpx[i];
        double y0 = vy[i], py0 = vpy[i];

        /* z update from quad focusing */
        vz[i] += zc_x1*x0*x0 + zc_x2*x0*px0 + zc_x3*px0*px0 +
                 zc_y1*y0*y0 + zc_y2*y0*py0 + zc_y3*py0*py0;

        /* Apply 2x2 matrices */
        double k_val_x = is_helical ? k1yy : k1x / (rel_p * rel_p);
        vx[i]  = cx_x * x0 + (sx_x / rel_p) * px0;
        vpx[i] = (k_val_x * sx_x * rel_p) * x0 + cx_x * px0;
        vy[i]  = cx_y * y0 + (sx_y / rel_p) * py0;
        vpy[i] = (k1yy * sx_y * rel_p) * y0 + cx_y * py0;

        /* Low energy z correction */
        vz[i] += low_energy_z_correction_dev(vpz[i], step_len, beta_val, beta_ref, mc2, e_tot_ele);

        /* Exit half octupole kick */
        k3l = 2.0 * step_len * k1yy;
        if (istep == n_step) k3l *= 0.5;

        vpy[i] += k3l * rel_p * kz2 * vy[i] * vy[i] * vy[i] / 3.0;
        if (is_helical) {
            vpx[i] += k3l * rel_p * kz2 * vx[i] * vx[i] * vx[i] / 3.0;
        }

        /* Magnetic multipole kick (half at last step, full otherwise) */
        if (has_mag) {
            double kx_m, ky_m;
            multipole_kick_dev(d_a2, d_b2, ix_mag_max, d_cm,
                               vx[i], vy[i], &kx_m, &ky_m);
            double scl = (istep == n_step) ? 0.5 : 1.0;
            vpx[i] += scl * kx_m;
            vpy[i] += scl * ky_m;
        }

        /* Electric multipole kick (half at last step, full otherwise) */
        if (has_elec) {
            double kx_e, ky_e;
            multipole_kick_dev(d_ea2, d_eb2, ix_elec_max, d_cm,
                               vx[i], vy[i], &kx_e, &ky_e);
            double scl = (istep == n_step) ? 0.5 : 1.0;
            if (apply_electric_kick_dev(scl * kx_e / beta_val, scl * ky_e / beta_val,
                    &vpx[i], &vpy[i], &vpz[i], &vz[i],
                    &beta_val, &beta_arr[i], mc2, p0c_val, &state[i])) return;
        }
    }

    /* Final time: standard formula matching CPU */
    double rel_p = 1.0 + vpz[i];
    t_arr[i] = t_start + delta_ref_time + (z_start - vz[i]) / (beta_val * C_LIGHT);

    /* Undulation path length correction -- matches CPU track_a_wiggler final section.
     * For helical: factor = length * (kz * osc_amp)^2 / 2
     * For planar:  factor = length * (kz * osc_amp)^2 / 4  */
    double factor;
    if (is_helical) {
        factor = ele_length * (kz * osc_amp) * (kz * osc_amp) / 2.0;
    } else {
        factor = ele_length * (kz * osc_amp) * (kz * osc_amp) / 4.0;
    }

    t_arr[i] += factor / (C_LIGHT * beta_val * rel_p * rel_p);
    vz[i]    += factor * (beta_val / beta_ref - 1.0 / (rel_p * rel_p));
}

/* =========================================================================
 * HOST WRAPPER: gpu_track_wiggler (combined upload + body + download)
 * ========================================================================= */
extern "C" void gpu_track_wiggler_(
    double *h_vx, double *h_vpx, double *h_vy, double *h_vpy,
    double *h_vz, double *h_vpz,
    int *h_state, double *h_beta, double *h_p0c, double *h_t,
    double mc2, double ele_length,
    double delta_ref_time, double e_tot_ele, double p0c_ele,
    double k1x, double k1y, double kz, int is_helical,
    double osc_amp,
    int n_particles, int n_step,
    double *h_a2, double *h_b2, double *h_cm,
    int ix_mag_max,
    double *h_ea2, double *h_eb2, int ix_elec_max)
{
    if (ensure_buffers(n_particles) != 0) return;

    if (upload_particle_data(n_particles, h_vx, h_vpx, h_vy, h_vpy, h_vz, h_vpz,
                             h_state, h_beta, h_p0c, h_t) != 0) return;
    if (upload_multipole_data(h_a2, h_b2, h_cm, h_ea2, h_eb2,
                              ix_mag_max, ix_elec_max) != 0) return;

    int threads = 256;
    int blocks = (n_particles + threads - 1) / threads;
    wiggler_kernel<<<blocks, threads>>>(
        d_vec[0], d_vec[1], d_vec[2], d_vec[3], d_vec[4], d_vec[5],
        d_state, d_beta, d_p0c, d_t,
        mc2, ele_length, delta_ref_time, e_tot_ele, p0c_ele,
        k1x, k1y, kz, is_helical, osc_amp,
        n_particles, n_step,
        d_a2, d_b2, d_cm, ix_mag_max,
        d_ea2, d_eb2, ix_elec_max);
    CUDA_CHECK_VOID(cudaGetLastError());
    CUDA_CHECK_VOID(cudaDeviceSynchronize());

    if (download_particle_data(n_particles, h_vx, h_vpx, h_vy, h_vpy, h_vz, h_vpz,
                               h_state, h_beta, h_p0c, h_t,
                               (ix_elec_max >= 0), 0) != 0) return;
}

/* Wiggler body-only: data already on device */
extern "C" void gpu_track_wiggler_dev_(
    double mc2, double ele_length,
    double delta_ref_time, double e_tot_ele, double p0c_ele,
    double k1x, double k1y, double kz, int is_helical,
    double osc_amp,
    int n_particles, int n_step,
    double *h_a2, double *h_b2, double *h_cm,
    int ix_mag_max,
    double *h_ea2, double *h_eb2, int ix_elec_max)
{
    if (upload_multipole_data(h_a2, h_b2, h_cm, h_ea2, h_eb2,
                              ix_mag_max, ix_elec_max) != 0) return;

    int threads = 256;
    int blocks = (n_particles + threads - 1) / threads;
    wiggler_kernel<<<blocks, threads>>>(
        d_vec[0], d_vec[1], d_vec[2], d_vec[3], d_vec[4], d_vec[5],
        d_state, d_beta, d_p0c, d_t,
        mc2, ele_length, delta_ref_time, e_tot_ele, p0c_ele,
        k1x, k1y, kz, is_helical, osc_amp,
        n_particles, n_step,
        d_a2, d_b2, d_cm, ix_mag_max,
        d_ea2, d_eb2, ix_elec_max);
    CUDA_CHECK_VOID(cudaGetLastError());
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
    double *h_ea2, double *h_eb2, int ix_elec_max,
    int is_exact,
    double *h_exact_an, double *h_exact_bn,
    int ix_exact_mag_max,
    double rho, double c_dir, double exact_f_scale)
{
    if (upload_multipole_data(h_a2, h_b2, h_cm, h_ea2, h_eb2,
                              ix_mag_max, ix_elec_max) != 0) return;
    if (is_exact) {
        if (upload_exact_multipole_data(h_exact_an, h_exact_bn, ix_exact_mag_max) != 0) return;
    }

    int threads = 256;
    int blocks = (n_particles + threads - 1) / threads;
    bend_kernel<<<blocks, threads>>>(
        d_vec[0], d_vec[1], d_vec[2], d_vec[3], d_vec[4], d_vec[5],
        d_state, d_beta, d_p0c, d_t,
        mc2, g, g_tot, dg, b1, ele_length, delta_ref_time, e_tot_ele,
        rel_charge_dir, p0c_ele,
        n_particles, d_a2, d_b2, d_cm, ix_mag_max, n_step,
        d_ea2, d_eb2, ix_elec_max,
        is_exact, d_exact_an, d_exact_bn, ix_exact_mag_max,
        rho, c_dir, exact_f_scale);
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

/* General hard multipole edge kick (port of hard_multipole_edge_kick).
 * Handles arbitrary n_max: n=1 for bends/quads (k1), n=2 for sextupoles (k2), etc.
 * bp/ap arrays are in constant memory, indexed [0..n_max-1] for orders n=1..n_max. */
#define MAX_HARD_MULTI_ORDER 8
static __constant__ double c_hard_bp[MAX_HARD_MULTI_ORDER];
static __constant__ double c_hard_ap[MAX_HARD_MULTI_ORDER];

__global__ void hard_multipole_edge_kernel(
    double *vx, double *vpx, double *vy, double *vpy, double *vz, double *vpz,
    int *state, int n_max, double charge_dir, int is_entrance, int n_particles)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n_particles) return;
    if (state[i] != ALIVE_ST) return;

    double x = vx[i], y = vy[i], px = vpx[i], py = vpy[i];
    double rel_p = 1.0 + vpz[i];
    if (rel_p <= 0) return;

    /* Complex polynomial iteration: poly = (x+iy)^{n+1}, poly_n1 = (x+iy)^n */
    double pn1_r = 1.0, pn1_i = 0.0;  /* poly_{n-1}, starts at 1 */
    double poly_r = x, poly_i = y;     /* poly, starts at x+iy */

    double fx = 0, dfx_dx = 0, dfx_dy = 0;
    double fy = 0, dfy_dx = 0, dfy_dy = 0;

    for (int nn = 1; nn <= n_max; nn++) {
        /* Advance: poly_n1 = poly, poly = poly * (x+iy) */
        pn1_r = poly_r; pn1_i = poly_i;
        double new_r = poly_r * x - poly_i * y;
        double new_i = poly_r * y + poly_i * x;
        poly_r = new_r; poly_i = new_i;

        double bp = c_hard_bp[nn - 1];
        double ap = c_hard_ap[nn - 1];
        if (bp == 0.0 && ap == 0.0) continue;

        /* cab = charge_dir * (bp + i*ap) / (4*(n+2)*rel_p) */
        double scale = charge_dir / (4.0 * (nn + 2) * rel_p);
        double cab_r = bp * scale;
        double cab_i = ap * scale;
        if (is_entrance) { cab_r = -cab_r; cab_i = -cab_i; }

        /* dpoly_dx = (n+1) * poly_n1 */
        double dpx_r = (nn + 1) * pn1_r;
        double dpx_i = (nn + 1) * pn1_i;
        /* dpoly_dy = i * dpoly_dx */
        double dpy_r = -dpx_i;
        double dpy_i = dpx_r;

        double cn = (double)(nn + 3) / (double)(nn + 1);

        /* --- fx terms: xny = (x, -cn*y) --- */
        double xny_r = x, xny_i = -cn * y;

        /* poly * xny */
        double pxny_r = poly_r * xny_r - poly_i * xny_i;
        double pxny_i = poly_r * xny_i + poly_i * xny_r;
        /* cab * poly * xny */
        fx += cab_r * pxny_r - cab_i * pxny_i;

        /* dfx_dx += Re(cab * (dpoly_dx * xny + poly)) */
        double t1_r = dpx_r * xny_r - dpx_i * xny_i + poly_r;
        double t1_i = dpx_r * xny_i + dpx_i * xny_r + poly_i;
        dfx_dx += cab_r * t1_r - cab_i * t1_i;

        /* dfx_dy += Re(cab * (dpoly_dy * xny + poly * dxny_dy))
         * dxny_dy = (0, -cn) */
        double pdxnydy_r = poly_i * cn;   /* Re(poly * (0 + -cn*i)) = poly_i * cn */
        double pdxnydy_i = -poly_r * cn;  /* Im(poly * (0 + -cn*i)) = -poly_r * cn */
        double t2_r = dpy_r * xny_r - dpy_i * xny_i + pdxnydy_r;
        double t2_i = dpy_r * xny_i + dpy_i * xny_r + pdxnydy_i;
        dfx_dy += cab_r * t2_r - cab_i * t2_i;

        /* --- fy terms: xny2 = (y, cn*x) --- */
        double xny2_r = y, xny2_i = cn * x;

        /* poly * xny2 */
        double pxny2_r = poly_r * xny2_r - poly_i * xny2_i;
        double pxny2_i = poly_r * xny2_i + poly_i * xny2_r;
        /* cab * poly * xny2 */
        fy += cab_r * pxny2_r - cab_i * pxny2_i;

        /* dfy_dx += Re(cab * (dpoly_dx * xny2 + poly * dxny2_dx))
         * dxny2_dx = (0, cn) */
        double pdxny2dx_r = -poly_i * cn;  /* Re(poly * (0 + cn*i)) = -poly_i * cn */
        double pdxny2dx_i = poly_r * cn;   /* Im(poly * (0 + cn*i)) = poly_r * cn */
        double t3_r = dpx_r * xny2_r - dpx_i * xny2_i + pdxny2dx_r;
        double t3_i = dpx_r * xny2_i + dpx_i * xny2_r + pdxny2dx_i;
        dfy_dx += cab_r * t3_r - cab_i * t3_i;

        /* dfy_dy += Re(cab * (dpoly_dy * xny2 + poly)) */
        double t4_r = dpy_r * xny2_r - dpy_i * xny2_i + poly_r;
        double t4_i = dpy_r * xny2_i + dpy_i * xny2_r + poly_i;
        dfy_dy += cab_r * t4_r - cab_i * t4_i;
    }

    /* Symplectic kick */
    double denom = (1.0 - dfx_dx) * (1.0 - dfy_dy) - dfx_dy * dfy_dx;
    if (fabs(denom) < 1e-30) return;

    vx[i] = x - fx;
    vpx[i] = px + ((1.0 - dfy_dy - denom) * px + dfy_dx * py) / denom;
    vy[i] = y - fy;
    vpy[i] = py + (dfx_dy * px + (1.0 - dfx_dx - denom) * py) / denom;
    vz[i] = vz[i] + (vpx[i] * fx + vpy[i] * fy) / rel_p;
}

extern "C" void gpu_hard_multipole_edge_(
    double *h_bp, double *h_ap, int n_max,
    double charge_dir, int is_entrance, int n_particles)
{
    if (n_particles <= 0 || n_max <= 0 || n_max > MAX_HARD_MULTI_ORDER) return;
    cudaMemcpyToSymbol(c_hard_bp, h_bp, n_max * sizeof(double));
    cudaMemcpyToSymbol(c_hard_ap, h_ap, n_max * sizeof(double));
    int threads = 256;
    int blocks = (n_particles + threads - 1) / threads;
    hard_multipole_edge_kernel<<<blocks, threads>>>(
        d_vec[0], d_vec[1], d_vec[2], d_vec[3], d_vec[4], d_vec[5],
        d_state, n_max, charge_dir, is_entrance, n_particles);
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


/* ==========================================================================
 * SAD soft-edge bend fringe kernel
 *
 * Port of sad_soft_bend_edge_kick from fringe_mod.f90 (sbend case only).
 * Parameters are pre-signed by the Fortran dispatch:
 *   g  = (g$ + dg$) * c_dir, negated if second_track_edge$
 *        where c_dir = charge_to_mass * orientation * direction * time_dir
 *   fb = 12 * fint * hgap  (or fintx * hgapx for exit end)
 * ========================================================================== */

__global__ void sad_bend_fringe_kernel(
    double *vx, double *vpx, double *vy, double *vpy, double *vz,
    const double *vpz, int *state,
    double g, double fb, int n)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    if (state[i] != ALIVE_ST) return;

    double px    = vpx[i];
    double y     = vy[i];
    double rel_p = 1.0 + vpz[i];

    double c1 = fb * fb * g / (24.0 * rel_p);
    double c2 = fb * g * g / (6.0 * rel_p);
    double c3 = 2.0 * g * g / (3.0 * fb * rel_p);

    vx[i]  = vx[i]  + c1 * vpz[i];
    vpy[i] = vpy[i] + c2 * y - c3 * y * y * y;
    vz[i]  = vz[i]  + (c1 * px + c2 * y * y / 2.0 - c3 * y * y * y * y / 4.0) / rel_p;
}

extern "C" void gpu_sad_bend_fringe_(double g, double fb, int n)
{
    if (n <= 0) return;
    int threads = 256;
    int blocks = (n + threads - 1) / threads;
    sad_bend_fringe_kernel<<<blocks, threads>>>(
        d_vec[0], d_vec[1], d_vec[2], d_vec[3], d_vec[4],
        d_vec[5], d_state,
        g, fb, n);
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
 * BEND OFFSET KERNEL -- curvature-aware offset_particle for sbends
 *
 * Replicates the CPU offset_particle bend path (set$ and unset$):
 *   set$:   lab -> body using bend_shift transforms
 *   unset$: body -> lab using inverse bend_shift transforms
 *
 * Assumes direction=1, time_dir=1, orientation=1 (GPU forward tracking).
 * ========================================================================== */

/* ---------- cos_one: cos(a)-1 to machine precision ---------- */
__device__ __forceinline__ double cos_one_dev(double a) {
    double s = sin(a * 0.5);
    return -2.0 * s * s;
}

/* ---------- 3x3 identity ---------- */
__device__ void mat3_identity(double M[3][3]) {
    for (int i = 0; i < 3; i++)
        for (int j = 0; j < 3; j++)
            M[i][j] = (i == j) ? 1.0 : 0.0;
}

/* ---------- left-multiply 3x3: M = R * M  (in-place) ---------- */
__device__ void mat3_lmul(double R[3][3], double M[3][3]) {
    double T[3][3];
    for (int i = 0; i < 3; i++)
        for (int j = 0; j < 3; j++) {
            T[i][j] = 0.0;
            for (int k = 0; k < 3; k++) T[i][j] += R[i][k] * M[k][j];
        }
    for (int i = 0; i < 3; i++)
        for (int j = 0; j < 3; j++) M[i][j] = T[i][j];
}

/* ---------- mat-vec: out = M * v ---------- */
__device__ void mat3_vec(double M[3][3], const double v[3], double out[3]) {
    for (int i = 0; i < 3; i++) {
        out[i] = 0.0;
        for (int j = 0; j < 3; j++) out[i] += M[i][j] * v[j];
    }
}

/* ---------- rotate_mat: left-multiply M by rotation about axis by angle ---------- */
/* axis: 0=x, 1=y, 2=z */
__device__ void rotate_mat_dev(double M[3][3], int axis, double angle) {
    if (angle == 0.0) return;
    double ca = cos(angle), sa = sin(angle);
    /* Build rotation R, then M <- R * M */
    double R[3][3];
    mat3_identity(R);
    switch (axis) {
    case 0: /* x-axis */
        R[1][1] = ca; R[1][2] = -sa;
        R[2][1] = sa; R[2][2] =  ca;
        break;
    case 1: /* y-axis */
        R[0][0] =  ca; R[0][2] = sa;
        R[2][0] = -sa; R[2][2] = ca;
        break;
    case 2: /* z-axis */
        R[0][0] = ca; R[0][1] = -sa;
        R[1][0] = sa; R[1][1] =  ca;
        break;
    }
    mat3_lmul(R, M);
}

/* ---------- rotate_vec: rotate vec about axis by angle ---------- */
__device__ void rotate_vec_dev(double v[3], int axis, double angle) {
    if (angle == 0.0) return;
    double ca = cos(angle), sa = sin(angle);
    double t;
    switch (axis) {
    case 0: /* x */
        t    = ca*v[1] - sa*v[2];
        v[2] = sa*v[1] + ca*v[2];
        v[1] = t;
        break;
    case 1: /* y */
        t    = sa*v[2] + ca*v[0];
        v[2] = ca*v[2] - sa*v[0];
        v[0] = t;
        break;
    case 2: /* z */
        t    = ca*v[0] - sa*v[1];
        v[1] = sa*v[0] + ca*v[1];
        v[0] = t;
        break;
    }
}

/* ---------- bend_shift_dev: port of Fortran bend_shift ---------- */
/* Transforms position (r, w) by shifting along a bend with curvature g
 * by arc length delta_s, with optional ref_tilt. */
__device__ void bend_shift_dev(double r[3], double w[3][3],
                               double g, double delta_s, double ref_tilt) {
    double angle = delta_s * g;

    if (angle == 0.0) {
        r[2] -= delta_s;
        return;
    }

    /* Build S_mat */
    double S[3][3];
    mat3_identity(S);
    if (ref_tilt != 0.0) {
        rotate_mat_dev(S, 2, -ref_tilt);  /* z-axis, -ref_tilt */
        rotate_mat_dev(S, 1, angle);      /* y-axis, angle */
        rotate_mat_dev(S, 2, ref_tilt);   /* z-axis, ref_tilt */
    } else {
        rotate_mat_dev(S, 1, angle);      /* y-axis, angle */
    }

    /* L_vec = [cos_one(angle), 0, -sin(angle)] / g */
    double L[3];
    L[0] = cos_one_dev(angle) / g;
    L[1] = 0.0;
    L[2] = -sin(angle) / g;
    if (ref_tilt != 0.0) rotate_vec_dev(L, 2, ref_tilt);

    /* r_new = S * r + L */
    double rn[3];
    mat3_vec(S, r, rn);
    rn[0] += L[0]; rn[1] += L[1]; rn[2] += L[2];
    r[0] = rn[0]; r[1] = rn[1]; r[2] = rn[2];

    /* w_new = S * w */
    mat3_lmul(S, w);
}

/* ---------- floor_angles_to_w_mat_dev: (theta=x_pitch, phi=y_pitch, psi=roll) ---------- */
__device__ void floor_angles_to_w_mat_dev(double theta, double phi, double psi,
                                           double wm[3][3]) {
    double st = sin(theta), ct = cos(theta);
    double sp = sin(phi),   cp = cos(phi);
    double ss = sin(psi),   cs = cos(psi);

    wm[0][0] =  ct*cs - st*sp*ss;
    wm[0][1] = -ct*ss - st*sp*cs;
    wm[0][2] =  st*cp;
    wm[1][0] =  cp*ss;
    wm[1][1] =  cp*cs;
    wm[1][2] =  sp;
    wm[2][0] = -st*cs - ct*sp*ss;
    wm[2][1] =  st*ss - ct*sp*cs;
    wm[2][2] =  ct*cp;
}

/* ---------- floor_angles_to_w_mat_inv_dev: inverse (= transpose for orthogonal) ---------- */
__device__ void floor_angles_to_w_mat_inv_dev(double theta, double phi, double psi,
                                               double wm[3][3]) {
    double st = sin(theta), ct = cos(theta);
    double sp = sin(phi),   cp = cos(phi);
    double ss = sin(psi),   cs = cos(psi);

    wm[0][0] =  ct*cs - st*sp*ss;
    wm[0][1] =  cp*ss;
    wm[0][2] = -st*cs - ct*sp*ss;
    wm[1][0] = -ct*ss - st*sp*cs;
    wm[1][1] =  cp*cs;
    wm[1][2] =  st*ss - ct*sp*cs;
    wm[2][0] =  st*cp;
    wm[2][1] =  sp;
    wm[2][2] =  ct*cp;
}

/* ---------- w_mat_for_tilt_dev: z-axis rotation matrix for tilt ---------- */
__device__ void w_mat_for_tilt_dev(double tilt, double wm[3][3]) {
    double c = cos(tilt), s = sin(tilt);
    wm[0][0] = c;   wm[0][1] = -s;  wm[0][2] = 0.0;
    wm[1][0] = s;   wm[1][1] =  c;  wm[1][2] = 0.0;
    wm[2][0] = 0.0; wm[2][1] = 0.0; wm[2][2] = 1.0;
}

/* ---------- ele_misalignment_L_S_calc_dev: for sbends ---------- */
__device__ void ele_misalignment_L_S_calc_dev(
    double x_off, double y_off, double z_off,
    double x_pitch, double y_pitch,  /* Note: x_pitch$ and y_pitch$ (not _tot) for bends */
    double roll_tot, double ref_tilt_tot,
    double bend_angle, double rho,
    double L_mis[3], double S_mis[3][3]) {

    L_mis[0] = x_off;
    L_mis[1] = y_off;
    L_mis[2] = z_off;

    /* Roll contribution to L_mis */
    if (roll_tot != 0.0) {
        double ha = bend_angle * 0.5;
        double Lc[3];
        Lc[0] = rho * cos_one_dev(ha);
        Lc[1] = 0.0;
        Lc[2] = rho * sin(ha);
        rotate_vec_dev(Lc, 1, ha);           /* y-axis by half angle */
        rotate_vec_dev(Lc, 2, ref_tilt_tot); /* z-axis by ref_tilt */
        L_mis[0] -= Lc[0];
        L_mis[1] -= Lc[1];
        L_mis[2] -= Lc[2];
        rotate_vec_dev(Lc, 2, roll_tot);     /* z-axis by roll */
        L_mis[0] += Lc[0];
        L_mis[1] += Lc[1];
        L_mis[2] += Lc[2];
    }

    /* S_mis = floor_angles_to_w_mat(x_pitch, y_pitch, roll_tot) */
    floor_angles_to_w_mat_dev(x_pitch, y_pitch, roll_tot, S_mis);
}

/* ---------- transpose a 3x3 in-place ---------- */
__device__ void mat3_transpose(double M[3][3]) {
    double t;
    t = M[0][1]; M[0][1] = M[1][0]; M[1][0] = t;
    t = M[0][2]; M[0][2] = M[2][0]; M[2][0] = t;
    t = M[1][2]; M[1][2] = M[2][1]; M[2][1] = t;
}

/* ---------- The main bend_offset kernel ----------
 *
 * Parameters (all pre-computed on host and passed as scalars):
 *   g          = curvature (1/rho)
 *   rho        = radius of curvature (rho$)
 *   L_half     = half-length
 *   bend_angle = g * length = angle$
 *   ref_tilt   = ref_tilt_tot$
 *   roll_tot   = roll_tot$
 *   x_off, y_off, z_off = offset_tot values
 *   x_pitch, y_pitch    = pitch$ values (not _tot)
 *   set_flag   = 1 for set (lab->body), -1 for unset (body->lab)
 *
 * Assumptions (GPU forward tracking):
 *   direction = 1, time_dir = 1, orientation = 1
 *   For set:   particle at entrance (s_pos0=0, ds_center=L_half)
 *   For unset: particle at exit     (s_pos0=length, ds_center=-L_half)
 */
__global__ void bend_offset_kernel(
    double *vx, double *vpx, double *vy, double *vpy,
    double *vz, double *vpz,
    int *state,
    double g, double rho, double L_half, double bend_angle,
    double ref_tilt, double roll_tot,
    double x_off, double y_off, double z_off,
    double x_pitch, double y_pitch,
    int set_flag, int n)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    if (state[i] != ALIVE_ST) return;

    double x = vx[i], px = vpx[i], y = vy[i], py = vpy[i];
    double rel_p = 1.0 + vpz[i];
    /* sign_z_vel = orientation * direction = 1 for GPU */
    int sign_z_vel = 1;

    /* ds_center: set -> L_half, unset -> -L_half */
    double ds_center = (set_flag == 1) ? L_half : -L_half;

    /* position: r = [x, y, 0], w = I */
    double r[3] = {x, y, 0.0};
    double w[3][3];
    mat3_identity(w);

    if (set_flag == 1) {
        /* ============== SET (lab -> body) ============== */

        /* Step 1: bend_shift from particle s_pos to element center */
        /* ds = orientation * ds_center = 1 * L_half = L_half */
        bend_shift_dev(r, w, g, L_half, ref_tilt);

        /* Step 2: ele_misalignment_L_S_calc -> L_mis, S_mis */
        double L_mis[3], S_mis[3][3];
        ele_misalignment_L_S_calc_dev(x_off, y_off, z_off,
            x_pitch, y_pitch, roll_tot, ref_tilt, bend_angle, rho,
            L_mis, S_mis);

        /* Step 3: position%r = transpose(S_mis) * (r - L_mis) */
        /*         position%w = transpose(S_mis) * w */
        double ws[3][3];
        /* ws = transpose(S_mis) */
        for (int ii = 0; ii < 3; ii++)
            for (int jj = 0; jj < 3; jj++)
                ws[ii][jj] = S_mis[jj][ii];

        double rd[3] = {r[0] - L_mis[0], r[1] - L_mis[1], r[2] - L_mis[2]};
        double rn[3];
        mat3_vec(ws, rd, rn);
        r[0] = rn[0]; r[1] = rn[1]; r[2] = rn[2];
        mat3_lmul(ws, w);

        /* Step 4: ref_tilt correction (if needed) */
        if (ref_tilt != 0.0) {
            bend_shift_dev(r, w, g, -L_half, ref_tilt);

            /* ws = w_mat_for_tilt(-ref_tilt) */
            w_mat_for_tilt_dev(-ref_tilt, ws);
            double rt[3];
            mat3_vec(ws, r, rt);
            r[0] = rt[0]; r[1] = rt[1]; r[2] = rt[2];
            mat3_lmul(ws, w);

            bend_shift_dev(r, w, g, L_half, 0.0);
        }

        /* Step 5: drift_to = upstream_end$
         * sign_z_vel * time_dir = 1, so shift by -L_half */
        bend_shift_dev(r, w, g, -L_half, 0.0);
        /* r[2] += (1 - sign_z_vel * time_dir) * L_half = 0 */
        /* s_body = r[2] (which should be ~0) */
        double s_body = r[2];

        /* Step 6: transform momenta */
        double pz_sq = rel_p * rel_p - px * px - py * py;
        if (pz_sq <= 0.0) { state[i] = LOST_PZ; return; }
        double pz_val = sign_z_vel * sqrt(pz_sq);

        double p_vec[3];
        double p_vec0[3] = {px, py, pz_val};
        mat3_vec(w, p_vec0, p_vec);

        vx[i]  = r[0];
        vpx[i] = p_vec[0];
        vy[i]  = r[1];
        vpy[i] = p_vec[1];

        /* Step 7: drift to edge (s_target = 0 for upstream_end$) */
        /* ds = s_target - s_body = 0 - s_body = -s_body */
        double ds = -s_body;
        if (fabs(ds) > 1e-14) {
            /* track_a_drift with include_ref_motion=false, orientation=1 */
            double rpx = vx[i], rppx = vpx[i], rpy = vy[i], rppy = vpy[i];
            double drp = 1.0 + vpz[i];
            double px_rel = vpx[i] / drp;
            double py_rel = vpy[i] / drp;
            double pxy2 = px_rel * px_rel + py_rel * py_rel;
            if (pxy2 >= 1.0) { state[i] = LOST_PZ; return; }
            double ps_rel = sqrt(1.0 - pxy2);
            vx[i] = vx[i] + ds * px_rel / ps_rel;
            vy[i] = vy[i] + ds * py_rel / ps_rel;
            /* vz change: dz = -ds / ps_rel  (include_ref_motion=false) */
            vz[i] = vz[i] - ds / ps_rel;
        }

    } else {
        /* ============== UNSET (body -> lab) ============== */

        /* Step 1: bend_shift from particle s_pos to center */
        /* ds_center = -L_half (particle is at exit, s_pos0=length=2*L_half) */
        bend_shift_dev(r, w, g, -L_half, 0.0);

        /* Step 2: ref_tilt correction (if needed) */
        if (ref_tilt != 0.0) {
            bend_shift_dev(r, w, g, -L_half, 0.0);

            /* ws = w_mat_for_tilt(ref_tilt) */
            double ws2[3][3];
            w_mat_for_tilt_dev(ref_tilt, ws2);
            double rt2[3];
            mat3_vec(ws2, r, rt2);
            r[0] = rt2[0]; r[1] = rt2[1]; r[2] = rt2[2];
            mat3_lmul(ws2, w);

            bend_shift_dev(r, w, g, L_half, ref_tilt);
        }

        /* Step 3: ele_misalignment_L_S_calc -> L_mis, S_mis */
        double L_mis[3], S_mis[3][3];
        ele_misalignment_L_S_calc_dev(x_off, y_off, z_off,
            x_pitch, y_pitch, roll_tot, ref_tilt, bend_angle, rho,
            L_mis, S_mis);

        /* Step 4: position%r = S_mis * r + L_mis */
        /*         position%w = S_mis * w */
        double rn[3];
        mat3_vec(S_mis, r, rn);
        r[0] = rn[0] + L_mis[0]; r[1] = rn[1] + L_mis[1]; r[2] = rn[2] + L_mis[2];
        mat3_lmul(S_mis, w);

        /* Step 5: drift_to = downstream_end$
         * sign_z_vel * time_dir = 1, shift by +L_half */
        bend_shift_dev(r, w, g, L_half, ref_tilt);
        /* r[2] += (1 + sign_z_vel * time_dir) * L_half = r[2] + 2*L_half */
        r[2] += 2.0 * L_half;

        /* s_lab = r[2]  (should be ~length = 2*L_half) */
        double s_lab = r[2];

        /* Step 6: transform momenta */
        double pz_sq = rel_p * rel_p - px * px - py * py;
        if (pz_sq <= 0.0) { state[i] = LOST_PZ; return; }
        double pz_val = sign_z_vel * sqrt(pz_sq);

        double p_vec[3];
        double p_vec0[3] = {px, py, pz_val};
        mat3_vec(w, p_vec0, p_vec);

        vx[i]  = r[0];
        vpx[i] = p_vec[0];
        vy[i]  = r[1];
        vpy[i] = p_vec[1];

        /* Step 7: drift to edge (s_target = 2*L_half for downstream_end$) */
        /* ds = s_target - s_lab = 2*L_half - s_lab */
        double ds = 2.0 * L_half - s_lab;
        if (fabs(ds) > 1e-14) {
            /* track_a_drift with include_ref_motion=false, orientation=+1 */
            double drp = 1.0 + vpz[i];
            double px_rel = vpx[i] / drp;
            double py_rel = vpy[i] / drp;
            double pxy2 = px_rel * px_rel + py_rel * py_rel;
            if (pxy2 >= 1.0) { state[i] = LOST_PZ; return; }
            double ps_rel = sqrt(1.0 - pxy2);
            vx[i] = vx[i] + ds * px_rel / ps_rel;
            vy[i] = vy[i] + ds * py_rel / ps_rel;
            /* vz change: dz = -ds / ps_rel (include_ref_motion=false) */
            vz[i] = vz[i] - ds / ps_rel;
        }
    }
}

extern "C" void gpu_bend_offset_(
    double g, double rho, double L_half, double bend_angle,
    double ref_tilt, double roll_tot,
    double x_off, double y_off, double z_off,
    double x_pitch, double y_pitch,
    int set_flag, int n)
{
    if (n <= 0) return;
    int threads = 256;
    int blocks = (n + threads - 1) / threads;
    bend_offset_kernel<<<blocks, threads>>>(
        d_vec[0], d_vec[1], d_vec[2], d_vec[3],
        d_vec[4], d_vec[5], d_state,
        g, rho, L_half, bend_angle,
        ref_tilt, roll_tot,
        x_off, y_off, z_off,
        x_pitch, y_pitch,
        set_flag, n);
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
 * gpu_download_first_p0c -- download first particle's p0c from device.
 * Used by precompute_multipole_arrays when host p0c is stale.
 * -------------------------------------------------------------------------- */
extern "C" void gpu_download_first_p0c_(double *h_p0c)
{
    if (d_p0c) {
        cudaMemcpy(h_p0c, d_p0c, sizeof(double), cudaMemcpyDeviceToHost);
    } else {
        *h_p0c = 1.0;  /* fallback: no device data */
    }
}

/* --------------------------------------------------------------------------
 * gpu_nan_check -- check for NaN/Inf in particle coordinates on device.
 * Returns the number of alive particles with NaN/Inf in any coordinate.
 * Enable with GPU_NAN_CHECK=1 environment variable.
 * -------------------------------------------------------------------------- */
__global__ void nan_check_kernel(
    const double *vx, const double *vpx, const double *vy, const double *vpy,
    const double *vz, const double *vpz, const int *state,
    int *nan_count, int n)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    if (state[i] != ALIVE_ST) return;
    if (isnan(vx[i]) || isinf(vx[i]) ||
        isnan(vpx[i]) || isinf(vpx[i]) ||
        isnan(vy[i]) || isinf(vy[i]) ||
        isnan(vpy[i]) || isinf(vpy[i]) ||
        isnan(vz[i]) || isinf(vz[i]) ||
        isnan(vpz[i]) || isinf(vpz[i])) {
        atomicAdd(nan_count, 1);
    }
}

static int *d_nan_count = NULL;

extern "C" int gpu_nan_check_(int n)
{
    if (n <= 0) return 0;
    if (!d_nan_count) cudaMalloc((void**)&d_nan_count, sizeof(int));
    cudaMemset(d_nan_count, 0, sizeof(int));
    int threads = 256;
    int blocks = (n + threads - 1) / threads;
    nan_check_kernel<<<blocks, threads>>>(
        d_vec[0], d_vec[1], d_vec[2], d_vec[3], d_vec[4], d_vec[5],
        d_state, d_nan_count, n);
    int h_count = 0;
    cudaMemcpy(&h_count, d_nan_count, sizeof(int), cudaMemcpyDeviceToHost);
    return h_count;
}

/* ==========================================================================
 * PATCH KERNEL
 *
 * Replicates track_a_patch.f90 for the forward-tracking case:
 *   direction=1, time_dir=1, orientation=1
 *   => entering from upstream, orbit%direction*orbit%time_dir*ele%orientation == 1
 *
 * Steps:
 *   1) Reconstruct 3D momentum p_vec from (px, py, pz_rel)
 *   2) Adjust pz sign for upstream_coord_dir
 *   3) Remove offsets: r_vec = (x - x_off, y - y_off, -z_off)
 *   4) Apply w_mat_inv rotation to both r_vec and p_vec (if non-identity)
 *   5) Apply time offset: z += beta * c_light * t_offset
 *   6) Energy reference correction (rescale px, py, pz when p0c changes)
 *   7) Drift to exit face
 *   8) Update beta, p0c
 *
 * Parameters passed from host (precomputed):
 *   ww[9]: w_mat_inv in column-major order (Fortran layout)
 *   x_off, y_off, z_off, t_offset: element offsets
 *   upstream_coord_dir: +1 or -1
 *   p0c_start, p0c_exit: reference momenta at entrance/exit
 *   e_tot_exit: total energy at exit
 *   ele_length: element length (for s update in z)
 *   mc2: rest mass energy
 *   has_rotation: 1 if rotation matrix is non-identity, 0 otherwise
 * ========================================================================== */

__global__ void patch_kernel(
    double *vx, double *vpx, double *vy, double *vpy, double *vz, double *vpz,
    int *state, double *beta_arr, double *p0c_arr, double *t_arr,
    const double *ww,
    double x_off, double y_off, double z_off, double t_offset,
    double upstream_coord_dir,
    double p0c_start, double p0c_exit, double e_tot_exit,
    double ele_length, double mc2,
    int has_rotation, int n_particles)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n_particles) return;
    if (state[i] != ALIVE_ST) return;

    /* Step 1: reconstruct 3D momentum (before any correction) */
    double px = vpx[i], py = vpy[i];
    double rel_p = 1.0 + vpz[i];
    double pz_sq = rel_p * rel_p - px * px - py * py;
    if (pz_sq <= 0.0) { state[i] = LOST_PZ; return; }
    double pz = sqrt(pz_sq);

    /* Step 2: pz sign for upstream_coord_dir (direction=1 for GPU) */
    pz = pz * upstream_coord_dir;

    /* Step 3: remove offsets (forward branch: entering from upstream) */
    double rx = vx[i] - x_off;
    double ry = vy[i] - y_off;
    double rz = -z_off;

    /* p_vec components (will be rotated if has_rotation) */
    double p1 = px, p2 = py, p3 = pz;

    /* Step 4: apply w_mat_inv rotation (column-major from Fortran) */
    if (has_rotation) {
        double w11 = ww[0], w21 = ww[1], w31 = ww[2];
        double w12 = ww[3], w22 = ww[4], w32 = ww[5];
        double w13 = ww[6], w23 = ww[7], w33 = ww[8];

        p1 = w11 * px + w12 * py + w13 * pz;
        p2 = w21 * px + w22 * py + w23 * pz;
        p3 = w31 * px + w32 * py + w33 * pz;

        double rx2 = w11 * rx + w12 * ry + w13 * rz;
        double ry2 = w21 * rx + w22 * ry + w23 * rz;
        double rz2 = w31 * rx + w32 * ry + w33 * rz;
        rx = rx2; ry = ry2; rz = rz2;
    }

    /* Store rotated position and momentum */
    vpx[i] = p1;
    vpy[i] = p2;
    vx[i] = rx;
    vy[i] = ry;

    /* Step 5: time offset */
    double beta_val = beta_arr[i];
    vz[i] = vz[i] + beta_val * C_LIGHT * t_offset;

    /* Step 6: energy reference correction
     * For forward tracking (dir*time_dir==1), first_track_edge$ => correct to p0c_exit.
     * IMPORTANT: The CPU drift (step 7) uses the PRE-correction p_vec and rel_p.
     * Also, orbit%beta is NOT updated by orbit_reference_energy_correction,
     * so the drift uses the entrance-side beta. We update beta only at the end. */
    if (p0c_start != p0c_exit) {
        double p0c_old = p0c_arr[i];
        double p_rel = p0c_old / p0c_exit;
        vpx[i] = vpx[i] * p_rel;
        vpy[i] = vpy[i] * p_rel;
        vpz[i] = (vpz[i] * p0c_old - (p0c_exit - p0c_old)) / p0c_exit;
        p0c_arr[i] = p0c_exit;
        /* Note: beta_val stays at entrance-side value for the drift */
    }

    /* Step 7: drift to exit face using PRE-correction p_vec and rel_p,
     * and entrance-side beta_val (matching CPU orbit%beta behavior).
     * NOTE: The CPU track_a_patch does NOT update orbit%beta, so we
     * leave beta_arr[i] unchanged to match. */
    double beta0 = p0c_exit / e_tot_exit;
    vx[i] = vx[i] - rz * p1 / p3;
    vy[i] = vy[i] - rz * p2 / p3;
    vz[i] = vz[i] + rz * rel_p / p3 + ele_length * beta_val / beta0;
    t_arr[i] = t_arr[i] - rz * rel_p / (p3 * beta_val * C_LIGHT);
}

/* HOST WRAPPERS for patch */

static double *d_patch_ww = NULL;

extern "C" void gpu_track_patch_(
    double *h_vx, double *h_vpx, double *h_vy, double *h_vpy,
    double *h_vz, double *h_vpz,
    int *h_state, double *h_beta, double *h_p0c, double *h_t,
    double *h_ww,
    double x_off, double y_off, double z_off, double t_offset,
    double upstream_coord_dir,
    double p0c_start, double p0c_exit, double e_tot_exit,
    double ele_length, double mc2,
    int has_rotation, int n_particles)
{
    if (ensure_buffers(n_particles) != 0) return;

    if (upload_particle_data(n_particles, h_vx, h_vpx, h_vy, h_vpy, h_vz, h_vpz,
                             h_state, h_beta, h_p0c, h_t) != 0) return;

    if (!d_patch_ww) {
        cudaMalloc((void**)&d_patch_ww, 9*sizeof(double));
    }
    CUDA_CHECK_VOID(cudaMemcpy(d_patch_ww, h_ww, 9*sizeof(double), cudaMemcpyHostToDevice));

    int threads = 256;
    int blocks = (n_particles + threads - 1) / threads;
    patch_kernel<<<blocks, threads>>>(
        d_vec[0], d_vec[1], d_vec[2], d_vec[3], d_vec[4], d_vec[5],
        d_state, d_beta, d_p0c, d_t,
        d_patch_ww,
        x_off, y_off, z_off, t_offset,
        upstream_coord_dir,
        p0c_start, p0c_exit, e_tot_exit,
        ele_length, mc2,
        has_rotation, n_particles);
    CUDA_CHECK_VOID(cudaGetLastError());
    CUDA_CHECK_VOID(cudaDeviceSynchronize());

    /* Patch can change beta and p0c, so always download them */
    if (download_particle_data(n_particles, h_vx, h_vpx, h_vy, h_vpy, h_vz, h_vpz,
                               h_state, h_beta, h_p0c, h_t,
                               1, 1) != 0) return;
}

/* Patch body-only: data already on device */
extern "C" void gpu_track_patch_dev_(
    double *h_ww,
    double x_off, double y_off, double z_off, double t_offset,
    double upstream_coord_dir,
    double p0c_start, double p0c_exit, double e_tot_exit,
    double ele_length, double mc2,
    int has_rotation, int n_particles)
{
    if (!d_patch_ww) {
        cudaMalloc((void**)&d_patch_ww, 9*sizeof(double));
    }
    CUDA_CHECK_VOID(cudaMemcpy(d_patch_ww, h_ww, 9*sizeof(double), cudaMemcpyHostToDevice));

    int threads = 256;
    int blocks = (n_particles + threads - 1) / threads;
    patch_kernel<<<blocks, threads>>>(
        d_vec[0], d_vec[1], d_vec[2], d_vec[3], d_vec[4], d_vec[5],
        d_state, d_beta, d_p0c, d_t,
        d_patch_ww,
        x_off, y_off, z_off, t_offset,
        upstream_coord_dir,
        p0c_start, p0c_exit, e_tot_exit,
        ele_length, mc2,
        has_rotation, n_particles);
    CUDA_CHECK_VOID(cudaGetLastError());
    CUDA_CHECK_VOID(cudaDeviceSynchronize());
}

/* --------------------------------------------------------------------------
 * gpu_save_bunch_buffers -- download current device particle buffers to
 * caller-supplied host arrays so another bunch can use the device.
 * Returns 0 on success, -1 on failure.
 * -------------------------------------------------------------------------- */
extern "C" int gpu_save_bunch_buffers_(
    double *h_vx, double *h_vpx, double *h_vy, double *h_vpy,
    double *h_vz, double *h_vpz,
    int *h_state, double *h_beta, double *h_p0c, double *h_t, double *h_s,
    int n)
{
    if (n <= 0 || n > d_cap) return -1;
    size_t db = (size_t)n * sizeof(double);
    size_t ib = (size_t)n * sizeof(int);
    CUDA_CHECK(cudaMemcpy(h_vx,    d_vec[0], db, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h_vpx,   d_vec[1], db, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h_vy,    d_vec[2], db, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h_vpy,   d_vec[3], db, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h_vz,    d_vec[4], db, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h_vpz,   d_vec[5], db, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h_state,  d_state,  ib, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h_beta,   d_beta,   db, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h_p0c,    d_p0c,    db, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h_t,      d_t,      db, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h_s,      d_s,      db, cudaMemcpyDeviceToHost));
    return 0;
}

/* --------------------------------------------------------------------------
 * gpu_restore_bunch_buffers -- upload previously saved host arrays back to
 * device buffers so tracking can resume for this bunch.
 * Calls ensure_buffers to (re-)allocate if needed.
 * Returns 0 on success, -1 on failure.
 * -------------------------------------------------------------------------- */
extern "C" int gpu_restore_bunch_buffers_(
    double *h_vx, double *h_vpx, double *h_vy, double *h_vpy,
    double *h_vz, double *h_vpz,
    int *h_state, double *h_beta, double *h_p0c, double *h_t, double *h_s,
    int n)
{
    if (n <= 0) return -1;
    if (ensure_buffers(n) != 0) return -1;
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
    CUDA_CHECK(cudaMemcpy(d_s,      h_s,      db, cudaMemcpyHostToDevice));
    return 0;
}

/* --------------------------------------------------------------------------
 * gpu_get_buffer_cap -- return current device buffer capacity (particles)
 * -------------------------------------------------------------------------- */
extern "C" int gpu_get_buffer_cap_(void)
{
    return d_cap;
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
    if (d_patch_ww)      cudaFree(d_patch_ww);      d_patch_ww      = NULL;
    d_rng_cap = 0;
    d_step_cap = 0;
    d_cap = 0;
}

#endif /* USE_GPU_TRACKING */
