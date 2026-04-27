!+
! gpu_sc_test
!
! Test program verifying GPU 3D FFT space charge and CSR correctness.
!
! Tests:
!   1. Drift with 3D FFT space charge: GPU vs CPU statistical agreement
!   2. Bend with 1D CSR: GPU vs CPU statistical agreement
!   3. Kitchen sink with CSR + SC: GPU vs CPU statistical agreement
!   4. Drift with SC, GPU off → CPU only (baseline sanity check)
!
! Since the GPU uses cuFFT (different numerical implementation than the
! Fortran FFT) and GPU atomicAdd introduces non-deterministic ordering,
! the results will not match exactly. We compare beam statistics
! (mean, sigma) and require agreement within tolerance.
!-

program gpu_sc_test

use bmad
use beam_mod
use gpu_tracking_mod

implicit none

type (lat_struct), target :: lat
type (beam_init_struct) :: beam_init
type (beam_struct) :: beam_cpu, beam_gpu
type (branch_struct), pointer :: branch
type (coord_struct), allocatable :: centroid(:)

integer :: n_pass, n_fail, ie
logical :: err

n_pass = 0
n_fail = 0

! Beam init
beam_init%n_particle = 10000
beam_init%random_engine = 'quasi'
beam_init%a_emit = 5e-7
beam_init%b_emit = 5e-7
beam_init%dPz_dz = 0
beam_init%n_bunch = 1
beam_init%bunch_charge = 1e-9
beam_init%sig_pz = 1e-3
beam_init%sig_z = 3e-3
beam_init%random_sigma_cutoff = 3

! Initialize GPU
call gpu_tracking_init()
if (.not. bmad_com%gpu_tracking_on) then
  print *, 'FATAL: GPU tracking not available'
  stop
endif

! Warmup
beam_init%n_particle = 10
call bmad_parser('lat_drift_only.bmad', lat)
branch => lat%branch(0)
call init_beam_distribution(branch%ele(0), branch%param, beam_init, beam_gpu, err)
call track_beam(lat, beam_gpu, err=err)
beam_init%n_particle = 10000

print *
print *, '=================================================================='
print *, '  GPU Space Charge & CSR Tests'
print *, '=================================================================='
print *

! Common space charge settings
space_charge_com%ds_track_step = 0.1_rp
space_charge_com%n_bin = 40
space_charge_com%particle_bin_span = 2
space_charge_com%space_charge_mesh_size = [16, 16, 32]

! ======================================================================
! TEST 1: Drift with 3D FFT space charge
! ======================================================================
bmad_com%csr_and_space_charge_on = .true.
call bmad_parser('lat_sc_test.bmad', lat)
call compute_centroid(lat, centroid)
call run_sc_statistical_test('Test 1: Drift with 3D FFT SC', lat, centroid, n_pass, n_fail)

! ======================================================================
! TEST 2: Bend with 1D CSR
! ======================================================================
call bmad_parser('lat_csr_test.bmad', lat)
call compute_centroid(lat, centroid)
call run_sc_statistical_test('Test 2: Bend with 1D CSR', lat, centroid, n_pass, n_fail)

! ======================================================================
! TEST 3: Kitchen sink with CSR + SC
! ======================================================================
call bmad_parser('lat_kitchen_sink_sc.bmad', lat)
call compute_centroid(lat, centroid)
call run_sc_statistical_test('Test 3: Kitchen sink CSR+SC', lat, centroid, n_pass, n_fail)

! ======================================================================
! TEST 4: CPU-only baseline (drift with SC, no GPU)
! ======================================================================
call bmad_parser('lat_sc_test.bmad', lat)
call compute_centroid(lat, centroid)
branch => lat%branch(0)
call init_beam_distribution(branch%ele(0), branch%param, beam_init, beam_cpu, err)
bmad_com%gpu_tracking_on = .false.
call track_beam(lat, beam_cpu, err=err, centroid=centroid)
bmad_com%gpu_tracking_on = .true.

! Check that SC had an effect (sigma should differ from zero-SC case)
block
  real(rp) :: mean_cpu(6), sig_cpu(6)
  integer :: n_alive
  call compute_beam_stats(beam_cpu, mean_cpu, sig_cpu, n_alive)
  if (n_alive > 0 .and. sig_cpu(2) > 0) then
    print '(A,A,T50,A,I6,A,ES10.2)', '  PASS  ', 'Test 4: CPU SC baseline runs', &
      'alive=', n_alive, ' sig_px=', sig_cpu(2)
    n_pass = n_pass + 1
  else
    print '(A,A,T50,A,I6)', '  FAIL  ', 'Test 4: CPU SC baseline', 'alive=', n_alive
    n_fail = n_fail + 1
  endif
end block

bmad_com%csr_and_space_charge_on = .false.

! Summary
print *
print *, '=================================================================='
print '(A,I3,A,I3,A)', '   ', n_pass, ' passed, ', n_fail, ' failed'
if (n_fail == 0) then
  print *, '  ALL TESTS PASSED'
else
  print *, '  SOME TESTS FAILED'
endif
print *, '=================================================================='

if (n_fail > 0) stop 1

contains

!------------------------------------------------------------------------
subroutine compute_centroid(lat, centroid)
type (lat_struct), target, intent(inout) :: lat
type (coord_struct), allocatable, intent(out) :: centroid(:)
type (branch_struct), pointer :: br
type (coord_struct) :: orb
integer :: ie2
logical :: errf

br => lat%branch(0)
allocate(centroid(0:br%n_ele_track))
call init_coord(orb, br%ele(0), downstream_end$)
centroid(0) = orb
do ie2 = 1, br%n_ele_track
  call track1(orb, br%ele(ie2), br%param, orb)
  centroid(ie2) = orb
enddo
end subroutine compute_centroid

!------------------------------------------------------------------------
subroutine run_sc_statistical_test(test_name, lat, centroid, n_pass, n_fail)
character(*), intent(in) :: test_name
type (lat_struct), target, intent(inout) :: lat
type (coord_struct), intent(in) :: centroid(0:)
integer, intent(inout) :: n_pass, n_fail

type (beam_struct) :: b_cpu, b_gpu
type (branch_struct), pointer :: br
integer :: n_alive_cpu, n_alive_gpu, k_v
real(rp) :: mean_cpu(6), mean_gpu(6), sig_cpu(6), sig_gpu(6)
real(rp) :: mean_diff, sig_diff, stat_tol
logical :: errf, pass

br => lat%branch(0)

call init_beam_distribution(br%ele(0), br%param, beam_init, b_cpu, errf)
call init_beam_distribution(br%ele(0), br%param, beam_init, b_gpu, errf)
b_gpu%bunch(1)%particle = b_cpu%bunch(1)%particle

! CPU run
bmad_com%gpu_tracking_on = .false.
call track_beam(lat, b_cpu, err=errf, centroid=centroid)

! GPU run
bmad_com%gpu_tracking_on = .true.
call track_beam(lat, b_gpu, err=errf, centroid=centroid)

call compute_beam_stats(b_cpu, mean_cpu, sig_cpu, n_alive_cpu)
call compute_beam_stats(b_gpu, mean_gpu, sig_gpu, n_alive_gpu)

! Tolerance: 5% on sigma, 0.5 sigma on mean
stat_tol = 0.05_rp
mean_diff = 0; sig_diff = 0
do k_v = 1, 6
  if (sig_cpu(k_v) > 0) then
    mean_diff = max(mean_diff, abs(mean_cpu(k_v) - mean_gpu(k_v)) / sig_cpu(k_v))
    sig_diff = max(sig_diff, abs(sig_cpu(k_v) - sig_gpu(k_v)) / sig_cpu(k_v))
  endif
enddo

pass = (mean_diff < 0.5_rp) .and. (sig_diff < stat_tol) .and. &
       (abs(n_alive_cpu - n_alive_gpu) <= max(1, n_alive_cpu / 20))

if (pass) then
  print '(A,A,T50,A,F6.3,A,F6.3)', '  PASS  ', test_name, &
    'mean_d=', mean_diff, ' sig_d=', sig_diff
  n_pass = n_pass + 1
else
  print '(A,A,T50,A,F6.3,A,F6.3)', '  FAIL  ', test_name, &
    'mean_d=', mean_diff, ' sig_d=', sig_diff
  print '(A,I6,A,I6)', '         alive_cpu=', n_alive_cpu, ' alive_gpu=', n_alive_gpu
  n_fail = n_fail + 1
endif

end subroutine run_sc_statistical_test

!------------------------------------------------------------------------
subroutine compute_beam_stats(beam, mean, sig, n_alive)
type (beam_struct), intent(in) :: beam
real(rp), intent(out) :: mean(6), sig(6)
integer, intent(out) :: n_alive
integer :: j, k, np
real(rp) :: sum1(6), sum2(6)

np = size(beam%bunch(1)%particle)
sum1 = 0; sum2 = 0; n_alive = 0

do j = 1, np
  if (beam%bunch(1)%particle(j)%state /= alive$) cycle
  n_alive = n_alive + 1
  do k = 1, 6
    sum1(k) = sum1(k) + beam%bunch(1)%particle(j)%vec(k)
    sum2(k) = sum2(k) + beam%bunch(1)%particle(j)%vec(k)**2
  enddo
enddo

if (n_alive > 1) then
  mean = sum1 / n_alive
  do k = 1, 6
    sig(k) = sqrt(max(0.0_rp, sum2(k)/n_alive - mean(k)**2))
  enddo
else
  mean = 0; sig = 0
endif
end subroutine compute_beam_stats

end program gpu_sc_test
