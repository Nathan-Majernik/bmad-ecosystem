!+
! gpu_fringe_test
!
! Test program verifying GPU quadrupole fringe tracking for all fringe modes.
!
! Tests exercise the GPU fringe kernel (gpu_quad_fringe) through the
! multi-element device-resident dispatch path, comparing against CPU.
!
! Single-element tests (per-element path, CPU fringe for comparison):
!   1. Quad soft_edge_only fringe (single element, per-element path)
!   2. Quad hard_edge_only fringe (single element, per-element path)
!
! Multi-element tests (device-resident path, GPU fringe):
!   3. Quad soft_edge_only fringe (d+q+d, multi-element dispatch)
!   4. Quad hard_edge_only fringe (d+q+d, multi-element dispatch)
!   5. Quad full fringe (d+q+d+q+d, multi-element dispatch)
!   6. Quad full fringe with misalignment (multi-element dispatch)
!
! Multi-element tests use a relaxed tolerance (1e-7) because the GPU
! fringe kernel has small numerical differences from the CPU fringe
! implementation (~1e-9 to 1e-11).
!-

program gpu_fringe_test

use bmad
use beam_mod
use gpu_tracking_mod

implicit none

type (lat_struct), target :: lat
type (beam_init_struct) :: beam_init
type (beam_struct) :: beam_cpu, beam_gpu
type (branch_struct), pointer :: branch

integer :: n_pass, n_fail
real(rp) :: max_diff
logical :: err

n_pass = 0
n_fail = 0

! Common beam init
beam_init%n_particle = 1000
beam_init%random_engine = 'quasi'
beam_init%a_emit = 1e-9
beam_init%b_emit = 1e-9
beam_init%dPz_dz = 0
beam_init%n_bunch = 1
beam_init%bunch_charge = 1e-9
beam_init%sig_pz = 1e-3
beam_init%sig_z = 1e-4
beam_init%random_sigma_cutoff = 4

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
beam_init%n_particle = 1000

print *
print *, '=================================================================='
print *, '  GPU Fringe Mode Tests'
print *, '=================================================================='
print *

! ======================================================================
! TEST 1: Single-element soft_edge_only (per-element path, CPU fringe)
! This verifies the CPU fringe path still works for soft_edge_only
! ======================================================================
call bmad_parser('lat_quad_fringe_soft.bmad', lat)
call run_comparison_test('Test 1: Soft fringe (per-ele)', lat, 1d-12, n_pass, n_fail)

! ======================================================================
! TEST 2: Single-element hard_edge_only (per-element path, CPU fringe)
! ======================================================================
call bmad_parser('lat_quad_fringe_hard.bmad', lat)
call run_comparison_test('Test 2: Hard fringe (per-ele)', lat, 1d-12, n_pass, n_fail)

! ======================================================================
! TEST 3: Multi-element soft_edge_only (device-resident, GPU fringe)
! ======================================================================
call bmad_parser('lat_quad_fringe_soft_multi.bmad', lat)
call run_comparison_test('Test 3: Soft fringe (GPU multi)', lat, 1d-7, n_pass, n_fail)

! ======================================================================
! TEST 4: Multi-element hard_edge_only (device-resident, GPU fringe)
! ======================================================================
call bmad_parser('lat_quad_fringe_hard_multi.bmad', lat)
call run_comparison_test('Test 4: Hard fringe (GPU multi)', lat, 1d-7, n_pass, n_fail)

! ======================================================================
! TEST 5: Multi-element full fringe, two quads (device-resident, GPU fringe)
! ======================================================================
call bmad_parser('lat_quad_fringe_full_multi.bmad', lat)
call run_comparison_test('Test 5: Full fringe (GPU multi)', lat, 1d-7, n_pass, n_fail)

! ======================================================================
! TEST 6: Full fringe + misalignment (device-resident, GPU fringe + misalign)
! ======================================================================
call bmad_parser('lat_quad_fringe_misalign.bmad', lat)
call run_comparison_test('Test 6: Full fringe+misalign (per-ele)', lat, 1d-12, n_pass, n_fail)

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
subroutine run_comparison_test(test_name, lat, tol, n_pass, n_fail)
character(*), intent(in) :: test_name
type (lat_struct), target, intent(inout) :: lat
real(rp), intent(in) :: tol
integer, intent(inout) :: n_pass, n_fail

type (beam_struct) :: b_cpu, b_gpu
type (branch_struct), pointer :: br
real(rp) :: mdiff
logical :: errf

br => lat%branch(0)

call init_beam_distribution(br%ele(0), br%param, beam_init, b_cpu, errf)
call init_beam_distribution(br%ele(0), br%param, beam_init, b_gpu, errf)
b_gpu%bunch(1)%particle = b_cpu%bunch(1)%particle

! CPU run
bmad_com%gpu_tracking_on = .false.
call track_beam(lat, b_cpu, err=errf)

! GPU run
bmad_com%gpu_tracking_on = .true.
call track_beam(lat, b_gpu, err=errf)

mdiff = compute_max_diff(b_cpu, b_gpu)
if (mdiff < tol) then
  print '(A,A,T50,A,ES10.2)', '  PASS  ', test_name, 'max_diff=', mdiff
  n_pass = n_pass + 1
else
  print '(A,A,T50,A,ES10.2)', '  FAIL  ', test_name, 'max_diff=', mdiff
  n_fail = n_fail + 1
endif

end subroutine run_comparison_test

!------------------------------------------------------------------------
function compute_max_diff(b1, b2) result(mdiff)
type (beam_struct), intent(in) :: b1, b2
real(rp) :: mdiff
integer :: j, k, np

mdiff = 0
np = min(size(b1%bunch(1)%particle), size(b2%bunch(1)%particle))
do j = 1, np
  do k = 1, 6
    mdiff = max(mdiff, abs(b1%bunch(1)%particle(j)%vec(k) - b2%bunch(1)%particle(j)%vec(k)))
  enddo
  mdiff = max(mdiff, abs(b1%bunch(1)%particle(j)%t - b2%bunch(1)%particle(j)%t))
  mdiff = max(mdiff, abs(b1%bunch(1)%particle(j)%s - b2%bunch(1)%particle(j)%s))
enddo
end function compute_max_diff

end program gpu_fringe_test
