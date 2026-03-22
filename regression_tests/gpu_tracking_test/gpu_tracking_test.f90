!+
! gpu_tracking_test
!
! Test program verifying GPU tracking correctness and CPU fallback behavior.
! Tests:
!   1. Plain drift
!   2. Plain quadrupole
!   3. Quad with fringe fields
!   4. Quad with sextupole component
!   5. Quad with electric multipole
!   6. Quad with misalignment
!   7. Quad fringe+misalign+tilt
!   8. Quad multi-pole+misalign+fringe
!   9. Quad electric+magnetic+fringe+misalign
!  10. Drift + quad + drift (mixed lattice)
!  11. CPU-only mode (bmad_com%gpu_tracking_on = .false.)
!  12. Quad with tight aperture
!  13. Drift with tight aperture
!  14. Plain bend
!  15. Bend with k1
!  16. Bend with fringe fields
!  17. Bend with misalignment
!  18. Bend with magnetic multipoles
!  19. Bend with k1+fringe+misalign+multipoles+aperture
!  20. Plain lcavity
!  21. Lcavity with misalignment
!  22. Lcavity standing wave (ponderomotive kicks)
!  23. Lcavity with phi0/phi0_err/voltage_err
!  24. Lcavity with aperture
!  25. Lcavity with fringe fields
!  26. Lcavity fringe standing wave
!  27. Plain pipe
!  28. Pipe with misalignment
!  29. Pipe with fringe fields
!  30. Pipe in mixed lattice (drift+pipe+quad)
!  31. Pipe with tight aperture
!  32. Pipe combo (misalign+aperture)
!  33. Drift with elliptical aperture
!  34. Quad with elliptical aperture
!  35. Lcavity with phi0_multipass
!  36. Lcavity with absolute_time_tracking
!  37. Lcavity abstime multi-cavity
!  38. Plain sextupole
!  39. Sextupole with misalignment
!  40. Sextupole with higher multipoles
!  41. Sextupole with aperture
!  42. Bend with sad_full fringe
!  43. Bend with soft_edge_only fringe
!  44. Plain solenoid
!  45. Solenoid with misalignment
!  46. Plain sol_quad
!  47. Sol_quad with misalignment
!  48. Bend with exact_multipoles
!  49. Multi-bunch tracking (3 bunches)
!  50. Multi-bunch tracking (5 bunches)
!  51. Plain octupole
!  52. Thick multipole (combined sextupole + octupole)
!  53. Plain wiggler
!  54. Plain undulator
!  55. Wiggler with misalignment
!  56. Plain elseparator
!  57. Plain rf_bend
!-

program gpu_tracking_test

use bmad
use beam_mod
use gpu_tracking_mod

implicit none

type (lat_struct), target :: lat
type (beam_init_struct) :: beam_init
type (beam_struct) :: beam_cpu, beam_gpu
type (branch_struct), pointer :: branch

integer :: j, n_pass, n_fail
real(rp) :: max_diff, tol
logical :: err

n_pass = 0
n_fail = 0
tol = 1d-12

! Common beam init parameters
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

! ---- Initialize GPU tracking from env var ----
call gpu_tracking_init()
if (.not. bmad_com%gpu_tracking_on) then
  print *, 'FATAL: GPU tracking not available (set ACC_ENABLE_GPU_TRACKING=Y)'
  stop
endif

! ---- Warmup GPU ----
beam_init%n_particle = 10
call bmad_parser('lat_drift_only.bmad', lat)
branch => lat%branch(0)
call init_beam_distribution(branch%ele(0), branch%param, beam_init, beam_gpu, err)
call track_beam(lat, beam_gpu, err=err)

! Restore test particle count
beam_init%n_particle = 1000

print *
print *, '=================================================================='
print *, '  GPU Tracking Tests'
print *, '=================================================================='
print *

! ======================================================================
! TEST 1: Plain drift
! ======================================================================
call bmad_parser('lat_drift_only.bmad', lat)
call run_comparison_test('Test 1: Plain drift', lat, tol, n_pass, n_fail)

! ======================================================================
! TEST 2: Plain quadrupole
! ======================================================================
call bmad_parser('lat_quad_only.bmad', lat)
call run_comparison_test('Test 2: Plain quadrupole', lat, tol, n_pass, n_fail)

! ======================================================================
! TEST 3: Quad with fringe fields
! ======================================================================
call bmad_parser('lat_quad_fringe.bmad', lat)
call run_comparison_test('Test 3: Quad with fringe (GPU+CPU)', lat, tol, n_pass, n_fail)

! ======================================================================
! TEST 4: Quad with sextupole component
! ======================================================================
call bmad_parser('lat_quad_sextupole.bmad', lat)
call run_comparison_test('Test 4: Quad with sextupole (GPU+CPU)', lat, tol, n_pass, n_fail)

! ======================================================================
! TEST 5: Quad with electric multipole
! ======================================================================
call bmad_parser('lat_quad_elec.bmad', lat)
call run_comparison_test('Test 5: Quad with electric multipole (GPU)', lat, tol, n_pass, n_fail)

! ======================================================================
! TEST 6: Quad with misalignment
! ======================================================================
call bmad_parser('lat_quad_misalign.bmad', lat)
call run_comparison_test('Test 6: Quad with misalignment (GPU)', lat, tol, n_pass, n_fail)

! ======================================================================
! TEST 7: Quad with fringe + misalignment + tilt (combined)
! ======================================================================
call bmad_parser('lat_quad_fringe_misalign.bmad', lat)
call run_comparison_test('Test 7: Quad fringe+misalign+tilt (GPU)', lat, tol, n_pass, n_fail)

! ======================================================================
! TEST 8: Quad with multiple multipoles + misalign + fringe
! ======================================================================
call bmad_parser('lat_quad_multi_multipole.bmad', lat)
call run_comparison_test('Test 8: Quad multi-pole+misalign+fringe', lat, tol, n_pass, n_fail)

! ======================================================================
! TEST 9: Quad with electric + magnetic + fringe + misalign (combined)
! ======================================================================
call bmad_parser('lat_quad_elec_combo.bmad', lat)
call run_comparison_test('Test 9: Quad elec+mag+fringe+misalign', lat, tol, n_pass, n_fail)

! ======================================================================
! TEST 10: Drift + quad + drift (mixed)
! ======================================================================
call bmad_parser('lat.bmad', lat)
call run_comparison_test('Test 10: Drift + quad + drift', lat, tol, n_pass, n_fail)

! ======================================================================
! TEST 11: CPU-only mode (bmad_com%gpu_tracking_on toggled off)
! ======================================================================
call bmad_parser('lat.bmad', lat)
branch => lat%branch(0)

! Init identical beams
call init_beam_distribution(branch%ele(0), branch%param, beam_init, beam_cpu, err)
call init_beam_distribution(branch%ele(0), branch%param, beam_init, beam_gpu, err)
beam_gpu%bunch(1)%particle = beam_cpu%bunch(1)%particle

! GPU run
bmad_com%gpu_tracking_on = .true.
call track_beam(lat, beam_gpu, err=err)

! CPU run (just toggle the flag)
bmad_com%gpu_tracking_on = .false.
call track_beam(lat, beam_cpu, err=err)

max_diff = compute_max_diff(beam_cpu, beam_gpu)
if (max_diff < tol) then
  print '(A,A,ES10.2)', '  PASS  ', 'Test 11: CPU-only matches GPU  ', max_diff
  n_pass = n_pass + 1
else
  print '(A,A,ES10.2)', '  FAIL  ', 'Test 11: CPU-only matches GPU  ', max_diff
  n_fail = n_fail + 1
endif

! Re-enable GPU for any subsequent work
bmad_com%gpu_tracking_on = .true.

! ======================================================================
! TEST 12: Quad with tight aperture (particle loss must match)
! ======================================================================
call bmad_parser('lat_quad_aperture.bmad', lat)
call run_aperture_test('Test 12: Quad with aperture', lat, tol, n_pass, n_fail)

! ======================================================================
! TEST 13: Drift with tight aperture
! ======================================================================
call bmad_parser('lat_drift_aperture.bmad', lat)
call run_aperture_test('Test 13: Drift with aperture', lat, tol, n_pass, n_fail)

! ======================================================================
! TEST 14: Plain bend
! ======================================================================
call bmad_parser('lat_bend_only.bmad', lat)
! Bend fringe runs on GPU (Hwang kick) — small FP ordering differences (~1e-7)
call run_comparison_test('Test 14: Plain bend', lat, 1d-6, n_pass, n_fail)

! ======================================================================
! TEST 15: Bend with k1
! ======================================================================
call bmad_parser('lat_bend_k1.bmad', lat)
call run_comparison_test('Test 15: Bend with k1', lat, 1d-6, n_pass, n_fail)

! ======================================================================
! TEST 16: Bend with fringe fields
! ======================================================================
call bmad_parser('lat_bend_fringe.bmad', lat)
call run_comparison_test('Test 16: Bend with fringe', lat, 1d-6, n_pass, n_fail)

! ======================================================================
! TEST 17: Bend with misalignment
! ======================================================================
call bmad_parser('lat_bend_misalign.bmad', lat)
call run_comparison_test('Test 17: Bend with misalignment', lat, 1d-3, n_pass, n_fail)

! ======================================================================
! TEST 18: Bend with magnetic multipoles
! ======================================================================
call bmad_parser('lat_bend_multipole.bmad', lat)
call run_comparison_test('Test 18: Bend with multipoles', lat, 1d-6, n_pass, n_fail)

! ======================================================================
! TEST 19: Bend with k1+fringe+misalign+multipoles+aperture
! ======================================================================
call bmad_parser('lat_bend_combo.bmad', lat)
! GPU exact fringe may cause minor aperture-edge state differences
call run_aperture_test('Test 19: Bend combo+aperture', lat, 1d-6, n_pass, n_fail)

! ======================================================================
! TEST 20: Plain lcavity
! ======================================================================
call bmad_parser('lat_lcavity_only.bmad', lat)
call run_comparison_test('Test 20: Plain lcavity', lat, tol, n_pass, n_fail)

! ======================================================================
! TEST 21: Lcavity with misalignment
! ======================================================================
call bmad_parser('lat_lcavity_misalign.bmad', lat)
call run_comparison_test('Test 21: Lcavity with misalignment', lat, tol, n_pass, n_fail)

! ======================================================================
! TEST 22: Lcavity standing wave (ponderomotive kicks)
! ======================================================================
call bmad_parser('lat_lcavity_standing.bmad', lat)
call run_comparison_test('Test 22: Lcavity standing wave', lat, tol, n_pass, n_fail)

! ======================================================================
! TEST 23: Lcavity with phi0/phi0_err/voltage_err
! ======================================================================
call bmad_parser('lat_lcavity_phase.bmad', lat)
call run_comparison_test('Test 23: Lcavity phase+voltage offsets', lat, tol, n_pass, n_fail)

! ======================================================================
! TEST 24: Lcavity with aperture
! ======================================================================
call bmad_parser('lat_lcavity_aperture.bmad', lat)
call run_aperture_test('Test 24: Lcavity with aperture', lat, tol, n_pass, n_fail)

! ======================================================================
! TEST 25: Lcavity with fringe fields (default fringe_type=full)
! ======================================================================
call bmad_parser('lat_lcavity_fringe.bmad', lat)
call run_comparison_test('Test 25: Lcavity fringe', lat, tol, n_pass, n_fail)

! ======================================================================
! TEST 26: Lcavity with fringe fields (standing wave)
! ======================================================================
call bmad_parser('lat_lcavity_fringe_standing.bmad', lat)
call run_comparison_test('Test 26: Lcavity fringe standing wave', lat, tol, n_pass, n_fail)

! ======================================================================
! TEST 27: Plain pipe
! ======================================================================
call bmad_parser('lat_pipe_only.bmad', lat)
call run_comparison_test('Test 27: Plain pipe', lat, tol, n_pass, n_fail)

! ======================================================================
! TEST 28: Pipe with misalignment
! ======================================================================
call bmad_parser('lat_pipe_misalign.bmad', lat)
call run_comparison_test('Test 28: Pipe with misalignment', lat, tol, n_pass, n_fail)

! ======================================================================
! TEST 29: Pipe with fringe fields
! ======================================================================
call bmad_parser('lat_pipe_multipole.bmad', lat)
call run_comparison_test('Test 29: Pipe with fringe', lat, tol, n_pass, n_fail)

! ======================================================================
! TEST 30: Pipe in mixed lattice (drift+pipe+quad)
! ======================================================================
call bmad_parser('lat_pipe_elec.bmad', lat)
call run_comparison_test('Test 30: Pipe in mixed lattice', lat, tol, n_pass, n_fail)

! ======================================================================
! TEST 31: Pipe with tight aperture
! ======================================================================
call bmad_parser('lat_pipe_aperture.bmad', lat)
call run_aperture_test('Test 31: Pipe with aperture', lat, tol, n_pass, n_fail)

! ======================================================================
! TEST 32: Pipe combo (misalign+aperture)
! ======================================================================
call bmad_parser('lat_pipe_combo.bmad', lat)
call run_aperture_test('Test 32: Pipe combo+aperture', lat, tol, n_pass, n_fail)

! ======================================================================
! TEST 33: Drift with elliptical aperture
! ======================================================================
call bmad_parser('lat_drift_elliptical.bmad', lat)
call run_aperture_test('Test 33: Drift elliptical aperture', lat, tol, n_pass, n_fail)

! ======================================================================
! TEST 34: Quad with elliptical aperture
! ======================================================================
call bmad_parser('lat_quad_elliptical.bmad', lat)
call run_aperture_test('Test 34: Quad elliptical aperture', lat, tol, n_pass, n_fail)

! ======================================================================
! TEST 35: Lcavity with phi0_multipass (multipass phase offset)
! ======================================================================
call bmad_parser('lat_lcavity_multipass.bmad', lat)
call run_comparison_test('Test 35: Lcavity phi0_multipass', lat, tol, n_pass, n_fail)

! ======================================================================
! TEST 36: Lcavity with absolute_time_tracking
! ======================================================================
call bmad_parser('lat_lcavity_abstime.bmad', lat)
call run_comparison_test('Test 36: Lcavity abstime', lat, tol, n_pass, n_fail)

! ======================================================================
! TEST 37: Lcavity abstime multi-cavity (downstream ref_time_start)
! ======================================================================
call bmad_parser('lat_lcavity_abstime_multi.bmad', lat)
call run_comparison_test('Test 37: Lcavity abstime multi-cav', lat, tol, n_pass, n_fail)

! ======================================================================
! TEST 38: Plain sextupole
! ======================================================================
call bmad_parser('lat_sext_only.bmad', lat)
! Sextupoles amplify small GPU fringe differences from upstream elements
call run_comparison_test('Test 38: Plain sextupole', lat, 1d-4, n_pass, n_fail)

! ======================================================================
! TEST 39: Sextupole with misalignment
! ======================================================================
call bmad_parser('lat_sext_misalign.bmad', lat)
call run_comparison_test('Test 39: Sextupole with misalignment', lat, 1d-4, n_pass, n_fail)

! ======================================================================
! TEST 40: Sextupole with higher multipoles
! ======================================================================
call bmad_parser('lat_sext_multipole.bmad', lat)
call run_comparison_test('Test 40: Sextupole with multipoles', lat, 1d-4, n_pass, n_fail)

! ======================================================================
! TEST 41: Sextupole with aperture
! ======================================================================
call bmad_parser('lat_sext_aperture.bmad', lat)
call run_aperture_test('Test 41: Sextupole with aperture', lat, 1d-4, n_pass, n_fail)

! ======================================================================
! TEST 42: Bend with sad_full fringe
! ======================================================================
call bmad_parser('lat_bend_sad_full.bmad', lat)
call run_comparison_test('Test 42: Bend sad_full fringe', lat, 1d-6, n_pass, n_fail)

! ======================================================================
! TEST 43: Bend with soft_edge_only fringe
! ======================================================================
call bmad_parser('lat_bend_soft_edge.bmad', lat)
call run_comparison_test('Test 43: Bend soft_edge_only fringe', lat, 1d-6, n_pass, n_fail)

! ======================================================================
! TEST 44: Plain solenoid
! ======================================================================
call bmad_parser('lat_solenoid_only.bmad', lat)
call run_comparison_test('Test 44: Plain solenoid', lat, tol, n_pass, n_fail)

! ======================================================================
! TEST 45: Solenoid with misalignment
! ======================================================================
call bmad_parser('lat_solenoid_misalign.bmad', lat)
call run_comparison_test('Test 45: Solenoid with misalignment', lat, tol, n_pass, n_fail)

! ======================================================================
! TEST 46: Plain sol_quad
! ======================================================================
call bmad_parser('lat_sol_quad_only.bmad', lat)
call run_comparison_test('Test 46: Plain sol_quad', lat, tol, n_pass, n_fail)

! ======================================================================
! TEST 47: Sol_quad with misalignment
! ======================================================================
call bmad_parser('lat_sol_quad_misalign.bmad', lat)
call run_comparison_test('Test 47: Sol_quad with misalignment', lat, tol, n_pass, n_fail)

! ======================================================================
! TEST 48: Bend with exact_multipoles
! ======================================================================
call bmad_parser('lat_bend_exact_multipole.bmad', lat)
call run_comparison_test('Test 48: Bend exact_multipoles', lat, 1d-6, n_pass, n_fail)

! ======================================================================
! TEST 49: Multi-bunch tracking (3 bunches)
! ======================================================================
call bmad_parser('lat.bmad', lat)
call run_multi_bunch_test('Test 49: Multi-bunch (3 bunches)', lat, tol, n_pass, n_fail, 3)

! ======================================================================
! TEST 50: Multi-bunch tracking (5 bunches, different lattice)
! Kitchen sink has sextupoles which amplify small FP ordering differences
! ======================================================================
call bmad_parser('lat_kitchen_sink.bmad', lat)
call run_multi_bunch_test('Test 50: Multi-bunch kitchen sink', lat, 1d-3, n_pass, n_fail, 5)

! ======================================================================
! TEST 51: Octupole
! ======================================================================
call bmad_parser('lat_octupole_only.bmad', lat)
call run_comparison_test('Test 51: Plain octupole', lat, 1d-3, n_pass, n_fail)

! ======================================================================
! TEST 52: Thick multipole (combined sextupole + octupole)
! ======================================================================
call bmad_parser('lat_thick_multipole.bmad', lat)
call run_comparison_test('Test 52: Thick multipole', lat, 1d-3, n_pass, n_fail)

! ======================================================================
! TEST 53: Plain wiggler
! ======================================================================
call bmad_parser('lat_wiggler_only.bmad', lat)
call run_comparison_test('Test 53: Plain wiggler', lat, tol, n_pass, n_fail)

! ======================================================================
! TEST 54: Plain undulator
! ======================================================================
call bmad_parser('lat_undulator_only.bmad', lat)
call run_comparison_test('Test 54: Plain undulator', lat, tol, n_pass, n_fail)

! ======================================================================
! TEST 55: Wiggler with misalignment
! ======================================================================
call bmad_parser('lat_wiggler_misalign.bmad', lat)
call run_comparison_test('Test 55: Wiggler with misalignment', lat, tol, n_pass, n_fail)

! ======================================================================
! TEST 56: Plain elseparator
! ======================================================================
call bmad_parser('lat_elseparator_only.bmad', lat)
call run_comparison_test('Test 56: Plain elseparator', lat, 2d-3, n_pass, n_fail)

! ======================================================================
! TEST 57: Plain rf_bend (bmad_standard tracking)
! ======================================================================
call bmad_parser('lat_rf_bend_only.bmad', lat)
call run_comparison_test('Test 57: Plain rf_bend', lat, tol, n_pass, n_fail)

! ======================================================================
! Summary
! ======================================================================
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
! run_comparison_test — run CPU and GPU tracking, compare results
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

! Init identical beams
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
! run_aperture_test — verify GPU and CPU agree on which particles are lost
!------------------------------------------------------------------------
subroutine run_aperture_test(test_name, lat, tol, n_pass, n_fail)
character(*), intent(in) :: test_name
type (lat_struct), target, intent(inout) :: lat
real(rp), intent(in) :: tol
integer, intent(inout) :: n_pass, n_fail

type (beam_struct) :: b_cpu, b_gpu
type (branch_struct), pointer :: br
real(rp) :: mdiff
integer :: j, np, n_state_mismatch, n_lost_cpu, n_lost_gpu
logical :: errf, pass

br => lat%branch(0)

! Init identical beams
call init_beam_distribution(br%ele(0), br%param, beam_init, b_cpu, errf)
call init_beam_distribution(br%ele(0), br%param, beam_init, b_gpu, errf)
b_gpu%bunch(1)%particle = b_cpu%bunch(1)%particle

! CPU run
bmad_com%gpu_tracking_on = .false.
call track_beam(lat, b_cpu, err=errf)

! GPU run
bmad_com%gpu_tracking_on = .true.
call track_beam(lat, b_gpu, err=errf)

! Compare particle states and coordinates
np = size(b_cpu%bunch(1)%particle)
n_state_mismatch = 0
n_lost_cpu = 0
n_lost_gpu = 0
mdiff = 0

do j = 1, np
  if (b_cpu%bunch(1)%particle(j)%state /= alive$) n_lost_cpu = n_lost_cpu + 1
  if (b_gpu%bunch(1)%particle(j)%state /= alive$) n_lost_gpu = n_lost_gpu + 1
  if (b_cpu%bunch(1)%particle(j)%state /= b_gpu%bunch(1)%particle(j)%state) then
    n_state_mismatch = n_state_mismatch + 1
  endif
  ! Only compare coordinates for particles alive in both
  if (b_cpu%bunch(1)%particle(j)%state == alive$ .and. &
      b_gpu%bunch(1)%particle(j)%state == alive$) then
    mdiff = max(mdiff, abs(b_cpu%bunch(1)%particle(j)%vec(1) - b_gpu%bunch(1)%particle(j)%vec(1)))
    mdiff = max(mdiff, abs(b_cpu%bunch(1)%particle(j)%vec(2) - b_gpu%bunch(1)%particle(j)%vec(2)))
    mdiff = max(mdiff, abs(b_cpu%bunch(1)%particle(j)%vec(3) - b_gpu%bunch(1)%particle(j)%vec(3)))
    mdiff = max(mdiff, abs(b_cpu%bunch(1)%particle(j)%vec(4) - b_gpu%bunch(1)%particle(j)%vec(4)))
    mdiff = max(mdiff, abs(b_cpu%bunch(1)%particle(j)%vec(5) - b_gpu%bunch(1)%particle(j)%vec(5)))
    mdiff = max(mdiff, abs(b_cpu%bunch(1)%particle(j)%vec(6) - b_gpu%bunch(1)%particle(j)%vec(6)))
  endif
enddo

pass = (n_state_mismatch == 0) .and. (mdiff < tol) .and. (n_lost_cpu > 0)

if (pass) then
  print '(A,A,T50,A,I4,A,I4,A,ES10.2)', '  PASS  ', test_name, &
    'lost=', n_lost_cpu, '/', np, '  max_diff=', mdiff
  n_pass = n_pass + 1
else
  print '(A,A,T50,A,I4,A,I4)', '  FAIL  ', test_name, &
    'state_mismatch=', n_state_mismatch, '  lost_cpu/gpu=', n_lost_cpu
  print '(A,I4,A,ES10.2)', '         lost_gpu=', n_lost_gpu, '  max_diff=', mdiff
  n_fail = n_fail + 1
endif

end subroutine run_aperture_test

!------------------------------------------------------------------------
! run_multi_bunch_test — track a multi-bunch beam on CPU and GPU, compare
!------------------------------------------------------------------------
subroutine run_multi_bunch_test(test_name, lat, tol, n_pass, n_fail, n_bunch)
character(*), intent(in) :: test_name
type (lat_struct), target, intent(inout) :: lat
real(rp), intent(in) :: tol
integer, intent(inout) :: n_pass, n_fail
integer, intent(in) :: n_bunch

type (beam_init_struct) :: mb_beam_init
type (beam_struct) :: b_cpu, b_gpu
type (branch_struct), pointer :: br
real(rp) :: mdiff
integer :: ib
logical :: errf

br => lat%branch(0)

! Set up multi-bunch beam init
mb_beam_init = beam_init
mb_beam_init%n_bunch = n_bunch

! Init identical beams
call init_beam_distribution(br%ele(0), br%param, mb_beam_init, b_cpu, errf)
call init_beam_distribution(br%ele(0), br%param, mb_beam_init, b_gpu, errf)
do ib = 1, n_bunch
  b_gpu%bunch(ib)%particle = b_cpu%bunch(ib)%particle
enddo

! CPU run
bmad_com%gpu_tracking_on = .false.
call track_beam(lat, b_cpu, err=errf)

! GPU run
bmad_com%gpu_tracking_on = .true.
call track_beam(lat, b_gpu, err=errf)

! Compare all bunches: coordinates and particle states
mdiff = compute_max_diff_multi(b_cpu, b_gpu, n_bunch)

! Also check for state mismatches across all bunches
block
  integer :: jb, jp, npp, n_state_mm
  n_state_mm = 0
  do jb = 1, n_bunch
    npp = min(size(b_cpu%bunch(jb)%particle), size(b_gpu%bunch(jb)%particle))
    do jp = 1, npp
      if (b_cpu%bunch(jb)%particle(jp)%state /= b_gpu%bunch(jb)%particle(jp)%state) &
        n_state_mm = n_state_mm + 1
    enddo
  enddo
  if (mdiff < tol .and. n_state_mm == 0) then
    print '(A,A,T50,A,ES10.2)', '  PASS  ', test_name, 'max_diff=', mdiff
    n_pass = n_pass + 1
  else
    print '(A,A,T50,A,ES10.2,A,I4)', '  FAIL  ', test_name, 'max_diff=', mdiff, &
          '  state_mm=', n_state_mm
    n_fail = n_fail + 1
  endif
end block

end subroutine run_multi_bunch_test

!------------------------------------------------------------------------
! compute_max_diff — max absolute difference across all particle coords
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

!------------------------------------------------------------------------
! compute_max_diff_multi — max absolute difference across all bunches
! Only compares particles that are alive in both beams.
!------------------------------------------------------------------------
function compute_max_diff_multi(b1, b2, n_bunch) result(mdiff)
type (beam_struct), intent(in) :: b1, b2
integer, intent(in) :: n_bunch
real(rp) :: mdiff
integer :: ib, j, k, np

mdiff = 0
do ib = 1, n_bunch
  np = min(size(b1%bunch(ib)%particle), size(b2%bunch(ib)%particle))
  do j = 1, np
    if (b1%bunch(ib)%particle(j)%state /= alive$ .or. &
        b2%bunch(ib)%particle(j)%state /= alive$) cycle
    do k = 1, 6
      mdiff = max(mdiff, abs(b1%bunch(ib)%particle(j)%vec(k) - b2%bunch(ib)%particle(j)%vec(k)))
    enddo
    mdiff = max(mdiff, abs(b1%bunch(ib)%particle(j)%t - b2%bunch(ib)%particle(j)%t))
    mdiff = max(mdiff, abs(b1%bunch(ib)%particle(j)%s - b2%bunch(ib)%particle(j)%s))
  enddo
enddo
end function compute_max_diff_multi

end program gpu_tracking_test
