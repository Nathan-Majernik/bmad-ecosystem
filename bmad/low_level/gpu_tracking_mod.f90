module gpu_tracking_mod

use bmad_struct
use bmad_routine_interface, dummy_gtm => track_a_drift

implicit none
private

public :: gpu_tracking_init
public :: gpu_tracking_reset
public :: gpu_tracking_is_active
public :: ele_gpu_eligible
public :: track_bunch_thru_drift_gpu
public :: track_bunch_thru_quad_gpu
public :: track_bunch_thru_sextupole_gpu
public :: track_bunch_thru_bend_gpu
public :: track_bunch_thru_lcavity_gpu
public :: track_bunch_thru_solenoid_gpu
public :: track_bunch_thru_sol_quad_gpu
public :: track_bunch_thru_wiggler_gpu
public :: track_bunch_thru_pipe_gpu
public :: check_entrance_aperture_for_gpu
public :: gpu_rad_eligible
public :: track_bunch_thru_elements_gpu
public :: ele_gpu_can_stay_on_device
public :: gpu_upload_particles, gpu_download_particles
public :: gpu_space_charge_3d, gpu_csr_bin_particles, gpu_csr_apply_kicks, gpu_csr_z_minmax
public :: gpu_csr_bin_kicks
public :: gpu_persistent_track_element, gpu_persistent_flush, gpu_persistent_seed
public :: gpu_persist_on_device
public :: gpu_track_body_on_device
public :: gpu_apply_fringe_on_device
public :: gpu_apply_misalign_on_device
public :: gpu_multi_bunch_save, gpu_multi_bunch_restore, gpu_multi_bunch_cleanup

! Whether gpu_tracking_init has been called
logical, save :: gpu_trk_initialized = .false.
! Whether CUDA hardware is present (checked once, does not change)
logical, save :: gpu_hw_available = .false.

! Persistent GPU session state (for cross-call device residence)
logical, save :: gpu_persist_on_device = .false.
integer, save :: gpu_persist_n = 0
integer(8), save :: gpu_persist_bunch_id = 0  ! bunch fingerprint to detect new beams
real(rp), allocatable, save :: gp_vx(:), gp_vpx(:), gp_vy(:), gp_vpy(:)
real(rp), allocatable, save :: gp_vz(:), gp_vpz(:)
real(rp), allocatable, save :: gp_beta(:), gp_p0c(:), gp_t(:), gp_s(:)
integer, allocatable, save :: gp_state(:)

! --------------------------------------------------------------------------
! Multi-bunch GPU buffer management
!
! When tracking multiple bunches, the single set of device buffers must be
! shared. Rather than discarding device data when switching bunches, we save
! the device state to per-bunch host-side slots and restore when switching back.
! This avoids costly AoS-to-SoA re-conversion and keeps the GPU pipeline hot.
! --------------------------------------------------------------------------

integer, parameter :: MAX_GPU_BUNCHES = 64   ! Max bunches we cache

type :: gpu_bunch_slot_struct
  logical :: valid = .false.          ! Does this slot contain saved data?
  integer :: n_particles = 0          ! Number of particles stored
  integer(8) :: bunch_id = 0          ! Bunch fingerprint (memory address)
  real(rp), allocatable :: vx(:), vpx(:), vy(:), vpy(:), vz(:), vpz(:)
  real(rp), allocatable :: beta(:), p0c(:), t(:), s(:)
  integer, allocatable :: state(:)
end type

type (gpu_bunch_slot_struct), save :: gpu_bunch_slots(MAX_GPU_BUNCHES)
integer, save :: gpu_active_slot = 0  ! Slot index currently on device (0 = none)

#ifdef USE_GPU_TRACKING
! ----- C interfaces (gpu_tracking_kernels.cu) ---------------------------------
interface
  subroutine gpu_track_drift(vx, vpx, vy, vpy, vz, vpz, &
                             state, beta, p0c, s_pos, t_time, &
                             mc2, length, n) bind(C, name='gpu_track_drift_')
    use, intrinsic :: iso_c_binding
    real(C_DOUBLE), intent(inout) :: vx(*), vpx(*), vy(*), vpy(*), vz(*), vpz(*)
    integer(C_INT), intent(inout) :: state(*)
    real(C_DOUBLE), intent(inout) :: beta(*), p0c(*), s_pos(*), t_time(*)
    real(C_DOUBLE), value, intent(in) :: mc2, length
    integer(C_INT), value, intent(in) :: n
  end subroutine

  subroutine gpu_track_quad(vx, vpx, vy, vpy, vz, vpz, &
                            state, beta, p0c, t_time, &
                            mc2, b1, ele_length, delta_ref_time, &
                            e_tot_ele, charge_dir, n_particles, &
                            a2_arr, b2_arr, cm_arr, &
                            ix_mag_max, n_step, &
                            ea2_arr, eb2_arr, ix_elec_max) bind(C, name='gpu_track_quad_')
    use, intrinsic :: iso_c_binding
    real(C_DOUBLE), intent(inout) :: vx(*), vpx(*), vy(*), vpy(*), vz(*), vpz(*)
    integer(C_INT), intent(inout) :: state(*)
    real(C_DOUBLE), intent(inout) :: beta(*), p0c(*), t_time(*)
    real(C_DOUBLE), value, intent(in) :: mc2, b1, ele_length
    real(C_DOUBLE), value, intent(in) :: delta_ref_time, e_tot_ele, charge_dir
    integer(C_INT), value, intent(in) :: n_particles
    real(C_DOUBLE), intent(in) :: a2_arr(*), b2_arr(*), cm_arr(*)
    integer(C_INT), value, intent(in) :: ix_mag_max, n_step
    real(C_DOUBLE), intent(in) :: ea2_arr(*), eb2_arr(*)
    integer(C_INT), value, intent(in) :: ix_elec_max
  end subroutine

  subroutine gpu_track_bend(vx, vpx, vy, vpy, vz, vpz, &
                            state, beta, p0c, t_time, &
                            mc2, g, g_tot, dg, b1, &
                            ele_length, delta_ref_time, e_tot_ele, &
                            rel_charge_dir, &
                            p0c_ele, n_particles, &
                            a2_arr, b2_arr, cm_arr, &
                            ix_mag_max, n_step, &
                            ea2_arr, eb2_arr, ix_elec_max, &
                            is_exact, exact_an, exact_bn, &
                            ix_exact_mag_max, rho_val, c_dir_val, &
                            exact_f_scale) bind(C, name='gpu_track_bend_')
    use, intrinsic :: iso_c_binding
    real(C_DOUBLE), intent(inout) :: vx(*), vpx(*), vy(*), vpy(*), vz(*), vpz(*)
    integer(C_INT), intent(inout) :: state(*)
    real(C_DOUBLE), intent(inout) :: beta(*), p0c(*), t_time(*)
    real(C_DOUBLE), value, intent(in) :: mc2, g, g_tot, dg, b1
    real(C_DOUBLE), value, intent(in) :: ele_length, delta_ref_time, e_tot_ele
    real(C_DOUBLE), value, intent(in) :: rel_charge_dir, p0c_ele
    integer(C_INT), value, intent(in) :: n_particles
    real(C_DOUBLE), intent(in) :: a2_arr(*), b2_arr(*), cm_arr(*)
    integer(C_INT), value, intent(in) :: ix_mag_max, n_step
    real(C_DOUBLE), intent(in) :: ea2_arr(*), eb2_arr(*)
    integer(C_INT), value, intent(in) :: ix_elec_max
    integer(C_INT), value, intent(in) :: is_exact
    real(C_DOUBLE), intent(in) :: exact_an(*), exact_bn(*)
    integer(C_INT), value, intent(in) :: ix_exact_mag_max
    real(C_DOUBLE), value, intent(in) :: rho_val, c_dir_val, exact_f_scale
  end subroutine

  subroutine gpu_track_lcavity(vx, vpx, vy, vpy, vz, vpz, &
                               state, beta, p0c, t_time, &
                               mc2, &
                               step_s0, step_s, step_p0c, step_p1c, &
                               step_scale, step_time, &
                               n_rf_steps, &
                               voltage, voltage_err, field_autoscale, &
                               rf_frequency, phi0_total, &
                               voltage_tot, l_active, &
                               cavity_type, &
                               fringe_at, charge_ratio, &
                               n_particles, &
                               abs_time, phi0_no_multi, &
                               ref_time_start) bind(C, name='gpu_track_lcavity_')
    use, intrinsic :: iso_c_binding
    real(C_DOUBLE), intent(inout) :: vx(*), vpx(*), vy(*), vpy(*), vz(*), vpz(*)
    integer(C_INT), intent(inout) :: state(*)
    real(C_DOUBLE), intent(inout) :: beta(*), p0c(*), t_time(*)
    real(C_DOUBLE), value, intent(in) :: mc2
    real(C_DOUBLE), intent(in) :: step_s0(*), step_s(*), step_p0c(*), step_p1c(*)
    real(C_DOUBLE), intent(in) :: step_scale(*), step_time(*)
    integer(C_INT), value, intent(in) :: n_rf_steps
    real(C_DOUBLE), value, intent(in) :: voltage, voltage_err, field_autoscale
    real(C_DOUBLE), value, intent(in) :: rf_frequency, phi0_total
    real(C_DOUBLE), value, intent(in) :: voltage_tot, l_active
    integer(C_INT), value, intent(in) :: cavity_type
    integer(C_INT), value, intent(in) :: fringe_at
    real(C_DOUBLE), value, intent(in) :: charge_ratio
    integer(C_INT), value, intent(in) :: n_particles
    integer(C_INT), value, intent(in) :: abs_time
    real(C_DOUBLE), value, intent(in) :: phi0_no_multi
    real(C_DOUBLE), value, intent(in) :: ref_time_start
  end subroutine

  function gpu_tracking_available() result(avail) bind(C, name='gpu_tracking_available_')
    use, intrinsic :: iso_c_binding
    integer(C_INT) :: avail
  end function

  subroutine gpu_tracking_cleanup() bind(C, name='gpu_tracking_cleanup_')
  end subroutine

  ! ----- Split upload/download/body wrappers for radiation support -----

  subroutine gpu_upload_particles(vx, vpx, vy, vpy, vz, vpz, &
                                  state, beta, p0c, t_time, n) bind(C, name='gpu_upload_particles_')
    use, intrinsic :: iso_c_binding
    real(C_DOUBLE), intent(in) :: vx(*), vpx(*), vy(*), vpy(*), vz(*), vpz(*)
    integer(C_INT), intent(in) :: state(*)
    real(C_DOUBLE), intent(in) :: beta(*), p0c(*), t_time(*)
    integer(C_INT), value, intent(in) :: n
  end subroutine

  subroutine gpu_download_particles(vx, vpx, vy, vpy, vz, vpz, &
                                    state, beta, p0c, t_time, &
                                    n, copy_beta, copy_p0c) bind(C, name='gpu_download_particles_')
    use, intrinsic :: iso_c_binding
    real(C_DOUBLE), intent(inout) :: vx(*), vpx(*), vy(*), vpy(*), vz(*), vpz(*)
    integer(C_INT), intent(inout) :: state(*)
    real(C_DOUBLE), intent(inout) :: beta(*), p0c(*), t_time(*)
    integer(C_INT), value, intent(in) :: n, copy_beta, copy_p0c
  end subroutine

  subroutine gpu_rad_kick(n, stoc_mat, damp_dmat, xfer_damp_vec, ref_orb, &
                           synch_rad_scale, apply_damp, apply_fluct, &
                           zero_average) bind(C, name='gpu_rad_kick_')
    use, intrinsic :: iso_c_binding
    integer(C_INT), value, intent(in) :: n
    real(C_DOUBLE), intent(in) :: stoc_mat(36), damp_dmat(36)
    real(C_DOUBLE), intent(in) :: xfer_damp_vec(6), ref_orb(6)
    real(C_DOUBLE), value, intent(in) :: synch_rad_scale
    integer(C_INT), value, intent(in) :: apply_damp, apply_fluct, zero_average
  end subroutine

  subroutine gpu_track_drift_dev(s_pos, mc2, length, n) bind(C, name='gpu_track_drift_dev_')
    use, intrinsic :: iso_c_binding
    real(C_DOUBLE), intent(inout) :: s_pos(*)
    real(C_DOUBLE), value, intent(in) :: mc2, length
    integer(C_INT), value, intent(in) :: n
  end subroutine

  subroutine gpu_track_drift_dev_no_s(mc2, length, n) bind(C, name='gpu_track_drift_dev_no_s_')
    use, intrinsic :: iso_c_binding
    real(C_DOUBLE), value, intent(in) :: mc2, length
    integer(C_INT), value, intent(in) :: n
  end subroutine

  subroutine gpu_track_sextupole(vx, vpx, vy, vpy, vz, vpz, &
                              state, beta, p0c, t_time, &
                              mc2, ele_length, delta_ref_time, &
                              e_tot_ele, charge_dir, n_particles, &
                              a2_arr, b2_arr, cm_arr, &
                              ix_mag_max, n_step, &
                              ea2_arr, eb2_arr, ix_elec_max) bind(C, name='gpu_track_sextupole_')
    use, intrinsic :: iso_c_binding
    real(C_DOUBLE), intent(inout) :: vx(*), vpx(*), vy(*), vpy(*), vz(*), vpz(*)
    integer(C_INT), intent(inout) :: state(*)
    real(C_DOUBLE), intent(inout) :: beta(*), p0c(*), t_time(*)
    real(C_DOUBLE), value, intent(in) :: mc2, ele_length
    real(C_DOUBLE), value, intent(in) :: delta_ref_time, e_tot_ele, charge_dir
    integer(C_INT), value, intent(in) :: n_particles
    real(C_DOUBLE), intent(in) :: a2_arr(*), b2_arr(*), cm_arr(*)
    integer(C_INT), value, intent(in) :: ix_mag_max, n_step
    real(C_DOUBLE), intent(in) :: ea2_arr(*), eb2_arr(*)
    integer(C_INT), value, intent(in) :: ix_elec_max
  end subroutine

  subroutine gpu_track_sextupole_dev(mc2, ele_length, delta_ref_time, &
                                 e_tot_ele, charge_dir, n_particles, &
                                 a2_arr, b2_arr, cm_arr, &
                                 ix_mag_max, n_step, &
                                 ea2_arr, eb2_arr, ix_elec_max) bind(C, name='gpu_track_sextupole_dev_')
    use, intrinsic :: iso_c_binding
    real(C_DOUBLE), value, intent(in) :: mc2, ele_length
    real(C_DOUBLE), value, intent(in) :: delta_ref_time, e_tot_ele, charge_dir
    integer(C_INT), value, intent(in) :: n_particles
    real(C_DOUBLE), intent(in) :: a2_arr(*), b2_arr(*), cm_arr(*)
    integer(C_INT), value, intent(in) :: ix_mag_max, n_step
    real(C_DOUBLE), intent(in) :: ea2_arr(*), eb2_arr(*)
    integer(C_INT), value, intent(in) :: ix_elec_max
  end subroutine

  subroutine gpu_track_quad_dev(mc2, b1, ele_length, delta_ref_time, &
                                 e_tot_ele, charge_dir, n_particles, &
                                 a2_arr, b2_arr, cm_arr, &
                                 ix_mag_max, n_step, &
                                 ea2_arr, eb2_arr, ix_elec_max) bind(C, name='gpu_track_quad_dev_')
    use, intrinsic :: iso_c_binding
    real(C_DOUBLE), value, intent(in) :: mc2, b1, ele_length
    real(C_DOUBLE), value, intent(in) :: delta_ref_time, e_tot_ele, charge_dir
    integer(C_INT), value, intent(in) :: n_particles
    real(C_DOUBLE), intent(in) :: a2_arr(*), b2_arr(*), cm_arr(*)
    integer(C_INT), value, intent(in) :: ix_mag_max, n_step
    real(C_DOUBLE), intent(in) :: ea2_arr(*), eb2_arr(*)
    integer(C_INT), value, intent(in) :: ix_elec_max
  end subroutine

  subroutine gpu_track_solenoid(vx, vpx, vy, vpy, vz, vpz, &
                            state, beta, p0c, t_time, &
                            mc2, ks0, ele_length, &
                            delta_ref_time, e_tot_ele, &
                            n_particles, n_step, &
                            a2_arr, b2_arr, cm_arr, &
                            ix_mag_max, &
                            ea2_arr, eb2_arr, ix_elec_max) bind(C, name='gpu_track_solenoid_')
    use, intrinsic :: iso_c_binding
    real(C_DOUBLE), intent(inout) :: vx(*), vpx(*), vy(*), vpy(*), vz(*), vpz(*)
    integer(C_INT), intent(inout) :: state(*)
    real(C_DOUBLE), intent(inout) :: beta(*), p0c(*), t_time(*)
    real(C_DOUBLE), value, intent(in) :: mc2, ks0, ele_length
    real(C_DOUBLE), value, intent(in) :: delta_ref_time, e_tot_ele
    integer(C_INT), value, intent(in) :: n_particles, n_step
    real(C_DOUBLE), intent(in) :: a2_arr(*), b2_arr(*), cm_arr(*)
    integer(C_INT), value, intent(in) :: ix_mag_max
    real(C_DOUBLE), intent(in) :: ea2_arr(*), eb2_arr(*)
    integer(C_INT), value, intent(in) :: ix_elec_max
  end subroutine

  subroutine gpu_track_solenoid_dev(mc2, ks0, ele_length, &
                                 delta_ref_time, e_tot_ele, &
                                 n_particles, n_step, &
                                 a2_arr, b2_arr, cm_arr, &
                                 ix_mag_max, &
                                 ea2_arr, eb2_arr, ix_elec_max) bind(C, name='gpu_track_solenoid_dev_')
    use, intrinsic :: iso_c_binding
    real(C_DOUBLE), value, intent(in) :: mc2, ks0, ele_length
    real(C_DOUBLE), value, intent(in) :: delta_ref_time, e_tot_ele
    integer(C_INT), value, intent(in) :: n_particles, n_step
    real(C_DOUBLE), intent(in) :: a2_arr(*), b2_arr(*), cm_arr(*)
    integer(C_INT), value, intent(in) :: ix_mag_max
    real(C_DOUBLE), intent(in) :: ea2_arr(*), eb2_arr(*)
    integer(C_INT), value, intent(in) :: ix_elec_max
  end subroutine

  subroutine gpu_track_sol_quad(vx, vpx, vy, vpy, vz, vpz, &
                            state, beta, p0c, t_time, &
                            mc2, ks_in, k1_in, ele_length, &
                            delta_ref_time, e_tot_ele, &
                            n_particles, n_step, &
                            a2_arr, b2_arr, cm_arr, &
                            ix_mag_max, &
                            ea2_arr, eb2_arr, ix_elec_max) bind(C, name='gpu_track_sol_quad_')
    use, intrinsic :: iso_c_binding
    real(C_DOUBLE), intent(inout) :: vx(*), vpx(*), vy(*), vpy(*), vz(*), vpz(*)
    integer(C_INT), intent(inout) :: state(*)
    real(C_DOUBLE), intent(inout) :: beta(*), p0c(*), t_time(*)
    real(C_DOUBLE), value, intent(in) :: mc2, ks_in, k1_in, ele_length
    real(C_DOUBLE), value, intent(in) :: delta_ref_time, e_tot_ele
    integer(C_INT), value, intent(in) :: n_particles, n_step
    real(C_DOUBLE), intent(in) :: a2_arr(*), b2_arr(*), cm_arr(*)
    integer(C_INT), value, intent(in) :: ix_mag_max
    real(C_DOUBLE), intent(in) :: ea2_arr(*), eb2_arr(*)
    integer(C_INT), value, intent(in) :: ix_elec_max
  end subroutine

  subroutine gpu_track_sol_quad_dev(mc2, ks_in, k1_in, ele_length, &
                                 delta_ref_time, e_tot_ele, &
                                 n_particles, n_step, &
                                 a2_arr, b2_arr, cm_arr, &
                                 ix_mag_max, &
                                 ea2_arr, eb2_arr, ix_elec_max) bind(C, name='gpu_track_sol_quad_dev_')
    use, intrinsic :: iso_c_binding
    real(C_DOUBLE), value, intent(in) :: mc2, ks_in, k1_in, ele_length
    real(C_DOUBLE), value, intent(in) :: delta_ref_time, e_tot_ele
    integer(C_INT), value, intent(in) :: n_particles, n_step
    real(C_DOUBLE), intent(in) :: a2_arr(*), b2_arr(*), cm_arr(*)
    integer(C_INT), value, intent(in) :: ix_mag_max
    real(C_DOUBLE), intent(in) :: ea2_arr(*), eb2_arr(*)
    integer(C_INT), value, intent(in) :: ix_elec_max
  end subroutine

  subroutine gpu_track_wiggler(vx, vpx, vy, vpy, vz, vpz, &
                            state, beta, p0c, t_time, &
                            mc2, ele_length, &
                            delta_ref_time, e_tot_ele, p0c_ele, &
                            k1x, k1y, kz, is_helical, &
                            osc_amp, &
                            n_particles, n_step, &
                            a2_arr, b2_arr, cm_arr, &
                            ix_mag_max, &
                            ea2_arr, eb2_arr, ix_elec_max) bind(C, name='gpu_track_wiggler_')
    use, intrinsic :: iso_c_binding
    real(C_DOUBLE), intent(inout) :: vx(*), vpx(*), vy(*), vpy(*), vz(*), vpz(*)
    integer(C_INT), intent(inout) :: state(*)
    real(C_DOUBLE), intent(inout) :: beta(*), p0c(*), t_time(*)
    real(C_DOUBLE), value, intent(in) :: mc2, ele_length
    real(C_DOUBLE), value, intent(in) :: delta_ref_time, e_tot_ele, p0c_ele
    real(C_DOUBLE), value, intent(in) :: k1x, k1y, kz
    integer(C_INT), value, intent(in) :: is_helical
    real(C_DOUBLE), value, intent(in) :: osc_amp
    integer(C_INT), value, intent(in) :: n_particles, n_step
    real(C_DOUBLE), intent(in) :: a2_arr(*), b2_arr(*), cm_arr(*)
    integer(C_INT), value, intent(in) :: ix_mag_max
    real(C_DOUBLE), intent(in) :: ea2_arr(*), eb2_arr(*)
    integer(C_INT), value, intent(in) :: ix_elec_max
  end subroutine

  subroutine gpu_track_wiggler_dev(mc2, ele_length, &
                                 delta_ref_time, e_tot_ele, p0c_ele, &
                                 k1x, k1y, kz, is_helical, &
                                 osc_amp, &
                                 n_particles, n_step, &
                                 a2_arr, b2_arr, cm_arr, &
                                 ix_mag_max, &
                                 ea2_arr, eb2_arr, ix_elec_max) bind(C, name='gpu_track_wiggler_dev_')
    use, intrinsic :: iso_c_binding
    real(C_DOUBLE), value, intent(in) :: mc2, ele_length
    real(C_DOUBLE), value, intent(in) :: delta_ref_time, e_tot_ele, p0c_ele
    real(C_DOUBLE), value, intent(in) :: k1x, k1y, kz
    integer(C_INT), value, intent(in) :: is_helical
    real(C_DOUBLE), value, intent(in) :: osc_amp
    integer(C_INT), value, intent(in) :: n_particles, n_step
    real(C_DOUBLE), intent(in) :: a2_arr(*), b2_arr(*), cm_arr(*)
    integer(C_INT), value, intent(in) :: ix_mag_max
    real(C_DOUBLE), intent(in) :: ea2_arr(*), eb2_arr(*)
    integer(C_INT), value, intent(in) :: ix_elec_max
  end subroutine

  subroutine gpu_track_bend_dev(mc2, g, g_tot, dg, b1, &
                                 ele_length, delta_ref_time, e_tot_ele, &
                                 rel_charge_dir, p0c_ele, n_particles, &
                                 a2_arr, b2_arr, cm_arr, &
                                 ix_mag_max, n_step, &
                                 ea2_arr, eb2_arr, ix_elec_max, &
                                 is_exact, exact_an, exact_bn, &
                                 ix_exact_mag_max, rho_val, c_dir_val, &
                                 exact_f_scale) bind(C, name='gpu_track_bend_dev_')
    use, intrinsic :: iso_c_binding
    real(C_DOUBLE), value, intent(in) :: mc2, g, g_tot, dg, b1
    real(C_DOUBLE), value, intent(in) :: ele_length, delta_ref_time, e_tot_ele
    real(C_DOUBLE), value, intent(in) :: rel_charge_dir, p0c_ele
    integer(C_INT), value, intent(in) :: n_particles
    real(C_DOUBLE), intent(in) :: a2_arr(*), b2_arr(*), cm_arr(*)
    integer(C_INT), value, intent(in) :: ix_mag_max, n_step
    real(C_DOUBLE), intent(in) :: ea2_arr(*), eb2_arr(*)
    integer(C_INT), value, intent(in) :: ix_elec_max
    integer(C_INT), value, intent(in) :: is_exact
    real(C_DOUBLE), intent(in) :: exact_an(*), exact_bn(*)
    integer(C_INT), value, intent(in) :: ix_exact_mag_max
    real(C_DOUBLE), value, intent(in) :: rho_val, c_dir_val, exact_f_scale
  end subroutine

  subroutine gpu_track_lcavity_dev(mc2, &
                                    step_s0, step_s, step_p0c, step_p1c, &
                                    step_scale, step_time, &
                                    n_rf_steps, &
                                    voltage, voltage_err, field_autoscale, &
                                    rf_frequency, phi0_total, &
                                    voltage_tot, l_active, &
                                    cavity_type, &
                                    fringe_at, charge_ratio, &
                                    n_particles, &
                                    abs_time, phi0_no_multi, &
                                    ref_time_start) bind(C, name='gpu_track_lcavity_dev_')
    use, intrinsic :: iso_c_binding
    real(C_DOUBLE), value, intent(in) :: mc2
    real(C_DOUBLE), intent(in) :: step_s0(*), step_s(*), step_p0c(*), step_p1c(*)
    real(C_DOUBLE), intent(in) :: step_scale(*), step_time(*)
    integer(C_INT), value, intent(in) :: n_rf_steps
    real(C_DOUBLE), value, intent(in) :: voltage, voltage_err, field_autoscale
    real(C_DOUBLE), value, intent(in) :: rf_frequency, phi0_total
    real(C_DOUBLE), value, intent(in) :: voltage_tot, l_active
    integer(C_INT), value, intent(in) :: cavity_type
    integer(C_INT), value, intent(in) :: fringe_at
    real(C_DOUBLE), value, intent(in) :: charge_ratio
    integer(C_INT), value, intent(in) :: n_particles
    integer(C_INT), value, intent(in) :: abs_time
    real(C_DOUBLE), value, intent(in) :: phi0_no_multi
    real(C_DOUBLE), value, intent(in) :: ref_time_start
  end subroutine

  ! ----- Cross-element persistence kernels -----

  subroutine gpu_quad_fringe(k1, fq1, fq2, charge_dir, &
      fringe_type, edge, time_dir, n) bind(C, name='gpu_quad_fringe_')
    use, intrinsic :: iso_c_binding
    real(C_DOUBLE), value, intent(in) :: k1, fq1, fq2, charge_dir
    integer(C_INT), value, intent(in) :: fringe_type, edge, time_dir, n
  end subroutine

  subroutine gpu_hard_multipole_edge(h_bp, h_ap, n_max, charge_dir, is_entrance, n_particles) &
      bind(C, name='gpu_hard_multipole_edge_')
    use, intrinsic :: iso_c_binding
    real(C_DOUBLE), intent(in) :: h_bp(*), h_ap(*)
    integer(C_INT), value, intent(in) :: n_max, is_entrance, n_particles
    real(C_DOUBLE), value, intent(in) :: charge_dir
  end subroutine

  subroutine gpu_exact_bend_fringe(g_tot, beta0, edge_angle, fint_signed, hgap, &
      is_exit, n) bind(C, name='gpu_exact_bend_fringe_')
    use, intrinsic :: iso_c_binding
    real(C_DOUBLE), value, intent(in) :: g_tot, beta0, edge_angle, fint_signed, hgap
    integer(C_INT), value, intent(in) :: is_exit, n
  end subroutine

  subroutine gpu_bend_fringe(g_tot, e_angle, fint_gap, k1, &
      entering, time_dir, n) bind(C, name='gpu_bend_fringe_')
    use, intrinsic :: iso_c_binding
    real(C_DOUBLE), value, intent(in) :: g_tot, e_angle, fint_gap, k1
    integer(C_INT), value, intent(in) :: entering, time_dir, n
  end subroutine

  subroutine gpu_sad_bend_fringe(g, fb, n) bind(C, name='gpu_sad_bend_fringe_')
    use, intrinsic :: iso_c_binding
    real(C_DOUBLE), value, intent(in) :: g, fb
    integer(C_INT), value, intent(in) :: n
  end subroutine

  subroutine gpu_misalign_3d(h_W, Lx, Ly, Lz, set_flag, n) bind(C, name='gpu_misalign_3d_')
    use, intrinsic :: iso_c_binding
    real(C_DOUBLE), intent(in) :: h_W(9)
    real(C_DOUBLE), value, intent(in) :: Lx, Ly, Lz
    integer(C_INT), value, intent(in) :: set_flag, n
  end subroutine

  subroutine gpu_misalign(x_off, y_off, tilt, set_flag, n) bind(C, name='gpu_misalign_')
    use, intrinsic :: iso_c_binding
    real(C_DOUBLE), value, intent(in) :: x_off, y_off, tilt
    integer(C_INT), value, intent(in) :: set_flag, n
  end subroutine

  subroutine gpu_bend_offset(g, rho, L_half, bend_angle, &
      ref_tilt, roll_tot, x_off, y_off, z_off, x_pitch, y_pitch, &
      set_flag, n) bind(C, name='gpu_bend_offset_')
    use, intrinsic :: iso_c_binding
    real(C_DOUBLE), value, intent(in) :: g, rho, L_half, bend_angle
    real(C_DOUBLE), value, intent(in) :: ref_tilt, roll_tot
    real(C_DOUBLE), value, intent(in) :: x_off, y_off, z_off
    real(C_DOUBLE), value, intent(in) :: x_pitch, y_pitch
    integer(C_INT), value, intent(in) :: set_flag, n
  end subroutine

  subroutine gpu_check_aperture_rect(x1_lim, x2_lim, y1_lim, y2_lim, n) &
      bind(C, name='gpu_check_aperture_rect_')
    use, intrinsic :: iso_c_binding
    real(C_DOUBLE), value, intent(in) :: x1_lim, x2_lim, y1_lim, y2_lim
    integer(C_INT), value, intent(in) :: n
  end subroutine

  subroutine gpu_check_aperture_ellipse(x1_lim, x2_lim, y1_lim, y2_lim, n) &
      bind(C, name='gpu_check_aperture_ellipse_')
    use, intrinsic :: iso_c_binding
    real(C_DOUBLE), value, intent(in) :: x1_lim, x2_lim, y1_lim, y2_lim
    integer(C_INT), value, intent(in) :: n
  end subroutine

  subroutine gpu_s_update(s_val, n) bind(C, name='gpu_s_update_')
    use, intrinsic :: iso_c_binding
    real(C_DOUBLE), value, intent(in) :: s_val
    integer(C_INT), value, intent(in) :: n
  end subroutine

  subroutine gpu_orbit_check(n) bind(C, name='gpu_orbit_check_')
    use, intrinsic :: iso_c_binding
    integer(C_INT), value, intent(in) :: n
  end subroutine

  ! ----- Space charge and CSR GPU wrappers -----

  subroutine gpu_space_charge_3d(h_charge, n_particles, &
      nx, ny, nz, gamma, ds_step, mc2, dct_ave) bind(C, name='gpu_space_charge_3d_')
    use, intrinsic :: iso_c_binding
    real(C_DOUBLE), intent(in) :: h_charge(*)
    integer(C_INT), value, intent(in) :: n_particles, nx, ny, nz
    real(C_DOUBLE), value, intent(in) :: gamma, ds_step, mc2, dct_ave
  end subroutine

  subroutine gpu_csr_z_minmax(h_z_min, h_z_max, n_particles) bind(C, name='gpu_csr_z_minmax_')
    use, intrinsic :: iso_c_binding
    real(C_DOUBLE), intent(out) :: h_z_min, h_z_max
    integer(C_INT), value, intent(in) :: n_particles
  end subroutine

  subroutine gpu_csr_bin_particles(h_charge, n_particles, &
      h_bin_charge, h_bin_x0_wt, h_bin_y0_wt, h_bin_n_particle, &
      z_min, dz_slice, dz_particle, &
      n_bin, particle_bin_span) bind(C, name='gpu_csr_bin_particles_')
    use, intrinsic :: iso_c_binding
    real(C_DOUBLE), intent(in) :: h_charge(*)
    integer(C_INT), value, intent(in) :: n_particles
    real(C_DOUBLE), intent(out) :: h_bin_charge(*), h_bin_x0_wt(*), h_bin_y0_wt(*), h_bin_n_particle(*)
    real(C_DOUBLE), value, intent(in) :: z_min, dz_slice, dz_particle
    integer(C_INT), value, intent(in) :: n_bin, particle_bin_span
  end subroutine

  subroutine gpu_csr_apply_kicks(h_kick_csr, h_kick_lsc, &
      z_center_0, dz_slice, apply_csr, apply_lsc, &
      n_bin, n_particles) bind(C, name='gpu_csr_apply_kicks_')
    use, intrinsic :: iso_c_binding
    real(C_DOUBLE), intent(in) :: h_kick_csr(*), h_kick_lsc(*)
    real(C_DOUBLE), value, intent(in) :: z_center_0, dz_slice
    integer(C_INT), value, intent(in) :: apply_csr, apply_lsc, n_bin, n_particles
  end subroutine

  subroutine gpu_csr_bin_kicks(h_floor0_x, h_floor0_z, h_floor0_theta, &
      h_floor1_x, h_floor1_z, h_floor1_theta, &
      h_L_chord, h_theta_chord, h_spline_coef, h_dL_s, h_ele_s, h_ele_key, &
      n_ele, ix_ele_kick, s_chord_kick, floor_k_x, floor_k_z, &
      gamma, gamma2, beta2, y_source, dz_slice, n_bin, kick_factor, &
      actual_track_step, species_radius, rel_mass, e_charge_abs, &
      csr_method_one_dim, h_edge_dcdz, h_slice_charge, &
      h_kick_csr, h_I_csr_out) bind(C, name='gpu_csr_bin_kicks_')
    use, intrinsic :: iso_c_binding
    real(C_DOUBLE), intent(in) :: h_floor0_x(*), h_floor0_z(*), h_floor0_theta(*)
    real(C_DOUBLE), intent(in) :: h_floor1_x(*), h_floor1_z(*), h_floor1_theta(*)
    real(C_DOUBLE), intent(in) :: h_L_chord(*), h_theta_chord(*), h_spline_coef(*)
    real(C_DOUBLE), intent(in) :: h_dL_s(*), h_ele_s(*)
    integer(C_INT), intent(in) :: h_ele_key(*)
    integer(C_INT), value, intent(in) :: n_ele, ix_ele_kick, n_bin, csr_method_one_dim
    real(C_DOUBLE), value, intent(in) :: s_chord_kick, floor_k_x, floor_k_z
    real(C_DOUBLE), value, intent(in) :: gamma, gamma2, beta2, y_source, dz_slice
    real(C_DOUBLE), value, intent(in) :: kick_factor, actual_track_step
    real(C_DOUBLE), value, intent(in) :: species_radius, rel_mass, e_charge_abs
    real(C_DOUBLE), intent(in) :: h_edge_dcdz(*), h_slice_charge(*)
    real(C_DOUBLE), intent(out) :: h_kick_csr(*), h_I_csr_out(*)
  end subroutine

  subroutine gpu_spacecharge_cleanup() bind(C, name='gpu_spacecharge_cleanup_')
  end subroutine

  ! ----- Multi-bunch buffer save/restore -----

  integer(C_INT) function gpu_save_bunch_buffers(vx, vpx, vy, vpy, vz, vpz, &
                              state, beta, p0c, t_time, s_pos, n) bind(C, name='gpu_save_bunch_buffers_')
    use, intrinsic :: iso_c_binding
    real(C_DOUBLE), intent(out) :: vx(*), vpx(*), vy(*), vpy(*), vz(*), vpz(*)
    integer(C_INT), intent(out) :: state(*)
    real(C_DOUBLE), intent(out) :: beta(*), p0c(*), t_time(*), s_pos(*)
    integer(C_INT), value, intent(in) :: n
  end function

  integer(C_INT) function gpu_restore_bunch_buffers(vx, vpx, vy, vpy, vz, vpz, &
                              state, beta, p0c, t_time, s_pos, n) bind(C, name='gpu_restore_bunch_buffers_')
    use, intrinsic :: iso_c_binding
    real(C_DOUBLE), intent(in) :: vx(*), vpx(*), vy(*), vpy(*), vz(*), vpz(*)
    integer(C_INT), intent(in) :: state(*)
    real(C_DOUBLE), intent(in) :: beta(*), p0c(*), t_time(*), s_pos(*)
    integer(C_INT), value, intent(in) :: n
  end function

  integer(C_INT) function gpu_get_buffer_cap() bind(C, name='gpu_get_buffer_cap_')
    use, intrinsic :: iso_c_binding
  end function

end interface
#endif

contains

!------------------------------------------------------------------------
! gpu_tracking_init — initialize GPU tracking from env var (call once)
!
! Reads ACC_ENABLE_GPU_TRACKING env var and checks for CUDA hardware.
! Sets bmad_com%gpu_tracking_on = .true. if env var is 'Y' and GPU is
! present. After this, callers can toggle bmad_com%gpu_tracking_on
! directly. Does nothing if called more than once.
!------------------------------------------------------------------------
subroutine gpu_tracking_init()
character(len=32) :: env_val
integer :: env_len, env_stat

if (gpu_trk_initialized) return
gpu_trk_initialized = .true.

#ifdef USE_GPU_TRACKING
if (gpu_tracking_available() == 1) then
  gpu_hw_available = .true.
endif

call get_environment_variable('ACC_ENABLE_GPU_TRACKING', env_val, env_len, env_stat)
if (env_stat == 0 .and. trim(env_val) == 'Y') then
  if (gpu_hw_available) then
    bmad_com%gpu_tracking_on = .true.
    print *, 'gpu_tracking: GPU tracking enabled (ACC_ENABLE_GPU_TRACKING=Y, CUDA GPU detected)'
  else
    print *, 'gpu_tracking: ACC_ENABLE_GPU_TRACKING=Y but no CUDA GPU found — using CPU tracking'
  endif
endif
#endif

end subroutine

!------------------------------------------------------------------------
! gpu_tracking_reset — reset initialization state so gpu_tracking_init
! can re-read the environment variable on next call. Also releases
! cached GPU device memory.
! Used by benchmarks to toggle GPU on/off between runs.
!------------------------------------------------------------------------
subroutine gpu_tracking_reset()
#ifdef USE_GPU_TRACKING
call gpu_tracking_cleanup()
#endif
call gpu_multi_bunch_cleanup()
gpu_trk_initialized = .false.
gpu_persist_on_device = .false.
gpu_persist_bunch_id = 0
gpu_active_slot = 0
bmad_com%gpu_tracking_on = .false.
end subroutine

!------------------------------------------------------------------------
! gpu_tracking_is_active — initialize if needed and return whether
! GPU tracking is currently enabled.
!------------------------------------------------------------------------
function gpu_tracking_is_active() result (is_active)
logical :: is_active
call gpu_tracking_init()
is_active = bmad_com%gpu_tracking_on
end function

!------------------------------------------------------------------------
! ele_gpu_eligible — check if an element can be GPU-tracked
!
! Returns .true. if the element's intrinsic properties allow GPU tracking.
! This checks element type, tracking method, and on/off state.
! Runtime conditions (bmad_com flags, particle direction, wakefields)
! are NOT checked here — those are evaluated at dispatch time.
!
! Currently supported element types: drift, quadrupole, sextupole, octupole,
! thick_multipole, elseparator, sbend, rf_bend, lcavity, pipe, monitor,
! instrument, kicker, hkicker, vkicker, marker, solenoid, sol_quad,
! wiggler, undulator.
!------------------------------------------------------------------------
function ele_gpu_eligible(ele) result (eligible)
type (ele_struct), intent(in) :: ele
logical :: eligible

eligible = .false.

! Must use bmad_standard tracking
if (ele%tracking_method /= bmad_standard$) return

! Must be turned on
if (.not. ele%is_on) return

! Check supported element types
select case (ele%key)
case (drift$, quadrupole$, sextupole$, octupole$, thick_multipole$, elseparator$, sbend$, rf_bend$, lcavity$, pipe$, &
      monitor$, instrument$, kicker$, hkicker$, vkicker$, marker$, solenoid$, sol_quad$, &
      wiggler$, undulator$)
  eligible = .true.
end select

end function ele_gpu_eligible

!------------------------------------------------------------------------
! gpu_rad_eligible — check if radiation should be applied on GPU for this element
!
! Returns .true. if radiation (damping or fluctuations) is requested and
! the element is of a type that generates radiation.
! Drifts do not produce radiation (consistent with track1_radiation).
!------------------------------------------------------------------------
function gpu_rad_eligible(ele) result (eligible)
type (ele_struct), intent(in) :: ele
logical :: eligible

eligible = .false.
if (.not. bmad_com%radiation_damping_on .and. .not. bmad_com%radiation_fluctuations_on) return
if (ele%value(l$) == 0) return

! Drifts, pipes, monitors, and instruments don't produce radiation
select case (ele%key)
case (drift$, pipe$, monitor$, instrument$, kicker$, hkicker$, vkicker$)
  return
end select

eligible = .true.
end function gpu_rad_eligible

!------------------------------------------------------------------------
! ensure_rad_map — ensure radiation map is computed for this element
!------------------------------------------------------------------------
subroutine ensure_rad_map(ele)
use radiation_mod, only: radiation_map_setup
type (ele_struct), intent(inout) :: ele
logical :: err

if (.not. associated(ele%rad_map)) then
  call radiation_map_setup(ele, err)
elseif (ele%rad_map%stale) then
  call radiation_map_setup(ele, err)
endif

end subroutine ensure_rad_map

!------------------------------------------------------------------------
! call_gpu_rad_kick — call the GPU radiation kick kernel
!
! Extracts the stoc_mat, damp_dmat, xfer_damp_vec, and ref_orb from
! the rad_map and calls the CUDA radiation kernel on already-uploaded
! device particle data.
!------------------------------------------------------------------------
subroutine call_gpu_rad_kick(n, rad_map)

use, intrinsic :: iso_c_binding

integer(C_INT),          intent(in) :: n
type (rad_map_struct),   intent(in) :: rad_map

real(C_DOUBLE) :: stoc_flat(36), damp_flat(36), xfer_vec(6), ref(6)
integer(C_INT) :: i_damp, i_fluct, i_zero_avg

! Flatten the 6x6 matrices to 1D column-major arrays
stoc_flat = reshape(rad_map%stoc_mat, [36])
damp_flat = reshape(rad_map%damp_dmat, [36])
xfer_vec  = rad_map%xfer_damp_vec
ref       = rad_map%ref_orb

i_damp = 0; i_fluct = 0; i_zero_avg = 0
if (bmad_com%radiation_damping_on) i_damp = 1
if (bmad_com%radiation_fluctuations_on) i_fluct = 1
if (bmad_com%radiation_zero_average) i_zero_avg = 1

call gpu_rad_kick(n, stoc_flat, damp_flat, xfer_vec, ref, &
                   bmad_com%synch_rad_scale, i_damp, i_fluct, i_zero_avg)

end subroutine call_gpu_rad_kick

!------------------------------------------------------------------------
! precompute_misalign_W — compute 3x3 rotation matrix and offset for
! an element's misalignment by probing offset_particle.
!
! Creates 3 test particles at unit offsets, applies offset_particle,
! and extracts the affine transformation matrix W and offset L.
!------------------------------------------------------------------------
subroutine precompute_misalign_W(ele, set_or_unset, W, Lx, Ly, Lz)

use, intrinsic :: iso_c_binding

type (ele_struct), intent(in) :: ele
logical,           intent(in) :: set_or_unset
real(C_DOUBLE),    intent(out) :: W(3,3), Lx, Ly, Lz

type (coord_struct) :: orb0, orb_dx, orb_dy, orb_dpx
real(rp) :: eps
integer :: edge

eps = 1d-8

! Choose the edge based on set/unset
if (set_or_unset .eqv. set$) then
  edge = upstream_end$
else
  edge = downstream_end$
endif

! Create base particle at on-axis orbit
call init_coord(orb0, ele, edge)
orb0%vec = 0; orb0%vec(6) = 0
call offset_particle(ele, set_or_unset, orb0, set_hvkicks = .false.)
Lx = orb0%vec(1)
Ly = orb0%vec(3)
Lz = 0

! Probe x direction
call init_coord(orb_dx, ele, edge)
orb_dx%vec = 0; orb_dx%vec(1) = eps
call offset_particle(ele, set_or_unset, orb_dx, set_hvkicks = .false.)
W(1,1) = (orb_dx%vec(1) - Lx) / eps
W(2,1) = (orb_dx%vec(3) - Ly) / eps
W(3,1) = 0

! Probe y direction
call init_coord(orb_dy, ele, edge)
orb_dy%vec = 0; orb_dy%vec(3) = eps
call offset_particle(ele, set_or_unset, orb_dy, set_hvkicks = .false.)
W(1,2) = (orb_dy%vec(1) - Lx) / eps
W(2,2) = (orb_dy%vec(3) - Ly) / eps
W(3,2) = 0

! Probe px direction (momentum column)
call init_coord(orb_dpx, ele, edge)
orb_dpx%vec = 0; orb_dpx%vec(2) = eps
call offset_particle(ele, set_or_unset, orb_dpx, set_hvkicks = .false.)
W(1,3) = (orb_dpx%vec(2) - orb0%vec(2)) / eps
W(2,3) = (orb_dpx%vec(4) - orb0%vec(4)) / eps
W(3,3) = 1

end subroutine precompute_misalign_W

!------------------------------------------------------------------------
! bunch_to_soa — extract particle data from AoS bunch to SoA arrays
!------------------------------------------------------------------------
subroutine bunch_to_soa(bunch, n, vx, vpx, vy, vpy, vz, vpz, state_a, beta_a, p0c_a, t_a)

use, intrinsic :: iso_c_binding

type (bunch_struct), intent(in) :: bunch
integer,             intent(in) :: n
real(C_DOUBLE),      intent(out) :: vx(n), vpx(n), vy(n), vpy(n), vz(n), vpz(n)
real(C_DOUBLE),      intent(out) :: beta_a(n), p0c_a(n), t_a(n)
integer(C_INT),      intent(out) :: state_a(n)

integer :: j

do j = 1, n
  vx(j)      = bunch%particle(j)%vec(1)
  vpx(j)     = bunch%particle(j)%vec(2)
  vy(j)      = bunch%particle(j)%vec(3)
  vpy(j)     = bunch%particle(j)%vec(4)
  vz(j)      = bunch%particle(j)%vec(5)
  vpz(j)     = bunch%particle(j)%vec(6)
  state_a(j) = bunch%particle(j)%state
  beta_a(j)  = bunch%particle(j)%beta
  p0c_a(j)   = bunch%particle(j)%p0c
  t_a(j)     = bunch%particle(j)%t
enddo

end subroutine bunch_to_soa

!------------------------------------------------------------------------
! soa_to_bunch — write SoA arrays back to AoS bunch structure
!
! copy_beta/copy_p0c control optional write-back of beta and p0c.
! lcavity: both true. quad/bend: copy_beta true. drift: both false.
!------------------------------------------------------------------------
subroutine soa_to_bunch(bunch, ele, n, vx, vpx, vy, vpy, vz, vpz, state_a, beta_a, p0c_a, t_a, &
                         copy_beta, copy_p0c)

use, intrinsic :: iso_c_binding

type (bunch_struct), intent(inout) :: bunch
type (ele_struct),   intent(in)    :: ele
integer,             intent(in)    :: n
real(C_DOUBLE),      intent(in)    :: vx(n), vpx(n), vy(n), vpy(n), vz(n), vpz(n)
real(C_DOUBLE),      intent(in)    :: beta_a(n), p0c_a(n), t_a(n)
integer(C_INT),      intent(in)    :: state_a(n)
logical,             intent(in)    :: copy_beta, copy_p0c

integer :: j

do j = 1, n
  bunch%particle(j)%vec(1)    = vx(j)
  bunch%particle(j)%vec(2)    = vpx(j)
  bunch%particle(j)%vec(3)    = vy(j)
  bunch%particle(j)%vec(4)    = vpy(j)
  bunch%particle(j)%vec(5)    = vz(j)
  bunch%particle(j)%vec(6)    = vpz(j)
  bunch%particle(j)%state     = state_a(j)
  bunch%particle(j)%t         = t_a(j)
  if (copy_beta) bunch%particle(j)%beta = beta_a(j)
  if (copy_p0c)  bunch%particle(j)%p0c  = p0c_a(j)
  if (state_a(j) == alive$) then
    bunch%particle(j)%location  = downstream_end$
  else
    bunch%particle(j)%location  = inside$
  endif
  bunch%particle(j)%ix_ele    = ele%ix_ele
  bunch%particle(j)%ix_branch = ele%ix_branch
enddo

end subroutine soa_to_bunch

!------------------------------------------------------------------------
! apply_misalign_to_bunch — apply misalignment transform to all alive particles
!------------------------------------------------------------------------
subroutine apply_misalign_to_bunch(bunch, ele, n, set_or_unset)

type (bunch_struct), intent(inout) :: bunch
type (ele_struct),   intent(in)    :: ele
integer,             intent(in)    :: n
logical,             intent(in)    :: set_or_unset  ! set$ or unset$

integer :: j

do j = 1, n
  if (bunch%particle(j)%state == alive$) then
    call offset_particle(ele, set_or_unset, bunch%particle(j), set_hvkicks = .false.)
  endif
enddo

end subroutine apply_misalign_to_bunch

!------------------------------------------------------------------------
! apply_fringe_to_bunch — apply fringe kicks to all alive particles
!------------------------------------------------------------------------
subroutine apply_fringe_to_bunch(bunch, ele, param, n, fringe_info, particle_at)

type (bunch_struct),              intent(inout) :: bunch
type (ele_struct),                intent(in)    :: ele
type (lat_param_struct),          intent(in)    :: param
integer,                          intent(in)    :: n
type (fringe_field_info_struct),  intent(inout) :: fringe_info
integer,                          intent(in)    :: particle_at

integer :: j

fringe_info%particle_at = particle_at
do j = 1, n
  if (bunch%particle(j)%state == alive$) then
    call apply_element_edge_kick(bunch%particle(j), fringe_info, ele, param, .false.)
    if (bunch%particle(j)%state /= alive$) cycle
  endif
enddo

end subroutine apply_fringe_to_bunch

!------------------------------------------------------------------------
! apply_sol_fringe_to_bunch — apply fringe with apply_sol_fringe = .false.
!
! For solenoid/sol_quad elements, the solenoid fringe is embedded in
! the body tracking, so the edge kick must NOT apply the solenoid fringe
! separately.
!------------------------------------------------------------------------
subroutine apply_sol_fringe_to_bunch(bunch, ele, param, n, fringe_info, particle_at)

type (bunch_struct),              intent(inout) :: bunch
type (ele_struct),                intent(in)    :: ele
type (lat_param_struct),          intent(in)    :: param
integer,                          intent(in)    :: n
type (fringe_field_info_struct),  intent(inout) :: fringe_info
integer,                          intent(in)    :: particle_at

integer :: j

fringe_info%particle_at = particle_at
do j = 1, n
  if (bunch%particle(j)%state == alive$) then
    call apply_element_edge_kick(bunch%particle(j), fringe_info, ele, param, .false., &
                                  apply_sol_fringe = .false.)
    if (bunch%particle(j)%state /= alive$) cycle
  endif
enddo

end subroutine apply_sol_fringe_to_bunch

!------------------------------------------------------------------------
! gpu_tracking_pre — common entrance sequence for GPU element tracking
!
! Allocates SoA arrays, applies entrance misalignment and fringe on CPU,
! then extracts particle data into SoA form for the CUDA kernel.
!------------------------------------------------------------------------
subroutine gpu_tracking_pre(bunch, ele, param, n, &
    vx, vpx, vy, vpy, vz, vpz, state_a, beta_a, p0c_a, t_a, &
    has_misalign, fringe_info, apply_fringe)

use, intrinsic :: iso_c_binding

type (bunch_struct),              intent(inout) :: bunch
type (ele_struct),                intent(in)    :: ele
type (lat_param_struct),          intent(in)    :: param
integer(C_INT),                   intent(in)    :: n
real(C_DOUBLE), allocatable,      intent(out)   :: vx(:), vpx(:), vy(:), vpy(:), vz(:), vpz(:)
real(C_DOUBLE), allocatable,      intent(out)   :: beta_a(:), p0c_a(:), t_a(:)
integer(C_INT), allocatable,      intent(out)   :: state_a(:)
logical,                          intent(in)    :: has_misalign
type (fringe_field_info_struct),  intent(inout) :: fringe_info
logical,                          intent(in)    :: apply_fringe

allocate(vx(n), vpx(n), vy(n), vpy(n), vz(n), vpz(n))
allocate(state_a(n), beta_a(n), p0c_a(n), t_a(n))

if (has_misalign) call apply_misalign_to_bunch(bunch, ele, n, set$)
if (apply_fringe .and. fringe_info%has_fringe) &
  call apply_fringe_to_bunch(bunch, ele, param, n, fringe_info, first_track_edge$)

call bunch_to_soa(bunch, n, vx, vpx, vy, vpy, vz, vpz, state_a, beta_a, p0c_a, t_a)

end subroutine gpu_tracking_pre

!------------------------------------------------------------------------
! gpu_tracking_post — common exit sequence for GPU element tracking
!
! Writes SoA arrays back to bunch, deallocates, applies exit fringe
! and misalignment on CPU, and optionally updates s position.
!------------------------------------------------------------------------
subroutine gpu_tracking_post(bunch, ele, param, n, &
    vx, vpx, vy, vpy, vz, vpz, state_a, beta_a, p0c_a, t_a, &
    has_misalign, fringe_info, apply_fringe, &
    copy_beta, copy_p0c, update_s)

use, intrinsic :: iso_c_binding

type (bunch_struct),              intent(inout) :: bunch
type (ele_struct),                intent(in)    :: ele
type (lat_param_struct),          intent(in)    :: param
integer(C_INT),                   intent(in)    :: n
real(C_DOUBLE), allocatable,      intent(inout) :: vx(:), vpx(:), vy(:), vpy(:), vz(:), vpz(:)
real(C_DOUBLE), allocatable,      intent(inout) :: beta_a(:), p0c_a(:), t_a(:)
integer(C_INT), allocatable,      intent(inout) :: state_a(:)
logical,                          intent(in)    :: has_misalign
type (fringe_field_info_struct),  intent(inout) :: fringe_info
logical,                          intent(in)    :: apply_fringe
logical,                          intent(in)    :: copy_beta, copy_p0c, update_s

integer :: j

call soa_to_bunch(bunch, ele, n, vx, vpx, vy, vpy, vz, vpz, state_a, beta_a, p0c_a, t_a, &
                   copy_beta, copy_p0c)

deallocate(vx, vpx, vy, vpy, vz, vpz)
deallocate(state_a, beta_a, p0c_a, t_a)

if (apply_fringe .and. fringe_info%has_fringe) &
  call apply_fringe_to_bunch(bunch, ele, param, n, fringe_info, second_track_edge$)
if (has_misalign) call apply_misalign_to_bunch(bunch, ele, n, unset$)

if (update_s) then
  do j = 1, n
    if (bunch%particle(j)%state == alive$) then
      bunch%particle(j)%s = ele%s
    endif
  enddo
endif

end subroutine gpu_tracking_post

!------------------------------------------------------------------------
! precompute_multipole_arrays — compute scaled multipole coefficients for CUDA
!
! Computes a2/b2 (magnetic), ea2/eb2 (electric), and cm (c_multi) arrays
! from the raw multipole data.  These include all element-level scaling
! factors; per-particle factors (1/beta for electric, (1+g*x) for bends)
! are applied in the CUDA kernels.
!------------------------------------------------------------------------
subroutine precompute_multipole_arrays(particle1, ele, &
    ix_mag_max, an, bn, ix_elec_max, an_elec, bn_elec, &
    ele_length, n_step, &
    a2_arr, b2_arr, ea2_arr, eb2_arr, cm_arr)

use, intrinsic :: iso_c_binding

type (coord_struct),  intent(in)  :: particle1
type (ele_struct),    intent(in)  :: ele
integer,              intent(in)  :: ix_mag_max, ix_elec_max, n_step
real(rp),             intent(in)  :: an(0:), bn(0:)
real(rp),             intent(in)  :: an_elec(0:), bn_elec(0:)
real(rp),             intent(in)  :: ele_length
real(C_DOUBLE),       intent(out) :: a2_arr(0:n_pole_maxx), b2_arr(0:n_pole_maxx)
real(C_DOUBLE),       intent(out) :: ea2_arr(0:n_pole_maxx), eb2_arr(0:n_pole_maxx)
real(C_DOUBLE),       intent(out) :: cm_arr(0:n_pole_maxx, 0:n_pole_maxx)

integer :: nn, mm
real(rp) :: r_ratio, r_step, step_len_val, f_charge, f_elec

a2_arr = 0; b2_arr = 0; ea2_arr = 0; eb2_arr = 0; cm_arr = 0

r_ratio = ele%value(p0c$) / particle1%p0c
r_step = 1.0_rp / n_step
step_len_val = ele_length / n_step

! Magnetic multipoles
if (ix_mag_max > -1) then
  f_charge = particle1%direction * ele%orientation * &
             charge_to_mass_of(particle1%species) / charge_to_mass_of(ele%ref_species)
  do nn = 0, ix_mag_max
    a2_arr(nn) = r_ratio * an(nn) * f_charge * r_step
    b2_arr(nn) = r_ratio * bn(nn) * f_charge * r_step
  enddo
endif

! Electric multipoles (1/beta applied per-particle in kernel)
if (ix_elec_max > -1) then
  f_elec = charge_of(particle1%species) / particle1%p0c
  do nn = 0, ix_elec_max
    ea2_arr(nn) =  r_ratio * an_elec(nn) * f_elec * step_len_val
    eb2_arr(nn) = -r_ratio * bn_elec(nn) * f_elec * step_len_val
  enddo
endif

! c_multi coefficient table
do nn = 0, max(ix_mag_max, ix_elec_max)
  do mm = 0, nn
    cm_arr(nn, mm) = c_multi(nn, mm, .true.)
  enddo
enddo

end subroutine precompute_multipole_arrays

!------------------------------------------------------------------------
! track_bunch_thru_drift_gpu
!
! GPU batch tracking of all particles in a bunch through a drift.
! Extracts particle data into SoA arrays, calls CUDA kernel, writes back.
!------------------------------------------------------------------------
subroutine track_bunch_thru_drift_gpu (bunch, ele, did_track)

use, intrinsic :: iso_c_binding

type (bunch_struct), intent(inout) :: bunch
type (ele_struct),   intent(in)    :: ele
logical,             intent(out)   :: did_track

#ifdef USE_GPU_TRACKING
integer(C_INT) :: n
integer :: j
real(rp) :: length, mc2

real(C_DOUBLE), allocatable :: vx(:), vpx(:), vy(:), vpy(:), vz(:), vpz(:)
real(C_DOUBLE), allocatable :: beta_a(:), p0c_a(:), s_a(:), t_a(:)
integer(C_INT), allocatable :: state_a(:)
#endif

did_track = .false.

#ifdef USE_GPU_TRACKING
n = size(bunch%particle)
if (n == 0) return
length = ele%value(l$)
if (length == 0) return

mc2 = mass_of(bunch%particle(1)%species)

! Allocate SoA arrays
allocate(vx(n), vpx(n), vy(n), vpy(n), vz(n), vpz(n))
allocate(state_a(n), beta_a(n), p0c_a(n), s_a(n), t_a(n))

! AoS -> SoA extraction
call bunch_to_soa(bunch, n, vx, vpx, vy, vpy, vz, vpz, state_a, beta_a, p0c_a, t_a)
do j = 1, n
  s_a(j) = bunch%particle(j)%s
enddo

did_track = .true.

! Call CUDA kernel
call gpu_track_drift(vx, vpx, vy, vpy, vz, vpz, &
                     state_a, beta_a, p0c_a, s_a, t_a, &
                     mc2, length, n)

! SoA -> AoS write-back
call soa_to_bunch(bunch, ele, n, vx, vpx, vy, vpy, vz, vpz, state_a, beta_a, p0c_a, t_a, &
                   .false., .false.)
do j = 1, n
  bunch%particle(j)%s = s_a(j)
enddo

deallocate(vx, vpx, vy, vpy, vz, vpz)
deallocate(state_a, beta_a, p0c_a, s_a, t_a)
#endif

end subroutine track_bunch_thru_drift_gpu

!------------------------------------------------------------------------
! track_bunch_thru_quad_gpu
!
! GPU batch tracking through a quadrupole.  Handles fringe fields and
! misalignment on CPU (before/after GPU body tracking), and magnetic/
! electric multipole kicks in the CUDA kernel via split-step integration.
!------------------------------------------------------------------------
subroutine track_bunch_thru_quad_gpu (bunch, ele, param, did_track)

use multipole_mod, only: ab_multipole_kicks
use, intrinsic :: iso_c_binding

type (bunch_struct),     intent(inout) :: bunch
type (ele_struct), target, intent(inout) :: ele
type (lat_param_struct), intent(in)    :: param
logical,                 intent(out)   :: did_track

#ifdef USE_GPU_TRACKING
integer, parameter :: n_multi = n_pole_maxx + 1  ! = 22
integer(C_INT) :: n
integer :: ix_mag_max, ix_elec_max, n_step
real(rp) :: ele_length, mc2, b1, delta_ref_time, e_tot_ele
real(rp) :: charge_dir, rel_tracking_charge, length
real(rp) :: an(0:n_pole_maxx), bn(0:n_pole_maxx)
real(rp) :: an_elec(0:n_pole_maxx), bn_elec(0:n_pole_maxx)
type (fringe_field_info_struct) :: fringe_info
logical :: has_misalign, has_mag_multipoles, apply_rad

! Precomputed scaled multipole arrays and c_multi coefficients for CUDA
real(C_DOUBLE) :: a2_arr(0:n_pole_maxx), b2_arr(0:n_pole_maxx)
real(C_DOUBLE) :: ea2_arr(0:n_pole_maxx), eb2_arr(0:n_pole_maxx)
real(C_DOUBLE) :: cm_arr(0:n_pole_maxx, 0:n_pole_maxx)

real(C_DOUBLE), allocatable :: vx(:), vpx(:), vy(:), vpy(:), vz(:), vpz(:)
real(C_DOUBLE), allocatable :: beta_a(:), p0c_a(:), t_a(:)
integer(C_INT), allocatable :: state_a(:)
#endif

did_track = .false.

#ifdef USE_GPU_TRACKING
n = size(bunch%particle)
if (n == 0) return
ele_length = ele%value(l$)
if (ele_length == 0) then
  did_track = .true.
  return
endif

mc2 = mass_of(bunch%particle(1)%species)
delta_ref_time = ele%value(delta_ref_time$)
e_tot_ele = ele%value(e_tot$)

! --- Safety checks: bail out to CPU if element has unsupported features ---

has_misalign = ele%bookkeeping_state%has_misalign

! Fringe fields are handled on CPU (before/after GPU body tracking)
call init_fringe_info(fringe_info, ele)

! Get the quad gradient b1 and check for extra multipoles
call multipole_ele_to_ab(ele, .false., ix_mag_max, an, bn, magnetic$, include_kicks$, b1)
call multipole_ele_to_ab(ele, .false., ix_elec_max, an_elec, bn_elec, electric$)

! Determine n_step for split-step integration
has_mag_multipoles = (ix_mag_max > -1)
length = bunch%particle(1)%time_dir * ele_length
n_step = 1
if (has_mag_multipoles .or. ix_elec_max > -1) &
  n_step = max(nint(abs(length) / ele%value(ds_step$)), 1)

! Compute charge_dir: rel_charge * orientation * direction * time_dir
rel_tracking_charge = rel_tracking_charge_to_mass(bunch%particle(1), param%particle)
charge_dir = rel_tracking_charge * ele%orientation * bunch%particle(1)%direction * bunch%particle(1)%time_dir

! Check if radiation should be applied
apply_rad = gpu_rad_eligible(ele)
if (apply_rad) call ensure_rad_map(ele)

! Entrance: allocate SoA, misalignment, fringe, AoS→SoA
call gpu_tracking_pre(bunch, ele, param, n, vx, vpx, vy, vpy, vz, vpz, &
    state_a, beta_a, p0c_a, t_a, has_misalign, fringe_info, .true.)

! Precompute scaled multipole arrays for CUDA kernel
call precompute_multipole_arrays(bunch%particle(1), ele, &
    ix_mag_max, an, bn, ix_elec_max, an_elec, bn_elec, &
    ele_length, n_step, a2_arr, b2_arr, ea2_arr, eb2_arr, cm_arr)

did_track = .true.

if (apply_rad .and. associated(ele%rad_map)) then
  ! Split approach: upload once, rad + body + rad, download once
  call gpu_upload_particles(vx, vpx, vy, vpy, vz, vpz, &
                            state_a, beta_a, p0c_a, t_a, n)
  ! Entrance radiation kick
  call call_gpu_rad_kick(n, ele%rad_map%rm0)
  ! Body kernel (data already on device)
  call gpu_track_quad_dev(mc2, b1, ele_length, delta_ref_time, &
                          e_tot_ele, charge_dir, n, &
                          a2_arr, b2_arr, cm_arr, &
                          int(ix_mag_max, C_INT), int(n_step, C_INT), &
                          ea2_arr, eb2_arr, int(ix_elec_max, C_INT))
  ! Exit radiation kick
  call call_gpu_rad_kick(n, ele%rad_map%rm1)
  ! Download
  call gpu_download_particles(vx, vpx, vy, vpy, vz, vpz, &
                              state_a, beta_a, p0c_a, t_a, &
                              n, merge(1, 0, ix_elec_max >= 0), 0)
else
  ! Original approach: combined upload + body + download
  call gpu_track_quad(vx, vpx, vy, vpy, vz, vpz, &
                      state_a, beta_a, p0c_a, t_a, &
                      mc2, b1, ele_length, delta_ref_time, &
                      e_tot_ele, charge_dir, n, &
                      a2_arr, b2_arr, cm_arr, &
                      int(ix_mag_max, C_INT), int(n_step, C_INT), &
                      ea2_arr, eb2_arr, int(ix_elec_max, C_INT))
endif

! Exit: SoA→AoS, deallocate, fringe, misalignment, update s
call gpu_tracking_post(bunch, ele, param, n, vx, vpx, vy, vpy, vz, vpz, &
    state_a, beta_a, p0c_a, t_a, has_misalign, fringe_info, .true., &
    .true., .false., .true.)
#endif

end subroutine track_bunch_thru_quad_gpu

!------------------------------------------------------------------------
! track_bunch_thru_sextupole_gpu
!
! Per-element GPU tracking for sextupole elements.
! Uses drift-kick-drift split-step integrator.
!------------------------------------------------------------------------
subroutine track_bunch_thru_sextupole_gpu (bunch, ele, param, did_track)

use multipole_mod, only: ab_multipole_kicks
use, intrinsic :: iso_c_binding

type (bunch_struct),     intent(inout) :: bunch
type (ele_struct), target, intent(inout) :: ele
type (lat_param_struct), intent(in)    :: param
logical,                 intent(out)   :: did_track

#ifdef USE_GPU_TRACKING
integer(C_INT) :: n
integer :: ix_mag_max, ix_elec_max, n_step, j
real(rp) :: mc2, ele_length, delta_ref_time, e_tot_ele
real(rp) :: charge_dir, rel_tracking_charge, length
real(rp) :: an(0:n_pole_maxx), bn(0:n_pole_maxx)
real(rp) :: an_elec(0:n_pole_maxx), bn_elec(0:n_pole_maxx)
real(C_DOUBLE) :: a2_arr(0:n_pole_maxx), b2_arr(0:n_pole_maxx)
real(C_DOUBLE) :: ea2_arr(0:n_pole_maxx), eb2_arr(0:n_pole_maxx)
real(C_DOUBLE) :: cm_arr(0:n_pole_maxx, 0:n_pole_maxx)
real(rp) :: b1_dummy
logical :: has_misalign
type (fringe_field_info_struct) :: fringe_info

real(C_DOUBLE), allocatable :: vx(:), vpx(:), vy(:), vpy(:), vz(:), vpz(:)
real(C_DOUBLE), allocatable :: beta_a(:), p0c_a(:), t_a(:)
integer(C_INT), allocatable :: state_a(:)
#endif

did_track = .false.

#ifdef USE_GPU_TRACKING
n = size(bunch%particle)
if (n == 0) return
ele_length = ele%value(l$)
if (ele_length == 0) return

mc2 = mass_of(bunch%particle(1)%species)
delta_ref_time = ele%value(delta_ref_time$)
e_tot_ele = ele%value(e_tot$)

! Extract multipole coefficients
call multipole_ele_to_ab(ele, .false., ix_mag_max, an, bn, magnetic$, include_kicks$, b1_dummy)
call multipole_ele_to_ab(ele, .false., ix_elec_max, an_elec, bn_elec, electric$)

length = bunch%particle(1)%time_dir * ele_length
n_step = 1
if (ix_mag_max > -1 .or. ix_elec_max > -1) &
  n_step = max(nint(abs(length) / ele%value(ds_step$)), 1)

rel_tracking_charge = rel_tracking_charge_to_mass(bunch%particle(1), param%particle)
charge_dir = rel_tracking_charge * ele%orientation * bunch%particle(1)%direction * bunch%particle(1)%time_dir

call precompute_multipole_arrays(bunch%particle(1), ele, &
    ix_mag_max, an, bn, ix_elec_max, an_elec, bn_elec, &
    ele_length, n_step, a2_arr, b2_arr, ea2_arr, eb2_arr, cm_arr)

! Pre: misalignment, fringe, allocate SoA, bunch_to_soa
call gpu_tracking_pre(bunch, ele, param, n, vx, vpx, vy, vpy, vz, vpz, &
    state_a, beta_a, p0c_a, t_a, has_misalign, fringe_info, .true.)

did_track = .true.

call gpu_track_sextupole(vx, vpx, vy, vpy, vz, vpz, &
                         state_a, beta_a, p0c_a, t_a, &
                         mc2, ele_length, delta_ref_time, &
                         e_tot_ele, charge_dir, n, &
                         a2_arr, b2_arr, cm_arr, &
                         int(ix_mag_max, C_INT), int(n_step, C_INT), &
                         ea2_arr, eb2_arr, int(ix_elec_max, C_INT))

! Post: SoA->AoS, fringe, misalignment, update s
call gpu_tracking_post(bunch, ele, param, n, vx, vpx, vy, vpy, vz, vpz, &
    state_a, beta_a, p0c_a, t_a, has_misalign, fringe_info, .true., &
    .true., .false., .true.)

if (allocated(vx)) deallocate(vx, vpx, vy, vpy, vz, vpz)
if (allocated(state_a)) deallocate(state_a, beta_a, p0c_a, t_a)
#endif

end subroutine track_bunch_thru_sextupole_gpu

!------------------------------------------------------------------------
! track_bunch_thru_bend_gpu
!
! GPU batch tracking through a bend (sbend).
! CPU sandwich pattern: misalignment + fringe on CPU, body on GPU.
! Handles all three body paths: general bend, k1 map, and drift fallback.
!------------------------------------------------------------------------
subroutine track_bunch_thru_bend_gpu (bunch, ele, param, did_track)

use multipole_mod, only: ab_multipole_kicks
use, intrinsic :: iso_c_binding

type (bunch_struct),     intent(inout) :: bunch
type (ele_struct), target, intent(inout) :: ele
type (lat_param_struct), intent(in)    :: param
logical,                 intent(out)   :: did_track

#ifdef USE_GPU_TRACKING
integer, parameter :: n_multi = n_pole_maxx + 1
integer(C_INT) :: n
integer :: ix_mag_max, ix_elec_max, n_step, ix_exact_mag_max
real(rp) :: ele_length, mc2, b1, delta_ref_time, e_tot_ele, p0c_ele
real(rp) :: g, g_tot, dg, rel_charge_dir, c_dir_val, rho_val, exact_f_scale_val
real(rp) :: r_step, length, step_len_val
real(rp) :: an(0:n_pole_maxx), bn(0:n_pole_maxx)
real(rp) :: an_elec(0:n_pole_maxx), bn_elec(0:n_pole_maxx)
real(rp) :: exact_an(0:n_pole_maxx), exact_bn(0:n_pole_maxx)
type (fringe_field_info_struct) :: fringe_info
logical :: has_misalign, has_mag_multipoles, has_elec_multipoles, apply_rad
integer(C_INT) :: is_exact

real(C_DOUBLE) :: a2_arr(0:n_pole_maxx), b2_arr(0:n_pole_maxx)
real(C_DOUBLE) :: ea2_arr(0:n_pole_maxx), eb2_arr(0:n_pole_maxx)
real(C_DOUBLE) :: cm_arr(0:n_pole_maxx, 0:n_pole_maxx)
real(C_DOUBLE) :: exact_an_arr(0:n_pole_maxx), exact_bn_arr(0:n_pole_maxx)

real(C_DOUBLE), allocatable :: vx(:), vpx(:), vy(:), vpy(:), vz(:), vpz(:)
real(C_DOUBLE), allocatable :: beta_a(:), p0c_a(:), t_a(:)
integer(C_INT), allocatable :: state_a(:)
#endif

did_track = .false.

#ifdef USE_GPU_TRACKING
n = size(bunch%particle)
if (n == 0) return
ele_length = ele%value(l$)
if (ele_length == 0) then
  did_track = .true.
  return
endif

mc2 = mass_of(bunch%particle(1)%species)
delta_ref_time = ele%value(delta_ref_time$)
e_tot_ele = ele%value(e_tot$)
p0c_ele = ele%value(p0c$)

has_misalign = ele%bookkeeping_state%has_misalign

! Compute charge/direction factors
rel_charge_dir = ele%orientation * bunch%particle(1)%direction * &
                 rel_tracking_charge_to_mass(bunch%particle(1), param%particle)
c_dir_val = ele%orientation * bunch%particle(1)%direction * charge_of(bunch%particle(1)%species)

! Determine if exact multipoles are in use
is_exact = 0
ix_exact_mag_max = -1
rho_val = 0
exact_f_scale_val = 0
exact_an_arr = 0
exact_bn_arr = 0

if (nint(ele%value(exact_multipoles$)) /= off$ .and. ele%value(g$) /= 0) then
  is_exact = 1
  ! For exact multipoles: b1 is folded into the multipole arrays
  call multipole_ele_to_ab(ele, .false., ix_mag_max, an, bn, magnetic$, include_kicks$)
  b1 = 0

  ! Prepare exact multipole arrays (convert to vertically_pure if horizontally_pure)
  call multipole_ele_to_ab(ele, .false., ix_exact_mag_max, exact_an, exact_bn, magnetic$, include_kicks$)
  if (nint(ele%value(exact_multipoles$)) == horizontally_pure$ .and. ix_exact_mag_max /= -1) then
    call convert_bend_exact_multipole(ele%value(g$), vertically_pure$, exact_an, exact_bn)
    ix_exact_mag_max = n_pole_maxx
  endif
  exact_an_arr = exact_an
  exact_bn_arr = exact_bn

  rho_val = ele%value(rho$)
  if (ele%value(l$) /= 0) then
    exact_f_scale_val = ele%value(p0c$) / (c_light * charge_of(param%particle) * ele%value(l$))
  endif
else
  ! Standard: extract b1 separately
  call multipole_ele_to_ab(ele, .false., ix_mag_max, an, bn, magnetic$, include_kicks$, b1)
  b1 = b1 * rel_charge_dir
  if (abs(b1) < 1d-10) then
    bn(1) = b1
    b1 = 0
  endif
endif

call multipole_ele_to_ab(ele, .false., ix_elec_max, an_elec, bn_elec, electric$)

! Compute g and g_tot
g = ele%value(g$)
length = bunch%particle(1)%time_dir * ele_length
if (length == 0) then
  dg = 0
else
  dg = bn(0) / ele_length
  bn(0) = 0
endif
g_tot = (g + dg) * rel_charge_dir

! Determine n_step
has_mag_multipoles = (ix_mag_max > -1)
has_elec_multipoles = (ix_elec_max > -1)
n_step = 1
if (has_mag_multipoles .or. has_elec_multipoles) &
  n_step = max(nint(abs(length) / ele%value(ds_step$)), 1)
r_step = real(bunch%particle(1)%time_dir, rp) / n_step
step_len_val = ele_length / n_step

! Fringe info
call init_fringe_info(fringe_info, ele)

! Check if radiation should be applied
apply_rad = gpu_rad_eligible(ele)
if (apply_rad) call ensure_rad_map(ele)

! Entrance: allocate SoA, misalignment, fringe, AoS→SoA
call gpu_tracking_pre(bunch, ele, param, n, vx, vpx, vy, vpy, vz, vpz, &
    state_a, beta_a, p0c_a, t_a, has_misalign, fringe_info, .true.)

! Precompute scaled multipole arrays for CUDA kernel
! Note: the (1+g*x) curvature factor for bends is applied per-particle in the kernel.
call precompute_multipole_arrays(bunch%particle(1), ele, &
    ix_mag_max, an, bn, ix_elec_max, an_elec, bn_elec, &
    ele_length, n_step, a2_arr, b2_arr, ea2_arr, eb2_arr, cm_arr)

did_track = .true.

if (apply_rad .and. associated(ele%rad_map)) then
  ! Split approach: upload once, rad + body + rad, download once
  call gpu_upload_particles(vx, vpx, vy, vpy, vz, vpz, &
                            state_a, beta_a, p0c_a, t_a, n)
  call call_gpu_rad_kick(n, ele%rad_map%rm0)
  call gpu_track_bend_dev(mc2, g, g_tot, dg, b1, &
                          ele_length, delta_ref_time, e_tot_ele, &
                          rel_charge_dir, p0c_ele, n, &
                          a2_arr, b2_arr, cm_arr, &
                          int(ix_mag_max, C_INT), int(n_step, C_INT), &
                          ea2_arr, eb2_arr, int(ix_elec_max, C_INT), &
                          is_exact, exact_an_arr, exact_bn_arr, &
                          int(ix_exact_mag_max, C_INT), &
                          real(rho_val, C_DOUBLE), real(c_dir_val, C_DOUBLE), &
                          real(exact_f_scale_val, C_DOUBLE))
  call call_gpu_rad_kick(n, ele%rad_map%rm1)
  call gpu_download_particles(vx, vpx, vy, vpy, vz, vpz, &
                              state_a, beta_a, p0c_a, t_a, &
                              n, merge(1, 0, ix_elec_max >= 0), 0)
else
  ! Original approach: combined upload + body + download
  call gpu_track_bend(vx, vpx, vy, vpy, vz, vpz, &
                      state_a, beta_a, p0c_a, t_a, &
                      mc2, g, g_tot, dg, b1, &
                      ele_length, delta_ref_time, e_tot_ele, &
                      rel_charge_dir, p0c_ele, n, &
                      a2_arr, b2_arr, cm_arr, &
                      int(ix_mag_max, C_INT), int(n_step, C_INT), &
                      ea2_arr, eb2_arr, int(ix_elec_max, C_INT), &
                      is_exact, exact_an_arr, exact_bn_arr, &
                      int(ix_exact_mag_max, C_INT), &
                      real(rho_val, C_DOUBLE), real(c_dir_val, C_DOUBLE), &
                      real(exact_f_scale_val, C_DOUBLE))
endif

! Exit: SoA→AoS, deallocate, fringe, misalignment, update s
call gpu_tracking_post(bunch, ele, param, n, vx, vpx, vy, vpy, vz, vpz, &
    state_a, beta_a, p0c_a, t_a, has_misalign, fringe_info, .true., &
    .true., .false., .true.)
#endif

end subroutine track_bunch_thru_bend_gpu

!------------------------------------------------------------------------
! track_bunch_thru_lcavity_gpu
!
! GPU batch tracking through an lcavity (linac cavity).
! Handles the stair-step RF approximation with energy kicks,
! ponderomotive transverse kicks (standing wave), and coordinate
! transformations.
!
! CPU sandwich: misalignment. Fringe kicks handled on GPU.
! Falls back to CPU if: multipoles present, solenoid (ks/=0),
!   zero length, zero rf_frequency, absolute time tracking,
!   coupler kicks (coupler_strength /= 0).
!------------------------------------------------------------------------
subroutine track_bunch_thru_lcavity_gpu (bunch, ele, param, did_track)

use, intrinsic :: iso_c_binding

type (bunch_struct),     intent(inout) :: bunch
type (ele_struct), target, intent(inout) :: ele
type (lat_param_struct), intent(in)    :: param
logical,                 intent(out)   :: did_track

#ifdef USE_GPU_TRACKING
integer(C_INT) :: n
integer :: j, nn, ix_mag_max, ix_elec_max, n_steps
real(rp) :: mc2, phi0_total, phi0_no_multi, ref_time_start_val
integer :: abs_time_flag
real(rp) :: an(0:n_pole_maxx), bn(0:n_pole_maxx)
real(rp) :: an_elec(0:n_pole_maxx), bn_elec(0:n_pole_maxx)
type (fringe_field_info_struct) :: fringe_info
logical :: has_misalign, apply_rad
integer :: i_fringe_at
real(rp) :: charge_ratio_val
type (ele_struct), pointer :: lord
type (rf_stair_step_struct), pointer :: step

real(C_DOUBLE), allocatable :: vx(:), vpx(:), vy(:), vpy(:), vz(:), vpz(:)
real(C_DOUBLE), allocatable :: beta_a(:), p0c_a(:), t_a(:)
integer(C_INT), allocatable :: state_a(:)
real(C_DOUBLE), allocatable :: h_step_s0(:), h_step_s(:)
real(C_DOUBLE), allocatable :: h_step_p0c(:), h_step_p1c(:)
real(C_DOUBLE), allocatable :: h_step_scale(:), h_step_time(:)
#endif

did_track = .false.

#ifdef USE_GPU_TRACKING
n = size(bunch%particle)
if (n == 0) return

! --- Bail out conditions ---

! Zero length → track on CPU
if (ele%value(l$) == 0) return

! Zero rf_frequency with non-zero voltage → CPU
if (ele%value(rf_frequency$) == 0) return

! Get the super lord for RF step data
lord => pointer_to_super_lord(ele)

! Solenoid → CPU (step_drift uses solenoid_track_and_mat)
if (lord%value(ks$) /= 0) return

! Multipoles → CPU (scale varies per step, complex interaction)
call multipole_ele_to_ab(ele, .false., ix_mag_max, an, bn, magnetic$, include_kicks$)
call multipole_ele_to_ab(ele, .false., ix_elec_max, an_elec, bn_elec, electric$)
if (ix_mag_max > -1) return
if (ix_elec_max > -1) return

! Check for coupler kicks → CPU
if (ele%value(coupler_strength$) /= 0) return

! Compute fringe parameters (fringe is handled on GPU)
if (nint(lord%value(fringe_type$)) == none$) then
  i_fringe_at = 0
else
  i_fringe_at = nint(lord%value(fringe_at$))
  if (i_fringe_at < 1 .or. i_fringe_at > 3) i_fringe_at = 0
endif
charge_ratio_val = charge_of(bunch%particle(1)%species) / (2.0_rp * charge_of(lord%ref_species))

mc2 = mass_of(bunch%particle(1)%species)
n_steps = nint(lord%value(n_rf_steps$))
has_misalign = ele%bookkeeping_state%has_misalign

! Phase offsets
phi0_no_multi = lord%value(phi0$) + lord%value(phi0_err$)
phi0_total = phi0_no_multi + lord%value(phi0_multipass$)
abs_time_flag = 0
ref_time_start_val = 0.0_rp
if (bmad_com%absolute_time_tracking) then
  abs_time_flag = 1
  if (bmad_com%absolute_time_ref_shift) then
    ! For multipass slaves, use the first-pass element's ref_time_start
    ! (matches this_rf_phase in track_a_lcavity.f90)
    if (lord%slave_status == multipass_slave$) then
      block
        type (ele_pointer_struct), allocatable :: mp_chain(:)
        integer :: ix_pass_mp, n_links_mp
        call multipass_chain(lord, ix_pass_mp, n_links_mp, mp_chain)
        ref_time_start_val = mp_chain(1)%ele%value(ref_time_start$)
      end block
    else
      ref_time_start_val = lord%value(ref_time_start$)
    endif
  endif
endif

! Extract step data from the lord's RF step array (indices 0..n_steps+1).
! For step_time, use the MULTIPASS LORD's values when the lord is a
! multipass slave (matches this_rf_phase which uses mlord%rf%steps%time).
! Other step fields (s0, s, p0c, p1c, scale) use the actual element's
! values since they encode the correct reference energy for this pass.
block
  type (ele_struct), pointer :: mlord
  mlord => lord
  if (lord%slave_status == multipass_slave$) mlord => pointer_to_lord(lord, 1)

  allocate(h_step_s0(n_steps+2), h_step_s(n_steps+2))
  allocate(h_step_p0c(n_steps+2), h_step_p1c(n_steps+2))
  allocate(h_step_scale(n_steps+2), h_step_time(n_steps+2))

  do j = 0, n_steps + 1
    step => lord%rf%steps(j)
    h_step_s0(j+1)    = step%s0
    h_step_s(j+1)     = step%s
    h_step_p0c(j+1)   = step%p0c
    h_step_p1c(j+1)   = step%p1c
    h_step_scale(j+1) = step%scale
    h_step_time(j+1)  = mlord%rf%steps(j)%time  ! Use multipass lord's step time
  enddo
end block

! Check if radiation should be applied
apply_rad = gpu_rad_eligible(ele)
if (apply_rad) call ensure_rad_map(ele)

! Entrance: allocate SoA, misalignment, AoS→SoA (no CPU fringe for lcavity)
call gpu_tracking_pre(bunch, ele, param, n, vx, vpx, vy, vpy, vz, vpz, &
    state_a, beta_a, p0c_a, t_a, has_misalign, fringe_info, .false.)

did_track = .true.

if (apply_rad .and. associated(ele%rad_map)) then
  ! Split approach: upload once, rad + body + rad, download once
  call gpu_upload_particles(vx, vpx, vy, vpy, vz, vpz, &
                            state_a, beta_a, p0c_a, t_a, n)
  call call_gpu_rad_kick(n, ele%rad_map%rm0)
  call gpu_track_lcavity_dev(mc2, &
                             h_step_s0, h_step_s, h_step_p0c, h_step_p1c, &
                             h_step_scale, h_step_time, &
                             int(n_steps, C_INT), &
                             lord%value(voltage$), lord%value(voltage_err$), &
                             lord%value(field_autoscale$), &
                             ele%value(rf_frequency$), phi0_total, &
                             lord%value(voltage_tot$), lord%value(l_active$), &
                             int(nint(lord%value(cavity_type$)), C_INT), &
                             int(i_fringe_at, C_INT), charge_ratio_val, &
                             int(n, C_INT), &
                             int(abs_time_flag, C_INT), phi0_no_multi, &
                             ref_time_start_val)
  call call_gpu_rad_kick(n, ele%rad_map%rm1)
  call gpu_download_particles(vx, vpx, vy, vpy, vz, vpz, &
                              state_a, beta_a, p0c_a, t_a, &
                              n, 1, 1)
else
  ! Original approach: combined upload + body + download
  call gpu_track_lcavity(vx, vpx, vy, vpy, vz, vpz, &
                         state_a, beta_a, p0c_a, t_a, &
                         mc2, &
                         h_step_s0, h_step_s, h_step_p0c, h_step_p1c, &
                         h_step_scale, h_step_time, &
                         int(n_steps, C_INT), &
                         lord%value(voltage$), lord%value(voltage_err$), &
                         lord%value(field_autoscale$), &
                         ele%value(rf_frequency$), phi0_total, &
                         lord%value(voltage_tot$), lord%value(l_active$), &
                         int(nint(lord%value(cavity_type$)), C_INT), &
                         int(i_fringe_at, C_INT), charge_ratio_val, &
                         int(n, C_INT), &
                         int(abs_time_flag, C_INT), phi0_no_multi, &
                         ref_time_start_val)
endif

! Exit: SoA→AoS, deallocate, misalignment, update s (no CPU fringe)
call gpu_tracking_post(bunch, ele, param, n, vx, vpx, vy, vpy, vz, vpz, &
    state_a, beta_a, p0c_a, t_a, has_misalign, fringe_info, .false., &
    .true., .true., .true.)

deallocate(h_step_s0, h_step_s, h_step_p0c, h_step_p1c, h_step_scale, h_step_time)
#endif

end subroutine track_bunch_thru_lcavity_gpu

!------------------------------------------------------------------------
! track_bunch_thru_solenoid_gpu
!
! GPU batch tracking through a solenoid element.
! Body tracking uses the solenoid kernel (4x4 rotation matrix).
! Misalignment and fringe are handled on CPU (fringe with
! apply_sol_fringe=.false. since the solenoid effect is in the body).
!------------------------------------------------------------------------
subroutine track_bunch_thru_solenoid_gpu (bunch, ele, param, did_track)

use multipole_mod, only: ab_multipole_kicks
use, intrinsic :: iso_c_binding

type (bunch_struct),     intent(inout) :: bunch
type (ele_struct), target, intent(inout) :: ele
type (lat_param_struct), intent(in)    :: param
logical,                 intent(out)   :: did_track

#ifdef USE_GPU_TRACKING
integer, parameter :: n_multi = n_pole_maxx + 1
integer(C_INT) :: n
integer :: ix_mag_max, ix_elec_max, n_step, j
real(rp) :: ele_length, mc2, b1, delta_ref_time, e_tot_ele
real(rp) :: rel_tracking_charge, length, ks0, r_step, step_len
real(rp) :: an(0:n_pole_maxx), bn(0:n_pole_maxx)
real(rp) :: an_elec(0:n_pole_maxx), bn_elec(0:n_pole_maxx)
type (fringe_field_info_struct) :: fringe_info
logical :: has_misalign, apply_rad

real(C_DOUBLE) :: a2_arr(0:n_pole_maxx), b2_arr(0:n_pole_maxx)
real(C_DOUBLE) :: ea2_arr(0:n_pole_maxx), eb2_arr(0:n_pole_maxx)
real(C_DOUBLE) :: cm_arr(0:n_pole_maxx, 0:n_pole_maxx)

real(C_DOUBLE), allocatable :: vx(:), vpx(:), vy(:), vpy(:), vz(:), vpz(:)
real(C_DOUBLE), allocatable :: beta_a(:), p0c_a(:), t_a(:)
integer(C_INT), allocatable :: state_a(:)
#endif

did_track = .false.

#ifdef USE_GPU_TRACKING
n = size(bunch%particle)
if (n == 0) return
ele_length = ele%value(l$)
if (ele_length == 0) then
  did_track = .true.
  return
endif

mc2 = mass_of(bunch%particle(1)%species)
delta_ref_time = ele%value(delta_ref_time$)
e_tot_ele = ele%value(e_tot$)

has_misalign = ele%bookkeeping_state%has_misalign
call init_fringe_info(fringe_info, ele)

! Compute ks0 = rel_tracking_charge * bs_field * charge * c_light / p0c
! This matches what solenoid_track_and_mat does.
rel_tracking_charge = rel_tracking_charge_to_mass(bunch%particle(1), param%particle)
ks0 = rel_tracking_charge * ele%value(bs_field$) * charge_of(bunch%particle(1)%species) * c_light / bunch%particle(1)%p0c

! Get multipoles
call multipole_ele_to_ab(ele, .false., ix_mag_max, an, bn, magnetic$, include_kicks$, b1)
call multipole_ele_to_ab(ele, .false., ix_elec_max, an_elec, bn_elec, electric$)

length = bunch%particle(1)%time_dir * ele_length
n_step = 1
if (ix_mag_max > -1 .or. ix_elec_max > -1) &
  n_step = max(nint(abs(length) / ele%value(ds_step$)), 1)

apply_rad = gpu_rad_eligible(ele)
if (apply_rad) call ensure_rad_map(ele)

! Entrance: misalignment + fringe on CPU (with apply_sol_fringe = .false.)
allocate(vx(n), vpx(n), vy(n), vpy(n), vz(n), vpz(n))
allocate(state_a(n), beta_a(n), p0c_a(n), t_a(n))

if (has_misalign) call apply_misalign_to_bunch(bunch, ele, n, set$)
if (fringe_info%has_fringe) then
  call apply_sol_fringe_to_bunch(bunch, ele, param, n, fringe_info, first_track_edge$)
endif

call bunch_to_soa(bunch, n, vx, vpx, vy, vpy, vz, vpz, state_a, beta_a, p0c_a, t_a)

! Precompute multipole arrays
call precompute_multipole_arrays(bunch%particle(1), ele, &
    ix_mag_max, an, bn, ix_elec_max, an_elec, bn_elec, &
    ele_length, n_step, a2_arr, b2_arr, ea2_arr, eb2_arr, cm_arr)

did_track = .true.

if (apply_rad .and. associated(ele%rad_map)) then
  call gpu_upload_particles(vx, vpx, vy, vpy, vz, vpz, &
                            state_a, beta_a, p0c_a, t_a, n)
  call call_gpu_rad_kick(n, ele%rad_map%rm0)
  call gpu_track_solenoid_dev(mc2, ks0, ele_length, delta_ref_time, &
                              e_tot_ele, n, int(n_step, C_INT), &
                              a2_arr, b2_arr, cm_arr, &
                              int(ix_mag_max, C_INT), &
                              ea2_arr, eb2_arr, int(ix_elec_max, C_INT))
  call call_gpu_rad_kick(n, ele%rad_map%rm1)
  call gpu_download_particles(vx, vpx, vy, vpy, vz, vpz, &
                              state_a, beta_a, p0c_a, t_a, &
                              n, merge(1, 0, ix_elec_max >= 0), 0)
else
  call gpu_track_solenoid(vx, vpx, vy, vpy, vz, vpz, &
                      state_a, beta_a, p0c_a, t_a, &
                      mc2, ks0, ele_length, delta_ref_time, &
                      e_tot_ele, n, int(n_step, C_INT), &
                      a2_arr, b2_arr, cm_arr, &
                      int(ix_mag_max, C_INT), &
                      ea2_arr, eb2_arr, int(ix_elec_max, C_INT))
endif

! Exit: SoA→AoS, fringe, misalignment
call soa_to_bunch(bunch, ele, n, vx, vpx, vy, vpy, vz, vpz, state_a, beta_a, p0c_a, t_a, &
                   .false., .false.)
deallocate(vx, vpx, vy, vpy, vz, vpz)
deallocate(state_a, beta_a, p0c_a, t_a)

if (fringe_info%has_fringe) then
  call apply_sol_fringe_to_bunch(bunch, ele, param, n, fringe_info, second_track_edge$)
endif
if (has_misalign) call apply_misalign_to_bunch(bunch, ele, n, unset$)

do j = 1, n
  if (bunch%particle(j)%state == alive$) then
    bunch%particle(j)%s = ele%s
  endif
enddo
#endif

end subroutine track_bunch_thru_solenoid_gpu

!------------------------------------------------------------------------
! track_bunch_thru_sol_quad_gpu
!
! GPU batch tracking through a sol_quad element (combined solenoid +
! quadrupole).  When b1==0 uses the solenoid kernel, otherwise uses
! the sol_quad kernel.
!------------------------------------------------------------------------
subroutine track_bunch_thru_sol_quad_gpu (bunch, ele, param, did_track)

use multipole_mod, only: ab_multipole_kicks
use, intrinsic :: iso_c_binding

type (bunch_struct),     intent(inout) :: bunch
type (ele_struct), target, intent(inout) :: ele
type (lat_param_struct), intent(in)    :: param
logical,                 intent(out)   :: did_track

#ifdef USE_GPU_TRACKING
integer, parameter :: n_multi = n_pole_maxx + 1
integer(C_INT) :: n
integer :: ix_mag_max, ix_elec_max, n_step, j
real(rp) :: ele_length, mc2, b1, delta_ref_time, e_tot_ele
real(rp) :: rel_tracking_charge, charge_dir, length
real(rp) :: ks0, ks_val, k1_val, r_step, step_len
real(rp) :: an(0:n_pole_maxx), bn(0:n_pole_maxx)
real(rp) :: an_elec(0:n_pole_maxx), bn_elec(0:n_pole_maxx)
type (fringe_field_info_struct) :: fringe_info
logical :: has_misalign, apply_rad

real(C_DOUBLE) :: a2_arr(0:n_pole_maxx), b2_arr(0:n_pole_maxx)
real(C_DOUBLE) :: ea2_arr(0:n_pole_maxx), eb2_arr(0:n_pole_maxx)
real(C_DOUBLE) :: cm_arr(0:n_pole_maxx, 0:n_pole_maxx)

real(C_DOUBLE), allocatable :: vx(:), vpx(:), vy(:), vpy(:), vz(:), vpz(:)
real(C_DOUBLE), allocatable :: beta_a(:), p0c_a(:), t_a(:)
integer(C_INT), allocatable :: state_a(:)
#endif

did_track = .false.

#ifdef USE_GPU_TRACKING
n = size(bunch%particle)
if (n == 0) return
ele_length = ele%value(l$)
if (ele_length == 0) then
  did_track = .true.
  return
endif

mc2 = mass_of(bunch%particle(1)%species)
delta_ref_time = ele%value(delta_ref_time$)
e_tot_ele = ele%value(e_tot$)

has_misalign = ele%bookkeeping_state%has_misalign
call init_fringe_info(fringe_info, ele)

rel_tracking_charge = rel_tracking_charge_to_mass(bunch%particle(1), param%particle)
charge_dir = rel_tracking_charge * ele%orientation * bunch%particle(1)%direction

! Get multipoles
call multipole_ele_to_ab(ele, .false., ix_mag_max, an, bn, magnetic$, include_kicks$, b1)
call multipole_ele_to_ab(ele, .false., ix_elec_max, an_elec, bn_elec, electric$)

length = bunch%particle(1)%time_dir * ele_length
n_step = 1
if (ix_mag_max > -1 .or. ix_elec_max > -1) &
  n_step = max(nint(abs(length) / ele%value(ds_step$)), 1)

apply_rad = gpu_rad_eligible(ele)
if (apply_rad) call ensure_rad_map(ele)

! Entrance: misalignment + fringe on CPU (with apply_sol_fringe = .false.)
allocate(vx(n), vpx(n), vy(n), vpy(n), vz(n), vpz(n))
allocate(state_a(n), beta_a(n), p0c_a(n), t_a(n))

if (has_misalign) call apply_misalign_to_bunch(bunch, ele, n, set$)
if (fringe_info%has_fringe) then
  call apply_sol_fringe_to_bunch(bunch, ele, param, n, fringe_info, first_track_edge$)
endif

call bunch_to_soa(bunch, n, vx, vpx, vy, vpy, vz, vpz, state_a, beta_a, p0c_a, t_a)

! Precompute multipole arrays
call precompute_multipole_arrays(bunch%particle(1), ele, &
    ix_mag_max, an, bn, ix_elec_max, an_elec, bn_elec, &
    ele_length, n_step, a2_arr, b2_arr, ea2_arr, eb2_arr, cm_arr)

did_track = .true.

! Choose kernel: solenoid (b1==0) or sol_quad (b1 /= 0)
if (b1 == 0) then
  ! Pure solenoid tracking (solenoid$ or sol_quad with no quad gradient)
  ks0 = rel_tracking_charge * ele%value(bs_field$) * charge_of(bunch%particle(1)%species) * c_light / bunch%particle(1)%p0c

  if (apply_rad .and. associated(ele%rad_map)) then
    call gpu_upload_particles(vx, vpx, vy, vpy, vz, vpz, &
                              state_a, beta_a, p0c_a, t_a, n)
    call call_gpu_rad_kick(n, ele%rad_map%rm0)
    call gpu_track_solenoid_dev(mc2, ks0, ele_length, delta_ref_time, &
                                e_tot_ele, n, int(n_step, C_INT), &
                                a2_arr, b2_arr, cm_arr, &
                                int(ix_mag_max, C_INT), &
                                ea2_arr, eb2_arr, int(ix_elec_max, C_INT))
    call call_gpu_rad_kick(n, ele%rad_map%rm1)
    call gpu_download_particles(vx, vpx, vy, vpy, vz, vpz, &
                                state_a, beta_a, p0c_a, t_a, &
                                n, merge(1, 0, ix_elec_max >= 0), 0)
  else
    call gpu_track_solenoid(vx, vpx, vy, vpy, vz, vpz, &
                        state_a, beta_a, p0c_a, t_a, &
                        mc2, ks0, ele_length, delta_ref_time, &
                        e_tot_ele, n, int(n_step, C_INT), &
                        a2_arr, b2_arr, cm_arr, &
                        int(ix_mag_max, C_INT), &
                        ea2_arr, eb2_arr, int(ix_elec_max, C_INT))
  endif
else
  ! Sol_quad kernel with combined solenoid + quadrupole
  ks_val = rel_tracking_charge * ele%value(ks$)
  k1_val = charge_dir * b1 / ele_length

  if (apply_rad .and. associated(ele%rad_map)) then
    call gpu_upload_particles(vx, vpx, vy, vpy, vz, vpz, &
                              state_a, beta_a, p0c_a, t_a, n)
    call call_gpu_rad_kick(n, ele%rad_map%rm0)
    call gpu_track_sol_quad_dev(mc2, ks_val, k1_val, ele_length, delta_ref_time, &
                                e_tot_ele, n, int(n_step, C_INT), &
                                a2_arr, b2_arr, cm_arr, &
                                int(ix_mag_max, C_INT), &
                                ea2_arr, eb2_arr, int(ix_elec_max, C_INT))
    call call_gpu_rad_kick(n, ele%rad_map%rm1)
    call gpu_download_particles(vx, vpx, vy, vpy, vz, vpz, &
                                state_a, beta_a, p0c_a, t_a, &
                                n, merge(1, 0, ix_elec_max >= 0), 0)
  else
    call gpu_track_sol_quad(vx, vpx, vy, vpy, vz, vpz, &
                        state_a, beta_a, p0c_a, t_a, &
                        mc2, ks_val, k1_val, ele_length, delta_ref_time, &
                        e_tot_ele, n, int(n_step, C_INT), &
                        a2_arr, b2_arr, cm_arr, &
                        int(ix_mag_max, C_INT), &
                        ea2_arr, eb2_arr, int(ix_elec_max, C_INT))
  endif
endif

! Exit: SoA→AoS, fringe, misalignment
call soa_to_bunch(bunch, ele, n, vx, vpx, vy, vpy, vz, vpz, state_a, beta_a, p0c_a, t_a, &
                   .false., .false.)
deallocate(vx, vpx, vy, vpy, vz, vpz)
deallocate(state_a, beta_a, p0c_a, t_a)

if (fringe_info%has_fringe) then
  call apply_sol_fringe_to_bunch(bunch, ele, param, n, fringe_info, second_track_edge$)
endif
if (has_misalign) call apply_misalign_to_bunch(bunch, ele, n, unset$)

do j = 1, n
  if (bunch%particle(j)%state == alive$) then
    bunch%particle(j)%s = ele%s
  endif
enddo
#endif

end subroutine track_bunch_thru_sol_quad_gpu

!------------------------------------------------------------------------
! track_bunch_thru_wiggler_gpu
!
! GPU batch tracking through a wiggler or undulator element.
! Uses the averaged-field model: quadrupole focusing with octupole
! kicks, matching track_a_wiggler.  Supports both planar and helical
! field models.
!------------------------------------------------------------------------
subroutine track_bunch_thru_wiggler_gpu (bunch, ele, param, did_track)

use multipole_mod, only: ab_multipole_kicks
use, intrinsic :: iso_c_binding

type (bunch_struct),     intent(inout) :: bunch
type (ele_struct), target, intent(inout) :: ele
type (lat_param_struct), intent(in)    :: param
logical,                 intent(out)   :: did_track

#ifdef USE_GPU_TRACKING
integer, parameter :: n_multi = n_pole_maxx + 1
integer(C_INT) :: n
integer :: ix_mag_max, ix_elec_max, n_step, j
real(rp) :: ele_length, mc2, delta_ref_time, e_tot_ele, p0c_ele_val
real(rp) :: rel_tracking_charge, length
real(rp) :: k1x_val, k1y_val, kz_val, ky2_val, factor_val, osc_amp_val
integer(C_INT) :: is_helical_val
real(rp) :: an(0:n_pole_maxx), bn(0:n_pole_maxx)
real(rp) :: an_elec(0:n_pole_maxx), bn_elec(0:n_pole_maxx)
type (ele_struct), pointer :: field_ele
type (fringe_field_info_struct) :: fringe_info
logical :: has_misalign, apply_rad

real(C_DOUBLE) :: a2_arr(0:n_pole_maxx), b2_arr(0:n_pole_maxx)
real(C_DOUBLE) :: ea2_arr(0:n_pole_maxx), eb2_arr(0:n_pole_maxx)
real(C_DOUBLE) :: cm_arr(0:n_pole_maxx, 0:n_pole_maxx)

real(C_DOUBLE), allocatable :: vx(:), vpx(:), vy(:), vpy(:), vz(:), vpz(:)
real(C_DOUBLE), allocatable :: beta_a(:), p0c_a(:), t_a(:)
integer(C_INT), allocatable :: state_a(:)
#endif

did_track = .false.

#ifdef USE_GPU_TRACKING
n = size(bunch%particle)
if (n == 0) return
ele_length = ele%value(l$)
if (ele_length == 0) then
  did_track = .true.
  return
endif

mc2 = mass_of(bunch%particle(1)%species)
delta_ref_time = ele%value(delta_ref_time$)
e_tot_ele = ele%value(e_tot$)
p0c_ele_val = ele%value(p0c$)
osc_amp_val = ele%value(osc_amplitude$)

has_misalign = ele%bookkeeping_state%has_misalign
call init_fringe_info(fringe_info, ele)

! Get field element to determine helical vs planar
field_ele => pointer_to_field_ele(ele, 1)

! Compute kz, ky2
if (ele%value(l_period$) == 0) then
  kz_val = 1d100
  ky2_val = 0
else
  kz_val = twopi / ele%value(l_period$)
  ky2_val = kz_val**2 + ele%value(kx$)**2
endif

! Compute averaged focusing strengths
rel_tracking_charge = rel_tracking_charge_to_mass(bunch%particle(1), param%particle)
factor_val = abs(rel_tracking_charge) * 0.5_rp * (c_light * ele%value(b_max$) / ele%value(p0c$))**2

if (field_ele%field_calc == helical_model$) then
  k1x_val = -factor_val
  k1y_val = -factor_val
  is_helical_val = 1
else
  k1x_val =  factor_val * (ele%value(kx$) / kz_val)**2
  k1y_val = -factor_val * ky2_val / kz_val**2
  is_helical_val = 0
endif

! Multipoles
call multipole_ele_to_ab(ele, .false., ix_mag_max, an, bn, magnetic$, include_kicks$)
call multipole_ele_to_ab(ele, .false., ix_elec_max, an_elec, bn_elec, electric$)

n_step = max(nint(ele%value(l$) / ele%value(ds_step$)), 1)
if (ix_mag_max < 0 .and. ix_elec_max < 0) n_step = 1

apply_rad = gpu_rad_eligible(ele)
if (apply_rad) call ensure_rad_map(ele)

! Entrance: misalignment on CPU
allocate(vx(n), vpx(n), vy(n), vpy(n), vz(n), vpz(n))
allocate(state_a(n), beta_a(n), p0c_a(n), t_a(n))

if (has_misalign) call apply_misalign_to_bunch(bunch, ele, n, set$)

call bunch_to_soa(bunch, n, vx, vpx, vy, vpy, vz, vpz, state_a, beta_a, p0c_a, t_a)

! Precompute multipole arrays
call precompute_multipole_arrays(bunch%particle(1), ele, &
    ix_mag_max, an, bn, ix_elec_max, an_elec, bn_elec, &
    ele_length, n_step, a2_arr, b2_arr, ea2_arr, eb2_arr, cm_arr)

did_track = .true.

if (apply_rad .and. associated(ele%rad_map)) then
  call gpu_upload_particles(vx, vpx, vy, vpy, vz, vpz, &
                            state_a, beta_a, p0c_a, t_a, n)
  call call_gpu_rad_kick(n, ele%rad_map%rm0)
  call gpu_track_wiggler_dev(mc2, ele_length, delta_ref_time, &
                              e_tot_ele, p0c_ele_val, &
                              k1x_val, k1y_val, kz_val, is_helical_val, &
                              osc_amp_val, &
                              n, int(n_step, C_INT), &
                              a2_arr, b2_arr, cm_arr, &
                              int(ix_mag_max, C_INT), &
                              ea2_arr, eb2_arr, int(ix_elec_max, C_INT))
  call call_gpu_rad_kick(n, ele%rad_map%rm1)
  call gpu_download_particles(vx, vpx, vy, vpy, vz, vpz, &
                              state_a, beta_a, p0c_a, t_a, &
                              n, merge(1, 0, ix_elec_max >= 0), 0)
else
  call gpu_track_wiggler(vx, vpx, vy, vpy, vz, vpz, &
                      state_a, beta_a, p0c_a, t_a, &
                      mc2, ele_length, &
                      delta_ref_time, e_tot_ele, p0c_ele_val, &
                      k1x_val, k1y_val, kz_val, is_helical_val, &
                      osc_amp_val, &
                      n, int(n_step, C_INT), &
                      a2_arr, b2_arr, cm_arr, &
                      int(ix_mag_max, C_INT), &
                      ea2_arr, eb2_arr, int(ix_elec_max, C_INT))
endif

! Exit: SoA->AoS, misalignment
call soa_to_bunch(bunch, ele, n, vx, vpx, vy, vpy, vz, vpz, state_a, beta_a, p0c_a, t_a, &
                   .false., .false.)
deallocate(vx, vpx, vy, vpy, vz, vpz)
deallocate(state_a, beta_a, p0c_a, t_a)

if (has_misalign) call apply_misalign_to_bunch(bunch, ele, n, unset$)

do j = 1, n
  if (bunch%particle(j)%state == alive$) then
    bunch%particle(j)%s = ele%s
  endif
enddo
#endif

end subroutine track_bunch_thru_wiggler_gpu

!------------------------------------------------------------------------
! track_bunch_thru_pipe_gpu
!
! GPU batch tracking through a pipe element.  A pipe is tracked as a
! thick multipole: split-step drift + multipole kicks.
!
! When there are no multipoles (the common case), the body is a pure
! drift and we use the drift kernel for bit-exact agreement with CPU.
! When multipoles are present, we use the quad kernel with b1=0.
! In both cases, misalignment and fringe are handled on CPU.
!------------------------------------------------------------------------
subroutine track_bunch_thru_pipe_gpu (bunch, ele, param, did_track)

use multipole_mod, only: ab_multipole_kicks
use, intrinsic :: iso_c_binding

type (bunch_struct),     intent(inout) :: bunch
type (ele_struct),       intent(in)    :: ele
type (lat_param_struct), intent(in)    :: param
logical,                 intent(out)   :: did_track

#ifdef USE_GPU_TRACKING
integer, parameter :: n_multi = n_pole_maxx + 1  ! = 22
integer(C_INT) :: n
integer :: j, ix_mag_max, ix_elec_max, n_step
real(rp) :: ele_length, mc2, delta_ref_time, e_tot_ele
real(rp) :: charge_dir, rel_tracking_charge, length
real(rp) :: an(0:n_pole_maxx), bn(0:n_pole_maxx)
real(rp) :: an_elec(0:n_pole_maxx), bn_elec(0:n_pole_maxx)
type (fringe_field_info_struct) :: fringe_info
logical :: has_misalign, has_multipoles

! Precomputed scaled multipole arrays and c_multi coefficients for CUDA
real(C_DOUBLE) :: a2_arr(0:n_pole_maxx), b2_arr(0:n_pole_maxx)
real(C_DOUBLE) :: ea2_arr(0:n_pole_maxx), eb2_arr(0:n_pole_maxx)
real(C_DOUBLE) :: cm_arr(0:n_pole_maxx, 0:n_pole_maxx)

real(C_DOUBLE), allocatable :: vx(:), vpx(:), vy(:), vpy(:), vz(:), vpz(:)
real(C_DOUBLE), allocatable :: beta_a(:), p0c_a(:), s_a(:), t_a(:)
integer(C_INT), allocatable :: state_a(:)
#endif

did_track = .false.

#ifdef USE_GPU_TRACKING
n = size(bunch%particle)
if (n == 0) return
ele_length = ele%value(l$)
if (ele_length == 0) then
  did_track = .true.
  return
endif

mc2 = mass_of(bunch%particle(1)%species)
delta_ref_time = ele%value(delta_ref_time$)
e_tot_ele = ele%value(e_tot$)

has_misalign = ele%bookkeeping_state%has_misalign

! Fringe fields are handled on CPU (before/after GPU body tracking)
call init_fringe_info(fringe_info, ele)

! Get multipoles (no quad gradient for pipe)
call multipole_ele_to_ab(ele, .false., ix_mag_max, an, bn, magnetic$, include_kicks$)
call multipole_ele_to_ab(ele, .false., ix_elec_max, an_elec, bn_elec, electric$)

has_multipoles = (ix_mag_max > -1) .or. (ix_elec_max > -1)

if (has_multipoles) then
  ! --- Multipole path: use quad kernel with b1=0 ---

  ! Determine n_step for split-step integration
  length = bunch%particle(1)%time_dir * ele_length
  n_step = max(nint(abs(length) / ele%value(ds_step$)), 1)

  ! Compute charge_dir: rel_charge * orientation * direction * time_dir
  rel_tracking_charge = rel_tracking_charge_to_mass(bunch%particle(1), param%particle)
  charge_dir = rel_tracking_charge * ele%orientation * bunch%particle(1)%direction * bunch%particle(1)%time_dir

  ! Entrance: allocate SoA, misalignment, fringe, AoS→SoA
  call gpu_tracking_pre(bunch, ele, param, n, vx, vpx, vy, vpy, vz, vpz, &
      state_a, beta_a, p0c_a, t_a, has_misalign, fringe_info, .true.)

  ! Precompute scaled multipole arrays for CUDA kernel
  call precompute_multipole_arrays(bunch%particle(1), ele, &
      ix_mag_max, an, bn, ix_elec_max, an_elec, bn_elec, &
      ele_length, n_step, a2_arr, b2_arr, ea2_arr, eb2_arr, cm_arr)

  did_track = .true.

  ! Call CUDA quad kernel with b1=0 (pipe has no quad gradient)
  call gpu_track_quad(vx, vpx, vy, vpy, vz, vpz, &
                      state_a, beta_a, p0c_a, t_a, &
                      mc2, 0.0_C_DOUBLE, ele_length, delta_ref_time, &
                      e_tot_ele, charge_dir, n, &
                      a2_arr, b2_arr, cm_arr, &
                      int(ix_mag_max, C_INT), int(n_step, C_INT), &
                      ea2_arr, eb2_arr, int(ix_elec_max, C_INT))

  ! Exit: SoA→AoS, deallocate, fringe, misalignment, update s
  call gpu_tracking_post(bunch, ele, param, n, vx, vpx, vy, vpy, vz, vpz, &
      state_a, beta_a, p0c_a, t_a, has_misalign, fringe_info, .true., &
      .true., .false., .true.)

else
  ! --- Pure drift path: use drift kernel for bit-exact CPU agreement ---

  ! Apply entrance misalignment and fringe on CPU
  if (has_misalign) call apply_misalign_to_bunch(bunch, ele, n, set$)
  if (fringe_info%has_fringe) &
    call apply_fringe_to_bunch(bunch, ele, param, n, fringe_info, first_track_edge$)

  ! Allocate SoA arrays (drift kernel needs s_pos)
  allocate(vx(n), vpx(n), vy(n), vpy(n), vz(n), vpz(n))
  allocate(state_a(n), beta_a(n), p0c_a(n), s_a(n), t_a(n))

  ! AoS -> SoA extraction
  call bunch_to_soa(bunch, n, vx, vpx, vy, vpy, vz, vpz, state_a, beta_a, p0c_a, t_a)
  do j = 1, n
    s_a(j) = bunch%particle(j)%s
  enddo

  did_track = .true.

  ! Call CUDA drift kernel
  call gpu_track_drift(vx, vpx, vy, vpy, vz, vpz, &
                       state_a, beta_a, p0c_a, s_a, t_a, &
                       mc2, ele_length, n)

  ! SoA -> AoS write-back
  call soa_to_bunch(bunch, ele, n, vx, vpx, vy, vpy, vz, vpz, state_a, beta_a, p0c_a, t_a, &
                     .false., .false.)
  do j = 1, n
    bunch%particle(j)%s = s_a(j)
  enddo

  deallocate(vx, vpx, vy, vpy, vz, vpz)
  deallocate(state_a, beta_a, p0c_a, s_a, t_a)

  ! Apply exit fringe and misalignment on CPU
  if (fringe_info%has_fringe) &
    call apply_fringe_to_bunch(bunch, ele, param, n, fringe_info, second_track_edge$)
  if (has_misalign) call apply_misalign_to_bunch(bunch, ele, n, unset$)
endif
#endif

end subroutine track_bunch_thru_pipe_gpu

!------------------------------------------------------------------------
! check_entrance_aperture_for_gpu — check entrance aperture for all alive particles
!
! The GPU dispatch path bypasses track1, which normally checks entrance
! apertures. This subroutine replicates that check.
!------------------------------------------------------------------------
subroutine check_entrance_aperture_for_gpu (bunch, ele, param)

type (bunch_struct), intent(inout) :: bunch
type (ele_struct),   intent(inout) :: ele
type (lat_param_struct), intent(inout) :: param

integer :: j

do j = 1, size(bunch%particle)
  if (bunch%particle(j)%state == alive$) then
    call check_aperture_limit(bunch%particle(j), ele, first_track_edge$, param)
  endif
enddo

end subroutine check_entrance_aperture_for_gpu

!------------------------------------------------------------------------
! ele_gpu_can_stay_on_device — check if an element can be tracked
! entirely on the GPU without downloading particle data to CPU.
!
! Returns .true. if:
! - Element is GPU-eligible (drift, quad, sbend, lcavity)
! - Any misalignment is "simple" (x_offset + y_offset + tilt only,
!   no pitches, no z_offset, non-bend)
! - Fringe can be handled: no fringe, or element is lcavity
!   (lcavity fringe already runs on GPU)
! - Aperture is rectangular (or none)
!
! Returns .false. if complex CPU-side operations are needed (complex
! fringe, pitches, bend misalignment, etc.).
!------------------------------------------------------------------------
function ele_gpu_can_stay_on_device(ele, from_csr_loop) result (can_stay)
type (ele_struct), intent(in) :: ele
logical, intent(in), optional :: from_csr_loop
logical :: can_stay
type (fringe_field_info_struct) :: fringe_info
logical :: has_fringe_needs_cpu

can_stay = .false.
if (.not. ele_gpu_eligible(ele)) return

! Elements with CSR or space charge need track1_bunch_csr sub-stepping.
! But when called FROM WITHIN the CSR sub-step loop (from_csr_loop=.true.),
! the sub-stepping is already being handled — body tracking can stay on GPU.
if (bmad_com%csr_and_space_charge_on .and. .not. logic_option(.false., from_csr_loop)) then
  if (ele%csr_method /= off$ .or. ele%space_charge_method /= off$) return
endif

! Pipe, monitor, instrument, kicker elements can have multipoles that require
! the pipe GPU path (quad kernel with b1=0). Without multipoles, they're
! simple drifts and can stay on device. Markers are always zero-length no-ops.
if (ele%key == pipe$ .or. ele%key == monitor$ .or. ele%key == instrument$ .or. &
    ele%key == kicker$ .or. ele%key == hkicker$ .or. ele%key == vkicker$) then
  if (allocated(ele%multipole_cache)) then
    if (ele%multipole_cache%ix_pole_mag_max > -1 .or. &
        ele%multipole_cache%ix_kick_mag_max > -1 .or. &
        ele%multipole_cache%ix_pole_elec_max > -1 .or. &
        ele%multipole_cache%ix_kick_elec_max > -1) return
  endif
  ! No multipoles: can stay on device as drift
endif

! All misalignment types handled on GPU:
! - x/y offset + tilt: gpu_misalign (2D rotation)
! - pitches + z_offset: gpu_misalign_3d (full 3D rotation matrix)
! - bends with curvature: gpu_bend_offset (curvature-aware transforms)

! Check for fringe that requires CPU
has_fringe_needs_cpu = .false.
select case (ele%key)
case (drift$, pipe$, monitor$, instrument$, kicker$, hkicker$, vkicker$, marker$)
  ! These elements have no fringe
case (octupole$, thick_multipole$, elseparator$)
  ! These elements use hard_multipole_edge_kick for fringe — not yet on GPU for these types.
  ! Conservative: fall back to CPU if fringe is enabled.
  call init_fringe_info(fringe_info, ele)
  if (fringe_info%has_fringe) has_fringe_needs_cpu = .true.
case (lcavity$)
  ! Lcavity fringe is already handled on GPU
case (quadrupole$)
  ! Quad fringe handled on GPU, but electric pole fringe needs CPU
  if (associated(ele%a_pole_elec)) has_fringe_needs_cpu = .true.
case (sextupole$)
  ! Sextupole fringe: full$ and hard_edge_only$ handled on GPU via hard_multipole_edge
  call init_fringe_info(fringe_info, ele)
  if (fringe_info%has_fringe) then
    select case (nint(ele%value(fringe_type$)))
    case (full$, hard_edge_only$)
      ! Handled on GPU
    case default
      has_fringe_needs_cpu = .true.
    end select
  endif
case (sbend$, rf_bend$)
  ! Bend fringe: basic_bend$, hard_edge_only$, full$, sad_full$, soft_edge_only$ handled on GPU.
  ! Other fringe types (linear_edge$) need CPU.
  call init_fringe_info(fringe_info, ele)
  if (fringe_info%has_fringe) then
    select case (nint(ele%value(fringe_type$)))
    case (basic_bend$, hard_edge_only$)
      ! Handled by gpu_bend_fringe kernel (Hwang)
    case (full$)
      ! Handled by gpu_exact_bend_fringe kernel (PTC-style)
    case (sad_full$, soft_edge_only$)
      ! Handled by gpu_sad_bend_fringe kernel + gpu_bend_fringe for sad_full$
    case default
      has_fringe_needs_cpu = .true.
    end select
  endif
case (solenoid$, sol_quad$)
  ! Solenoid/sol_quad fringe requires apply_sol_fringe=.false. which needs CPU dispatch.
  ! For persistent on-device path, these always fall through to per-element tracking.
  call init_fringe_info(fringe_info, ele)
  if (fringe_info%has_fringe) has_fringe_needs_cpu = .true.
case (wiggler$, undulator$)
  ! Wiggler/undulator: init_fringe_info always sets has_fringe=.true., but the actual
  ! fringe is handled implicitly by the body tracking (offset_particle in CPU).
  ! For persistent mode, conservatively require CPU dispatch.
  call init_fringe_info(fringe_info, ele)
  if (fringe_info%has_fringe) has_fringe_needs_cpu = .true.
end select
if (has_fringe_needs_cpu) return

! Rectangular and elliptical apertures can be checked on GPU.
! Wall3d and offset_moves_aperture need full CPU check_aperture_limit.
if (ele%value(x1_limit$) /= 0 .or. ele%value(x2_limit$) /= 0 .or. &
    ele%value(y1_limit$) /= 0 .or. ele%value(y2_limit$) /= 0) then
  if (ele%aperture_type /= rectangular$ .and. ele%aperture_type /= elliptical$) return
  if (ele%offset_moves_aperture) return
endif

can_stay = .true.
end function ele_gpu_can_stay_on_device

!------------------------------------------------------------------------
! track_bunch_thru_elements_gpu — track a bunch through multiple
! consecutive elements, keeping particle data on the GPU device between
! elements that support it.
!
! Elements are processed in order from ix_start to ix_end.
! For "device-resident" elements (ele_gpu_can_stay_on_device), the
! particle data stays on the GPU between elements, avoiding redundant
! host-device transfers. For elements that need CPU-side operations
! (complex fringe, pitches, bend misalignment), data is downloaded
! to the CPU, the operations are done, and data is re-uploaded.
!
! did_track_to: index of last element successfully tracked.
!------------------------------------------------------------------------
subroutine track_bunch_thru_elements_gpu(bunch, branch, ix_start, ix_end, did_track_to)

use multipole_mod, only: ab_multipole_kicks
use radiation_mod, only: radiation_map_setup
use, intrinsic :: iso_c_binding

type (bunch_struct),     intent(inout) :: bunch
type (branch_struct), target, intent(inout) :: branch
integer,                 intent(in)    :: ix_start, ix_end
integer,                 intent(out)   :: did_track_to

#ifdef USE_GPU_TRACKING
integer, parameter :: n_multi = n_pole_maxx + 1
integer(C_INT) :: n
integer :: ie, j
type (ele_struct), pointer :: ele
logical :: on_device, did_track, apply_rad
logical :: has_misalign, can_stay
real(C_DOUBLE) :: misalign_W(3,3), misalign_Lx, misalign_Ly, misalign_Lz

! Persistent host SoA buffers (reused across elements)
real(C_DOUBLE), allocatable, save :: vx(:), vpx(:), vy(:), vpy(:), vz(:), vpz(:)
real(C_DOUBLE), allocatable, save :: beta_a(:), p0c_a(:), t_a(:), s_a(:)
integer(C_INT), allocatable, save :: state_a(:)
integer, save :: soa_cap = 0

! Per-element scratch
integer :: ix_mag_max, ix_elec_max, n_step
real(rp) :: ele_length, mc2, b1, delta_ref_time, e_tot_ele, p0c_ele
real(rp) :: charge_dir, rel_tracking_charge, length
real(rp) :: g, g_tot, dg, rel_charge_dir, r_step, step_len_val
real(rp) :: an(0:n_pole_maxx), bn(0:n_pole_maxx)
real(rp) :: an_elec(0:n_pole_maxx), bn_elec(0:n_pole_maxx)
real(C_DOUBLE) :: a2_arr(0:n_pole_maxx), b2_arr(0:n_pole_maxx)
real(C_DOUBLE) :: ea2_arr(0:n_pole_maxx), eb2_arr(0:n_pole_maxx)
real(C_DOUBLE) :: cm_arr(0:n_pole_maxx, 0:n_pole_maxx)
type (fringe_field_info_struct) :: fringe_info
logical :: has_mag_multipoles, has_elec_multipoles
! lcavity scratch
type (ele_struct), pointer :: lord
type (rf_stair_step_struct), pointer :: step
integer :: nn, n_steps, i_fringe_at
real(rp) :: phi0_total, phi0_no_multi, charge_ratio_val, ref_time_start_val
integer :: abs_time_flag
real(C_DOUBLE), allocatable :: h_step_s0(:), h_step_s(:)
real(C_DOUBLE), allocatable :: h_step_p0c(:), h_step_p1c(:)
real(C_DOUBLE), allocatable :: h_step_scale(:), h_step_time(:)
! Exact bend multipole scratch (elements_gpu)
integer :: ix_exact_mag_max
integer(C_INT) :: is_exact
real(rp) :: c_dir_val, rho_val, exact_f_scale_val
real(rp) :: exact_an(0:n_pole_maxx), exact_bn(0:n_pole_maxx)
real(C_DOUBLE) :: exact_an_arr(0:n_pole_maxx), exact_bn_arr(0:n_pole_maxx)
#endif

did_track_to = ix_start - 1

#ifdef USE_GPU_TRACKING
n = size(bunch%particle)
if (n == 0) return

! Ensure persistent SoA buffers are large enough
if (n > soa_cap) then
  if (allocated(vx)) deallocate(vx, vpx, vy, vpy, vz, vpz, state_a, beta_a, p0c_a, t_a, s_a)
  allocate(vx(n), vpx(n), vy(n), vpy(n), vz(n), vpz(n))
  allocate(state_a(n), beta_a(n), p0c_a(n), t_a(n), s_a(n))
  soa_cap = n
endif

on_device = .false.

do ie = ix_start, ix_end
  ele => branch%ele(ie)

  ! Check if this element can be GPU-tracked
  if (.not. ele_gpu_eligible(ele)) exit
  if (bmad_com%spin_tracking_on) exit
  if (bmad_com%high_energy_space_charge_on) exit
  ! CSR/SC elements need track1_bunch_csr sub-stepping — exit to caller
  if (bmad_com%csr_and_space_charge_on .and. &
      (ele%csr_method /= off$ .or. ele%space_charge_method /= off$)) exit
  if (bunch%particle(1)%direction /= 1) exit
  if (bunch%particle(1)%time_dir /= 1) exit

  can_stay = ele_gpu_can_stay_on_device(ele)

  ! If element needs CPU ops and data is on device, download first
  if (.not. can_stay .and. on_device) then
    call gpu_download_particles(vx, vpx, vy, vpy, vz, vpz, &
                                state_a, beta_a, p0c_a, t_a, n, 1, 1)
    call soa_to_bunch(bunch, branch%ele(ie-1), n, vx, vpx, vy, vpy, vz, vpz, &
                       state_a, beta_a, p0c_a, t_a, .true., .true.)
    ! Also write back s
    do j = 1, n
      bunch%particle(j)%s = s_a(j)
    enddo
    on_device = .false.
  endif

  ! --- Device-resident path: everything on GPU ---
  if (can_stay) then
    if (.not. on_device) then
      ! First element in a GPU run: AoS → SoA → upload
      call bunch_to_soa(bunch, n, vx, vpx, vy, vpy, vz, vpz, state_a, beta_a, p0c_a, t_a)
      do j = 1, n
        s_a(j) = bunch%particle(j)%s
      enddo
      call gpu_upload_particles(vx, vpx, vy, vpy, vz, vpz, &
                                state_a, beta_a, p0c_a, t_a, n)
      ! Also upload s_pos
      call upload_s_array()
      on_device = .true.
    endif

    ! Entrance aperture check on device (before misalignment, in lab frame)
    call dispatch_aperture_on_device(ele, n, entrance_end$)

    ! Misalignment on device.
    ! For bends with curvature/ref_tilt/roll, use full curvature-aware bend_offset.
    ! For others, use simple 2D offset+tilt kernel.
    has_misalign = ele%bookkeeping_state%has_misalign
    if (has_misalign) then
      if (ele%key == sbend$ .and. (ele%value(g$) /= 0 .or. ele%value(ref_tilt_tot$) /= 0 &
                                   .or. ele%value(roll$) /= 0)) then
        call gpu_bend_offset(ele%value(g$), ele%value(rho$), &
             ele%value(l$) * 0.5_rp, ele%value(angle$), &
             ele%value(ref_tilt_tot$), ele%value(roll_tot$), &
             ele%value(x_offset_tot$), ele%value(y_offset_tot$), ele%value(z_offset_tot$), &
             ele%value(x_pitch$), ele%value(y_pitch$), 1, n)
      elseif (ele%key == sbend$) then
        call gpu_misalign(ele%value(x_offset_tot$), ele%value(y_offset_tot$), &
                           ele%value(ref_tilt_tot$), 1, n)
      elseif (ele%value(x_pitch_tot$) /= 0 .or. ele%value(y_pitch_tot$) /= 0 .or. &
              ele%value(z_offset_tot$) /= 0) then
        ! 3D misalignment: pitches and/or z_offset require full rotation matrix
        block
          real(rp) :: W_3d(3,3)
          call floor_angles_to_w_mat(ele%value(x_pitch_tot$), ele%value(y_pitch_tot$), &
                                     ele%value(tilt_tot$), w_mat_inv = W_3d)
          call gpu_misalign_3d(W_3d, ele%value(x_offset_tot$), ele%value(y_offset_tot$), &
                               ele%value(z_offset_tot$), 1, n)
        end block
      else
        call gpu_misalign(ele%value(x_offset_tot$), ele%value(y_offset_tot$), &
                           ele%value(tilt_tot$), 1, n)
      endif
    endif

    ! Entrance fringe on device (quad fringe)
    call dispatch_fringe_on_device(ele, branch%param, n, first_track_edge$)

    ! Radiation entrance
    apply_rad = gpu_rad_eligible(ele)
    if (apply_rad) then
      call ensure_rad_map(ele)
      if (associated(ele%rad_map)) call call_gpu_rad_kick(n, ele%rad_map%rm0)
    endif

    ! Body kernel
    call dispatch_body_kernel_on_device(ele, branch%param, n)

    ! Radiation exit
    if (apply_rad .and. associated(ele%rad_map)) call call_gpu_rad_kick(n, ele%rad_map%rm1)

    ! Exit fringe on device (quad fringe)
    call dispatch_fringe_on_device(ele, branch%param, n, second_track_edge$)

    ! Remove misalignment
    if (has_misalign) then
      if (ele%key == sbend$ .and. (ele%value(g$) /= 0 .or. ele%value(ref_tilt_tot$) /= 0 &
                                   .or. ele%value(roll$) /= 0)) then
        call gpu_bend_offset(ele%value(g$), ele%value(rho$), &
             ele%value(l$) * 0.5_rp, ele%value(angle$), &
             ele%value(ref_tilt_tot$), ele%value(roll_tot$), &
             ele%value(x_offset_tot$), ele%value(y_offset_tot$), ele%value(z_offset_tot$), &
             ele%value(x_pitch$), ele%value(y_pitch$), -1, n)
      elseif (ele%key == sbend$) then
        call gpu_misalign(ele%value(x_offset_tot$), ele%value(y_offset_tot$), &
                           ele%value(ref_tilt_tot$), -1, n)
      elseif (ele%value(x_pitch_tot$) /= 0 .or. ele%value(y_pitch_tot$) /= 0 .or. &
              ele%value(z_offset_tot$) /= 0) then
        block
          real(rp) :: W_3d(3,3)
          call floor_angles_to_w_mat(ele%value(x_pitch_tot$), ele%value(y_pitch_tot$), &
                                     ele%value(tilt_tot$), w_mat = W_3d)
          call gpu_misalign_3d(W_3d, ele%value(x_offset_tot$), ele%value(y_offset_tot$), &
                               ele%value(z_offset_tot$), -1, n)
        end block
      else
        call gpu_misalign(ele%value(x_offset_tot$), ele%value(y_offset_tot$), &
                           ele%value(tilt_tot$), -1, n)
      endif
    endif

    ! Exit aperture check on device (after misalignment removed, in lab frame)
    call dispatch_aperture_on_device(ele, n, exit_end$)

    ! Update s position on device and host
    call gpu_s_update(ele%s, n)
    s_a(1:n) = ele%s

    ! Orbit-too-large check on device
    call gpu_orbit_check(n)

    did_track_to = ie

  ! --- CPU-fallback path: use existing per-element routines ---
  else
    select case (ele%key)
    case (drift$)
      call track_bunch_thru_drift_gpu(bunch, ele, did_track)
    case (quadrupole$)
      call track_bunch_thru_quad_gpu(bunch, ele, branch%param, did_track)
    case (sbend$, rf_bend$)
      call track_bunch_thru_bend_gpu(bunch, ele, branch%param, did_track)
    case (lcavity$)
      call track_bunch_thru_lcavity_gpu(bunch, ele, branch%param, did_track)
    case (pipe$, monitor$, instrument$)
      call track_bunch_thru_pipe_gpu(bunch, ele, branch%param, did_track)
    case (solenoid$)
      call track_bunch_thru_solenoid_gpu(bunch, ele, branch%param, did_track)
    case (sol_quad$)
      call track_bunch_thru_sol_quad_gpu(bunch, ele, branch%param, did_track)
    end select

    if (.not. did_track) exit

    ! Run the post-tracking checks that beam_utils normally does
    call check_apertures_after_gpu(bunch, ele, branch%param)
    do j = 1, size(bunch%particle)
      if (bunch%particle(j)%state == alive$) then
        if (orbit_too_large(bunch%particle(j), branch%param)) cycle
      endif
    enddo

    did_track_to = ie
  endif
enddo

! Final download if data is still on device
if (on_device) then
  call gpu_download_particles(vx, vpx, vy, vpy, vz, vpz, &
                              state_a, beta_a, p0c_a, t_a, n, 1, 1)
  call soa_to_bunch(bunch, branch%ele(did_track_to), n, vx, vpx, vy, vpy, vz, vpz, &
                     state_a, beta_a, p0c_a, t_a, .true., .true.)
  do j = 1, n
    bunch%particle(j)%s = s_a(j)
  enddo
  on_device = .false.
endif

! Update charge_live
bunch%charge_live = sum(bunch%particle(:)%charge, mask = (bunch%particle(:)%state == alive$))
#endif

contains

!------------------------------------------------------------------------
! upload_s_array — upload s_pos data to device d_s buffer
!------------------------------------------------------------------------
subroutine upload_s_array()
use, intrinsic :: iso_c_binding
integer(C_INT) :: nb
nb = n * 8  ! sizeof(double) = 8
! Use the gpu_track_drift_dev approach — d_s is managed by ensure_buffers
! We need a direct cudaMemcpy. Use the upload function with s as part of it.
! Actually, d_s is already allocated by ensure_buffers. We upload via a trick:
! pass s_a through a tiny drift of length 0 — no, that's wasteful.
! Instead, let's use a dedicated upload. But we don't have one for s alone.
! For now, just pass s through the existing drift body wrapper with length=0.
! That will upload s, run a no-op kernel, and download s back — not great.
! Better: just upload s via the existing cuda memcpy wrapper.
! We already have gpu_upload_particles which calls ensure_buffers.
! The d_s buffer exists. We need a way to upload to it.
! Let me use a zero-length drift dev call which uploads+downloads s.
! Actually that's wasteful. Let me just leave s on the CPU side and update
! it only during download. We track s_a on the host and just set it.
! For the device-resident path, s_pos isn't needed by most kernels
! (only drift uses it). The drift kernel writes to d_s on device.
! For quad/bend/lcavity, s isn't modified by the kernel.
! So we can: upload s once, let drift modify it, and download at the end.
! But gpu_upload_particles doesn't upload s. Let me just not upload s
! and handle it differently.
!
! Solution: don't upload s. Instead:
! - For drifts: the drift kernel needs d_s. We upload/download it per drift.
!   (The cost is minimal — it's just one array for drift elements.)
! - For other elements: s is set to ele%s on device via gpu_s_update.
! This is already what we do above with gpu_s_update.
! For drifts, the drift kernel updates d_s. We need to upload d_s before the
! drift and download after. Let's handle this in dispatch_body_kernel_on_device.
end subroutine upload_s_array

!------------------------------------------------------------------------
! dispatch_aperture_on_device — apply rectangular aperture check on device
!------------------------------------------------------------------------
subroutine dispatch_aperture_on_device(ele, np, at_edge)

type (ele_struct), intent(in) :: ele
integer(C_INT), intent(in) :: np
integer, intent(in) :: at_edge  ! entrance_end$ or exit_end$

! Check if aperture should be applied at this edge
! aperture_at: entrance_end$=1, exit_end$=2, both_ends$=3, no_end$/no_aperture$=4
if (ele%aperture_at == no_aperture$) return
if (ele%aperture_at == entrance_end$ .and. at_edge /= entrance_end$) return
if (ele%aperture_at == exit_end$ .and. at_edge /= exit_end$) return

if (ele%value(x1_limit$) == 0 .and. ele%value(x2_limit$) == 0 .and. &
    ele%value(y1_limit$) == 0 .and. ele%value(y2_limit$) == 0) return

select case (ele%aperture_type)
case (rectangular$)
  call gpu_check_aperture_rect(ele%value(x1_limit$), ele%value(x2_limit$), &
                                ele%value(y1_limit$), ele%value(y2_limit$), np)
case (elliptical$)
  call gpu_check_aperture_ellipse(ele%value(x1_limit$), ele%value(x2_limit$), &
                                   ele%value(y1_limit$), ele%value(y2_limit$), np)
end select

end subroutine dispatch_aperture_on_device

!------------------------------------------------------------------------
! dispatch_fringe_on_device — apply fringe kicks on device
!------------------------------------------------------------------------
subroutine dispatch_fringe_on_device(ele, param, np, edge)

use multipole_mod, only: ab_multipole_kicks
type (ele_struct), intent(in) :: ele
type (lat_param_struct), intent(in) :: param
integer(C_INT), intent(in) :: np
integer, intent(in) :: edge

integer :: fringe_type_val
real(rp) :: charge_dir_val, k1_val
real(rp) :: an_tmp(0:n_pole_maxx), bn_tmp(0:n_pole_maxx)
real(C_DOUBLE) :: bp_arr(8), ap_arr(8)
integer :: ix_tmp

select case (ele%key)
case (quadrupole$)
  fringe_type_val = nint(ele%value(fringe_type$))
  if (fringe_type_val == none$) return

  ! Compute charge_dir for fringe
  charge_dir_val = rel_tracking_charge_to_mass(bunch%particle(1), param%particle) * &
                   ele%orientation * bunch%particle(1)%direction

  call gpu_quad_fringe(ele%value(k1$), ele%value(fq1$), ele%value(fq2$), &
                       charge_dir_val, &
                       int(fringe_type_val, C_INT), int(edge, C_INT), &
                       int(bunch%particle(1)%time_dir, C_INT), np)

case (sextupole$)
  fringe_type_val = nint(ele%value(fringe_type$))
  if (fringe_type_val == none$) return
  if (fringe_type_val /= full$ .and. fringe_type_val /= hard_edge_only$) return

  charge_dir_val = rel_tracking_charge_to_mass(bunch%particle(1), param%particle) * &
                   ele%orientation * bunch%particle(1)%direction
  bp_arr = 0; ap_arr = 0
  bp_arr(2) = ele%value(k2$)   ! n=2 only for sextupoles
  block
    integer :: is_ent
    is_ent = merge(1, 0, edge == first_track_edge$)
    call gpu_hard_multipole_edge(bp_arr, ap_arr, int(2, C_INT), charge_dir_val, &
                                 int(is_ent, C_INT), np)
  end block

case (sbend$, rf_bend$)
  fringe_type_val = nint(ele%value(fringe_type$))
  if (fringe_type_val == none$) return
  ! Full (PTC-style) fringe
  if (fringe_type_val == full$) then
    block
      real(rp) :: g_tot_exact, beta0_exact, e_ang_exact, fint_exact, hgap_exact
      integer :: is_exit_exact, physical_end_exact

      if (ele%is_on) then
        g_tot_exact = ele%value(g$) + ele%value(dg$)  ! NO charge_dir for exact fringe
      else
        g_tot_exact = 0
      endif
      beta0_exact = ele%value(p0c$) / ele%value(e_tot$)
      physical_end_exact = physical_ele_end(edge, bunch%particle(1), ele%orientation)

      ! Hard multipole edge kick at entrance (before exact fringe)
      if (physical_end_exact == entrance_end$) then
        charge_dir_val = rel_tracking_charge_to_mass(bunch%particle(1), param%particle) * &
                         ele%orientation * bunch%particle(1)%direction
        bp_arr = 0; ap_arr = 0
        bp_arr(1) = ele%value(k1$)
        call gpu_hard_multipole_edge(bp_arr, ap_arr, int(1, C_INT), charge_dir_val, int(1, C_INT), np)
      endif

      if (physical_end_exact == entrance_end$) then
        e_ang_exact = bunch%particle(1)%time_dir * ele%value(e1$)
        fint_exact = bunch%particle(1)%time_dir * ele%value(fint$)
        hgap_exact = ele%value(hgap$)
        is_exit_exact = 0
      else
        e_ang_exact = bunch%particle(1)%time_dir * ele%value(e2$)
        fint_exact = bunch%particle(1)%time_dir * ele%value(fintx$)
        hgap_exact = ele%value(hgapx$)
        is_exit_exact = 1
      endif

      call gpu_exact_bend_fringe(g_tot_exact, beta0_exact, &
          e_ang_exact, fint_exact, hgap_exact, &
          int(is_exit_exact, C_INT), np)

      ! Hard multipole edge kick at exit (after exact fringe)
      if (physical_end_exact == exit_end$) then
        charge_dir_val = rel_tracking_charge_to_mass(bunch%particle(1), param%particle) * &
                         ele%orientation * bunch%particle(1)%direction
        bp_arr = 0; ap_arr = 0
        bp_arr(1) = ele%value(k1$)
        call gpu_hard_multipole_edge(bp_arr, ap_arr, int(1, C_INT), charge_dir_val, int(0, C_INT), np)
      endif
    end block
    return
  endif

  ! SAD fringe types: sad_full$ and soft_edge_only$
  if (fringe_type_val == sad_full$ .or. fringe_type_val == soft_edge_only$) then
    block
      real(rp) :: g_sad, fb_sad, c_dir_sad
      integer :: physical_end_sad

      physical_end_sad = physical_ele_end(edge, bunch%particle(1), ele%orientation)

      ! Compute fb = 12 * fint * hgap (entrance) or 12 * fintx * hgapx (exit)
      if (physical_end_sad == entrance_end$) then
        fb_sad = 12 * ele%value(fint$) * ele%value(hgap$)
      else
        fb_sad = 12 * ele%value(fintx$) * ele%value(hgapx$)
      endif

      ! Compute g with all sign factors baked in
      ! c_dir includes time_dir (matching sad_soft_bend_edge_kick)
      c_dir_sad = rel_tracking_charge_to_mass(bunch%particle(1), param%particle) * &
                  ele%orientation * bunch%particle(1)%direction * bunch%particle(1)%time_dir
      g_sad = (ele%value(g$) + ele%value(dg$)) * c_dir_sad
      if (edge == second_track_edge$) g_sad = -g_sad

      ! Hard multipole edge kick at entrance (before bend fringe)
      if (physical_end_sad == entrance_end$) then
        charge_dir_val = rel_tracking_charge_to_mass(bunch%particle(1), param%particle) * &
                         ele%orientation * bunch%particle(1)%direction
        bp_arr = 0; ap_arr = 0
        bp_arr(1) = ele%value(k1$)
        call gpu_hard_multipole_edge(bp_arr, ap_arr, int(1, C_INT), charge_dir_val, int(1, C_INT), np)
      endif

      if (fringe_type_val == sad_full$) then
        ! sad_full$: sad_soft then hwang at entrance, hwang then sad_soft at exit
        if (edge == first_track_edge$) then
          ! SAD soft kick first, then Hwang (with fint_gap=0)
          if (fb_sad /= 0 .and. g_sad /= 0) &
            call gpu_sad_bend_fringe(g_sad, fb_sad, np)
          ! Hwang kick with fint_gap = 0 (hard_edge_only behavior)
          charge_dir_val = rel_tracking_charge_to_mass(bunch%particle(1), param%particle) * &
                           ele%orientation * bunch%particle(1)%direction
          block
            real(rp) :: g_hwang, e_ang_hwang
            integer :: entering_hwang
            if (ele%is_on) then
              g_hwang = (ele%value(g$) + ele%value(dg$)) * charge_dir_val
              k1_val = ele%value(k1$)
            else
              g_hwang = 0; k1_val = 0
            endif
            if (physical_end_sad == entrance_end$) then
              e_ang_hwang = ele%value(e1$)
            else
              e_ang_hwang = ele%value(e2$)
            endif
            entering_hwang = 0
            if ((edge == first_track_edge$ .and. bunch%particle(1)%direction == 1) .or. &
                (edge == second_track_edge$ .and. bunch%particle(1)%direction == -1)) entering_hwang = 1
            call gpu_bend_fringe(g_hwang, e_ang_hwang, 0.0_rp, k1_val, &
                                  int(entering_hwang, C_INT), &
                                  int(bunch%particle(1)%time_dir, C_INT), np)
          end block
        else
          ! Hwang kick first (with fint_gap=0), then SAD soft kick
          charge_dir_val = rel_tracking_charge_to_mass(bunch%particle(1), param%particle) * &
                           ele%orientation * bunch%particle(1)%direction
          block
            real(rp) :: g_hwang, e_ang_hwang
            integer :: entering_hwang
            if (ele%is_on) then
              g_hwang = (ele%value(g$) + ele%value(dg$)) * charge_dir_val
              k1_val = ele%value(k1$)
            else
              g_hwang = 0; k1_val = 0
            endif
            if (physical_end_sad == entrance_end$) then
              e_ang_hwang = ele%value(e1$)
            else
              e_ang_hwang = ele%value(e2$)
            endif
            entering_hwang = 0
            if ((edge == first_track_edge$ .and. bunch%particle(1)%direction == 1) .or. &
                (edge == second_track_edge$ .and. bunch%particle(1)%direction == -1)) entering_hwang = 1
            call gpu_bend_fringe(g_hwang, e_ang_hwang, 0.0_rp, k1_val, &
                                  int(entering_hwang, C_INT), &
                                  int(bunch%particle(1)%time_dir, C_INT), np)
          end block
          if (fb_sad /= 0 .and. g_sad /= 0) &
            call gpu_sad_bend_fringe(g_sad, fb_sad, np)
        endif
      else
        ! soft_edge_only$: just the SAD soft kick
        if (fb_sad /= 0 .and. g_sad /= 0) &
          call gpu_sad_bend_fringe(g_sad, fb_sad, np)
      endif

      ! Hard multipole edge kick at exit (after bend fringe)
      if (physical_end_sad == exit_end$) then
        charge_dir_val = rel_tracking_charge_to_mass(bunch%particle(1), param%particle) * &
                         ele%orientation * bunch%particle(1)%direction
        bp_arr = 0; ap_arr = 0
        bp_arr(1) = ele%value(k1$)
        call gpu_hard_multipole_edge(bp_arr, ap_arr, int(1, C_INT), charge_dir_val, int(0, C_INT), np)
      endif
    end block
    return
  endif

  ! Basic/hard_edge fringe (Hwang)
  if (fringe_type_val /= basic_bend$ .and. fringe_type_val /= hard_edge_only$) return

  charge_dir_val = rel_tracking_charge_to_mass(bunch%particle(1), param%particle) * &
                   ele%orientation * bunch%particle(1)%direction

  block
    real(rp) :: g_tot_val, e_ang, fint_gap_val
    integer :: entering, physical_end

    if (ele%is_on) then
      g_tot_val = (ele%value(g$) + ele%value(dg$)) * charge_dir_val
      k1_val = ele%value(k1$)
    else
      g_tot_val = 0; k1_val = 0
    endif

    physical_end = physical_ele_end(edge, bunch%particle(1), ele%orientation)
    if (physical_end == entrance_end$) then
      e_ang = ele%value(e1$)
      fint_gap_val = ele%value(fint$) * ele%value(hgap$)
    else
      e_ang = ele%value(e2$)
      fint_gap_val = ele%value(fintx$) * ele%value(hgapx$)
    endif

    if (fringe_type_val == hard_edge_only$) fint_gap_val = 0

    entering = 0
    if ((edge == first_track_edge$ .and. bunch%particle(1)%direction == 1) .or. &
        (edge == second_track_edge$ .and. bunch%particle(1)%direction == -1)) entering = 1

    call gpu_bend_fringe(g_tot_val, e_ang, fint_gap_val, k1_val, &
                          int(entering, C_INT), &
                          int(bunch%particle(1)%time_dir, C_INT), np)
  end block

end select

end subroutine dispatch_fringe_on_device

!------------------------------------------------------------------------
! dispatch_body_kernel_on_device — launch the appropriate body kernel
!------------------------------------------------------------------------
subroutine dispatch_body_kernel_on_device(ele, param, np)

type (ele_struct), target, intent(inout) :: ele
type (lat_param_struct), intent(in) :: param
integer(C_INT), intent(in) :: np

select case (ele%key)
case (drift$, pipe$, monitor$, instrument$, kicker$, hkicker$, vkicker$)
  call dispatch_drift_body(ele, np)
case (quadrupole$)
  call dispatch_quad_body(ele, param, np)
case (sbend$, rf_bend$)
  call dispatch_bend_body(ele, param, np)
case (lcavity$)
  call dispatch_lcavity_body(ele, param, np)
case (solenoid$)
  call dispatch_solenoid_body(ele, param, np)
case (sol_quad$)
  call dispatch_sol_quad_body(ele, param, np)
case (wiggler$, undulator$)
  call dispatch_wiggler_body(ele, param, np)
end select

end subroutine dispatch_body_kernel_on_device

!------------------------------------------------------------------------
subroutine dispatch_drift_body(ele, np)
type (ele_struct), intent(in) :: ele
integer(C_INT), intent(in) :: np
real(rp) :: mc2_val, len_val
integer :: j2
real(C_DOUBLE) :: dummy_s(1)

mc2_val = mass_of(bunch%particle(1)%species)
len_val = ele%value(l$)
if (len_val == 0) return

! For drift, need s on device. Upload current s_a, run kernel, download.
call gpu_track_drift_dev(s_a, mc2_val, len_val, np)
! s_a is now updated on host (drift_dev does upload+download of s)
end subroutine dispatch_drift_body

!------------------------------------------------------------------------
subroutine dispatch_quad_body(ele, param, np)
type (ele_struct), intent(in) :: ele
type (lat_param_struct), intent(in) :: param
integer(C_INT), intent(in) :: np

ele_length = ele%value(l$)
if (ele_length == 0) return

mc2 = mass_of(bunch%particle(1)%species)
delta_ref_time = ele%value(delta_ref_time$)
e_tot_ele = ele%value(e_tot$)

call multipole_ele_to_ab(ele, .false., ix_mag_max, an, bn, magnetic$, include_kicks$, b1)
call multipole_ele_to_ab(ele, .false., ix_elec_max, an_elec, bn_elec, electric$)

has_mag_multipoles = (ix_mag_max > -1)
length = bunch%particle(1)%time_dir * ele_length
n_step = 1
if (has_mag_multipoles .or. ix_elec_max > -1) &
  n_step = max(nint(abs(length) / ele%value(ds_step$)), 1)

rel_tracking_charge = rel_tracking_charge_to_mass(bunch%particle(1), param%particle)
charge_dir = rel_tracking_charge * ele%orientation * bunch%particle(1)%direction * bunch%particle(1)%time_dir

call precompute_multipole_arrays(bunch%particle(1), ele, &
    ix_mag_max, an, bn, ix_elec_max, an_elec, bn_elec, &
    ele_length, n_step, a2_arr, b2_arr, ea2_arr, eb2_arr, cm_arr)

call gpu_track_quad_dev(mc2, b1, ele_length, delta_ref_time, &
                        e_tot_ele, charge_dir, np, &
                        a2_arr, b2_arr, cm_arr, &
                        int(ix_mag_max, C_INT), int(n_step, C_INT), &
                        ea2_arr, eb2_arr, int(ix_elec_max, C_INT))
end subroutine dispatch_quad_body

!------------------------------------------------------------------------
subroutine dispatch_bend_body(ele, param, np)
type (ele_struct), target, intent(inout) :: ele
type (lat_param_struct), intent(in) :: param
integer(C_INT), intent(in) :: np

ele_length = ele%value(l$)
if (ele_length == 0) return

mc2 = mass_of(bunch%particle(1)%species)
delta_ref_time = ele%value(delta_ref_time$)
e_tot_ele = ele%value(e_tot$)
p0c_ele = ele%value(p0c$)

rel_charge_dir = ele%orientation * bunch%particle(1)%direction * &
                 rel_tracking_charge_to_mass(bunch%particle(1), param%particle)
c_dir_val = ele%orientation * bunch%particle(1)%direction * charge_of(bunch%particle(1)%species)

is_exact = 0; ix_exact_mag_max = -1; rho_val = 0; exact_f_scale_val = 0
exact_an_arr = 0; exact_bn_arr = 0

if (nint(ele%value(exact_multipoles$)) /= off$ .and. ele%value(g$) /= 0) then
  is_exact = 1
  call multipole_ele_to_ab(ele, .false., ix_mag_max, an, bn, magnetic$, include_kicks$)
  b1 = 0
  call multipole_ele_to_ab(ele, .false., ix_exact_mag_max, exact_an, exact_bn, magnetic$, include_kicks$)
  if (nint(ele%value(exact_multipoles$)) == horizontally_pure$ .and. ix_exact_mag_max /= -1) then
    call convert_bend_exact_multipole(ele%value(g$), vertically_pure$, exact_an, exact_bn)
    ix_exact_mag_max = n_pole_maxx
  endif
  exact_an_arr = exact_an; exact_bn_arr = exact_bn
  rho_val = ele%value(rho$)
  if (ele%value(l$) /= 0) exact_f_scale_val = ele%value(p0c$) / (c_light * charge_of(param%particle) * ele%value(l$))
else
  call multipole_ele_to_ab(ele, .false., ix_mag_max, an, bn, magnetic$, include_kicks$, b1)
  b1 = b1 * rel_charge_dir
  if (abs(b1) < 1d-10) then; bn(1) = b1; b1 = 0; endif
endif

call multipole_ele_to_ab(ele, .false., ix_elec_max, an_elec, bn_elec, electric$)
g = ele%value(g$)
length = bunch%particle(1)%time_dir * ele_length
if (length == 0) then; dg = 0; else; dg = bn(0)/ele_length; bn(0) = 0; endif
g_tot = (g + dg) * rel_charge_dir
has_mag_multipoles = (ix_mag_max > -1)
has_elec_multipoles = (ix_elec_max > -1)
n_step = 1
if (has_mag_multipoles .or. has_elec_multipoles) &
  n_step = max(nint(abs(length) / ele%value(ds_step$)), 1)
call precompute_multipole_arrays(bunch%particle(1), ele, &
    ix_mag_max, an, bn, ix_elec_max, an_elec, bn_elec, &
    ele_length, n_step, a2_arr, b2_arr, ea2_arr, eb2_arr, cm_arr)
call gpu_track_bend_dev(mc2, g, g_tot, dg, b1, &
                        ele_length, delta_ref_time, e_tot_ele, &
                        rel_charge_dir, p0c_ele, np, &
                        a2_arr, b2_arr, cm_arr, &
                        int(ix_mag_max, C_INT), int(n_step, C_INT), &
                        ea2_arr, eb2_arr, int(ix_elec_max, C_INT), &
                        is_exact, exact_an_arr, exact_bn_arr, &
                        int(ix_exact_mag_max, C_INT), &
                        real(rho_val, C_DOUBLE), real(c_dir_val, C_DOUBLE), &
                        real(exact_f_scale_val, C_DOUBLE))
end subroutine dispatch_bend_body

!------------------------------------------------------------------------
subroutine dispatch_lcavity_body(ele, param, np)
type (ele_struct), target, intent(inout) :: ele
type (lat_param_struct), intent(in) :: param
integer(C_INT), intent(in) :: np

if (ele%value(l$) == 0) return
if (ele%value(rf_frequency$) == 0) return

lord => pointer_to_super_lord(ele)
if (lord%value(ks$) /= 0) return

call multipole_ele_to_ab(ele, .false., ix_mag_max, an, bn, magnetic$, include_kicks$)
call multipole_ele_to_ab(ele, .false., ix_elec_max, an_elec, bn_elec, electric$)
if (ix_mag_max > -1) return
if (ix_elec_max > -1) return
if (ele%value(coupler_strength$) /= 0) return

if (nint(lord%value(fringe_type$)) == none$) then
  i_fringe_at = 0
else
  i_fringe_at = nint(lord%value(fringe_at$))
  if (i_fringe_at < 1 .or. i_fringe_at > 3) i_fringe_at = 0
endif
charge_ratio_val = charge_of(bunch%particle(1)%species) / (2.0_rp * charge_of(lord%ref_species))

mc2 = mass_of(bunch%particle(1)%species)
n_steps = nint(lord%value(n_rf_steps$))
phi0_no_multi = lord%value(phi0$) + lord%value(phi0_err$)
phi0_total = phi0_no_multi + lord%value(phi0_multipass$)
abs_time_flag = 0
ref_time_start_val = 0.0_rp
if (bmad_com%absolute_time_tracking) then
  abs_time_flag = 1
  if (bmad_com%absolute_time_ref_shift) then
    ! For multipass slaves, use the first-pass element's ref_time_start
    ! (matches this_rf_phase in track_a_lcavity.f90)
    if (lord%slave_status == multipass_slave$) then
      block
        type (ele_pointer_struct), allocatable :: mp_chain(:)
        integer :: ix_pass_mp, n_links_mp
        call multipass_chain(lord, ix_pass_mp, n_links_mp, mp_chain)
        ref_time_start_val = mp_chain(1)%ele%value(ref_time_start$)
      end block
    else
      ref_time_start_val = lord%value(ref_time_start$)
    endif
  endif
endif

block
  type (ele_struct), pointer :: mlord2
  mlord2 => lord
  if (lord%slave_status == multipass_slave$) mlord2 => pointer_to_lord(lord, 1)
  allocate(h_step_s0(n_steps+2), h_step_s(n_steps+2))
  allocate(h_step_p0c(n_steps+2), h_step_p1c(n_steps+2))
  allocate(h_step_scale(n_steps+2), h_step_time(n_steps+2))
  do j = 0, n_steps + 1
    step => lord%rf%steps(j)
    h_step_s0(j+1)    = step%s0
    h_step_s(j+1)     = step%s
    h_step_p0c(j+1)   = step%p0c
    h_step_p1c(j+1)   = step%p1c
    h_step_scale(j+1) = step%scale
    h_step_time(j+1)  = mlord2%rf%steps(j)%time
  enddo
end block

call gpu_track_lcavity_dev(mc2, &
                           h_step_s0, h_step_s, h_step_p0c, h_step_p1c, &
                           h_step_scale, h_step_time, &
                           int(n_steps, C_INT), &
                           lord%value(voltage$), lord%value(voltage_err$), &
                           lord%value(field_autoscale$), &
                           ele%value(rf_frequency$), phi0_total, &
                           lord%value(voltage_tot$), lord%value(l_active$), &
                           int(nint(lord%value(cavity_type$)), C_INT), &
                           int(i_fringe_at, C_INT), charge_ratio_val, &
                           int(np, C_INT), &
                           int(abs_time_flag, C_INT), phi0_no_multi, &
                           ref_time_start_val)

deallocate(h_step_s0, h_step_s, h_step_p0c, h_step_p1c, h_step_scale, h_step_time)
end subroutine dispatch_lcavity_body

!------------------------------------------------------------------------
subroutine dispatch_solenoid_body(ele, param, np)
type (ele_struct), intent(in) :: ele
type (lat_param_struct), intent(in) :: param
integer(C_INT), intent(in) :: np
real(rp) :: ks0_val

ele_length = ele%value(l$)
if (ele_length == 0) return

mc2 = mass_of(bunch%particle(1)%species)
delta_ref_time = ele%value(delta_ref_time$)
e_tot_ele = ele%value(e_tot$)

rel_tracking_charge = rel_tracking_charge_to_mass(bunch%particle(1), param%particle)
ks0_val = rel_tracking_charge * ele%value(bs_field$) * charge_of(bunch%particle(1)%species) * c_light / bunch%particle(1)%p0c

call multipole_ele_to_ab(ele, .false., ix_mag_max, an, bn, magnetic$, include_kicks$, b1)
call multipole_ele_to_ab(ele, .false., ix_elec_max, an_elec, bn_elec, electric$)

length = bunch%particle(1)%time_dir * ele_length
n_step = 1
if (ix_mag_max > -1 .or. ix_elec_max > -1) &
  n_step = max(nint(abs(length) / ele%value(ds_step$)), 1)

call precompute_multipole_arrays(bunch%particle(1), ele, &
    ix_mag_max, an, bn, ix_elec_max, an_elec, bn_elec, &
    ele_length, n_step, a2_arr, b2_arr, ea2_arr, eb2_arr, cm_arr)

call gpu_track_solenoid_dev(mc2, ks0_val, ele_length, delta_ref_time, &
                            e_tot_ele, np, int(n_step, C_INT), &
                            a2_arr, b2_arr, cm_arr, &
                            int(ix_mag_max, C_INT), &
                            ea2_arr, eb2_arr, int(ix_elec_max, C_INT))
end subroutine dispatch_solenoid_body

!------------------------------------------------------------------------
subroutine dispatch_sol_quad_body(ele, param, np)
type (ele_struct), intent(in) :: ele
type (lat_param_struct), intent(in) :: param
integer(C_INT), intent(in) :: np
real(rp) :: ks0_val, ks_val, k1_val

ele_length = ele%value(l$)
if (ele_length == 0) return

mc2 = mass_of(bunch%particle(1)%species)
delta_ref_time = ele%value(delta_ref_time$)
e_tot_ele = ele%value(e_tot$)

rel_tracking_charge = rel_tracking_charge_to_mass(bunch%particle(1), param%particle)
charge_dir = rel_tracking_charge * ele%orientation * bunch%particle(1)%direction * bunch%particle(1)%time_dir

call multipole_ele_to_ab(ele, .false., ix_mag_max, an, bn, magnetic$, include_kicks$, b1)
call multipole_ele_to_ab(ele, .false., ix_elec_max, an_elec, bn_elec, electric$)

length = bunch%particle(1)%time_dir * ele_length
n_step = 1
if (ix_mag_max > -1 .or. ix_elec_max > -1) &
  n_step = max(nint(abs(length) / ele%value(ds_step$)), 1)

call precompute_multipole_arrays(bunch%particle(1), ele, &
    ix_mag_max, an, bn, ix_elec_max, an_elec, bn_elec, &
    ele_length, n_step, a2_arr, b2_arr, ea2_arr, eb2_arr, cm_arr)

if (b1 == 0) then
  ! Pure solenoid tracking
  ks0_val = rel_tracking_charge * ele%value(bs_field$) * charge_of(bunch%particle(1)%species) * c_light / bunch%particle(1)%p0c
  call gpu_track_solenoid_dev(mc2, ks0_val, ele_length, delta_ref_time, &
                              e_tot_ele, np, int(n_step, C_INT), &
                              a2_arr, b2_arr, cm_arr, &
                              int(ix_mag_max, C_INT), &
                              ea2_arr, eb2_arr, int(ix_elec_max, C_INT))
else
  ! Sol_quad tracking
  ks_val = rel_tracking_charge * ele%value(ks$)
  k1_val = charge_dir * b1 / ele_length
  call gpu_track_sol_quad_dev(mc2, ks_val, k1_val, ele_length, delta_ref_time, &
                              e_tot_ele, np, int(n_step, C_INT), &
                              a2_arr, b2_arr, cm_arr, &
                              int(ix_mag_max, C_INT), &
                              ea2_arr, eb2_arr, int(ix_elec_max, C_INT))
endif
end subroutine dispatch_sol_quad_body

!------------------------------------------------------------------------
subroutine dispatch_wiggler_body(ele, param, np)
type (ele_struct), target, intent(inout) :: ele
type (lat_param_struct), intent(in) :: param
integer(C_INT), intent(in) :: np
type (ele_struct), pointer :: field_ele
real(rp) :: k1x_val, k1y_val, kz_val, factor_val, ky2_val, osc_amp_val, p0c_ele_val
integer(C_INT) :: is_helical_val

ele_length = ele%value(l$)
if (ele_length == 0) return

mc2 = mass_of(bunch%particle(1)%species)
delta_ref_time = ele%value(delta_ref_time$)
e_tot_ele = ele%value(e_tot$)
p0c_ele_val = ele%value(p0c$)
osc_amp_val = ele%value(osc_amplitude$)

field_ele => pointer_to_field_ele(ele, 1)

! Compute kz, ky2
if (ele%value(l_period$) == 0) then
  kz_val = 1d100
  ky2_val = 0
else
  kz_val = twopi / ele%value(l_period$)
  ky2_val = kz_val**2 + ele%value(kx$)**2
endif

! Compute averaged focusing strengths (sign-independent of charge and direction)
rel_tracking_charge = rel_tracking_charge_to_mass(bunch%particle(1), param%particle)
factor_val = abs(rel_tracking_charge) * 0.5_rp * (c_light * ele%value(b_max$) / ele%value(p0c$))**2

if (field_ele%field_calc == helical_model$) then
  k1x_val = -factor_val
  k1y_val = -factor_val
  is_helical_val = 1
else
  k1x_val =  factor_val * (ele%value(kx$) / kz_val)**2
  k1y_val = -factor_val * ky2_val / kz_val**2
  is_helical_val = 0
endif

! Multipoles
call multipole_ele_to_ab(ele, .false., ix_mag_max, an, bn, magnetic$, include_kicks$)
call multipole_ele_to_ab(ele, .false., ix_elec_max, an_elec, bn_elec, electric$)

length = bunch%particle(1)%time_dir * ele_length
n_step = max(nint(ele%value(l$) / ele%value(ds_step$)), 1)
if (ix_mag_max < 0 .and. ix_elec_max < 0) n_step = 1

call precompute_multipole_arrays(bunch%particle(1), ele, &
    ix_mag_max, an, bn, ix_elec_max, an_elec, bn_elec, &
    ele_length, n_step, a2_arr, b2_arr, ea2_arr, eb2_arr, cm_arr)

call gpu_track_wiggler_dev(mc2, ele_length, delta_ref_time, &
                            e_tot_ele, p0c_ele_val, &
                            k1x_val, k1y_val, kz_val, is_helical_val, &
                            osc_amp_val, &
                            np, int(n_step, C_INT), &
                            a2_arr, b2_arr, cm_arr, &
                            int(ix_mag_max, C_INT), &
                            ea2_arr, eb2_arr, int(ix_elec_max, C_INT))
end subroutine dispatch_wiggler_body

end subroutine track_bunch_thru_elements_gpu

!------------------------------------------------------------------------
! check_apertures_after_gpu — check exit aperture for all alive particles
!------------------------------------------------------------------------
subroutine check_apertures_after_gpu (bunch, ele, param)

type (bunch_struct), intent(inout) :: bunch
type (ele_struct),   intent(inout) :: ele
type (lat_param_struct), intent(inout) :: param

integer :: j

do j = 1, size(bunch%particle)
  if (bunch%particle(j)%state == alive$) then
    call check_aperture_limit(bunch%particle(j), ele, second_track_edge$, param)
  endif
enddo

end subroutine check_apertures_after_gpu

!------------------------------------------------------------------------
! PERSISTENT GPU SESSION
!
! These routines keep particle data on the GPU across multiple calls
! to track1_bunch_hom. This solves the problem where Tao (or other
! callers) tracks one element at a time, preventing the multi-element
! dispatch from batching.
!
! gpu_persistent_track_element: Track one element on device. If data
!   is not yet on device, upload. Leave data on device for next call.
!   Returns did_track=.true. if handled on GPU.
!
! gpu_persistent_flush: Download data from device back to bunch.
!   Must be called before any CPU-side access to bunch particle data.
!------------------------------------------------------------------------

subroutine gpu_persistent_track_element(bunch, ele, param, did_track)

use multipole_mod, only: ab_multipole_kicks
use radiation_mod, only: radiation_map_setup
use, intrinsic :: iso_c_binding

type (bunch_struct),     intent(inout) :: bunch
type (ele_struct), target, intent(inout) :: ele
type (lat_param_struct), intent(in)    :: param
logical,                 intent(out)   :: did_track

#ifdef USE_GPU_TRACKING
integer(C_INT) :: n
integer :: j
logical :: can_stay, has_misalign, apply_rad
real(C_DOUBLE) :: misalign_W(3,3), misalign_Lx, misalign_Ly, misalign_Lz

! Per-element scratch for lcavity dispatch
integer :: ix_mag_max, ix_elec_max, n_steps, n_step, i_fringe_at, abs_time_flag
real(rp) :: mc2, ele_length, b1, delta_ref_time, e_tot_ele, p0c_ele
real(rp) :: charge_dir, rel_tracking_charge, length, g, g_tot, dg, rel_charge_dir
real(rp) :: phi0_total, phi0_no_multi, charge_ratio_val, ref_time_start_val
real(rp) :: ks0, ks_val, k1_val
real(rp) :: an(0:n_pole_maxx), bn(0:n_pole_maxx)
real(rp) :: an_elec(0:n_pole_maxx), bn_elec(0:n_pole_maxx)
real(C_DOUBLE) :: a2_arr(0:n_pole_maxx), b2_arr(0:n_pole_maxx)
real(C_DOUBLE) :: ea2_arr(0:n_pole_maxx), eb2_arr(0:n_pole_maxx)
real(C_DOUBLE) :: cm_arr(0:n_pole_maxx, 0:n_pole_maxx)
logical :: has_mag_multipoles, has_elec_multipoles
type (ele_struct), pointer :: lord
type (rf_stair_step_struct), pointer :: step
type (fringe_field_info_struct) :: fringe_info
real(C_DOUBLE), allocatable :: h_step_s0(:), h_step_s(:)
real(C_DOUBLE), allocatable :: h_step_p0c(:), h_step_p1c(:)
real(C_DOUBLE), allocatable :: h_step_scale(:), h_step_time(:)
! Exact bend multipole scratch (persistent)
integer :: ix_exact_mag_max_p
integer(C_INT) :: is_exact_p
real(rp) :: c_dir_val_p, rho_val_p, exact_f_scale_val_p
real(rp) :: exact_an_p(0:n_pole_maxx), exact_bn_p(0:n_pole_maxx)
real(C_DOUBLE) :: exact_an_arr_p(0:n_pole_maxx), exact_bn_arr_p(0:n_pole_maxx)
#endif

did_track = .false.

#ifdef USE_GPU_TRACKING
can_stay = ele_gpu_can_stay_on_device(ele)
if (.not. can_stay) then
  ! Element can't stay on device — flush and let caller handle it
  if (gpu_persist_on_device) call gpu_persistent_flush(bunch, ele)
  return
endif

! First element: data not yet on device -- upload will happen below.
! Handle via persistent path directly to avoid redundant per-element transfer.

n = size(bunch%particle)
if (n == 0) return

! Ensure persistent SoA buffers
if (n > gpu_persist_n) then
  if (allocated(gp_vx)) deallocate(gp_vx, gp_vpx, gp_vy, gp_vpy, gp_vz, gp_vpz, &
                                    gp_state, gp_beta, gp_p0c, gp_t, gp_s)
  allocate(gp_vx(n), gp_vpx(n), gp_vy(n), gp_vpy(n), gp_vz(n), gp_vpz(n))
  allocate(gp_state(n), gp_beta(n), gp_p0c(n), gp_t(n), gp_s(n))
  gpu_persist_n = n
endif

! Detect bunch switch: if bunch identity changed (different allocation),
! save current device data to its slot and try to restore the new bunch.
block
  integer(8) :: bunch_id
  bunch_id = transfer(loc(bunch%particle(1)%vec(1)), bunch_id)
  if (gpu_persist_on_device .and. bunch_id /= gpu_persist_bunch_id) then
    ! Save current bunch's device data to its slot
    call gpu_multi_bunch_save(gpu_persist_bunch_id, gpu_persist_n)
    ! Try to restore the new bunch from a saved slot
    call gpu_multi_bunch_restore(bunch_id, n, gpu_persist_on_device)
  endif
  gpu_persist_bunch_id = bunch_id
end block

! Upload if not already on device
if (.not. gpu_persist_on_device) then
  call bunch_to_soa(bunch, n, gp_vx, gp_vpx, gp_vy, gp_vpy, gp_vz, gp_vpz, &
                     gp_state, gp_beta, gp_p0c, gp_t)
  do j = 1, n
    gp_s(j) = bunch%particle(j)%s
  enddo
  call gpu_upload_particles(gp_vx, gp_vpx, gp_vy, gp_vpy, gp_vz, gp_vpz, &
                            gp_state, gp_beta, gp_p0c, gp_t, n)
  gpu_persist_on_device = .true.
endif

! --- Track one element on device (same logic as track_bunch_thru_elements_gpu) ---

! Entrance aperture
call dispatch_aperture_on_device_pub(ele, n, entrance_end$)

! Misalignment
has_misalign = ele%bookkeeping_state%has_misalign
if (has_misalign) then
  if (ele%key == sbend$ .and. (ele%value(g$) /= 0 .or. ele%value(ref_tilt_tot$) /= 0 &
                               .or. ele%value(roll$) /= 0)) then
    call gpu_bend_offset(ele%value(g$), ele%value(rho$), &
         ele%value(l$) * 0.5_rp, ele%value(angle$), &
         ele%value(ref_tilt_tot$), ele%value(roll_tot$), &
         ele%value(x_offset_tot$), ele%value(y_offset_tot$), ele%value(z_offset_tot$), &
         ele%value(x_pitch$), ele%value(y_pitch$), 1, n)
  elseif (ele%key == sbend$) then
    call gpu_misalign(ele%value(x_offset_tot$), ele%value(y_offset_tot$), &
                       ele%value(ref_tilt_tot$), 1, n)
  elseif (ele%value(x_pitch_tot$) /= 0 .or. ele%value(y_pitch_tot$) /= 0 .or. &
          ele%value(z_offset_tot$) /= 0) then
    block
      real(rp) :: W_3d(3,3)
      call floor_angles_to_w_mat(ele%value(x_pitch_tot$), ele%value(y_pitch_tot$), &
                                 ele%value(tilt_tot$), w_mat_inv = W_3d)
      call gpu_misalign_3d(W_3d, ele%value(x_offset_tot$), ele%value(y_offset_tot$), &
                           ele%value(z_offset_tot$), 1, n)
    end block
  else
    call gpu_misalign(ele%value(x_offset_tot$), ele%value(y_offset_tot$), &
                       ele%value(tilt_tot$), 1, n)
  endif
endif

! Entrance fringe
call dispatch_fringe_on_device_pub(ele, param, n, first_track_edge$)

! Radiation entrance
apply_rad = gpu_rad_eligible(ele)
if (apply_rad) then
  call ensure_rad_map(ele)
  if (associated(ele%rad_map)) call call_gpu_rad_kick(n, ele%rad_map%rm0)
endif

! Body kernel
call dispatch_body_kernel_pub(bunch, ele, param, n)

! Radiation exit
if (apply_rad .and. associated(ele%rad_map)) call call_gpu_rad_kick(n, ele%rad_map%rm1)

! Exit fringe
call dispatch_fringe_on_device_pub(ele, param, n, second_track_edge$)

! Remove misalignment
if (has_misalign) then
  if (ele%key == sbend$ .and. (ele%value(g$) /= 0 .or. ele%value(ref_tilt_tot$) /= 0 &
                               .or. ele%value(roll$) /= 0)) then
    call gpu_bend_offset(ele%value(g$), ele%value(rho$), &
         ele%value(l$) * 0.5_rp, ele%value(angle$), &
         ele%value(ref_tilt_tot$), ele%value(roll_tot$), &
         ele%value(x_offset_tot$), ele%value(y_offset_tot$), ele%value(z_offset_tot$), &
         ele%value(x_pitch$), ele%value(y_pitch$), -1, n)
  elseif (ele%key == sbend$) then
    call gpu_misalign(ele%value(x_offset_tot$), ele%value(y_offset_tot$), &
                       ele%value(ref_tilt_tot$), -1, n)
  elseif (ele%value(x_pitch_tot$) /= 0 .or. ele%value(y_pitch_tot$) /= 0 .or. &
          ele%value(z_offset_tot$) /= 0) then
    block
      real(rp) :: W_3d(3,3)
      call floor_angles_to_w_mat(ele%value(x_pitch_tot$), ele%value(y_pitch_tot$), &
                                 ele%value(tilt_tot$), w_mat = W_3d)
      call gpu_misalign_3d(W_3d, ele%value(x_offset_tot$), ele%value(y_offset_tot$), &
                           ele%value(z_offset_tot$), -1, n)
    end block
  else
    call gpu_misalign(ele%value(x_offset_tot$), ele%value(y_offset_tot$), &
                       ele%value(tilt_tot$), -1, n)
  endif
endif

! Exit aperture
call dispatch_aperture_on_device_pub(ele, n, exit_end$)

! Update s on device and host
call gpu_s_update(ele%s, n)
gp_s(1:n) = ele%s

! Orbit check
call gpu_orbit_check(n)

did_track = .true.
#endif

contains

subroutine dispatch_aperture_on_device_pub(ele, np, at_edge)
type (ele_struct), intent(in) :: ele
integer(C_INT), intent(in) :: np
integer, intent(in) :: at_edge

if (ele%aperture_at == no_aperture$) return
if (ele%aperture_at == entrance_end$ .and. at_edge /= entrance_end$) return
if (ele%aperture_at == exit_end$ .and. at_edge /= exit_end$) return
if (ele%value(x1_limit$) == 0 .and. ele%value(x2_limit$) == 0 .and. &
    ele%value(y1_limit$) == 0 .and. ele%value(y2_limit$) == 0) return
select case (ele%aperture_type)
case (rectangular$)
  call gpu_check_aperture_rect(ele%value(x1_limit$), ele%value(x2_limit$), &
                                ele%value(y1_limit$), ele%value(y2_limit$), np)
case (elliptical$)
  call gpu_check_aperture_ellipse(ele%value(x1_limit$), ele%value(x2_limit$), &
                                   ele%value(y1_limit$), ele%value(y2_limit$), np)
end select
end subroutine

subroutine dispatch_fringe_on_device_pub(ele, param, np, edge)
use multipole_mod, only: ab_multipole_kicks
type (ele_struct), intent(in) :: ele
type (lat_param_struct), intent(in) :: param
integer(C_INT), intent(in) :: np
integer, intent(in) :: edge
integer :: fringe_type_val
real(rp) :: charge_dir_val

select case (ele%key)
case (quadrupole$)
  fringe_type_val = nint(ele%value(fringe_type$))
  if (fringe_type_val == none$) return
  charge_dir_val = rel_tracking_charge_to_mass(bunch%particle(1), param%particle) * &
                   ele%orientation * bunch%particle(1)%direction
  call gpu_quad_fringe(ele%value(k1$), ele%value(fq1$), ele%value(fq2$), &
                       charge_dir_val, &
                       int(fringe_type_val, C_INT), int(edge, C_INT), &
                       int(bunch%particle(1)%time_dir, C_INT), np)
end select
end subroutine

subroutine dispatch_body_kernel_pub(bunch, ele, param, np)
type (bunch_struct), intent(in) :: bunch
type (ele_struct), target, intent(inout) :: ele
type (lat_param_struct), intent(in) :: param
integer(C_INT), intent(in) :: np

select case (ele%key)
case (drift$, pipe$, monitor$, instrument$, kicker$, hkicker$, vkicker$)
  mc2 = mass_of(bunch%particle(1)%species)
  ele_length = ele%value(l$)
  if (ele_length == 0) return
  call gpu_track_drift_dev_no_s(mc2, ele_length, np)

case (quadrupole$)
  ele_length = ele%value(l$)
  if (ele_length == 0) return
  mc2 = mass_of(bunch%particle(1)%species)
  delta_ref_time = ele%value(delta_ref_time$)
  e_tot_ele = ele%value(e_tot$)
  call multipole_ele_to_ab(ele, .false., ix_mag_max, an, bn, magnetic$, include_kicks$, b1)
  call multipole_ele_to_ab(ele, .false., ix_elec_max, an_elec, bn_elec, electric$)
  has_mag_multipoles = (ix_mag_max > -1)
  length = bunch%particle(1)%time_dir * ele_length
  n_step = 1
  if (has_mag_multipoles .or. ix_elec_max > -1) &
    n_step = max(nint(abs(length) / ele%value(ds_step$)), 1)
  rel_tracking_charge = rel_tracking_charge_to_mass(bunch%particle(1), param%particle)
  charge_dir = rel_tracking_charge * ele%orientation * bunch%particle(1)%direction * bunch%particle(1)%time_dir
  call precompute_multipole_arrays(bunch%particle(1), ele, &
      ix_mag_max, an, bn, ix_elec_max, an_elec, bn_elec, &
      ele_length, n_step, a2_arr, b2_arr, ea2_arr, eb2_arr, cm_arr)
  call gpu_track_quad_dev(mc2, b1, ele_length, delta_ref_time, &
                          e_tot_ele, charge_dir, np, &
                          a2_arr, b2_arr, cm_arr, &
                          int(ix_mag_max, C_INT), int(n_step, C_INT), &
                          ea2_arr, eb2_arr, int(ix_elec_max, C_INT))

case (sextupole$, octupole$, thick_multipole$, elseparator$)
  ele_length = ele%value(l$)
  if (ele_length == 0) return
  mc2 = mass_of(bunch%particle(1)%species)
  delta_ref_time = ele%value(delta_ref_time$)
  e_tot_ele = ele%value(e_tot$)
  call multipole_ele_to_ab(ele, .false., ix_mag_max, an, bn, magnetic$, include_kicks$, b1)
  call multipole_ele_to_ab(ele, .false., ix_elec_max, an_elec, bn_elec, electric$)
  length = bunch%particle(1)%time_dir * ele_length
  n_step = 1
  if (ix_mag_max > -1 .or. ix_elec_max > -1) &
    n_step = max(nint(abs(length) / ele%value(ds_step$)), 1)
  rel_tracking_charge = rel_tracking_charge_to_mass(bunch%particle(1), param%particle)
  charge_dir = rel_tracking_charge * ele%orientation * bunch%particle(1)%direction * bunch%particle(1)%time_dir
  call precompute_multipole_arrays(bunch%particle(1), ele, &
      ix_mag_max, an, bn, ix_elec_max, an_elec, bn_elec, &
      ele_length, n_step, a2_arr, b2_arr, ea2_arr, eb2_arr, cm_arr)
  call gpu_track_sextupole_dev(mc2, ele_length, delta_ref_time, &
                               e_tot_ele, charge_dir, np, &
                               a2_arr, b2_arr, cm_arr, &
                               int(ix_mag_max, C_INT), int(n_step, C_INT), &
                               ea2_arr, eb2_arr, int(ix_elec_max, C_INT))

case (sbend$, rf_bend$)
  ele_length = ele%value(l$)
  if (ele_length == 0) return
  mc2 = mass_of(bunch%particle(1)%species)
  delta_ref_time = ele%value(delta_ref_time$)
  e_tot_ele = ele%value(e_tot$)
  p0c_ele = ele%value(p0c$)
  rel_charge_dir = ele%orientation * bunch%particle(1)%direction * &
                   rel_tracking_charge_to_mass(bunch%particle(1), param%particle)
  c_dir_val_p = ele%orientation * bunch%particle(1)%direction * charge_of(bunch%particle(1)%species)
  is_exact_p = 0; ix_exact_mag_max_p = -1; rho_val_p = 0; exact_f_scale_val_p = 0
  exact_an_arr_p = 0; exact_bn_arr_p = 0
  if (nint(ele%value(exact_multipoles$)) /= off$ .and. ele%value(g$) /= 0) then
    is_exact_p = 1
    call multipole_ele_to_ab(ele, .false., ix_mag_max, an, bn, magnetic$, include_kicks$)
    b1 = 0
    call multipole_ele_to_ab(ele, .false., ix_exact_mag_max_p, exact_an_p, exact_bn_p, magnetic$, include_kicks$)
    if (nint(ele%value(exact_multipoles$)) == horizontally_pure$ .and. ix_exact_mag_max_p /= -1) then
      call convert_bend_exact_multipole(ele%value(g$), vertically_pure$, exact_an_p, exact_bn_p)
      ix_exact_mag_max_p = n_pole_maxx
    endif
    exact_an_arr_p = exact_an_p; exact_bn_arr_p = exact_bn_p
    rho_val_p = ele%value(rho$)
    if (ele%value(l$) /= 0) exact_f_scale_val_p = ele%value(p0c$) / (c_light * charge_of(param%particle) * ele%value(l$))
  else
    call multipole_ele_to_ab(ele, .false., ix_mag_max, an, bn, magnetic$, include_kicks$, b1)
    b1 = b1 * rel_charge_dir
    if (abs(b1) < 1d-10) then; bn(1) = b1; b1 = 0; endif
  endif
  call multipole_ele_to_ab(ele, .false., ix_elec_max, an_elec, bn_elec, electric$)
  g = ele%value(g$)
  length = bunch%particle(1)%time_dir * ele_length
  if (length == 0) then; dg = 0; else; dg = bn(0)/ele_length; bn(0) = 0; endif
  g_tot = (g + dg) * rel_charge_dir
  has_mag_multipoles = (ix_mag_max > -1)
  has_elec_multipoles = (ix_elec_max > -1)
  n_step = 1
  if (has_mag_multipoles .or. has_elec_multipoles) &
    n_step = max(nint(abs(length) / ele%value(ds_step$)), 1)
  call precompute_multipole_arrays(bunch%particle(1), ele, &
      ix_mag_max, an, bn, ix_elec_max, an_elec, bn_elec, &
      ele_length, n_step, a2_arr, b2_arr, ea2_arr, eb2_arr, cm_arr)
  call gpu_track_bend_dev(mc2, g, g_tot, dg, b1, &
                          ele_length, delta_ref_time, e_tot_ele, &
                          rel_charge_dir, p0c_ele, np, &
                          a2_arr, b2_arr, cm_arr, &
                          int(ix_mag_max, C_INT), int(n_step, C_INT), &
                          ea2_arr, eb2_arr, int(ix_elec_max, C_INT), &
                          is_exact_p, exact_an_arr_p, exact_bn_arr_p, &
                          int(ix_exact_mag_max_p, C_INT), &
                          real(rho_val_p, C_DOUBLE), real(c_dir_val_p, C_DOUBLE), &
                          real(exact_f_scale_val_p, C_DOUBLE))

case (lcavity$)
  if (ele%value(l$) == 0) return
  if (ele%value(rf_frequency$) == 0) return
  lord => pointer_to_super_lord(ele)
  if (lord%value(ks$) /= 0) return
  call multipole_ele_to_ab(ele, .false., ix_mag_max, an, bn, magnetic$, include_kicks$)
  call multipole_ele_to_ab(ele, .false., ix_elec_max, an_elec, bn_elec, electric$)
  if (ix_mag_max > -1 .or. ix_elec_max > -1) return
  if (ele%value(coupler_strength$) /= 0) return
  if (nint(lord%value(fringe_type$)) == none$) then
    i_fringe_at = 0
  else
    i_fringe_at = nint(lord%value(fringe_at$))
    if (i_fringe_at < 1 .or. i_fringe_at > 3) i_fringe_at = 0
  endif
  charge_ratio_val = charge_of(bunch%particle(1)%species) / (2.0_rp * charge_of(lord%ref_species))
  mc2 = mass_of(bunch%particle(1)%species)
  n_steps = nint(lord%value(n_rf_steps$))
  phi0_no_multi = lord%value(phi0$) + lord%value(phi0_err$)
  phi0_total = phi0_no_multi + lord%value(phi0_multipass$)
  abs_time_flag = 0
  ref_time_start_val = 0.0_rp
  if (bmad_com%absolute_time_tracking) then
    abs_time_flag = 1
    if (bmad_com%absolute_time_ref_shift) then
    ! For multipass slaves, use the first-pass element's ref_time_start
    ! (matches this_rf_phase in track_a_lcavity.f90)
    if (lord%slave_status == multipass_slave$) then
      block
        type (ele_pointer_struct), allocatable :: mp_chain(:)
        integer :: ix_pass_mp, n_links_mp
        call multipass_chain(lord, ix_pass_mp, n_links_mp, mp_chain)
        ref_time_start_val = mp_chain(1)%ele%value(ref_time_start$)
      end block
    else
      ref_time_start_val = lord%value(ref_time_start$)
    endif
  endif
  endif
  allocate(h_step_s0(n_steps+2), h_step_s(n_steps+2))
  allocate(h_step_p0c(n_steps+2), h_step_p1c(n_steps+2))
  allocate(h_step_scale(n_steps+2), h_step_time(n_steps+2))
  block
    type (ele_struct), pointer :: mlord3
    mlord3 => lord
    if (lord%slave_status == multipass_slave$) mlord3 => pointer_to_lord(lord, 1)
    do j = 0, n_steps + 1
      step => lord%rf%steps(j)
      h_step_s0(j+1) = step%s0; h_step_s(j+1) = step%s
      h_step_p0c(j+1) = step%p0c; h_step_p1c(j+1) = step%p1c
      h_step_scale(j+1) = step%scale; h_step_time(j+1) = mlord3%rf%steps(j)%time
    enddo
  end block
  call gpu_track_lcavity_dev(mc2, h_step_s0, h_step_s, h_step_p0c, h_step_p1c, &
                             h_step_scale, h_step_time, int(n_steps, C_INT), &
                             lord%value(voltage$), lord%value(voltage_err$), &
                             lord%value(field_autoscale$), &
                             ele%value(rf_frequency$), phi0_total, &
                             lord%value(voltage_tot$), lord%value(l_active$), &
                             int(nint(lord%value(cavity_type$)), C_INT), &
                             int(i_fringe_at, C_INT), charge_ratio_val, &
                             int(np, C_INT), int(abs_time_flag, C_INT), phi0_no_multi, &
                             ref_time_start_val)
  deallocate(h_step_s0, h_step_s, h_step_p0c, h_step_p1c, h_step_scale, h_step_time)

case (solenoid$)
  ele_length = ele%value(l$)
  if (ele_length == 0) return
  mc2 = mass_of(bunch%particle(1)%species)
  delta_ref_time = ele%value(delta_ref_time$)
  e_tot_ele = ele%value(e_tot$)
  rel_tracking_charge = rel_tracking_charge_to_mass(bunch%particle(1), param%particle)
  ks0 = rel_tracking_charge * ele%value(bs_field$) * charge_of(bunch%particle(1)%species) * c_light / bunch%particle(1)%p0c
  call multipole_ele_to_ab(ele, .false., ix_mag_max, an, bn, magnetic$, include_kicks$, b1)
  call multipole_ele_to_ab(ele, .false., ix_elec_max, an_elec, bn_elec, electric$)
  length = bunch%particle(1)%time_dir * ele_length
  n_step = 1
  if (ix_mag_max > -1 .or. ix_elec_max > -1) &
    n_step = max(nint(abs(length) / ele%value(ds_step$)), 1)
  call precompute_multipole_arrays(bunch%particle(1), ele, &
      ix_mag_max, an, bn, ix_elec_max, an_elec, bn_elec, &
      ele_length, n_step, a2_arr, b2_arr, ea2_arr, eb2_arr, cm_arr)
  call gpu_track_solenoid_dev(mc2, ks0, ele_length, delta_ref_time, &
                              e_tot_ele, np, int(n_step, C_INT), &
                              a2_arr, b2_arr, cm_arr, &
                              int(ix_mag_max, C_INT), &
                              ea2_arr, eb2_arr, int(ix_elec_max, C_INT))

case (sol_quad$)
  ele_length = ele%value(l$)
  if (ele_length == 0) return
  mc2 = mass_of(bunch%particle(1)%species)
  delta_ref_time = ele%value(delta_ref_time$)
  e_tot_ele = ele%value(e_tot$)
  rel_tracking_charge = rel_tracking_charge_to_mass(bunch%particle(1), param%particle)
  charge_dir = rel_tracking_charge * ele%orientation * bunch%particle(1)%direction * bunch%particle(1)%time_dir
  call multipole_ele_to_ab(ele, .false., ix_mag_max, an, bn, magnetic$, include_kicks$, b1)
  call multipole_ele_to_ab(ele, .false., ix_elec_max, an_elec, bn_elec, electric$)
  length = bunch%particle(1)%time_dir * ele_length
  n_step = 1
  if (ix_mag_max > -1 .or. ix_elec_max > -1) &
    n_step = max(nint(abs(length) / ele%value(ds_step$)), 1)
  call precompute_multipole_arrays(bunch%particle(1), ele, &
      ix_mag_max, an, bn, ix_elec_max, an_elec, bn_elec, &
      ele_length, n_step, a2_arr, b2_arr, ea2_arr, eb2_arr, cm_arr)
  if (b1 == 0) then
    ks0 = rel_tracking_charge * ele%value(bs_field$) * charge_of(bunch%particle(1)%species) * c_light / bunch%particle(1)%p0c
    call gpu_track_solenoid_dev(mc2, ks0, ele_length, delta_ref_time, &
                                e_tot_ele, np, int(n_step, C_INT), &
                                a2_arr, b2_arr, cm_arr, &
                                int(ix_mag_max, C_INT), &
                                ea2_arr, eb2_arr, int(ix_elec_max, C_INT))
  else
    ks_val = rel_tracking_charge * ele%value(ks$)
    k1_val = charge_dir * b1 / ele_length
    call gpu_track_sol_quad_dev(mc2, ks_val, k1_val, ele_length, delta_ref_time, &
                                e_tot_ele, np, int(n_step, C_INT), &
                                a2_arr, b2_arr, cm_arr, &
                                int(ix_mag_max, C_INT), &
                                ea2_arr, eb2_arr, int(ix_elec_max, C_INT))
  endif

case (wiggler$, undulator$)
  ele_length = ele%value(l$)
  if (ele_length == 0) return
  mc2 = mass_of(bunch%particle(1)%species)
  delta_ref_time = ele%value(delta_ref_time$)
  e_tot_ele = ele%value(e_tot$)
  block
    type (ele_struct), pointer :: field_ele_p
    real(rp) :: k1x_p, k1y_p, kz_p, factor_p, ky2_p, osc_amp_p, p0c_ele_p
    integer(C_INT) :: is_helical_p

    p0c_ele_p = ele%value(p0c$)
    osc_amp_p = ele%value(osc_amplitude$)
    field_ele_p => pointer_to_field_ele(ele, 1)

    if (ele%value(l_period$) == 0) then
      kz_p = 1d100
      ky2_p = 0
    else
      kz_p = twopi / ele%value(l_period$)
      ky2_p = kz_p**2 + ele%value(kx$)**2
    endif

    rel_tracking_charge = rel_tracking_charge_to_mass(bunch%particle(1), param%particle)
    factor_p = abs(rel_tracking_charge) * 0.5_rp * (c_light * ele%value(b_max$) / ele%value(p0c$))**2

    if (field_ele_p%field_calc == helical_model$) then
      k1x_p = -factor_p
      k1y_p = -factor_p
      is_helical_p = 1
    else
      k1x_p =  factor_p * (ele%value(kx$) / kz_p)**2
      k1y_p = -factor_p * ky2_p / kz_p**2
      is_helical_p = 0
    endif

    call multipole_ele_to_ab(ele, .false., ix_mag_max, an, bn, magnetic$, include_kicks$)
    call multipole_ele_to_ab(ele, .false., ix_elec_max, an_elec, bn_elec, electric$)

    n_step = max(nint(ele%value(l$) / ele%value(ds_step$)), 1)
    if (ix_mag_max < 0 .and. ix_elec_max < 0) n_step = 1

    call precompute_multipole_arrays(bunch%particle(1), ele, &
        ix_mag_max, an, bn, ix_elec_max, an_elec, bn_elec, &
        ele_length, n_step, a2_arr, b2_arr, ea2_arr, eb2_arr, cm_arr)

    call gpu_track_wiggler_dev(mc2, ele_length, delta_ref_time, &
                                e_tot_ele, p0c_ele_p, &
                                k1x_p, k1y_p, kz_p, is_helical_p, &
                                osc_amp_p, &
                                np, int(n_step, C_INT), &
                                a2_arr, b2_arr, cm_arr, &
                                int(ix_mag_max, C_INT), &
                                ea2_arr, eb2_arr, int(ix_elec_max, C_INT))
  end block
end select
end subroutine

end subroutine gpu_persistent_track_element

!------------------------------------------------------------------------
! gpu_persistent_flush — download particle data from device to bunch
!
! Must be called before any CPU-side access to bunch particles,
! and when transitioning to a non-GPU element.
!------------------------------------------------------------------------
subroutine gpu_persistent_flush(bunch, ele)

use, intrinsic :: iso_c_binding

type (bunch_struct), intent(inout) :: bunch
type (ele_struct),   intent(in)    :: ele
integer :: j, n

if (.not. gpu_persist_on_device) return

n = size(bunch%particle)
call gpu_download_particles(gp_vx, gp_vpx, gp_vy, gp_vpy, gp_vz, gp_vpz, &
                            gp_state, gp_beta, gp_p0c, gp_t, n, 1, 1)
call soa_to_bunch(bunch, ele, n, gp_vx, gp_vpx, gp_vy, gp_vpy, gp_vz, gp_vpz, &
                   gp_state, gp_beta, gp_p0c, gp_t, .true., .true.)
do j = 1, n
  bunch%particle(j)%s = gp_s(j)
enddo
bunch%charge_live = sum(bunch%particle(:)%charge, mask = (bunch%particle(:)%state == alive$))
gpu_persist_on_device = .false.

! Invalidate the saved multi-bunch slot for this bunch so stale data isn't restored
block
  integer :: ib
  integer(8) :: bid
  bid = gpu_persist_bunch_id
  do ib = 1, MAX_GPU_BUNCHES
    if (gpu_bunch_slots(ib)%valid .and. gpu_bunch_slots(ib)%bunch_id == bid) then
      gpu_bunch_slots(ib)%valid = .false.
      exit
    endif
  enddo
end block

end subroutine gpu_persistent_flush

!------------------------------------------------------------------------
! gpu_persistent_seed — upload bunch data to device after per-element
! GPU tracking, so the next element can use the persistent path.
!------------------------------------------------------------------------
subroutine gpu_persistent_seed(bunch, ele, force)

use, intrinsic :: iso_c_binding

type (bunch_struct), intent(in) :: bunch
type (ele_struct),   intent(in) :: ele
logical, optional, intent(in) :: force
integer :: j, n

#ifdef USE_GPU_TRACKING
if (.not. logic_option(.false., force)) then
  if (.not. ele_gpu_can_stay_on_device(ele)) return
endif

n = size(bunch%particle)

! Ensure persistent SoA buffers
if (n > gpu_persist_n) then
  if (allocated(gp_vx)) deallocate(gp_vx, gp_vpx, gp_vy, gp_vpy, gp_vz, gp_vpz, &
                                    gp_state, gp_beta, gp_p0c, gp_t, gp_s)
  allocate(gp_vx(n), gp_vpx(n), gp_vy(n), gp_vpy(n), gp_vz(n), gp_vpz(n))
  allocate(gp_state(n), gp_beta(n), gp_p0c(n), gp_t(n), gp_s(n))
  gpu_persist_n = n
endif

call bunch_to_soa(bunch, n, gp_vx, gp_vpx, gp_vy, gp_vpy, gp_vz, gp_vpz, &
                   gp_state, gp_beta, gp_p0c, gp_t)
do j = 1, n
  gp_s(j) = bunch%particle(j)%s
enddo
call gpu_upload_particles(gp_vx, gp_vpx, gp_vy, gp_vpy, gp_vz, gp_vpz, &
                          gp_state, gp_beta, gp_p0c, gp_t, n)
gpu_persist_on_device = .true.
gpu_persist_bunch_id = transfer(loc(bunch%particle(1)%vec(1)), gpu_persist_bunch_id)
#endif

end subroutine gpu_persistent_seed

!------------------------------------------------------------------------
! gpu_multi_bunch_save — save current device buffers to a per-bunch slot
!
! Called when switching away from a bunch whose data is on device.
! The device data is downloaded to a host-side slot indexed by bunch_id.
!------------------------------------------------------------------------
subroutine gpu_multi_bunch_save(bunch_id, n_part)

use, intrinsic :: iso_c_binding

integer(8), intent(in) :: bunch_id
integer, intent(in) :: n_part

#ifdef USE_GPU_TRACKING
integer :: slot, i, oldest_slot
integer(C_INT) :: rc

! Find existing slot for this bunch_id, or allocate a new one
slot = 0
do i = 1, MAX_GPU_BUNCHES
  if (gpu_bunch_slots(i)%valid .and. gpu_bunch_slots(i)%bunch_id == bunch_id) then
    slot = i
    exit
  endif
enddo

! No existing slot found — find an empty one
if (slot == 0) then
  do i = 1, MAX_GPU_BUNCHES
    if (.not. gpu_bunch_slots(i)%valid) then
      slot = i
      exit
    endif
  enddo
endif

! All slots full — reuse slot 1 (oldest)
if (slot == 0) slot = 1

! Ensure slot arrays are large enough
if (.not. allocated(gpu_bunch_slots(slot)%vx) .or. &
    size(gpu_bunch_slots(slot)%vx) < n_part) then
  if (allocated(gpu_bunch_slots(slot)%vx)) then
    deallocate(gpu_bunch_slots(slot)%vx, gpu_bunch_slots(slot)%vpx)
    deallocate(gpu_bunch_slots(slot)%vy, gpu_bunch_slots(slot)%vpy)
    deallocate(gpu_bunch_slots(slot)%vz, gpu_bunch_slots(slot)%vpz)
    deallocate(gpu_bunch_slots(slot)%state)
    deallocate(gpu_bunch_slots(slot)%beta, gpu_bunch_slots(slot)%p0c)
    deallocate(gpu_bunch_slots(slot)%t, gpu_bunch_slots(slot)%s)
  endif
  allocate(gpu_bunch_slots(slot)%vx(n_part), gpu_bunch_slots(slot)%vpx(n_part))
  allocate(gpu_bunch_slots(slot)%vy(n_part), gpu_bunch_slots(slot)%vpy(n_part))
  allocate(gpu_bunch_slots(slot)%vz(n_part), gpu_bunch_slots(slot)%vpz(n_part))
  allocate(gpu_bunch_slots(slot)%state(n_part))
  allocate(gpu_bunch_slots(slot)%beta(n_part), gpu_bunch_slots(slot)%p0c(n_part))
  allocate(gpu_bunch_slots(slot)%t(n_part), gpu_bunch_slots(slot)%s(n_part))
endif

! Download device buffers into the slot
rc = gpu_save_bunch_buffers( &
      gpu_bunch_slots(slot)%vx, gpu_bunch_slots(slot)%vpx, &
      gpu_bunch_slots(slot)%vy, gpu_bunch_slots(slot)%vpy, &
      gpu_bunch_slots(slot)%vz, gpu_bunch_slots(slot)%vpz, &
      gpu_bunch_slots(slot)%state, &
      gpu_bunch_slots(slot)%beta, gpu_bunch_slots(slot)%p0c, &
      gpu_bunch_slots(slot)%t, gpu_bunch_slots(slot)%s, &
      int(n_part, C_INT))

if (rc == 0) then
  gpu_bunch_slots(slot)%valid = .true.
  gpu_bunch_slots(slot)%n_particles = n_part
  gpu_bunch_slots(slot)%bunch_id = bunch_id
  gpu_active_slot = slot
endif
#endif

end subroutine gpu_multi_bunch_save

!------------------------------------------------------------------------
! gpu_multi_bunch_restore — restore a previously saved bunch to device
!
! Called when switching to a bunch that may have saved device state.
! Sets was_restored=.true. if the bunch was found and restored, so the
! caller knows data is on device and doesn't need a fresh upload.
!------------------------------------------------------------------------
subroutine gpu_multi_bunch_restore(bunch_id, n_part, was_restored)

use, intrinsic :: iso_c_binding

integer(8), intent(in) :: bunch_id
integer, intent(in) :: n_part
logical, intent(out) :: was_restored

#ifdef USE_GPU_TRACKING
integer :: i
integer(C_INT) :: rc

was_restored = .false.

do i = 1, MAX_GPU_BUNCHES
  if (gpu_bunch_slots(i)%valid .and. gpu_bunch_slots(i)%bunch_id == bunch_id .and. &
      gpu_bunch_slots(i)%n_particles == n_part) then
    ! Found saved data — upload it back to device
    rc = gpu_restore_bunch_buffers( &
          gpu_bunch_slots(i)%vx, gpu_bunch_slots(i)%vpx, &
          gpu_bunch_slots(i)%vy, gpu_bunch_slots(i)%vpy, &
          gpu_bunch_slots(i)%vz, gpu_bunch_slots(i)%vpz, &
          gpu_bunch_slots(i)%state, &
          gpu_bunch_slots(i)%beta, gpu_bunch_slots(i)%p0c, &
          gpu_bunch_slots(i)%t, gpu_bunch_slots(i)%s, &
          int(n_part, C_INT))
    if (rc == 0) then
      was_restored = .true.
      gpu_active_slot = i
      ! Also update the module-level SoA arrays so they stay consistent
      ! (flush uses gp_vx etc. for the D->H transfer target)
      if (n_part > gpu_persist_n) then
        if (allocated(gp_vx)) deallocate(gp_vx, gp_vpx, gp_vy, gp_vpy, gp_vz, gp_vpz, &
                                          gp_state, gp_beta, gp_p0c, gp_t, gp_s)
        allocate(gp_vx(n_part), gp_vpx(n_part), gp_vy(n_part), gp_vpy(n_part))
        allocate(gp_vz(n_part), gp_vpz(n_part))
        allocate(gp_state(n_part), gp_beta(n_part), gp_p0c(n_part), gp_t(n_part), gp_s(n_part))
        gpu_persist_n = n_part
      endif
      ! Copy the slot's s array to gp_s for consistency with gpu_persistent_flush
      gp_s(1:n_part) = gpu_bunch_slots(i)%s(1:n_part)
    endif
    return
  endif
enddo

! Not found — caller will do a fresh upload
was_restored = .false.
#else
was_restored = .false.
#endif

end subroutine gpu_multi_bunch_restore

!------------------------------------------------------------------------
! gpu_multi_bunch_cleanup — invalidate all saved bunch slots
!
! Should be called when the beam is re-initialized or when tracking
! is complete, to free memory and prevent stale data.
!------------------------------------------------------------------------
subroutine gpu_multi_bunch_cleanup()

integer :: i

do i = 1, MAX_GPU_BUNCHES
  gpu_bunch_slots(i)%valid = .false.
  gpu_bunch_slots(i)%n_particles = 0
  gpu_bunch_slots(i)%bunch_id = 0
  if (allocated(gpu_bunch_slots(i)%vx)) then
    deallocate(gpu_bunch_slots(i)%vx, gpu_bunch_slots(i)%vpx)
    deallocate(gpu_bunch_slots(i)%vy, gpu_bunch_slots(i)%vpy)
    deallocate(gpu_bunch_slots(i)%vz, gpu_bunch_slots(i)%vpz)
    deallocate(gpu_bunch_slots(i)%state)
    deallocate(gpu_bunch_slots(i)%beta, gpu_bunch_slots(i)%p0c)
    deallocate(gpu_bunch_slots(i)%t, gpu_bunch_slots(i)%s)
  endif
enddo
gpu_active_slot = 0

end subroutine gpu_multi_bunch_cleanup

!------------------------------------------------------------------------
! gpu_track_body_on_device — run ONLY the body kernel for an element
!
! Assumes data is already on device. Does NOT apply misalignment, fringe,
! aperture, or radiation. Used by track1_bunch_csr for sub-element tracking
! where the CSR loop handles its own fringe/misalignment.
!------------------------------------------------------------------------
subroutine gpu_track_body_on_device(bunch, ele, param, did_track)

use multipole_mod, only: ab_multipole_kicks, multipole_ele_to_ab
use, intrinsic :: iso_c_binding

type (bunch_struct),     intent(inout) :: bunch
type (ele_struct), target, intent(inout) :: ele
type (lat_param_struct), intent(in)    :: param
logical,                 intent(out)   :: did_track

#ifdef USE_GPU_TRACKING
integer(C_INT) :: n
integer :: ix_mag_max, ix_elec_max, n_step
real(rp) :: mc2, ele_length, b1, delta_ref_time, e_tot_ele, p0c_ele
real(rp) :: charge_dir, rel_tracking_charge, length, g, g_tot, dg, rel_charge_dir
real(rp) :: ks0_body, ks_val_body, k1_val_body
real(rp) :: an(0:n_pole_maxx), bn(0:n_pole_maxx)
real(rp) :: an_elec(0:n_pole_maxx), bn_elec(0:n_pole_maxx)
real(C_DOUBLE) :: a2_arr(0:n_pole_maxx), b2_arr(0:n_pole_maxx)
real(C_DOUBLE) :: ea2_arr(0:n_pole_maxx), eb2_arr(0:n_pole_maxx)
real(C_DOUBLE) :: cm_arr(0:n_pole_maxx, 0:n_pole_maxx)
logical :: has_mag_multipoles, has_elec_multipoles
! Wiggler scratch
real(rp) :: k1x_wig, k1y_wig, kz_wig, ky2_wig, factor_wig, osc_amp_wig, p0c_ele_wig
integer(C_INT) :: is_helical_wig
type (ele_struct), pointer :: field_ele_wig
! Exact bend multipole scratch
integer :: ix_exact_mag_max
integer(C_INT) :: is_exact
real(rp) :: c_dir_val, rho_val, exact_f_scale_val
real(rp) :: exact_an(0:n_pole_maxx), exact_bn(0:n_pole_maxx)
real(C_DOUBLE) :: exact_an_arr(0:n_pole_maxx), exact_bn_arr(0:n_pole_maxx)
#endif

did_track = .false.

#ifdef USE_GPU_TRACKING
if (.not. gpu_persist_on_device) return

n = size(bunch%particle)
if (n == 0) return

select case (ele%key)
case (drift$, pipe$, monitor$, instrument$, kicker$, hkicker$, vkicker$)
  mc2 = mass_of(bunch%particle(1)%species)
  ele_length = ele%value(l$)
  if (ele_length == 0) then; did_track = .true.; return; endif
  call gpu_track_drift_dev_no_s(mc2, ele_length, n)
  did_track = .true.

case (quadrupole$)
  ele_length = ele%value(l$)
  if (ele_length == 0) then; did_track = .true.; return; endif
  mc2 = mass_of(bunch%particle(1)%species)
  delta_ref_time = ele%value(delta_ref_time$)
  e_tot_ele = ele%value(e_tot$)
  call multipole_ele_to_ab(ele, .false., ix_mag_max, an, bn, magnetic$, include_kicks$, b1)
  call multipole_ele_to_ab(ele, .false., ix_elec_max, an_elec, bn_elec, electric$)
  length = bunch%particle(1)%time_dir * ele_length
  n_step = 1
  if (ix_mag_max > -1 .or. ix_elec_max > -1) &
    n_step = max(nint(abs(length) / ele%value(ds_step$)), 1)
  rel_tracking_charge = rel_tracking_charge_to_mass(bunch%particle(1), param%particle)
  charge_dir = rel_tracking_charge * ele%orientation * bunch%particle(1)%direction * bunch%particle(1)%time_dir
  call precompute_multipole_arrays(bunch%particle(1), ele, &
      ix_mag_max, an, bn, ix_elec_max, an_elec, bn_elec, &
      ele_length, n_step, a2_arr, b2_arr, ea2_arr, eb2_arr, cm_arr)
  call gpu_track_quad_dev(mc2, b1, ele_length, delta_ref_time, &
                          e_tot_ele, charge_dir, n, &
                          a2_arr, b2_arr, cm_arr, &
                          int(ix_mag_max, C_INT), int(n_step, C_INT), &
                          ea2_arr, eb2_arr, int(ix_elec_max, C_INT))
  did_track = .true.

case (sextupole$, octupole$, thick_multipole$, elseparator$)
  ele_length = ele%value(l$)
  if (ele_length == 0) then; did_track = .true.; return; endif
  mc2 = mass_of(bunch%particle(1)%species)
  delta_ref_time = ele%value(delta_ref_time$)
  e_tot_ele = ele%value(e_tot$)
  call multipole_ele_to_ab(ele, .false., ix_mag_max, an, bn, magnetic$, include_kicks$, b1)
  call multipole_ele_to_ab(ele, .false., ix_elec_max, an_elec, bn_elec, electric$)
  length = bunch%particle(1)%time_dir * ele_length
  n_step = 1
  if (ix_mag_max > -1 .or. ix_elec_max > -1) &
    n_step = max(nint(abs(length) / ele%value(ds_step$)), 1)
  rel_tracking_charge = rel_tracking_charge_to_mass(bunch%particle(1), param%particle)
  charge_dir = rel_tracking_charge * ele%orientation * bunch%particle(1)%direction * bunch%particle(1)%time_dir
  call precompute_multipole_arrays(bunch%particle(1), ele, &
      ix_mag_max, an, bn, ix_elec_max, an_elec, bn_elec, &
      ele_length, n_step, a2_arr, b2_arr, ea2_arr, eb2_arr, cm_arr)
  call gpu_track_sextupole_dev(mc2, ele_length, delta_ref_time, &
                               e_tot_ele, charge_dir, n, &
                               a2_arr, b2_arr, cm_arr, &
                               int(ix_mag_max, C_INT), int(n_step, C_INT), &
                               ea2_arr, eb2_arr, int(ix_elec_max, C_INT))
  did_track = .true.

case (sbend$, rf_bend$)
  ele_length = ele%value(l$)
  if (ele_length == 0) then; did_track = .true.; return; endif
  ! Cache multipole setup across CSR sub-steps for the same element.
  ! Only ele_length changes between sub-steps; all coefficients are the same.
  block
    integer, save :: cached_bend_ix = -1
    real(rp), save :: cached_mc2, cached_drt, cached_etot, cached_p0c
    real(rp), save :: cached_rcd, cached_g, cached_g_tot, cached_dg, cached_b1
    real(rp), save :: cached_c_dir, cached_rho, cached_efs
    integer(C_INT), save :: cached_is_exact, cached_ix_emm
    integer, save :: cached_ix_mm, cached_ix_em
    real(C_DOUBLE), save :: ca2(0:n_pole_maxx), cb2(0:n_pole_maxx)
    real(C_DOUBLE), save :: cea2(0:n_pole_maxx), ceb2(0:n_pole_maxx)
    real(C_DOUBLE), save :: ccm(0:n_pole_maxx, 0:n_pole_maxx)
    real(C_DOUBLE), save :: cexa(0:n_pole_maxx), cexb(0:n_pole_maxx)
    logical, save :: cached_has_mag, cached_has_elec

    if (ele%ix_ele /= cached_bend_ix) then
      ! Cache miss: recompute everything
      cached_bend_ix = ele%ix_ele
      cached_mc2 = mass_of(bunch%particle(1)%species)
      cached_drt = ele%value(delta_ref_time$)
      cached_etot = ele%value(e_tot$)
      cached_p0c = ele%value(p0c$)
      cached_rcd = ele%orientation * bunch%particle(1)%direction * &
                   rel_tracking_charge_to_mass(bunch%particle(1), param%particle)
      cached_c_dir = ele%orientation * bunch%particle(1)%direction * charge_of(bunch%particle(1)%species)
      cached_is_exact = 0; cached_ix_emm = -1; cached_rho = 0; cached_efs = 0
      cexa = 0; cexb = 0
      if (nint(ele%value(exact_multipoles$)) /= off$ .and. ele%value(g$) /= 0) then
        cached_is_exact = 1
        call multipole_ele_to_ab(ele, .false., ix_mag_max, an, bn, magnetic$, include_kicks$)
        b1 = 0
        call multipole_ele_to_ab(ele, .false., ix_exact_mag_max, exact_an, exact_bn, magnetic$, include_kicks$)
        if (nint(ele%value(exact_multipoles$)) == horizontally_pure$ .and. ix_exact_mag_max /= -1) then
          call convert_bend_exact_multipole(ele%value(g$), vertically_pure$, exact_an, exact_bn)
          ix_exact_mag_max = n_pole_maxx
        endif
        cexa = exact_an; cexb = exact_bn
        cached_rho = ele%value(rho$)
        if (ele%value(l$) /= 0) cached_efs = ele%value(p0c$) / (c_light * charge_of(param%particle) * ele%value(l$))
        cached_ix_emm = ix_exact_mag_max
      else
        call multipole_ele_to_ab(ele, .false., ix_mag_max, an, bn, magnetic$, include_kicks$, b1)
        b1 = b1 * cached_rcd
        if (abs(b1) < 1d-10) then; bn(1) = b1; b1 = 0; endif
      endif
      call multipole_ele_to_ab(ele, .false., ix_elec_max, an_elec, bn_elec, electric$)
      cached_g = ele%value(g$)
      length = bunch%particle(1)%time_dir * ele_length
      if (length == 0) then; dg = 0; else; dg = bn(0)/ele_length; bn(0) = 0; endif
      cached_g_tot = (cached_g + dg) * cached_rcd
      cached_dg = dg
      cached_b1 = b1
      cached_has_mag = (ix_mag_max > -1)
      cached_has_elec = (ix_elec_max > -1)
      cached_ix_mm = ix_mag_max
      cached_ix_em = ix_elec_max
      n_step = 1
      if (cached_has_mag .or. cached_has_elec) &
        n_step = max(nint(abs(length) / ele%value(ds_step$)), 1)
      call precompute_multipole_arrays(bunch%particle(1), ele, &
          ix_mag_max, an, bn, ix_elec_max, an_elec, bn_elec, &
          ele_length, n_step, ca2, cb2, cea2, ceb2, ccm)
    else
      ! Cache hit: only recompute n_step for the new element length
      length = bunch%particle(1)%time_dir * ele_length
      n_step = 1
      if (cached_has_mag .or. cached_has_elec) &
        n_step = max(nint(abs(length) / ele%value(ds_step$)), 1)
    endif

    call gpu_track_bend_dev(cached_mc2, cached_g, cached_g_tot, cached_dg, cached_b1, &
                            ele_length, cached_drt, cached_etot, &
                            cached_rcd, cached_p0c, n, &
                            ca2, cb2, ccm, &
                            int(cached_ix_mm, C_INT), int(n_step, C_INT), &
                            cea2, ceb2, int(cached_ix_em, C_INT), &
                            cached_is_exact, cexa, cexb, &
                            int(cached_ix_emm, C_INT), &
                            real(cached_rho, C_DOUBLE), real(cached_c_dir, C_DOUBLE), &
                            real(cached_efs, C_DOUBLE))
  end block
  did_track = .true.

case (solenoid$)
  ele_length = ele%value(l$)
  if (ele_length == 0) then; did_track = .true.; return; endif
  mc2 = mass_of(bunch%particle(1)%species)
  delta_ref_time = ele%value(delta_ref_time$)
  e_tot_ele = ele%value(e_tot$)
  rel_tracking_charge = rel_tracking_charge_to_mass(bunch%particle(1), param%particle)
  ks0_body = rel_tracking_charge * ele%value(bs_field$) * charge_of(bunch%particle(1)%species) * c_light / bunch%particle(1)%p0c
  call multipole_ele_to_ab(ele, .false., ix_mag_max, an, bn, magnetic$, include_kicks$, b1)
  call multipole_ele_to_ab(ele, .false., ix_elec_max, an_elec, bn_elec, electric$)
  length = bunch%particle(1)%time_dir * ele_length
  n_step = 1
  if (ix_mag_max > -1 .or. ix_elec_max > -1) &
    n_step = max(nint(abs(length) / ele%value(ds_step$)), 1)
  call precompute_multipole_arrays(bunch%particle(1), ele, &
      ix_mag_max, an, bn, ix_elec_max, an_elec, bn_elec, &
      ele_length, n_step, a2_arr, b2_arr, ea2_arr, eb2_arr, cm_arr)
  call gpu_track_solenoid_dev(mc2, ks0_body, ele_length, delta_ref_time, &
                              e_tot_ele, n, int(n_step, C_INT), &
                              a2_arr, b2_arr, cm_arr, &
                              int(ix_mag_max, C_INT), &
                              ea2_arr, eb2_arr, int(ix_elec_max, C_INT))
  did_track = .true.

case (sol_quad$)
  ele_length = ele%value(l$)
  if (ele_length == 0) then; did_track = .true.; return; endif
  mc2 = mass_of(bunch%particle(1)%species)
  delta_ref_time = ele%value(delta_ref_time$)
  e_tot_ele = ele%value(e_tot$)
  rel_tracking_charge = rel_tracking_charge_to_mass(bunch%particle(1), param%particle)
  charge_dir = rel_tracking_charge * ele%orientation * bunch%particle(1)%direction * bunch%particle(1)%time_dir
  call multipole_ele_to_ab(ele, .false., ix_mag_max, an, bn, magnetic$, include_kicks$, b1)
  call multipole_ele_to_ab(ele, .false., ix_elec_max, an_elec, bn_elec, electric$)
  length = bunch%particle(1)%time_dir * ele_length
  n_step = 1
  if (ix_mag_max > -1 .or. ix_elec_max > -1) &
    n_step = max(nint(abs(length) / ele%value(ds_step$)), 1)
  call precompute_multipole_arrays(bunch%particle(1), ele, &
      ix_mag_max, an, bn, ix_elec_max, an_elec, bn_elec, &
      ele_length, n_step, a2_arr, b2_arr, ea2_arr, eb2_arr, cm_arr)
  if (b1 == 0) then
    ks0_body = rel_tracking_charge * ele%value(bs_field$) * charge_of(bunch%particle(1)%species) * c_light / bunch%particle(1)%p0c
    call gpu_track_solenoid_dev(mc2, ks0_body, ele_length, delta_ref_time, &
                                e_tot_ele, n, int(n_step, C_INT), &
                                a2_arr, b2_arr, cm_arr, &
                                int(ix_mag_max, C_INT), &
                                ea2_arr, eb2_arr, int(ix_elec_max, C_INT))
  else
    ks_val_body = rel_tracking_charge * ele%value(ks$)
    k1_val_body = charge_dir * b1 / ele_length
    call gpu_track_sol_quad_dev(mc2, ks_val_body, k1_val_body, ele_length, delta_ref_time, &
                                e_tot_ele, n, int(n_step, C_INT), &
                                a2_arr, b2_arr, cm_arr, &
                                int(ix_mag_max, C_INT), &
                                ea2_arr, eb2_arr, int(ix_elec_max, C_INT))
  endif
  did_track = .true.

case (wiggler$, undulator$)
  ele_length = ele%value(l$)
  if (ele_length == 0) then; did_track = .true.; return; endif
  mc2 = mass_of(bunch%particle(1)%species)
  delta_ref_time = ele%value(delta_ref_time$)
  e_tot_ele = ele%value(e_tot$)
  p0c_ele_wig = ele%value(p0c$)
  osc_amp_wig = ele%value(osc_amplitude$)

  field_ele_wig => pointer_to_field_ele(ele, 1)

  if (ele%value(l_period$) == 0) then
    kz_wig = 1d100
    ky2_wig = 0
  else
    kz_wig = twopi / ele%value(l_period$)
    ky2_wig = kz_wig**2 + ele%value(kx$)**2
  endif

  rel_tracking_charge = rel_tracking_charge_to_mass(bunch%particle(1), param%particle)
  factor_wig = abs(rel_tracking_charge) * 0.5_rp * (c_light * ele%value(b_max$) / ele%value(p0c$))**2

  if (field_ele_wig%field_calc == helical_model$) then
    k1x_wig = -factor_wig
    k1y_wig = -factor_wig
    is_helical_wig = 1
  else
    k1x_wig =  factor_wig * (ele%value(kx$) / kz_wig)**2
    k1y_wig = -factor_wig * ky2_wig / kz_wig**2
    is_helical_wig = 0
  endif

  call multipole_ele_to_ab(ele, .false., ix_mag_max, an, bn, magnetic$, include_kicks$)
  call multipole_ele_to_ab(ele, .false., ix_elec_max, an_elec, bn_elec, electric$)

  n_step = max(nint(ele%value(l$) / ele%value(ds_step$)), 1)
  if (ix_mag_max < 0 .and. ix_elec_max < 0) n_step = 1

  call precompute_multipole_arrays(bunch%particle(1), ele, &
      ix_mag_max, an, bn, ix_elec_max, an_elec, bn_elec, &
      ele_length, n_step, a2_arr, b2_arr, ea2_arr, eb2_arr, cm_arr)

  call gpu_track_wiggler_dev(mc2, ele_length, delta_ref_time, &
                              e_tot_ele, p0c_ele_wig, &
                              k1x_wig, k1y_wig, kz_wig, is_helical_wig, &
                              osc_amp_wig, &
                              n, int(n_step, C_INT), &
                              a2_arr, b2_arr, cm_arr, &
                              int(ix_mag_max, C_INT), &
                              ea2_arr, eb2_arr, int(ix_elec_max, C_INT))
  did_track = .true.
end select

! Update s on device
if (did_track) call gpu_s_update(ele%s, n)
#endif

end subroutine gpu_track_body_on_device

!------------------------------------------------------------------------
! gpu_apply_fringe_on_device — apply fringe kicks on device (public wrapper)
!
! For use in the CSR sub-step loop to apply entrance/exit fringe
! without flushing particles to CPU.
!------------------------------------------------------------------------
subroutine gpu_apply_fringe_on_device(bunch, ele, param, edge)

use multipole_mod, only: ab_multipole_kicks
use, intrinsic :: iso_c_binding

type (bunch_struct),     intent(inout) :: bunch
type (ele_struct),       intent(in)    :: ele
type (lat_param_struct), intent(in)    :: param
integer,                 intent(in)    :: edge

#ifdef USE_GPU_TRACKING
integer(C_INT) :: np
integer :: fringe_type_val, physical_end_val
real(rp) :: charge_dir_val
real(C_DOUBLE) :: bp_arr(8), ap_arr(8)

np = size(bunch%particle)
if (np == 0) return
if (.not. gpu_persist_on_device) return

select case (ele%key)
case (quadrupole$)
  fringe_type_val = nint(ele%value(fringe_type$))
  if (fringe_type_val == none$) return
  charge_dir_val = rel_tracking_charge_to_mass(bunch%particle(1), param%particle) * &
                   ele%orientation * bunch%particle(1)%direction
  call gpu_quad_fringe(ele%value(k1$), ele%value(fq1$), ele%value(fq2$), &
                       charge_dir_val, &
                       int(fringe_type_val, C_INT), int(edge, C_INT), &
                       int(bunch%particle(1)%time_dir, C_INT), np)

case (sbend$, rf_bend$)
  fringe_type_val = nint(ele%value(fringe_type$))
  if (fringe_type_val == none$) return

  if (fringe_type_val == full$) then
    block
      real(rp) :: g_tot_ex, beta0_ex, e_ang_ex, fint_ex, hgap_ex
      integer :: is_exit_ex, phys_end_ex

      if (ele%is_on) then
        g_tot_ex = ele%value(g$) + ele%value(dg$)
      else
        g_tot_ex = 0
      endif
      beta0_ex = ele%value(p0c$) / ele%value(e_tot$)
      phys_end_ex = physical_ele_end(edge, bunch%particle(1), ele%orientation)

      ! Hard multipole edge kick at entrance
      if (phys_end_ex == entrance_end$) then
        charge_dir_val = rel_tracking_charge_to_mass(bunch%particle(1), param%particle) * &
                         ele%orientation * bunch%particle(1)%direction
        bp_arr = 0; ap_arr = 0; bp_arr(1) = ele%value(k1$)
        call gpu_hard_multipole_edge(bp_arr, ap_arr, int(1, C_INT), charge_dir_val, int(1, C_INT), np)
      endif

      if (phys_end_ex == entrance_end$) then
        e_ang_ex = bunch%particle(1)%time_dir * ele%value(e1$)
        fint_ex = bunch%particle(1)%time_dir * ele%value(fint$)
        hgap_ex = ele%value(hgap$); is_exit_ex = 0
      else
        e_ang_ex = bunch%particle(1)%time_dir * ele%value(e2$)
        fint_ex = bunch%particle(1)%time_dir * ele%value(fintx$)
        hgap_ex = ele%value(hgapx$); is_exit_ex = 1
      endif

      call gpu_exact_bend_fringe(g_tot_ex, beta0_ex, e_ang_ex, fint_ex, hgap_ex, &
          int(is_exit_ex, C_INT), np)

      ! Hard multipole edge kick at exit
      if (phys_end_ex == exit_end$) then
        charge_dir_val = rel_tracking_charge_to_mass(bunch%particle(1), param%particle) * &
                         ele%orientation * bunch%particle(1)%direction
        bp_arr = 0; ap_arr = 0; bp_arr(1) = ele%value(k1$)
        call gpu_hard_multipole_edge(bp_arr, ap_arr, int(1, C_INT), charge_dir_val, int(0, C_INT), np)
      endif
    end block
    return
  endif

  ! SAD fringe types: sad_full$ and soft_edge_only$
  if (fringe_type_val == sad_full$ .or. fringe_type_val == soft_edge_only$) then
    block
      real(rp) :: g_sad, fb_sad, c_dir_sad, k1_sad
      integer :: phys_end_sad, entering_sad

      phys_end_sad = physical_ele_end(edge, bunch%particle(1), ele%orientation)
      if (phys_end_sad == entrance_end$) then
        fb_sad = 12 * ele%value(fint$) * ele%value(hgap$)
      else
        fb_sad = 12 * ele%value(fintx$) * ele%value(hgapx$)
      endif
      c_dir_sad = rel_tracking_charge_to_mass(bunch%particle(1), param%particle) * &
                  ele%orientation * bunch%particle(1)%direction * bunch%particle(1)%time_dir
      g_sad = (ele%value(g$) + ele%value(dg$)) * c_dir_sad
      if (edge == second_track_edge$) g_sad = -g_sad

      charge_dir_val = rel_tracking_charge_to_mass(bunch%particle(1), param%particle) * &
                       ele%orientation * bunch%particle(1)%direction

      ! Hard multipole edge kick at entrance
      if (phys_end_sad == entrance_end$) then
        bp_arr = 0; ap_arr = 0; bp_arr(1) = ele%value(k1$)
        call gpu_hard_multipole_edge(bp_arr, ap_arr, int(1, C_INT), charge_dir_val, int(1, C_INT), np)
      endif

      if (fringe_type_val == sad_full$) then
        ! sad_full$: sad_soft then hwang at entrance, hwang then sad_soft at exit
        if (ele%is_on) then
          k1_sad = ele%value(k1$)
        else
          k1_sad = 0
        endif
        entering_sad = 0
        if ((edge == first_track_edge$ .and. bunch%particle(1)%direction == 1) .or. &
            (edge == second_track_edge$ .and. bunch%particle(1)%direction == -1)) entering_sad = 1

        if (edge == first_track_edge$) then
          if (fb_sad /= 0 .and. g_sad /= 0) call gpu_sad_bend_fringe(g_sad, fb_sad, np)
          block
            real(rp) :: g_hw2, e_hw2
            g_hw2 = (ele%value(g$) + ele%value(dg$)) * charge_dir_val
            if (phys_end_sad == entrance_end$) then; e_hw2 = ele%value(e1$); else; e_hw2 = ele%value(e2$); endif
            call gpu_bend_fringe(g_hw2, e_hw2, 0.0_rp, k1_sad, &
                int(entering_sad, C_INT), int(bunch%particle(1)%time_dir, C_INT), np)
          end block
        else
          block
            real(rp) :: g_hw2, e_hw2
            g_hw2 = (ele%value(g$) + ele%value(dg$)) * charge_dir_val
            if (phys_end_sad == entrance_end$) then; e_hw2 = ele%value(e1$); else; e_hw2 = ele%value(e2$); endif
            call gpu_bend_fringe(g_hw2, e_hw2, 0.0_rp, k1_sad, &
                int(entering_sad, C_INT), int(bunch%particle(1)%time_dir, C_INT), np)
          end block
          if (fb_sad /= 0 .and. g_sad /= 0) call gpu_sad_bend_fringe(g_sad, fb_sad, np)
        endif
      else
        ! soft_edge_only$: just the SAD soft kick
        if (fb_sad /= 0 .and. g_sad /= 0) call gpu_sad_bend_fringe(g_sad, fb_sad, np)
      endif

      ! Hard multipole edge kick at exit
      if (phys_end_sad == exit_end$) then
        bp_arr = 0; ap_arr = 0; bp_arr(1) = ele%value(k1$)
        call gpu_hard_multipole_edge(bp_arr, ap_arr, int(1, C_INT), charge_dir_val, int(0, C_INT), np)
      endif
    end block
    return
  endif

  ! SAD fringe types: sad_full$ and soft_edge_only$
  if (fringe_type_val == sad_full$ .or. fringe_type_val == soft_edge_only$) then
    block
      real(rp) :: g_sad, fb_sad, c_dir_sad, k1_sad
      integer :: phys_end_sad, entering_sad
      phys_end_sad = physical_ele_end(edge, bunch%particle(1), ele%orientation)
      if (phys_end_sad == entrance_end$) then
        fb_sad = 12 * ele%value(fint$) * ele%value(hgap$)
      else
        fb_sad = 12 * ele%value(fintx$) * ele%value(hgapx$)
      endif
      c_dir_sad = rel_tracking_charge_to_mass(bunch%particle(1), param%particle) * &
                  ele%orientation * bunch%particle(1)%direction * bunch%particle(1)%time_dir
      g_sad = (ele%value(g$) + ele%value(dg$)) * c_dir_sad
      if (edge == second_track_edge$) g_sad = -g_sad
      charge_dir_val = rel_tracking_charge_to_mass(bunch%particle(1), param%particle) * &
                       ele%orientation * bunch%particle(1)%direction
      if (phys_end_sad == entrance_end$) then
        bp_arr = 0; ap_arr = 0; bp_arr(1) = ele%value(k1$)
        call gpu_hard_multipole_edge(bp_arr, ap_arr, int(1, C_INT), charge_dir_val, int(1, C_INT), np)
      endif
      if (fringe_type_val == sad_full$) then
        k1_sad = merge(ele%value(k1$), 0.0_rp, ele%is_on)
        entering_sad = merge(1, 0, (edge == first_track_edge$ .and. bunch%particle(1)%direction == 1) .or. &
            (edge == second_track_edge$ .and. bunch%particle(1)%direction == -1))
        if (edge == first_track_edge$) then
          if (fb_sad /= 0 .and. g_sad /= 0) call gpu_sad_bend_fringe(g_sad, fb_sad, np)
          block
            real(rp) :: g_hw2, e_hw2
            g_hw2 = (ele%value(g$) + ele%value(dg$)) * charge_dir_val
            e_hw2 = merge(ele%value(e1$), ele%value(e2$), phys_end_sad == entrance_end$)
            call gpu_bend_fringe(g_hw2, e_hw2, 0.0_rp, k1_sad, &
                int(entering_sad, C_INT), int(bunch%particle(1)%time_dir, C_INT), np)
          end block
        else
          block
            real(rp) :: g_hw2, e_hw2
            g_hw2 = (ele%value(g$) + ele%value(dg$)) * charge_dir_val
            e_hw2 = merge(ele%value(e1$), ele%value(e2$), phys_end_sad == entrance_end$)
            call gpu_bend_fringe(g_hw2, e_hw2, 0.0_rp, k1_sad, &
                int(entering_sad, C_INT), int(bunch%particle(1)%time_dir, C_INT), np)
          end block
          if (fb_sad /= 0 .and. g_sad /= 0) call gpu_sad_bend_fringe(g_sad, fb_sad, np)
        endif
      else
        if (fb_sad /= 0 .and. g_sad /= 0) call gpu_sad_bend_fringe(g_sad, fb_sad, np)
      endif
      if (phys_end_sad == exit_end$) then
        bp_arr = 0; ap_arr = 0; bp_arr(1) = ele%value(k1$)
        call gpu_hard_multipole_edge(bp_arr, ap_arr, int(1, C_INT), charge_dir_val, int(0, C_INT), np)
      endif
    end block
    return
  endif

  ! Basic/hard_edge bend fringe (Hwang)
  if (fringe_type_val /= basic_bend$ .and. fringe_type_val /= hard_edge_only$) return
  charge_dir_val = rel_tracking_charge_to_mass(bunch%particle(1), param%particle) * &
                   ele%orientation * bunch%particle(1)%direction
  block
    real(rp) :: g_tot_hw, e_ang_hw, fint_gap_hw, k1_hw
    integer :: entering_hw, phys_end_hw
    if (ele%is_on) then
      g_tot_hw = (ele%value(g$) + ele%value(dg$)) * charge_dir_val; k1_hw = ele%value(k1$)
    else
      g_tot_hw = 0; k1_hw = 0
    endif
    phys_end_hw = physical_ele_end(edge, bunch%particle(1), ele%orientation)
    if (phys_end_hw == entrance_end$) then
      e_ang_hw = ele%value(e1$); fint_gap_hw = ele%value(fint$) * ele%value(hgap$)
    else
      e_ang_hw = ele%value(e2$); fint_gap_hw = ele%value(fintx$) * ele%value(hgapx$)
    endif
    if (fringe_type_val == hard_edge_only$) fint_gap_hw = 0
    entering_hw = 0
    if ((edge == first_track_edge$ .and. bunch%particle(1)%direction == 1) .or. &
        (edge == second_track_edge$ .and. bunch%particle(1)%direction == -1)) entering_hw = 1
    call gpu_bend_fringe(g_tot_hw, e_ang_hw, fint_gap_hw, k1_hw, &
                          int(entering_hw, C_INT), int(bunch%particle(1)%time_dir, C_INT), np)
  end block

end select
#endif

end subroutine gpu_apply_fringe_on_device

!------------------------------------------------------------------------
! gpu_apply_misalign_on_device — apply misalignment on device (public)
!------------------------------------------------------------------------
subroutine gpu_apply_misalign_on_device(ele, is_set, np)

use, intrinsic :: iso_c_binding

type (ele_struct), intent(in) :: ele
logical, intent(in) :: is_set  ! set$ (.true.) or unset$ (.false.)
integer(C_INT), intent(in) :: np

#ifdef USE_GPU_TRACKING
integer :: sf

if (.not. gpu_persist_on_device) return
if (.not. ele%bookkeeping_state%has_misalign) return
if (np == 0) return

sf = merge(1, -1, is_set)

if (ele%key == sbend$ .and. (ele%value(g$) /= 0 .or. ele%value(ref_tilt_tot$) /= 0 &
                             .or. ele%value(roll$) /= 0)) then
  call gpu_bend_offset(ele%value(g$), ele%value(rho$), &
       ele%value(l$) * 0.5_rp, ele%value(angle$), &
       ele%value(ref_tilt_tot$), ele%value(roll_tot$), &
       ele%value(x_offset_tot$), ele%value(y_offset_tot$), ele%value(z_offset_tot$), &
       ele%value(x_pitch$), ele%value(y_pitch$), sf, np)
elseif (ele%key == sbend$) then
  call gpu_misalign(ele%value(x_offset_tot$), ele%value(y_offset_tot$), &
                     ele%value(ref_tilt_tot$), sf, np)
elseif (ele%value(x_pitch_tot$) /= 0 .or. ele%value(y_pitch_tot$) /= 0 .or. &
        ele%value(z_offset_tot$) /= 0) then
  block
    real(rp) :: W3(3,3)
    if (is_set) then
      call floor_angles_to_w_mat(ele%value(x_pitch_tot$), ele%value(y_pitch_tot$), &
                                 ele%value(tilt_tot$), w_mat_inv = W3)
    else
      call floor_angles_to_w_mat(ele%value(x_pitch_tot$), ele%value(y_pitch_tot$), &
                                 ele%value(tilt_tot$), w_mat = W3)
    endif
    call gpu_misalign_3d(W3, ele%value(x_offset_tot$), ele%value(y_offset_tot$), &
                         ele%value(z_offset_tot$), sf, np)
  end block
else
  call gpu_misalign(ele%value(x_offset_tot$), ele%value(y_offset_tot$), &
                     ele%value(tilt_tot$), sf, np)
endif
#endif

end subroutine gpu_apply_misalign_on_device

end module gpu_tracking_mod
