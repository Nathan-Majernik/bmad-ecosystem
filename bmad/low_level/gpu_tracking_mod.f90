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
public :: track_bunch_thru_bend_gpu
public :: track_bunch_thru_lcavity_gpu
public :: track_bunch_thru_pipe_gpu
public :: check_entrance_aperture_for_gpu
public :: gpu_rad_eligible
public :: track_bunch_thru_elements_gpu
public :: ele_gpu_can_stay_on_device
public :: gpu_upload_particles, gpu_download_particles
public :: gpu_space_charge_3d, gpu_csr_bin_particles, gpu_csr_apply_kicks
public :: gpu_persistent_track_element, gpu_persistent_flush, gpu_persistent_seed

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
                            ea2_arr, eb2_arr, ix_elec_max) bind(C, name='gpu_track_bend_')
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

  subroutine gpu_track_bend_dev(mc2, g, g_tot, dg, b1, &
                                 ele_length, delta_ref_time, e_tot_ele, &
                                 rel_charge_dir, p0c_ele, n_particles, &
                                 a2_arr, b2_arr, cm_arr, &
                                 ix_mag_max, n_step, &
                                 ea2_arr, eb2_arr, ix_elec_max) bind(C, name='gpu_track_bend_dev_')
    use, intrinsic :: iso_c_binding
    real(C_DOUBLE), value, intent(in) :: mc2, g, g_tot, dg, b1
    real(C_DOUBLE), value, intent(in) :: ele_length, delta_ref_time, e_tot_ele
    real(C_DOUBLE), value, intent(in) :: rel_charge_dir, p0c_ele
    integer(C_INT), value, intent(in) :: n_particles
    real(C_DOUBLE), intent(in) :: a2_arr(*), b2_arr(*), cm_arr(*)
    integer(C_INT), value, intent(in) :: ix_mag_max, n_step
    real(C_DOUBLE), intent(in) :: ea2_arr(*), eb2_arr(*)
    integer(C_INT), value, intent(in) :: ix_elec_max
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

  subroutine gpu_spacecharge_cleanup() bind(C, name='gpu_spacecharge_cleanup_')
  end subroutine

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
gpu_trk_initialized = .false.
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
! Currently supported element types: drift, quadrupole, sbend, lcavity,
! pipe, monitor, instrument.
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
case (drift$, quadrupole$, sbend$, lcavity$, pipe$, monitor$, instrument$)
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
case (drift$, pipe$, monitor$, instrument$)
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
integer :: ix_mag_max, ix_elec_max, n_step
real(rp) :: ele_length, mc2, b1, delta_ref_time, e_tot_ele, p0c_ele
real(rp) :: g, g_tot, dg, rel_charge_dir
real(rp) :: r_step, length, step_len_val
real(rp) :: an(0:n_pole_maxx), bn(0:n_pole_maxx)
real(rp) :: an_elec(0:n_pole_maxx), bn_elec(0:n_pole_maxx)
type (fringe_field_info_struct) :: fringe_info
logical :: has_misalign, has_mag_multipoles, has_elec_multipoles, apply_rad

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
p0c_ele = ele%value(p0c$)

! Bail if exact_multipoles is on — too complex for GPU
if (nint(ele%value(exact_multipoles$)) /= off$) return

has_misalign = ele%bookkeeping_state%has_misalign

! Compute charge/direction factors
rel_charge_dir = ele%orientation * bunch%particle(1)%direction * &
                 rel_tracking_charge_to_mass(bunch%particle(1), param%particle)
! Get multipoles
call multipole_ele_to_ab(ele, .false., ix_mag_max, an, bn, magnetic$, include_kicks$, b1)
b1 = b1 * rel_charge_dir
if (abs(b1) < 1d-10) then
  bn(1) = b1
  b1 = 0
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
                          ea2_arr, eb2_arr, int(ix_elec_max, C_INT))
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
                      ea2_arr, eb2_arr, int(ix_elec_max, C_INT))
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
  if (bmad_com%absolute_time_ref_shift) ref_time_start_val = lord%value(ref_time_start$)
endif

! Extract step data from the lord's RF step array (indices 0..n_steps+1)
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
  h_step_time(j+1)  = step%time
enddo

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
function ele_gpu_can_stay_on_device(ele) result (can_stay)
type (ele_struct), intent(in) :: ele
logical :: can_stay
type (fringe_field_info_struct) :: fringe_info
logical :: has_fringe_needs_cpu

can_stay = .false.
if (.not. ele_gpu_eligible(ele)) return

! Elements with CSR or space charge need track1_bunch_csr sub-stepping
if (bmad_com%csr_and_space_charge_on) then
  if (ele%csr_method /= off$ .or. ele%space_charge_method /= off$) return
endif

! Pipe, monitor, and instrument elements can have multipoles that require
! the pipe GPU path (quad kernel with b1=0). Without multipoles, they're
! simple drifts and can stay on device.
if (ele%key == pipe$ .or. ele%key == monitor$ .or. ele%key == instrument$) then
  if (allocated(ele%multipole_cache)) then
    if (ele%multipole_cache%ix_pole_mag_max > -1 .or. &
        ele%multipole_cache%ix_kick_mag_max > -1 .or. &
        ele%multipole_cache%ix_pole_elec_max > -1 .or. &
        ele%multipole_cache%ix_kick_elec_max > -1) return
  endif
  ! No multipoles: can stay on device as drift
endif

! Simple misalignment (x_offset, y_offset, tilt/ref_tilt) handled on GPU
! for all element types including bends. Pitches and z_offset still need CPU.
if (ele%bookkeeping_state%has_misalign) then
  if (ele%value(x_pitch_tot$) /= 0 .or. ele%value(y_pitch_tot$) /= 0 .or. &
      ele%value(z_offset_tot$) /= 0) return
endif

! Check for fringe that requires CPU
has_fringe_needs_cpu = .false.
select case (ele%key)
case (drift$, pipe$, monitor$, instrument$)
  ! Drifts, pipes, monitors, and instruments never have fringe
case (lcavity$)
  ! Lcavity fringe is already handled on GPU
case (quadrupole$)
  ! Quad fringe is now handled on GPU via gpu_quad_fringe
case (sbend$)
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

    ! Misalignment on device via 2D offset+tilt kernel.
    ! For bends, use ref_tilt_tot; for others, use tilt_tot (=roll_tot).
    has_misalign = ele%bookkeeping_state%has_misalign
    if (has_misalign) then
      if (ele%key == sbend$) then
        call gpu_misalign(ele%value(x_offset_tot$), ele%value(y_offset_tot$), &
                           ele%value(ref_tilt_tot$), 1, n)
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
      if (ele%key == sbend$) then
        call gpu_misalign(ele%value(x_offset_tot$), ele%value(y_offset_tot$), &
                           ele%value(ref_tilt_tot$), -1, n)
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
    case (sbend$)
      call track_bunch_thru_bend_gpu(bunch, ele, branch%param, did_track)
    case (lcavity$)
      call track_bunch_thru_lcavity_gpu(bunch, ele, branch%param, did_track)
    case (pipe$, monitor$, instrument$)
      call track_bunch_thru_pipe_gpu(bunch, ele, branch%param, did_track)
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
case (drift$, pipe$, monitor$, instrument$)
  call dispatch_drift_body(ele, np)
case (quadrupole$)
  call dispatch_quad_body(ele, param, np)
case (sbend$)
  call dispatch_bend_body(ele, param, np)
case (lcavity$)
  call dispatch_lcavity_body(ele, param, np)
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

if (nint(ele%value(exact_multipoles$)) /= off$) return

rel_charge_dir = ele%orientation * bunch%particle(1)%direction * &
                 rel_tracking_charge_to_mass(bunch%particle(1), param%particle)

call multipole_ele_to_ab(ele, .false., ix_mag_max, an, bn, magnetic$, include_kicks$, b1)
b1 = b1 * rel_charge_dir
if (abs(b1) < 1d-10) then
  bn(1) = b1
  b1 = 0
endif
call multipole_ele_to_ab(ele, .false., ix_elec_max, an_elec, bn_elec, electric$)

g = ele%value(g$)
length = bunch%particle(1)%time_dir * ele_length
if (length == 0) then
  dg = 0
else
  dg = bn(0) / ele_length
  bn(0) = 0
endif
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
                        ea2_arr, eb2_arr, int(ix_elec_max, C_INT))
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
  if (bmad_com%absolute_time_ref_shift) ref_time_start_val = lord%value(ref_time_start$)
endif

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
  h_step_time(j+1)  = step%time
enddo

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
#endif

did_track = .false.

#ifdef USE_GPU_TRACKING
can_stay = ele_gpu_can_stay_on_device(ele)
if (.not. can_stay) then
  ! Element can't stay on device — flush and let caller handle it
  if (gpu_persist_on_device) call gpu_persistent_flush(bunch, ele)
  return
endif

! Only use persistent path if data is already on device from a previous
! element. First element in a sequence falls through to per-element path,
! which then seeds the device for subsequent elements.
if (.not. gpu_persist_on_device) return

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

! Detect new beam: if bunch identity changed (different allocation),
! invalidate persistent state so we re-upload.
block
  integer(8) :: bunch_id
  bunch_id = transfer(loc(bunch%particle(1)%vec(1)), bunch_id)
  if (gpu_persist_on_device .and. bunch_id /= gpu_persist_bunch_id) then
    gpu_persist_on_device = .false.
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
  if (ele%key == sbend$) then
    call gpu_misalign(ele%value(x_offset_tot$), ele%value(y_offset_tot$), &
                       ele%value(ref_tilt_tot$), 1, n)
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
  if (ele%key == sbend$) then
    call gpu_misalign(ele%value(x_offset_tot$), ele%value(y_offset_tot$), &
                       ele%value(ref_tilt_tot$), -1, n)
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
case (drift$, pipe$, monitor$, instrument$)
  mc2 = mass_of(bunch%particle(1)%species)
  ele_length = ele%value(l$)
  if (ele_length == 0) return
  call gpu_track_drift_dev(gp_s, mc2, ele_length, np)

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

case (sbend$)
  ele_length = ele%value(l$)
  if (ele_length == 0) return
  mc2 = mass_of(bunch%particle(1)%species)
  delta_ref_time = ele%value(delta_ref_time$)
  e_tot_ele = ele%value(e_tot$)
  p0c_ele = ele%value(p0c$)
  if (nint(ele%value(exact_multipoles$)) /= off$) return
  rel_charge_dir = ele%orientation * bunch%particle(1)%direction * &
                   rel_tracking_charge_to_mass(bunch%particle(1), param%particle)
  call multipole_ele_to_ab(ele, .false., ix_mag_max, an, bn, magnetic$, include_kicks$, b1)
  b1 = b1 * rel_charge_dir
  if (abs(b1) < 1d-10) then; bn(1) = b1; b1 = 0; endif
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
                          ea2_arr, eb2_arr, int(ix_elec_max, C_INT))

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
    if (bmad_com%absolute_time_ref_shift) ref_time_start_val = lord%value(ref_time_start$)
  endif
  allocate(h_step_s0(n_steps+2), h_step_s(n_steps+2))
  allocate(h_step_p0c(n_steps+2), h_step_p1c(n_steps+2))
  allocate(h_step_scale(n_steps+2), h_step_time(n_steps+2))
  do j = 0, n_steps + 1
    step => lord%rf%steps(j)
    h_step_s0(j+1) = step%s0; h_step_s(j+1) = step%s
    h_step_p0c(j+1) = step%p0c; h_step_p1c(j+1) = step%p1c
    h_step_scale(j+1) = step%scale; h_step_time(j+1) = step%time
  enddo
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

end subroutine gpu_persistent_flush

!------------------------------------------------------------------------
! gpu_persistent_seed — upload bunch data to device after per-element
! GPU tracking, so the next element can use the persistent path.
!------------------------------------------------------------------------
subroutine gpu_persistent_seed(bunch, ele)

use, intrinsic :: iso_c_binding

type (bunch_struct), intent(in) :: bunch
type (ele_struct),   intent(in) :: ele
integer :: j, n

#ifdef USE_GPU_TRACKING
if (.not. ele_gpu_can_stay_on_device(ele)) return

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

end module gpu_tracking_mod
