!+
! See the paper:
!   "Coherent Synchrotron Radiation Simulations for Off-Axis Beams Using the Bmad Toolkit"
!   D. Sagan & C. Mayes
!   Proceedings of IPAC2017, Copenhagen, Denmark THPAB076
!   https://accelconf.web.cern.ch/ipac2017/papers/thpab076.pdf
!-

module csr_and_space_charge_mod

use beam_utils
use bmad_interface
use spline_mod
use super_recipes_mod, only: super_zbrent
use open_spacecharge_mod
use csr3d_mod, only: csr3d_steady_state_solver

! csr_ele_info_struct holds info for a particular lattice element
! The centroid "chord" is the line from the centroid position at the element entrance to
! the centroid position at the element exit end.

type csr_ele_info_struct
  type (ele_struct), pointer :: ele            ! lattice element
  type (coord_struct) orbit0, orbit1           ! centroid orbit at entrance/exit ends
  type (floor_position_struct) floor0, floor1  ! Floor position of centroid at entrance/exit ends
  type (floor_position_struct) ref_floor0, ref_floor1  ! Floor position of element ref coords at entrance/exit ends
  type (spline_struct) spline                  ! Spline for centroid orbit. spline%x = distance along chord.
                                               !   The spline is zero at the ends by construction.
  real(rp) theta_chord                         ! Reference angle of chord in z-x plane
  real(rp) L_chord                             ! Chord Length. Negative if bunch moves backwards in element.
  real(rp) dL_s                                ! L_s(of element) - L_chord
end type

type csr_bunch_slice_struct  ! Structure for a single particle bin.
  real(rp) :: x0 = 0, y0 = 0 ! Transverse center of the particle distrubution
  real(rp) :: z0_edge = 0    ! Left (min z) edge of bin
  real(rp) :: z1_edge = 0    ! Right (max z) edge of bin
  real(rp) :: z_center = 0   ! z at center of bin.
  real(rp) :: sig_x = 0      ! particle's RMS width
  real(rp) :: sig_y = 0      ! particle's RMS width
  real(rp) :: charge = 0     ! charge of the particles
  real(rp) :: dcharge_density_dz = 0      ! Charge density gradient 
  real(rp) :: edge_dcharge_density_dz = 0 ! gradient between this and preceeding bin. [Evaluated at bin edge.]
  real(rp) :: kick_csr = 0                ! CSR kick
  real(rp) :: coef_lsc_plus(0:2,0:2) = 0  ! LSC Kick coefs.
  real(rp) :: coef_lsc_minus(0:2,0:2) = 0 ! LSC Kick coefs.
  real(rp) :: kick_lsc = 0
  real(rp) :: n_particle = 0 ! Number of particles in slice can be a fraction since particles span multiple bins.
end type

! csr_kick1_struct stores the CSR kick, kick integral etc. for a given source and kick positions.
! This structure also holds info on parameters that go into the kick calculation.
! Since an integration step involves one kick position and many source positions,
! the information that is only kick position dependent is held in the csr_struct and
! the csr_struct holds an array of csr_kick1_structs, one for each dz.

type csr_kick1_struct ! Sub-structure for csr calculation cache
  real(rp) I_csr            ! Kick integral.
  real(rp) I_int_csr        ! Integrated Kick integral.
  real(rp) image_kick_csr   ! kick.
  real(rp) L_vec(3)         ! L vector in global coordinates.
  real(rp) L                ! Distance between source and kick points.
  real(rp) dL               ! = epsilon_L = Ls - L
  real(rp) dz_particles     ! Kicked particle - source particle position at constant time.
  real(rp) s_chord_source   ! Source point coordinate along chord.
  real(rp) theta_L          ! Angle of L vector
  real(rp) theta_sl         ! Angle between velocity of particle at source pt and L
  real(rp) theta_lk         ! Angle between L and velocity of kicked particle
  integer ix_ele_source     ! Source element index.
  type (floor_position_struct) floor_s  ! Floor position of source pt
end type

type csr_particle_position_struct
  real(rp) :: r(3)       ! particle position
  real(rp) :: charge     ! particle charge
end type

type csr_struct           ! Structurture for binning particle averages
  real(rp) gamma, gamma2        ! Relativistic gamma factor.
  real(rp) rel_mass             ! m_particle / m_electron
  real(rp) beta                 ! Relativistic beta factor.
  real(rp) :: dz_slice = 0      ! Bin width
  real(rp) ds_track_step        ! True step size
  real(rp) s_kick               ! Kick point longitudinal location (element ref coords) from entrance end
  real(rp) s_chord_kick         ! Kick point along beam centroid line
  real(rp) y_source             ! Height of source particle.
  real(rp) kick_factor          ! Coefficient to scale the kick
  real(rp) actual_track_step    ! ds_track_step scalled by Length_centroid_chord / Length_element ratio
  real(rp) x0_bunch, y0_bunch   ! Bunch centroid
  type(floor_position_struct) floor_k   ! Floor coords at kick point
  integer species                       ! Particle type
  integer ix_ele_kick                   ! Same as element being tracked through.
  type (csr_bunch_slice_struct), allocatable :: slice(:)    ! slice(i) refers to the i^th bunch slice.
  type (csr_kick1_struct), allocatable :: kick1(:)          ! kick1(i) referes to the kick between two slices i bins apart.
  type (csr_ele_info_struct), allocatable :: eleinfo(:)     ! Element-by-element information.
  type (ele_struct), pointer :: kick_ele                    ! Element where the kick pt is == ele tracked through.
  type (mesh3d_struct) :: mesh3d
  type (csr_particle_position_struct), allocatable :: position(:)
end type

contains

!----------------------------------------------------------------------------
!----------------------------------------------------------------------------
!----------------------------------------------------------------------------
!+
! Subroutine track1_bunch_csr (bunch, ele, centroid, err, s_start, s_end, bunch_track)
!
! Routine to track a bunch of particles through an element with csr radiation effects.
!
! Input:
!   bunch         -- Bunch_struct: Starting bunch position.
!   ele           -- Ele_struct: The element to track through. Must be part of a lattice.
!   centroid(0:)  -- coord_struct, Approximate beam centroid orbit for the lattice branch.
!                      Calculate this before beam tracking by tracking a single particle.
!   s_start       -- real(rp), optional: Starting position relative to ele. Default = 0
!   s_end         -- real(rp), optional: Ending position. Default is ele length.
!   bunch_track   -- bunch_track_struct, optional: Existing tracks. If bunch_track%n_pt = -1 then
!                        Overwrite any existing track.
!
! Output:
!   bunch       -- Bunch_struct: Ending bunch position.
!   err         -- Logical: Set true if there is an error. EG: Too many particles lost.
!   bunch_track -- bunch_track_struct, optional: track information if the tracking method does
!                        tracking step-by-step. When tracking through multiple elements, the 
!                        trajectory in an element is appended to the existing trajectory. 
!-

subroutine track1_bunch_csr (bunch, ele, centroid, err, s_start, s_end, bunch_track)

use gpu_tracking_mod, only: gpu_persistent_flush, &
                             gpu_persistent_seed, gpu_persist_on_device, &
                             gpu_track_body_on_device, ele_gpu_eligible

implicit none

type (bunch_struct), target :: bunch
type (coord_struct), pointer :: pt, c0
type (ele_struct), target :: ele
type (bunch_track_struct), optional :: bunch_track
type (branch_struct), pointer :: branch
type (ele_struct) :: runt
logical :: gpu_csr_active, gpu_did_track
type (ele_struct), pointer :: ele0, s_ele
type (csr_struct), target :: csr
type (csr_ele_info_struct), pointer :: eleinfo
type (coord_struct), target :: centroid(0:)
type (floor_position_struct) floor

real(rp), optional :: s_start, s_end
real(rp) s0_step, vec0(6), vec(6), theta_chord, theta0, theta1, L
real(rp) e_tot, f1, x, z, ds_step, s_save_last

integer i, j, n, ie, ns, nb, n_step, n_live, i_step
integer :: iu_wake

character(*), parameter :: r_name = 'track1_bunch_csr'
logical err, err_flag, parallel0, parallel1

! Init

err = .true.
csr%kick_ele => ele    ! Element where the kick pt is == ele tracked through.
branch => pointer_to_branch(ele)

if (ele%space_charge_method == fft_3d$) then
  c0 => centroid(ele%ix_ele)
  call convert_pc_to((1+c0%vec(6)) * c0%p0c, c0%species, gamma = csr%mesh3d%gamma)
  csr%mesh3d%nhi = space_charge_com%space_charge_mesh_size
endif

! No CSR for a zero length element.
! And taylor elements get ignored.

if (ele%value(l$) == 0 .or. ele%key == taylor$) then
  call track1_bunch_hom (bunch, ele)
  err = .false.
  return
endif

! n_step is the number of steps to take when tracking through the element.
! csr%ds_step is the true step length.

if (ele%csr_method == one_dim$ .and. (ele%key == wiggler$ .or. ele%key == undulator$) .and. &
                                (ele%field_calc == planar_model$ .or. ele%field_calc == helical_model$)) then
  call out_io (s_warn$, r_name, 'CALCULATION OF CSR EFFECTS IN PLANAR OR HELICAL MODEL WIGGLERS MAY BE INVALID SINCE', &
                                'CSR TRACKING USES A SPLINE FIT FOR THE CENTROID ORBIT WHICH IS INACCURATE FOR A WIGGLING ORBIT.', &
                                'FOR: ' // ele%name)
endif

if (space_charge_com%n_bin <= space_charge_com%particle_bin_span + 1) then
  if (space_charge_com%n_bin == 0) then
    call out_io (s_error$, r_name, &
            'SPACE_CHARGE_COM%N_BIN IS ZERO WHICH INDICATES THAT THE SPACE_CHARGE_COM STRUCTURE HAS NOT BEEN SET BY THE USER.', &
            'ALL PARTICLES IN THE BUNCH WILL BE MARKED AS LOST.')
  else
    call out_io (s_error$, r_name, &
            'SPACE_CHARGE_COM%N_BIN (= \i0\) MUST BE GREATER THAN SPACE_CHARGE_COM%PARTICLE_BIN_SPAN+1 (= \i0\+1)!', &
            'ALL PARTICLES IN THE BUNCH WILL BE MARKED AS LOST.', &
             i_array = [space_charge_com%n_bin, space_charge_com%particle_bin_span])
  endif
  bunch%particle%state = lost$
  return
endif

! make sure that ele_len / track_step is an integer.

csr%ds_track_step = ele%value(csr_ds_step$)
if (csr%ds_track_step == 0) csr%ds_track_step = space_charge_com%ds_track_step
if (csr%ds_track_step == 0) then
  call out_io (s_fatal$, r_name, 'NEITHER SPACE_CHARGE_COM%DS_TRACK_STEP NOR CSR_TRACK_STEP FOR THIS ELEMENT ARE SET! ' // ele%name)
  if (global_com%exit_on_error) call err_exit
  return
endif

n_step = max (1, nint(ele%value(l$) / csr%ds_track_step))
csr%ds_track_step = ele%value(l$) / n_step

! Calculate beam centroid info at element edges, etc.

n = min(ele%ix_ele+10, branch%n_ele_track)
allocate (csr%eleinfo(0:n))

do i = 0, n
  eleinfo => csr%eleinfo(i)
  eleinfo%ele => branch%ele(i)  ! Pointer to the P' element
  s_ele => eleinfo%ele
  eleinfo%ref_floor1 = branch%ele(i)%floor
  eleinfo%ref_floor1%r(2) = 0  ! Make sure in horizontal plane

  eleinfo%orbit1 = centroid(i)
  vec = eleinfo%orbit1%vec
  floor%r = [vec(1), vec(3), s_ele%value(l$)]
  eleinfo%floor1 = coords_local_curvilinear_to_floor (floor, s_ele)
  eleinfo%floor1%r(2) = 0  ! Make sure in horizontal plane
  ! The 1d-14 is inserted to avoid 0/0 error at the beginning of an e_gun.
  eleinfo%floor1%theta = s_ele%floor%theta + asin(vec(2) / (1.0_rp + 1d-14 + vec(6)))

  if (i == 0) then
    eleinfo%ref_floor0 = csr%eleinfo(i)%ref_floor1
    eleinfo%floor0   = csr%eleinfo(i)%floor1
    eleinfo%orbit0   = csr%eleinfo(i)%orbit1
  else
    eleinfo%ref_floor0 = csr%eleinfo(i-1)%ref_floor1
    eleinfo%floor0   = csr%eleinfo(i-1)%floor1
    eleinfo%orbit0   = csr%eleinfo(i-1)%orbit1
  endif

  vec0 = eleinfo%orbit0%vec
  vec = eleinfo%orbit1%vec
  theta_chord = atan2(eleinfo%floor1%r(1)-eleinfo%floor0%r(1), eleinfo%floor1%r(3)-eleinfo%floor0%r(3))
  eleinfo%theta_chord = theta_chord

  ! An element with a negative step length (EG off-center beam in a patch which just has an x-pitch), can
  ! have floor%theta and theta_chord 180 degress off. Hence, restrict theta0 & theta1 to be within [-pi/2,pi/2]
  theta0 = modulo2(eleinfo%floor0%theta - theta_chord, pi/2)
  theta1 = modulo2(eleinfo%floor1%theta - theta_chord, pi/2)
  eleinfo%L_chord = sqrt((eleinfo%floor1%r(1)-eleinfo%floor0%r(1))**2 + (eleinfo%floor1%r(3)-eleinfo%floor0%r(3))**2)
  if (abs(eleinfo%L_chord) < 1d-8) then   ! 1d-8 is rather arbitrary.
    eleinfo%spline = spline_struct()
    eleinfo%dL_s = 0
    cycle
  endif

  if (s_ele%key == match$) cycle  ! Match elements offsets are ignored so essentially they are like markers.

  ! With a negative step length, %L_chord is negative
  parallel0 = (abs(modulo2(eleinfo%floor0%theta - theta_chord, pi)) < pi/2)
  parallel1 = (abs(modulo2(eleinfo%floor1%theta - theta_chord, pi)) < pi/2)
  if (parallel0 .neqv. parallel1) then
    call out_io (s_fatal$, r_name, 'VERY CONFUSED CSR CALCULATION! PLEASE SEEK HELP ' // ele%name)
    if (global_com%exit_on_error) call err_exit
    return
  endif
  if (.not. parallel0) eleinfo%L_chord = -eleinfo%L_chord

  eleinfo%spline = create_a_spline ([0.0_rp, 0.0_rp], [eleinfo%L_chord, 0.0_rp], theta0, theta1)
  eleinfo%dL_s = dspline_len(0.0_rp, eleinfo%L_chord, eleinfo%spline)
enddo

!

csr%species = bunch%particle(1)%species
csr%ix_ele_kick = ele%ix_ele
csr%actual_track_step = csr%ds_track_step * (csr%eleinfo(ele%ix_ele)%L_chord / ele%value(l$))

! Determine if GPU CSR path is viable.
! We check ele_gpu_eligible (body kernel exists) rather than ele_gpu_can_stay_on_device
! (which excludes CSR elements and elements with fringe — both handled by this routine).
! If data isn't already on device, upload it now.
! GPU CSR path: use GPU for body tracking + binning + kicks when:
! 1. GPU is available and element is eligible
! 2. Particle count is large enough to amortize per-sub-step GPU overhead
!    (below ~100K particles, the CPU is faster for CSR sub-stepping)
gpu_csr_active = .false.
if (bmad_com%gpu_tracking_on .and. ele_gpu_eligible(ele) .and. &
    .not. bmad_com%spin_tracking_on .and. &
    bunch%particle(1)%direction == 1 .and. bunch%particle(1)%time_dir == 1 .and. &
    size(bunch%particle) >= 100000) then
  gpu_csr_active = .true.
  if (.not. gpu_persist_on_device) then
    call gpu_persistent_seed(bunch, ele, force=.true.)
  endif
endif

! If GPU CSR is NOT active but data is on device from a previous element,
! flush to CPU so the CPU CSR loop can access particle data.
if (.not. gpu_csr_active .and. gpu_persist_on_device) then
  call gpu_persistent_flush(bunch, ele)
endif

call save_a_bunch_step (ele, bunch, bunch_track, s_start)

!----------------------------------------------------------------------------------------
! Loop over the tracking steps
! runt is the element that is tracked through at each step.
!
! GPU path: when data is on device, use GPU kernels for body tracking,
! binning, and kick application. The bin-level kick computation stays on CPU.

do i_step = 0, n_step

  ! track through the runt

  if (i_step /= 0) then
    call element_slice_iterator (ele, branch%param, i_step, n_step, runt, s_start, s_end)
    if (gpu_csr_active) then
      ! Track the runt body on GPU (data already on device).
      call gpu_track_body_on_device(bunch, runt, branch%param, gpu_did_track)
      if (.not. gpu_did_track) then
        ! GPU body failed for this runt — flush and fall back to CPU
        call gpu_persistent_flush(bunch, runt)
        gpu_csr_active = .false.
        call track1_bunch_hom (bunch, runt)
      endif
    else
      call track1_bunch_hom (bunch, runt)
    endif
  endif

  s0_step = i_step * csr%ds_track_step
  if (present(s_start)) s0_step = s0_step + s_start

  ! Cannot do a realistic calculation if there are less particles than bins
  ! When GPU CSR is active, CPU bunch%particle%state is stale — skip check
  ! (the GPU binning kernel handles dead particles correctly).

  if (.not. gpu_csr_active) then
    n_live = count(bunch%particle%state == alive$)
    if (n_live < space_charge_com%n_bin) then
      call out_io (s_error$, r_name, 'NUMBER OF LIVE PARTICLES: \i0\ ', &
                            'LESS THAN NUMBER OF BINS FOR CSR CALC.', &
                            'AT ELEMENT: ' // trim(ele%name) // '  [# \i0\] ', &
                            i_array = [n_live, ele%ix_ele ])
      return
    endif
  endif

  ! Assume a linear energy gain in a cavity

  f1 = s0_step / ele%value(l$)
  e_tot = f1 * branch%ele(ele%ix_ele-1)%value(e_tot$) + (1 - f1) * ele%value(e_tot$)
  call convert_total_energy_to (e_tot, branch%param%particle, csr%gamma, beta = csr%beta)
  csr%gamma2 = csr%gamma**2
  csr%rel_mass = mass_of(branch%param%particle) / m_electron

  ! Bin particles — only needed for CSR or slice SC (not for fft_3d-only)
  if (ele%space_charge_method == slice$ .or. ele%csr_method == one_dim$) then
    if (gpu_csr_active) then
      call csr_bin_particles_gpu(ele, bunch, csr, err_flag)
    else
      call csr_bin_particles (ele, bunch%particle, csr, err_flag)
    endif
    if (err_flag) return
  endif

  ! CSR kick computation — skip entirely for fft_3d-only elements
  if (ele%space_charge_method == slice$ .or. ele%csr_method == one_dim$) then
    csr%s_kick = s0_step
    csr%s_chord_kick = s_ref_to_s_chord (s0_step, csr%eleinfo(ele%ix_ele))
    z = csr%s_chord_kick
    x = spline1(csr%eleinfo(ele%ix_ele)%spline, z)
    theta_chord = csr%eleinfo(ele%ix_ele)%theta_chord
    csr%floor_k%r = [x*cos(theta_chord)+z*sin(theta_chord), 0.0_rp, -x*sin(theta_chord)+z*cos(theta_chord)] + &
                        csr%eleinfo(ele%ix_ele)%floor0%r
    csr%floor_k%theta = theta_chord + spline1(csr%eleinfo(ele%ix_ele)%spline, z, 1)
  endif

  if (ele%space_charge_method == slice$ .or. ele%csr_method == one_dim$) then
    do ns = 0, space_charge_com%n_shield_images
      ! The factor of -1^ns accounts for the sign of the image currents
      ! Take into account that at the endpoints we are only putting in a half kick.
      ! The factor of two is due to there being image currents both above and below.

      csr%kick_factor = (-1)**ns
      if (i_step == 0 .or. i_step == n_step) csr%kick_factor = csr%kick_factor / 2
      if (ns /= 0) csr%kick_factor = 2 * csr%kick_factor

      csr%y_source = ns * space_charge_com%beam_chamber_height

      if (gpu_csr_active) then
        call csr_bin_kicks_gpu_wrap(ele, csr, err_flag)
      else
        call csr_bin_kicks (ele, s0_step, csr, err_flag)
      endif
      if (err_flag) return
    enddo
  endif

  ! Apply kicks to particles — GPU path modifies device data directly
  if (gpu_csr_active) then
    call csr_apply_kicks_gpu(ele, csr, bunch)
  else
    call csr_and_sc_apply_kicks (ele, csr, bunch%particle)
  endif

  call save_a_bunch_step (ele, bunch, bunch_track, s0_step)

  ! Record wake to file?

  if (space_charge_com%diagnostic_output_file /= '') then
    iu_wake = lunget()
    open (iu_wake, file = space_charge_com%diagnostic_output_file, access = 'append')
    if (i_step == 0) then
      write (iu_wake, '(a)') '!------------------------------------------------------------'
      write (iu_wake, '(a, i6, 2x, a)') '! ', ele%ix_ele, trim(ele%name)
    endif
    write (iu_wake, '(a)') '!#-----------------------------'
    write (iu_wake, '(a, i4, f12.6)') '! Step index:', i_step
    write (iu_wake, '(a, f12.6)') '! S-position:', s0_step
    write (iu_wake, '(a)') '!         Z   Charge/Meter    CSR_Kick/m       I_CSR/m      S_Source' 
    if (allocated(csr%kick1)) then
      ds_step = csr%kick_factor * csr%actual_track_step
      do j = 1, space_charge_com%n_bin
        ele0 => branch%ele(csr%kick1(j)%ix_ele_source)
        write (iu_wake, '(f14.10, 4es14.6)') csr%slice(j)%z_center, &
                    csr%slice(j)%charge/csr%dz_slice, csr%slice(j)%kick_csr/ds_step, &
                    csr%kick1(j)%I_csr/ds_step, ele0%s_start + csr%kick1(j)%s_chord_source
      enddo
    elseif (i_step == 0) then
      write (iu_wake, '(a)') 'Note: CSR wake not calculated for element: ' // ele%name
      write (iu_wake, '(a)') '      Check settings of ele%space_charge_method and ele%csr_method.'
    endif
    close (iu_wake)
  endif

enddo

err = .false.

contains

!------------------------------------------------------------------------
! GPU-accelerated CSR particle binning.
! Uses GPU z-reduction for min/max, GPU kernel for per-particle binning,
! then copies results into csr%slice struct for CPU bin-kick computation.
!------------------------------------------------------------------------
subroutine csr_bin_particles_gpu(ele_in, bunch_in, csr_in, err_flag_out)
use gpu_tracking_mod, only: gpu_csr_z_minmax, gpu_csr_bin_particles
use, intrinsic :: iso_c_binding
type (ele_struct), intent(in) :: ele_in
type (bunch_struct), target, intent(in) :: bunch_in
type (csr_struct), target, intent(inout) :: csr_in
logical, intent(out) :: err_flag_out

real(rp) :: z_minval, z_maxval, dz_l, z_center_l, z_min_l, dz_particle_l
real(C_DOUBLE), allocatable :: h_charge_l(:), h_bin_charge_l(:), h_bin_x0_wt_l(:), h_bin_y0_wt_l(:), h_bin_n_particle_l(:)
integer :: ii, np_l, nb_l, n_bin_eff_l

err_flag_out = .false.

if (ele_in%space_charge_method /= slice$ .and. ele_in%csr_method /= one_dim$) return

nb_l = space_charge_com%n_bin
np_l = size(bunch_in%particle)
n_bin_eff_l = nb_l - 2 - (space_charge_com%particle_bin_span + 1)
if (n_bin_eff_l < 1) then
  err_flag_out = .true.
  return
endif

! Get z min/max from device data
call gpu_csr_z_minmax(z_minval, z_maxval, int(np_l, C_INT))
dz_l = z_maxval - z_minval
if (dz_l == 0) then
  err_flag_out = .true.
  return
endif

csr_in%dz_slice = 1.0000001_rp * dz_l / n_bin_eff_l
z_center_l = (z_maxval + z_minval) / 2
z_min_l = z_center_l - nb_l * csr_in%dz_slice / 2
dz_particle_l = space_charge_com%particle_bin_span * csr_in%dz_slice

! Allocate bin arrays
if (allocated(csr_in%slice)) then
  if (size(csr_in%slice, 1) < nb_l) deallocate (csr_in%slice)
endif
if (.not. allocated(csr_in%slice)) &
    allocate (csr_in%slice(nb_l), csr_in%kick1(-nb_l:nb_l))
csr_in%slice(:) = csr_bunch_slice_struct()

! Fill in z geometry
do ii = 1, nb_l
  csr_in%slice(ii)%z0_edge  = z_min_l + (ii - 1) * csr_in%dz_slice
  csr_in%slice(ii)%z_center = csr_in%slice(ii)%z0_edge + csr_in%dz_slice / 2
  csr_in%slice(ii)%z1_edge  = csr_in%slice(ii)%z0_edge + csr_in%dz_slice
enddo

! Prepare per-particle charge array (host)
allocate(h_charge_l(np_l))
do ii = 1, np_l
  if (bunch_in%particle(ii)%state == alive$) then
    h_charge_l(ii) = bunch_in%particle(ii)%charge * charge_of(bunch_in%particle(ii)%species)
  else
    h_charge_l(ii) = 0
  endif
enddo

! GPU binning kernel (reads vz, vx, vy, state from device; outputs bin arrays to host)
allocate(h_bin_charge_l(nb_l), h_bin_x0_wt_l(nb_l), h_bin_y0_wt_l(nb_l), h_bin_n_particle_l(nb_l))
call gpu_csr_bin_particles(h_charge_l, int(np_l, C_INT), &
    h_bin_charge_l, h_bin_x0_wt_l, h_bin_y0_wt_l, h_bin_n_particle_l, &
    z_min_l, csr_in%dz_slice, dz_particle_l, &
    int(nb_l, C_INT), int(space_charge_com%particle_bin_span, C_INT))

! Copy results into csr%slice
do ii = 1, nb_l
  csr_in%slice(ii)%charge = h_bin_charge_l(ii)
  if (h_bin_n_particle_l(ii) > 0 .and. h_bin_charge_l(ii) /= 0) then
    csr_in%slice(ii)%x0 = h_bin_x0_wt_l(ii) / h_bin_charge_l(ii)
    csr_in%slice(ii)%y0 = h_bin_y0_wt_l(ii) / h_bin_charge_l(ii)
  endif
  csr_in%slice(ii)%n_particle = h_bin_n_particle_l(ii)
enddo

deallocate(h_charge_l, h_bin_charge_l, h_bin_x0_wt_l, h_bin_y0_wt_l, h_bin_n_particle_l)

end subroutine csr_bin_particles_gpu

!------------------------------------------------------------------------
! GPU-accelerated CSR kick application.
! Uploads bin-level kick arrays to device, applies kicks on GPU.
!------------------------------------------------------------------------
subroutine csr_apply_kicks_gpu(ele_in, csr_in, bunch_in)
use gpu_tracking_mod, only: gpu_csr_apply_kicks, gpu_space_charge_3d, gpu_persist_on_device
use, intrinsic :: iso_c_binding
type (ele_struct), intent(in) :: ele_in
type (csr_struct), target, intent(in) :: csr_in
type (bunch_struct), target, intent(inout) :: bunch_in

real(C_DOUBLE), allocatable :: h_kick_csr_l(:), h_kick_lsc_l(:), h_charge_l(:)
integer :: ii, np_l, nb_l
integer(C_INT) :: apply_csr_l, apply_lsc_l
real(rp) :: mc2_l

nb_l = space_charge_com%n_bin
np_l = size(bunch_in%particle)

! CSR/LSC kicks
allocate(h_kick_csr_l(nb_l), h_kick_lsc_l(nb_l))
do ii = 1, nb_l
  h_kick_csr_l(ii) = csr_in%slice(ii)%kick_csr
  h_kick_lsc_l(ii) = 0
enddo

apply_csr_l = 0; apply_lsc_l = 0
if (ele_in%csr_method == one_dim$) apply_csr_l = 1
if (ele_in%space_charge_method == slice$) apply_lsc_l = 1

call gpu_csr_apply_kicks(h_kick_csr_l, h_kick_lsc_l, &
    csr_in%slice(1)%z_center, csr_in%dz_slice, &
    apply_csr_l, apply_lsc_l, &
    int(nb_l, C_INT), int(np_l, C_INT))

deallocate(h_kick_csr_l, h_kick_lsc_l)

! 3D FFT space charge (on device — no upload/download needed)
if (ele_in%space_charge_method == fft_3d$ .and. gpu_persist_on_device) then
  ! Cache charge array — it doesn't change between sub-steps.
  ! Use module-level saved array to avoid per-sub-step allocation + 1M-particle loop.
  block
    real(C_DOUBLE), allocatable, save :: h_charge_cached(:)
    integer, save :: h_charge_cached_n = 0
    if (np_l /= h_charge_cached_n) then
      if (allocated(h_charge_cached)) deallocate(h_charge_cached)
      allocate(h_charge_cached(np_l))
      do ii = 1, np_l
        h_charge_cached(ii) = bunch_in%particle(ii)%charge
      enddo
      h_charge_cached_n = np_l
    endif
    mc2_l = mass_of(bunch_in%particle(1)%species)
    call gpu_space_charge_3d(h_charge_cached, int(np_l, C_INT), &
        int(csr_in%mesh3d%nhi(1), C_INT), int(csr_in%mesh3d%nhi(2), C_INT), &
        int(csr_in%mesh3d%nhi(3), C_INT), &
        csr_in%mesh3d%gamma, csr_in%actual_track_step, mc2_l, 1.0e31_rp)
  end block
endif

end subroutine csr_apply_kicks_gpu

!------------------------------------------------------------------------
! GPU-accelerated CSR bin kicks.
! Uploads element geometry to device, runs parallel root-finding for all
! kick1 bins, then computes convolution to get per-slice kick_csr.
!------------------------------------------------------------------------
subroutine csr_bin_kicks_gpu_wrap(ele_in, csr_in, err_flag_out)
use gpu_tracking_mod, only: gpu_csr_bin_kicks
use, intrinsic :: iso_c_binding
type (ele_struct), intent(in) :: ele_in
type (csr_struct), target, intent(inout) :: csr_in
logical, intent(out) :: err_flag_out

real(C_DOUBLE), allocatable :: h_f0x(:), h_f0z(:), h_f0t(:)
real(C_DOUBLE), allocatable :: h_f1x(:), h_f1z(:), h_f1t(:)
real(C_DOUBLE), allocatable :: h_lc(:), h_tc(:), h_spl(:), h_dls(:), h_es(:)
integer(C_INT), allocatable :: h_ek(:)
real(C_DOUBLE), allocatable :: h_edge_dcdz(:), h_slice_charge(:)
real(C_DOUBLE), allocatable :: h_kick_csr_l(:), h_I_csr_out(:)
integer :: ie, nb, n_ele_geom, n_kick1
real(rp) :: species_radius, coef

err_flag_out = .false.
nb = space_charge_com%n_bin
n_ele_geom = size(csr_in%eleinfo) - 1  ! 0:n_ele_geom
n_kick1 = 2 * nb + 1

! Build geometry arrays from csr%eleinfo
allocate(h_f0x(0:n_ele_geom), h_f0z(0:n_ele_geom), h_f0t(0:n_ele_geom))
allocate(h_f1x(0:n_ele_geom), h_f1z(0:n_ele_geom), h_f1t(0:n_ele_geom))
allocate(h_lc(0:n_ele_geom), h_tc(0:n_ele_geom))
allocate(h_spl(3*(n_ele_geom+1)))
allocate(h_dls(0:n_ele_geom), h_es(0:n_ele_geom))
allocate(h_ek(0:n_ele_geom))

do ie = 0, n_ele_geom
  h_f0x(ie) = csr_in%eleinfo(ie)%floor0%r(1)
  h_f0z(ie) = csr_in%eleinfo(ie)%floor0%r(3)
  h_f0t(ie) = csr_in%eleinfo(ie)%floor0%theta
  h_f1x(ie) = csr_in%eleinfo(ie)%floor1%r(1)
  h_f1z(ie) = csr_in%eleinfo(ie)%floor1%r(3)
  h_f1t(ie) = csr_in%eleinfo(ie)%floor1%theta
  h_lc(ie) = csr_in%eleinfo(ie)%L_chord
  h_tc(ie) = csr_in%eleinfo(ie)%theta_chord
  h_spl(3*ie+1) = csr_in%eleinfo(ie)%spline%coef(1)
  h_spl(3*ie+2) = csr_in%eleinfo(ie)%spline%coef(2)
  h_spl(3*ie+3) = csr_in%eleinfo(ie)%spline%coef(3)
  h_dls(ie) = csr_in%eleinfo(ie)%dL_s
  h_es(ie) = csr_in%eleinfo(ie)%ele%s
  h_ek(ie) = csr_in%eleinfo(ie)%ele%key
enddo

! Build bin data arrays
allocate(h_edge_dcdz(nb), h_slice_charge(nb))
do ie = 1, nb
  h_edge_dcdz(ie) = csr_in%slice(ie)%edge_dcharge_density_dz
  h_slice_charge(ie) = csr_in%slice(ie)%charge
enddo

species_radius = classical_radius(csr_in%species)

allocate(h_kick_csr_l(nb), h_I_csr_out(n_kick1))
h_kick_csr_l = 0

call gpu_csr_bin_kicks( &
    h_f0x, h_f0z, h_f0t, h_f1x, h_f1z, h_f1t, &
    h_lc, h_tc, h_spl, h_dls, h_es, h_ek, &
    int(n_ele_geom, C_INT), &
    int(csr_in%ix_ele_kick, C_INT), &
    csr_in%s_chord_kick, &
    csr_in%floor_k%r(1), csr_in%floor_k%r(3), &
    csr_in%gamma, csr_in%gamma2, csr_in%beta**2, &
    csr_in%y_source, csr_in%dz_slice, &
    int(nb, C_INT), csr_in%kick_factor, &
    csr_in%actual_track_step, species_radius, &
    csr_in%rel_mass, abs(e_charge * charge_of(csr_in%species)), &
    merge(1, 0, ele_in%csr_method == one_dim$), &
    h_edge_dcdz, h_slice_charge, &
    h_kick_csr_l, h_I_csr_out)

! Copy results back to csr%slice
if (csr_in%y_source == 0 .and. ele_in%csr_method == one_dim$) then
  do ie = 1, nb
    csr_in%slice(ie)%kick_csr = h_kick_csr_l(ie)
  enddo
else
  ! Image charge: GPU accumulates into h_kick_csr_l
  do ie = 1, nb
    csr_in%slice(ie)%kick_csr = h_kick_csr_l(ie)
  enddo
endif

! Copy I_csr back to kick1 for image charge accumulation
do ie = -nb, nb
  csr_in%kick1(ie)%I_csr = h_I_csr_out(ie + nb + 1)
enddo

deallocate(h_f0x, h_f0z, h_f0t, h_f1x, h_f1z, h_f1t)
deallocate(h_lc, h_tc, h_spl, h_dls, h_es, h_ek)
deallocate(h_edge_dcdz, h_slice_charge, h_kick_csr_l, h_I_csr_out)

end subroutine csr_bin_kicks_gpu_wrap

end subroutine track1_bunch_csr

!----------------------------------------------------------------------------
!----------------------------------------------------------------------------
!----------------------------------------------------------------------------
!+
! Subroutine csr_bin_particles (ele, particle, csr)
!
! Routine to bin the particles longitudinally in s. 
!
! To avoid noise in the cacluation, every particle is considered to have a 
! triangular distribution with a base length  given by
!   space_charge_com%particle_bin_span * csr%dz_slice.
! That is, particles will, in general, overlap multiple bins. 
!
! Input:
!   ele                  -- ele_struct: Element being tracked through.
!   particle(:)          -- coord_struct: Array of particles
!
! Other:
!   space_charge_com     -- space_charge_common_struct: Space charge/CSR common block
!     %n_bin                 -- Number of bins.
!     %particle_bin_span     -- Particle length / dz_slice. 
!
! Output:
!   csr                  -- csr_struct: The bin structure.
!     %dz_slice             -- Bin longitudinal length
!     %slice(1:)            -- Array of bins.
!-

subroutine csr_bin_particles (ele, particle, csr, err_flag)

implicit none

type (ele_struct) ele
type (coord_struct), target :: particle(:)
type (coord_struct), pointer :: p
type (csr_struct), target :: csr
type (csr_bunch_slice_struct), pointer :: slice

real(rp) z_center, z_min, z_max, dz_particle, dz, z_maxval, z_minval, c_tot, n_tot
real(rp) zp_center, zp0, zp1, zb0, zb1, charge, overlap_fraction, charge_tot, sig_x_ave, sig_y_ave, f
integer i, j, n, ix0, ib, ib2, ib_center, n_bin_eff

logical err_flag

character(*), parameter :: r_name = 'csr_bin_particles'

! Init bins...
! The left edge of csr%slice(1) is at z_min
! The right edge of csr%slice(n_bin) is at z_max
! The first and last bins are empty.

err_flag = .false.

if (ele%space_charge_method /= slice$ .and. ele%csr_method /= one_dim$) return

n_bin_eff = space_charge_com%n_bin - 2 - (space_charge_com%particle_bin_span + 1)
if (n_bin_eff < 1) then
  call out_io (s_abort$, r_name, 'NUMBER OF CSR BINS TOO SMALL: \i0\ ', &
              'MUST BE GREATER THAN 3 + PARTICLE_BIN_SPAN.', i_array = [space_charge_com%n_bin])
  if (global_com%exit_on_error) call err_exit
  particle%state = lost$
  err_flag = .true.
  return
endif

z_maxval = maxval(particle(:)%vec(5), mask = (particle(:)%state == alive$))
z_minval = minval(particle(:)%vec(5), mask = (particle(:)%state == alive$))
dz = z_maxval - z_minval
csr%dz_slice = dz / n_bin_eff
csr%dz_slice = 1.0000001 * csr%dz_slice     ! to prevent round off problems
z_center = (z_maxval + z_minval) / 2
z_min = z_center - space_charge_com%n_bin * csr%dz_slice / 2
z_max = z_center + space_charge_com%n_bin * csr%dz_slice / 2
dz_particle = space_charge_com%particle_bin_span * csr%dz_slice

if (dz == 0) then
  call out_io (s_fatal$, r_name, 'LONGITUDINAL WIDTH OF BEAM IS ZERO!')
  if (global_com%exit_on_error) call err_exit
  err_flag = .true.
  return
endif

! allocate memeory for the bins

if (allocated(csr%slice)) then
  if (size(csr%slice, 1) < space_charge_com%n_bin) deallocate (csr%slice)
endif

if (.not. allocated(csr%slice)) &
    allocate (csr%slice(space_charge_com%n_bin), csr%kick1(-space_charge_com%n_bin:space_charge_com%n_bin))

csr%slice(:) = csr_bunch_slice_struct()  ! Zero everything

! Fill in some z information

do i = 1, space_charge_com%n_bin
  csr%slice(i)%z0_edge  = z_min + (i - 1) * csr%dz_slice
  csr%slice(i)%z_center = csr%slice(i)%z0_edge + csr%dz_slice / 2
  csr%slice(i)%z1_edge  = csr%slice(i)%z0_edge + csr%dz_slice
enddo

! Compute the particle distribution center in each bin

! The contribution to the charge in a bin from a particle is computed from the overlap
! between the particle and the bin.
 
c_tot = 0    ! Used for debugging sanity check
do i = 1, size(particle)
  p => particle(i)
  if (p%state /= alive$) cycle
  zp_center = p%vec(5) ! center of particle
  zp0 = zp_center - dz_particle / 2       ! particle left edge 
  zp1 = zp_center + dz_particle / 2       ! particle right edge 
  ix0 = nint((zp0 - z_min) / csr%dz_slice)  ! left most bin index
  do j = 0, space_charge_com%particle_bin_span+1
    ib = j + ix0
    slice => csr%slice(ib)
    zb0 = csr%slice(ib)%z0_edge
    zb1 = csr%slice(ib)%z1_edge   ! edges of the bin
    overlap_fraction = particle_overlap_in_bin (zb0, zb1)
    slice%n_particle = slice%n_particle + overlap_fraction
    charge = overlap_fraction * p%charge
    slice%charge = slice%charge + charge
    slice%x0 = slice%x0 + p%vec(1) * charge
    slice%y0 = slice%y0 + p%vec(3) * charge
    c_tot = c_tot + charge
  enddo
enddo

do ib = 1, space_charge_com%n_bin
  if (ib /= 1) csr%slice(ib)%edge_dcharge_density_dz = (csr%slice(ib)%charge - csr%slice(ib-1)%charge) / csr%dz_slice**2
  if (ib == 1) then
    csr%slice(ib)%dcharge_density_dz = (csr%slice(ib+1)%charge - csr%slice(ib)%charge) / csr%dz_slice**2
  elseif (ib == space_charge_com%n_bin) then
    csr%slice(ib)%dcharge_density_dz = (csr%slice(ib)%charge - csr%slice(ib-1)%charge) / csr%dz_slice**2
  else
    csr%slice(ib)%dcharge_density_dz = (csr%slice(ib+1)%charge - csr%slice(ib-1)%charge) / (2 * csr%dz_slice**2)
  endif

  if (csr%slice(ib)%charge == 0) cycle
  csr%slice(ib)%x0 = csr%slice(ib)%x0 / csr%slice(ib)%charge
  csr%slice(ib)%y0 = csr%slice(ib)%y0 / csr%slice(ib)%charge
enddo

csr%x0_bunch = sum(csr%slice%x0 * csr%slice%charge) / sum(csr%slice%charge)
csr%y0_bunch = sum(csr%slice%y0 * csr%slice%charge) / sum(csr%slice%charge)

! Compute the particle distribution sigmas in each bin.
! Abs(x-x0) is used instead of the usual formula involving (x-x0)^2 to lessen the effect
! of non-Gaussian tails.

do i = 1, size(particle)
  p => particle(i)
  if (p%state /= alive$) cycle
  zp_center = p%vec(5) ! center of particle
  zp0 = zp_center - dz_particle / 2       ! particle left edge 
  zp1 = zp_center + dz_particle / 2       ! particle right edge 
  ix0 = nint((zp0 - z_min) / csr%dz_slice)  ! left most bin index
  do j = 0, space_charge_com%particle_bin_span+1
    ib = j + ix0
    slice => csr%slice(ib)
    zb0 = csr%slice(ib)%z0_edge
    zb1 = csr%slice(ib)%z1_edge   ! edges of the bin
    overlap_fraction = particle_overlap_in_bin (zb0, zb1)
    charge = overlap_fraction * p%charge
    slice%sig_x = slice%sig_x + abs(p%vec(1) - slice%x0) * charge
    slice%sig_y = slice%sig_y + abs(p%vec(3) - slice%y0) * charge
  enddo
enddo

charge_tot = 0;  sig_x_ave = 0;  sig_y_ave = 0
f = sqrt(pi/2)  ! This corrects for the fact that |x - x0| is used instead of (x - x0)^2 to compute the sigma.
do ib = 1, space_charge_com%n_bin
  slice => csr%slice(ib)
  if (slice%n_particle < space_charge_com%sc_min_in_bin) cycle
  slice%sig_x = f * slice%sig_x / slice%charge
  slice%sig_y = f * slice%sig_y / slice%charge
  charge_tot = charge_tot + slice%charge
  sig_x_ave = slice%sig_x * slice%charge
  sig_y_ave = slice%sig_y * slice%charge
enddo

if (charge_tot == 0) then   ! Not enough particles for calc
  csr%slice%sig_x = 1  ! Something large
  csr%slice%sig_y = 1  ! Something large
  return
endif

sig_x_ave = sig_x_ave / charge_tot
sig_y_ave = sig_y_ave / charge_tot

! At the ends, for bins where there are not enough particles to calculate sigma, use
! the sigmas of nearest bin that has a valid sigmas

do ib = space_charge_com%n_bin/2, space_charge_com%n_bin
  slice => csr%slice(ib)
  if (slice%n_particle < space_charge_com%sc_min_in_bin) then
    slice%sig_x = csr%slice(ib-1)%sig_x
    slice%sig_y = csr%slice(ib-1)%sig_y
  else
    if (slice%sig_x < sig_x_ave * space_charge_com%lsc_sigma_cutoff) slice%sig_x = sig_x_ave * space_charge_com%lsc_sigma_cutoff
    if (slice%sig_y < sig_y_ave * space_charge_com%lsc_sigma_cutoff) slice%sig_y = sig_y_ave * space_charge_com%lsc_sigma_cutoff
  endif
enddo

do ib = space_charge_com%n_bin/2, 1, -1
  slice => csr%slice(ib)
  if (slice%n_particle < space_charge_com%sc_min_in_bin) then
    slice%sig_x = csr%slice(ib+1)%sig_x
    slice%sig_y = csr%slice(ib+1)%sig_y
  else
    if (slice%sig_x < sig_x_ave * space_charge_com%lsc_sigma_cutoff) slice%sig_x = sig_x_ave * space_charge_com%lsc_sigma_cutoff
    if (slice%sig_y < sig_y_ave * space_charge_com%lsc_sigma_cutoff) slice%sig_y = sig_y_ave * space_charge_com%lsc_sigma_cutoff
  endif
enddo


!---------------------------------------------------------------------------
contains

! computes the contribution to the charge in a bin from a given particle.
! z0_bin, z1_bin are the edge positions of the bin

function particle_overlap_in_bin (z0_bin, z1_bin) result (overlap)

real(rp) z0_bin, z1_bin, overlap, z1, z2

! Integrate over left triangular half of particle distribution

z1 = max(zp0, z0_bin)        ! left integration edge
z2 = min(zp_center, z1_bin)  ! right integration edge
if (z2 > z1) then            ! If left particle half is in bin ...
  overlap = 2 * real(((z2 - zp0)**2 - (z1 - zp0)**2), rp) / dz_particle**2
else
  overlap = 0
endif

! Integrate over right triangular half of particle distribution

z1 = max(zp_center, z0_bin)  ! left integration edge
z2 = min(zp1, z1_bin)        ! right integration edge
if (z2 > z1) then            ! If right particle half is in bin ...
  overlap = overlap + 2 * real(((z1 - zp1)**2 - (z2 - zp1)**2), rp) / dz_particle**2
endif

end function particle_overlap_in_bin

end subroutine csr_bin_particles

!----------------------------------------------------------------------------
!----------------------------------------------------------------------------
!----------------------------------------------------------------------------
!+
! Subroutine csr_bin_kicks (ele, ds_kick_pt, csr, err_flag)
!
! Routine to cache intermediate values needed for the csr calculations.
!
! Input:
!   ele          -- ele_struct: Element being tracked through.
!   ds_kick_pt   -- real(rp): Distance between the beginning of the element we are
!                    tracking through and the kick point (which is within this element).
!   csr          -- csr_struct: 
!
! Output:
!   csr          -- csr_struct: 
!     %kick1(:)          -- CSR kick calculation bin array. 
!     %slice(:)%kick_csr -- Integrated kick
!   err_flag     -- logical: Set True if there is an error. False otherwise
!-

subroutine csr_bin_kicks (ele, ds_kick_pt, csr, err_flag)

implicit none

type (ele_struct) ele
type (csr_struct), target :: csr
type (branch_struct), pointer :: branch
type (csr_kick1_struct), pointer :: kick1

real(rp) ds_kick_pt, coef, dr_match(3)

integer i, n_bin
logical err_flag

character(16) :: r_name = 'csr_bin_kicks'

! The kick point P is fixed.
! Loop over all kick1 bins and compute the kick.

err_flag = .false.

do i = lbound(csr%kick1, 1), ubound(csr%kick1, 1)

  kick1 => csr%kick1(i)
  kick1%dz_particles = i * csr%dz_slice

  if (i == lbound(csr%kick1, 1)) then
    kick1%ix_ele_source = csr%ix_ele_kick  ! Initial guess where source point is
    dr_match = 0  ! Discontinuity factor for match element. See s_source_calc routine.
  else
    kick1%ix_ele_source = csr%kick1(i-1)%ix_ele_source
  endif

  ! Calculate what element the source point is in.

  kick1%s_chord_source = s_source_calc(kick1, csr, err_flag, dr_match)
  if (err_flag) return

  ! calculate csr.
  ! I_csr is only calculated for particles with y = 0 and not for image currents.

  if (csr%y_source == 0) then
    call I_csr (kick1, i, csr)
  else
    call image_charge_kick_calc (kick1, csr)
  endif

enddo

!

coef = csr%actual_track_step * classical_radius(csr%species) / &
                              (csr%rel_mass * e_charge * abs(charge_of(csr%species)) * csr%gamma)
n_bin = space_charge_com%n_bin

! CSR & Image charge kick

if (csr%y_source == 0) then
  if (ele%csr_method == one_dim$) then
    do i = 1, n_bin
      csr%slice(i)%kick_csr = coef * dot_product(csr%kick1(i:1:-1)%I_int_csr, csr%slice(1:i)%edge_dcharge_density_dz)
    enddo
  endif

else  ! Image charge
  do i = 1, n_bin
    csr%slice(i)%kick_csr = csr%slice(i)%kick_csr + coef * &
                  dot_product(csr%kick1(i-1:i-n_bin:-1)%image_kick_csr, csr%slice(1:n_bin)%charge)
  enddo
endif

! Longitudinal space charge kick
! Note: image charges do not contribute to LSC.

if (ele%space_charge_method == slice$ .and. csr%y_source == 0) then
  call lsc_kick_params_calc (ele, csr)
endif

end subroutine csr_bin_kicks
  
!----------------------------------------------------------------------------
!----------------------------------------------------------------------------
!----------------------------------------------------------------------------
!+
! Function s_source_calc (kick1, csr, err_flag, dr_match) result (s_source)
!
! Routine to calculate the distance between source and kick points.
!
! Input:
!   kick1         -- csr_kick1_struct:
!   csr           -- csr_struct:
!   dr_match(3)   -- real(rp): Discontinuity factor if there is a match element between source and kick elements.
!
! Output:
!   s_source      -- real(rp): source s-position.
!   kick1         -- csr_kick1_struct:
!   err_flag      -- logical: Set True if there is an error. Untouched otherwise.
!   dr_match(3)   -- real(rp): Discontinuity factor if there is a match element between source and kick elements.
!-

function s_source_calc (kick1, csr, err_flag, dr_match) result (s_source)

implicit none

type (csr_kick1_struct), target :: kick1
type (csr_struct), target :: csr
type (csr_ele_info_struct), pointer :: einfo_s, einfo_k
type (floor_position_struct), pointer :: fk, f0, fs
type (ele_struct), pointer :: ele

real(rp) a, b, c, dz, s_source, beta2, L0, Lz, ds_source
real(rp) z0, z1, sz_kick, sz0, Lsz0, ddz0, ddz1, dr_match(3)

integer i, last_step, status
logical err_flag

character(*), parameter :: r_name = 's_source_calc'

! Each interation of the loop looks for a possible source point in the lattice element with 
! index kick1%ix_ele_source. If found, return. If not, move on to another element

dz = kick1%dz_particles   ! Target distance.
beta2 = csr%beta**2
last_step = 0
s_source = 0        ! To prevent uninitalized complaints from the compiler.

do

  einfo_s => csr%eleinfo(kick1%ix_ele_source)

  ! If at beginning of lattice assume an infinite drift.
  ! s_source will be negative

  if (kick1%ix_ele_source == 0) then
    fk => csr%floor_k
    fs => kick1%floor_s
    f0 => einfo_s%floor1

    L0 = sqrt((fk%r(1) - f0%r(1))**2 + (fk%r(3) - f0%r(3))**2 + csr%y_source**2)
    ! L_z is the z-component from lat start to the kick point.
    Lz = (fk%r(1) - f0%r(1)) * sin(f0%theta) + (fk%r(3) - f0%r(3)) * cos(f0%theta) 
    ! Lsz0 is Ls from the lat start to the kick point
    Lsz0 = dspline_len(0.0_rp, csr%s_chord_kick, csr%eleinfo(csr%ix_ele_kick)%spline) + csr%s_chord_kick
    do i = 1, csr%ix_ele_kick - 1
      Lsz0 = Lsz0 + csr%eleinfo(i)%dL_s + csr%eleinfo(i)%L_chord
    enddo

    a = 1/csr%gamma2
    b = 2 * (Lsz0 - dz - beta2 * Lz)
    c = (Lsz0 - dz)**2 - beta2 * L0**2
    ds_source = -(-b + sqrt(b**2 - 4 * a * c)) / (2 * a)
    s_source = einfo_s%ele%s + ds_source

    fs%r = [f0%r(1) + ds_source * sin(f0%theta), csr%y_source, f0%r(3) + ds_source * cos(f0%theta)]
    fs%theta = f0%theta
    kick1%L_vec = fk%r - fs%r
    kick1%L = sqrt(dot_product(kick1%L_vec, kick1%L_vec))
    kick1%dL = lsz0 - ds_source - kick1%L  ! Remember s_source is negative
    kick1%theta_L = atan2(kick1%L_vec(1), kick1%L_vec(3))
    kick1%theta_sl = f0%theta - kick1%theta_L
    einfo_k => csr%eleinfo(csr%ix_ele_kick)
    kick1%theta_lk = kick1%theta_L - (spline1(einfo_k%spline, csr%s_chord_kick, 1) + einfo_k%theta_chord)
    return
  endif

  ! Match elements can have an orbit discontinuity which is non-physical.
  ! So if there is a match element then shift the orbit using dr_match to remove the discontinuity.
  ! Any angle discontinuity is ignored.

  ele => einfo_s%ele

  if (ele%key == floor_shift$) then
    call out_io (s_fatal$, r_name, &
        'CSR CALC IS NOT ABLE TO HANDLE A FLOOR_SHIFT ELEMENT: ' // ele%name, &
        'WHILE TRACKING THROUGH ELEMENT: ' // csr%kick_ele%name)
    if (global_com%exit_on_error) call err_exit
    err_flag = .true.
    return
  endif

  ! Look at ends of the source element and check if the source point is within the element or not.
  ! Generally dz decreases with increasing s but this may not be true for patch elements.

  ddz0 = ddz_calc_csr(0.0_rp, status)
  ddz1 = ddz_calc_csr(einfo_s%L_chord, status)

  if (last_step == -1 .and. ddz1 > 0) then  ! Roundoff error is causing ddz1 to be positive.
    s_source = ddz_calc_csr(einfo_s%L_chord, status)
    return
  endif

  if (last_step == 1 .and. ddz0 < 0) then  ! Roundoff error is causing ddz0 to be negative.
    s_source = ddz_calc_csr(0.0_rp, status)
    return
  endif

  if (ddz0 < 0 .and. ddz1 < 0) then
    if (last_step == 1) exit       ! Round off error can cause problems
    last_step = -1
    kick1%ix_ele_source = kick1%ix_ele_source - 1

    einfo_s => csr%eleinfo(kick1%ix_ele_source)
    if (einfo_s%ele%key == match$) then
      dr_match = einfo_s%floor1%r - einfo_s%floor0%r ! discontinuity in x.
      kick1%ix_ele_source = kick1%ix_ele_source - 1
    endif

    cycle
  endif

  if (ddz0 > 0 .and. ddz1 > 0) then
    if (kick1%ix_ele_source == csr%ix_ele_kick) return  ! Source ahead of kick pt => ignore.
    if (last_step == -1) exit      ! Round off error can cause problems
    last_step = 1
    kick1%ix_ele_source = kick1%ix_ele_source + 1

    if (csr%eleinfo(kick1%ix_ele_source)%ele%key == match$) then
      dr_match = 0
      kick1%ix_ele_source = kick1%ix_ele_source + 1
    endif

    cycle
  endif

  ! Only possibility left is that root is bracketed.

  s_source = super_zbrent (ddz_calc_csr, 0.0_rp, einfo_s%L_chord, 1e-12_rp, 1e-8_rp, status)
  return
    
enddo

call out_io (s_fatal$, r_name, &
    'CSR CALCULATION ERROR. PLEASE REPORT THIS...', &
    'WHILE TRACKING THROUGH ELEMENT: ', csr%kick_ele%name)
if (global_com%exit_on_error) call err_exit
err_flag = .true.

!----------------------------------------------------------------------------
contains

!+
! Function ddz_calc_csr (s_chord_source, status) result (ddz_this)
!
! Routine to calculate the distance between the source particle and the
! kicked particle at constant time minus the target distance.
!
! Input:
!   s_chord_source  -- real(rp): Chord distance from start of element.
!
! Output:
!   ddz_this        -- real(rp): Distance between source and kick particles: Calculated - Wanted.
!-

function ddz_calc_csr (s_chord_source, status) result (ddz_this)

implicit none

type (csr_ele_info_struct), pointer :: ce
real(rp), intent(in) :: s_chord_source
real(rp) ddz_this, x, z, c, s, dtheta_L
real(rp) s0, s1, ds, theta_L, dL

integer status
integer i

character(*), parameter :: r_name = 'ddz_calc_csr'

! 

x = spline1(einfo_s%spline, s_chord_source)
c = cos(einfo_s%theta_chord)
s = sin(einfo_s%theta_chord)
kick1%floor_s%r = [x*c + s_chord_source*s, csr%y_source, -x*s + s_chord_source*c] + einfo_s%floor0%r + dr_match   ! Floor coordinates

kick1%L_vec = csr%floor_k%r - kick1%floor_s%r
kick1%L = sqrt(dot_product(kick1%L_vec, kick1%L_vec))
kick1%theta_L = atan2(kick1%L_vec(1), kick1%L_vec(3))

s0 = s_chord_source
s1 = csr%s_chord_kick

if (kick1%ix_ele_source == csr%ix_ele_kick) then
  ds = s1 - s0
  ! dtheta_L = angle of L line in centroid chord ref frame
  dtheta_L = einfo_s%spline%coef(1) + einfo_s%spline%coef(2) * (2*s0 + ds) + einfo_s%spline%coef(3) * (3*s0**2 + 3*s0*ds + ds**2)
  dL = dspline_len(s0, s1, einfo_s%spline, dtheta_L) ! = Ls - L
  ! Ls is negative if the source pt is ahead of the kick pt (ds < 0). But L is always positive. 
  if (ds < 0) dL = dL + 2 * ds    ! Correct for L always being positive.
  kick1%theta_sl = spline1(einfo_s%spline, s0, 1) - dtheta_L
  kick1%theta_lk = dtheta_L - spline1(einfo_s%spline, s1, 1)

else
  ! In an element where the beam centroid takes a backstep, %theta_chord will be anti-parallel to
  ! other angles. This is why pi/2 is used with modulo2 to make sure angles are in the range [-pi/2, pi/2]
  theta_L = kick1%theta_L
  dL = dspline_len(s0, einfo_s%L_chord, einfo_s%spline, modulo2(theta_L-einfo_s%theta_chord, pi/2))
  do i = kick1%ix_ele_source+1, csr%ix_ele_kick-1
    ce => csr%eleinfo(i)
    if (ce%ele%key == match$) cycle  ! Match elements are adjusted to give zero displacement.
    dL = dL + dspline_len(0.0_rp, ce%L_chord, ce%spline, modulo2(theta_L-ce%theta_chord, pi/2))
  enddo
  ce => csr%eleinfo(csr%ix_ele_kick)
  dL = dL + dspline_len(0.0_rp, s1, ce%spline, modulo2(theta_L-ce%theta_chord, pi/2))
  kick1%theta_sl = modulo2((spline1(einfo_s%spline, s0, 1) + einfo_s%theta_chord) - theta_L, pi/2)
  kick1%theta_lk = modulo2(theta_L - (spline1(ce%spline, s1, 1) + ce%theta_chord), pi/2)
endif

kick1%floor_s%theta = kick1%theta_sl + kick1%theta_L

! The above calc for dL neglected csr%y_source. So must correct for this.

if (csr%y_source /= 0) dL = dL - (kick1%L - sqrt(kick1%L_vec(1)**2 + kick1%L_vec(3)**2))
kick1%dL = dL
ddz_this = kick1%L / (2 * csr%gamma2) + dL
ddz_this = ddz_this - kick1%dz_particles

end function ddz_calc_csr

end function s_source_calc

!----------------------------------------------------------------------------
!----------------------------------------------------------------------------
!----------------------------------------------------------------------------
!+
! Subroutine lsc_kick_params_calc (ele, csr)
!
! Routine to cache intermediate values needed for the lsc calculation.
! This routine is not for image currents.
!
! Input:
!   ele       -- Element_struct: Element to set up cache for.
!   csr       -- csr_struct: 
!     %slice(:)   -- bin array of particle averages.
!
! Output:
!   csr     -- csr_struct: Binned particle averages.
!     %slice(:)   -- bin array of particle averages.
!-

subroutine lsc_kick_params_calc (ele, csr)

use da2_mod

implicit none

type (ele_struct) ele
type (csr_struct), target :: csr

real(rp), pointer :: f(:,:)
real(rp) factor, f00, c_val, dz_half, z_center_i
real(rp) sx2, sy2, g_z1_s, g_z2_s, h_z1_s, h_z2_s, alph, bet, ss, f1(0:2,0:2)

integer i, j, n_bin

! Vectorized working arrays for inner loop over source slices
real(rp), allocatable :: sx_v(:), sy_v(:), a_v(:), b_v(:)
real(rp), allocatable :: charge_v(:), dcdz_v(:), z_center_v(:)
real(rp), allocatable :: abs_z_v(:), z1_v(:), z2_v(:), rho0_v(:), drho_v(:)
real(rp), allocatable :: radix_v(:), sr_v(:)
real(rp), allocatable :: b2cz1_v(:), b2cz2_v(:), abcz1_v(:), abcz2_v(:)
real(rp), allocatable :: atz1_v(:), atz2_v(:), bcd_v(:), dk0_v(:)
integer, allocatable :: sign_v(:)

character(*), parameter :: r_name = 'lsc_kick_params_calc'

!

if (ele%space_charge_method /= slice$) return

n_bin = space_charge_com%n_bin

factor = csr%kick_factor * csr%actual_track_step * classical_radius(csr%species) / &
                              (csr%rel_mass * e_charge * abs(charge_of(csr%species)) * csr%gamma)

c_val = csr%gamma**2
dz_half = csr%dz_slice / 2

! Precompute j-dependent arrays (source slice properties, independent of kick index i)

allocate(sx_v(n_bin), sy_v(n_bin), a_v(n_bin), b_v(n_bin))
allocate(charge_v(n_bin), dcdz_v(n_bin), z_center_v(n_bin))
allocate(abs_z_v(n_bin), z1_v(n_bin), z2_v(n_bin), rho0_v(n_bin), drho_v(n_bin))
allocate(radix_v(n_bin), sr_v(n_bin))
allocate(b2cz1_v(n_bin), b2cz2_v(n_bin), abcz1_v(n_bin), abcz2_v(n_bin))
allocate(atz1_v(n_bin), atz2_v(n_bin), bcd_v(n_bin), dk0_v(n_bin))
allocate(sign_v(n_bin))

do j = 1, n_bin
  sx_v(j) = csr%slice(j)%sig_x
  sy_v(j) = csr%slice(j)%sig_y
  charge_v(j) = csr%slice(j)%charge
  dcdz_v(j) = csr%slice(j)%dcharge_density_dz
  z_center_v(j) = csr%slice(j)%z_center
enddo

! Quantities that depend only on the source slice (j), not on kick slice (i)
a_v = sx_v * sy_v
b_v = csr%gamma * (sx_v**2 + sy_v**2) / (sx_v + sy_v)
radix_v = -b_v**2 + 4 * a_v * c_val
sr_v = sqrt(abs(radix_v))

! Compute the kick at the center of each bin
! i = index of slice where kick is computed

do i = 1, n_bin
  z_center_i = csr%slice(i)%z_center

  ! Vectorized computation of z-separations and signs
  abs_z_v = abs(z_center_i - z_center_v)

  do j = 1, n_bin
    if (z_center_i > z_center_v(j)) then
      sign_v(j) = 1
    elseif (z_center_i < z_center_v(j)) then
      sign_v(j) = -1
    else
      sign_v(j) = 0
    endif
  enddo

  ! General case: compute z1, z2, drho, rho0 for all source slices
  z1_v = abs_z_v - dz_half
  z2_v = abs_z_v + dz_half
  drho_v = dcdz_v * sign_v
  rho0_v = charge_v / csr%dz_slice - drho_v * abs_z_v

  ! Self-slice override (diagonal: i == j)
  drho_v(i) = dcdz_v(i)
  rho0_v(i) = 0
  z1_v(i) = 0
  z2_v(i) = dz_half
  sign_v(i) = -2        ! Factor of 2 accounts for 1/2 we did not integrate over.

  ! Vectorized intermediate computations
  bcd_v = 2 * c_val * rho0_v - b_v * drho_v
  abcz1_v = a_v + b_v * z1_v + c_val * z1_v**2
  abcz2_v = a_v + b_v * z2_v + c_val * z2_v**2
  b2cz1_v = b_v + 2 * c_val * z1_v
  b2cz2_v = b_v + 2 * c_val * z2_v

  ! Compute atan for all elements (always safe), then override with log where radix <= 0
  atz1_v = atan(b2cz1_v / sr_v) / sr_v
  atz2_v = atan(b2cz2_v / sr_v) / sr_v

  do j = 1, n_bin
    if (radix_v(j) <= 0) then
      atz1_v(j) = log((b2cz1_v(j) - sr_v(j)) / (b2cz1_v(j) + sr_v(j))) / (2 * sr_v(j))
      atz2_v(j) = log((b2cz2_v(j) - sr_v(j)) / (b2cz2_v(j) + sr_v(j))) / (2 * sr_v(j))
    endif
  enddo

  ! Vectorized kick computation
  dk0_v = factor * ((2 * atz2_v * bcd_v + drho_v * log(abcz2_v)) - &
                     (2 * atz1_v * bcd_v + drho_v * log(abcz1_v))) / (2 * c_val)

  ! Accumulate kick_lsc (sum over all source slices)
  csr%slice(i)%kick_lsc = csr%slice(i)%kick_lsc + sum(sign_v * dk0_v)

  ! Transverse dependence: scalar loop required for DA2 accumulation
  if (space_charge_com%lsc_kick_transverse_dependence) then
    do j = 1, n_bin
      if (dk0_v(j) == 0) cycle

      f00 = 1 / dk0_v(j)
      sx2 = sx_v(j)**2
      sy2 = sy_v(j)**2

      g_z1_s = a_v(j) * (b2cz1_v(j)*bcd_v(j) + 4*abcz1_v(j)*atz1_v(j)*bcd_v(j)*c_val - drho_v(j)*radix_v(j)) / &
               (2*abcz1_v(j)*c_val*radix_v(j))
      g_z2_s = a_v(j) * (b2cz2_v(j)*bcd_v(j) + 4*abcz2_v(j)*atz2_v(j)*bcd_v(j)*c_val - drho_v(j)*radix_v(j)) / &
               (2*abcz2_v(j)*c_val*radix_v(j))
      h_z1_s = (b_v(j)*b_v(j)*b2cz1_v(j)*bcd_v(j) - 6*a_v(j)*b2cz1_v(j)*bcd_v(j)*c_val - &
                4*abcz1_v(j)*b2cz1_v(j)*bcd_v(j)*c_val - 24*abcz1_v(j)**2*atz1_v(j)*bcd_v(j)*c_val**2 + &
                drho_v(j)*radix_v(j)**2 - 2*b_v(j)*b2cz1_v(j)*bcd_v(j)*c_val*z1_v(j) - &
                2*b2cz1_v(j)*bcd_v(j)*(c_val*z1_v(j))**2)
      h_z2_s = (b_v(j)*b_v(j)*b2cz2_v(j)*bcd_v(j) - 6*a_v(j)*b2cz2_v(j)*bcd_v(j)*c_val - &
                4*abcz2_v(j)*b2cz2_v(j)*bcd_v(j)*c_val - 24*abcz2_v(j)**2*atz2_v(j)*bcd_v(j)*c_val**2 + &
                drho_v(j)*radix_v(j)**2 - 2*b_v(j)*b2cz2_v(j)*bcd_v(j)*c_val*z2_v(j) - &
                2*b2cz2_v(j)*bcd_v(j)*(c_val*z2_v(j))**2)

      alph = f00**2 * (g_z2_s - g_z1_s)
      bet = f00**3 * (g_z2_s - g_z1_s)**2 + (f00 * a_v(j))**2 * &
            (h_z2_s/abcz2_v(j)**2 - h_z1_s/abcz1_v(j)**2) / (4 * c_val * radix_v(j)**2)

      ss = 1.0_rp / sign_v(j)
      f1(0,0) = ss * f00
      f1(0,1) = ss * alph / (2 * sy2)
      f1(0,2) = ss * (alph + 2*bet) / (8 * sy2**2)
      f1(1,0) = ss * alph / (2 * sx2)
      f1(1,1) = ss * (alph + 2*bet) / (4 * sx2 * sy2)
      f1(2,0) = ss * (alph + 2*bet) / (8 * sx2**2)

      if (f1(0,0) > 0) then
        f => csr%slice(i)%coef_lsc_plus
      else
        f => csr%slice(i)%coef_lsc_minus
      endif

      if (f(0,0) == 0) then  ! If first time
        f = f1
      else
        f = da2_div(da2_mult(f1, f), f + f1)
      endif
    enddo
  endif

enddo

deallocate(sx_v, sy_v, a_v, b_v, charge_v, dcdz_v, z_center_v)
deallocate(abs_z_v, z1_v, z2_v, rho0_v, drho_v, radix_v, sr_v)
deallocate(b2cz1_v, b2cz2_v, abcz1_v, abcz2_v, atz1_v, atz2_v, bcd_v, dk0_v)
deallocate(sign_v)

end subroutine lsc_kick_params_calc

!----------------------------------------------------------------------------
!----------------------------------------------------------------------------
!----------------------------------------------------------------------------
!+
! Subroutine I_csr (kick1, i_bin, csr) 
!
! Routine to calculate the CSR kick integral (at y = 0)
!
! Input:
!   kick1      -- csr_kick1_struct: 
!   i_bin      -- integer: Bin index.
!   csr    -- csr_struct:
!
! Output:
!   kick1     -- csr_kick1_struct: 
!     %I_csr     -- real(rp): CSR kick integral.
!     %I_int_csr -- real(rp): Integral of I_csr. Only calculated for i_bin = 1 since it is not needed otherwise.
!-

subroutine I_csr (kick1, i_bin, csr)

implicit none

type (csr_kick1_struct), target :: kick1
type (csr_struct), target :: csr
type (csr_kick1_struct), pointer :: k
type (csr_ele_info_struct), pointer :: eleinfo
type (spline_struct), pointer :: spl

real(rp) g_bend, z, zz, Ls, L, dtheta_L, dL, s_chord_kick
integer i_bin, ix_ele_kick

!

kick1%I_int_csr = 0
kick1%I_csr = 0

! No kick when source particle is ahead of the kicked particle

z = kick1%dz_particles
if (z <= 0) return

!

k => kick1
k%I_csr = -csr%kick_factor * 2 * (k%dL / z + csr%gamma2 * k%theta_sl * k%theta_lk / (1 + csr%gamma2 * k%theta_sl**2)) / k%L

! I_csr Integral 

if (i_bin == 1) then
  ix_ele_kick = csr%ix_ele_kick
  s_chord_kick = csr%s_chord_kick
  do
    if (s_chord_kick /= 0) exit  ! Not at edge of element and element has finite length
    ix_ele_kick = ix_ele_kick - 1
    if (ix_ele_kick == 0) return  ! No kick from before beginning of lattice
    s_chord_kick = csr%eleinfo(ix_ele_kick)%L_chord
  enddo

  spl => csr%eleinfo(ix_ele_kick)%spline
  g_bend = -spline1(spl, s_chord_kick, 2) / sqrt(1 + spline1(spl, s_chord_kick, 1)**2)**3

  if (k%ix_ele_source == ix_ele_kick) then
    Ls = k%L + k%dL
    k%I_int_csr = -csr%kick_factor * ((g_bend * Ls/2)**2 - log(2 * csr%gamma2 * z / Ls) / csr%gamma2)  
  else
    ! Since source pt is in another element, split integral into pieces.
    ! First integrate over element containing the kick point.
    L = s_chord_kick
    eleinfo => csr%eleinfo(ix_ele_kick)
    dtheta_L = eleinfo%spline%coef(1) + eleinfo%spline%coef(2) * L + eleinfo%spline%coef(3) * L**2
    dL = dspline_len(0.0_rp, L, eleinfo%spline, dtheta_L) ! = Ls - L
    Ls = L + dL
    zz = L / (2 * csr%gamma2) + dL
    k%I_int_csr = -csr%kick_factor * ((g_bend * Ls/2)**2 - log(2 * csr%gamma2 * zz / Ls) / csr%gamma2)
    ! Now add on rest of integral
    k%I_int_csr = k%I_int_csr + k%I_csr * (z - zz)
  endif

else
  kick1%I_int_csr = (kick1%I_csr + csr%kick1(i_bin-1)%I_csr) * csr%dz_slice / 2
endif

end subroutine I_csr

!----------------------------------------------------------------------------
!----------------------------------------------------------------------------
!----------------------------------------------------------------------------
!+
! Subroutine image_charge_kick_calc (kick1, csr) 
!
! Routine to calculate the image charge kick.
!
! Input:
!   kick1    -- csr_kick1_struct: 
!   csr      -- csr_struct:
!
! Output:
!   kick1             -- csr_kick1_struct: 
!     %image_kick_csr -- real(rp): Image charge kick.
!-

subroutine image_charge_kick_calc (kick1, csr)

implicit none

type (csr_kick1_struct), target :: kick1
type (csr_struct), target :: csr
type (csr_kick1_struct), pointer :: k
type (spline_struct), pointer :: sp

real(rp) N_vec(3), G_vec(3), B_vec(3), Bp_vec(3), NBp_vec(3), NBpG_vec(3), rad_cross_vec(3)
real(rp) z, sin_phi, cos_phi, OneNBp, OneNBp3, radiate, coulomb1, theta, g_bend

!

k => kick1
sp => csr%eleinfo(k%ix_ele_source)%spline

g_bend = -spline1(sp, k%s_chord_source, 2) / sqrt(1 + spline1(sp, k%s_chord_source, 1)**2)**3
theta = k%floor_s%theta

Bp_vec = csr%beta * [sin(theta), 0.0_rp, cos(theta) ]             ! beta vector at source point
G_vec = csr%beta**2 * g_bend * [-cos(theta), 0.0_rp, sin(theta) ] ! Acceleration vector

k%image_kick_csr = 0
z = k%dz_particles

N_vec = (csr%floor_k%r - k%floor_s%r) / k%L

theta = csr%floor_k%theta
B_vec = [sin(theta), 0.0_rp, cos(theta)]        ! Beta vector at kicked point


OneNBp = 1 - sum(N_vec * Bp_vec)
OneNBp3 = OneNBp**3

NBp_vec = N_vec - Bp_vec
NBpG_vec = cross_product(NBp_vec, G_vec)
rad_cross_vec = cross_product(N_vec, NBpG_vec)

radiate  = dot_product (B_vec, rad_cross_vec) / (k%L * OneNBp3)
coulomb1 = dot_product (B_vec, NBp_vec) / (csr%gamma2 * k%L**2 * OneNBp3)
kick1%image_kick_csr = csr%kick_factor * (radiate + coulomb1)

end subroutine image_charge_kick_calc

!----------------------------------------------------------------------------
!----------------------------------------------------------------------------
!----------------------------------------------------------------------------
!+
! Subroutine csr_and_sc_apply_kicks (ele, csr, particle)
!
! Routine to calculate the longitudinal coherent synchrotron radiation kick.
!
! Input:
!   ele         -- ele_struct: Element being tracked through.
!   csr         -- csr_struct: 
!   particle(:) -- coord_struct: Particles to kick.
!
! Output:
!   particle(:) -- Coord_struct: Particles with kick applied.
!-

subroutine csr_and_sc_apply_kicks (ele, csr, particle)

use da2_mod

implicit none

type (ele_struct) ele
type (csr_struct), target :: csr
type (coord_struct), target :: particle(:)
type (coord_struct), pointer :: p
type (csr_bunch_slice_struct), pointer :: slice

real(rp) zp, r1, r0, dz, dpz, nk(2), dnk(2,2), f0, f, beta0, dpx, dpy, x, y, f_tot
real(rp) Evec(3), factor, pz0, dct_ave, new_beta, ef
integer i, n, i0, i_del, ip

! CSR kick and Slice space charge kick
! We use a weighted average so that the integral varies smoothly as a function of particle%vec(5).

if (ele%csr_method == one_dim$ .or. ele%space_charge_method == slice$) then
  do ip = 1, size(particle)
    p => particle(ip)
    if (p%state /= alive$) cycle
    zp = p%vec(5)
    i0 = int((zp - csr%slice(1)%z_center) / csr%dz_slice) + 1
    r1 = (zp - csr%slice(i0)%z_center) / csr%dz_slice
    r0 = 1 - r1

    ! r1 should be in [0,1] but allow for some round-off error
    if (r1 < -0.01_rp .or. r1 > 1.01_rp .or. i0 < 1 .or. i0 >= space_charge_com%n_bin) then
      print *, 'CSR INTERNAL ERROR!'
      if (global_com%exit_on_error) call err_exit
    endif

    ! CSR kick
    ! We use a weighted average so that the integral varies smoothly as a function of particle%vec(5).

    if (ele%csr_method == one_dim$) then
      p%vec(6) = p%vec(6) + r0 * csr%slice(i0)%kick_csr + r1 * csr%slice(i0+1)%kick_csr
    endif

  ! Slice space charge kick

  if (ele%space_charge_method == slice$) then
    if (space_charge_com%lsc_kick_transverse_dependence) then
      x = p%vec(1)
      y = p%vec(3)
      dpz = 0

      slice => csr%slice(i0)
      if (slice%coef_lsc_plus(0,0) /= 0)  dpz = dpz + r0 / da2_evaluate(slice%coef_lsc_plus, (x-csr%x0_bunch)**2, (y-csr%y0_bunch)**2)
      if (slice%coef_lsc_minus(0,0) /= 0) dpz = dpz + r0 / da2_evaluate(slice%coef_lsc_minus, (x-csr%x0_bunch)**2, (y-csr%y0_bunch)**2)

      slice => csr%slice(i0+1)
      if (slice%coef_lsc_plus(0,0) /= 0)  dpz = dpz + r1 / da2_evaluate(slice%coef_lsc_plus, (x-csr%x0_bunch)**2, (y-csr%y0_bunch)**2)
      if (slice%coef_lsc_minus(0,0) /= 0) dpz = dpz + r1 / da2_evaluate(slice%coef_lsc_minus, (x-csr%x0_bunch)**2, (y-csr%y0_bunch)**2)

      p%vec(6) = p%vec(6) + dpz

    else
      p%vec(6) = p%vec(6) + r0 * csr%slice(i0)%kick_lsc + r1 * csr%slice(i0+1)%kick_lsc
    endif

    ! Must update beta and z due to the energy change

    beta0 = p%beta
    call convert_pc_to ((1+p%vec(6))* p%p0c, p%species, beta = p%beta)
    p%vec(5) = p%vec(5) * p%beta / beta0

    ! Transverse space charge. Like the beam-beam interaction but in this case the particles are going
    ! in the same direction longitudinally. In this case, the kicks due to the electric and magnetic fields
    ! tend to cancel instead of add as in the bbi case.

    f0 = csr%kick_factor * csr%actual_track_step * classical_radius(csr%species) / (twopi * &
             csr%dz_slice * csr%rel_mass * e_charge * abs(charge_of(p%species)) * (1 + p%vec(6)) * csr%gamma**3)

    dpx = 0;  dpy = 0

    ! Interpolate between slices. If one of the slices does not have 

    slice => csr%slice(i0)
    call bbi_kick (p%vec(1)-slice%x0, p%vec(3)-slice%y0, [slice%sig_x, slice%sig_y], nk, dnk)
    f = f0 * r0 * slice%charge / (slice%sig_x + slice%sig_y)
    ! The kick is negative of the bbi kick. That is, the kick is outward.
    dpx = -nk(1) * f
    dpy = -nk(2) * f

    slice => csr%slice(i0+1)
    call bbi_kick (p%vec(1)-slice%x0, p%vec(3)-slice%y0, [slice%sig_x, slice%sig_y], nk, dnk)
    f = f0 * r1 * slice%charge / (slice%sig_x + slice%sig_y)
    ! The kick is negative of the bbi kick. That is, the kick is outward.
    dpx = dpx - nk(1) * f   
    dpy = dpy - nk(2) * f

    if (slice%sig_x == 0 .or. slice%sig_y == 0) call err_exit

    ! 1/2 of the kick is electric and this leads to an energy change.
    ! The formula for p%vec(6) assumes that dpx and dpy are small so only the linear terms are kept.

    p%vec(2) = p%vec(2) + dpx
    p%vec(4) = p%vec(4) + dpy
    p%vec(6) = p%vec(6) + 0.5_rp * (dpx*p%vec(2) + dpy*p%vec(4))
  endif


  enddo
endif

! Mesh Space charge kick

if (ele%space_charge_method == fft_3d$) then

  dct_ave = sum(particle%vec(5)/particle%beta, particle%state == alive$) / count(particle%state == alive$)

#ifdef USE_GPU_TRACKING
  ! GPU path: use persistent device buffers if available, otherwise upload/download
  if (bmad_com%gpu_tracking_on) then
    block
    use gpu_tracking_mod, only: gpu_upload_particles, gpu_download_particles, &
                                gpu_space_charge_3d, gpu_persist_on_device
    use, intrinsic :: iso_c_binding
    real(C_DOUBLE), allocatable :: vx(:), vpx(:), vy(:), vpy(:), vz(:), vpz(:)
    real(C_DOUBLE), allocatable :: beta_a(:), p0c_a(:), t_a(:), charge_a(:)
    integer(C_INT), allocatable :: state_a(:)
    integer(C_INT) :: np
    real(rp) :: mc2_val

    np = size(particle)
    mc2_val = mass_of(particle(1)%species)

    ! Build charge array (always needed on host for upload)
    allocate(charge_a(np))
    do i = 1, np
      charge_a(i) = particle(i)%charge
    enddo

    if (gpu_persist_on_device) then
      ! Data is already on device from persistent session — no upload/download needed!
      ! Pass huge sentinel for dct_ave so the CUDA kernel computes it from device data.
      call gpu_space_charge_3d(charge_a, np, &
          int(csr%mesh3d%nhi(1), C_INT), int(csr%mesh3d%nhi(2), C_INT), int(csr%mesh3d%nhi(3), C_INT), &
          csr%mesh3d%gamma, csr%actual_track_step, mc2_val, 1.0e31_rp)
    else
      ! Data not on device — full upload/download cycle
      allocate(vx(np), vpx(np), vy(np), vpy(np), vz(np), vpz(np))
      allocate(state_a(np), beta_a(np), p0c_a(np), t_a(np))

      do i = 1, np
        vx(i) = particle(i)%vec(1); vpx(i) = particle(i)%vec(2)
        vy(i) = particle(i)%vec(3); vpy(i) = particle(i)%vec(4)
        vz(i) = particle(i)%vec(5); vpz(i) = particle(i)%vec(6)
        state_a(i) = particle(i)%state; beta_a(i) = particle(i)%beta
        p0c_a(i) = particle(i)%p0c; t_a(i) = particle(i)%t
      enddo

      call gpu_upload_particles(vx, vpx, vy, vpy, vz, vpz, state_a, beta_a, p0c_a, t_a, np)
      call gpu_space_charge_3d(charge_a, np, &
          int(csr%mesh3d%nhi(1), C_INT), int(csr%mesh3d%nhi(2), C_INT), int(csr%mesh3d%nhi(3), C_INT), &
          csr%mesh3d%gamma, csr%actual_track_step, mc2_val, dct_ave)
      call gpu_download_particles(vx, vpx, vy, vpy, vz, vpz, state_a, beta_a, p0c_a, t_a, np, 1, 0)

      do i = 1, np
        particle(i)%vec(1) = vx(i); particle(i)%vec(2) = vpx(i)
        particle(i)%vec(3) = vy(i); particle(i)%vec(4) = vpy(i)
        particle(i)%vec(5) = vz(i); particle(i)%vec(6) = vpz(i)
        particle(i)%state = state_a(i); particle(i)%beta = beta_a(i)
      enddo

      deallocate(vx, vpx, vy, vpy, vz, vpz, state_a, beta_a, p0c_a, t_a)
    endif

    deallocate(charge_a)
    end block
  else
#endif

  if (.not. allocated(csr%position)) allocate(csr%position(size(particle)))
  if (size(csr%position) < size(particle)) then
    deallocate(csr%position)
    allocate(csr%position(size(particle)))
  endif

  n = 0
  do i = 1, size(particle)
    p => particle(i)
    if (p%state /= alive$) cycle
    n = n + 1
    csr%position(n)%r = [p%vec(1), p%vec(3), p%vec(5) - dct_ave * p%beta]
    csr%position(n)%charge = p%charge
  enddo

  call deposit_particles (csr%position(1:n)%r(1), csr%position(1:n)%r(2), csr%position(1:n)%r(3), csr%mesh3d, qa=csr%position(1:n)%charge)
  call space_charge_3d(csr%mesh3d)

  do i = 1, size(particle)
    p => particle(i)
    if (p%state /= alive$) cycle
    call interpolate_field(p%vec(1), p%vec(3), p%vec(5)-dct_ave*p%beta,  csr%mesh3d, E=Evec)
    factor = csr%actual_track_step / (p%p0c  * p%beta)
    pz0 = sqrt( (1.0_rp + p%vec(6))**2 - p%vec(2)**2 - p%vec(4)**2 )
    p%vec(2) = p%vec(2) + Evec(1)*factor / csr%mesh3d%gamma**2
    p%vec(4) = p%vec(4) + Evec(2)*factor / csr%mesh3d%gamma**2
    ef = Evec(3) * factor
    dpz = sqrt_alpha(1 + p%vec(6), ef*ef + 2 * ef * pz0)
    p%vec(6) = p%vec(6) + dpz
    call convert_pc_to (p%p0c * (1 + p%vec(6)), p%species, beta = new_beta)
    p%vec(5) = p%vec(5) * new_beta / p%beta
    p%beta = new_beta
  enddo

#ifdef USE_GPU_TRACKING
  endif
#endif

endif

end subroutine csr_and_sc_apply_kicks

!----------------------------------------------------------------------------
!----------------------------------------------------------------------------
!----------------------------------------------------------------------------
!+
! Function dspline_len (s_chord0, s_chord1, spline, dtheta_ref) result (dlen)
!
! Routine to calculate the difference in length between the spline curve length and a referece line.
! Referece line is centroid chord (referece system of the spline) rotated by dtheta_ref.
!
! Input:
!   s_chord0    -- real(rp): Start position along centroid chord.
!   s_chord1    -- real(rp): Stop position along central_chord.
!   spline      -- spline_struct: Spline of x-position as a function of s.
!   dtheta_ref  -- real(rp), optional: angle to rotate the reference line from the centroid chord.
!                    Default is 0.
!
! Output:
!   dlen -- real(rp): L_spline - L_chord
!-

function dspline_len (s_chord0, s_chord1, spline, dtheta_ref) result (dlen)

implicit none

type (spline_struct) spline

real(rp) s_chord0, s_chord1, dlen
real(rp), optional :: dtheta_ref
real(rp) c(0:3), s0, ds

! x' = c(1) + 2*c2*s + 3*c3*s^2
! dlen = Integral: x'^2/2 ds

c = spline%coef
s0 = s_chord0
ds = s_chord1 - s_chord0

if (present(dtheta_ref)) then
  c(1) = c(1) - dtheta_ref
endif

dlen = (ds / 2) * ( &
        c(1)**2 + &
        (2*s0 + ds) * 2*c(1)*c(2) + &
        (3*s0*s0 + 3*s0*ds + ds*ds) * (6*c(1)*c(3) + 4*c(2)**2) / 3 + &
        (4*s0**3 + 6*s0*s0*ds + 4*s0*ds*ds + ds**3) * 3*c(2)*c(3) + &
        (5*s0**4 + 10*s0**3*ds + 10*s0*s0*ds*ds + 5*s0*ds**3 + ds**4) * 9*c(3)**2 &
       )

end function dspline_len

!----------------------------------------------------------------------------
!----------------------------------------------------------------------------
!----------------------------------------------------------------------------
!+
! Function s_ref_to_s_chord (s_ref, eleinfo) result (s_chord)
!
! Routine to calculate s_chord given s_ref.
!
! Input:
!   s_ref     -- real(rp): s-position along element ref coords.
!   eleinfo     -- csr_ele_info_struct: Element info
!
! Output:
!   s_chord   -- real(rp): s-position along centroid chord.
!-

function s_ref_to_s_chord (s_ref, eleinfo) result (s_chord)

implicit none

type (csr_ele_info_struct), target :: eleinfo
type (ele_struct), pointer :: ele

real(rp) s_ref, s_chord, dtheta, dr(3), x, g, t

!

ele => eleinfo%ele
g = ele%value(g$)

if ((ele%key == sbend$ .or. ele%key == rf_bend$) .and. abs(g) > 1d-5) then
  dtheta = eleinfo%floor0%theta - eleinfo%ref_floor0%theta
  dr = eleinfo%ref_floor0%r - eleinfo%floor0%r
  t = eleinfo%ref_floor0%theta + pi/2
  x = dr(1) * cos(t) + dr(3) * sin(t)
  s_chord = abs(ele%value(rho$)) * atan2(s_ref * cos(dtheta), abs(ele%value(rho$) + x + s_ref * sin(dtheta)))

else
  s_chord = s_ref * eleinfo%L_chord /ele%value(l$)
endif

end function s_ref_to_s_chord







!----------------------------------------------------------------------------
!----------------------------------------------------------------------------
!----------------------------------------------------------------------------
!+
! Subroutine track1_bunch_csr3d (bunch, ele, centroid, err, bunch_track)
!
! EXPERIMENTAL. NOT CURRENTLY OPERATIONAL!
!
! Routine to track a bunch of particles through an element using
! steady-state 3D CSR.
!
!
! Input:
!   bunch         -- Bunch_struct: Starting bunch position.
!   ele           -- Ele_struct: The element to track through. Must be part of a lattice.
!   centroid(0:)  -- coord_struct, Approximate beam centroid orbit for the lattice branch.
!                      Calculate this before beam tracking by tracking a single particle.
!   s_start       -- real(rp), optional: Starting position relative to ele. Default = 0
!   s_end         -- real(rp), optional: Ending position. Default is ele length.
!   bunch_track   -- bunch_track_struct, optional: Existing tracks. If bunch_track%n_pt = -1 then
!                        Overwrite any existing track.
!
! Output:
!   bunch       -- Bunch_struct: Ending bunch position.
!   err         -- Logical: Set true if there is an error. EG: Too many particles lost.
!   bunch_track -- bunch_track_struct, optional: track information if the tracking method does
!                        tracking step-by-step. When tracking through multiple elements, the 
!                        trajectory in an element is appended to the existing trajectory. 
!
!
! Notes:
!   The core routines are from the OpenCSR package developed at:
!   https://github.com/ChristopherMayes/OpenCSR
!
!-

subroutine track1_bunch_csr3d (bunch, ele, centroid, err, s_start, s_end, bunch_track)

implicit none

type (bunch_struct), target :: bunch
type (coord_struct), pointer :: c0, p
type (ele_struct), target :: ele
type (bunch_track_struct), optional :: bunch_track
type (branch_struct), pointer :: branch
type (ele_struct) :: runt
type (csr_struct), target :: csr
type (coord_struct), target :: centroid(0:)
type (coord_struct), pointer :: particle(:)

real(rp), optional :: s_start, s_end
real(rp) s0_step
real(rp) e_tot, ds_step
real(rp) Evec(3), factor, pz0
integer i, j, n, ie, ns, nb, n_step, n_live, i_step

character(*), parameter :: r_name = 'track1_bunch_csr3d'
logical err, err_flag, parallel0, parallel1

! EXPERIMENTAL. NOT CURRENTLY OPERATIONAL!

! Init

err = .true.
branch => pointer_to_branch(ele)

! This calc is only valid in SBEND elements, nonzero length
! No CSR for a zero length element.
! And taylor elements get ignored.
if (ele%value(l$) == 0 .or. ele%key /= sbend$) then
  call track1_bunch_hom (bunch, ele)
  err = .false.
  return
endif

! Set gamma, mesh size
c0 => centroid(ele%ix_ele)
call convert_pc_to((1+c0%vec(6)) * c0%p0c, c0%species, gamma = csr%mesh3d%gamma)
csr%mesh3d%nhi = space_charge_com%csr3d_mesh_size

! n_step is the number of steps to take when tracking through the element.
! csr%ds_step is the true step length.

particle => bunch%particle
! make sure that ele_len / track_step is an integer.

csr%ds_track_step = ele%value(csr_ds_step$)
if (csr%ds_track_step == 0) csr%ds_track_step = space_charge_com%ds_track_step
if (csr%ds_track_step == 0) then
  call out_io (s_fatal$, r_name, 'NEITHER SPACE_CHARGE_COM%DS_TRACK_STEP NOR CSR_TRACK_STEP FOR THIS ELEMENT ARE SET! ' // ele%name)
  if (global_com%exit_on_error) call err_exit
  return
endif

n_step = max (1, nint(ele%value(l$) / csr%ds_track_step))
csr%ds_track_step = ele%value(l$) / n_step
csr%species = bunch%particle(1)%species

if (.not. allocated(csr%position)) allocate(csr%position(size(particle)))
if (size(csr%position) < size(particle)) then
  deallocate(csr%position)
  allocate(csr%position(size(particle)))
endif

!----------------------------------------------------------------------------------------
! Loop over the tracking steps
! runt is the element that is tracked through at each step.

call save_a_bunch_step (ele, bunch, bunch_track, s_start)

do i_step = 0, n_step

  ! track through the runt

  if (i_step /= 0) then
    call element_slice_iterator (ele, branch%param, i_step, n_step, runt, s_start, s_end)
    call track1_bunch_hom (bunch, runt)
  endif
  particle => bunch%particle
  
  s0_step = i_step * csr%ds_track_step
  if (present(s_start)) s0_step = s0_step + s_start

  e_tot = ele%value(e_tot$)
  call convert_total_energy_to (e_tot, branch%param%particle, csr%gamma, beta = csr%beta)
  csr%gamma2 = csr%gamma**2

  ! Bin particles
  ! TODO: simplify this into a function, and use the function above for fft_3d
  n = 0
  do i = 1, size(particle)
    p => particle(i)
    if (p%state /= alive$) cycle
    n = n + 1
    csr%position(n)%r = p%vec(1:5:2)
    csr%position(n)%charge = p%charge
  enddo
  call deposit_particles (csr%position(1:n)%r(1), csr%position(1:n)%r(2), csr%position(1:n)%r(3), csr%mesh3d, qa=csr%position(1:n)%charge)  

  ! Give particles a kick
  ! TODO: simplify with fft_3d
  print *, '---------- CSR Steady_State_3D ----------'
  
  call csr3d_steady_state_solver(csr%mesh3d%rho/product(csr%mesh3d%delta), &
              csr%mesh3d%gamma, ele%value(rho$), csr%mesh3d%delta, csr%mesh3d%efield, normalize=.false.)
  
  ! Handle first and last steps for half-kicks
  csr%kick_factor = csr%ds_track_step
  if (i_step == 0 .or. i_step == n_step) csr%kick_factor = csr%kick_factor / 2
  print *, 'kick by ds', csr%kick_factor, i_step
  
  
  print *, 'Interpolating field and kicking particles'
  do i = 1, size(particle)
    p => particle(i)
    if (p%state /= alive$) cycle
    call interpolate_field(p%vec(1), p%vec(3), p%vec(5), csr%mesh3d, E=Evec)

    factor = csr%kick_factor / (p%p0c  * p%beta) 

    pz0 = sqrt( (1.0_rp + p%vec(6))**2 - p%vec(2)**2 - p%vec(4)**2 ) ! * p0 
    p%vec(2) = p%vec(2) + Evec(1)*factor
    p%vec(4) = p%vec(4) + Evec(2)*factor
    p%vec(6) = sqrt(p%vec(2)**2 + p%vec(4)**2 + (Evec(3)*factor + pz0)**2) -1.0_rp
    ! Set beta
    call convert_pc_to (p%p0c * (1 + p%vec(6)), p%species, beta = p%beta)
  enddo

  call save_a_bunch_step (ele, bunch, bunch_track, s0_step+s_start)

enddo

err = .false.

end subroutine track1_bunch_csr3d

end module
