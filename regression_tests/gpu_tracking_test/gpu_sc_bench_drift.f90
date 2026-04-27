!+
! gpu_sc_bench_drift
!
! Benchmark: 10 x 1m drifts, 1M particles
! Four cases:
!   1. CPU, no space charge
!   2. CPU, 3D FFT space charge
!   3. GPU, no space charge
!   4. GPU, 3D FFT space charge
!
! init_beam_distribution is called ONCE before the timing loop.
! Each pass restores particles from a saved copy (fast memcpy).
!-

program gpu_sc_bench_drift

use bmad
use beam_mod
use gpu_tracking_mod

implicit none

type (lat_struct), target :: lat_nosc, lat_sc
type (beam_init_struct) :: beam_init
type (beam_struct) :: beam, beam_save
type (branch_struct), pointer :: branch
type (coord_struct), allocatable :: centroid(:)
type (coord_struct), allocatable :: saved_particles(:)

integer :: n_pass, ie, np
real(rp) :: t_start, t_end
real(rp) :: t_cpu_nosc, t_cpu_sc, t_gpu_nosc, t_gpu_sc
logical :: err

! Initialize GPU
call gpu_tracking_init()
if (.not. bmad_com%gpu_tracking_on) then
  print *, 'FATAL: GPU tracking not available'
  stop
endif

! Beam init: 1M particles
beam_init%n_particle = 1000000
beam_init%random_engine = 'quasi'
beam_init%a_emit = 5e-7
beam_init%b_emit = 5e-7
beam_init%dPz_dz = 0
beam_init%n_bunch = 1
beam_init%bunch_charge = 1e-9
beam_init%sig_pz = 1e-3
beam_init%sig_z = 3e-3
beam_init%random_sigma_cutoff = 3

! Space charge settings
space_charge_com%ds_track_step = 0.1_rp
space_charge_com%n_bin = 40
space_charge_com%particle_bin_span = 2
space_charge_com%space_charge_mesh_size = [16, 16, 32]

n_pass = 5  ! Average over 5 passes

! Parse both lattices
call bmad_parser('lat_sc_bench_drift_nosc.bmad', lat_nosc)
call bmad_parser('lat_sc_bench_drift.bmad', lat_sc)

! Compute centroids for SC lattice
call compute_centroid(lat_sc, centroid)

! Create beam ONCE and save a copy for restoring between passes
branch => lat_nosc%branch(0)
call init_beam_distribution(branch%ele(0), branch%param, beam_init, beam, err)
np = size(beam%bunch(1)%particle)
allocate(saved_particles(np))
saved_particles = beam%bunch(1)%particle

! Warmup GPU (small beam)
block
  type (beam_struct) :: warmup
  beam_init%n_particle = 100
  call init_beam_distribution(branch%ele(0), branch%param, beam_init, warmup, err)
  bmad_com%gpu_tracking_on = .true.
  call track_beam(lat_nosc, warmup, err=err)
  beam_init%n_particle = 1000000
end block

print *
print *, '=================================================================='
print *, '  Drift + Space Charge Benchmark'
print *, '  10 x 1m drifts, 1M particles, 5 passes'
print *, '=================================================================='
print *

! ======================================================================
! Case 1: CPU, no space charge
! ======================================================================
bmad_com%gpu_tracking_on = .false.
bmad_com%csr_and_space_charge_on = .false.
call cpu_time(t_start)
do ie = 1, n_pass
  beam%bunch(1)%particle = saved_particles
  call track_beam(lat_nosc, beam, err=err)
enddo
call cpu_time(t_end)
t_cpu_nosc = (t_end - t_start) / n_pass
print '(A,T45,F10.3,A)', '  CPU, no SC:', t_cpu_nosc, ' s'

! ======================================================================
! Case 2: CPU, 3D FFT space charge
! ======================================================================
bmad_com%gpu_tracking_on = .false.
bmad_com%csr_and_space_charge_on = .true.
call cpu_time(t_start)
do ie = 1, n_pass
  beam%bunch(1)%particle = saved_particles
  call track_beam(lat_sc, beam, err=err, centroid=centroid)
enddo
call cpu_time(t_end)
t_cpu_sc = (t_end - t_start) / n_pass
print '(A,T45,F10.3,A)', '  CPU, 3D FFT SC:', t_cpu_sc, ' s'

! ======================================================================
! Case 3: GPU, no space charge
! ======================================================================
bmad_com%gpu_tracking_on = .true.
bmad_com%csr_and_space_charge_on = .false.
call cpu_time(t_start)
do ie = 1, n_pass
  beam%bunch(1)%particle = saved_particles
  call track_beam(lat_nosc, beam, err=err)
enddo
call cpu_time(t_end)
t_gpu_nosc = (t_end - t_start) / n_pass
print '(A,T45,F10.3,A)', '  GPU, no SC:', t_gpu_nosc, ' s'

! ======================================================================
! Case 4: GPU, 3D FFT space charge
! ======================================================================
bmad_com%gpu_tracking_on = .true.
bmad_com%csr_and_space_charge_on = .true.
call cpu_time(t_start)
do ie = 1, n_pass
  beam%bunch(1)%particle = saved_particles
  call track_beam(lat_sc, beam, err=err, centroid=centroid)
enddo
call cpu_time(t_end)
t_gpu_sc = (t_end - t_start) / n_pass
print '(A,T45,F10.3,A)', '  GPU, 3D FFT SC:', t_gpu_sc, ' s'

! Summary
print *
print *, '------------------------------------------------------------------'
print *, '  Speedups:'
print '(A,T45,F10.2,A)', '  GPU vs CPU (no SC):', t_cpu_nosc / t_gpu_nosc, 'x'
print '(A,T45,F10.2,A)', '  GPU vs CPU (with SC):', t_cpu_sc / t_gpu_sc, 'x'
print '(A,T45,F10.2,A)', '  SC overhead (CPU):', t_cpu_sc / t_cpu_nosc, 'x'
print '(A,T45,F10.2,A)', '  SC overhead (GPU):', t_gpu_sc / t_gpu_nosc, 'x'
print *, '=================================================================='

contains

subroutine compute_centroid(lat, centroid)
type (lat_struct), target, intent(inout) :: lat
type (coord_struct), allocatable, intent(out) :: centroid(:)
type (branch_struct), pointer :: br
type (coord_struct) :: orb
integer :: ie2
br => lat%branch(0)
allocate(centroid(0:br%n_ele_track))
call init_coord(orb, br%ele(0), downstream_end$)
centroid(0) = orb
do ie2 = 1, br%n_ele_track
  call track1(orb, br%ele(ie2), br%param, orb)
  centroid(ie2) = orb
enddo
end subroutine compute_centroid

end program gpu_sc_bench_drift
