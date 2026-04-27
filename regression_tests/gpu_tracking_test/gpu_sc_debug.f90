program gpu_sc_debug
use bmad
use beam_mod
use gpu_tracking_mod
implicit none
type (lat_struct), target :: lat
type (beam_init_struct) :: beam_init
type (beam_struct) :: b_cpu, b_gpu
type (branch_struct), pointer :: branch
type (coord_struct), allocatable :: centroid(:)
type (coord_struct) :: orb
integer :: ie, j, n_alive_cpu, n_alive_gpu, np
real(rp) :: max_diff, d
logical :: err

call gpu_tracking_init()
if (.not. bmad_com%gpu_tracking_on) stop 'No GPU'

beam_init%n_particle = 100
beam_init%random_engine = 'quasi'
beam_init%a_emit = 5e-7; beam_init%b_emit = 5e-7
beam_init%n_bunch = 1; beam_init%bunch_charge = 1e-9
beam_init%sig_pz = 1e-3; beam_init%sig_z = 3e-3
beam_init%random_sigma_cutoff = 3

space_charge_com%ds_track_step = 0.1_rp
space_charge_com%n_bin = 40
space_charge_com%particle_bin_span = 2
space_charge_com%space_charge_mesh_size = [16, 16, 32]

bmad_com%csr_and_space_charge_on = .true.

call bmad_parser('lat_sc_test.bmad', lat)
branch => lat%branch(0)

! Compute centroid
allocate(centroid(0:branch%n_ele_track))
call init_coord(orb, branch%ele(0), downstream_end$)
centroid(0) = orb
do ie = 1, branch%n_ele_track
  call track1(orb, branch%ele(ie), branch%param, orb)
  centroid(ie) = orb
enddo

! Init beams
call init_beam_distribution(branch%ele(0), branch%param, beam_init, b_cpu, err)
call init_beam_distribution(branch%ele(0), branch%param, beam_init, b_gpu, err)
b_gpu%bunch(1)%particle = b_cpu%bunch(1)%particle

! CPU run
print *, 'Starting CPU run...'
bmad_com%gpu_tracking_on = .false.
call track_beam(lat, b_cpu, err=err, centroid=centroid)
print *, 'CPU done. err=', err

! GPU run
print *, 'Starting GPU run...'
bmad_com%gpu_tracking_on = .true.
call track_beam(lat, b_gpu, err=err, centroid=centroid)
print *, 'GPU done. err=', err

! Compare
np = size(b_cpu%bunch(1)%particle)
n_alive_cpu = 0; n_alive_gpu = 0; max_diff = 0
do j = 1, np
  if (b_cpu%bunch(1)%particle(j)%state == alive$) n_alive_cpu = n_alive_cpu + 1
  if (b_gpu%bunch(1)%particle(j)%state == alive$) n_alive_gpu = n_alive_gpu + 1
  if (b_cpu%bunch(1)%particle(j)%state == alive$ .and. b_gpu%bunch(1)%particle(j)%state == alive$) then
    do ie = 1, 6
      d = abs(b_cpu%bunch(1)%particle(j)%vec(ie) - b_gpu%bunch(1)%particle(j)%vec(ie))
      max_diff = max(max_diff, d)
    enddo
  endif
enddo

print *, 'alive_cpu=', n_alive_cpu, ' alive_gpu=', n_alive_gpu
print *, 'max_diff=', max_diff

! Print first 3 particles
do j = 1, min(3, np)
  print '(A,I3,A,6ES12.4)', ' CPU  p', j, ':', b_cpu%bunch(1)%particle(j)%vec
  print '(A,I3,A,6ES12.4)', ' GPU  p', j, ':', b_gpu%bunch(1)%particle(j)%vec
  print *
enddo

end program gpu_sc_debug
