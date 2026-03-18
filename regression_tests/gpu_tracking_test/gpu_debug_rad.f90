! Debug program: track full lattice with track_beam (which uses the
! multi-element dispatch when GPU is on), compare with CPU track_beam.
! Print per-particle differences for the first divergent particle.
program gpu_debug_rad
use bmad
use beam_mod
use gpu_tracking_mod
implicit none

type (lat_struct), target :: lat
type (beam_init_struct) :: beam_init
type (beam_struct) :: b_cpu, b_gpu
type (branch_struct), pointer :: br
real(rp) :: mdiff, ediff
logical :: err
integer :: j, j2, k, np

beam_init%n_particle = 5000
beam_init%random_engine = 'quasi'
beam_init%a_emit = 1e-9; beam_init%b_emit = 1e-9
beam_init%sig_pz = 1e-3; beam_init%sig_z = 1e-4
beam_init%n_bunch = 1; beam_init%bunch_charge = 1e-9
beam_init%random_sigma_cutoff = 4

call gpu_tracking_init()

call bmad_parser('lat_kitchen_sink.bmad', lat)
br => lat%branch(0)

bmad_com%radiation_damping_on = .true.
bmad_com%radiation_fluctuations_on = .true.
bmad_com%synch_rad_scale = 0.0_rp

call init_beam_distribution(br%ele(0), br%param, beam_init, b_cpu, err)
call init_beam_distribution(br%ele(0), br%param, beam_init, b_gpu, err)
b_gpu%bunch(1)%particle = b_cpu%bunch(1)%particle

np = size(b_cpu%bunch(1)%particle)

! CPU: full lattice tracking
bmad_com%gpu_tracking_on = .false.
call track_beam(lat, b_cpu, err=err)

! GPU: full lattice tracking (uses multi-element dispatch if enabled)
bmad_com%gpu_tracking_on = .true.
call track_beam(lat, b_gpu, err=err)

! Compare
mdiff = 0
k = 0  ! count state mismatches
do j = 1, np
  if (b_cpu%bunch(1)%particle(j)%state /= b_gpu%bunch(1)%particle(j)%state) then
    k = k + 1
    if (k <= 3) print '(A,I4,A,I2,A,I2)', 'STATE MISMATCH particle ', j, &
      '  cpu=', b_cpu%bunch(1)%particle(j)%state, '  gpu=', b_gpu%bunch(1)%particle(j)%state
  endif
  do j2 = 1, 6
    ediff = abs(b_cpu%bunch(1)%particle(j)%vec(j2) - b_gpu%bunch(1)%particle(j)%vec(j2))
    if (ediff > mdiff) mdiff = ediff
  enddo
  ediff = maxval(abs(b_cpu%bunch(1)%particle(j)%vec - b_gpu%bunch(1)%particle(j)%vec))
  if (ediff > 1e-6) then
    print '(A,I4,A,I2,A,I2)', 'Particle ', j, &
      '  cpu_state=', b_cpu%bunch(1)%particle(j)%state, &
      '  gpu_state=', b_gpu%bunch(1)%particle(j)%state
    print '(A,6ES14.6)', '  CPU:', b_cpu%bunch(1)%particle(j)%vec
    print '(A,6ES14.6)', '  GPU:', b_gpu%bunch(1)%particle(j)%vec
    print '(A,ES14.6,A,ES14.6)', '  CPU s=', b_cpu%bunch(1)%particle(j)%s, &
      '  GPU s=', b_gpu%bunch(1)%particle(j)%s
    exit  ! Only print first divergent particle
  endif
enddo
! Also check t and s
do j = 1, np
  ediff = abs(b_cpu%bunch(1)%particle(j)%t - b_gpu%bunch(1)%particle(j)%t)
  if (ediff > mdiff) mdiff = ediff
  if (ediff > 1e-6) then
    print '(A,I4,A,ES14.6,A,ES14.6)', '  t DIFF particle ', j, &
      '  cpu_t=', b_cpu%bunch(1)%particle(j)%t, '  gpu_t=', b_gpu%bunch(1)%particle(j)%t
    exit
  endif
  ediff = abs(b_cpu%bunch(1)%particle(j)%s - b_gpu%bunch(1)%particle(j)%s)
  if (ediff > mdiff) mdiff = ediff
  if (ediff > 1e-6) then
    print '(A,I4,A,ES14.6,A,ES14.6)', '  s DIFF particle ', j, &
      '  cpu_s=', b_cpu%bunch(1)%particle(j)%s, '  gpu_s=', b_gpu%bunch(1)%particle(j)%s
    exit
  endif
enddo
print '(A,I6)', 'state_mismatches = ', k
print '(A,ES14.6)', 'max_diff (incl t,s) = ', mdiff

end program gpu_debug_rad
