! RUN: %python %S/test_folding.py %s %flang_fc1

! Test if windows build crashes or not

module real_tests
  !WARN: invalid argument on intrinsic function
  real(4), parameter :: nan_r4_mod = mod(3.5, 0.)
end module
