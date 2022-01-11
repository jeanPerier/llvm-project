! RUN: %flang_fc1 -fdebug-unparse %s 2>&1 | FileCheck %s

! Test if windows build crashes or not
module real_tests
  !CHECK: REAL, PARAMETER :: x
  real, parameter :: x = 1
  
  ! Lines that I did not suspect may cause the crash, but since
  ! the others seem to pass, let's try them.

  real(8), parameter :: nan_r8_dasin1 = dasin(-1.1_8)
  real(8), parameter :: nan_r8_dlog1 = dlog(-0.1_8)
  real(4), parameter :: ok_r4_gamma = gamma(-1.1)
  real(4), parameter :: r4_gamma2 = gamma(-1.)
  real(4), parameter :: ok_r4_log_gamma = log_gamma(-2.001)

end module
