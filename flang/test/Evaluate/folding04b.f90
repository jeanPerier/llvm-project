! RUN: %flang_fc1 -fdebug-unparse %s 2>&1 | FileCheck %s

! Test if windows build crashes or not
module real_tests
  !CHECK: REAL, PARAMETER :: x
  real, parameter :: x = 1
  
  ! Lines that I suspect may cause the crash
  complex(4), parameter :: c4_clog1 = clog((0., 0.))

  real(4), parameter :: r4_gamma1 = gamma(0.)

  real(4), parameter :: r4_log_gamma1 = log_gamma(0.)

  real(4), parameter :: r4_atan2 = atan2(0., 0.) 
end module
