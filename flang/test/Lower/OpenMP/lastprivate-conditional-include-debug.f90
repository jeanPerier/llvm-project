! Test that lastprivate(conditional:) preserves the source line in its record
! type name when nested FusedLocs wrap an INCLUDE'd directive inside a BLOCK
! construct with debug scopes enabled.

! RUN: split-file %s %t
! RUN: %flang_fc1 -fopenmp -fopenmp-version=50 -debug-info-kind=standalone -emit-hlfir -I %t %t/main.f90 -o - | FileCheck %s

! CHECK: omp.declare_reduction @lp_cond_byref_rec__lp_cond_t.l3.0 byref_element_type(!fir.type<_lp_cond_t.l3.0{x:i32,$x:i64}>) : !fir.ref<!fir.type<_lp_cond_t.l3.0{x:i32,$x:i64}>>

!--- inc.h
  ! line 1
  ! line 2
  !$omp parallel do lastprivate(conditional: x)
  do i = 1, n
    if (i > 0) x = i
  end do

!--- main.f90
subroutine s(n, x)
  implicit none
  integer, intent(in) :: n
  integer, intent(inout) :: x
  integer :: i
  block
    include "inc.h"
  end block
end subroutine
