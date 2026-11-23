! Check that diagnostics do not print the source scopes and the debug
! information that -g fuses into locations, both during lowering and after it.

! RUN: %not_todo_cmd %flang_fc1 -emit-hlfir -fopenmp -fopenmp-version=50 -debug-info-kind=standalone -cpp -DLOWERING %s -o /dev/null 2>&1 | FileCheck %s --check-prefix=LOWERING
! RUN: not %flang_fc1 -emit-llvm -fopenmp -fopenmp-version=50 -debug-info-kind=standalone %s -o /dev/null 2>&1 | FileCheck %s --check-prefix=TRANSLATION

! LOWERING: error: loc("{{[^"]*}}debug-diagnostic-location.f90":[[#@LINE+20]]:{{[0-9]+}}): {{.*}}not yet implemented: OpenMPDepobjConstruct

! TRANSLATION: loc("{{[^"]*}}debug-diagnostic-location.f90":[[#@LINE+20]]:{{[0-9]+}}): error: not yet implemented: Unhandled clause order in omp.wsloop operation
! TRANSLATION: loc("{{[^"]*}}debug-diagnostic-location.f90":[[#@LINE+19]]:{{[0-9]+}}): error: LLVM Translation failed for operation: omp.wsloop
! TRANSLATION: loc("{{[^"]*}}debug-diagnostic-location.f90":[[#@LINE+18]]:{{[0-9]+}}): error: LLVM Translation failed for operation: omp.parallel

module used
  integer :: y
end module

module m
  use used
  integer :: x
contains
  subroutine s(a, n)
    integer :: a(:), n, i
    block
      use used, only: z => y
#ifdef LOWERING
      integer :: obj
      !$omp depobj(obj) depend(in: z)
#else
      !$omp parallel do order(concurrent)
      do i = 1, n
        a(i) = z + x
      end do
#endif
    end block
  end subroutine
end module
