! The location of a module scope is what tells AddDebugInfo the line of the
! MODULE statement, so the builtin.module location lists a scope for every
! module defined here, even one with no USE statement, and only when full debug
! info is requested.

! RUN: %flang_fc1 -emit-hlfir -debug-info-kind=standalone %s -o - \
! RUN:   -mmlir -mlir-print-debuginfo -mmlir -mlir-print-local-scope \
! RUN:   | FileCheck %s --check-prefix=WITH_DEBUG
! RUN: %flang_fc1 -emit-hlfir %s -o - -mmlir -mlir-print-debuginfo \
! RUN:   | FileCheck %s --check-prefix=NO_DEBUG
! RUN: %flang_fc1 -emit-hlfir -debug-info-kind=line-tables-only %s -o - \
! RUN:   -mmlir -mlir-print-debuginfo | FileCheck %s --check-prefix=NO_DEBUG

! NO_DEBUG-NOT: #fir.scope
! NO_DEBUG-NOT: #fir.use

! WITH_DEBUG-LABEL: func.func @_QMno_use_modPtest_sub()
! WITH_DEBUG: } loc(fused<#fir.scope<procedure "test_sub", parent = #fir.scope<module "no_use_mod", exports = [<"mod_var">], loc({{[^)]*}})>, loc({{[^)]*}})>>
! WITH_DEBUG-LABEL: func.func @_QQmain()
! WITH_DEBUG: } loc(fused<#fir.scope<procedure "MAIN", uses = [#fir.use<"using_mod", loc({{[^)]*}})>], loc({{[^)]*}})>>

! WITH_DEBUG: {{^}}} loc(fused<[#fir.scope<module "no_use_mod", exports = [<"mod_var">], loc("{{[^"]*}}":[[@LINE+1]]:1)>,
module no_use_mod
  integer :: mod_var
contains
  subroutine test_sub()
    mod_var = 100
  end subroutine test_sub
end module no_use_mod

! WITH_DEBUG-SAME: #fir.scope<module "using_mod", uses = [#fir.use<"no_use_mod", loc("{{[^"]*}}":[[@LINE+3]]:3)>],
! WITH_DEBUG-SAME: loc("{{[^"]*}}":[[@LINE+1]]:1)>]>
module using_mod
  use no_use_mod
  real :: x
end module using_mod

program main
  use using_mod
  call test_sub()
  x = 1.0
end program main
