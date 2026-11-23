! RUN: %flang_fc1 -emit-hlfir -debug-info-kind=standalone -mmlir --mlir-print-debuginfo %s -o - | FileCheck %s --check-prefix=WITH_DEBUG
! RUN: %flang_fc1 -emit-hlfir -mmlir --mlir-print-debuginfo %s -o - | FileCheck %s --check-prefix=NO_DEBUG
! RUN: %flang_fc1 -emit-hlfir -debug-info-kind=line-tables-only -mmlir --mlir-print-debuginfo %s -o - | FileCheck %s --check-prefix=NO_DEBUG

module mod_ma
  integer :: va = 1
end module mod_ma

module mod_mb
  integer :: vb = 2
end module mod_mb

module mod_mc
  integer :: vc = 3
end module mod_mc

! A module that uses three different modules in its specification part.
module mod_multi
  use mod_ma, only: va
  use mod_mb
  use mod_mc, renamed_vc => vc
  integer :: vx = 0
end module mod_multi

subroutine test_sub_uses_multi
  use mod_multi
  implicit none
  print *, vx
end subroutine

! The USE statements of a module are listed in its module scope, in the
! builtin.module location, and are not copied into the scope of a procedure
! that uses the module.
! WITH_DEBUG-LABEL: func.func @_QPtest_sub_uses_multi()
! WITH_DEBUG: } loc(#[[SUB_LOC:loc[0-9]+]])
! WITH_DEBUG: {{^}}} loc(#[[MOD_LOC:loc[0-9]+]])

! WITH_DEBUG-DAG: #scope_mod_mod_ma = #fir.scope<module "mod_ma", #{{loc[0-9]*}}>
! WITH_DEBUG-DAG: #scope_mod_mod_mb = #fir.scope<module "mod_mb", #{{loc[0-9]*}}>
! WITH_DEBUG-DAG: #scope_mod_mod_mc = #fir.scope<module "mod_mc", #{{loc[0-9]*}}>
! WITH_DEBUG-DAG: #scope_mod_mod_multi = #fir.scope<module "mod_multi", uses = [#fir.use<#scope_mod_mod_ma, only [<"va">], #[[USE_MA:loc[0-9]+]]>, #fir.use<#scope_mod_mod_mb, #[[USE_MB:loc[0-9]+]]>, #fir.use<#scope_mod_mod_mc, [<"renamed_vc" -> "vc">], #[[USE_MC:loc[0-9]+]]>], #{{loc[0-9]*}}>
! WITH_DEBUG-DAG: #[[MOD_LOC]] = loc(fused<[#scope_mod_mod_ma, #scope_mod_mod_mb, #scope_mod_mod_mc, #scope_mod_mod_multi]>[#{{loc[0-9]*}}])
! WITH_DEBUG-DAG: #[[USE_MA]] = loc("{{.*}}debug-use-stmt-module-multiple.f90":19:3)
! WITH_DEBUG-DAG: #[[USE_MB]] = loc("{{.*}}debug-use-stmt-module-multiple.f90":20:3)
! WITH_DEBUG-DAG: #[[USE_MC]] = loc("{{.*}}debug-use-stmt-module-multiple.f90":21:3)
! WITH_DEBUG-DAG: #[[SUB:scope[0-9]*]] = #fir.scope<procedure "test_sub_uses_multi", uses = [#fir.use<#scope_mod_mod_multi, #{{loc[0-9]+}}>], #{{loc[0-9]*}}>
! WITH_DEBUG-DAG: #[[SUB_LOC]] = loc(fused<#[[SUB]]>[#{{loc[0-9]*}}])

! NO_DEBUG-NOT: #fir.scope
! NO_DEBUG-NOT: #fir.use
