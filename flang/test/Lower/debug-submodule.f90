! RUN: %flang_fc1 -emit-fir -debug-info-kind=standalone -mmlir --mlir-print-debuginfo %s -o - | FileCheck %s
! RUN: %flang_fc1 -emit-fir -mmlir --mlir-print-debuginfo %s -o - | FileCheck %s --check-prefix=NO_DEBUG
! RUN: %flang_fc1 -emit-fir -debug-info-kind=line-tables-only -mmlir --mlir-print-debuginfo %s -o - | FileCheck %s --check-prefix=NO_DEBUG

! Test that lowering records the ancestry of a submodule, and the submodule
! that defines a separate module procedure, only when debug info asks for it.

! NO_DEBUG-NOT: #fir.scope

! CHECK-LABEL: func.func @_QMshapesPsquare(
! CHECK: } loc(#[[SQUARE_LOC:loc[0-9]+]])
! CHECK-LABEL: func.func @_QMshapesSimplSdeepPdeep_helper()
! CHECK: } loc(#[[HELPER_LOC:loc[0-9]+]])
! CHECK: {{^}}} loc(#[[MOD_LOC:loc[0-9]+]])

! The parent of a first level submodule is the module at the root, and that of
! a nested one is the submodule containing it. All three are defined here.
! CHECK-DAG: #scope_mod_shapes = #fir.scope<module "shapes", #{{loc[0-9]*}}>
! CHECK-DAG: #scope_mod_impl = #fir.scope<module "impl", parent = #scope_mod_shapes, #{{loc[0-9]*}}>
! CHECK-DAG: #scope_mod_deep = #fir.scope<module "deep", parent = #scope_mod_impl, #{{loc[0-9]*}}>
! CHECK-DAG: #[[MOD_LOC]] = loc(fused<[#scope_mod_shapes, #scope_mod_impl, #scope_mod_deep]>[#{{loc[0-9]*}}])

! The parent of a separate module procedure is the submodule that defines it,
! not the module that declares its interface. The parent of any other
! procedure is the submodule that contains it.
! CHECK-DAG: #[[SQUARE:scope[0-9]*]] = #fir.scope<procedure "square", parent = #scope_mod_impl, #{{loc[0-9]*}}>
! CHECK-DAG: #[[SQUARE_LOC]] = loc(fused<#[[SQUARE]]>[#{{loc[0-9]*}}])
! CHECK-DAG: #[[HELPER:scope[0-9]*]] = #fir.scope<procedure "deep_helper", parent = #scope_mod_deep, #{{loc[0-9]*}}>
! CHECK-DAG: #[[HELPER_LOC]] = loc(fused<#[[HELPER]]>[#{{loc[0-9]*}}])

module shapes
  interface
    module function square(n) result(r)
      integer, intent(in) :: n
      integer :: r
    end function square
  end interface
end module shapes

submodule (shapes) impl
  integer :: sub_var = 7
contains
  module function square(n) result(r)
    integer, intent(in) :: n
    integer :: r
    r = n * n
  end function square
end submodule impl

submodule (shapes:impl) deep
  integer :: deep_var = 11
contains
  subroutine deep_helper()
    deep_var = deep_var + sub_var
  end subroutine deep_helper
end submodule deep
