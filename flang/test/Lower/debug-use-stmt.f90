! RUN: %flang_fc1 -emit-hlfir -debug-info-kind=standalone -mmlir --mlir-print-debuginfo %s -o - | FileCheck %s --check-prefix=WITH_DEBUG
! RUN: %flang_fc1 -emit-hlfir -mmlir --mlir-print-debuginfo %s -o - | FileCheck %s --check-prefix=NO_DEBUG
! RUN: %flang_fc1 -emit-hlfir -debug-info-kind=line-tables-only -mmlir --mlir-print-debuginfo %s -o - | FileCheck %s --check-prefix=NO_DEBUG

module mod1
  integer :: a = 10, b = 20, c = 30
end module mod1

module mod2
  real :: x = 1.0, y = 2.0, z = 3.0
end module mod2

module mod3
  logical :: flag = .true.
end module mod3

! Test 1: Program with USE ONLY and USE with renames
program test_main
  use mod1, only: b, c
  use mod2, renamed_y => y
  implicit none
  print *, b, c, renamed_y
end program

! Test 2: Subroutine with USE (all) and different renames
subroutine test_sub()
  use mod1
  use mod2, only: x
  use mod3, my_flag => flag
  implicit none
  print *, a, b, c, x, my_flag
end subroutine

! Test 3: Function with multiple USE patterns
function test_func() result(res)
  use mod1, only: a
  use mod2, renamed_x => x, renamed_z => z
  use mod3
  implicit none
  integer :: res
  res = a
end function

! Test 4: Subroutine with USE ONLY with renames (combined)
subroutine test_only_rename()
  use mod1, only: local_b => b, c, local_a => a
  use mod2, only: local_x => x, local_y => y
  implicit none
  print *, local_a, local_b, c, local_x, local_y
end subroutine

! The USE statements of each procedure are listed in the procedure scope fused
! into the location of the function, each with the location of its statement.

! WITH_DEBUG-LABEL: func.func @_QQmain()
! WITH_DEBUG: } loc(#[[MAIN_LOC:loc[0-9]+]])
! WITH_DEBUG-LABEL: func.func @_QPtest_sub()
! WITH_DEBUG: } loc(#[[SUB_LOC:loc[0-9]+]])
! WITH_DEBUG-LABEL: func.func @_QPtest_func()
! WITH_DEBUG: } loc(#[[FUNC_LOC:loc[0-9]+]])
! WITH_DEBUG-LABEL: func.func @_QPtest_only_rename()
! WITH_DEBUG: } loc(#[[ONLY_RENAME_LOC:loc[0-9]+]])

! WITH_DEBUG-DAG: #[[MAIN_LOC]] = loc(fused<#[[MAIN:scope[0-9]*]]>[#{{loc[0-9]*}}])
! WITH_DEBUG-DAG: #scope_mod_mod1 = #fir.scope<module "mod1", #{{loc[0-9]*}}>
! WITH_DEBUG-DAG: #scope_mod_mod2 = #fir.scope<module "mod2", #{{loc[0-9]*}}>
! WITH_DEBUG-DAG: #scope_mod_mod3 = #fir.scope<module "mod3", #{{loc[0-9]*}}>
! WITH_DEBUG-DAG: #[[MAIN]] = #fir.scope<procedure "TEST_MAIN", uses = [#fir.use<"mod1", only [<"b">, <"c">], #[[MAIN_USE1:loc[0-9]+]]>, #fir.use<"mod2", [<"renamed_y" -> "y">], #[[MAIN_USE2:loc[0-9]+]]>], #{{loc[0-9]*}}>
! WITH_DEBUG-DAG: #[[MAIN_USE1]] = loc("{{.*}}debug-use-stmt.f90":19:3)
! WITH_DEBUG-DAG: #[[MAIN_USE2]] = loc("{{.*}}debug-use-stmt.f90":20:3)

! WITH_DEBUG-DAG: #[[SUB_LOC]] = loc(fused<#[[SUB:scope[0-9]*]]>[#{{loc[0-9]*}}])
! WITH_DEBUG-DAG: #[[SUB]] = #fir.scope<procedure "test_sub", uses = [#fir.use<"mod1", #[[SUB_USE1:loc[0-9]+]]>, #fir.use<"mod2", only [<"x">], #[[SUB_USE2:loc[0-9]+]]>, #fir.use<"mod3", [<"my_flag" -> "flag">], #[[SUB_USE3:loc[0-9]+]]>], #{{loc[0-9]*}}>
! WITH_DEBUG-DAG: #[[SUB_USE1]] = loc("{{.*}}debug-use-stmt.f90":27:3)
! WITH_DEBUG-DAG: #[[SUB_USE2]] = loc("{{.*}}debug-use-stmt.f90":28:3)
! WITH_DEBUG-DAG: #[[SUB_USE3]] = loc("{{.*}}debug-use-stmt.f90":29:3)

! WITH_DEBUG-DAG: #[[FUNC_LOC]] = loc(fused<#[[FUNC:scope[0-9]*]]>[#{{loc[0-9]*}}])
! WITH_DEBUG-DAG: #[[FUNC]] = #fir.scope<procedure "test_func", uses = [#fir.use<"mod1", only [<"a">], #{{loc[0-9]+}}>, #fir.use<"mod2", [<"renamed_x" -> "x">, <"renamed_z" -> "z">], #{{loc[0-9]+}}>, #fir.use<"mod3", #{{loc[0-9]+}}>], #{{loc[0-9]*}}>

! An ONLY list keeps its entries in source order, renamed or not.
! WITH_DEBUG-DAG: #[[ONLY_RENAME_LOC]] = loc(fused<#[[ONLY_RENAME:scope[0-9]*]]>[#{{loc[0-9]*}}])
! WITH_DEBUG-DAG: #[[ONLY_RENAME]] = #fir.scope<procedure "test_only_rename", uses = [#fir.use<"mod1", only [<"local_b" -> "b">, <"c">, <"local_a" -> "a">], #{{loc[0-9]+}}>, #fir.use<"mod2", only [<"local_x" -> "x">, <"local_y" -> "y">], #{{loc[0-9]+}}>], #{{loc[0-9]*}}>

! NO_DEBUG-NOT: #fir.scope
! NO_DEBUG-NOT: #fir.use
