! RUN: split-file %s %t
! RUN: %flang_fc1 -emit-hlfir -debug-info-kind=standalone -mmlir --mlir-print-debuginfo -module-dir %t %t/mods.f90 -o %t/mods.hlfir
! RUN: FileCheck %s --check-prefix=MODS-FIR --input-file=%t/mods.hlfir
! RUN: FileCheck %s --check-prefix=NAMES --input-file=%t/mods.hlfir
! RUN: %flang_fc1 -emit-llvm -debug-info-kind=standalone -module-dir %t %t/mods.f90 -o - | FileCheck %s --check-prefix=MODS
! RUN: %flang_fc1 -emit-hlfir -debug-info-kind=standalone -mmlir --mlir-print-debuginfo -I %t %t/main.f90 -o - | FileCheck %s --check-prefix=MAIN-FIR
! RUN: %flang_fc1 -emit-llvm -debug-info-kind=standalone -I %t %t/main.f90 -o - | FileCheck %s --check-prefix=MAIN
! RUN: %flang_fc1 -emit-llvm -debug-info-kind=standalone -I %t %t/main.f90 -o - | FileCheck %s --check-prefix=MAIN-STORAGE

! RUN: %flang_fc1 -emit-llvm -debug-info-kind=standalone -module-dir %t %t/bound-common.f90 -o - | FileCheck %s --check-prefix=BOUND

! Check the module variables that their globals do not describe completely:
! those with a binding label, in a COMMON block or an EQUIVALENCE, or whose
! lower bounds are not all 1.

! The global locations record identities for ordinary, bound, COMMON and
! EQUIVALENCE storage, including an ordinary array that needs no declaration.
! NAMES-DAG: loc(fused<#fir.global_name<"_QMmvEz">>
! NAMES-DAG: loc(fused<#fir.global_name<"c_x">>
! NAMES-DAG: loc(fused<#fir.global_name<"blk_">>
! NAMES-DAG: loc(fused<#fir.global_name<"_QMmvEe1">>

! The unit that defines a module lists all of them, and no variable that the
! compiler creates.
! MODS-FIR: #fir.scope<module "mv", exports = [{{.*}}], variables = [
! MODS-FIR-SAME: #fir.module_var<"a" : i32, common "blk" ("blk_"), #{{loc[0-9]*}}>,
! MODS-FIR-SAME: #fir.module_var<"b" : !fir.array<2xi32>, common "blk" ("blk_"), offset 4, lbounds [2], #{{loc[0-9]*}}>,
! MODS-FIR-SAME: #fir.module_var<"e1" : !fir.array<4xi32>, equivalence "e1", offset 4, lbounds [0], #{{loc[0-9]*}}>,
! MODS-FIR-SAME: #fir.module_var<"e2" : !fir.array<4xi32>, equivalence "e1", #{{loc[0-9]*}}>,
! MODS-FIR-SAME: #fir.module_var<"x", bind_c "c_x", #{{loc[0-9]*}}>,
! MODS-FIR-SAME: #fir.module_var<"z", lbounds [5], #{{loc[0-9]*}}>], #{{loc[0-9]*}}>

! MODS-DAG: @blk_ = common global [12 x i8] zeroinitializer, align 4, !dbg ![[A_EXPR:[0-9]+]], !dbg ![[B_EXPR:[0-9]+]]
! MODS-DAG: @_QMmvEe1 = global [20 x i8] zeroinitializer, align 64, !dbg ![[E1_EXPR:[0-9]+]], !dbg ![[E2_EXPR:[0-9]+]]
! MODS-DAG: @c_x = global i32 1, !dbg ![[X_EXPR:[0-9]+]]
! MODS-DAG: @_QMmvEz = global [2 x i32] [i32 3, i32 4], align 64, !dbg ![[Z_EXPR:[0-9]+]]
! MODS-DAG: ![[A_EXPR]] = !DIGlobalVariableExpression(var: ![[A:[0-9]+]], expr: !DIExpression())
! MODS-DAG: ![[A]] = distinct !DIGlobalVariable(name: "a", scope: ![[BLK:[0-9]+]], {{.*}}isDefinition: true)
! MODS-DAG: ![[BLK]] = !DICommonBlock(scope: ![[MV:[0-9]+]], declaration: null, name: "blk"
! MODS-DAG: ![[MV]] = !DIModule(scope: ![[CU:[0-9]+]], name: "mv"
! MODS-DAG: ![[CU]] = distinct !DICompileUnit({{.*}}globals: ![[GLOBALS:[0-9]+]]
! MODS-DAG: ![[GLOBALS]] = !{![[A_EXPR]], ![[B_EXPR]], ![[E1_EXPR]], ![[E2_EXPR]], !{{[0-9]+}}, ![[X_EXPR]], ![[Z_EXPR]]}
! MODS-DAG: ![[B_EXPR]] = !DIGlobalVariableExpression(var: ![[B:[0-9]+]], expr: !DIExpression(DW_OP_plus_uconst, 4))
! MODS-DAG: ![[B]] = distinct !DIGlobalVariable(name: "b", scope: ![[BLK]], {{.*}}type: ![[B_TYPE:[0-9]+]], isLocal: false, isDefinition: true)
! MODS-DAG: ![[B_TYPE]] = !DICompositeType(tag: DW_TAG_array_type, {{.*}}elements: ![[B_ELTS:[0-9]+]])
! MODS-DAG: ![[B_ELTS]] = !{![[B_SR:[0-9]+]]}
! MODS-DAG: ![[B_SR]] = !DISubrange(count: 2, lowerBound: 2)
! MODS-DAG: ![[E1_EXPR]] = !DIGlobalVariableExpression(var: ![[E1:[0-9]+]], expr: !DIExpression(DW_OP_plus_uconst, 4))
! MODS-DAG: ![[E1]] = distinct !DIGlobalVariable(name: "e1", scope: ![[MV]], {{.*}}isDefinition: true)
! MODS-DAG: ![[E2_EXPR]] = !DIGlobalVariableExpression(var: ![[E2:[0-9]+]], expr: !DIExpression())
! MODS-DAG: ![[E2]] = distinct !DIGlobalVariable(name: "e2", scope: ![[MV]], {{.*}}isDefinition: true)
! MODS-DAG: ![[X_EXPR]] = !DIGlobalVariableExpression(var: ![[X:[0-9]+]], expr: !DIExpression())
! MODS-DAG: ![[X]] = distinct !DIGlobalVariable(name: "x", linkageName: "c_x", scope: ![[MV]],
! MODS-DAG: ![[Z_EXPR]] = !DIGlobalVariableExpression(var: ![[Z:[0-9]+]], expr: !DIExpression())
! MODS-DAG: ![[Z]] = distinct !DIGlobalVariable(name: "z", linkageName: "_QMmvEz", scope: ![[MV]], {{.*}}type: ![[Z_TYPE:[0-9]+]],
! MODS-DAG: ![[Z_TYPE]] = !DICompositeType(tag: DW_TAG_array_type, {{.*}}elements: ![[Z_ELTS:[0-9]+]])
! MODS-DAG: ![[Z_ELTS]] = !{![[Z_SR:[0-9]+]]}
! MODS-DAG: ![[Z_SR]] = !DISubrange(count: 2, lowerBound: 5)

! A unit that uses a module lists the variables that are not in a global
! named after them. The compilation unit records the interfaces of the used
! module and of the module that declares its re-exported object.
! MAIN-FIR: {{^}}} loc(#[[MAIN_MOD_LOC:loc[0-9]*]])
! MAIN-FIR-DAG: #[[MAIN_MOD_LOC]] = loc(fused<[#scope_mod_mv, #scope_mod_top]>[
! MAIN-FIR-DAG: #scope_mod_mv = #fir.scope<module "mv", exports = [{{.*}}], variables = [#fir.module_var<"a" : i32, common "blk" ("blk_"), #{{loc[0-9]*}}>, #fir.module_var<"b" : !fir.array<2xi32>, common "blk" ("blk_"), offset 4, lbounds [2], #{{loc[0-9]*}}>, #fir.module_var<"e1" : !fir.array<4xi32>, equivalence "e1", offset 4, lbounds [0], #{{loc[0-9]*}}>, #fir.module_var<"e2" : !fir.array<4xi32>, equivalence "e1", #{{loc[0-9]*}}>, #fir.module_var<"x", bind_c "c_x", #{{loc[0-9]*}}>], #{{loc[0-9]*}}>
! MAIN-FIR-DAG: #scope_mod_top = #fir.scope<module "top", exports = [<"e2" from "mv">], #{{loc[0-9]*}}>
! MAIN-FIR-DAG: #fir.scope<procedure "s", uses = [#fir.use<"top", #{{loc[0-9]*}}>, #fir.use<"mv", only [<"x">, <"b">], #{{loc[0-9]*}}>], #{{loc[0-9]*}}>

! The storage of the EQUIVALENCE is not described as the variable it is named
! after.
! MAIN-STORAGE-NOT: name: "e1"
! MAIN-DAG: @blk_ = common global [12 x i8] zeroinitializer, align 4, !dbg ![[B_EXPR:[0-9]+]], !dbg !{{[0-9]+}}
! MAIN-DAG: @_QMmvEe1 = external global [20 x i8], align 64, !dbg ![[E2_EXPR:[0-9]+]]
! MAIN-DAG: @c_x = external global i32, !dbg ![[X_EXPR:[0-9]+]]
! MAIN-DAG: ![[B_EXPR]] = !DIGlobalVariableExpression(var: ![[B:[0-9]+]], expr: !DIExpression(DW_OP_plus_uconst, 4))
! MAIN-DAG: ![[B]] = distinct !DIGlobalVariable(name: "b", scope: ![[BLK:[0-9]+]], {{.*}}isDefinition: false)
! MAIN-DAG: ![[BLK]] = !DICommonBlock(scope: ![[MV:[0-9]+]], declaration: null, name: "blk"
! MAIN-DAG: ![[MV]] = !DIModule(scope: null, name: "mv", isDecl: true)
! MAIN-DAG: ![[E2_EXPR]] = !DIGlobalVariableExpression(var: ![[E2:[0-9]+]], expr: !DIExpression())
! MAIN-DAG: ![[E2]] = distinct !DIGlobalVariable(name: "e2", scope: ![[MV]], {{.*}}isDefinition: false)
! MAIN-DAG: ![[X_EXPR]] = !DIGlobalVariableExpression(var: ![[X:[0-9]+]], expr: !DIExpression())
! MAIN-DAG: ![[X]] = distinct !DIGlobalVariable(name: "x", linkageName: "c_x", scope: ![[MV]],
! MAIN-DAG: !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: ![[S:[0-9]+]], entity: ![[E2]], file: !{{[0-9]+}}, line: 2)
! MAIN-DAG: !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: ![[S]], entity: ![[X]], file: !{{[0-9]+}}, line: 3)
! MAIN-DAG: !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: ![[S]], entity: ![[B]], file: !{{[0-9]+}}, line: 3)

!--- mods.f90
module mv
  type :: t
    integer :: i = 1
  end type
  type(t) :: tv
  integer :: z(5:6) = [3, 4]
  integer, bind(c, name="c_x") :: x = 1
  integer :: a, b(2:3)
  common /blk/ a, b
  integer :: e1(0:3), e2(4)
  equivalence (e1(1), e2(3))
end module

module top
  use mv, only: e2
end module

!--- main.f90
subroutine s()
  use top
  use mv, only: x, b
  print *, e2, x, b
end subroutine

!--- bound-common.f90
module bindings
  integer :: x
  common /blk/ x
  bind(c, name="actual") /blk/
  integer, bind(c, name="blk_") :: decoy = 91
end module

! The COMMON block's source name and its actual object-file label differ.
! Its default spelling happens to name an unrelated variable in this unit.
! BOUND-DAG: @actual = common global [4 x i8] zeroinitializer, align 4, !dbg ![[XEX:[0-9]+]]
! BOUND-DAG: @blk_ = global i32 91, !dbg ![[DEX:[0-9]+]]
! BOUND-DAG: ![[XEX]] = !DIGlobalVariableExpression(var: ![[X:[0-9]+]],
! BOUND-DAG: ![[X]] = distinct !DIGlobalVariable(name: "x", scope: ![[BLK:[0-9]+]],
! BOUND-DAG: ![[BLK]] = !DICommonBlock(scope: ![[MOD:[0-9]+]], declaration: null, name: "blk",
! BOUND-DAG: ![[DEX]] = !DIGlobalVariableExpression(var: ![[D:[0-9]+]],
! BOUND-DAG: ![[D]] = distinct !DIGlobalVariable(name: "decoy", linkageName: "blk_", scope: ![[MOD]],
