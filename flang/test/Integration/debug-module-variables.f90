! RUN: split-file %s %t
! RUN: %flang_fc1 -emit-hlfir -debug-info-kind=standalone -mmlir --mlir-print-debuginfo -module-dir %t %t/mods.f90 -o %t/mods.hlfir
! RUN: FileCheck %s --check-prefix=MODS-FIR --input-file=%t/mods.hlfir
! RUN: %flang_fc1 -emit-llvm -debug-info-kind=standalone -module-dir %t %t/mods.f90 -o - | FileCheck %s --check-prefix=MODS
! RUN: %flang_fc1 -emit-hlfir -debug-info-kind=standalone -mmlir --mlir-print-debuginfo -I %t %t/main.f90 -o - | FileCheck %s --check-prefix=MAIN-FIR
! RUN: %flang_fc1 -emit-llvm -debug-info-kind=standalone -I %t %t/main.f90 -o - | FileCheck %s --check-prefix=MAIN --implicit-check-not='linkageName: "_QMmvEe1"'
! RUN: %flang_fc1 -emit-llvm -debug-info-kind=standalone -module-dir %t %t/bound-common.f90 -o - | FileCheck %s --check-prefix=BOUND
! RUN: %flang_fc1 -emit-llvm -debug-info-kind=standalone -module-dir %t %t/shared.f90 -o - | FileCheck %s --check-prefix=SHARED
! RUN: %flang_fc1 -emit-llvm -debug-info-kind=standalone -I %t %t/use-shared.f90 -o - | FileCheck %s --check-prefix=USE-SHARED
! RUN: %flang_fc1 -emit-hlfir -debug-info-kind=standalone -module-dir %t %t/equiv-used.f90 -o - | FileCheck %s --check-prefix=EQUIV-USED
! RUN: %flang_fc1 -fsyntax-only -module-dir %t %t/typed-storage.f90
! RUN: %flang_fc1 -emit-hlfir -debug-info-kind=standalone -I %t %t/use-typed-storage.f90 -o - | FileCheck %s --check-prefix=TYPED --implicit-check-not=fir.type_info --implicit-check-not=EXdtX
! RUN: %flang_fc1 -emit-llvm -debug-info-kind=standalone -I %t %t/submod-common-and-block.f90 -o - | FileCheck %s --check-prefix=SUBMOD-BLK --implicit-check-not='!DILocalVariable(name: "e2"'
! RUN: %flang_fc1 -emit-fir -debug-info-kind=standalone -module-dir %t %t/mods.f90 -o %t/mods-no-loc.fir
! RUN: %flang_fc1 -emit-llvm -debug-info-kind=standalone %t/mods-no-loc.fir -o - | FileCheck %s --check-prefix=ROUNDTRIP

! Check the module variables that their globals do not describe completely:
! those with a binding label, in a COMMON block or an EQUIVALENCE, or whose
! lower bounds are not all 1.

! Each source descriptor is attached to the global providing its storage.
! Only members sharing storage need an explicit type.
! MODS-FIR-DAG: module_vars = [#fir.module_var<"mv", "x", #{{loc[0-9]*}}>]}>
! MODS-FIR-DAG: module_vars = [#fir.module_var<"mv", "z", lbounds [5], #{{loc[0-9]*}}>]}>
! MODS-FIR-DAG: module_vars = [#fir.module_var<"mv", "a", type i32, common "blk", #{{loc[0-9]*}}>, #fir.module_var<"mv", "b", type !fir.array<2xi32>, common "blk", offset 4, lbounds [2], #{{loc[0-9]*}}>]}>
! MODS-FIR-DAG: module_vars = [#fir.module_var<"mv", "e1", type !fir.array<4xi32>, offset 4, lbounds [0], #{{loc[0-9]*}}>, #fir.module_var<"mv", "e2", type !fir.array<4xi32>, #{{loc[0-9]*}}>]}>

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

! The using unit reconstructs member information from .mod files.
! MAIN-FIR-DAG: #fir.module_var<"mv", "x", #{{loc[0-9]*}}>
! MAIN-FIR-DAG: #fir.module_var<"mv", "b", type !fir.array<2xi32>, common "blk", offset 4, lbounds [2], #{{loc[0-9]*}}>
! MAIN-FIR-DAG: #fir.module_var<"mv", "e2", type !fir.array<4xi32>, #{{loc[0-9]*}}>
! MAIN-FIR: {{^}}} loc(#[[MAIN_MOD_LOC:loc[0-9]*]])
! MAIN-FIR-DAG: #[[MAIN_MOD_LOC]] = loc(fused<[#scope_mod_mv, #scope_mod_top]>[
! MAIN-FIR-DAG: #scope_mod_top = #fir.scope<module "top", #{{loc[0-9]*}}>
! MAIN-FIR-DAG: #fir.scope<procedure "s", uses = [#fir.use<"top", only [<"e2" from "mv">], #{{loc[0-9]*}}>, #fir.use<"mv", only [<"x">, <"b">], #{{loc[0-9]*}}>], #{{loc[0-9]*}}>

! The storage of the EQUIVALENCE is not described as the variable it is named
! after.
! MAIN-DAG: @blk_ = common global [12 x i8] zeroinitializer, align 4, !dbg !{{[0-9]+}}
! MAIN-DAG: @_QMmvEe1 = external global [20 x i8], align 64, !dbg !{{[0-9]+}}
! MAIN-DAG: @c_x = external global i32, !dbg ![[X_EXPR:[0-9]+]]
! MAIN-DAG: ![[B:[0-9]+]] = distinct !DIGlobalVariable(name: "b", scope: ![[BLK:[0-9]+]], {{.*}}isDefinition: false)
! MAIN-DAG: !DIGlobalVariableExpression(var: ![[B]], expr: !DIExpression(DW_OP_plus_uconst, 4))
! MAIN-DAG: ![[BLK]] = !DICommonBlock(scope: ![[MV:[0-9]+]], declaration: null, name: "blk"
! MAIN-DAG: ![[MV]] = !DIModule(scope: null, name: "mv", isDecl: true)
! MAIN-DAG: ![[E2:[0-9]+]] = distinct !DIGlobalVariable(name: "e2", scope: ![[MV]], {{.*}}isDefinition: false)
! MAIN-DAG: !DIGlobalVariableExpression(var: ![[E2]], expr: !DIExpression())
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
  use top, only: e2
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

!--- shared.f90
module left_view
  integer :: tag, data(0:1)
  common /shared/ tag, data
end module

module right_view
  integer :: data(2), tag
  common /shared/ data, tag
end module

! The same three words have different names and array bounds in each module.
! SHARED-DAG: @shared_ = common global [12 x i8] zeroinitializer, align 4, !dbg !{{[0-9]+}}, !dbg !{{[0-9]+}}, !dbg !{{[0-9]+}}, !dbg !{{[0-9]+}}
! SHARED-DAG: ![[LEFT:[0-9]+]] = !DIModule({{.*}}name: "left_view",
! SHARED-DAG: ![[RIGHT:[0-9]+]] = !DIModule({{.*}}name: "right_view",
! SHARED-DAG: ![[LB:[0-9]+]] = !DICommonBlock(scope: ![[LEFT]], declaration: null, name: "shared",
! SHARED-DAG: ![[RB:[0-9]+]] = !DICommonBlock(scope: ![[RIGHT]], declaration: null, name: "shared",
! SHARED-DAG: ![[LD:[0-9]+]] = distinct !DIGlobalVariable(name: "data", scope: ![[LB]], {{.*}}type: ![[LTYPE:[0-9]+]],
! SHARED-DAG: ![[LT:[0-9]+]] = distinct !DIGlobalVariable(name: "tag", scope: ![[LB]],
! SHARED-DAG: ![[RD:[0-9]+]] = distinct !DIGlobalVariable(name: "data", scope: ![[RB]], {{.*}}type: ![[RTYPE:[0-9]+]],
! SHARED-DAG: ![[RT:[0-9]+]] = distinct !DIGlobalVariable(name: "tag", scope: ![[RB]],
! SHARED-DAG: !DIGlobalVariableExpression(var: ![[LD]], expr: !DIExpression(DW_OP_plus_uconst, 4))
! SHARED-DAG: !DIGlobalVariableExpression(var: ![[LT]], expr: !DIExpression())
! SHARED-DAG: !DIGlobalVariableExpression(var: ![[RD]], expr: !DIExpression())
! SHARED-DAG: !DIGlobalVariableExpression(var: ![[RT]], expr: !DIExpression(DW_OP_plus_uconst, 8))
! SHARED-DAG: ![[LTYPE]] = !DICompositeType(tag: DW_TAG_array_type, {{.*}}elements: ![[LDIMS:[0-9]+]])
! SHARED-DAG: ![[LDIMS]] = !{![[LRANGE:[0-9]+]]}
! SHARED-DAG: ![[LRANGE]] = !DISubrange(count: 2, lowerBound: 0)
! SHARED-DAG: ![[RTYPE]] = !DICompositeType(tag: DW_TAG_array_type, {{.*}}elements: ![[RDIMS:[0-9]+]])
! SHARED-DAG: ![[RDIMS]] = !{![[RRANGE:[0-9]+]]}
! SHARED-DAG: ![[RRANGE]] = !DISubrange(count: 2{{(, lowerBound: 1)?}})

!--- use-shared.f90
subroutine read_views
  use left_view, only: first => tag
  use right_view, only: last => tag
  print *, first, last
end subroutine

! Both .mod files refer to the same storage. Their imports select different
! variables named tag, at the first and last words of the block.
! USE-SHARED-DAG: !DIImportedEntity(tag: DW_TAG_imported_declaration, name: "first", scope: !{{[0-9]+}}, entity: ![[FIRST:[0-9]+]],
! USE-SHARED-DAG: !DIImportedEntity(tag: DW_TAG_imported_declaration, name: "last", scope: !{{[0-9]+}}, entity: ![[LAST:[0-9]+]],
! USE-SHARED-DAG: ![[CU_SHARED:[0-9]+]] = distinct !DICompileUnit({{.*}}globals: ![[SHARED_GLOBALS:[0-9]+]]
! USE-SHARED-DAG: ![[SHARED_GLOBALS]] = !{![[FIRST_EX:[0-9]+]], ![[LAST_EX:[0-9]+]]}
! USE-SHARED-DAG: ![[FIRST_EX]] = !DIGlobalVariableExpression(var: ![[FIRST]], expr: !DIExpression())
! USE-SHARED-DAG: ![[LAST_EX]] = !DIGlobalVariableExpression(var: ![[LAST]], expr: !DIExpression(DW_OP_plus_uconst, 8))

!--- equiv-used.f90
module leaf
  integer :: za, zb
  equivalence (za, zb)
end module

module user
  use leaf
  integer :: ma, mb
  equivalence (ma, mb)
end module

! The storage of an EQUIVALENCE describes only the variables of its module,
! although a module using another one has the storage of both at offset 0.
! EQUIV-USED-DAG: fir.global @_QMleafEza <{alignment = 64 : i64, module_vars = [#fir.module_var<"leaf", "za", type i32, #{{loc[0-9]*}}>, #fir.module_var<"leaf", "zb", type i32, #{{loc[0-9]*}}>]}>
! EQUIV-USED-DAG: fir.global @_QMuserEma <{alignment = 64 : i64, module_vars = [#fir.module_var<"user", "ma", type i32, #{{loc[0-9]*}}>, #fir.module_var<"user", "mb", type i32, #{{loc[0-9]*}}>]}>

!--- typed-storage.f90
module typed_storage
  type t
    sequence
    integer :: i
    real :: r
  end type
  type t2
    sequence
    integer :: k = 3
  end type
  integer :: other
  type(t) :: dt
  common /dc/ other, dt
  type(t2) :: e1
  real :: e2
  equivalence (e1, e2)
end module

!--- use-typed-storage.f90
subroutine use_typed_storage
  use typed_storage, only: other, e2
  print *, other, e2
end subroutine

! Translating the types of the members only for their description does not
! create runtime type information.
! TYPED-DAG: #fir.module_var<"typed_storage", "dt", type !fir.type<_QMtyped_storageTt,sequence{i:i32,r:f32}>, common "dc", offset 4, #{{loc[0-9]*}}>
! TYPED-DAG: #fir.module_var<"typed_storage", "e1", type !fir.type<_QMtyped_storageTt2,sequence{k:i32}>, #{{loc[0-9]*}}>

!--- submod-common-and-block.f90
submodule(mv) mv_sub
contains
  subroutine sub_proc()
    integer :: z
    integer :: local_c
    common /proc_cb/ local_c
    z = 0
    block
      integer :: z
      local_c = b(2) + e2(1)
      z = local_c
    end block
  end subroutine
end submodule

! Host-associated COMMON and EQUIVALENCE members in a submodule and a procedure
! COMMON block declared inside a BLOCK construct verify cleanly without
! creating duplicate COMMON blocks in lexical blocks or local variables for
! EQUIVALENCE members.
! SUBMOD-BLK-DAG: @blk_ = common global [12 x i8] zeroinitializer, align 4, !dbg ![[SB_BEX:[0-9]+]]
! SUBMOD-BLK-DAG: @proc_cb_ = common global [4 x i8] zeroinitializer, align 4, !dbg ![[SB_LCEX:[0-9]+]]
! SUBMOD-BLK-DAG: ![[SB_MV:[0-9]+]] = !DIModule(scope: null, name: "mv", isDecl: true)
! SUBMOD-BLK-DAG: ![[SB_BLK:[0-9]+]] = !DICommonBlock(scope: ![[SB_MV]], declaration: null, name: "blk"
! SUBMOD-BLK-DAG: ![[SB_B:[0-9]+]] = distinct !DIGlobalVariable(name: "b", scope: ![[SB_BLK]],
! SUBMOD-BLK-DAG: ![[SB_BEX]] = !DIGlobalVariableExpression(var: ![[SB_B]], expr: !DIExpression(DW_OP_plus_uconst, 4))
! SUBMOD-BLK-DAG: ![[SB_LCEX]] = !DIGlobalVariableExpression(var: ![[SB_LC:[0-9]+]], expr: !DIExpression())
! SUBMOD-BLK-DAG: ![[SB_LC]] = distinct !DIGlobalVariable(name: "local_c", linkageName: "_QMmvSmv_subFsub_procElocal_c", scope: ![[SB_PCB:[0-9]+]],
! SUBMOD-BLK-DAG: ![[SB_PCB]] = !DICommonBlock(scope: ![[SB_SP:[0-9]+]], declaration: null, name: "proc_cb"
! SUBMOD-BLK-DAG: ![[SB_SP]] = distinct !DISubprogram(name: "sub_proc",

! Textual FIR round-tripping without --mlir-print-debuginfo still preserves
! module variable DWARF via module_vars.
! ROUNDTRIP-DAG: ![[RT_CU:[0-9]+]] = distinct !DICompileUnit({{.*}}globals: ![[RT_GLOBALS:[0-9]+]]
! ROUNDTRIP-DAG: ![[RT_MV:[0-9]+]] = !DIModule(scope: ![[RT_CU]], name: "mv"
! ROUNDTRIP-DAG: distinct !DIGlobalVariable(name: "x", linkageName: "c_x", scope: ![[RT_MV]],
