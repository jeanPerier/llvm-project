! Test that a user-defined non-intrinsic module with the same name as an
! intrinsic module (F2023 19.2 & 14.2.2) keeps its own DWARF module, variables,
! and USE imports distinct from USE, INTRINSIC associations in the same or
! separate compilation units.

! RUN: split-file %s %t
! RUN: %flang_fc1 -emit-hlfir -debug-info-kind=standalone -mmlir --mlir-print-debuginfo -module-dir %t %t/same-tu.f90 -o - | FileCheck %s --check-prefix=FIR
! RUN: %flang_fc1 -emit-llvm -debug-info-kind=standalone -module-dir %t %t/same-tu.f90 -o - | FileCheck %s --check-prefix=LLVM
! RUN: %flang_fc1 -emit-llvm -debug-info-kind=standalone -I %t %t/sep-tu.f90 -o - | FileCheck %s --check-prefix=SEP

! FIR-DAG: #fir.module_var<"iso_c_binding", "user_x", #{{loc[0-9]*}}>
! FIR-DAG: #fir.module_var<"iso_c_binding", "c_int", #{{loc[0-9]*}}>
! FIR-DAG: #fir.module_var<"__fortran_intrinsic__iso_c_binding", "c_int", #{{loc[0-9]*}}>
! FIR-DAG: #scope_mod_iso_c_binding = #fir.scope<module "iso_c_binding", uses = [#fir.use<"mod_helper", #{{loc[0-9]*}}>], #{{loc[0-9]*}}>
! FIR-DAG: #scope_mod___fortran_intrinsic__iso_c_binding = #fir.scope<module "__fortran_intrinsic__iso_c_binding", #{{loc[0-9]*}}>
! FIR-DAG: #scope_mod_bridge = #fir.scope<module "bridge", uses = [#fir.use<"__fortran_intrinsic__iso_c_binding", only [<"intr_k" -> "c_int">], #{{loc[0-9]*}}>], #{{loc[0-9]*}}>
! FIR-DAG: #fir.scope<procedure "sub_user", uses = [#fir.use<"iso_c_binding", #{{loc[0-9]*}}>, #fir.use<"bridge", only [<"intr_k" from "__fortran_intrinsic__iso_c_binding" named "c_int">], #{{loc[0-9]*}}>], #{{loc[0-9]*}}>
! FIR-DAG: #fir.scope<procedure "sub_intr", uses = [#fir.use<"__fortran_intrinsic__iso_c_binding", #{{loc[0-9]*}}>], #{{loc[0-9]*}}>

! The user module remains a definition on the compile unit (not overwritten by
! the intrinsic module's unknown location), and the intrinsic module carries its
! own prefixed DWARF module name.
! LLVM-DAG: ![[CU:[0-9]+]] = distinct !DICompileUnit({{.*}}globals: ![[GLOBALS:[0-9]+]]
! LLVM-DAG: ![[USER_MOD:[0-9]+]] = !DIModule(scope: ![[CU]], name: "iso_c_binding", {{.*}}line:
! LLVM-DAG: ![[INTR_MOD:[0-9]+]] = !DIModule(scope: null, name: "__fortran_intrinsic__iso_c_binding", isDecl: true)
! LLVM-DAG: ![[USER_X:[0-9]+]] = distinct !DIGlobalVariable(name: "user_x", linkageName: "_QMiso_c_bindingEuser_x", scope: ![[USER_MOD]], {{.*}}isDefinition: true)
! LLVM-DAG: ![[USER_CINT:[0-9]+]] = distinct !DIGlobalVariable(name: "c_int", linkageName: "_QMiso_c_bindingEc_int", scope: ![[USER_MOD]], {{.*}}isDefinition: true)
! LLVM-DAG: ![[INTR_CINT:[0-9]+]] = distinct !DIGlobalVariable(name: "c_int", linkageName: "_QMiso_c_bindingECc_int", scope: ![[INTR_MOD]], {{.*}}isDefinition: true)
! LLVM-DAG: ![[HELPER_V:[0-9]+]] = distinct !DIGlobalVariable(name: "helper_v", linkageName: "_QMmod_helperEhelper_v",
! LLVM-DAG: ![[SUB_USER:[0-9]+]] = distinct !DISubprogram(name: "sub_user",
! LLVM-DAG: !DIImportedEntity(tag: DW_TAG_imported_module, scope: ![[SUB_USER]], entity: ![[USER_MOD]],
! LLVM-DAG: !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: ![[SUB_USER]], entity: ![[HELPER_V]],
! LLVM-DAG: !DIImportedEntity(tag: DW_TAG_imported_declaration, name: "intr_k", scope: ![[SUB_USER]], entity: ![[INTR_CINT]],
! LLVM-DAG: ![[SUB_INTR:[0-9]+]] = distinct !DISubprogram(name: "sub_intr",
! LLVM-DAG: !DIImportedEntity(tag: DW_TAG_imported_module, scope: ![[SUB_INTR]], entity: ![[INTR_MOD]],

! Separate-TU consumer importing both the user .mod and the intrinsic module.
! SEP-DAG: ![[SEP_USER_MOD:[0-9]+]] = !DIModule(scope: null, name: "iso_c_binding", isDecl: true)
! SEP-DAG: ![[SEP_INTR_MOD:[0-9]+]] = !DIModule(scope: null, name: "__fortran_intrinsic__iso_c_binding", isDecl: true)
! SEP-DAG: ![[SEP_USER_CINT:[0-9]+]] = distinct !DIGlobalVariable(name: "c_int", linkageName: "_QMiso_c_bindingEc_int", scope: ![[SEP_USER_MOD]],
! SEP-DAG: ![[SEP_INTR_CINT:[0-9]+]] = distinct !DIGlobalVariable(name: "c_int", linkageName: "_QMiso_c_bindingECc_int", scope: ![[SEP_INTR_MOD]],
! SEP-DAG: !DIImportedEntity(tag: DW_TAG_imported_declaration, name: "uk", scope: !{{[0-9]+}}, entity: ![[SEP_USER_CINT]],
! SEP-DAG: !DIImportedEntity(tag: DW_TAG_imported_declaration, name: "ik", scope: !{{[0-9]+}}, entity: ![[SEP_INTR_CINT]],

!--- same-tu.f90
module mod_helper
  implicit none
  integer :: helper_v = 99
end module

module iso_c_binding
  use mod_helper
  implicit none
  integer :: user_x = 42
  integer :: c_int = 777
end module

module bridge
  use, intrinsic :: iso_c_binding, only: intr_k => c_int
  implicit none
end module

subroutine sub_user(res)
  use, non_intrinsic :: iso_c_binding
  use bridge, only: intr_k
  implicit none
  integer, intent(out) :: res
  res = user_x + c_int + helper_v + intr_k
end subroutine

subroutine sub_intr(res)
  use, intrinsic :: iso_c_binding
  implicit none
  integer(c_int), intent(out) :: res
  res = c_int
end subroutine

!--- sep-tu.f90
subroutine sep_both(res)
  use, non_intrinsic :: iso_c_binding, only: uk => c_int
  implicit none
  integer, intent(out) :: res
  block
    use, intrinsic :: iso_c_binding, only: ik => c_int
    import, only: res
    integer :: tmp
    tmp = ik
    res = tmp
  end block
  res = res + uk
end subroutine
