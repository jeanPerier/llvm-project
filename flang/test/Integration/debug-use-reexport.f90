! RUN: mkdir -p %t && %flang_fc1 -module-dir %t -debug-info-kind=standalone -emit-llvm %s -o - | FileCheck %s
! RUN: %flang_fc1 -module-dir %t -emit-hlfir -debug-info-kind=standalone -mmlir --mlir-print-debuginfo %s -o - | FileCheck %s --check-prefix=FIR

module origin
  integer :: x
end module
module facade
  use origin, only: exported => x
end module
subroutine client
  use facade, only: local => exported
  print *, local
end subroutine

! The declared variable belongs to origin even though the client imports it
! through a different module and under a different spelling.
! FIR-DAG: #fir.scope<procedure "client", uses = [#fir.use<"facade", only [<"local" -> "exported" from "origin" named "x">], #{{loc[0-9]*}}>], #{{loc[0-9]*}}>
! CHECK-DAG: ![[ORIGIN:[0-9]+]] = !DIModule({{.*}}name: "origin"
! CHECK-DAG: ![[X:[0-9]+]] = distinct !DIGlobalVariable(name: "x", {{.*}}scope: ![[ORIGIN]],
! CHECK-DAG: ![[CLIENT:[0-9]+]] = distinct !DISubprogram(name: "client",
! CHECK-DAG: !DIImportedEntity(tag: DW_TAG_imported_declaration, name: "local", scope: ![[CLIENT]], entity: ![[X]]

module names
  use origin, only: a => x, b => x
end module
subroutine show()
  use names, c => a
  print *, b, c
end subroutine

! A whole-module USE imports what a module defined in this unit re-exports,
! through the USE statements of that module. A rename of a re-export leaves
! the other binding of the same object visible.
! FIR-DAG: #fir.scope<module "names", uses = [#fir.use<"origin", only [<"a" -> "x">, <"b" -> "x">], #{{loc[0-9]*}}>], #{{loc[0-9]*}}>
! FIR-DAG: #fir.scope<procedure "show", uses = [#fir.use<"names", [<"c" -> "a" from "origin" named "x">], #{{loc[0-9]*}}>], #{{loc[0-9]*}}>
! CHECK-DAG: ![[NAMES:[0-9]+]] = !DIModule({{.*}}name: "names"
! CHECK-DAG: ![[SHOW:[0-9]+]] = distinct !DISubprogram(name: "show", {{.*}}retainedNodes: ![[NODES:[0-9]+]]
! CHECK-DAG: ![[NODES]] = !{![[IMPORT:[0-9]+]], ![[B:[0-9]+]]}
! CHECK-DAG: ![[IMPORT]] = !DIImportedEntity(tag: DW_TAG_imported_module, scope: ![[SHOW]], entity: ![[NAMES]], {{.*}}elements: ![[RENAMES:[0-9]+]])
! CHECK-DAG: ![[RENAMES]] = !{![[C:[0-9]+]]}
! CHECK-DAG: ![[C]] = !DIImportedEntity(tag: DW_TAG_imported_declaration, name: "c", scope: ![[SHOW]], entity: ![[X]],
! CHECK-DAG: ![[B]] = !DIImportedEntity(tag: DW_TAG_imported_declaration, name: "b", scope: ![[SHOW]], entity: ![[X]],

module priv_leaf
  integer :: hidden_x = 11
  integer :: exposed_z = 22
end module
module priv_mid
  use priv_leaf
  private
  public :: exposed_z
  integer, public :: mid_y = 33
end module
subroutine priv_consumer()
  integer :: hidden_x
  hidden_x = 99
  block
    use priv_mid
    print *, hidden_x, exposed_z, mid_y
  end block
end subroutine

! A module with PRIVATE accessibility only records its PUBLIC USE-associated
! bindings, so hidden_x is not imported into the nested BLOCK to shadow the
! host local variable of the same name.
! FIR-DAG: #fir.scope<module "priv_mid", uses = [#fir.use<"priv_leaf", only [<"exposed_z">], #{{loc[0-9]*}}>], #{{loc[0-9]*}}>
! CHECK-DAG: ![[PRIV_MID:[0-9]+]] = !DIModule({{.*}}name: "priv_mid"
! CHECK-DAG: ![[EXPOSED_GV:[0-9]+]] = distinct !DIGlobalVariable(name: "exposed_z", {{.*}}scope: !{{[0-9]+}},
! CHECK-DAG: ![[MID_Y_GV:[0-9]+]] = distinct !DIGlobalVariable(name: "mid_y", {{.*}}scope: ![[PRIV_MID]],
! CHECK-DAG: distinct !DISubprogram(name: "priv_consumer", {{.*}}retainedNodes: ![[PRIV_NODES:[0-9]+]]
! CHECK-DAG: ![[PRIV_NODES]] = !{![[IMP_PM:[0-9]+]], ![[IMP_MY:[0-9]+]], ![[IMP_EZ:[0-9]+]]}
! CHECK-DAG: ![[IMP_PM]] = !DIImportedEntity(tag: DW_TAG_imported_module, scope: ![[PRIV_LB:[0-9]+]], entity: ![[PRIV_MID]],
! CHECK-DAG: ![[IMP_MY]] = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: ![[PRIV_LB]], entity: ![[MID_Y_GV]],
! CHECK-DAG: ![[IMP_EZ]] = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: ![[PRIV_LB]], entity: ![[EXPOSED_GV]],
