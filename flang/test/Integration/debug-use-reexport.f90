! RUN: mkdir -p %t && %flang_fc1 -module-dir %t -debug-info-kind=standalone -emit-llvm %s -o - | FileCheck %s

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
! CHECK-DAG: ![[ORIGIN:[0-9]+]] = !DIModule({{.*}}name: "origin"
! CHECK-DAG: ![[X:[0-9]+]] = distinct !DIGlobalVariable(name: "x", {{.*}}scope: ![[ORIGIN]],
! CHECK-DAG: ![[CLIENT:[0-9]+]] = distinct !DISubprogram(name: "client",
! CHECK-DAG: !DIImportedEntity(tag: DW_TAG_imported_declaration, name: "local", scope: ![[CLIENT]], entity: ![[X]]
