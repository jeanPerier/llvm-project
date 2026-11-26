! RUN: %flang_fc1 -emit-llvm -debug-info-kind=standalone %s -o - | FileCheck %s
! RUN: %flang_fc1 -emit-hlfir -debug-info-kind=standalone -mmlir --mlir-print-debuginfo %s -o - | FileCheck %s --check-prefix=FIR

module host_data
  integer :: x = 11
end module
module local_data
  integer, private :: x = 99
  integer :: y = 22
end module

subroutine host()
  use host_data, only: x
  call inner()
contains
  subroutine inner()
    use local_data
    print *, x, y
  end subroutine
end subroutine

! The storage of the private x records that it is private.
! FIR-DAG: #fir.module_var<"local_data", "x", private, #{{loc[0-9]*}}>
! FIR-DAG: #fir.module_var<"local_data", "y", #{{loc[0-9]*}}>

! Both x globals exist in this unit. The unrestricted USE in inner imports
! y alone by name, so x still resolves to the host's variable.
! CHECK-DAG: ![[HOST_X:[0-9]+]] = distinct !DIGlobalVariable(name: "x", linkageName: "_QMhost_dataEx"
! CHECK-DAG: distinct !DIGlobalVariable(name: "x", linkageName: "_QMlocal_dataEx"
! CHECK-DAG: ![[Y:[0-9]+]] = distinct !DIGlobalVariable(name: "y", linkageName: "_QMlocal_dataEy"
! CHECK-DAG: ![[MOD:[0-9]+]] = !DIModule({{.*}}name: "local_data"
! CHECK-DAG: ![[HOST:[0-9]+]] = distinct !DISubprogram(name: "host",
! CHECK-DAG: !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: ![[HOST]], entity: ![[HOST_X]],
! CHECK-DAG: ![[INNER:[0-9]+]] = distinct !DISubprogram(name: "inner", {{.*}}retainedNodes: ![[NODES:[0-9]+]]
! CHECK-DAG: ![[NODES]] = !{![[IMPORT_MOD:[0-9]+]], ![[IMPORT_Y:[0-9]+]]}
! CHECK-DAG: ![[IMPORT_MOD]] = !DIImportedEntity(tag: DW_TAG_imported_module, scope: ![[INNER]], entity: ![[MOD]],
! CHECK-DAG: ![[IMPORT_Y]] = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: ![[INNER]], entity: ![[Y]],
