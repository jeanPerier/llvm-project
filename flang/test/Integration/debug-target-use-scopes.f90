! RUN: %flang_fc1 -fopenmp -emit-llvm -debug-info-kind=standalone %s -o - | FileCheck %s --implicit-check-not='c"loc(fused<'

module target_data
  integer :: x, y, z
  !$omp declare target(x,y,z)
end module

! The two targets have the same host import. Each has a different BLOCK
! binding of v which shadows the host's v at its executable statements.
! CHECK-DAG: ![[X:[0-9]+]] = distinct !DIGlobalVariable(name: "x", linkageName: "_QMtarget_dataEx"
! CHECK-DAG: ![[Y:[0-9]+]] = distinct !DIGlobalVariable(name: "y", linkageName: "_QMtarget_dataEy"
! CHECK-DAG: ![[Z:[0-9]+]] = distinct !DIGlobalVariable(name: "z", linkageName: "_QMtarget_dataEz"
subroutine work(i)
  use target_data, only: v => z
  integer :: i
  block
    use target_data, only: v => x
! CHECK-DAG: ![[ONE:[0-9]+]] = distinct !DISubprogram(name: "__omp_offloading_{{.*}}_work__l[[@LINE+1]]", {{.*}}retainedNodes: ![[ONE_NODES:[0-9]+]]
    !$omp target map(tofrom:i)
! CHECK-DAG: !DILocation(line: [[@LINE+1]], column: {{[0-9]+}}, scope: ![[ONE_BLOCK:[0-9]+]])
      i = i + v
      block
        use target_data, only: w => z
! CHECK-DAG: !DILocation(line: [[@LINE+1]], column: {{[0-9]+}}, scope: ![[INNER:[0-9]+]])
        i = i + w
      end block
    !$omp end target
  end block
  block
    use target_data, only: v => y
! CHECK-DAG: ![[TWO:[0-9]+]] = distinct !DISubprogram(name: "__omp_offloading_{{.*}}_work__l[[@LINE+1]]", {{.*}}retainedNodes: ![[TWO_NODES:[0-9]+]]
    !$omp target map(tofrom:i)
! CHECK-DAG: !DILocation(line: [[@LINE+1]], column: {{[0-9]+}}, scope: ![[TWO_BLOCK:[0-9]+]])
      i = i + v
    !$omp end target
  end block
end subroutine

! CHECK-DAG: ![[ONE_NODES]] = !{![[ONE_HOST:[0-9]+]], ![[ONE_V:[0-9]+]], ![[INNER_W:[0-9]+]]}
! CHECK-DAG: ![[TWO_NODES]] = !{![[TWO_HOST:[0-9]+]], ![[TWO_V:[0-9]+]]}
! CHECK-DAG: ![[ONE_BLOCK]] = distinct !DILexicalBlock(scope: ![[ONE]],
! CHECK-DAG: ![[INNER]] = distinct !DILexicalBlock(scope: ![[ONE_BLOCK]],
! CHECK-DAG: ![[TWO_BLOCK]] = distinct !DILexicalBlock(scope: ![[TWO]],
! CHECK-DAG: ![[ONE_HOST]] = !DIImportedEntity(tag: DW_TAG_imported_declaration, name: "v", scope: ![[ONE]], entity: ![[Z]],
! CHECK-DAG: ![[ONE_V]] = !DIImportedEntity(tag: DW_TAG_imported_declaration, name: "v", scope: ![[ONE_BLOCK]], entity: ![[X]],
! CHECK-DAG: ![[INNER_W]] = !DIImportedEntity(tag: DW_TAG_imported_declaration, name: "w", scope: ![[INNER]], entity: ![[Z]],
! CHECK-DAG: ![[TWO_HOST]] = !DIImportedEntity(tag: DW_TAG_imported_declaration, name: "v", scope: ![[TWO]], entity: ![[Z]],
! CHECK-DAG: ![[TWO_V]] = !DIImportedEntity(tag: DW_TAG_imported_declaration, name: "v", scope: ![[TWO_BLOCK]], entity: ![[Y]],

! The mapped dummy has the lifetime of the outlined procedure.
! CHECK-DAG: !DILocalVariable(name: "i", arg: 1, scope: ![[ONE]],
! CHECK-DAG: !DILocalVariable(name: "i", arg: 1, scope: ![[TWO]],
