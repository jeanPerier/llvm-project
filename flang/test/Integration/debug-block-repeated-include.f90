! RUN: split-file %s %t && %flang_fc1 -emit-llvm -debug-info-kind=standalone -module-dir %t -I %t %t/test.F90 -o - | FileCheck %s

! The two blocks open at the same position in block.inc but import different x's.
! Their lexical blocks and code locations must stay distinct in LLVM IR.
! CHECK-DAG: ![[X1:[0-9]+]] = distinct !DIGlobalVariable(name: "x", linkageName: "_QMm1Ex"
! CHECK-DAG: ![[X2:[0-9]+]] = distinct !DIGlobalVariable(name: "x", linkageName: "_QMm2Ex"
! CHECK-DAG: ![[S1:[0-9]+]] = distinct !DILexicalBlock(scope: !{{[0-9]+}}, file: !{{[0-9]+}}, line: 9,
! CHECK-DAG: ![[S2:[0-9]+]] = distinct !DILexicalBlock(scope: !{{[0-9]+}}, file: !{{[0-9]+}}, line: 12,
! CHECK-DAG: ![[B1:[0-9]+]] = distinct !DILexicalBlock(scope: ![[S1]], file: !{{[0-9]+}}, line: 1,
! CHECK-DAG: ![[B2:[0-9]+]] = distinct !DILexicalBlock(scope: ![[S2]], file: !{{[0-9]+}}, line: 1,
! CHECK-DAG: !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: ![[B1]], entity: ![[X1]]
! CHECK-DAG: !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: ![[B2]], entity: ![[X2]]
! CHECK-DAG: !DILocation({{.*}}scope: ![[B1]])
! CHECK-DAG: !DILocation({{.*}}scope: ![[B2]])

!--- test.F90
module m1
  integer :: x=1
end module
module m2
  integer :: x=2
end module
program example
#define M m1
#include "block.inc"
#undef M
#define M m2
#include "block.inc"
end program

!--- block.inc
block
  use M, only: x
  print *, x
end block
