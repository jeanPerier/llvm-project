! RUN: split-file %s %t
! RUN: %flang_fc1 -emit-llvm-bc -debug-info-kind=standalone -module-dir %t %t/mods.f90 -o %t/mods.bc
! RUN: %flang_fc1 -emit-hlfir -debug-info-kind=standalone -mmlir --mlir-print-debuginfo -I %t %t/main.f90 -o - | FileCheck %s --check-prefix=FIR
! RUN: %flang_fc1 -emit-llvm -debug-info-kind=standalone -I %t %t/main.f90 -o - | FileCheck %s

! Check how USE statements in procedures and BLOCK constructs are described
! when the used modules are compiled in another file: USE in a BLOCK is
! imported in a lexical block, ONLY lists only import what they list, a
! variable that a used module re-exports is imported too, and variables that a
! BLOCK declares do not hide the outer ones in the whole procedure.

!--- mods.f90
module ancestor
  integer :: item = 11
end module

module first
  integer :: item = 22
  integer :: first_unused = 23
end module

module second
  integer :: item = 33
  integer :: second_only = 34
end module

module pm
  integer :: used_var = 1
  integer :: unused_var = 2
  integer :: other = 3
end module

module pa
  use pm, only: used_var, pa_ren => other
  integer :: anchor = 44
end module

module facade
  use pm, only: a => other, b => other, t => used_var
end module

!--- main.f90
subroutine blocks(n)
  use ancestor
  integer :: n, x
  x = item
  block
    use first, only: item, first_unused
    integer :: x
    x = item + first_unused + n
    print *, x
    block
      use second
      print *, item, second_only
    end block
  end block
  print *, x
end subroutine

subroutine onlys()
  use pm, only:
  use pm, only: used_var
  use pm, only: ren => other
  print *, used_var, ren
end subroutine

subroutine viapa()
  use pa
  print *, anchor, used_var
end subroutine

subroutine layered()
  use facade, c => a
  use facade, d => t
  use facade, only: a
  print *, a, b, c, d
end subroutine

subroutine rename_first()
  use pm, only: a => other
  use pm
  print *, a, used_var
end subroutine

subroutine rename_last()
  use pm
  use pm, a => other
  print *, a, used_var
end subroutine

subroutine host_block()
  integer :: x
  x = 11
  block
    integer :: x
    x = 22
    print *, x
  end block
  print *, x
end subroutine

! The x of the BLOCK is declared in the BLOCK scope.
! FIR-LABEL: func.func @_QPblocks
! FIR: hlfir.declare %{{.*}} uniq_name("_QFblocksEx") : {{.*}} loc(#[[X:loc[0-9]*]])
! FIR: hlfir.declare %{{.*}} uniq_name("_QFblocksB1Ex") : {{.*}} loc(#[[B1X:loc[0-9]*]])

! The compilation unit lists the imported modules. A module read from a module
! file has no USE statements. A USE names the declaring module of an entity
! that the used module re-exports.
! FIR-DAG: #scope_mod_pa = #fir.scope<module "pa", #{{loc[0-9]*}}>
! FIR-DAG: #[[BLOCKS:scope[0-9]*]] = #fir.scope<procedure "blocks", uses = [#fir.use<"ancestor", #{{.*}}>], #{{.*}}>
! FIR-DAG: #fir.scope<procedure "onlys", uses = [#fir.use<"pm", only [<"used_var">], #{{.*}}>, #fir.use<"pm", only [<"ren" -> "other">], #{{.*}}>], #{{.*}}>
! FIR-DAG: #fir.scope<procedure "viapa", uses = [#fir.use<"pa", #{{.*}}>], #{{.*}}>
! FIR-DAG: #scope_mod_facade = #fir.scope<module "facade", #{{loc[0-9]*}}>
! FIR-DAG: #fir.scope<procedure "layered", uses = [#fir.use<"facade", [<"c" -> "a" from "pm" named "other">], #{{loc[0-9]*}}>, #fir.use<"facade", [<"d" -> "t" from "pm" named "used_var">], #{{loc[0-9]*}}>, #fir.use<"facade", only [<"a" from "pm" named "other">], #{{loc[0-9]*}}>]
! FIR-DAG: #[[B1:scope[0-9]*]] = #fir.scope<construct, parent = #[[BLOCKS]], uses = [#fir.use<"first", only [<"item">, <"first_unused">], #{{.*}}>], #{{.*}}>
! FIR-DAG: #[[B1X]] = loc(fused<#[[B1]]>[#{{.*}}])
! FIR-DAG: #fir.scope<construct, parent = #[[B1]], uses = [#fir.use<"second", #{{.*}}>], #{{.*}}>

! CHECK-DAG: ![[ANCESTOR_MOD:.*]] = !DIModule(scope: null, name: "ancestor", isDecl: true)
! CHECK-DAG: ![[SECOND_MOD:.*]] = !DIModule(scope: null, name: "second", isDecl: true)
! CHECK-DAG: ![[PA_MOD:.*]] = !DIModule(scope: null, name: "pa", isDecl: true)
! CHECK-DAG: ![[SECOND_ONLY:.*]] = distinct !DIGlobalVariable(name: "second_only", linkageName: "_QMsecondEsecond_only", scope: !{{.*}}, {{.*}}isDefinition: false)

! CHECK-DAG: ![[ITEM1:.*]] = distinct !DIGlobalVariable(name: "item", linkageName: "_QMfirstEitem", scope: !{{.*}}, {{.*}}isDefinition: false)
! CHECK-DAG: ![[UNUSED1:.*]] = distinct !DIGlobalVariable(name: "first_unused", linkageName: "_QMfirstEfirst_unused", scope: !{{.*}}, {{.*}}isDefinition: false)
! CHECK-DAG: ![[ITEM2:.*]] = distinct !DIGlobalVariable(name: "item", linkageName: "_QMsecondEitem", scope: !{{.*}}, {{.*}}isDefinition: false)
! CHECK-DAG: ![[USED:.*]] = distinct !DIGlobalVariable(name: "used_var", linkageName: "_QMpmEused_var", scope: !{{.*}}, {{.*}}isDefinition: false)
! CHECK-DAG: ![[OTHER:.*]] = distinct !DIGlobalVariable(name: "other", linkageName: "_QMpmEother", scope: !{{.*}}, {{.*}}isDefinition: false)

! The USE in the procedure is imported in the subprogram. The ones in the
! BLOCK constructs are imported in their lexical blocks, the same lexical
! blocks that hold the code and the x of the BLOCK. USE of a whole module in
! a nested BLOCK also imports its variables one by one so that they hide the
! ones imported in outer scopes.
! CHECK-DAG: ![[BLOCKS_SP:.*]] = distinct !DISubprogram(name: "blocks", {{.*}}retainedNodes: ![[BLOCKS_NODES:[0-9]+]]
! CHECK-DAG: ![[BLOCKS_NODES]] = !{![[IMP_ANCESTOR:[0-9]+]], {{.*}}}
! CHECK-DAG: ![[IMP_ANCESTOR]] = !DIImportedEntity(tag: DW_TAG_imported_module, scope: ![[BLOCKS_SP]], entity: ![[ANCESTOR_MOD]], file: !{{.*}}, line: 2)
! CHECK-DAG: ![[LB1:.*]] = distinct !DILexicalBlock(scope: ![[BLOCKS_SP]], file: !{{.*}}, line: 5, column: 3)
! CHECK-DAG: ![[LB2:.*]] = distinct !DILexicalBlock(scope: ![[LB1]], file: !{{.*}}, line: 10, column: 5)
! CHECK-DAG: !DILocalVariable(name: "x", scope: ![[LB1]]
! CHECK-DAG: !DILocalVariable(name: "x", scope: ![[BLOCKS_SP]]
! CHECK-DAG: !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: ![[LB1]], entity: ![[ITEM1]], file: !{{.*}}, line: 6)
! CHECK-DAG: !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: ![[LB1]], entity: ![[UNUSED1]], file: !{{.*}}, line: 6)
! CHECK-DAG: !DIImportedEntity(tag: DW_TAG_imported_module, scope: ![[LB2]], entity: ![[SECOND_MOD]], file: !{{.*}}, line: 11)
! CHECK-DAG: !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: ![[LB2]], entity: ![[ITEM2]], file: !{{.*}}, line: 11)
! CHECK-DAG: !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: ![[LB2]], entity: ![[SECOND_ONLY]], file: !{{.*}}, line: 11)
! CHECK-DAG: !DILocation(line: 12, column: {{.*}}, scope: ![[LB2]])

! "use pm, only:" imports nothing, and the ONLY lists import what they list.
! CHECK-DAG: ![[ONLYS_SP:.*]] = distinct !DISubprogram(name: "onlys", {{.*}}retainedNodes: ![[ONLYS_NODES:[0-9]+]]
! CHECK-DAG: ![[ONLYS_NODES]] = !{![[IMP_USED:[0-9]+]], ![[IMP_REN:[0-9]+]]}
! CHECK-DAG: ![[IMP_USED]] = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: ![[ONLYS_SP]], entity: ![[USED]], file: !{{.*}}, line: 20)
! CHECK-DAG: ![[IMP_REN]] = !DIImportedEntity(tag: DW_TAG_imported_declaration, name: "ren", scope: ![[ONLYS_SP]], entity: ![[OTHER]], file: !{{.*}}, line: 21)

! "use pa" imports module pa. What pa re-exports from pm is not imported, since
! this unit reads pa from a module file and does not know its USE statements.
! CHECK-DAG: ![[VIAPA_SP:.*]] = distinct !DISubprogram(name: "viapa", {{.*}}retainedNodes: ![[VIAPA_NODES:[0-9]+]]
! CHECK-DAG: ![[VIAPA_NODES]] = !{![[IMP_PA:[0-9]+]]}
! CHECK-DAG: ![[IMP_PA]] = !DIImportedEntity(tag: DW_TAG_imported_module, scope: ![[VIAPA_SP]], entity: ![[PA_MOD]], file: !{{.*}}, line: 26)

! The renames and the ONLY list resolve the re-exports of facade to pm.
! A second unrestricted USE renames t, and an explicit ONLY restores a. The
! other binding b of the same object is not imported, for the same reason as
! the re-exports of pa.
! CHECK-DAG: ![[LAYER:[0-9]+]] = distinct !DISubprogram(name: "layered", {{.*}}retainedNodes: ![[NODES:[0-9]+]]
! CHECK-DAG: ![[NODES]] = !{![[MOD:[0-9]+]], ![[A:[0-9]+]]}
! CHECK-DAG: ![[MOD]] = !DIImportedEntity(tag: DW_TAG_imported_module, scope: ![[LAYER]], {{.*}}elements: ![[RENAMES:[0-9]+]])
! CHECK-DAG: ![[RENAMES]] = !{![[C:[0-9]+]], ![[D:[0-9]+]]}
! CHECK-DAG: ![[C]] = !DIImportedEntity(tag: DW_TAG_imported_declaration, name: "c", scope: ![[LAYER]], entity: ![[OTHER]], file: !{{.*}}, line: 31)
! CHECK-DAG: ![[D]] = !DIImportedEntity(tag: DW_TAG_imported_declaration, name: "d", scope: ![[LAYER]], entity: ![[USED]], file: !{{.*}}, line: 32)
! CHECK-DAG: ![[A]] = !DIImportedEntity(tag: DW_TAG_imported_declaration, name: "a", scope: ![[LAYER]], entity: ![[OTHER]], file: !{{.*}}, line: 33)

! The whole-module import excludes the renamed variable in either statement
! order, including when the rename comes from an ONLY list.
! CHECK-DAG: ![[FIRST:[0-9]+]] = distinct !DISubprogram(name: "rename_first",
! CHECK-DAG: !DIImportedEntity(tag: DW_TAG_imported_module, scope: ![[FIRST]], {{.*}}line: 39, elements: ![[FIRST_LIST:[0-9]+]])
! CHECK-DAG: ![[FIRST_LIST]] = !{![[FIRST_A:[0-9]+]]}
! CHECK-DAG: ![[FIRST_A]] = !DIImportedEntity(tag: DW_TAG_imported_declaration, name: "a", scope: ![[FIRST]], entity: ![[OTHER]], file: !{{.*}}, line: 38)
! CHECK-DAG: ![[LAST:[0-9]+]] = distinct !DISubprogram(name: "rename_last",
! CHECK-DAG: !DIImportedEntity(tag: DW_TAG_imported_module, scope: ![[LAST]], {{.*}}line: 44, elements: ![[LAST_LIST:[0-9]+]])
! CHECK-DAG: ![[LAST_LIST]] = !{![[LAST_A:[0-9]+]]}
! CHECK-DAG: ![[LAST_A]] = !DIImportedEntity(tag: DW_TAG_imported_declaration, name: "a", scope: ![[LAST]], entity: ![[OTHER]], file: !{{.*}}, line: 45)

! A BLOCK declaration can shadow its host's declaration even when neither
! scope has a USE. The two print statements must use the corresponding scope.
! CHECK-DAG: ![[HOST:[0-9]+]] = distinct !DISubprogram(name: "host_block",
! CHECK-DAG: ![[BLOCK:[0-9]+]] = distinct !DILexicalBlock(scope: ![[HOST]],
! CHECK-DAG: !DILocalVariable(name: "x", scope: ![[HOST]], {{.*}}line: 50,
! CHECK-DAG: !DILocalVariable(name: "x", scope: ![[BLOCK]], {{.*}}line: 53,
! CHECK-DAG: !DILocation(line: 55, column: {{[0-9]+}}, scope: ![[BLOCK]])
! CHECK-DAG: !DILocation(line: 57, column: {{[0-9]+}}, scope: ![[HOST]])
