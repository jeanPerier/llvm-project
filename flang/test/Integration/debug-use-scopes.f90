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

! The x of the BLOCK is declared in the BLOCK scope.
! FIR-LABEL: func.func @_QPblocks
! FIR: hlfir.declare %{{.*}} uniq_name("_QFblocksEx") : {{.*}} loc(#[[X:loc[0-9]*]])
! FIR: hlfir.declare %{{.*}} uniq_name("_QFblocksB1Ex") : {{.*}} loc(#[[B1X:loc[0-9]*]])

! The modules are not defined here, so the module location lists none of them.
! The used modules have no location, and the USE statements of pa describe what
! it re-exports. The USE with an empty ONLY list is dropped.
! FIR: {{^}}} loc(#[[MODLOC:loc[0-9]*]])
! FIR-DAG: #[[MODLOC]] = loc("{{.*}}main.f90":0:0)
! FIR-DAG: #[[UNKNOWN:loc[0-9]*]] = loc(unknown)
! FIR-DAG: #scope_mod_ancestor = #fir.scope<module "ancestor", #[[UNKNOWN]]>
! FIR-DAG: #scope_mod_first = #fir.scope<module "first", #[[UNKNOWN]]>
! FIR-DAG: #scope_mod_second = #fir.scope<module "second", #[[UNKNOWN]]>
! FIR-DAG: #scope_mod_pm = #fir.scope<module "pm", #[[UNKNOWN]]>
! FIR-DAG: #scope_mod_pa = #fir.scope<module "pa", uses = [#fir.use<#scope_mod_pm, only [<"pa_ren" -> "other">, <"used_var">], #[[UNKNOWN]]>], #[[UNKNOWN]]>
! FIR-DAG: #[[BLOCKS:scope[0-9]*]] = #fir.scope<procedure "blocks", uses = [#fir.use<#scope_mod_ancestor, #{{.*}}>], #{{.*}}>
! FIR-DAG: #fir.scope<procedure "onlys", uses = [#fir.use<#scope_mod_pm, only [<"used_var">], #{{.*}}>, #fir.use<#scope_mod_pm, only [<"ren" -> "other">], #{{.*}}>], #{{.*}}>
! FIR-DAG: #fir.scope<procedure "viapa", uses = [#fir.use<#scope_mod_pa, #{{.*}}>], #{{.*}}>
! FIR-DAG: #[[B1:scope[0-9]*]] = #fir.scope<construct, parent = #[[BLOCKS]], uses = [#fir.use<#scope_mod_first, only [<"item">, <"first_unused">], #{{.*}}>], #{{.*}}>
! FIR-DAG: #[[B1X]] = loc(fused<#[[B1]]>[#{{.*}}])
! FIR-DAG: #fir.scope<construct, parent = #[[B1]], uses = [#fir.use<#scope_mod_second, #{{.*}}>], #{{.*}}>

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

! "use pa" imports module pa and what pa re-exports from pm.
! CHECK-DAG: ![[VIAPA_SP:.*]] = distinct !DISubprogram(name: "viapa", {{.*}}retainedNodes: ![[VIAPA_NODES:[0-9]+]]
! CHECK-DAG: ![[VIAPA_NODES]] = !{![[IMP_PA:[0-9]+]], ![[IMP_PA_REN:[0-9]+]], ![[IMP_PA_USED:[0-9]+]]}
! CHECK-DAG: ![[IMP_PA]] = !DIImportedEntity(tag: DW_TAG_imported_module, scope: ![[VIAPA_SP]], entity: ![[PA_MOD]], file: !{{.*}}, line: 26)
! CHECK-DAG: ![[IMP_PA_USED]] = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: ![[VIAPA_SP]], entity: ![[USED]], file: !{{.*}}, line: 26)
! CHECK-DAG: ![[IMP_PA_REN]] = !DIImportedEntity(tag: DW_TAG_imported_declaration, name: "pa_ren", scope: ![[VIAPA_SP]], entity: ![[OTHER]], file: !{{.*}}, line: 26)
