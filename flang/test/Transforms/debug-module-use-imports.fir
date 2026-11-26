// RUN: fir-opt --split-input-file --add-debug-info --mlir-print-debuginfo %s | FileCheck %s

// AddDebugInfo imports the re-exports of the module into the
// subprogram that uses it. Each re-export refers to the defining module.

module {
  fir.global @_QMchildEx : i32 {
    %0 = fir.zero_bits i32
    fir.has_value %0 : i32
  }
  fir.global @_QMparentEy : i32 {
    %0 = fir.zero_bits i32
    fir.has_value %0 : i32
  }
  func.func @_QQmain() attributes {fir.bindc_name = "MAIN_USE"} {
    return
  } loc(#loc_main)
} loc(#loc_mod)

#child = #fir.scope<module "child", loc("m.f90":1:1)>
#parent = #fir.scope<module "parent", uses = [#fir.use<"child", loc("m.f90":5:3)>], loc("m.f90":4:1)>
#main = #fir.scope<procedure "main_use", uses = [#fir.use<"parent", loc("m.f90":10:3)>], loc("m.f90":9:1)>
#loc_mod = loc(fused<[#child, #parent]>["m.f90":0:0])
#loc_main = loc(fused<#main>["m.f90":9:1])

// CHECK-DAG: #[[MOD_CHILD:.+]] = #llvm.di_module<{{.*}}name = "child"{{.*}}>
// CHECK-DAG: #[[MOD_PARENT:.+]] = #llvm.di_module<{{.*}}name = "parent"{{.*}}>

// The re-exported variable and the module import are at the USE statement of the subprogram.
// CHECK-DAG: #[[SP_REC:.+]] = #llvm.di_subprogram<recId = distinct[[[RECID:[0-9]+]]]<>, isRecSelf = true{{.*}}name = "main_use"
// CHECK-DAG: #[[X:.+]] = #llvm.di_global_variable<scope = #[[MOD_CHILD]], name = "x"{{.*}}>
// CHECK-DAG: #llvm.di_imported_entity<tag = DW_TAG_imported_declaration, scope = #[[SP_REC]], entity = #[[X]], file = #{{.*}}, line = 10>
// CHECK-DAG: #llvm.di_imported_entity<tag = DW_TAG_imported_module, scope = #[[SP_REC]], entity = #[[MOD_PARENT]], file = #{{.*}}, line = 10>
// CHECK-DAG: #llvm.di_subprogram<recId = distinct[[[RECID]]]<>{{.*}}name = "main_use"{{.*}}retainedNodes = {{.+}}>

// -----

// When a module defined in this unit re-exports an external module via an
// unrestricted USE, the external module is also imported so unreferenced
// variables described in its own object file remain visible.
module {
  fir.global @_QMmidEy : i32 {
    %0 = fir.zero_bits i32
    fir.has_value %0 : i32
  }
  func.func @_QQmain() attributes {fir.bindc_name = "EXT_REEXP"} {
    return
  } loc(#loc_ext_main)
} loc(#loc_ext_mod)

#ext_leaf = #fir.scope<module "ext_leaf", loc(unknown)>
#mid = #fir.scope<module "mid", uses = [#fir.use<"ext_leaf", loc("m.f90":2:3)>], loc("m.f90":1:1)>
#ext_main = #fir.scope<procedure "ext_reexp", uses = [#fir.use<"mid", loc("m.f90":6:3)>], loc("m.f90":5:1)>
#loc_ext_mod = loc(fused<[#ext_leaf, #mid]>["m.f90":0:0])
#loc_ext_main = loc(fused<#ext_main>["m.f90":5:1])

// CHECK-DAG: #[[EXT_LEAF:.+]] = #llvm.di_module<name = "ext_leaf", isDecl = true>
// CHECK-DAG: #[[MID:.+]] = #llvm.di_module<{{.*}}name = "mid"{{.*}}>
// CHECK-DAG: #[[SP2_REC:.+]] = #llvm.di_subprogram<recId = distinct[[[REC2:[0-9]+]]]<>, isRecSelf = true{{.*}}name = "ext_reexp"
// CHECK-DAG: #llvm.di_imported_entity<tag = DW_TAG_imported_module, scope = #[[SP2_REC]], entity = #[[MID]], file = #{{.*}}, line = 6>
// CHECK-DAG: #llvm.di_imported_entity<tag = DW_TAG_imported_module, scope = #[[SP2_REC]], entity = #[[EXT_LEAF]], file = #{{.*}}, line = 6>
