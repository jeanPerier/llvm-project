// RUN: fir-opt --add-debug-info --mlir-print-debuginfo %s | FileCheck %s

// The scope of a module defined here lists the modules it uses; AddDebugInfo
// flattens these transitive imports onto the subprogram that uses the module.

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
#parent = #fir.scope<module "parent", uses = [#fir.use<#child, loc("m.f90":5:3)>], loc("m.f90":4:1)>
#main = #fir.scope<procedure "main_use", uses = [#fir.use<#parent, loc("m.f90":10:3)>], loc("m.f90":9:1)>
#loc_mod = loc(fused<[#child, #parent]>["m.f90":0:0])
#loc_main = loc(fused<#main>["m.f90":9:1])

// CHECK-DAG: #[[MOD_CHILD:.+]] = #llvm.di_module<{{.*}}name = "child"{{.*}}>
// CHECK-DAG: #[[MOD_PARENT:.+]] = #llvm.di_module<{{.*}}name = "parent"{{.*}}>

// Both imports are at the USE statement of the subprogram.
// CHECK-DAG: #[[SP_REC:.+]] = #llvm.di_subprogram<recId = distinct[[[RECID:[0-9]+]]]<>, isRecSelf = true{{.*}}name = "main_use"
// CHECK-DAG: #llvm.di_imported_entity<tag = DW_TAG_imported_module, scope = #[[SP_REC]], entity = #[[MOD_CHILD]], file = #{{.*}}, line = 10>
// CHECK-DAG: #llvm.di_imported_entity<tag = DW_TAG_imported_module, scope = #[[SP_REC]], entity = #[[MOD_PARENT]], file = #{{.*}}, line = 10>
// CHECK-DAG: #llvm.di_subprogram<recId = distinct[[[RECID]]]<>{{.*}}name = "main_use"{{.*}}retainedNodes = {{.+}}>
