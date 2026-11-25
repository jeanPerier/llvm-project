// RUN: fir-opt --add-debug-info --mlir-print-debuginfo %s | FileCheck %s --implicit-check-not='#llvm.di_global_variable<' --implicit-check-not='DW_TAG_imported_declaration'

// A pass deleted the original globals of a described module, then created
// unrelated storage under the freed names. None of these four operations
// carries the original source identity. The surviving USE and module
// information must not attach source variables or imports to the new storage.
#m = #fir.scope<module "m", exports = [<"plain">, <"bound">, <"member">, <"eq">],
    variables = [#fir.module_var<"bound", bind_c "slot", loc(unknown)>,
                 #fir.module_var<"member" : i32, common "blk" ("blk_"), loc(unknown)>,
                 #fir.module_var<"eq" : i32, equivalence "eq", loc(unknown)>],
    loc("m.f90":1:1)>
#s = #fir.scope<procedure "s", uses = [#fir.use<"m", only [<"plain">, <"bound">, <"member">, <"eq">], loc("m.f90":10:3)>], loc("m.f90":9:1)>
module {
  fir.global @_QMmEplain(91 : i32) : i32
  fir.global @slot(91 : i32) : i32
  fir.global @blk_(91 : i32) : i32
  fir.global @_QMmEeq(91 : i32) : i32
  func.func @_QPs() {
    return
  } loc(fused<#s>["m.f90":9:1])
} loc(fused<[#m]>["m.f90":0:0])

// CHECK: #llvm.di_subprogram<
