// Test the parsing and printing of the attributes that describe the scopes
// for the debug information.

// RUN: fir-opt --mlir-print-debuginfo %s | fir-opt --mlir-print-debuginfo | FileCheck %s

// CHECK-LABEL: func.func @f()
// CHECK: fir.call @g() : () -> () loc(#[[CALL1:.*]])
// CHECK: fir.call @g() : () -> () loc(#[[CALL2:.*]])
// CHECK: } loc(#[[FUNC:.*]])
// CHECK: } loc(#[[MODULE:.*]])

// CHECK-DAG: #scope_mod_p = #fir.scope<module "p", #{{.*}}>
// CHECK-DAG: #scope_mod_n = #fir.scope<module "n", uses = [#fir.use<#scope_mod_p, only [<"x">], #{{.*}}>], #{{.*}}>
// CHECK-DAG: #scope_mod_o = #fir.scope<module "o", #{{.*}}>
// CHECK-DAG: #scope_mod_m = #fir.scope<module "m", uses = [#fir.use<#scope_mod_n, #{{.*}}>, #fir.use<#scope_mod_o, [<"a" -> "b">, <"c" -> "d" from #scope_mod_p>], #{{.*}}>], variables = [#fir.module_var<"z", lbounds [5, -1], #{{.*}}>, #fir.module_var<"x", bind_c "c_x", #{{.*}}>, #fir.module_var<"a" : i32, common "blk", #{{.*}}>, #fir.module_var<"b" : !fir.array<2xi32>, common "", offset 4, lbounds [0], #{{.*}}>, #fir.module_var<"e" : !fir.array<4xf32>, equivalence "e", offset 8, #{{.*}}>], #{{.*}}>
// CHECK-DAG: #scope_mod_s = #fir.scope<module "s", parent = #scope_mod_m, variables = [#fir.module_var<"y", bind_c "y", lbounds [0], #{{.*}}>], #{{.*}}>
// CHECK-DAG: #[[PROC:.*]] = #fir.scope<procedure "f", parent = #scope_mod_s, uses = [#fir.use<#scope_mod_m, #{{.*}}>, #fir.use<#scope_mod_n, only [<"x">, <"y" -> "z">, <"w" from #scope_mod_p>], #{{.*}}>], #{{.*}}>
// CHECK-DAG: #[[BLOCK:.*]] = #fir.scope<construct, parent = #[[PROC]], uses = [#fir.use<#scope_mod_o, #{{.*}}>], #{{.*}}>
// CHECK-DAG: #[[BRANCH:.*]] = #fir.scope<construct, parent = #[[BLOCK]], #{{.*}}>
// CHECK-DAG: #[[CALL1]] = loc(fused<#[[BLOCK]]>[
// CHECK-DAG: #[[CALL2]] = loc(fused<#[[BRANCH]]>[
// CHECK-DAG: #[[FUNC]] = loc(fused<#[[PROC]]>[
// CHECK-DAG: #[[MODULE]] = loc(fused<[#scope_mod_m, #scope_mod_s]>[

#p = #fir.scope<module "p", loc(unknown)>
#n = #fir.scope<module "n", uses = [#fir.use<#p, only [<"x">], loc(unknown)>], loc(unknown)>
#o = #fir.scope<module "o", loc(unknown)>
#m = #fir.scope<module "m", uses = [#fir.use<#n, loc("test.f90":2:3)>, #fir.use<#o, [<"a" -> "b">, <"c" -> "d" from #p>], loc("test.f90":3:3)>],
                variables = [#fir.module_var<"z", lbounds [5, -1], loc("test.f90":3:3)>,
                             #fir.module_var<"x", bind_c "c_x", loc(unknown)>,
                             #fir.module_var<"a" : i32, common "blk", loc(unknown)>,
                             #fir.module_var<"b" : !fir.array<2xi32>, common "", offset 4, lbounds [0], loc(unknown)>,
                             #fir.module_var<"e" : !fir.array<4xf32>, equivalence "e", offset 8, loc(unknown)>],
                loc("test.f90":1:1)>
#s = #fir.scope<module "s", parent = #m, variables = [#fir.module_var<"y", bind_c "y", lbounds [0], loc(unknown)>], loc("test.f90":10:1)>
#f = #fir.scope<procedure "f", parent = #s, uses = [#fir.use<#m, loc("test.f90":13:5)>, #fir.use<#n, only [<"x">, <"y" -> "z">, <"w" from #p>], loc("test.f90":14:5)>], loc("test.f90":12:3)>
#b = #fir.scope<construct, parent = #f, uses = [#fir.use<#o, loc("test.f90":15:7)>], loc("test.f90":14:5)>
#c = #fir.scope<construct, parent = #b, loc("test.f90":16:7)>

module {
  func.func private @g()
  func.func @f() {
    fir.call @g() : () -> () loc(fused<#b>["test.f90":15:7])
    fir.call @g() : () -> () loc(fused<#c>["test.f90":17:9])
    return
  } loc(fused<#f>["test.f90":12:3])
} loc(fused<[#m, #s]>["test.f90":0:0])
