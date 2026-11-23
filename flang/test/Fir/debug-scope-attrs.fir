// Test the parsing and printing of the attributes that describe the scopes
// for the debug information.

// RUN: fir-opt --mlir-print-debuginfo %s | fir-opt --mlir-print-debuginfo | FileCheck %s

// RUN: fir-opt --strip-debuginfo --mlir-print-debuginfo %s | FileCheck %s --check-prefix=STRIP --implicit-check-not='#fir.scope' --implicit-check-not='#fir.use'
// RUN: fir-opt --symbol-dce --mlir-print-debuginfo %s | FileCheck %s --check-prefix=DCE --implicit-check-not='fir.global'

// STRIP: func.func @f()
// DCE: func.func @f()
// DCE: #fir.scope<module "o", uses = [#fir.use<"n", only [<"d" from "p">]

// CHECK-LABEL: func.func @f()
// CHECK: fir.call @g() : () -> () loc(#[[CALL1:.*]])
// CHECK: fir.call @g() : () -> () loc(#[[CALL2:.*]])
// CHECK: } loc(#[[FUNC:.*]])
// CHECK: } loc(#[[MODULE:.*]])

// CHECK-DAG: #scope_mod_p = #fir.scope<module "p", #{{.*}}>
// CHECK-DAG: #scope_mod_n = #fir.scope<module "n", uses = [#fir.use<"p", only [<"x">], #{{.*}}>], #{{.*}}>
// CHECK-DAG: #scope_mod_o = #fir.scope<module "o", uses = [#fir.use<"n", only [<"d" from "p">], #{{.*}}>], #{{.*}}>
// CHECK-DAG: #scope_mod_m = #fir.scope<module "m", uses = [#fir.use<"n", #{{.*}}>, #fir.use<"o", [<"a" -> "b">, <"c" -> "d">], #{{.*}}>], #{{.*}}>
// CHECK-DAG: #scope_mod_s = #fir.scope<module "s", parent = #scope_mod_m, #{{.*}}>
// CHECK-DAG: #[[PROC:.*]] = #fir.scope<procedure "f", parent = #scope_mod_s, uses = [#fir.use<"m", #{{.*}}>, #fir.use<"n", only [<"x">, <"y" -> "z">, <"w">], #{{.*}}>], #{{.*}}>
// CHECK-DAG: #[[BLOCK:.*]] = #fir.scope<construct, parent = #[[PROC]], uses = [#fir.use<"o", #{{.*}}>], #{{.*}}>
// CHECK-DAG: #[[BRANCH:.*]] = #fir.scope<construct, parent = #[[BLOCK]], #{{.*}}>
// CHECK-DAG: #[[CALL1]] = loc(fused<#[[BLOCK]]>[
// CHECK-DAG: #[[CALL2]] = loc(fused<#[[BRANCH]]>[
// CHECK-DAG: #[[FUNC]] = loc(fused<#[[PROC]]>[
// CHECK-DAG: #[[MODULE]] = loc(fused<[#scope_mod_p, #scope_mod_n, #scope_mod_o, #scope_mod_m, #scope_mod_s]>[

#p = #fir.scope<module "p", loc(unknown)>
#n = #fir.scope<module "n", uses = [#fir.use<"p", only [<"x">], loc(unknown)>], loc(unknown)>
#o = #fir.scope<module "o", uses = [#fir.use<"n", only [<"d" from "p">], loc(unknown)>], loc(unknown)>
#m = #fir.scope<module "m", uses = [#fir.use<"n", loc("test.f90":2:3)>, #fir.use<"o", [<"a" -> "b">, <"c" -> "d">], loc("test.f90":3:3)>], loc("test.f90":1:1)>
#s = #fir.scope<module "s", parent = #m, loc("test.f90":10:1)>
#f = #fir.scope<procedure "f", parent = #s, uses = [#fir.use<"m", loc("test.f90":13:5)>, #fir.use<"n", only [<"x">, <"y" -> "z">, <"w">], loc("test.f90":14:5)>], loc("test.f90":12:3)>
#b = #fir.scope<construct, parent = #f, uses = [#fir.use<"o", loc("test.f90":15:7)>], loc("test.f90":14:5)>
#c = #fir.scope<construct, parent = #b, loc("test.f90":16:7)>

module {
  fir.global private @_QMpEd : i32
  func.func private @g()
  func.func @f() {
    fir.call @g() : () -> () loc(fused<#b>["test.f90":15:7])
    fir.call @g() : () -> () loc(fused<#c>["test.f90":17:9])
    return
  } loc(fused<#f>["test.f90":12:3])
} loc(fused<[#p, #n, #o, #m, #s]>["test.f90":0:0])
