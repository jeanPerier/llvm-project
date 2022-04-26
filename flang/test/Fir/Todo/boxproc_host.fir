// RUN: tco -disable-boxed-procedure-rewrite %s | FileCheck %s

// The codegen lowering converts a boxproc into a pair of pointers.
// fir_boxproc_host extracts the second element of the pair.
// The boxed-procedure-rewrite pass, when enabled, converts all boxproc to
// thunks, and the boxproc_host op is not needed nor used.

// CHECK-LABEL: define { i32, double }* @test(
// CHECK: %2 = extractvalue { void (i32)*, i8* } %0, 1, !dbg !7
// CHECK: %3 = bitcast i8* %2 to { i32, double }*, !dbg !9
// CHECK: ret { i32, double }* %3, !dbg !10

func @test(%bproc: !fir.boxproc<(i32) -> ()>) -> !fir.ref<tuple<i32, f64>> {
  %tuple = fir.boxproc_host %bproc : (!fir.boxproc<(i32) -> ()>) -> (!fir.ref<i8>)
  %cast = fir.convert %tuple : (!fir.ref<i8>) -> !fir.ref<tuple<i32, f64>>
  return %cast : !fir.ref<tuple<i32, f64>>
}
