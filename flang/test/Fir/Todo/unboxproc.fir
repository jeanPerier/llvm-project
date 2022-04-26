// RUN: tco -disable-boxed-procedure-rewrite %s | FileCheck %s

// CHECK-LABEL: define i8* @boxing_match({ void (i32)*, i8* }
// CHECK-SAME: %[[arg:[0-9]+]]
// CHECK: extractvalue { void (i32)*, i8* } %[[arg]], 0
// CHECK: %[[res:.*]] = extractvalue { void (i32)*, i8* } %[[arg]], 1
// CHECK: ret i8* %[[res]]

func @boxing_match(%arg0 : !fir.boxproc<(i32) -> ()>) -> !fir.ref<i8> {
  %proc:2 = fir.unboxproc %arg0 : (!fir.boxproc<(i32) -> ()>) -> ((i32) -> (), !fir.ref<tuple<i32, f32>>)
  %cast = fir.convert %proc#1 : (!fir.ref<tuple<i32, f32>>) -> !fir.ref<i8>
  return %cast : !fir.ref<i8>
}
