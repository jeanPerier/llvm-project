// Test data layout from targets is correctly applied in codegen
// REQUIRES: x86-registered-target
// RUN: %flang_fc1 -emit-llvm -triple x86_64-unknown-linux-gnu %s -o - | FileCheck %s --check-prefix=ALIGN_I64_64

// i686-pc-elfiamcu is an ABI selected for this test because it aligns i64 to 32 bits.
// There is no claim that flang supports this target correctly end-to-end.
// RUN: %flang_fc1 -emit-llvm -triple i686-pc-elfiamcu %s -o - | FileCheck %s --check-prefix=ALIGN_I64_32

func.func @_QPtest() -> !fir.heap<!fir.array<2x!fir.type<_QFtestTt{i:i64,j:i32}>>> {
  %0 = fir.allocmem !fir.array<2x!fir.type<_QFtestTt{i:i64,j:i32}>>
  return %0 : !fir.heap<!fir.array<2x!fir.type<_QFtestTt{i:i64,j:i32}>>>
}
// ALIGN_I64_64: call ptr @malloc(i64 32)
// ALIGN_I64_32: call ptr @malloc(i64 24)
