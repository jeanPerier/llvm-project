<<<<<<< HEAD
// RUN: tco %s | FileCheck %s
// REQUIRES: shell
=======
// RUN: tco --target=x86_64-unknown-linux-gnu %s | FileCheck %s
>>>>>>> 853e79d8d8af... [flang] Update tco tool pipline and add translation to LLVM IR

// Check that tco is working with a basic test.

func @_QQmain() {
  return
}

// CHECK: ; ModuleID = 'FIRModule'
// CHECK-LABEL: define void @_QQmain()
// CHECK:       ret void
