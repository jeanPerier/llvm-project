// RUN: fir-opt %s --stack-arrays | FileCheck %s
// RUN: fir-opt %s --allocation-placement | FileCheck %s

func.func private @extent() -> index
func.func private @use(!fir.heap<!fir.char<1,?>>)
func.func private @use_ref(!fir.ref<!fir.char<1,?>>)

// An existing alloca before the temporary must not be reclaimed at block exit.
// CHECK-LABEL: func.func @existing_alloca
// CHECK-NOT: llvm.intr.stacksave
// CHECK: fir.alloca
// CHECK: fir.allocmem
// CHECK: fir.freemem
// CHECK-NOT: llvm.intr.stackrestore
// CHECK: return
func.func @existing_alloca(%again: i1) {
  cf.br ^loop
^loop:
  %n = fir.call @extent() : () -> index
  %a = fir.alloca !fir.char<1,?>(%n : index)
  %s = fir.allocmem !fir.char<1,?>(%n : index)
  fir.call @use(%s) : (!fir.heap<!fir.char<1,?>>) -> ()
  fir.freemem %s : !fir.heap<!fir.char<1,?>>
  cf.cond_br %again, ^loop, ^exit
^exit:
  fir.call @use_ref(%a) : (!fir.ref<!fir.char<1,?>>) -> ()
  return
}

// An allocation whose free is in another block cannot join this block scope.
// CHECK-LABEL: func.func @free_in_successor
// CHECK-NOT: llvm.intr.stacksave
// CHECK: fir.allocmem
// CHECK: fir.freemem
// CHECK-NOT: llvm.intr.stackrestore
// CHECK: return
func.func @free_in_successor(%again: i1) {
  cf.br ^loop
^loop:
  %n = fir.call @extent() : () -> index
  %s = fir.allocmem !fir.char<1,?>(%n : index)
  fir.call @use(%s) : (!fir.heap<!fir.char<1,?>>) -> ()
  cf.br ^cleanup
^cleanup:
  fir.freemem %s : !fir.heap<!fir.char<1,?>>
  cf.cond_br %again, ^loop, ^exit
^exit:
  return
}

// The character stack preference must be suppressed in device regions by the
// shared placement policy, including when stack-arrays is not enabled.
// CHECK-LABEL: func.func @offload
// CHECK: acc.parallel
// CHECK-NOT: fir.alloca
// CHECK: fir.allocmem !fir.char
// CHECK: fir.freemem
// CHECK: acc.yield
func.func @offload(%n: index) {
  acc.parallel {
    %s = fir.allocmem !fir.char<1,?>(%n : index)
    fir.call @use(%s) : (!fir.heap<!fir.char<1,?>>) -> ()
    fir.freemem %s : !fir.heap<!fir.char<1,?>>
    acc.yield
  }
  return
}

// Each block can reclaim its own promoted allocations. A nested allocmem
// does not prevent promotion of the outer block's temporary.
// CHECK-LABEL: func.func @nested_allocations
// CHECK: %[[OUTER:.*]] = llvm.intr.stacksave
// CHECK: fir.alloca !fir.char
// CHECK: fir.if
// CHECK: %[[INNER:.*]] = llvm.intr.stacksave
// CHECK: fir.alloca !fir.char
// CHECK: fir.call @use
// CHECK: llvm.intr.stackrestore %[[INNER]]
// CHECK: }
// CHECK: fir.call @use
// CHECK: llvm.intr.stackrestore %[[OUTER]]
// CHECK: cf.cond_br
// CHECK: return
func.func @nested_allocations(%again: i1, %condition: i1) {
  cf.br ^loop
^loop:
  %n = fir.call @extent() : () -> index
  %outer = fir.allocmem !fir.char<1,?>(%n : index)
  fir.if %condition {
    %m = fir.call @extent() : () -> index
    %inner = fir.allocmem !fir.char<1,?>(%m : index)
    fir.call @use(%inner) : (!fir.heap<!fir.char<1,?>>) -> ()
    fir.freemem %inner : !fir.heap<!fir.char<1,?>>
  }
  fir.call @use(%outer) : (!fir.heap<!fir.char<1,?>>) -> ()
  fir.freemem %outer : !fir.heap<!fir.char<1,?>>
  cf.cond_br %again, ^loop, ^exit
^exit:
  return
}
