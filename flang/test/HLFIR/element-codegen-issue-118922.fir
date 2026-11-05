// Test issue 113843 and 118922 fix: do not elide hlfir.elemental final as_expr
// copy if this is not the last operation.
// RUN: fir-opt %s --bufferize-hlfir | FileCheck %s

func.func @_QMmPbug(%val: !fir.char<1>, %var: !fir.ref<!fir.array<10x!fir.char<1>>>) {
  %c1 = arith.constant 1 : index
  %c10 = arith.constant 10 : index
  %1 = fir.shape %c10 : (index) -> !fir.shape<1>
  %expr = hlfir.elemental %1 typeparams %c1 unordered : (!fir.shape<1>, index) -> !hlfir.expr<10x!fir.char<1>> {
  ^bb0(%arg2: index):
    %alloc = fir.allocmem !fir.char<1>
    fir.store %val to %alloc : !fir.heap<!fir.char<1>>
    %addr = fir.convert %alloc: (!fir.heap<!fir.char<1>>) -> !fir.ref<!fir.char<1>>
    %9 = hlfir.as_expr %addr : (!fir.ref<!fir.char<1>>) -> !hlfir.expr<!fir.char<1>>
    fir.freemem  %alloc : !fir.heap<!fir.char<1>>
    hlfir.yield_element %9 : !hlfir.expr<!fir.char<1>> cleanup {
      hlfir.destroy %9 : !hlfir.expr<!fir.char<1>>
    }
  }
  hlfir.assign %expr to %var : !hlfir.expr<10x!fir.char<1>>, !fir.ref<!fir.array<10x!fir.char<1>>>
  hlfir.destroy %expr : !hlfir.expr<10x!fir.char<1>>
  return
}

// CHECK-LABEL:   func.func @_QMmPbug(
// CHECK-SAME:      %[[ARG0:[0-9]+|[a-zA-Z$._-][a-zA-Z0-9$._-]*]]: !fir.char<1>,
// CHECK-SAME:      %[[ARG1:[0-9]+|[a-zA-Z$._-][a-zA-Z0-9$._-]*]]: !fir.ref<!fir.array<10x!fir.char<1>>>) {
// CHECK:           %[[ALLOCA_0:.*]] = fir.alloca !fir.char<1> <{bindc_name = ".tmp"}>
// CHECK:           %[[CONSTANT_0:.*]] = arith.constant 1 : index
// CHECK:           %[[CONSTANT_1:.*]] = arith.constant 10 : index
// CHECK:           %[[SHAPE_0:.*]] = fir.shape %[[CONSTANT_1]] : (index) -> !fir.shape<1>
// CHECK:           %[[ALLOCMEM_0:.*]] = fir.allocmem !fir.array<10x!fir.char<1>> <{bindc_name = ".tmp.array", uniq_name = ""}>
// CHECK:           %[[DECLARE_0:.*]]:2 = hlfir.declare %[[ALLOCMEM_0]](%[[SHAPE_0]]) typeparams %[[CONSTANT_0]] uniq_name(".tmp.array") : (!fir.heap<!fir.array<10x!fir.char<1>>>, !fir.shape<1>, index) -> (!fir.heap<!fir.array<10x!fir.char<1>>>, !fir.heap<!fir.array<10x!fir.char<1>>>)
// CHECK:           %[[CONSTANT_2:.*]] = arith.constant true
// CHECK:           %[[CONSTANT_3:.*]] = arith.constant 1 : index
// CHECK:           fir.do_loop %[[VAL_0:.*]] = %[[CONSTANT_3]] to %[[CONSTANT_1]] step %[[CONSTANT_3]] unordered {
// CHECK:             %[[ALLOCMEM_1:.*]] = fir.allocmem !fir.char<1>
// CHECK:             fir.store %[[ARG0]] to %[[ALLOCMEM_1]] : !fir.heap<!fir.char<1>>
// CHECK:             %[[CONVERT_0:.*]] = fir.convert %[[ALLOCMEM_1]] : (!fir.heap<!fir.char<1>>) -> !fir.ref<!fir.char<1>>
// CHECK:             %[[CONSTANT_4:.*]] = arith.constant 1 : index
// CHECK:             %[[DECLARE_1:.*]]:2 = hlfir.declare %[[ALLOCA_0]] typeparams %[[CONSTANT_4]] uniq_name(".tmp") : (!fir.ref<!fir.char<1>>, index) -> (!fir.ref<!fir.char<1>>, !fir.ref<!fir.char<1>>)
// CHECK:             hlfir.assign %[[CONVERT_0]] to %[[DECLARE_1]]#0 temporary_lhs : !fir.ref<!fir.char<1>>, !fir.ref<!fir.char<1>>
// CHECK:             %[[CONSTANT_5:.*]] = arith.constant false
// CHECK:             %[[UNDEFINED_0:.*]] = fir.undefined tuple<!fir.ref<!fir.char<1>>, i1>
// CHECK:             %[[INSERT_VALUE_0:.*]] = fir.insert_value %[[UNDEFINED_0]], %[[CONSTANT_5]], [1 : index] : (tuple<!fir.ref<!fir.char<1>>, i1>, i1) -> tuple<!fir.ref<!fir.char<1>>, i1>
// CHECK:             %[[INSERT_VALUE_1:.*]] = fir.insert_value %[[INSERT_VALUE_0]], %[[DECLARE_1]]#0, [0 : index] : (tuple<!fir.ref<!fir.char<1>>, i1>, !fir.ref<!fir.char<1>>) -> tuple<!fir.ref<!fir.char<1>>, i1>
// CHECK:             fir.freemem %[[ALLOCMEM_1]] : !fir.heap<!fir.char<1>>
// CHECK:             %[[DESIGNATE_0:.*]] = hlfir.designate %[[DECLARE_0]]#0 (%[[VAL_0]])  typeparams %[[CONSTANT_0]] : (!fir.heap<!fir.array<10x!fir.char<1>>>, index, index) -> !fir.ref<!fir.char<1>>
// CHECK:             hlfir.assign %[[DECLARE_1]]#0 to %[[DESIGNATE_0]] temporary_lhs : !fir.ref<!fir.char<1>>, !fir.ref<!fir.char<1>>
// CHECK:           }
// CHECK:           %[[UNDEFINED_1:.*]] = fir.undefined tuple<!fir.heap<!fir.array<10x!fir.char<1>>>, i1>
// CHECK:           %[[INSERT_VALUE_2:.*]] = fir.insert_value %[[UNDEFINED_1]], %[[CONSTANT_2]], [1 : index] : (tuple<!fir.heap<!fir.array<10x!fir.char<1>>>, i1>, i1) -> tuple<!fir.heap<!fir.array<10x!fir.char<1>>>, i1>
// CHECK:           %[[INSERT_VALUE_3:.*]] = fir.insert_value %[[INSERT_VALUE_2]], %[[DECLARE_0]]#0, [0 : index] : (tuple<!fir.heap<!fir.array<10x!fir.char<1>>>, i1>, !fir.heap<!fir.array<10x!fir.char<1>>>) -> tuple<!fir.heap<!fir.array<10x!fir.char<1>>>, i1>
// CHECK:           hlfir.assign %[[DECLARE_0]]#0 to %[[ARG1]] : !fir.heap<!fir.array<10x!fir.char<1>>>, !fir.ref<!fir.array<10x!fir.char<1>>>
// CHECK:           fir.freemem %[[DECLARE_0]]#0 : !fir.heap<!fir.array<10x!fir.char<1>>>
// CHECK:           return
// CHECK:         }
