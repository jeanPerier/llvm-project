// Test hlfir.concat operation lowering to operations operating on memory.

// RUN: fir-opt %s -bufferize-hlfir | FileCheck %s


func.func @concat(%arg0: !fir.boxchar<1>, %arg1: !fir.boxchar<1>, %arg2: !fir.boxchar<1>) {
  %0:2 = fir.unboxchar %arg0 : (!fir.boxchar<1>) -> (!fir.ref<!fir.char<1,?>>, index)
  %1:2 = hlfir.declare %0#0 typeparams %0#1 uniq_name("c1") : (!fir.ref<!fir.char<1,?>>, index) -> (!fir.boxchar<1>, !fir.ref<!fir.char<1,?>>)
  %2:2 = fir.unboxchar %arg1 : (!fir.boxchar<1>) -> (!fir.ref<!fir.char<1,?>>, index)
  %3:2 = hlfir.declare %2#0 typeparams %2#1 uniq_name("c2") : (!fir.ref<!fir.char<1,?>>, index) -> (!fir.boxchar<1>, !fir.ref<!fir.char<1,?>>)
  %4:2 = fir.unboxchar %arg2 : (!fir.boxchar<1>) -> (!fir.ref<!fir.char<1,?>>, index)
  %5:2 = hlfir.declare %4#0 typeparams %4#1 uniq_name("c3") : (!fir.ref<!fir.char<1,?>>, index) -> (!fir.boxchar<1>, !fir.ref<!fir.char<1,?>>)
  %6 = arith.addi %2#1, %4#1 : index
  %7 = hlfir.concat %3#0, %5#0 len %6 : (!fir.boxchar<1>, !fir.boxchar<1>, index) -> !hlfir.expr<!fir.char<1,?>>
  hlfir.assign %7 to %1#0 : !hlfir.expr<!fir.char<1,?>>, !fir.boxchar<1>
  hlfir.destroy %7 : !hlfir.expr<!fir.char<1,?>>
  return
}
// CHECK-LABEL:   func.func @concat(
// CHECK-SAME:                      %[[ARG0:[0-9]+|[a-zA-Z$._-][a-zA-Z0-9$._-]*]]: !fir.boxchar<1>,
// CHECK-SAME:                      %[[ARG1:[0-9]+|[a-zA-Z$._-][a-zA-Z0-9$._-]*]]: !fir.boxchar<1>,
// CHECK-SAME:                      %[[ARG2:[0-9]+|[a-zA-Z$._-][a-zA-Z0-9$._-]*]]: !fir.boxchar<1>) {
// CHECK:           %[[UNBOXCHAR_0:.*]]:2 = fir.unboxchar %[[ARG0]] : (!fir.boxchar<1>) -> (!fir.ref<!fir.char<1,?>>, index)
// CHECK:           %[[DECLARE_0:.*]]:2 = hlfir.declare %[[UNBOXCHAR_0]]#0 typeparams %[[UNBOXCHAR_0]]#1 uniq_name("c1") : (!fir.ref<!fir.char<1,?>>, index) -> (!fir.boxchar<1>, !fir.ref<!fir.char<1,?>>)
// CHECK:           %[[UNBOXCHAR_1:.*]]:2 = fir.unboxchar %[[ARG1]] : (!fir.boxchar<1>) -> (!fir.ref<!fir.char<1,?>>, index)
// CHECK:           %[[DECLARE_1:.*]]:2 = hlfir.declare %[[UNBOXCHAR_1]]#0 typeparams %[[UNBOXCHAR_1]]#1 uniq_name("c2") : (!fir.ref<!fir.char<1,?>>, index) -> (!fir.boxchar<1>, !fir.ref<!fir.char<1,?>>)
// CHECK:           %[[UNBOXCHAR_2:.*]]:2 = fir.unboxchar %[[ARG2]] : (!fir.boxchar<1>) -> (!fir.ref<!fir.char<1,?>>, index)
// CHECK:           %[[DECLARE_2:.*]]:2 = hlfir.declare %[[UNBOXCHAR_2]]#0 typeparams %[[UNBOXCHAR_2]]#1 uniq_name("c3") : (!fir.ref<!fir.char<1,?>>, index) -> (!fir.boxchar<1>, !fir.ref<!fir.char<1,?>>)
// CHECK:           %[[ADDI_0:.*]] = arith.addi %[[UNBOXCHAR_1]]#1, %[[UNBOXCHAR_2]]#1 : index
// CHECK:           %[[ALLOCMEM_0:.*]] = fir.allocmem !fir.char<1,?>(%[[ADDI_0]] : index) <{bindc_name = ".chrtmp", uniq_name = ""}>
// CHECK:           %[[CONSTANT_0:.*]] = arith.constant 0 : index
// CHECK:           %[[CONSTANT_1:.*]] = arith.constant 1 : index
// CHECK:           %[[ADDI_1:.*]] = arith.addi %[[CONSTANT_0]], %[[CONSTANT_1]] : index
// CHECK:           %[[SUBI_0:.*]] = arith.subi %[[ADDI_1]], %[[CONSTANT_1]] : index
// CHECK:           %[[CONVERT_0:.*]] = fir.convert %[[ALLOCMEM_0]] : (!fir.heap<!fir.char<1,?>>) -> !fir.ref<!fir.array<?x!fir.char<1>>>
// CHECK:           %[[COORDINATE_OF_0:.*]] = fir.coordinate_of %[[CONVERT_0]], %[[SUBI_0]] : (!fir.ref<!fir.array<?x!fir.char<1>>>, index) -> !fir.ref<!fir.char<1>>
// CHECK:           %[[CONVERT_1:.*]] = fir.convert %[[COORDINATE_OF_0]] : (!fir.ref<!fir.char<1>>) -> !fir.ref<!fir.char<1,?>>
// CHECK:           %[[CONSTANT_2:.*]] = arith.constant 1 : i64
// CHECK:           %[[CONVERT_2:.*]] = fir.convert %[[UNBOXCHAR_1]]#1 : (index) -> i64
// CHECK:           %[[MULI_0:.*]] = arith.muli %[[CONSTANT_2]], %[[CONVERT_2]] : i64
// CHECK:           %[[CONVERT_3:.*]] = fir.convert %[[CONVERT_1]] : (!fir.ref<!fir.char<1,?>>) -> !llvm.ptr
// CHECK:           %[[CONVERT_4:.*]] = fir.convert %[[DECLARE_1]]#1 : (!fir.ref<!fir.char<1,?>>) -> !llvm.ptr
// CHECK:           "llvm.intr.memmove"(%[[CONVERT_3]], %[[CONVERT_4]], %[[MULI_0]]) <{isVolatile = false}> : (!llvm.ptr, !llvm.ptr, i64) -> ()
// CHECK:           %[[ADDI_2:.*]] = arith.addi %[[CONSTANT_0]], %[[UNBOXCHAR_1]]#1 : index
// CHECK:           %[[ADDI_3:.*]] = arith.addi %[[ADDI_2]], %[[CONSTANT_1]] : index
// CHECK:           %[[SUBI_1:.*]] = arith.subi %[[ADDI_3]], %[[CONSTANT_1]] : index
// CHECK:           %[[CONVERT_5:.*]] = fir.convert %[[ALLOCMEM_0]] : (!fir.heap<!fir.char<1,?>>) -> !fir.ref<!fir.array<?x!fir.char<1>>>
// CHECK:           %[[COORDINATE_OF_1:.*]] = fir.coordinate_of %[[CONVERT_5]], %[[SUBI_1]] : (!fir.ref<!fir.array<?x!fir.char<1>>>, index) -> !fir.ref<!fir.char<1>>
// CHECK:           %[[CONVERT_6:.*]] = fir.convert %[[COORDINATE_OF_1]] : (!fir.ref<!fir.char<1>>) -> !fir.ref<!fir.char<1,?>>
// CHECK:           %[[CONSTANT_3:.*]] = arith.constant 1 : i64
// CHECK:           %[[CONVERT_7:.*]] = fir.convert %[[UNBOXCHAR_2]]#1 : (index) -> i64
// CHECK:           %[[MULI_1:.*]] = arith.muli %[[CONSTANT_3]], %[[CONVERT_7]] : i64
// CHECK:           %[[CONVERT_8:.*]] = fir.convert %[[CONVERT_6]] : (!fir.ref<!fir.char<1,?>>) -> !llvm.ptr
// CHECK:           %[[CONVERT_9:.*]] = fir.convert %[[DECLARE_2]]#1 : (!fir.ref<!fir.char<1,?>>) -> !llvm.ptr
// CHECK:           "llvm.intr.memmove"(%[[CONVERT_8]], %[[CONVERT_9]], %[[MULI_1]]) <{isVolatile = false}> : (!llvm.ptr, !llvm.ptr, i64) -> ()
// CHECK:           %[[ADDI_4:.*]] = arith.addi %[[ADDI_2]], %[[UNBOXCHAR_2]]#1 : index
// CHECK:           %[[CONVERT_10:.*]] = fir.convert %[[ALLOCMEM_0]] : (!fir.heap<!fir.char<1,?>>) -> !fir.ref<!fir.char<1,?>>
// CHECK:           %[[DECLARE_3:.*]]:2 = hlfir.declare %[[CONVERT_10]] typeparams %[[ADDI_0]] uniq_name("tmp") : (!fir.ref<!fir.char<1,?>>, index) -> (!fir.boxchar<1>, !fir.ref<!fir.char<1,?>>)
// CHECK:           %[[CONSTANT_4:.*]] = arith.constant true
// CHECK:           %[[UNDEFINED_0:.*]] = fir.undefined tuple<!fir.boxchar<1>, i1>
// CHECK:           %[[INSERT_VALUE_0:.*]] = fir.insert_value %[[UNDEFINED_0]], %[[CONSTANT_4]], [1 : index] : (tuple<!fir.boxchar<1>, i1>, i1) -> tuple<!fir.boxchar<1>, i1>
// CHECK:           %[[INSERT_VALUE_1:.*]] = fir.insert_value %[[INSERT_VALUE_0]], %[[DECLARE_3]]#0, [0 : index] : (tuple<!fir.boxchar<1>, i1>, !fir.boxchar<1>) -> tuple<!fir.boxchar<1>, i1>
// CHECK:           hlfir.assign %[[DECLARE_3]]#0 to %[[DECLARE_0]]#0 : !fir.boxchar<1>, !fir.boxchar<1>
// CHECK:           %[[BOX_ADDR_0:.*]] = fir.box_addr %[[DECLARE_3]]#0 : (!fir.boxchar<1>) -> !fir.heap<!fir.char<1,?>>
// CHECK:           fir.freemem %[[BOX_ADDR_0]] : !fir.heap<!fir.char<1,?>>
// CHECK:           return
// CHECK:         }


func.func @concat_chained(%arg0: !fir.boxchar<1>, %arg1: !fir.boxchar<1>, %arg2: !fir.boxchar<1>, %arg3: !fir.boxchar<1>) {
  %0:2 = fir.unboxchar %arg0 : (!fir.boxchar<1>) -> (!fir.ref<!fir.char<1,?>>, index)
  %1:2 = hlfir.declare %0#0 typeparams %0#1 uniq_name("c1") : (!fir.ref<!fir.char<1,?>>, index) -> (!fir.boxchar<1>, !fir.ref<!fir.char<1,?>>)
  %2:2 = fir.unboxchar %arg1 : (!fir.boxchar<1>) -> (!fir.ref<!fir.char<1,?>>, index)
  %3:2 = hlfir.declare %2#0 typeparams %2#1 uniq_name("c2") : (!fir.ref<!fir.char<1,?>>, index) -> (!fir.boxchar<1>, !fir.ref<!fir.char<1,?>>)
  %4:2 = fir.unboxchar %arg2 : (!fir.boxchar<1>) -> (!fir.ref<!fir.char<1,?>>, index)
  %5:2 = hlfir.declare %4#0 typeparams %4#1 uniq_name("c3") : (!fir.ref<!fir.char<1,?>>, index) -> (!fir.boxchar<1>, !fir.ref<!fir.char<1,?>>)
  %6:2 = fir.unboxchar %arg3 : (!fir.boxchar<1>) -> (!fir.ref<!fir.char<1,?>>, index)
  %7:2 = hlfir.declare %6#0 typeparams %6#1 uniq_name("c4") : (!fir.ref<!fir.char<1,?>>, index) -> (!fir.boxchar<1>, !fir.ref<!fir.char<1,?>>)
  %8 = arith.addi %2#1, %4#1 : index
  %10 = arith.addi %8, %6#1 : index
  %11 = hlfir.concat %3#0, %5#0, %7#0 len %10 : (!fir.boxchar<1>, !fir.boxchar<1>, !fir.boxchar<1>, index) -> !hlfir.expr<!fir.char<1,?>>
  hlfir.assign %11 to %1#0 : !hlfir.expr<!fir.char<1,?>>, !fir.boxchar<1>
  hlfir.destroy %11 : !hlfir.expr<!fir.char<1,?>>
  return
}

// CHECK-LABEL:   func.func @concat_chained(
// CHECK-SAME:      %[[ARG0:[0-9]+|[a-zA-Z$._-][a-zA-Z0-9$._-]*]]: !fir.boxchar<1>,
// CHECK-SAME:      %[[ARG1:[0-9]+|[a-zA-Z$._-][a-zA-Z0-9$._-]*]]: !fir.boxchar<1>,
// CHECK-SAME:      %[[ARG2:[0-9]+|[a-zA-Z$._-][a-zA-Z0-9$._-]*]]: !fir.boxchar<1>,
// CHECK-SAME:      %[[ARG3:[0-9]+|[a-zA-Z$._-][a-zA-Z0-9$._-]*]]: !fir.boxchar<1>) {
// CHECK:           %[[UNBOXCHAR_0:.*]]:2 = fir.unboxchar %[[ARG0]] : (!fir.boxchar<1>) -> (!fir.ref<!fir.char<1,?>>, index)
// CHECK:           %[[DECLARE_0:.*]]:2 = hlfir.declare %[[UNBOXCHAR_0]]#0 typeparams %[[UNBOXCHAR_0]]#1 uniq_name("c1") : (!fir.ref<!fir.char<1,?>>, index) -> (!fir.boxchar<1>, !fir.ref<!fir.char<1,?>>)
// CHECK:           %[[UNBOXCHAR_1:.*]]:2 = fir.unboxchar %[[ARG1]] : (!fir.boxchar<1>) -> (!fir.ref<!fir.char<1,?>>, index)
// CHECK:           %[[DECLARE_1:.*]]:2 = hlfir.declare %[[UNBOXCHAR_1]]#0 typeparams %[[UNBOXCHAR_1]]#1 uniq_name("c2") : (!fir.ref<!fir.char<1,?>>, index) -> (!fir.boxchar<1>, !fir.ref<!fir.char<1,?>>)
// CHECK:           %[[UNBOXCHAR_2:.*]]:2 = fir.unboxchar %[[ARG2]] : (!fir.boxchar<1>) -> (!fir.ref<!fir.char<1,?>>, index)
// CHECK:           %[[DECLARE_2:.*]]:2 = hlfir.declare %[[UNBOXCHAR_2]]#0 typeparams %[[UNBOXCHAR_2]]#1 uniq_name("c3") : (!fir.ref<!fir.char<1,?>>, index) -> (!fir.boxchar<1>, !fir.ref<!fir.char<1,?>>)
// CHECK:           %[[UNBOXCHAR_3:.*]]:2 = fir.unboxchar %[[ARG3]] : (!fir.boxchar<1>) -> (!fir.ref<!fir.char<1,?>>, index)
// CHECK:           %[[DECLARE_3:.*]]:2 = hlfir.declare %[[UNBOXCHAR_3]]#0 typeparams %[[UNBOXCHAR_3]]#1 uniq_name("c4") : (!fir.ref<!fir.char<1,?>>, index) -> (!fir.boxchar<1>, !fir.ref<!fir.char<1,?>>)
// CHECK:           %[[ADDI_0:.*]] = arith.addi %[[UNBOXCHAR_1]]#1, %[[UNBOXCHAR_2]]#1 : index
// CHECK:           %[[ADDI_1:.*]] = arith.addi %[[ADDI_0]], %[[UNBOXCHAR_3]]#1 : index
// CHECK:           %[[ALLOCMEM_0:.*]] = fir.allocmem !fir.char<1,?>(%[[ADDI_1]] : index) <{bindc_name = ".chrtmp", uniq_name = ""}>
// CHECK:           %[[CONSTANT_0:.*]] = arith.constant 0 : index
// CHECK:           %[[CONSTANT_1:.*]] = arith.constant 1 : index
// CHECK:           %[[ADDI_2:.*]] = arith.addi %[[CONSTANT_0]], %[[CONSTANT_1]] : index
// CHECK:           %[[SUBI_0:.*]] = arith.subi %[[ADDI_2]], %[[CONSTANT_1]] : index
// CHECK:           %[[CONVERT_0:.*]] = fir.convert %[[ALLOCMEM_0]] : (!fir.heap<!fir.char<1,?>>) -> !fir.ref<!fir.array<?x!fir.char<1>>>
// CHECK:           %[[COORDINATE_OF_0:.*]] = fir.coordinate_of %[[CONVERT_0]], %[[SUBI_0]] : (!fir.ref<!fir.array<?x!fir.char<1>>>, index) -> !fir.ref<!fir.char<1>>
// CHECK:           %[[CONVERT_1:.*]] = fir.convert %[[COORDINATE_OF_0]] : (!fir.ref<!fir.char<1>>) -> !fir.ref<!fir.char<1,?>>
// CHECK:           %[[CONSTANT_2:.*]] = arith.constant 1 : i64
// CHECK:           %[[CONVERT_2:.*]] = fir.convert %[[UNBOXCHAR_1]]#1 : (index) -> i64
// CHECK:           %[[MULI_0:.*]] = arith.muli %[[CONSTANT_2]], %[[CONVERT_2]] : i64
// CHECK:           %[[CONVERT_3:.*]] = fir.convert %[[CONVERT_1]] : (!fir.ref<!fir.char<1,?>>) -> !llvm.ptr
// CHECK:           %[[CONVERT_4:.*]] = fir.convert %[[DECLARE_1]]#1 : (!fir.ref<!fir.char<1,?>>) -> !llvm.ptr
// CHECK:           "llvm.intr.memmove"(%[[CONVERT_3]], %[[CONVERT_4]], %[[MULI_0]]) <{isVolatile = false}> : (!llvm.ptr, !llvm.ptr, i64) -> ()
// CHECK:           %[[ADDI_3:.*]] = arith.addi %[[CONSTANT_0]], %[[UNBOXCHAR_1]]#1 : index
// CHECK:           %[[ADDI_4:.*]] = arith.addi %[[ADDI_3]], %[[CONSTANT_1]] : index
// CHECK:           %[[SUBI_1:.*]] = arith.subi %[[ADDI_4]], %[[CONSTANT_1]] : index
// CHECK:           %[[CONVERT_5:.*]] = fir.convert %[[ALLOCMEM_0]] : (!fir.heap<!fir.char<1,?>>) -> !fir.ref<!fir.array<?x!fir.char<1>>>
// CHECK:           %[[COORDINATE_OF_1:.*]] = fir.coordinate_of %[[CONVERT_5]], %[[SUBI_1]] : (!fir.ref<!fir.array<?x!fir.char<1>>>, index) -> !fir.ref<!fir.char<1>>
// CHECK:           %[[CONVERT_6:.*]] = fir.convert %[[COORDINATE_OF_1]] : (!fir.ref<!fir.char<1>>) -> !fir.ref<!fir.char<1,?>>
// CHECK:           %[[CONSTANT_3:.*]] = arith.constant 1 : i64
// CHECK:           %[[CONVERT_7:.*]] = fir.convert %[[UNBOXCHAR_2]]#1 : (index) -> i64
// CHECK:           %[[MULI_1:.*]] = arith.muli %[[CONSTANT_3]], %[[CONVERT_7]] : i64
// CHECK:           %[[CONVERT_8:.*]] = fir.convert %[[CONVERT_6]] : (!fir.ref<!fir.char<1,?>>) -> !llvm.ptr
// CHECK:           %[[CONVERT_9:.*]] = fir.convert %[[DECLARE_2]]#1 : (!fir.ref<!fir.char<1,?>>) -> !llvm.ptr
// CHECK:           "llvm.intr.memmove"(%[[CONVERT_8]], %[[CONVERT_9]], %[[MULI_1]]) <{isVolatile = false}> : (!llvm.ptr, !llvm.ptr, i64) -> ()
// CHECK:           %[[ADDI_5:.*]] = arith.addi %[[ADDI_3]], %[[UNBOXCHAR_2]]#1 : index
// CHECK:           %[[ADDI_6:.*]] = arith.addi %[[ADDI_5]], %[[CONSTANT_1]] : index
// CHECK:           %[[SUBI_2:.*]] = arith.subi %[[ADDI_6]], %[[CONSTANT_1]] : index
// CHECK:           %[[CONVERT_10:.*]] = fir.convert %[[ALLOCMEM_0]] : (!fir.heap<!fir.char<1,?>>) -> !fir.ref<!fir.array<?x!fir.char<1>>>
// CHECK:           %[[COORDINATE_OF_2:.*]] = fir.coordinate_of %[[CONVERT_10]], %[[SUBI_2]] : (!fir.ref<!fir.array<?x!fir.char<1>>>, index) -> !fir.ref<!fir.char<1>>
// CHECK:           %[[CONVERT_11:.*]] = fir.convert %[[COORDINATE_OF_2]] : (!fir.ref<!fir.char<1>>) -> !fir.ref<!fir.char<1,?>>
// CHECK:           %[[CONSTANT_4:.*]] = arith.constant 1 : i64
// CHECK:           %[[CONVERT_12:.*]] = fir.convert %[[UNBOXCHAR_3]]#1 : (index) -> i64
// CHECK:           %[[MULI_2:.*]] = arith.muli %[[CONSTANT_4]], %[[CONVERT_12]] : i64
// CHECK:           %[[CONVERT_13:.*]] = fir.convert %[[CONVERT_11]] : (!fir.ref<!fir.char<1,?>>) -> !llvm.ptr
// CHECK:           %[[CONVERT_14:.*]] = fir.convert %[[DECLARE_3]]#1 : (!fir.ref<!fir.char<1,?>>) -> !llvm.ptr
// CHECK:           "llvm.intr.memmove"(%[[CONVERT_13]], %[[CONVERT_14]], %[[MULI_2]]) <{isVolatile = false}> : (!llvm.ptr, !llvm.ptr, i64) -> ()
// CHECK:           %[[ADDI_7:.*]] = arith.addi %[[ADDI_5]], %[[UNBOXCHAR_3]]#1 : index
// CHECK:           %[[CONVERT_15:.*]] = fir.convert %[[ALLOCMEM_0]] : (!fir.heap<!fir.char<1,?>>) -> !fir.ref<!fir.char<1,?>>
// CHECK:           %[[DECLARE_4:.*]]:2 = hlfir.declare %[[CONVERT_15]] typeparams %[[ADDI_1]] uniq_name("tmp") : (!fir.ref<!fir.char<1,?>>, index) -> (!fir.boxchar<1>, !fir.ref<!fir.char<1,?>>)
// CHECK:           %[[CONSTANT_5:.*]] = arith.constant true
// CHECK:           %[[UNDEFINED_0:.*]] = fir.undefined tuple<!fir.boxchar<1>, i1>
// CHECK:           %[[INSERT_VALUE_0:.*]] = fir.insert_value %[[UNDEFINED_0]], %[[CONSTANT_5]], [1 : index] : (tuple<!fir.boxchar<1>, i1>, i1) -> tuple<!fir.boxchar<1>, i1>
// CHECK:           %[[INSERT_VALUE_1:.*]] = fir.insert_value %[[INSERT_VALUE_0]], %[[DECLARE_4]]#0, [0 : index] : (tuple<!fir.boxchar<1>, i1>, !fir.boxchar<1>) -> tuple<!fir.boxchar<1>, i1>
// CHECK:           hlfir.assign %[[DECLARE_4]]#0 to %[[DECLARE_0]]#0 : !fir.boxchar<1>, !fir.boxchar<1>
// CHECK:           %[[BOX_ADDR_0:.*]] = fir.box_addr %[[DECLARE_4]]#0 : (!fir.boxchar<1>) -> !fir.heap<!fir.char<1,?>>
// CHECK:           fir.freemem %[[BOX_ADDR_0]] : !fir.heap<!fir.char<1,?>>
// CHECK:           return
// CHECK:         }
