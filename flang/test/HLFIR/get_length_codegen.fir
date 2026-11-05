// Test hlfir.get_length code generation
// RUN: fir-opt %s --bufferize-hlfir | FileCheck %s

func.func @_QPtest_char_get_length(%arg0: !fir.boxchar<1> {fir.bindc_name = "ch"}) -> index {
  %c3 = arith.constant 3 : index
  %0:2 = fir.unboxchar %arg0 : (!fir.boxchar<1>) -> (!fir.ref<!fir.char<1,?>>, index)
  %1:2 = hlfir.declare %0#0 typeparams %0#1 uniq_name("_QFtest_char_get_lengthEch") : (!fir.ref<!fir.char<1,?>>, index) -> (!fir.boxchar<1>, !fir.ref<!fir.char<1,?>>)
  %2 = fir.alloca i32 {bindc_name = "x", uniq_name = "_QFtest_char_get_lengthEx"}
  %3:2 = hlfir.declare %2 uniq_name("_QFtest_char_get_lengthEx") : (!fir.ref<i32>) -> (!fir.ref<i32>, !fir.ref<i32>)
  %4 = fir.address_of(@_QQclX616263) : !fir.ref<!fir.char<1,3>>
  %5:2 = hlfir.declare %4 typeparams %c3 uniq_name("_QQclX616263") fortran_attrs<parameter> : (!fir.ref<!fir.char<1,3>>, index) -> (!fir.ref<!fir.char<1,3>>, !fir.ref<!fir.char<1,3>>)
  %6 = arith.addi %0#1, %c3 : index
  %7 = hlfir.concat %5#0, %1#0 len %6 : (!fir.ref<!fir.char<1,3>>, !fir.boxchar<1>, index) -> !hlfir.expr<!fir.char<1,?>>
  %8:3 = hlfir.associate %7 typeparams %6 uniq_name(".tmp.assign") : (!hlfir.expr<!fir.char<1,?>>, index) -> (!fir.boxchar<1>, !fir.ref<!fir.char<1,?>>, i1)
  %9 = hlfir.as_expr %8#0 : (!fir.boxchar<1>) -> !hlfir.expr<!fir.char<1,?>>
  %10 = hlfir.get_length %9 : (!hlfir.expr<!fir.char<1,?>>) -> index
  hlfir.destroy %9 : !hlfir.expr<!fir.char<1,?>>
  hlfir.end_associate %8#1, %8#2 : !fir.ref<!fir.char<1,?>>, i1
  return %10 : index
}
fir.global linkonce @_QQclX616263 constant : !fir.char<1,3> {
  %0 = fir.string_lit "abc"(3) : !fir.char<1,3>
  fir.has_value %0 : !fir.char<1,3>
}
// CHECK-LABEL:   func.func @_QPtest_char_get_length(
// CHECK-SAME:      %[[ARG0:[0-9]+|[a-zA-Z$._-][a-zA-Z0-9$._-]*]]: !fir.boxchar<1> {fir.bindc_name = "ch"}) -> index {
// CHECK:           %[[CONSTANT_0:.*]] = arith.constant 3 : index
// CHECK:           %[[UNBOXCHAR_0:.*]]:2 = fir.unboxchar %[[ARG0]] : (!fir.boxchar<1>) -> (!fir.ref<!fir.char<1,?>>, index)
// CHECK:           %[[DECLARE_0:.*]]:2 = hlfir.declare %[[UNBOXCHAR_0]]#0 typeparams %[[UNBOXCHAR_0]]#1 uniq_name("_QFtest_char_get_lengthEch") : (!fir.ref<!fir.char<1,?>>, index) -> (!fir.boxchar<1>, !fir.ref<!fir.char<1,?>>)
// CHECK:           %[[ALLOCA_0:.*]] = fir.alloca i32 <{bindc_name = "x", uniq_name = "_QFtest_char_get_lengthEx"}>
// CHECK:           %[[DECLARE_1:.*]]:2 = hlfir.declare %[[ALLOCA_0]] uniq_name("_QFtest_char_get_lengthEx") : (!fir.ref<i32>) -> (!fir.ref<i32>, !fir.ref<i32>)
// CHECK:           %[[ADDRESS_OF_0:.*]] = fir.address_of(@_QQclX616263) : !fir.ref<!fir.char<1,3>>
// CHECK:           %[[DECLARE_2:.*]]:2 = hlfir.declare %[[ADDRESS_OF_0]] typeparams %[[CONSTANT_0]] uniq_name("_QQclX616263") fortran_attrs<parameter> : (!fir.ref<!fir.char<1,3>>, index) -> (!fir.ref<!fir.char<1,3>>, !fir.ref<!fir.char<1,3>>)
// CHECK:           %[[ADDI_0:.*]] = arith.addi %[[UNBOXCHAR_0]]#1, %[[CONSTANT_0]] : index
// CHECK:           %[[ALLOCMEM_0:.*]] = fir.allocmem !fir.char<1,?>(%[[ADDI_0]] : index) <{bindc_name = ".chrtmp", uniq_name = ""}>
// CHECK:           %[[CONSTANT_1:.*]] = arith.constant 0 : index
// CHECK:           %[[CONSTANT_2:.*]] = arith.constant 1 : index
// CHECK:           %[[ADDI_1:.*]] = arith.addi %[[CONSTANT_1]], %[[CONSTANT_2]] : index
// CHECK:           %[[SUBI_0:.*]] = arith.subi %[[ADDI_1]], %[[CONSTANT_2]] : index
// CHECK:           %[[CONVERT_0:.*]] = fir.convert %[[ALLOCMEM_0]] : (!fir.heap<!fir.char<1,?>>) -> !fir.ref<!fir.array<?x!fir.char<1>>>
// CHECK:           %[[COORDINATE_OF_0:.*]] = fir.coordinate_of %[[CONVERT_0]], %[[SUBI_0]] : (!fir.ref<!fir.array<?x!fir.char<1>>>, index) -> !fir.ref<!fir.char<1>>
// CHECK:           %[[CONVERT_1:.*]] = fir.convert %[[COORDINATE_OF_0]] : (!fir.ref<!fir.char<1>>) -> !fir.ref<!fir.char<1,?>>
// CHECK:           %[[CONSTANT_3:.*]] = arith.constant 1 : i64
// CHECK:           %[[CONVERT_2:.*]] = fir.convert %[[CONSTANT_0]] : (index) -> i64
// CHECK:           %[[MULI_0:.*]] = arith.muli %[[CONSTANT_3]], %[[CONVERT_2]] : i64
// CHECK:           %[[CONVERT_3:.*]] = fir.convert %[[CONVERT_1]] : (!fir.ref<!fir.char<1,?>>) -> !llvm.ptr
// CHECK:           %[[CONVERT_4:.*]] = fir.convert %[[DECLARE_2]]#0 : (!fir.ref<!fir.char<1,3>>) -> !llvm.ptr
// CHECK:           "llvm.intr.memmove"(%[[CONVERT_3]], %[[CONVERT_4]], %[[MULI_0]]) <{isVolatile = false}> : (!llvm.ptr, !llvm.ptr, i64) -> ()
// CHECK:           %[[ADDI_2:.*]] = arith.addi %[[CONSTANT_1]], %[[CONSTANT_0]] : index
// CHECK:           %[[ADDI_3:.*]] = arith.addi %[[ADDI_2]], %[[CONSTANT_2]] : index
// CHECK:           %[[SUBI_1:.*]] = arith.subi %[[ADDI_3]], %[[CONSTANT_2]] : index
// CHECK:           %[[CONVERT_5:.*]] = fir.convert %[[ALLOCMEM_0]] : (!fir.heap<!fir.char<1,?>>) -> !fir.ref<!fir.array<?x!fir.char<1>>>
// CHECK:           %[[COORDINATE_OF_1:.*]] = fir.coordinate_of %[[CONVERT_5]], %[[SUBI_1]] : (!fir.ref<!fir.array<?x!fir.char<1>>>, index) -> !fir.ref<!fir.char<1>>
// CHECK:           %[[CONVERT_6:.*]] = fir.convert %[[COORDINATE_OF_1]] : (!fir.ref<!fir.char<1>>) -> !fir.ref<!fir.char<1,?>>
// CHECK:           %[[CONSTANT_4:.*]] = arith.constant 1 : i64
// CHECK:           %[[CONVERT_7:.*]] = fir.convert %[[UNBOXCHAR_0]]#1 : (index) -> i64
// CHECK:           %[[MULI_1:.*]] = arith.muli %[[CONSTANT_4]], %[[CONVERT_7]] : i64
// CHECK:           %[[CONVERT_8:.*]] = fir.convert %[[CONVERT_6]] : (!fir.ref<!fir.char<1,?>>) -> !llvm.ptr
// CHECK:           %[[CONVERT_9:.*]] = fir.convert %[[DECLARE_0]]#1 : (!fir.ref<!fir.char<1,?>>) -> !llvm.ptr
// CHECK:           "llvm.intr.memmove"(%[[CONVERT_8]], %[[CONVERT_9]], %[[MULI_1]]) <{isVolatile = false}> : (!llvm.ptr, !llvm.ptr, i64) -> ()
// CHECK:           %[[ADDI_4:.*]] = arith.addi %[[ADDI_2]], %[[UNBOXCHAR_0]]#1 : index
// CHECK:           %[[CONVERT_10:.*]] = fir.convert %[[ALLOCMEM_0]] : (!fir.heap<!fir.char<1,?>>) -> !fir.ref<!fir.char<1,?>>
// CHECK:           %[[DECLARE_3:.*]]:2 = hlfir.declare %[[CONVERT_10]] typeparams %[[ADDI_0]] uniq_name("tmp") : (!fir.ref<!fir.char<1,?>>, index) -> (!fir.boxchar<1>, !fir.ref<!fir.char<1,?>>)
// CHECK:           %[[CONSTANT_5:.*]] = arith.constant true
// CHECK:           %[[UNDEFINED_0:.*]] = fir.undefined tuple<!fir.boxchar<1>, i1>
// CHECK:           %[[INSERT_VALUE_0:.*]] = fir.insert_value %[[UNDEFINED_0]], %[[CONSTANT_5]], [1 : index] : (tuple<!fir.boxchar<1>, i1>, i1) -> tuple<!fir.boxchar<1>, i1>
// CHECK:           %[[INSERT_VALUE_1:.*]] = fir.insert_value %[[INSERT_VALUE_0]], %[[DECLARE_3]]#0, [0 : index] : (tuple<!fir.boxchar<1>, i1>, !fir.boxchar<1>) -> tuple<!fir.boxchar<1>, i1>
// CHECK:           %[[ALLOCMEM_1:.*]] = fir.allocmem !fir.char<1,?>(%[[ADDI_0]] : index) <{bindc_name = ".tmp", uniq_name = ""}>
// CHECK:           %[[DECLARE_4:.*]]:2 = hlfir.declare %[[ALLOCMEM_1]] typeparams %[[ADDI_0]] uniq_name(".tmp") : (!fir.heap<!fir.char<1,?>>, index) -> (!fir.boxchar<1>, !fir.heap<!fir.char<1,?>>)
// CHECK:           hlfir.assign %[[DECLARE_3]]#0 to %[[DECLARE_4]]#0 temporary_lhs : !fir.boxchar<1>, !fir.boxchar<1>
// CHECK:           %[[CONSTANT_6:.*]] = arith.constant true
// CHECK:           %[[UNDEFINED_1:.*]] = fir.undefined tuple<!fir.boxchar<1>, i1>
// CHECK:           %[[INSERT_VALUE_2:.*]] = fir.insert_value %[[UNDEFINED_1]], %[[CONSTANT_6]], [1 : index] : (tuple<!fir.boxchar<1>, i1>, i1) -> tuple<!fir.boxchar<1>, i1>
// CHECK:           %[[INSERT_VALUE_3:.*]] = fir.insert_value %[[INSERT_VALUE_2]], %[[DECLARE_4]]#0, [0 : index] : (tuple<!fir.boxchar<1>, i1>, !fir.boxchar<1>) -> tuple<!fir.boxchar<1>, i1>
// CHECK:           %[[BOX_ADDR_0:.*]] = fir.box_addr %[[DECLARE_4]]#0 : (!fir.boxchar<1>) -> !fir.heap<!fir.char<1,?>>
// CHECK:           fir.freemem %[[BOX_ADDR_0]] : !fir.heap<!fir.char<1,?>>
// CHECK:           %[[CONVERT_11:.*]] = fir.convert %[[DECLARE_3]]#1 : (!fir.ref<!fir.char<1,?>>) -> !fir.heap<!fir.char<1,?>>
// CHECK:           fir.freemem %[[CONVERT_11]] : !fir.heap<!fir.char<1,?>>
// CHECK:           return %[[ADDI_0]] : index
// CHECK:         }

// Test get_length taking the length from an expression with i32 length parameter.
func.func @i32_length(%char: !fir.boxchar<1>, %shape : i32, %len : i32) -> index {
  %14 = fir.shape %shape : (i32) -> !fir.shape<1>
  %15 = hlfir.elemental %14 typeparams %len unordered : (!fir.shape<1>, i32) -> !hlfir.expr<?x!fir.char<1,?>> {
  ^bb0(%arg0: index):
    hlfir.yield_element %char : !fir.boxchar<1>
  }
  %18 = hlfir.get_length %15 : (!hlfir.expr<?x!fir.char<1,?>>) -> index
  hlfir.destroy %15 : !hlfir.expr<?x!fir.char<1,?>>
  return %18 : index
}
// CHECK-LABEL:   func.func @i32_length(
// CHECK-SAME:                          %[[VAL_0:.*]]: !fir.boxchar<1>,
// CHECK-SAME:                          %[[VAL_1:.*]]: i32,
// CHECK-SAME:                          %[[VAL_2:.*]]: i32) -> index {
// CHECK:           %[[VAL_14:.*]] = fir.convert %[[VAL_2]] : (i32) -> index
// CHECK:           return %[[VAL_14]] : index
// CHECK:         }
