// Test hlfir.concat operation lowering to operations operating on memory.

// RUN: fir-opt %s -bufferize-hlfir | FileCheck %s

func.func @_QPmax1(%arg0: !fir.boxchar<1> {fir.bindc_name = "c1"}, %arg1: !fir.boxchar<1> {fir.bindc_name = "c2"}, %arg2: !fir.boxchar<1> {fir.bindc_name = "c3"}) {
  %0:2 = fir.unboxchar %arg0 : (!fir.boxchar<1>) -> (!fir.ref<!fir.char<1,?>>, index)
  %1:2 = hlfir.declare %0#0 typeparams %0#1 uniq_name("_QFmax1Ec1") : (!fir.ref<!fir.char<1,?>>, index) -> (!fir.boxchar<1>, !fir.ref<!fir.char<1,?>>)
  %2:2 = fir.unboxchar %arg1 : (!fir.boxchar<1>) -> (!fir.ref<!fir.char<1,?>>, index)
  %3:2 = hlfir.declare %2#0 typeparams %2#1 uniq_name("_QFmax1Ec2") : (!fir.ref<!fir.char<1,?>>, index) -> (!fir.boxchar<1>, !fir.ref<!fir.char<1,?>>)
  %4:2 = fir.unboxchar %arg2 : (!fir.boxchar<1>) -> (!fir.ref<!fir.char<1,?>>, index)
  %5:2 = hlfir.declare %4#0 typeparams %4#1 uniq_name("_QFmax1Ec3") : (!fir.ref<!fir.char<1,?>>, index) -> (!fir.boxchar<1>, !fir.ref<!fir.char<1,?>>)
  %6 = hlfir.char_extremum max, %3#0, %5#0 : (!fir.boxchar<1>, !fir.boxchar<1>) -> !hlfir.expr<!fir.char<1,?>>
  hlfir.assign %6 to %1#0 : !hlfir.expr<!fir.char<1,?>>, !fir.boxchar<1>
  hlfir.destroy %6 : !hlfir.expr<!fir.char<1,?>>
  return
}

// CHECK-LABEL:   func.func @_QPmax1(
// CHECK-SAME:      %[[ARG0:[0-9]+|[a-zA-Z$._-][a-zA-Z0-9$._-]*]]: !fir.boxchar<1> {fir.bindc_name = "c1"},
// CHECK-SAME:      %[[ARG1:[0-9]+|[a-zA-Z$._-][a-zA-Z0-9$._-]*]]: !fir.boxchar<1> {fir.bindc_name = "c2"},
// CHECK-SAME:      %[[ARG2:[0-9]+|[a-zA-Z$._-][a-zA-Z0-9$._-]*]]: !fir.boxchar<1> {fir.bindc_name = "c3"}) {
// CHECK:           %[[UNBOXCHAR_0:.*]]:2 = fir.unboxchar %[[ARG0]] : (!fir.boxchar<1>) -> (!fir.ref<!fir.char<1,?>>, index)
// CHECK:           %[[DECLARE_0:.*]]:2 = hlfir.declare %[[UNBOXCHAR_0]]#0 typeparams %[[UNBOXCHAR_0]]#1 uniq_name("_QFmax1Ec1") : (!fir.ref<!fir.char<1,?>>, index) -> (!fir.boxchar<1>, !fir.ref<!fir.char<1,?>>)
// CHECK:           %[[UNBOXCHAR_1:.*]]:2 = fir.unboxchar %[[ARG1]] : (!fir.boxchar<1>) -> (!fir.ref<!fir.char<1,?>>, index)
// CHECK:           %[[DECLARE_1:.*]]:2 = hlfir.declare %[[UNBOXCHAR_1]]#0 typeparams %[[UNBOXCHAR_1]]#1 uniq_name("_QFmax1Ec2") : (!fir.ref<!fir.char<1,?>>, index) -> (!fir.boxchar<1>, !fir.ref<!fir.char<1,?>>)
// CHECK:           %[[UNBOXCHAR_2:.*]]:2 = fir.unboxchar %[[ARG2]] : (!fir.boxchar<1>) -> (!fir.ref<!fir.char<1,?>>, index)
// CHECK:           %[[DECLARE_2:.*]]:2 = hlfir.declare %[[UNBOXCHAR_2]]#0 typeparams %[[UNBOXCHAR_2]]#1 uniq_name("_QFmax1Ec3") : (!fir.ref<!fir.char<1,?>>, index) -> (!fir.boxchar<1>, !fir.ref<!fir.char<1,?>>)
// CHECK:           %[[CMPI_0:.*]] = arith.cmpi uge, %[[UNBOXCHAR_1]]#1, %[[UNBOXCHAR_2]]#1 : index
// CHECK:           %[[SELECT_0:.*]] = arith.select %[[CMPI_0]], %[[UNBOXCHAR_1]]#1, %[[UNBOXCHAR_2]]#1 : index
// CHECK:           %[[CONVERT_0:.*]] = fir.convert %[[DECLARE_2]]#1 : (!fir.ref<!fir.char<1,?>>) -> !fir.ref<i8>
// CHECK:           %[[CONVERT_1:.*]] = fir.convert %[[DECLARE_1]]#1 : (!fir.ref<!fir.char<1,?>>) -> !fir.ref<i8>
// CHECK:           %[[CONVERT_2:.*]] = fir.convert %[[UNBOXCHAR_2]]#1 : (index) -> i64
// CHECK:           %[[CONVERT_3:.*]] = fir.convert %[[UNBOXCHAR_1]]#1 : (index) -> i64
// CHECK:           %[[CALL_0:.*]] = fir.call @_FortranACharacterCompareScalar1(%[[CONVERT_0]], %[[CONVERT_1]], %[[CONVERT_2]], %[[CONVERT_3]]) : (!fir.ref<i8>, !fir.ref<i8>, i64, i64) -> i32
// CHECK:           %[[CONSTANT_0:.*]] = arith.constant 0 : i32
// CHECK:           %[[CMPI_1:.*]] = arith.cmpi sgt, %[[CALL_0]], %[[CONSTANT_0]] : i32
// CHECK:           %[[SELECT_1:.*]] = arith.select %[[CMPI_1]], %[[DECLARE_2]]#1, %[[DECLARE_1]]#1 : !fir.ref<!fir.char<1,?>>
// CHECK:           %[[SELECT_2:.*]] = arith.select %[[CMPI_1]], %[[UNBOXCHAR_2]]#1, %[[UNBOXCHAR_1]]#1 : index
// CHECK:           %[[ALLOCMEM_0:.*]] = fir.allocmem !fir.char<1,?>(%[[SELECT_0]] : index) <{bindc_name = ".chrtmp", uniq_name = ""}>
// CHECK:           %[[CMPI_2:.*]] = arith.cmpi slt, %[[SELECT_0]], %[[SELECT_2]] : index
// CHECK:           %[[SELECT_3:.*]] = arith.select %[[CMPI_2]], %[[SELECT_0]], %[[SELECT_2]] : index
// CHECK:           %[[CONSTANT_1:.*]] = arith.constant 1 : i64
// CHECK:           %[[CONVERT_4:.*]] = fir.convert %[[SELECT_3]] : (index) -> i64
// CHECK:           %[[MULI_0:.*]] = arith.muli %[[CONSTANT_1]], %[[CONVERT_4]] : i64
// CHECK:           %[[CONVERT_5:.*]] = fir.convert %[[ALLOCMEM_0]] : (!fir.heap<!fir.char<1,?>>) -> !llvm.ptr
// CHECK:           %[[CONVERT_6:.*]] = fir.convert %[[SELECT_1]] : (!fir.ref<!fir.char<1,?>>) -> !llvm.ptr
// CHECK:           "llvm.intr.memmove"(%[[CONVERT_5]], %[[CONVERT_6]], %[[MULI_0]]) <{isVolatile = false}> : (!llvm.ptr, !llvm.ptr, i64) -> ()
// CHECK:           %[[CONSTANT_2:.*]] = arith.constant 1 : index
// CHECK:           %[[SUBI_0:.*]] = arith.subi %[[SELECT_0]], %[[CONSTANT_2]] : index
// CHECK:           %[[CONSTANT_3:.*]] = arith.constant 32 : i8
// CHECK:           %[[UNDEFINED_0:.*]] = fir.undefined !fir.char<1>
// CHECK:           %[[INSERT_VALUE_0:.*]] = fir.insert_value %[[UNDEFINED_0]], %[[CONSTANT_3]], [0 : index] : (!fir.char<1>, i8) -> !fir.char<1>
// CHECK:           %[[CONSTANT_4:.*]] = arith.constant 1 : index
// CHECK:           fir.do_loop %[[VAL_0:.*]] = %[[SELECT_3]] to %[[SUBI_0]] step %[[CONSTANT_4]] {
// CHECK:             %[[CONVERT_7:.*]] = fir.convert %[[ALLOCMEM_0]] : (!fir.heap<!fir.char<1,?>>) -> !fir.ref<!fir.array<?x!fir.char<1>>>
// CHECK:             %[[COORDINATE_OF_0:.*]] = fir.coordinate_of %[[CONVERT_7]], %[[VAL_0]] : (!fir.ref<!fir.array<?x!fir.char<1>>>, index) -> !fir.ref<!fir.char<1>>
// CHECK:             fir.store %[[INSERT_VALUE_0]] to %[[COORDINATE_OF_0]] : !fir.ref<!fir.char<1>>
// CHECK:           }
// CHECK:           %[[CONVERT_8:.*]] = fir.convert %[[ALLOCMEM_0]] : (!fir.heap<!fir.char<1,?>>) -> !fir.ref<!fir.char<1,?>>
// CHECK:           %[[DECLARE_3:.*]]:2 = hlfir.declare %[[CONVERT_8]] typeparams %[[SELECT_0]] uniq_name(".tmp.char_extremum") : (!fir.ref<!fir.char<1,?>>, index) -> (!fir.boxchar<1>, !fir.ref<!fir.char<1,?>>)
// CHECK:           %[[CONSTANT_5:.*]] = arith.constant true
// CHECK:           %[[UNDEFINED_1:.*]] = fir.undefined tuple<!fir.boxchar<1>, i1>
// CHECK:           %[[INSERT_VALUE_1:.*]] = fir.insert_value %[[UNDEFINED_1]], %[[CONSTANT_5]], [1 : index] : (tuple<!fir.boxchar<1>, i1>, i1) -> tuple<!fir.boxchar<1>, i1>
// CHECK:           %[[INSERT_VALUE_2:.*]] = fir.insert_value %[[INSERT_VALUE_1]], %[[DECLARE_3]]#0, [0 : index] : (tuple<!fir.boxchar<1>, i1>, !fir.boxchar<1>) -> tuple<!fir.boxchar<1>, i1>
// CHECK:           hlfir.assign %[[DECLARE_3]]#0 to %[[DECLARE_0]]#0 : !fir.boxchar<1>, !fir.boxchar<1>
// CHECK:           %[[BOX_ADDR_0:.*]] = fir.box_addr %[[DECLARE_3]]#0 : (!fir.boxchar<1>) -> !fir.heap<!fir.char<1,?>>
// CHECK:           fir.freemem %[[BOX_ADDR_0]] : !fir.heap<!fir.char<1,?>>
// CHECK:           return
// CHECK:         }

func.func @_QPmin1(%arg0: !fir.boxchar<1> {fir.bindc_name = "c1"}, %arg1: !fir.boxchar<1> {fir.bindc_name = "c2"}, %arg2: !fir.boxchar<1> {fir.bindc_name = "c3"}) {
  %0:2 = fir.unboxchar %arg0 : (!fir.boxchar<1>) -> (!fir.ref<!fir.char<1,?>>, index)
  %1:2 = hlfir.declare %0#0 typeparams %0#1 uniq_name("_QFmin1Ec1") : (!fir.ref<!fir.char<1,?>>, index) -> (!fir.boxchar<1>, !fir.ref<!fir.char<1,?>>)
  %2:2 = fir.unboxchar %arg1 : (!fir.boxchar<1>) -> (!fir.ref<!fir.char<1,?>>, index)
  %3:2 = hlfir.declare %2#0 typeparams %2#1 uniq_name("_QFmin1Ec2") : (!fir.ref<!fir.char<1,?>>, index) -> (!fir.boxchar<1>, !fir.ref<!fir.char<1,?>>)
  %4:2 = fir.unboxchar %arg2 : (!fir.boxchar<1>) -> (!fir.ref<!fir.char<1,?>>, index)
  %5:2 = hlfir.declare %4#0 typeparams %4#1 uniq_name("_QFmin1Ec3") : (!fir.ref<!fir.char<1,?>>, index) -> (!fir.boxchar<1>, !fir.ref<!fir.char<1,?>>)
  %6 = hlfir.char_extremum min, %3#0, %5#0 : (!fir.boxchar<1>, !fir.boxchar<1>) -> !hlfir.expr<!fir.char<1,?>>
  hlfir.assign %6 to %1#0 : !hlfir.expr<!fir.char<1,?>>, !fir.boxchar<1>
  hlfir.destroy %6 : !hlfir.expr<!fir.char<1,?>>
  return
}

// CHECK-LABEL:   func.func @_QPmin1(
// CHECK-SAME:      %[[ARG0:[0-9]+|[a-zA-Z$._-][a-zA-Z0-9$._-]*]]: !fir.boxchar<1> {fir.bindc_name = "c1"},
// CHECK-SAME:      %[[ARG1:[0-9]+|[a-zA-Z$._-][a-zA-Z0-9$._-]*]]: !fir.boxchar<1> {fir.bindc_name = "c2"},
// CHECK-SAME:      %[[ARG2:[0-9]+|[a-zA-Z$._-][a-zA-Z0-9$._-]*]]: !fir.boxchar<1> {fir.bindc_name = "c3"}) {
// CHECK:           %[[UNBOXCHAR_0:.*]]:2 = fir.unboxchar %[[ARG0]] : (!fir.boxchar<1>) -> (!fir.ref<!fir.char<1,?>>, index)
// CHECK:           %[[DECLARE_0:.*]]:2 = hlfir.declare %[[UNBOXCHAR_0]]#0 typeparams %[[UNBOXCHAR_0]]#1 uniq_name("_QFmin1Ec1") : (!fir.ref<!fir.char<1,?>>, index) -> (!fir.boxchar<1>, !fir.ref<!fir.char<1,?>>)
// CHECK:           %[[UNBOXCHAR_1:.*]]:2 = fir.unboxchar %[[ARG1]] : (!fir.boxchar<1>) -> (!fir.ref<!fir.char<1,?>>, index)
// CHECK:           %[[DECLARE_1:.*]]:2 = hlfir.declare %[[UNBOXCHAR_1]]#0 typeparams %[[UNBOXCHAR_1]]#1 uniq_name("_QFmin1Ec2") : (!fir.ref<!fir.char<1,?>>, index) -> (!fir.boxchar<1>, !fir.ref<!fir.char<1,?>>)
// CHECK:           %[[UNBOXCHAR_2:.*]]:2 = fir.unboxchar %[[ARG2]] : (!fir.boxchar<1>) -> (!fir.ref<!fir.char<1,?>>, index)
// CHECK:           %[[DECLARE_2:.*]]:2 = hlfir.declare %[[UNBOXCHAR_2]]#0 typeparams %[[UNBOXCHAR_2]]#1 uniq_name("_QFmin1Ec3") : (!fir.ref<!fir.char<1,?>>, index) -> (!fir.boxchar<1>, !fir.ref<!fir.char<1,?>>)
// CHECK:           %[[CMPI_0:.*]] = arith.cmpi uge, %[[UNBOXCHAR_1]]#1, %[[UNBOXCHAR_2]]#1 : index
// CHECK:           %[[SELECT_0:.*]] = arith.select %[[CMPI_0]], %[[UNBOXCHAR_1]]#1, %[[UNBOXCHAR_2]]#1 : index
// CHECK:           %[[CONVERT_0:.*]] = fir.convert %[[DECLARE_2]]#1 : (!fir.ref<!fir.char<1,?>>) -> !fir.ref<i8>
// CHECK:           %[[CONVERT_1:.*]] = fir.convert %[[DECLARE_1]]#1 : (!fir.ref<!fir.char<1,?>>) -> !fir.ref<i8>
// CHECK:           %[[CONVERT_2:.*]] = fir.convert %[[UNBOXCHAR_2]]#1 : (index) -> i64
// CHECK:           %[[CONVERT_3:.*]] = fir.convert %[[UNBOXCHAR_1]]#1 : (index) -> i64
// CHECK:           %[[CALL_0:.*]] = fir.call @_FortranACharacterCompareScalar1(%[[CONVERT_0]], %[[CONVERT_1]], %[[CONVERT_2]], %[[CONVERT_3]]) : (!fir.ref<i8>, !fir.ref<i8>, i64, i64) -> i32
// CHECK:           %[[CONSTANT_0:.*]] = arith.constant 0 : i32
// CHECK:           %[[CMPI_1:.*]] = arith.cmpi slt, %[[CALL_0]], %[[CONSTANT_0]] : i32
// CHECK:           %[[SELECT_1:.*]] = arith.select %[[CMPI_1]], %[[DECLARE_2]]#1, %[[DECLARE_1]]#1 : !fir.ref<!fir.char<1,?>>
// CHECK:           %[[SELECT_2:.*]] = arith.select %[[CMPI_1]], %[[UNBOXCHAR_2]]#1, %[[UNBOXCHAR_1]]#1 : index
// CHECK:           %[[ALLOCMEM_0:.*]] = fir.allocmem !fir.char<1,?>(%[[SELECT_0]] : index) <{bindc_name = ".chrtmp", uniq_name = ""}>
// CHECK:           %[[CMPI_2:.*]] = arith.cmpi slt, %[[SELECT_0]], %[[SELECT_2]] : index
// CHECK:           %[[SELECT_3:.*]] = arith.select %[[CMPI_2]], %[[SELECT_0]], %[[SELECT_2]] : index
// CHECK:           %[[CONSTANT_1:.*]] = arith.constant 1 : i64
// CHECK:           %[[CONVERT_4:.*]] = fir.convert %[[SELECT_3]] : (index) -> i64
// CHECK:           %[[MULI_0:.*]] = arith.muli %[[CONSTANT_1]], %[[CONVERT_4]] : i64
// CHECK:           %[[CONVERT_5:.*]] = fir.convert %[[ALLOCMEM_0]] : (!fir.heap<!fir.char<1,?>>) -> !llvm.ptr
// CHECK:           %[[CONVERT_6:.*]] = fir.convert %[[SELECT_1]] : (!fir.ref<!fir.char<1,?>>) -> !llvm.ptr
// CHECK:           "llvm.intr.memmove"(%[[CONVERT_5]], %[[CONVERT_6]], %[[MULI_0]]) <{isVolatile = false}> : (!llvm.ptr, !llvm.ptr, i64) -> ()
// CHECK:           %[[CONSTANT_2:.*]] = arith.constant 1 : index
// CHECK:           %[[SUBI_0:.*]] = arith.subi %[[SELECT_0]], %[[CONSTANT_2]] : index
// CHECK:           %[[CONSTANT_3:.*]] = arith.constant 32 : i8
// CHECK:           %[[UNDEFINED_0:.*]] = fir.undefined !fir.char<1>
// CHECK:           %[[INSERT_VALUE_0:.*]] = fir.insert_value %[[UNDEFINED_0]], %[[CONSTANT_3]], [0 : index] : (!fir.char<1>, i8) -> !fir.char<1>
// CHECK:           %[[CONSTANT_4:.*]] = arith.constant 1 : index
// CHECK:           fir.do_loop %[[VAL_0:.*]] = %[[SELECT_3]] to %[[SUBI_0]] step %[[CONSTANT_4]] {
// CHECK:             %[[CONVERT_7:.*]] = fir.convert %[[ALLOCMEM_0]] : (!fir.heap<!fir.char<1,?>>) -> !fir.ref<!fir.array<?x!fir.char<1>>>
// CHECK:             %[[COORDINATE_OF_0:.*]] = fir.coordinate_of %[[CONVERT_7]], %[[VAL_0]] : (!fir.ref<!fir.array<?x!fir.char<1>>>, index) -> !fir.ref<!fir.char<1>>
// CHECK:             fir.store %[[INSERT_VALUE_0]] to %[[COORDINATE_OF_0]] : !fir.ref<!fir.char<1>>
// CHECK:           }
// CHECK:           %[[CONVERT_8:.*]] = fir.convert %[[ALLOCMEM_0]] : (!fir.heap<!fir.char<1,?>>) -> !fir.ref<!fir.char<1,?>>
// CHECK:           %[[DECLARE_3:.*]]:2 = hlfir.declare %[[CONVERT_8]] typeparams %[[SELECT_0]] uniq_name(".tmp.char_extremum") : (!fir.ref<!fir.char<1,?>>, index) -> (!fir.boxchar<1>, !fir.ref<!fir.char<1,?>>)
// CHECK:           %[[CONSTANT_5:.*]] = arith.constant true
// CHECK:           %[[UNDEFINED_1:.*]] = fir.undefined tuple<!fir.boxchar<1>, i1>
// CHECK:           %[[INSERT_VALUE_1:.*]] = fir.insert_value %[[UNDEFINED_1]], %[[CONSTANT_5]], [1 : index] : (tuple<!fir.boxchar<1>, i1>, i1) -> tuple<!fir.boxchar<1>, i1>
// CHECK:           %[[INSERT_VALUE_2:.*]] = fir.insert_value %[[INSERT_VALUE_1]], %[[DECLARE_3]]#0, [0 : index] : (tuple<!fir.boxchar<1>, i1>, !fir.boxchar<1>) -> tuple<!fir.boxchar<1>, i1>
// CHECK:           hlfir.assign %[[DECLARE_3]]#0 to %[[DECLARE_0]]#0 : !fir.boxchar<1>, !fir.boxchar<1>
// CHECK:           %[[BOX_ADDR_0:.*]] = fir.box_addr %[[DECLARE_3]]#0 : (!fir.boxchar<1>) -> !fir.heap<!fir.char<1,?>>
// CHECK:           fir.freemem %[[BOX_ADDR_0]] : !fir.heap<!fir.char<1,?>>
// CHECK:           return
// CHECK:         }

func.func @_QPmax2(%arg0: !fir.boxchar<1> {fir.bindc_name = "c1"}, %arg1: !fir.boxchar<1> {fir.bindc_name = "c2"}, %arg2: !fir.boxchar<1> {fir.bindc_name = "c3"}) {
  %0:2 = fir.unboxchar %arg0 : (!fir.boxchar<1>) -> (!fir.ref<!fir.char<1,?>>, index)
  %1 = fir.convert %0#0 : (!fir.ref<!fir.char<1,?>>) -> !fir.ref<!fir.array<100x!fir.char<1,?>>>
  %c100 = arith.constant 100 : index
  %2 = fir.shape %c100 : (index) -> !fir.shape<1>
  %3:2 = hlfir.declare %1(%2) typeparams %0#1 uniq_name("_QFmax2Ec1") : (!fir.ref<!fir.array<100x!fir.char<1,?>>>, !fir.shape<1>, index) -> (!fir.box<!fir.array<100x!fir.char<1,?>>>, !fir.ref<!fir.array<100x!fir.char<1,?>>>)
  %4:2 = fir.unboxchar %arg1 : (!fir.boxchar<1>) -> (!fir.ref<!fir.char<1,?>>, index)
  %c10 = arith.constant 10 : index
  %5 = fir.convert %4#0 : (!fir.ref<!fir.char<1,?>>) -> !fir.ref<!fir.array<100x!fir.char<1,10>>>
  %c100_0 = arith.constant 100 : index
  %6 = fir.shape %c100_0 : (index) -> !fir.shape<1>
  %7:2 = hlfir.declare %5(%6) typeparams %c10 uniq_name("_QFmax2Ec2") : (!fir.ref<!fir.array<100x!fir.char<1,10>>>, !fir.shape<1>, index) -> (!fir.ref<!fir.array<100x!fir.char<1,10>>>, !fir.ref<!fir.array<100x!fir.char<1,10>>>)
  %8:2 = fir.unboxchar %arg2 : (!fir.boxchar<1>) -> (!fir.ref<!fir.char<1,?>>, index)
  %c20 = arith.constant 20 : index
  %9 = fir.convert %8#0 : (!fir.ref<!fir.char<1,?>>) -> !fir.ref<!fir.array<100x!fir.char<1,20>>>
  %c100_1 = arith.constant 100 : index
  %10 = fir.shape %c100_1 : (index) -> !fir.shape<1>
  %11:2 = hlfir.declare %9(%10) typeparams %c20 uniq_name("_QFmax2Ec3") : (!fir.ref<!fir.array<100x!fir.char<1,20>>>, !fir.shape<1>, index) -> (!fir.ref<!fir.array<100x!fir.char<1,20>>>, !fir.ref<!fir.array<100x!fir.char<1,20>>>)
  %c1 = arith.constant 1 : index
  %12 = hlfir.designate %7#0 (%c1)  typeparams %c10 : (!fir.ref<!fir.array<100x!fir.char<1,10>>>, index, index) -> !fir.ref<!fir.char<1,10>>
  %c1_2 = arith.constant 1 : index
  %13 = hlfir.designate %11#0 (%c1_2)  typeparams %c20 : (!fir.ref<!fir.array<100x!fir.char<1,20>>>, index, index) -> !fir.ref<!fir.char<1,20>>
  %14 = hlfir.char_extremum max, %12, %13 : (!fir.ref<!fir.char<1,10>>, !fir.ref<!fir.char<1,20>>) -> !hlfir.expr<!fir.char<1,20>>
  %c1_3 = arith.constant 1 : index
  %15 = hlfir.designate %3#0 (%c1_3)  typeparams %0#1 : (!fir.box<!fir.array<100x!fir.char<1,?>>>, index, index) -> !fir.boxchar<1>
  hlfir.assign %14 to %15 : !hlfir.expr<!fir.char<1,20>>, !fir.boxchar<1>
  hlfir.destroy %14 : !hlfir.expr<!fir.char<1,20>>
  return
}

// CHECK-LABEL:   func.func @_QPmax2(
// CHECK-SAME:      %[[ARG0:[0-9]+|[a-zA-Z$._-][a-zA-Z0-9$._-]*]]: !fir.boxchar<1> {fir.bindc_name = "c1"},
// CHECK-SAME:      %[[ARG1:[0-9]+|[a-zA-Z$._-][a-zA-Z0-9$._-]*]]: !fir.boxchar<1> {fir.bindc_name = "c2"},
// CHECK-SAME:      %[[ARG2:[0-9]+|[a-zA-Z$._-][a-zA-Z0-9$._-]*]]: !fir.boxchar<1> {fir.bindc_name = "c3"}) {
// CHECK:           %[[UNBOXCHAR_0:.*]]:2 = fir.unboxchar %[[ARG0]] : (!fir.boxchar<1>) -> (!fir.ref<!fir.char<1,?>>, index)
// CHECK:           %[[CONVERT_0:.*]] = fir.convert %[[UNBOXCHAR_0]]#0 : (!fir.ref<!fir.char<1,?>>) -> !fir.ref<!fir.array<100x!fir.char<1,?>>>
// CHECK:           %[[CONSTANT_0:.*]] = arith.constant 100 : index
// CHECK:           %[[SHAPE_0:.*]] = fir.shape %[[CONSTANT_0]] : (index) -> !fir.shape<1>
// CHECK:           %[[DECLARE_0:.*]]:2 = hlfir.declare %[[CONVERT_0]](%[[SHAPE_0]]) typeparams %[[UNBOXCHAR_0]]#1 uniq_name("_QFmax2Ec1") : (!fir.ref<!fir.array<100x!fir.char<1,?>>>, !fir.shape<1>, index) -> (!fir.box<!fir.array<100x!fir.char<1,?>>>, !fir.ref<!fir.array<100x!fir.char<1,?>>>)
// CHECK:           %[[UNBOXCHAR_1:.*]]:2 = fir.unboxchar %[[ARG1]] : (!fir.boxchar<1>) -> (!fir.ref<!fir.char<1,?>>, index)
// CHECK:           %[[CONSTANT_1:.*]] = arith.constant 10 : index
// CHECK:           %[[CONVERT_1:.*]] = fir.convert %[[UNBOXCHAR_1]]#0 : (!fir.ref<!fir.char<1,?>>) -> !fir.ref<!fir.array<100x!fir.char<1,10>>>
// CHECK:           %[[CONSTANT_2:.*]] = arith.constant 100 : index
// CHECK:           %[[SHAPE_1:.*]] = fir.shape %[[CONSTANT_2]] : (index) -> !fir.shape<1>
// CHECK:           %[[DECLARE_1:.*]]:2 = hlfir.declare %[[CONVERT_1]](%[[SHAPE_1]]) typeparams %[[CONSTANT_1]] uniq_name("_QFmax2Ec2") : (!fir.ref<!fir.array<100x!fir.char<1,10>>>, !fir.shape<1>, index) -> (!fir.ref<!fir.array<100x!fir.char<1,10>>>, !fir.ref<!fir.array<100x!fir.char<1,10>>>)
// CHECK:           %[[UNBOXCHAR_2:.*]]:2 = fir.unboxchar %[[ARG2]] : (!fir.boxchar<1>) -> (!fir.ref<!fir.char<1,?>>, index)
// CHECK:           %[[CONSTANT_3:.*]] = arith.constant 20 : index
// CHECK:           %[[CONVERT_2:.*]] = fir.convert %[[UNBOXCHAR_2]]#0 : (!fir.ref<!fir.char<1,?>>) -> !fir.ref<!fir.array<100x!fir.char<1,20>>>
// CHECK:           %[[CONSTANT_4:.*]] = arith.constant 100 : index
// CHECK:           %[[SHAPE_2:.*]] = fir.shape %[[CONSTANT_4]] : (index) -> !fir.shape<1>
// CHECK:           %[[DECLARE_2:.*]]:2 = hlfir.declare %[[CONVERT_2]](%[[SHAPE_2]]) typeparams %[[CONSTANT_3]] uniq_name("_QFmax2Ec3") : (!fir.ref<!fir.array<100x!fir.char<1,20>>>, !fir.shape<1>, index) -> (!fir.ref<!fir.array<100x!fir.char<1,20>>>, !fir.ref<!fir.array<100x!fir.char<1,20>>>)
// CHECK:           %[[CONSTANT_5:.*]] = arith.constant 1 : index
// CHECK:           %[[DESIGNATE_0:.*]] = hlfir.designate %[[DECLARE_1]]#0 (%[[CONSTANT_5]])  typeparams %[[CONSTANT_1]] : (!fir.ref<!fir.array<100x!fir.char<1,10>>>, index, index) -> !fir.ref<!fir.char<1,10>>
// CHECK:           %[[CONSTANT_6:.*]] = arith.constant 1 : index
// CHECK:           %[[DESIGNATE_1:.*]] = hlfir.designate %[[DECLARE_2]]#0 (%[[CONSTANT_6]])  typeparams %[[CONSTANT_3]] : (!fir.ref<!fir.array<100x!fir.char<1,20>>>, index, index) -> !fir.ref<!fir.char<1,20>>
// CHECK:           %[[CMPI_0:.*]] = arith.cmpi uge, %[[CONSTANT_1]], %[[CONSTANT_3]] : index
// CHECK:           %[[SELECT_0:.*]] = arith.select %[[CMPI_0]], %[[CONSTANT_1]], %[[CONSTANT_3]] : index
// CHECK:           %[[CONVERT_3:.*]] = fir.convert %[[DESIGNATE_1]] : (!fir.ref<!fir.char<1,20>>) -> !fir.ref<i8>
// CHECK:           %[[CONVERT_4:.*]] = fir.convert %[[DESIGNATE_0]] : (!fir.ref<!fir.char<1,10>>) -> !fir.ref<i8>
// CHECK:           %[[CONVERT_5:.*]] = fir.convert %[[CONSTANT_3]] : (index) -> i64
// CHECK:           %[[CONVERT_6:.*]] = fir.convert %[[CONSTANT_1]] : (index) -> i64
// CHECK:           %[[CALL_0:.*]] = fir.call @_FortranACharacterCompareScalar1(%[[CONVERT_3]], %[[CONVERT_4]], %[[CONVERT_5]], %[[CONVERT_6]]) : (!fir.ref<i8>, !fir.ref<i8>, i64, i64) -> i32
// CHECK:           %[[CONSTANT_7:.*]] = arith.constant 0 : i32
// CHECK:           %[[CMPI_1:.*]] = arith.cmpi sgt, %[[CALL_0]], %[[CONSTANT_7]] : i32
// CHECK:           %[[CONVERT_7:.*]] = fir.convert %[[DESIGNATE_0]] : (!fir.ref<!fir.char<1,10>>) -> !fir.ref<!fir.char<1,?>>
// CHECK:           %[[CONVERT_8:.*]] = fir.convert %[[DESIGNATE_1]] : (!fir.ref<!fir.char<1,20>>) -> !fir.ref<!fir.char<1,?>>
// CHECK:           %[[SELECT_1:.*]] = arith.select %[[CMPI_1]], %[[CONVERT_8]], %[[CONVERT_7]] : !fir.ref<!fir.char<1,?>>
// CHECK:           %[[SELECT_2:.*]] = arith.select %[[CMPI_1]], %[[CONSTANT_3]], %[[CONSTANT_1]] : index
// CHECK:           %[[ALLOCMEM_0:.*]] = fir.allocmem !fir.char<1,?>(%[[SELECT_0]] : index) <{bindc_name = ".chrtmp", uniq_name = ""}>
// CHECK:           %[[CMPI_2:.*]] = arith.cmpi slt, %[[SELECT_0]], %[[SELECT_2]] : index
// CHECK:           %[[SELECT_3:.*]] = arith.select %[[CMPI_2]], %[[SELECT_0]], %[[SELECT_2]] : index
// CHECK:           %[[CONSTANT_8:.*]] = arith.constant 1 : i64
// CHECK:           %[[CONVERT_9:.*]] = fir.convert %[[SELECT_3]] : (index) -> i64
// CHECK:           %[[MULI_0:.*]] = arith.muli %[[CONSTANT_8]], %[[CONVERT_9]] : i64
// CHECK:           %[[CONVERT_10:.*]] = fir.convert %[[ALLOCMEM_0]] : (!fir.heap<!fir.char<1,?>>) -> !llvm.ptr
// CHECK:           %[[CONVERT_11:.*]] = fir.convert %[[SELECT_1]] : (!fir.ref<!fir.char<1,?>>) -> !llvm.ptr
// CHECK:           "llvm.intr.memmove"(%[[CONVERT_10]], %[[CONVERT_11]], %[[MULI_0]]) <{isVolatile = false}> : (!llvm.ptr, !llvm.ptr, i64) -> ()
// CHECK:           %[[CONSTANT_9:.*]] = arith.constant 1 : index
// CHECK:           %[[SUBI_0:.*]] = arith.subi %[[SELECT_0]], %[[CONSTANT_9]] : index
// CHECK:           %[[CONSTANT_10:.*]] = arith.constant 32 : i8
// CHECK:           %[[UNDEFINED_0:.*]] = fir.undefined !fir.char<1>
// CHECK:           %[[INSERT_VALUE_0:.*]] = fir.insert_value %[[UNDEFINED_0]], %[[CONSTANT_10]], [0 : index] : (!fir.char<1>, i8) -> !fir.char<1>
// CHECK:           %[[CONSTANT_11:.*]] = arith.constant 1 : index
// CHECK:           fir.do_loop %[[VAL_0:.*]] = %[[SELECT_3]] to %[[SUBI_0]] step %[[CONSTANT_11]] {
// CHECK:             %[[CONVERT_12:.*]] = fir.convert %[[ALLOCMEM_0]] : (!fir.heap<!fir.char<1,?>>) -> !fir.ref<!fir.array<?x!fir.char<1>>>
// CHECK:             %[[COORDINATE_OF_0:.*]] = fir.coordinate_of %[[CONVERT_12]], %[[VAL_0]] : (!fir.ref<!fir.array<?x!fir.char<1>>>, index) -> !fir.ref<!fir.char<1>>
// CHECK:             fir.store %[[INSERT_VALUE_0]] to %[[COORDINATE_OF_0]] : !fir.ref<!fir.char<1>>
// CHECK:           }
// CHECK:           %[[CONVERT_13:.*]] = fir.convert %[[ALLOCMEM_0]] : (!fir.heap<!fir.char<1,?>>) -> !fir.ref<!fir.char<1,20>>
// CHECK:           %[[DECLARE_3:.*]]:2 = hlfir.declare %[[CONVERT_13]] typeparams %[[SELECT_0]] uniq_name(".tmp.char_extremum") : (!fir.ref<!fir.char<1,20>>, index) -> (!fir.ref<!fir.char<1,20>>, !fir.ref<!fir.char<1,20>>)
// CHECK:           %[[CONSTANT_12:.*]] = arith.constant true
// CHECK:           %[[UNDEFINED_1:.*]] = fir.undefined tuple<!fir.ref<!fir.char<1,20>>, i1>
// CHECK:           %[[INSERT_VALUE_1:.*]] = fir.insert_value %[[UNDEFINED_1]], %[[CONSTANT_12]], [1 : index] : (tuple<!fir.ref<!fir.char<1,20>>, i1>, i1) -> tuple<!fir.ref<!fir.char<1,20>>, i1>
// CHECK:           %[[INSERT_VALUE_2:.*]] = fir.insert_value %[[INSERT_VALUE_1]], %[[DECLARE_3]]#0, [0 : index] : (tuple<!fir.ref<!fir.char<1,20>>, i1>, !fir.ref<!fir.char<1,20>>) -> tuple<!fir.ref<!fir.char<1,20>>, i1>
// CHECK:           %[[CONSTANT_13:.*]] = arith.constant 1 : index
// CHECK:           %[[DESIGNATE_2:.*]] = hlfir.designate %[[DECLARE_0]]#0 (%[[CONSTANT_13]])  typeparams %[[UNBOXCHAR_0]]#1 : (!fir.box<!fir.array<100x!fir.char<1,?>>>, index, index) -> !fir.boxchar<1>
// CHECK:           hlfir.assign %[[DECLARE_3]]#0 to %[[DESIGNATE_2]] : !fir.ref<!fir.char<1,20>>, !fir.boxchar<1>
// CHECK:           %[[CONVERT_14:.*]] = fir.convert %[[DECLARE_3]]#0 : (!fir.ref<!fir.char<1,20>>) -> !fir.heap<!fir.char<1,20>>
// CHECK:           fir.freemem %[[CONVERT_14]] : !fir.heap<!fir.char<1,20>>
// CHECK:           return
// CHECK:         }

func.func @_QPmin2(%arg0: !fir.boxchar<1> {fir.bindc_name = "c1"}, %arg1: !fir.boxchar<1> {fir.bindc_name = "c2"}, %arg2: !fir.boxchar<1> {fir.bindc_name = "c3"}) {
  %0:2 = fir.unboxchar %arg0 : (!fir.boxchar<1>) -> (!fir.ref<!fir.char<1,?>>, index)
  %1 = fir.convert %0#0 : (!fir.ref<!fir.char<1,?>>) -> !fir.ref<!fir.array<100x!fir.char<1,?>>>
  %c100 = arith.constant 100 : index
  %2 = fir.shape %c100 : (index) -> !fir.shape<1>
  %3:2 = hlfir.declare %1(%2) typeparams %0#1 uniq_name("_QFmin2Ec1") : (!fir.ref<!fir.array<100x!fir.char<1,?>>>, !fir.shape<1>, index) -> (!fir.box<!fir.array<100x!fir.char<1,?>>>, !fir.ref<!fir.array<100x!fir.char<1,?>>>)
  %4:2 = fir.unboxchar %arg1 : (!fir.boxchar<1>) -> (!fir.ref<!fir.char<1,?>>, index)
  %c10 = arith.constant 10 : index
  %5 = fir.convert %4#0 : (!fir.ref<!fir.char<1,?>>) -> !fir.ref<!fir.array<100x!fir.char<1,10>>>
  %c100_0 = arith.constant 100 : index
  %6 = fir.shape %c100_0 : (index) -> !fir.shape<1>
  %7:2 = hlfir.declare %5(%6) typeparams %c10 uniq_name("_QFmin2Ec2") : (!fir.ref<!fir.array<100x!fir.char<1,10>>>, !fir.shape<1>, index) -> (!fir.ref<!fir.array<100x!fir.char<1,10>>>, !fir.ref<!fir.array<100x!fir.char<1,10>>>)
  %8:2 = fir.unboxchar %arg2 : (!fir.boxchar<1>) -> (!fir.ref<!fir.char<1,?>>, index)
  %c20 = arith.constant 20 : index
  %9 = fir.convert %8#0 : (!fir.ref<!fir.char<1,?>>) -> !fir.ref<!fir.array<100x!fir.char<1,20>>>
  %c100_1 = arith.constant 100 : index
  %10 = fir.shape %c100_1 : (index) -> !fir.shape<1>
  %11:2 = hlfir.declare %9(%10) typeparams %c20 uniq_name("_QFmin2Ec3") : (!fir.ref<!fir.array<100x!fir.char<1,20>>>, !fir.shape<1>, index) -> (!fir.ref<!fir.array<100x!fir.char<1,20>>>, !fir.ref<!fir.array<100x!fir.char<1,20>>>)
  %c1 = arith.constant 1 : index
  %12 = hlfir.designate %7#0 (%c1)  typeparams %c10 : (!fir.ref<!fir.array<100x!fir.char<1,10>>>, index, index) -> !fir.ref<!fir.char<1,10>>
  %c1_2 = arith.constant 1 : index
  %13 = hlfir.designate %11#0 (%c1_2)  typeparams %c20 : (!fir.ref<!fir.array<100x!fir.char<1,20>>>, index, index) -> !fir.ref<!fir.char<1,20>>
  %14 = hlfir.char_extremum min, %12, %13 : (!fir.ref<!fir.char<1,10>>, !fir.ref<!fir.char<1,20>>) -> !hlfir.expr<!fir.char<1,20>>
  %c1_3 = arith.constant 1 : index
  %15 = hlfir.designate %3#0 (%c1_3)  typeparams %0#1 : (!fir.box<!fir.array<100x!fir.char<1,?>>>, index, index) -> !fir.boxchar<1>
  hlfir.assign %14 to %15 : !hlfir.expr<!fir.char<1,20>>, !fir.boxchar<1>
  hlfir.destroy %14 : !hlfir.expr<!fir.char<1,20>>
  return
}

// CHECK-LABEL:   func.func @_QPmin2(
// CHECK-SAME:      %[[ARG0:[0-9]+|[a-zA-Z$._-][a-zA-Z0-9$._-]*]]: !fir.boxchar<1> {fir.bindc_name = "c1"},
// CHECK-SAME:      %[[ARG1:[0-9]+|[a-zA-Z$._-][a-zA-Z0-9$._-]*]]: !fir.boxchar<1> {fir.bindc_name = "c2"},
// CHECK-SAME:      %[[ARG2:[0-9]+|[a-zA-Z$._-][a-zA-Z0-9$._-]*]]: !fir.boxchar<1> {fir.bindc_name = "c3"}) {
// CHECK:           %[[UNBOXCHAR_0:.*]]:2 = fir.unboxchar %[[ARG0]] : (!fir.boxchar<1>) -> (!fir.ref<!fir.char<1,?>>, index)
// CHECK:           %[[CONVERT_0:.*]] = fir.convert %[[UNBOXCHAR_0]]#0 : (!fir.ref<!fir.char<1,?>>) -> !fir.ref<!fir.array<100x!fir.char<1,?>>>
// CHECK:           %[[CONSTANT_0:.*]] = arith.constant 100 : index
// CHECK:           %[[SHAPE_0:.*]] = fir.shape %[[CONSTANT_0]] : (index) -> !fir.shape<1>
// CHECK:           %[[DECLARE_0:.*]]:2 = hlfir.declare %[[CONVERT_0]](%[[SHAPE_0]]) typeparams %[[UNBOXCHAR_0]]#1 uniq_name("_QFmin2Ec1") : (!fir.ref<!fir.array<100x!fir.char<1,?>>>, !fir.shape<1>, index) -> (!fir.box<!fir.array<100x!fir.char<1,?>>>, !fir.ref<!fir.array<100x!fir.char<1,?>>>)
// CHECK:           %[[UNBOXCHAR_1:.*]]:2 = fir.unboxchar %[[ARG1]] : (!fir.boxchar<1>) -> (!fir.ref<!fir.char<1,?>>, index)
// CHECK:           %[[CONSTANT_1:.*]] = arith.constant 10 : index
// CHECK:           %[[CONVERT_1:.*]] = fir.convert %[[UNBOXCHAR_1]]#0 : (!fir.ref<!fir.char<1,?>>) -> !fir.ref<!fir.array<100x!fir.char<1,10>>>
// CHECK:           %[[CONSTANT_2:.*]] = arith.constant 100 : index
// CHECK:           %[[SHAPE_1:.*]] = fir.shape %[[CONSTANT_2]] : (index) -> !fir.shape<1>
// CHECK:           %[[DECLARE_1:.*]]:2 = hlfir.declare %[[CONVERT_1]](%[[SHAPE_1]]) typeparams %[[CONSTANT_1]] uniq_name("_QFmin2Ec2") : (!fir.ref<!fir.array<100x!fir.char<1,10>>>, !fir.shape<1>, index) -> (!fir.ref<!fir.array<100x!fir.char<1,10>>>, !fir.ref<!fir.array<100x!fir.char<1,10>>>)
// CHECK:           %[[UNBOXCHAR_2:.*]]:2 = fir.unboxchar %[[ARG2]] : (!fir.boxchar<1>) -> (!fir.ref<!fir.char<1,?>>, index)
// CHECK:           %[[CONSTANT_3:.*]] = arith.constant 20 : index
// CHECK:           %[[CONVERT_2:.*]] = fir.convert %[[UNBOXCHAR_2]]#0 : (!fir.ref<!fir.char<1,?>>) -> !fir.ref<!fir.array<100x!fir.char<1,20>>>
// CHECK:           %[[CONSTANT_4:.*]] = arith.constant 100 : index
// CHECK:           %[[SHAPE_2:.*]] = fir.shape %[[CONSTANT_4]] : (index) -> !fir.shape<1>
// CHECK:           %[[DECLARE_2:.*]]:2 = hlfir.declare %[[CONVERT_2]](%[[SHAPE_2]]) typeparams %[[CONSTANT_3]] uniq_name("_QFmin2Ec3") : (!fir.ref<!fir.array<100x!fir.char<1,20>>>, !fir.shape<1>, index) -> (!fir.ref<!fir.array<100x!fir.char<1,20>>>, !fir.ref<!fir.array<100x!fir.char<1,20>>>)
// CHECK:           %[[CONSTANT_5:.*]] = arith.constant 1 : index
// CHECK:           %[[DESIGNATE_0:.*]] = hlfir.designate %[[DECLARE_1]]#0 (%[[CONSTANT_5]])  typeparams %[[CONSTANT_1]] : (!fir.ref<!fir.array<100x!fir.char<1,10>>>, index, index) -> !fir.ref<!fir.char<1,10>>
// CHECK:           %[[CONSTANT_6:.*]] = arith.constant 1 : index
// CHECK:           %[[DESIGNATE_1:.*]] = hlfir.designate %[[DECLARE_2]]#0 (%[[CONSTANT_6]])  typeparams %[[CONSTANT_3]] : (!fir.ref<!fir.array<100x!fir.char<1,20>>>, index, index) -> !fir.ref<!fir.char<1,20>>
// CHECK:           %[[CMPI_0:.*]] = arith.cmpi uge, %[[CONSTANT_1]], %[[CONSTANT_3]] : index
// CHECK:           %[[SELECT_0:.*]] = arith.select %[[CMPI_0]], %[[CONSTANT_1]], %[[CONSTANT_3]] : index
// CHECK:           %[[CONVERT_3:.*]] = fir.convert %[[DESIGNATE_1]] : (!fir.ref<!fir.char<1,20>>) -> !fir.ref<i8>
// CHECK:           %[[CONVERT_4:.*]] = fir.convert %[[DESIGNATE_0]] : (!fir.ref<!fir.char<1,10>>) -> !fir.ref<i8>
// CHECK:           %[[CONVERT_5:.*]] = fir.convert %[[CONSTANT_3]] : (index) -> i64
// CHECK:           %[[CONVERT_6:.*]] = fir.convert %[[CONSTANT_1]] : (index) -> i64
// CHECK:           %[[CALL_0:.*]] = fir.call @_FortranACharacterCompareScalar1(%[[CONVERT_3]], %[[CONVERT_4]], %[[CONVERT_5]], %[[CONVERT_6]]) : (!fir.ref<i8>, !fir.ref<i8>, i64, i64) -> i32
// CHECK:           %[[CONSTANT_7:.*]] = arith.constant 0 : i32
// CHECK:           %[[CMPI_1:.*]] = arith.cmpi slt, %[[CALL_0]], %[[CONSTANT_7]] : i32
// CHECK:           %[[CONVERT_7:.*]] = fir.convert %[[DESIGNATE_0]] : (!fir.ref<!fir.char<1,10>>) -> !fir.ref<!fir.char<1,?>>
// CHECK:           %[[CONVERT_8:.*]] = fir.convert %[[DESIGNATE_1]] : (!fir.ref<!fir.char<1,20>>) -> !fir.ref<!fir.char<1,?>>
// CHECK:           %[[SELECT_1:.*]] = arith.select %[[CMPI_1]], %[[CONVERT_8]], %[[CONVERT_7]] : !fir.ref<!fir.char<1,?>>
// CHECK:           %[[SELECT_2:.*]] = arith.select %[[CMPI_1]], %[[CONSTANT_3]], %[[CONSTANT_1]] : index
// CHECK:           %[[ALLOCMEM_0:.*]] = fir.allocmem !fir.char<1,?>(%[[SELECT_0]] : index) <{bindc_name = ".chrtmp", uniq_name = ""}>
// CHECK:           %[[CMPI_2:.*]] = arith.cmpi slt, %[[SELECT_0]], %[[SELECT_2]] : index
// CHECK:           %[[SELECT_3:.*]] = arith.select %[[CMPI_2]], %[[SELECT_0]], %[[SELECT_2]] : index
// CHECK:           %[[CONSTANT_8:.*]] = arith.constant 1 : i64
// CHECK:           %[[CONVERT_9:.*]] = fir.convert %[[SELECT_3]] : (index) -> i64
// CHECK:           %[[MULI_0:.*]] = arith.muli %[[CONSTANT_8]], %[[CONVERT_9]] : i64
// CHECK:           %[[CONVERT_10:.*]] = fir.convert %[[ALLOCMEM_0]] : (!fir.heap<!fir.char<1,?>>) -> !llvm.ptr
// CHECK:           %[[CONVERT_11:.*]] = fir.convert %[[SELECT_1]] : (!fir.ref<!fir.char<1,?>>) -> !llvm.ptr
// CHECK:           "llvm.intr.memmove"(%[[CONVERT_10]], %[[CONVERT_11]], %[[MULI_0]]) <{isVolatile = false}> : (!llvm.ptr, !llvm.ptr, i64) -> ()
// CHECK:           %[[CONSTANT_9:.*]] = arith.constant 1 : index
// CHECK:           %[[SUBI_0:.*]] = arith.subi %[[SELECT_0]], %[[CONSTANT_9]] : index
// CHECK:           %[[CONSTANT_10:.*]] = arith.constant 32 : i8
// CHECK:           %[[UNDEFINED_0:.*]] = fir.undefined !fir.char<1>
// CHECK:           %[[INSERT_VALUE_0:.*]] = fir.insert_value %[[UNDEFINED_0]], %[[CONSTANT_10]], [0 : index] : (!fir.char<1>, i8) -> !fir.char<1>
// CHECK:           %[[CONSTANT_11:.*]] = arith.constant 1 : index
// CHECK:           fir.do_loop %[[VAL_0:.*]] = %[[SELECT_3]] to %[[SUBI_0]] step %[[CONSTANT_11]] {
// CHECK:             %[[CONVERT_12:.*]] = fir.convert %[[ALLOCMEM_0]] : (!fir.heap<!fir.char<1,?>>) -> !fir.ref<!fir.array<?x!fir.char<1>>>
// CHECK:             %[[COORDINATE_OF_0:.*]] = fir.coordinate_of %[[CONVERT_12]], %[[VAL_0]] : (!fir.ref<!fir.array<?x!fir.char<1>>>, index) -> !fir.ref<!fir.char<1>>
// CHECK:             fir.store %[[INSERT_VALUE_0]] to %[[COORDINATE_OF_0]] : !fir.ref<!fir.char<1>>
// CHECK:           }
// CHECK:           %[[CONVERT_13:.*]] = fir.convert %[[ALLOCMEM_0]] : (!fir.heap<!fir.char<1,?>>) -> !fir.ref<!fir.char<1,20>>
// CHECK:           %[[DECLARE_3:.*]]:2 = hlfir.declare %[[CONVERT_13]] typeparams %[[SELECT_0]] uniq_name(".tmp.char_extremum") : (!fir.ref<!fir.char<1,20>>, index) -> (!fir.ref<!fir.char<1,20>>, !fir.ref<!fir.char<1,20>>)
// CHECK:           %[[CONSTANT_12:.*]] = arith.constant true
// CHECK:           %[[UNDEFINED_1:.*]] = fir.undefined tuple<!fir.ref<!fir.char<1,20>>, i1>
// CHECK:           %[[INSERT_VALUE_1:.*]] = fir.insert_value %[[UNDEFINED_1]], %[[CONSTANT_12]], [1 : index] : (tuple<!fir.ref<!fir.char<1,20>>, i1>, i1) -> tuple<!fir.ref<!fir.char<1,20>>, i1>
// CHECK:           %[[INSERT_VALUE_2:.*]] = fir.insert_value %[[INSERT_VALUE_1]], %[[DECLARE_3]]#0, [0 : index] : (tuple<!fir.ref<!fir.char<1,20>>, i1>, !fir.ref<!fir.char<1,20>>) -> tuple<!fir.ref<!fir.char<1,20>>, i1>
// CHECK:           %[[CONSTANT_13:.*]] = arith.constant 1 : index
// CHECK:           %[[DESIGNATE_2:.*]] = hlfir.designate %[[DECLARE_0]]#0 (%[[CONSTANT_13]])  typeparams %[[UNBOXCHAR_0]]#1 : (!fir.box<!fir.array<100x!fir.char<1,?>>>, index, index) -> !fir.boxchar<1>
// CHECK:           hlfir.assign %[[DECLARE_3]]#0 to %[[DESIGNATE_2]] : !fir.ref<!fir.char<1,20>>, !fir.boxchar<1>
// CHECK:           %[[CONVERT_14:.*]] = fir.convert %[[DECLARE_3]]#0 : (!fir.ref<!fir.char<1,20>>) -> !fir.heap<!fir.char<1,20>>
// CHECK:           fir.freemem %[[CONVERT_14]] : !fir.heap<!fir.char<1,20>>
// CHECK:           return
// CHECK:         }

func.func @_QPmax3(%arg0: !fir.boxchar<1> {fir.bindc_name = "c1"}, %arg1: !fir.boxchar<1> {fir.bindc_name = "c2"}, %arg2: !fir.boxchar<1> {fir.bindc_name = "c3"}, %arg3: !fir.boxchar<1> {fir.bindc_name = "c4"}) {
  %0:2 = fir.unboxchar %arg0 : (!fir.boxchar<1>) -> (!fir.ref<!fir.char<1,?>>, index)
  %1:2 = hlfir.declare %0#0 typeparams %0#1 uniq_name("_QFmax3Ec1") : (!fir.ref<!fir.char<1,?>>, index) -> (!fir.boxchar<1>, !fir.ref<!fir.char<1,?>>)
  %2:2 = fir.unboxchar %arg1 : (!fir.boxchar<1>) -> (!fir.ref<!fir.char<1,?>>, index)
  %3:2 = hlfir.declare %2#0 typeparams %2#1 uniq_name("_QFmax3Ec2") : (!fir.ref<!fir.char<1,?>>, index) -> (!fir.boxchar<1>, !fir.ref<!fir.char<1,?>>)
  %4:2 = fir.unboxchar %arg2 : (!fir.boxchar<1>) -> (!fir.ref<!fir.char<1,?>>, index)
  %5:2 = hlfir.declare %4#0 typeparams %4#1 uniq_name("_QFmax3Ec3") : (!fir.ref<!fir.char<1,?>>, index) -> (!fir.boxchar<1>, !fir.ref<!fir.char<1,?>>)
  %6:2 = fir.unboxchar %arg3 : (!fir.boxchar<1>) -> (!fir.ref<!fir.char<1,?>>, index)
  %7:2 = hlfir.declare %6#0 typeparams %6#1 uniq_name("_QFmax3Ec4") : (!fir.ref<!fir.char<1,?>>, index) -> (!fir.boxchar<1>, !fir.ref<!fir.char<1,?>>)
  %8 = hlfir.char_extremum max, %3#0, %5#0, %7#0 : (!fir.boxchar<1>, !fir.boxchar<1>, !fir.boxchar<1>) -> !hlfir.expr<!fir.char<1,?>>
  hlfir.assign %8 to %1#0 : !hlfir.expr<!fir.char<1,?>>, !fir.boxchar<1>
  hlfir.destroy %8 : !hlfir.expr<!fir.char<1,?>>
  return
}

// CHECK-LABEL:   func.func @_QPmax3(
// CHECK-SAME:      %[[ARG0:[0-9]+|[a-zA-Z$._-][a-zA-Z0-9$._-]*]]: !fir.boxchar<1> {fir.bindc_name = "c1"},
// CHECK-SAME:      %[[ARG1:[0-9]+|[a-zA-Z$._-][a-zA-Z0-9$._-]*]]: !fir.boxchar<1> {fir.bindc_name = "c2"},
// CHECK-SAME:      %[[ARG2:[0-9]+|[a-zA-Z$._-][a-zA-Z0-9$._-]*]]: !fir.boxchar<1> {fir.bindc_name = "c3"},
// CHECK-SAME:      %[[ARG3:[0-9]+|[a-zA-Z$._-][a-zA-Z0-9$._-]*]]: !fir.boxchar<1> {fir.bindc_name = "c4"}) {
// CHECK:           %[[UNBOXCHAR_0:.*]]:2 = fir.unboxchar %[[ARG0]] : (!fir.boxchar<1>) -> (!fir.ref<!fir.char<1,?>>, index)
// CHECK:           %[[DECLARE_0:.*]]:2 = hlfir.declare %[[UNBOXCHAR_0]]#0 typeparams %[[UNBOXCHAR_0]]#1 uniq_name("_QFmax3Ec1") : (!fir.ref<!fir.char<1,?>>, index) -> (!fir.boxchar<1>, !fir.ref<!fir.char<1,?>>)
// CHECK:           %[[UNBOXCHAR_1:.*]]:2 = fir.unboxchar %[[ARG1]] : (!fir.boxchar<1>) -> (!fir.ref<!fir.char<1,?>>, index)
// CHECK:           %[[DECLARE_1:.*]]:2 = hlfir.declare %[[UNBOXCHAR_1]]#0 typeparams %[[UNBOXCHAR_1]]#1 uniq_name("_QFmax3Ec2") : (!fir.ref<!fir.char<1,?>>, index) -> (!fir.boxchar<1>, !fir.ref<!fir.char<1,?>>)
// CHECK:           %[[UNBOXCHAR_2:.*]]:2 = fir.unboxchar %[[ARG2]] : (!fir.boxchar<1>) -> (!fir.ref<!fir.char<1,?>>, index)
// CHECK:           %[[DECLARE_2:.*]]:2 = hlfir.declare %[[UNBOXCHAR_2]]#0 typeparams %[[UNBOXCHAR_2]]#1 uniq_name("_QFmax3Ec3") : (!fir.ref<!fir.char<1,?>>, index) -> (!fir.boxchar<1>, !fir.ref<!fir.char<1,?>>)
// CHECK:           %[[UNBOXCHAR_3:.*]]:2 = fir.unboxchar %[[ARG3]] : (!fir.boxchar<1>) -> (!fir.ref<!fir.char<1,?>>, index)
// CHECK:           %[[DECLARE_3:.*]]:2 = hlfir.declare %[[UNBOXCHAR_3]]#0 typeparams %[[UNBOXCHAR_3]]#1 uniq_name("_QFmax3Ec4") : (!fir.ref<!fir.char<1,?>>, index) -> (!fir.boxchar<1>, !fir.ref<!fir.char<1,?>>)
// CHECK:           %[[CMPI_0:.*]] = arith.cmpi uge, %[[UNBOXCHAR_1]]#1, %[[UNBOXCHAR_2]]#1 : index
// CHECK:           %[[SELECT_0:.*]] = arith.select %[[CMPI_0]], %[[UNBOXCHAR_1]]#1, %[[UNBOXCHAR_2]]#1 : index
// CHECK:           %[[CONVERT_0:.*]] = fir.convert %[[DECLARE_2]]#1 : (!fir.ref<!fir.char<1,?>>) -> !fir.ref<i8>
// CHECK:           %[[CONVERT_1:.*]] = fir.convert %[[DECLARE_1]]#1 : (!fir.ref<!fir.char<1,?>>) -> !fir.ref<i8>
// CHECK:           %[[CONVERT_2:.*]] = fir.convert %[[UNBOXCHAR_2]]#1 : (index) -> i64
// CHECK:           %[[CONVERT_3:.*]] = fir.convert %[[UNBOXCHAR_1]]#1 : (index) -> i64
// CHECK:           %[[CALL_0:.*]] = fir.call @_FortranACharacterCompareScalar1(%[[CONVERT_0]], %[[CONVERT_1]], %[[CONVERT_2]], %[[CONVERT_3]]) : (!fir.ref<i8>, !fir.ref<i8>, i64, i64) -> i32
// CHECK:           %[[CONSTANT_0:.*]] = arith.constant 0 : i32
// CHECK:           %[[CMPI_1:.*]] = arith.cmpi sgt, %[[CALL_0]], %[[CONSTANT_0]] : i32
// CHECK:           %[[SELECT_1:.*]] = arith.select %[[CMPI_1]], %[[DECLARE_2]]#1, %[[DECLARE_1]]#1 : !fir.ref<!fir.char<1,?>>
// CHECK:           %[[SELECT_2:.*]] = arith.select %[[CMPI_1]], %[[UNBOXCHAR_2]]#1, %[[UNBOXCHAR_1]]#1 : index
// CHECK:           %[[CMPI_2:.*]] = arith.cmpi uge, %[[SELECT_0]], %[[UNBOXCHAR_3]]#1 : index
// CHECK:           %[[SELECT_3:.*]] = arith.select %[[CMPI_2]], %[[SELECT_0]], %[[UNBOXCHAR_3]]#1 : index
// CHECK:           %[[CONVERT_4:.*]] = fir.convert %[[DECLARE_3]]#1 : (!fir.ref<!fir.char<1,?>>) -> !fir.ref<i8>
// CHECK:           %[[CONVERT_5:.*]] = fir.convert %[[SELECT_1]] : (!fir.ref<!fir.char<1,?>>) -> !fir.ref<i8>
// CHECK:           %[[CONVERT_6:.*]] = fir.convert %[[UNBOXCHAR_3]]#1 : (index) -> i64
// CHECK:           %[[CONVERT_7:.*]] = fir.convert %[[SELECT_2]] : (index) -> i64
// CHECK:           %[[CALL_1:.*]] = fir.call @_FortranACharacterCompareScalar1(%[[CONVERT_4]], %[[CONVERT_5]], %[[CONVERT_6]], %[[CONVERT_7]]) : (!fir.ref<i8>, !fir.ref<i8>, i64, i64) -> i32
// CHECK:           %[[CONSTANT_1:.*]] = arith.constant 0 : i32
// CHECK:           %[[CMPI_3:.*]] = arith.cmpi sgt, %[[CALL_1]], %[[CONSTANT_1]] : i32
// CHECK:           %[[SELECT_4:.*]] = arith.select %[[CMPI_3]], %[[DECLARE_3]]#1, %[[SELECT_1]] : !fir.ref<!fir.char<1,?>>
// CHECK:           %[[SELECT_5:.*]] = arith.select %[[CMPI_3]], %[[UNBOXCHAR_3]]#1, %[[SELECT_2]] : index
// CHECK:           %[[ALLOCMEM_0:.*]] = fir.allocmem !fir.char<1,?>(%[[SELECT_3]] : index) <{bindc_name = ".chrtmp", uniq_name = ""}>
// CHECK:           %[[CMPI_4:.*]] = arith.cmpi slt, %[[SELECT_3]], %[[SELECT_5]] : index
// CHECK:           %[[SELECT_6:.*]] = arith.select %[[CMPI_4]], %[[SELECT_3]], %[[SELECT_5]] : index
// CHECK:           %[[CONSTANT_2:.*]] = arith.constant 1 : i64
// CHECK:           %[[CONVERT_8:.*]] = fir.convert %[[SELECT_6]] : (index) -> i64
// CHECK:           %[[MULI_0:.*]] = arith.muli %[[CONSTANT_2]], %[[CONVERT_8]] : i64
// CHECK:           %[[CONVERT_9:.*]] = fir.convert %[[ALLOCMEM_0]] : (!fir.heap<!fir.char<1,?>>) -> !llvm.ptr
// CHECK:           %[[CONVERT_10:.*]] = fir.convert %[[SELECT_4]] : (!fir.ref<!fir.char<1,?>>) -> !llvm.ptr
// CHECK:           "llvm.intr.memmove"(%[[CONVERT_9]], %[[CONVERT_10]], %[[MULI_0]]) <{isVolatile = false}> : (!llvm.ptr, !llvm.ptr, i64) -> ()
// CHECK:           %[[CONSTANT_3:.*]] = arith.constant 1 : index
// CHECK:           %[[SUBI_0:.*]] = arith.subi %[[SELECT_3]], %[[CONSTANT_3]] : index
// CHECK:           %[[CONSTANT_4:.*]] = arith.constant 32 : i8
// CHECK:           %[[UNDEFINED_0:.*]] = fir.undefined !fir.char<1>
// CHECK:           %[[INSERT_VALUE_0:.*]] = fir.insert_value %[[UNDEFINED_0]], %[[CONSTANT_4]], [0 : index] : (!fir.char<1>, i8) -> !fir.char<1>
// CHECK:           %[[CONSTANT_5:.*]] = arith.constant 1 : index
// CHECK:           fir.do_loop %[[VAL_0:.*]] = %[[SELECT_6]] to %[[SUBI_0]] step %[[CONSTANT_5]] {
// CHECK:             %[[CONVERT_11:.*]] = fir.convert %[[ALLOCMEM_0]] : (!fir.heap<!fir.char<1,?>>) -> !fir.ref<!fir.array<?x!fir.char<1>>>
// CHECK:             %[[COORDINATE_OF_0:.*]] = fir.coordinate_of %[[CONVERT_11]], %[[VAL_0]] : (!fir.ref<!fir.array<?x!fir.char<1>>>, index) -> !fir.ref<!fir.char<1>>
// CHECK:             fir.store %[[INSERT_VALUE_0]] to %[[COORDINATE_OF_0]] : !fir.ref<!fir.char<1>>
// CHECK:           }
// CHECK:           %[[CONVERT_12:.*]] = fir.convert %[[ALLOCMEM_0]] : (!fir.heap<!fir.char<1,?>>) -> !fir.ref<!fir.char<1,?>>
// CHECK:           %[[DECLARE_4:.*]]:2 = hlfir.declare %[[CONVERT_12]] typeparams %[[SELECT_3]] uniq_name(".tmp.char_extremum") : (!fir.ref<!fir.char<1,?>>, index) -> (!fir.boxchar<1>, !fir.ref<!fir.char<1,?>>)
// CHECK:           %[[CONSTANT_6:.*]] = arith.constant true
// CHECK:           %[[UNDEFINED_1:.*]] = fir.undefined tuple<!fir.boxchar<1>, i1>
// CHECK:           %[[INSERT_VALUE_1:.*]] = fir.insert_value %[[UNDEFINED_1]], %[[CONSTANT_6]], [1 : index] : (tuple<!fir.boxchar<1>, i1>, i1) -> tuple<!fir.boxchar<1>, i1>
// CHECK:           %[[INSERT_VALUE_2:.*]] = fir.insert_value %[[INSERT_VALUE_1]], %[[DECLARE_4]]#0, [0 : index] : (tuple<!fir.boxchar<1>, i1>, !fir.boxchar<1>) -> tuple<!fir.boxchar<1>, i1>
// CHECK:           hlfir.assign %[[DECLARE_4]]#0 to %[[DECLARE_0]]#0 : !fir.boxchar<1>, !fir.boxchar<1>
// CHECK:           %[[BOX_ADDR_0:.*]] = fir.box_addr %[[DECLARE_4]]#0 : (!fir.boxchar<1>) -> !fir.heap<!fir.char<1,?>>
// CHECK:           fir.freemem %[[BOX_ADDR_0]] : !fir.heap<!fir.char<1,?>>
// CHECK:           return
// CHECK:         }

func.func @_QPmin3(%arg0: !fir.boxchar<1> {fir.bindc_name = "c1"}, %arg1: !fir.boxchar<1> {fir.bindc_name = "c2"}, %arg2: !fir.boxchar<1> {fir.bindc_name = "c3"}, %arg3: !fir.boxchar<1> {fir.bindc_name = "c4"}) {
  %0:2 = fir.unboxchar %arg0 : (!fir.boxchar<1>) -> (!fir.ref<!fir.char<1,?>>, index)
  %1:2 = hlfir.declare %0#0 typeparams %0#1 uniq_name("_QFmin3Ec1") : (!fir.ref<!fir.char<1,?>>, index) -> (!fir.boxchar<1>, !fir.ref<!fir.char<1,?>>)
  %2:2 = fir.unboxchar %arg1 : (!fir.boxchar<1>) -> (!fir.ref<!fir.char<1,?>>, index)
  %3:2 = hlfir.declare %2#0 typeparams %2#1 uniq_name("_QFmin3Ec2") : (!fir.ref<!fir.char<1,?>>, index) -> (!fir.boxchar<1>, !fir.ref<!fir.char<1,?>>)
  %4:2 = fir.unboxchar %arg2 : (!fir.boxchar<1>) -> (!fir.ref<!fir.char<1,?>>, index)
  %5:2 = hlfir.declare %4#0 typeparams %4#1 uniq_name("_QFmin3Ec3") : (!fir.ref<!fir.char<1,?>>, index) -> (!fir.boxchar<1>, !fir.ref<!fir.char<1,?>>)
  %6:2 = fir.unboxchar %arg3 : (!fir.boxchar<1>) -> (!fir.ref<!fir.char<1,?>>, index)
  %7:2 = hlfir.declare %6#0 typeparams %6#1 uniq_name("_QFmin3Ec4") : (!fir.ref<!fir.char<1,?>>, index) -> (!fir.boxchar<1>, !fir.ref<!fir.char<1,?>>)
  %8 = hlfir.char_extremum min, %3#0, %5#0, %7#0 : (!fir.boxchar<1>, !fir.boxchar<1>, !fir.boxchar<1>) -> !hlfir.expr<!fir.char<1,?>>
  hlfir.assign %8 to %1#0 : !hlfir.expr<!fir.char<1,?>>, !fir.boxchar<1>
  hlfir.destroy %8 : !hlfir.expr<!fir.char<1,?>>
  return
}

// CHECK-LABEL:   func.func @_QPmin3(
// CHECK-SAME:      %[[ARG0:[0-9]+|[a-zA-Z$._-][a-zA-Z0-9$._-]*]]: !fir.boxchar<1> {fir.bindc_name = "c1"},
// CHECK-SAME:      %[[ARG1:[0-9]+|[a-zA-Z$._-][a-zA-Z0-9$._-]*]]: !fir.boxchar<1> {fir.bindc_name = "c2"},
// CHECK-SAME:      %[[ARG2:[0-9]+|[a-zA-Z$._-][a-zA-Z0-9$._-]*]]: !fir.boxchar<1> {fir.bindc_name = "c3"},
// CHECK-SAME:      %[[ARG3:[0-9]+|[a-zA-Z$._-][a-zA-Z0-9$._-]*]]: !fir.boxchar<1> {fir.bindc_name = "c4"}) {
// CHECK:           %[[UNBOXCHAR_0:.*]]:2 = fir.unboxchar %[[ARG0]] : (!fir.boxchar<1>) -> (!fir.ref<!fir.char<1,?>>, index)
// CHECK:           %[[DECLARE_0:.*]]:2 = hlfir.declare %[[UNBOXCHAR_0]]#0 typeparams %[[UNBOXCHAR_0]]#1 uniq_name("_QFmin3Ec1") : (!fir.ref<!fir.char<1,?>>, index) -> (!fir.boxchar<1>, !fir.ref<!fir.char<1,?>>)
// CHECK:           %[[UNBOXCHAR_1:.*]]:2 = fir.unboxchar %[[ARG1]] : (!fir.boxchar<1>) -> (!fir.ref<!fir.char<1,?>>, index)
// CHECK:           %[[DECLARE_1:.*]]:2 = hlfir.declare %[[UNBOXCHAR_1]]#0 typeparams %[[UNBOXCHAR_1]]#1 uniq_name("_QFmin3Ec2") : (!fir.ref<!fir.char<1,?>>, index) -> (!fir.boxchar<1>, !fir.ref<!fir.char<1,?>>)
// CHECK:           %[[UNBOXCHAR_2:.*]]:2 = fir.unboxchar %[[ARG2]] : (!fir.boxchar<1>) -> (!fir.ref<!fir.char<1,?>>, index)
// CHECK:           %[[DECLARE_2:.*]]:2 = hlfir.declare %[[UNBOXCHAR_2]]#0 typeparams %[[UNBOXCHAR_2]]#1 uniq_name("_QFmin3Ec3") : (!fir.ref<!fir.char<1,?>>, index) -> (!fir.boxchar<1>, !fir.ref<!fir.char<1,?>>)
// CHECK:           %[[UNBOXCHAR_3:.*]]:2 = fir.unboxchar %[[ARG3]] : (!fir.boxchar<1>) -> (!fir.ref<!fir.char<1,?>>, index)
// CHECK:           %[[DECLARE_3:.*]]:2 = hlfir.declare %[[UNBOXCHAR_3]]#0 typeparams %[[UNBOXCHAR_3]]#1 uniq_name("_QFmin3Ec4") : (!fir.ref<!fir.char<1,?>>, index) -> (!fir.boxchar<1>, !fir.ref<!fir.char<1,?>>)
// CHECK:           %[[CMPI_0:.*]] = arith.cmpi uge, %[[UNBOXCHAR_1]]#1, %[[UNBOXCHAR_2]]#1 : index
// CHECK:           %[[SELECT_0:.*]] = arith.select %[[CMPI_0]], %[[UNBOXCHAR_1]]#1, %[[UNBOXCHAR_2]]#1 : index
// CHECK:           %[[CONVERT_0:.*]] = fir.convert %[[DECLARE_2]]#1 : (!fir.ref<!fir.char<1,?>>) -> !fir.ref<i8>
// CHECK:           %[[CONVERT_1:.*]] = fir.convert %[[DECLARE_1]]#1 : (!fir.ref<!fir.char<1,?>>) -> !fir.ref<i8>
// CHECK:           %[[CONVERT_2:.*]] = fir.convert %[[UNBOXCHAR_2]]#1 : (index) -> i64
// CHECK:           %[[CONVERT_3:.*]] = fir.convert %[[UNBOXCHAR_1]]#1 : (index) -> i64
// CHECK:           %[[CALL_0:.*]] = fir.call @_FortranACharacterCompareScalar1(%[[CONVERT_0]], %[[CONVERT_1]], %[[CONVERT_2]], %[[CONVERT_3]]) : (!fir.ref<i8>, !fir.ref<i8>, i64, i64) -> i32
// CHECK:           %[[CONSTANT_0:.*]] = arith.constant 0 : i32
// CHECK:           %[[CMPI_1:.*]] = arith.cmpi slt, %[[CALL_0]], %[[CONSTANT_0]] : i32
// CHECK:           %[[SELECT_1:.*]] = arith.select %[[CMPI_1]], %[[DECLARE_2]]#1, %[[DECLARE_1]]#1 : !fir.ref<!fir.char<1,?>>
// CHECK:           %[[SELECT_2:.*]] = arith.select %[[CMPI_1]], %[[UNBOXCHAR_2]]#1, %[[UNBOXCHAR_1]]#1 : index
// CHECK:           %[[CMPI_2:.*]] = arith.cmpi uge, %[[SELECT_0]], %[[UNBOXCHAR_3]]#1 : index
// CHECK:           %[[SELECT_3:.*]] = arith.select %[[CMPI_2]], %[[SELECT_0]], %[[UNBOXCHAR_3]]#1 : index
// CHECK:           %[[CONVERT_4:.*]] = fir.convert %[[DECLARE_3]]#1 : (!fir.ref<!fir.char<1,?>>) -> !fir.ref<i8>
// CHECK:           %[[CONVERT_5:.*]] = fir.convert %[[SELECT_1]] : (!fir.ref<!fir.char<1,?>>) -> !fir.ref<i8>
// CHECK:           %[[CONVERT_6:.*]] = fir.convert %[[UNBOXCHAR_3]]#1 : (index) -> i64
// CHECK:           %[[CONVERT_7:.*]] = fir.convert %[[SELECT_2]] : (index) -> i64
// CHECK:           %[[CALL_1:.*]] = fir.call @_FortranACharacterCompareScalar1(%[[CONVERT_4]], %[[CONVERT_5]], %[[CONVERT_6]], %[[CONVERT_7]]) : (!fir.ref<i8>, !fir.ref<i8>, i64, i64) -> i32
// CHECK:           %[[CONSTANT_1:.*]] = arith.constant 0 : i32
// CHECK:           %[[CMPI_3:.*]] = arith.cmpi slt, %[[CALL_1]], %[[CONSTANT_1]] : i32
// CHECK:           %[[SELECT_4:.*]] = arith.select %[[CMPI_3]], %[[DECLARE_3]]#1, %[[SELECT_1]] : !fir.ref<!fir.char<1,?>>
// CHECK:           %[[SELECT_5:.*]] = arith.select %[[CMPI_3]], %[[UNBOXCHAR_3]]#1, %[[SELECT_2]] : index
// CHECK:           %[[ALLOCMEM_0:.*]] = fir.allocmem !fir.char<1,?>(%[[SELECT_3]] : index) <{bindc_name = ".chrtmp", uniq_name = ""}>
// CHECK:           %[[CMPI_4:.*]] = arith.cmpi slt, %[[SELECT_3]], %[[SELECT_5]] : index
// CHECK:           %[[SELECT_6:.*]] = arith.select %[[CMPI_4]], %[[SELECT_3]], %[[SELECT_5]] : index
// CHECK:           %[[CONSTANT_2:.*]] = arith.constant 1 : i64
// CHECK:           %[[CONVERT_8:.*]] = fir.convert %[[SELECT_6]] : (index) -> i64
// CHECK:           %[[MULI_0:.*]] = arith.muli %[[CONSTANT_2]], %[[CONVERT_8]] : i64
// CHECK:           %[[CONVERT_9:.*]] = fir.convert %[[ALLOCMEM_0]] : (!fir.heap<!fir.char<1,?>>) -> !llvm.ptr
// CHECK:           %[[CONVERT_10:.*]] = fir.convert %[[SELECT_4]] : (!fir.ref<!fir.char<1,?>>) -> !llvm.ptr
// CHECK:           "llvm.intr.memmove"(%[[CONVERT_9]], %[[CONVERT_10]], %[[MULI_0]]) <{isVolatile = false}> : (!llvm.ptr, !llvm.ptr, i64) -> ()
// CHECK:           %[[CONSTANT_3:.*]] = arith.constant 1 : index
// CHECK:           %[[SUBI_0:.*]] = arith.subi %[[SELECT_3]], %[[CONSTANT_3]] : index
// CHECK:           %[[CONSTANT_4:.*]] = arith.constant 32 : i8
// CHECK:           %[[UNDEFINED_0:.*]] = fir.undefined !fir.char<1>
// CHECK:           %[[INSERT_VALUE_0:.*]] = fir.insert_value %[[UNDEFINED_0]], %[[CONSTANT_4]], [0 : index] : (!fir.char<1>, i8) -> !fir.char<1>
// CHECK:           %[[CONSTANT_5:.*]] = arith.constant 1 : index
// CHECK:           fir.do_loop %[[VAL_0:.*]] = %[[SELECT_6]] to %[[SUBI_0]] step %[[CONSTANT_5]] {
// CHECK:             %[[CONVERT_11:.*]] = fir.convert %[[ALLOCMEM_0]] : (!fir.heap<!fir.char<1,?>>) -> !fir.ref<!fir.array<?x!fir.char<1>>>
// CHECK:             %[[COORDINATE_OF_0:.*]] = fir.coordinate_of %[[CONVERT_11]], %[[VAL_0]] : (!fir.ref<!fir.array<?x!fir.char<1>>>, index) -> !fir.ref<!fir.char<1>>
// CHECK:             fir.store %[[INSERT_VALUE_0]] to %[[COORDINATE_OF_0]] : !fir.ref<!fir.char<1>>
// CHECK:           }
// CHECK:           %[[CONVERT_12:.*]] = fir.convert %[[ALLOCMEM_0]] : (!fir.heap<!fir.char<1,?>>) -> !fir.ref<!fir.char<1,?>>
// CHECK:           %[[DECLARE_4:.*]]:2 = hlfir.declare %[[CONVERT_12]] typeparams %[[SELECT_3]] uniq_name(".tmp.char_extremum") : (!fir.ref<!fir.char<1,?>>, index) -> (!fir.boxchar<1>, !fir.ref<!fir.char<1,?>>)
// CHECK:           %[[CONSTANT_6:.*]] = arith.constant true
// CHECK:           %[[UNDEFINED_1:.*]] = fir.undefined tuple<!fir.boxchar<1>, i1>
// CHECK:           %[[INSERT_VALUE_1:.*]] = fir.insert_value %[[UNDEFINED_1]], %[[CONSTANT_6]], [1 : index] : (tuple<!fir.boxchar<1>, i1>, i1) -> tuple<!fir.boxchar<1>, i1>
// CHECK:           %[[INSERT_VALUE_2:.*]] = fir.insert_value %[[INSERT_VALUE_1]], %[[DECLARE_4]]#0, [0 : index] : (tuple<!fir.boxchar<1>, i1>, !fir.boxchar<1>) -> tuple<!fir.boxchar<1>, i1>
// CHECK:           hlfir.assign %[[DECLARE_4]]#0 to %[[DECLARE_0]]#0 : !fir.boxchar<1>, !fir.boxchar<1>
// CHECK:           %[[BOX_ADDR_0:.*]] = fir.box_addr %[[DECLARE_4]]#0 : (!fir.boxchar<1>) -> !fir.heap<!fir.char<1,?>>
// CHECK:           fir.freemem %[[BOX_ADDR_0]] : !fir.heap<!fir.char<1,?>>
// CHECK:           return
// CHECK:         }
// CHECK:         func.func private @_FortranACharacterCompareScalar1(!fir.ref<i8>, !fir.ref<i8>, i64, i64) -> i32 attributes {fir.runtime}
