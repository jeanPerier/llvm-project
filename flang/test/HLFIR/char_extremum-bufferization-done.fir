// Test hlfir.concat operation lowering to operations operating on memory.

// RUN: fir-opt %s -bufferize-hlfir | FileCheck %s
// XFAIL: *

// Along with the TODOs, we'll still need to go and change all of the RHS stuff

func.func @_QPmax1(%arg0: !fir.boxchar<1> {fir.bindc_name = "c1"}, %arg1: !fir.boxchar<1> {fir.bindc_name = "c2"}, %arg2: !fir.boxchar<1> {fir.bindc_name = "c3"}) {
  %0:2 = fir.unboxchar %arg0 : (!fir.boxchar<1>) -> (!fir.ref<!fir.char<1,?>>, index)
  %1:2 = hlfir.declare %0#0 typeparams %0#1 {uniq_name = "_QFmax1Ec1"} : (!fir.ref<!fir.char<1,?>>, index) -> (!fir.boxchar<1>, !fir.ref<!fir.char<1,?>>)
  %2:2 = fir.unboxchar %arg1 : (!fir.boxchar<1>) -> (!fir.ref<!fir.char<1,?>>, index)
  %3:2 = hlfir.declare %2#0 typeparams %2#1 {uniq_name = "_QFmax1Ec2"} : (!fir.ref<!fir.char<1,?>>, index) -> (!fir.boxchar<1>, !fir.ref<!fir.char<1,?>>)
  %4:2 = fir.unboxchar %arg2 : (!fir.boxchar<1>) -> (!fir.ref<!fir.char<1,?>>, index)
  %5:2 = hlfir.declare %4#0 typeparams %4#1 {uniq_name = "_QFmax1Ec3"} : (!fir.ref<!fir.char<1,?>>, index) -> (!fir.boxchar<1>, !fir.ref<!fir.char<1,?>>)
  %6 = hlfir.char_extremum max, %3#0, %5#0 : (!fir.boxchar<1>, !fir.boxchar<1>) -> !hlfir.expr<!fir.char<1,?>>
  hlfir.assign %6 to %1#0 : !hlfir.expr<!fir.char<1,?>>, !fir.boxchar<1>
  hlfir.destroy %6 : !hlfir.expr<!fir.char<1,?>>
  return
}

// CHECK-LABEL:   func.func @_QPmax1(
// CHECK-SAME:    %[[ARG0:[^:]*]]: !fir.boxchar<1> {{[^,]*}},
// CHECK-SAME:    %[[ARG1:[^:]*]]: !fir.boxchar<1> {{[^,].*}},
// CHECK-SAME:    %[[ARG2:[^:]*]]: !fir.boxchar<1> {{[^)].*}}) {
// CHECK: %[[ARG0_UNBOXED:.*]]:2 = fir.unboxchar %[[ARG0]] : (!fir.boxchar<1>) -> (!fir.ref<!fir.char<1,?>>, index)
// CHECK: %[[ARG0_VAL:[^:]*]]:2 = hlfir.declare %[[ARG0_UNBOXED]]#0 typeparams %[[ARG0_UNBOXED]]#1 {uniq_name = "_QFmax1Ec1"} : (!fir.ref<!fir.char<1,?>>, index) -> (!fir.boxchar<1>, !fir.ref<!fir.char<1,?>>)
// CHECK: %[[ARG1_UNBOXED:[^:]*]]:2 = fir.unboxchar %[[ARG1]] : (!fir.boxchar<1>) -> (!fir.ref<!fir.char<1,?>>, index)
// CHECK: %[[ARG1_VAL:[^:]*]]:2 = hlfir.declare %[[ARG1_UNBOXED]]#0 typeparams %[[ARG1_UNBOXED]]#1 {uniq_name = "_QFmax1Ec2"} : (!fir.ref<!fir.char<1,?>>, index) -> (!fir.boxchar<1>, !fir.ref<!fir.char<1,?>>)
// CHECK: %[[ARG2_UNBOXED:[^:]*]]:2 = fir.unboxchar %[[ARG2]] : (!fir.boxchar<1>) -> (!fir.ref<!fir.char<1,?>>, index)
// CHECK: %[[ARG2_VAL:[^:]*]]:2 = hlfir.declare %[[ARG2_UNBOXED]]#0 typeparams %[[ARG2_UNBOXED]]#1 {uniq_name = "_QFmax1Ec3"} : (!fir.ref<!fir.char<1,?>>, index) -> (!fir.boxchar<1>, !fir.ref<!fir.char<1,?>>)
// CHECK: %[[BIGGEST_LEN_CMP:[^:]*]] = arith.cmpi uge, %[[ARG1_UNBOXED]]#1, %[[ARG2_UNBOXED]]#1 : index
// CHECK: %[[BIGGEST_LEN:[^:]*]] = arith.select %[[BIGGEST_LEN_CMP]], %[[ARG1_UNBOXED]]#1, %[[ARG2_UNBOXED]]#1 : index
// CHECK: %[[CAST_ARG2_BUF_TO_UNKNOWN:[^:]*]] = fir.convert %[[ARG2_VAL]]#1 : (!fir.ref<!fir.char<1,?>>) -> !fir.ref<i8>
// CHECK: %[[CAST_ARG1_BUF_TO_UNKNOWN:[^:]*]] = fir.convert %[[ARG1_VAL]]#1 : (!fir.ref<!fir.char<1,?>>) -> !fir.ref<i8>
// CHECK: %[[CAST_ARG2_LEN_TO_LENTYPE:[^:]*]] = fir.convert %[[ARG2_UNBOXED]]#1 : (index) -> i64
// CHECK: %[[CAST_ARG1_LEN_TO_LENTYPE:[^:]*]] = fir.convert %[[ARG1_UNBOXED]]#1 : (index) -> i64
// CHECK: %[[LEX_CMP_RT:[^:]*]] = fir.call @_FortranACharacterCompareScalar1(%[[CAST_ARG2_BUF_TO_UNKNOWN]], %[[CAST_ARG1_BUF_TO_UNKNOWN]], %[[CAST_ARG2_LEN_TO_LENTYPE]], %[[CAST_ARG1_LEN_TO_LENTYPE]]) : (!fir.ref<i8>, !fir.ref<i8>, i64, i64) -> i32
// CHECK: %[[C0_I32:[^:]*]] = arith.constant 0 : i32
// CHECK: %[[LEX_CMP:[^:]*]] = arith.cmpi sgt, %[[LEX_CMP_RT]], %[[C0_I32]] : i32
// CHECK: %[[LEX_CMP_RES_BUF:[^:]*]] = arith.select %[[LEX_CMP]], %[[ARG2_VAL]]#1, %[[ARG1_VAL]]#1 : !fir.ref<!fir.char<1,?>>
// CHECK: %[[LEX_CMP_RES_LEN:[^:]*]] = arith.select %[[LEX_CMP]], %[[ARG2_UNBOXED]]#1, %[[ARG1_UNBOXED]]#1 : index
// CHECK: %[[TMP_BUF:[^:]*]] = fir.alloca !fir.char<1,?>(%[[BIGGEST_LEN]] : index) {bindc_name = ".chrtmp"}
// CHECK: %[[LHS_SMALLER_CMP:[^:]*]] = arith.cmpi slt, %[[BIGGEST_LEN]], %[[LEX_CMP_RES_LEN]] : index
// CHECK: %[[LHS_SMALLER_RES:[^:]*]] = arith.select %[[LHS_SMALLER_CMP]], %[[BIGGEST_LEN]], %[[LEX_CMP_RES_LEN]] : index
// CHECK: %[[C1_I64:[^:]*]] = arith.constant 1 : i64
// CHECK: %[[CAST_CONSTANT_1:[^:]*]] = fir.convert %[[LHS_SMALLER_RES]] : (index) -> i64
// CHECK: %[[NUM_BYTES_TO_MOVE:[^:]*]] = arith.muli %[[C1_I64]], %[[CAST_CONSTANT_1]] : i64
// CHECK: %[[FALSE:[^:]*]] = arith.constant false
// CHECK: %[[DST_BUF:[^:]*]] = fir.convert %[[TMP_BUF]] : (!fir.ref<!fir.char<1,?>>) -> !fir.ref<i8>
// CHECK: %[[SRC_BUF:[^:]*]] = fir.convert %[[LEX_CMP_RES_BUF]] : (!fir.ref<!fir.char<1,?>>) -> !fir.ref<i8>
// CHECK: fir.call @llvm.memmove.p0.p0.i64(%[[DST_BUF]], %[[SRC_BUF]], %[[NUM_BYTES_TO_MOVE]], %[[FALSE]]) : (!fir.ref<i8>, !fir.ref<i8>, i64, i1) -> ()
// CHECK: %[[C1:[^:]*]] = arith.constant 1 : index
// CHECK: %[[PAD_LENGTH:[^:]*]] = arith.subi %[[BIGGEST_LEN]], %[[C1]] : index
// CHECK: %[[C32_I8:[^:]*]] = arith.constant 32 : i8
// CHECK: %[[BLANK_CONSTANT:[^:]*]] = fir.undefined !fir.char<1>
// CHECK: %[[PAD_CHAR:[^:]*]] = fir.insert_value %[[BLANK_CONSTANT]], %[[C32_I8]], [0 : index] : (!fir.char<1>, i8) -> !fir.char<1>
// CHECK: %[[C1_0:[^:]*]] = arith.constant 1 : index
// CHECK: fir.do_loop %arg3 = %[[LHS_SMALLER_RES]] to %[[PAD_LENGTH]] step %[[C1_0]] {
// CHECK:   %[[FINAL_DST:[^:]*]] = fir.convert %[[TMP_BUF]] : (!fir.ref<!fir.char<1,?>>) -> !fir.ref<!fir.array<?x!fir.char<1>>>
// CHECK:   %[[FINAL_DST_POS:[^:]*]] = fir.coordinate_of %[[FINAL_DST]], %arg3 : (!fir.ref<!fir.array<?x!fir.char<1>>>, index) -> !fir.ref<!fir.char<1>>
// CHECK:   fir.store %[[PAD_CHAR]] to %[[FINAL_DST_POS]] : !fir.ref<!fir.char<1>>
// CHECK: %[[FINAL_DEST_SSA:[^:]*]]:2 = hlfir.declare %[[TMP_BUF]] typeparams %[[BIGGEST_LEN]] {uniq_name = "tmp"} : (!fir.ref<!fir.char<1,?>>, index) -> (!fir.boxchar<1>, !fir.ref<!fir.char<1,?>>)
// CHECK: %[[FALSE_1:[^:]*]] = arith.constant false
// CHECK: %[[FINAL_CLEANUP_0:[^:]*]] = fir.undefined tuple<!fir.boxchar<1>, i1>
// CHECK: %[[FINAL_CLEANUP_1:[^:]*]] = fir.insert_value %[[FINAL_CLEANUP_0]], %[[FALSE_1]], [1 : index] : (tuple<!fir.boxchar<1>, i1>, i1) -> tuple<!fir.boxchar<1>, i1>
// CHECK: %[[FINAL_CLEANUP_2:[^:]*]] = fir.insert_value %[[FINAL_CLEANUP_1]], %[[FINAL_DEST_SSA]]#0, [0 : index] : (tuple<!fir.boxchar<1>, i1>, !fir.boxchar<1>) -> tuple<!fir.boxchar<1>, i1>
// CHECK: hlfir.assign %[[FINAL_DEST_SSA]]#0 to %[[ARG0_VAL]]#0 : !fir.boxchar<1>, !fir.boxchar<1>
