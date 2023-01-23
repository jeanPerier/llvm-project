// Test hlfir.copy_in and hlfir.copy_out operation parse, verify (no errors),
// and unparse.

// RUN: fir-opt %s | fir-opt | FileCheck %s

func.func @test_copy_in(%box: !fir.box<!fir.array<?xf64>>) {
  %0:2 = hlfir.copy_in %box : (!fir.box<!fir.array<?xf64>>) -> (!fir.ref<!fir.array<?xf64>>, i1)
  %1:2 = hlfir.copy_in %box : (!fir.box<!fir.array<?xf64>>) -> (!fir.box<!fir.array<?xf64>>, i1)
  %2:2 = hlfir.copy_in %box handle_null_box: (!fir.box<!fir.array<?xf64>>) -> (!fir.ref<!fir.array<?xf64>>, i1)
  %3:2 = hlfir.copy_in %box handle_null_box: (!fir.box<!fir.array<?xf64>>) -> (!fir.box<!fir.array<?xf64>>, i1)
  %4:2 = hlfir.copy_in %box handle_null_addr: (!fir.box<!fir.array<?xf64>>) -> (!fir.ref<!fir.array<?xf64>>, i1)
  %5:2 = hlfir.copy_in %box handle_null_addr: (!fir.box<!fir.array<?xf64>>) -> (!fir.box<!fir.array<?xf64>>, i1)
  return
}
// CHECK-LABEL:   func.func @test_copy_in(
// CHECK-SAME:    %[[VAL_0:.*]]: !fir.box<!fir.array<?xf64>>) {
// CHECK:  %[[VAL_1:.*]]:2 = hlfir.copy_in %[[VAL_0]]  : (!fir.box<!fir.array<?xf64>>) -> (!fir.ref<!fir.array<?xf64>>, i1)
// CHECK:  %[[VAL_2:.*]]:2 = hlfir.copy_in %[[VAL_0]]  : (!fir.box<!fir.array<?xf64>>) -> (!fir.box<!fir.array<?xf64>>, i1)
// CHECK:  %[[VAL_3:.*]]:2 = hlfir.copy_in %[[VAL_0]] handle_null_box : (!fir.box<!fir.array<?xf64>>) -> (!fir.ref<!fir.array<?xf64>>, i1)
// CHECK:  %[[VAL_4:.*]]:2 = hlfir.copy_in %[[VAL_0]] handle_null_box : (!fir.box<!fir.array<?xf64>>) -> (!fir.box<!fir.array<?xf64>>, i1)
// CHECK:  %[[VAL_5:.*]]:2 = hlfir.copy_in %[[VAL_0]] handle_null_addr : (!fir.box<!fir.array<?xf64>>) -> (!fir.ref<!fir.array<?xf64>>, i1)
// CHECK:  %[[VAL_6:.*]]:2 = hlfir.copy_in %[[VAL_0]] handle_null_addr : (!fir.box<!fir.array<?xf64>>) -> (!fir.box<!fir.array<?xf64>>, i1)

func.func @test_copy_out_with_raw_temp(%box: !fir.box<!fir.array<?xf64>>, %temp: !fir.ref<!fir.array<?xf64>>, %was_copied: i1) {
  hlfir.copy_out %temp, %was_copied : (!fir.ref<!fir.array<?xf64>>, i1) -> ()
  hlfir.copy_out %temp, %was_copied to %box : (!fir.ref<!fir.array<?xf64>>, i1, !fir.box<!fir.array<?xf64>>) -> ()
  return
}
// CHECK-LABEL:   func.func @test_copy_out_with_raw_temp(
// CHECK-SAME:    %[[VAL_0:.*]]: !fir.box<!fir.array<?xf64>>,
// CHECK-SAME:    %[[VAL_1:.*]]: !fir.ref<!fir.array<?xf64>>,
// CHECK-SAME:    %[[VAL_2:.*]]: i1) {
// CHECK:  hlfir.copy_out %[[VAL_1]], %[[VAL_2]] : (!fir.ref<!fir.array<?xf64>>, i1) -> ()
// CHECK:  hlfir.copy_out %[[VAL_1]], %[[VAL_2]] to %[[VAL_0]] : (!fir.ref<!fir.array<?xf64>>, i1, !fir.box<!fir.array<?xf64>>) -> ()

func.func @test_copy_out_with_box_temp(%box: !fir.box<!fir.array<?xf64>>, %temp: !fir.box<!fir.array<?xf64>>, %was_copied: i1) {
  hlfir.copy_out %temp, %was_copied : (!fir.box<!fir.array<?xf64>>, i1) -> ()
  hlfir.copy_out %temp, %was_copied to %box : (!fir.box<!fir.array<?xf64>>, i1, !fir.box<!fir.array<?xf64>>) -> ()
  return
}
// CHECK-LABEL:   func.func @test_copy_out_with_box_temp(
// CHECK-SAME:    %[[VAL_0:[^:]*]]: !fir.box<!fir.array<?xf64>>,
// CHECK-SAME:    %[[VAL_1:.*]]: !fir.box<!fir.array<?xf64>>,
// CHECK-SAME:    %[[VAL_2:.*]]: i1) {
// CHECK:  hlfir.copy_out %[[VAL_1]], %[[VAL_2]] : (!fir.box<!fir.array<?xf64>>, i1) -> ()
// CHECK:  hlfir.copy_out %[[VAL_1]], %[[VAL_2]] to %[[VAL_0]] : (!fir.box<!fir.array<?xf64>>, i1, !fir.box<!fir.array<?xf64>>) -> ()
