! RUN: bbc -emit-fir %s -o - | FileCheck %s

! CHECK-LABEL: system_clock_test
subroutine system_clock_test()
  integer(4) :: c
  integer(8) :: m
  real :: r
  ! CHECK-DAG: %[[c:.*]] = fir.alloca i32 {bindc_name = "c"
  ! CHECK-DAG: %[[m:.*]] = fir.alloca i64 {bindc_name = "m"
  ! CHECK-DAG: %[[r:.*]] = fir.alloca f32 {bindc_name = "r"
  ! CHECK: %[[c4:.*]] = arith.constant 4 : i32
  ! CHECK: %[[Count:.*]] = fir.call @_FortranASystemClockCount(%[[c4]]) : (i32) -> i64
  ! CHECK: %[[Count1:.*]] = fir.convert %[[Count]] : (i64) -> i32
  ! CHECK: fir.store %[[Count1]] to %[[c]] : !fir.ref<i32>
  ! CHECK: %[[c8:.*]] = arith.constant 8 : i32
  ! CHECK: %[[Rate:.*]] = fir.call @_FortranASystemClockCountRate(%[[c8]]) : (i32) -> i64
  ! CHECK: %[[Rate1:.*]] = fir.convert %[[Rate]] : (i64) -> f32
  ! CHECK: fir.store %[[Rate1]] to %[[r]] : !fir.ref<f32>
  ! CHECK: %[[c8_2:.*]] = arith.constant 8 : i32
  ! CHECK: %[[Max:.*]] = fir.call @_FortranASystemClockCountMax(%[[c8_2]]) : (i32) -> i64
  ! CHECK: fir.store %[[Max]] to %[[m]] : !fir.ref<i64>
  call system_clock(c, r, m)
! print*, c, r, m
  ! CHECK-NOT: fir.call
  ! CHECK: %[[c8_3:.*]] = arith.constant 8 : i32
  ! CHECK: %[[Rate:.*]] = fir.call @_FortranASystemClockCountRate(%[[c8_3]]) : (i32) -> i64
  ! CHECK: fir.store %[[Rate]] to %[[m]] : !fir.ref<i64>
  call system_clock(count_rate=m)
  ! CHECK-NOT: fir.call
! print*, m
end subroutine

! CHECK-LABEL: system_clock_optional_test
subroutine system_clock_optional_test()
  ! CHECK-DAG: %[[VAL_0:[0-9]*]] = fir.alloca i64 {bindc_name = "count", uniq_name = "_QFsystem_clock_optional_testEcount"}
  ! CHECK-DAG: %[[VAL_1:[0-9]*]] = fir.alloca !fir.box<!fir.heap<i64>> {bindc_name = "count_max", uniq_name = "_QFsystem_clock_optional_testEcount_max"}
  ! CHECK-DAG: %[[VAL_2:[0-9]*]] = fir.alloca !fir.heap<i64> {uniq_name = "_QFsystem_clock_optional_testEcount_max.addr"}
  ! CHECK-DAG: %[[VAL_4:[0-9]*]] = fir.alloca !fir.box<!fir.ptr<i64>> {bindc_name = "count_rate", uniq_name = "_QFsystem_clock_optional_testEcount_rate"}
  ! CHECK-DAG: %[[VAL_5:[0-9]*]] = fir.alloca !fir.ptr<i64> {uniq_name = "_QFsystem_clock_optional_testEcount_rate.addr"}
  ! CHECK-DAG: %[[VAL_7:[0-9]*]] = fir.alloca i64 {bindc_name = "count_rate_", fir.target, uniq_name = "_QFsystem_clock_optional_testEcount_rate_"}
  ! CHECK-DAG: %[[VAL_9:[0-9]*]] = fir.allocmem i64 {uniq_name = "_QFsystem_clock_optional_testEcount_max.alloc"}
  ! CHECK: fir.store %[[VAL_9]] to %[[VAL_2]] : !fir.ref<!fir.heap<i64>>
  integer(8) :: count
  integer(8), target :: count_rate_
  integer(8), pointer :: count_rate
  integer(8), allocatable :: count_max

  ! CHECK: %[[VAL_10:[0-9]*]] = fir.load %[[VAL_5]] : !fir.ref<!fir.ptr<i64>>
  ! CHECK: %[[VAL_11:[0-9]*]] = fir.load %[[VAL_2]] : !fir.ref<!fir.heap<i64>>
  ! CHECK: %[[VAL_12:[0-9]*]] = fir.call @_FortranASystemClockCount(%c8{{.*}}) : (i32) -> i64
  ! CHECK: fir.store %[[VAL_12]] to %[[VAL_0]] : !fir.ref<i64>
  ! CHECK: %[[VAL_13:[0-9]*]] = fir.convert %[[VAL_10]] : (!fir.ptr<i64>) -> i64
  ! CHECK: %[[VAL_14:[0-9]*]] = arith.cmpi ne, %[[VAL_13]], %c0{{.*}} : i64
  ! CHECK: fir.if %[[VAL_14]] {
  ! CHECK:   %[[VAL_46:[0-9]*]] = fir.call @_FortranASystemClockCountRate(%c8{{.*}}) : (i32) -> i64
  ! CHECK:   fir.store %[[VAL_46]] to %[[VAL_10]] : !fir.ptr<i64>
  ! CHECK: }
  ! CHECK: %[[VAL_15:[0-9]*]] = fir.convert %[[VAL_11]] : (!fir.heap<i64>) -> i64
  ! CHECK: %[[VAL_16:[0-9]*]] = arith.cmpi ne, %[[VAL_15]], %c0{{.*}} : i64
  ! CHECK: fir.if %[[VAL_16]] {
  ! CHECK:   %[[VAL_46:[0-9]*]] = fir.call @_FortranASystemClockCountMax(%c8{{.*}}) : (i32) -> i64
  ! CHECK:   fir.store %[[VAL_46]] to %[[VAL_11]] : !fir.heap<i64>
  ! CHECK: }
  count_rate => count_rate_
  allocate(count_max)
  call system_clock(count, count_rate, count_max)
  print*, count, count_rate, count_max

  ! CHECK:                    = fir.load %[[VAL_5]] : !fir.ref<!fir.ptr<i64>>
  ! CHECK: %[[VAL_33:[0-9]*]] = fir.load %[[VAL_5]] : !fir.ref<!fir.ptr<i64>>
  ! CHECK: %[[VAL_34:[0-9]*]] = fir.load %[[VAL_2]] : !fir.ref<!fir.heap<i64>>
  ! CHECK: %[[VAL_35:[0-9]*]] = fir.call @_FortranASystemClockCount(%c8{{.*}}) : (i32) -> i64
  ! CHECK: fir.store %[[VAL_35]] to %[[VAL_0]] : !fir.ref<i64>
  ! CHECK: %[[VAL_36:[0-9]*]] = fir.convert %[[VAL_33]] : (!fir.ptr<i64>) -> i64
  ! CHECK: %[[VAL_37:[0-9]*]] = arith.cmpi ne, %[[VAL_36]], %c0{{.*}} : i64
  ! CHECK: fir.if %[[VAL_37]] {
  ! CHECK:   %[[VAL_46:[0-9]*]] = fir.call @_FortranASystemClockCountRate(%c8{{.*}}) : (i32) -> i64
  ! CHECK:   fir.store %[[VAL_46]] to %[[VAL_33]] : !fir.ptr<i64>
  ! CHECK: }
  ! CHECK: %[[VAL_38:[0-9]*]] = fir.convert %[[VAL_34]] : (!fir.heap<i64>) -> i64
  ! CHECK: %[[VAL_39:[0-9]*]] = arith.cmpi ne, %[[VAL_38]], %c0{{.*}} : i64
  ! CHECK: fir.if %[[VAL_39]] {
  ! CHECK:   %[[VAL_46:[0-9]*]] = fir.call @_FortranASystemClockCountMax(%c8{{.*}}) : (i32) -> i64
  ! CHECK:   fir.store %[[VAL_46]] to %[[VAL_34]] : !fir.heap<i64>
  ! CHECK: }
  count = 0
  count_rate => null()
  deallocate(count_max)
  call system_clock(count, count_rate, count_max)
  print*, count
end
