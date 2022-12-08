! Test plugging of the intrinsic lowering framework into HLFIR lowering.
! RUN: bbc -emit-fir -hlfir -o - %s 2>&1 | FileCheck %s

subroutine test_as_value(x, y)
  real :: x, y
  x = acos(y)
end subroutine
! CHECK-LABEL: func.func @_QPtest_as_value(
! CHECK:  %[[VAL_4:.*]] = fir.load %{{.*}}: !fir.ref<f32>
! CHECK:  %[[VAL_5:.*]] = fir.call @acosf(%[[VAL_4]]) {{.*}} : (f32) -> f32
! CHECK:  hlfir.assign %[[VAL_5]] to %{{.*}} : f32, !fir.ref<f32>

subroutine test_as_addr(x, y)
  character(*) :: x, y
  y = trim(x)
end subroutine
! CHECK-LABEL: func.func @_QPtest_as_addr(
! CHECK:  %[[VAL_2:.*]] = fir.alloca !fir.box<!fir.heap<!fir.char<1,?>>>
! CHECK:  %[[VAL_4:.*]]:2 = hlfir.declare %[[VAL_3:.*]]#0 typeparams %[[VAL_3]]#1 {uniq_name = "_QFtest_as_addrEx"} : (!fir.ref<!fir.char<1,?>>, index) -> (!fir.boxchar<1>, !fir.ref<!fir.char<1,?>>)
! CHECK:  %[[VAL_7:.*]] = fir.embox %[[VAL_4]]#1 typeparams %[[VAL_3]]#1 : (!fir.ref<!fir.char<1,?>>, index) -> !fir.box<!fir.char<1,?>>
! CHECK:  %[[VAL_13:.*]] = fir.convert %[[VAL_2]] : (!fir.ref<!fir.box<!fir.heap<!fir.char<1,?>>>>) -> !fir.ref<!fir.box<none>>
! CHECK:  %[[VAL_14:.*]] = fir.convert %[[VAL_7]] : (!fir.box<!fir.char<1,?>>) -> !fir.box<none>
! CHECK:  %[[VAL_16:.*]] = fir.call @_FortranATrim(%[[VAL_13]], %[[VAL_14]], %{{.*}}, %{{.*}}) fastmath<contract> : (!fir.ref<!fir.box<none>>, !fir.box<none>, !fir.ref<i8>, i32) -> none
! CHECK:  %[[VAL_17:.*]] = fir.load %[[VAL_2]] : !fir.ref<!fir.box<!fir.heap<!fir.char<1,?>>>>
! CHECK:  %[[VAL_18:.*]] = fir.box_elesize %[[VAL_17]] : (!fir.box<!fir.heap<!fir.char<1,?>>>) -> index
! CHECK:  %[[VAL_19:.*]] = fir.box_addr %[[VAL_17]] : (!fir.box<!fir.heap<!fir.char<1,?>>>) -> !fir.heap<!fir.char<1,?>>
! CHECK:  %[[VAL_20:.*]]:2 = hlfir.declare %[[VAL_19]] typeparams %[[VAL_18]] {uniq_name = ".tmp.intrinsic_result"} : (!fir.heap<!fir.char<1,?>>, index) -> (!fir.boxchar<1>, !fir.heap<!fir.char<1,?>>)
! CHECK:  hlfir.assign %[[VAL_20]]#0 to %{{.*}} : !fir.boxchar<1>, !fir.boxchar<1>
! CHECK:  fir.freemem %[[VAL_19]] : !fir.heap<!fir.char<1,?>>

subroutine test_as_box(x, y)
  integer :: x(100), y
  y = iany(x)
end subroutine
! CHECK-LABEL: func.func @_QPtest_as_box(
! CHECK:  %[[VAL_2:.*]] = arith.constant 100 : index
! CHECK:  %[[VAL_4:.*]]:2 = hlfir.declare %{{.*}}x"
! CHECK:  %[[VAL_6:.*]] = fir.shape %[[VAL_2]] : (index) -> !fir.shape<1>
! CHECK:  %[[VAL_7:.*]] = fir.embox %[[VAL_4]]#1(%[[VAL_6]]) : (!fir.ref<!fir.array<100xi32>>, !fir.shape<1>) -> !fir.box<!fir.array<100xi32>>
! CHECK:  %[[VAL_12:.*]] = fir.convert %[[VAL_7]] : (!fir.box<!fir.array<100xi32>>) -> !fir.box<none>
! CHECK:  %[[VAL_16:.*]] = fir.call @_FortranAIAny4(%[[VAL_12]], %{{.*}}, %{{.*}}, %{{.*}}, %{{.*}}) {{.*}} : (!fir.box<none>, !fir.ref<i8>, i32, i32, !fir.box<none>) -> i32
! CHECK:  hlfir.assign %[[VAL_16]] to %{{.*}} : i32, !fir.ref<i32>
