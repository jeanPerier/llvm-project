! Test OpenACC directives referencing a COMMON block from a USEd module when
! none of the COMMON block's member variables are referenced by name.

! RUN: %flang_fc1 -fopenacc -emit-hlfir %s -o - | FileCheck %s

module acc_mod_common
  integer :: a, b
  common /cb/ a, b
  integer, private :: priv_c
  common /cb_priv/ priv_c
  integer :: unreferenced
end module

! CHECK-LABEL: func.func @_QPtest_update()
! CHECK-NOT: @_QMacc_mod_commonEunreferenced
! CHECK: %[[CB:.*]] = fir.address_of(@cb_) : !fir.ref<!fir.array<8xi8>>
! CHECK-DAG: hlfir.declare %{{.*}} storage(%[[CB]][0]) uniq_name("_QMacc_mod_commonEa")
! CHECK-DAG: hlfir.declare %{{.*}} storage(%[[CB]][4]) uniq_name("_QMacc_mod_commonEb")
! CHECK: acc.update_device varPtr(%[[CB]] : !fir.ref<!fir.array<8xi8>>) structured(false) name("cb") -> !fir.ref<!fir.array<8xi8>>
! CHECK: acc.update dataOperands(
subroutine test_update()
  use acc_mod_common
  !$acc update device(/cb/)
end subroutine

! CHECK-LABEL: func.func @_QPtest_parallel_copy()
! CHECK-NOT: @_QMacc_mod_commonEunreferenced
! CHECK: %[[CB2:.*]] = fir.address_of(@cb_) : !fir.ref<!fir.array<8xi8>>
! CHECK: %[[COPYIN:.*]] = acc.copyin varPtr(%[[CB2]] : !fir.ref<!fir.array<8xi8>>) dataClause(acc_copy) name("cb") -> !fir.ref<!fir.array<8xi8>>
! CHECK: acc.parallel dataOperands(%[[COPYIN]] : !fir.ref<!fir.array<8xi8>>) {
! CHECK-DAG: hlfir.declare %{{.*}} uniq_name("_QMacc_mod_commonEa")
! CHECK-DAG: hlfir.declare %{{.*}} uniq_name("_QMacc_mod_commonEb")
! CHECK: acc.yield
subroutine test_parallel_copy()
  use acc_mod_common
  !$acc parallel copy(/cb/)
  !$acc end parallel
end subroutine

! CHECK-LABEL: func.func @_QPtest_private_member()
! CHECK: %[[CBP:.*]] = fir.address_of(@cb_priv_) : !fir.ref<!fir.array<4xi8>>
! CHECK: hlfir.declare %{{.*}} storage(%[[CBP]][0]) uniq_name("_QMacc_mod_commonEpriv_c")
! CHECK: %[[CP_PRIV:.*]] = acc.copyin varPtr(%[[CBP]] : !fir.ref<!fir.array<4xi8>>) dataClause(acc_copy) name("cb_priv") -> !fir.ref<!fir.array<4xi8>>
! CHECK: acc.parallel dataOperands(%[[CP_PRIV]] : !fir.ref<!fir.array<4xi8>>) {
! CHECK: hlfir.declare %{{.*}} uniq_name("_QMacc_mod_commonEpriv_c")
! CHECK: acc.yield
subroutine test_private_member()
  use acc_mod_common, only:
  !$acc parallel copy(/cb_priv/)
  !$acc end parallel
end subroutine

! CHECK-LABEL: func.func @_QPtest_block_common()
! CHECK: %[[CBB:.*]] = fir.address_of(@cb_) : !fir.ref<!fir.array<8xi8>>
! CHECK: acc.update_device varPtr(%[[CBB]] : !fir.ref<!fir.array<8xi8>>) structured(false) name("cb") -> !fir.ref<!fir.array<8xi8>>
subroutine test_block_common()
  block
    use acc_mod_common
    !$acc update device(/cb/)
  end block
end subroutine
