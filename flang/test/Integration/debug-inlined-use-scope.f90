! RUN: %flang_fc1 -emit-fir -debug-info-kind=standalone -mmlir --mlir-print-debuginfo %s -o - | fir-opt --inline-all --inline --mlir-print-debuginfo | FileCheck %s

module m
  integer :: a
end module
subroutine callee()
  use m
  call sink(a)
end subroutine
subroutine caller()
  call callee()
end subroutine

! The inlined call to sink still carries its callee's USE context.
! Its call site carries the caller's own scope.
! CHECK-LABEL: func.func @_QPcaller()
! CHECK: fir.call @_QPsink({{.*}}) {{.*}} loc(#[[INLINED:loc[0-9]*]])
! CHECK-DAG: #[[CALLEE:scope[0-9]*]] = #fir.scope<procedure "callee", uses = [#fir.use<"m",
! CHECK-DAG: #[[CALLER:scope[0-9]*]] = #fir.scope<procedure "caller", #{{loc[0-9]*}}>
! CHECK-DAG: #[[INLINED]] = loc(callsite(#[[CALLEE_LOC:loc[0-9]*]] at #[[CALLER_LOC:loc[0-9]*]]))
! CHECK-DAG: #[[CALLEE_LOC]] = loc(fused<#[[CALLEE]]>[#{{loc[0-9]*}}])
! CHECK-DAG: #[[CALLER_LOC]] = loc(fused<#[[CALLER]]>[#{{loc[0-9]*}}])
