! A generic can hide a procedure pointer of the same name. A USE of the generic
! then makes the procedure pointer accessible, unlike a USE of a generic that
! hides a subprogram or that has the name of a derived type.

! RUN: %flang_fc1 -emit-hlfir -debug-info-kind=standalone -mmlir --mlir-print-debuginfo %s -o - | FileCheck %s
! RUN: %flang_fc1 -emit-llvm -debug-info-kind=standalone %s -o - | FileCheck %s --check-prefix=LLVM

module mg
  procedure(proc2), pointer :: p
  interface p
    procedure proc1, p
  end interface
  interface f
    procedure f, proc1
  end interface
  type t
  end type
  interface t
    procedure mkt
  end interface
  integer :: other = 3
contains
  subroutine f(x)
    real :: x
  end subroutine
  subroutine proc1(i)
    integer :: i
  end subroutine
  subroutine proc2(x)
    real :: x
  end subroutine
  function mkt()
    type(t) :: mkt
    mkt = t()
  end function
end module

module mid
  use mg, only: p
end module

subroutine direct
  use mg, only: p, other, f, t
  call p(1.0)
end subroutine

subroutine renamed
  use mg, only: q => p
  call q(1.0)
end subroutine

subroutine reexported
  use mid, only: p
  call p(1.0)
end subroutine

! CHECK-DAG: fir.global @_QMmgEp <{module_vars = [#fir.module_var<"mg", "p", #{{.*}}>]}>
! CHECK-DAG: #fir.scope<module "mid", uses = [#fir.use<"mg", only [<"p">], #{{.*}}>], #{{.*}}>
! CHECK-DAG: #fir.use<"mg", only [<"p">, <"other">], #{{.*}}>
! CHECK-DAG: #fir.use<"mg", only [<"q" -> "p">], #{{.*}}>
! CHECK-DAG: #fir.use<"mid", only [<"p" from "mg">], #{{.*}}>

! LLVM-DAG: ![[P:[0-9]+]] = distinct !DIGlobalVariable(name: "p", linkageName: "_QMmgEp",
! LLVM-DAG: ![[DIRECT:[0-9]+]] = distinct !DISubprogram(name: "direct",
! LLVM-DAG: ![[RENAMED:[0-9]+]] = distinct !DISubprogram(name: "renamed",
! LLVM-DAG: ![[REEXPORTED:[0-9]+]] = distinct !DISubprogram(name: "reexported",
! LLVM-DAG: !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: ![[DIRECT]], entity: ![[P]],
! LLVM-DAG: !DIImportedEntity(tag: DW_TAG_imported_declaration, name: "q", scope: ![[RENAMED]], entity: ![[P]],
! LLVM-DAG: !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: ![[REEXPORTED]], entity: ![[P]],
