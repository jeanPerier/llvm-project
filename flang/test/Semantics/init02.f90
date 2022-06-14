! RUN: %flang_fc1 -fdebug-dump-parse-tree %s 2>&1 | FileCheck %s --check-prefix=SEMA_ON
! RUN: %flang_fc1 -fdebug-dump-parse-tree-no-sema %s 2>&1 | FileCheck %s --check-prefix=SEMA_OFF

!-----------------
! EXPECTEED OUTPUT
!-----------------

!SEMA_ON:      Name = 'a0'
!SEMA_ON-NEXT: Initialization -> Constant -> Expr -> Designator -> DataRef -> Name = 'kind'
!SEMA_ON:      Name = 'a1'
!SEMA_ON-NEXT: ComponentArraySpec -> ExplicitShapeSpec
!SEMA_ON-NEXT: | SpecificationExpr -> Scalar -> Integer -> Expr = '2_4'
!SEMA_ON-NEXT: | | LiteralConstant -> IntLiteralConstant = '2'
!SEMA_ON-NEXT: Initialization -> Constant -> Expr -> FunctionReference -> Call
!SEMA_ON-NEXT: | ProcedureDesignator -> Name = 'int'
!SEMA_ON-NEXT: | ActualArgSpec
!SEMA_ON-NEXT: | | ActualArg -> Expr -> FunctionReference -> Call
!SEMA_ON-NEXT: | | | ProcedureDesignator -> Name = 'kind'
!SEMA_ON-NEXT: | | | ActualArgSpec
!SEMA_ON-NEXT: | | | | ActualArg -> Expr -> LiteralConstant -> IntLiteralConstant = '0'
!SEMA_ON-NEXT: | ActualArgSpec
!SEMA_ON-NEXT: | | ActualArg -> Expr -> FunctionReference -> Call
!SEMA_ON-NEXT: | | | ProcedureDesignator -> Name = 'kind'
!SEMA_ON-NEXT: | | | ActualArgSpec
!SEMA_ON-NEXT: | | | | ActualArg -> Expr -> Designator -> DataRef -> Name = 'kind'
!SEMA_ON:      Name = 'a2'
!SEMA_ON-NEXT: ComponentArraySpec -> ExplicitShapeSpec
!SEMA_ON-NEXT: | SpecificationExpr -> Scalar -> Integer -> Expr = 'int(kind,kind=4)'
!SEMA_ON-NEXT: | | Designator -> DataRef -> Name = 'kind'
!SEMA_ON-NEXT: Initialization -> Constant -> Expr -> ArrayConstructor -> AcSpec
!SEMA_ON-NEXT: | AcValue -> Expr -> LiteralConstant -> IntLiteralConstant = '1'
!SEMA_ON-NEXT: | AcValue -> Expr -> LiteralConstant -> IntLiteralConstant = '2'
!SEMA_ON:      Name = 'a3'
!SEMA_ON-NEXT: ComponentArraySpec -> ExplicitShapeSpec
!SEMA_ON-NEXT: | SpecificationExpr -> Scalar -> Integer -> Expr = 'int(kind,kind=4)'
!SEMA_ON-NEXT: | | Designator -> DataRef -> Name = 'kind'
!SEMA_ON-NEXT: Initialization -> Constant -> Expr -> Designator -> DataRef -> Name = 'l'
!SEMA_ON:      Name = 'a4'
!SEMA_ON-NEXT: ComponentArraySpec -> ExplicitShapeSpec
!SEMA_ON-NEXT: | SpecificationExpr -> Scalar -> Integer -> Expr = '4_4'
!SEMA_ON-NEXT: | | FunctionReference -> Call
!SEMA_ON-NEXT: | | | ProcedureDesignator -> Name = 'kind'
!SEMA_ON-NEXT: | | | ActualArgSpec
!SEMA_ON-NEXT: | | | | ActualArg -> Expr = 'int(kind,kind=4)'
!SEMA_ON-NEXT: | | | | | Designator -> DataRef -> Name = 'kind'
!SEMA_ON-NEXT: Initialization -> Constant -> Expr = '2_4'
!SEMA_ON-NEXT: | LiteralConstant -> IntLiteralConstant = '2'
!SEMA_ON:      Name = 'x'
!SEMA_ON-NEXT: Initialization -> Constant -> Expr = '10_4'
!SEMA_ON-NEXT: | LiteralConstant -> IntLiteralConstant = '10'
!SEMA_ON:      Name = 'll'
!SEMA_ON-NEXT: Initialization -> Constant -> Expr = '.true._4'
!SEMA_ON-NEXT: | LiteralConstant -> LogicalLiteralConstant
!SEMA_ON-NEXT: | | bool
!SEMA_ON:      Name = 'r'
!SEMA_ON-NEXT: Initialization -> Constant -> Expr = '1._4'
!SEMA_ON-NEXT: | LiteralConstant -> RealLiteralConstant
!SEMA_ON-NEXT: | | Real = '1.0'
!SEMA_ON:      Name = 'c'
!SEMA_ON-NEXT: Initialization -> Constant -> Expr = '(2._4,1._4)'
!SEMA_ON-NEXT: | LiteralConstant -> ComplexLiteralConstant
!SEMA_ON-NEXT: | | ComplexPart -> SignedRealLiteralConstant
!SEMA_ON-NEXT: | | | RealLiteralConstant
!SEMA_ON-NEXT: | | | | Real = '2.0'
!SEMA_ON-NEXT: | | ComplexPart -> SignedRealLiteralConstant
!SEMA_ON-NEXT: | | | RealLiteralConstant
!SEMA_ON-NEXT: | | | | Real = '1.0'
!SEMA_ON:      Name = 's'
!SEMA_ON-NEXT: Initialization -> Constant -> Expr = '"s"'
!SEMA_ON-NEXT: | LiteralConstant -> CharLiteralConstant
!SEMA_ON-NEXT: | | string = 's'

!SEMA_OFF:      Name = 'a0'
!SEMA_OFF-NEXT: Initialization -> Constant -> Expr -> Designator -> DataRef -> Name = 'kind'
!SEMA_OFF:      Name = 'a1'
!SEMA_OFF-NEXT: ComponentArraySpec -> ExplicitShapeSpec
!SEMA_OFF-NEXT: | SpecificationExpr -> Scalar -> Integer -> Expr -> LiteralConstant -> IntLiteralConstant = '2'
!SEMA_OFF-NEXT: Initialization -> Constant -> Expr -> FunctionReference -> Call
!SEMA_OFF-NEXT: | ProcedureDesignator -> Name = 'int'
!SEMA_OFF-NEXT: | ActualArgSpec
!SEMA_OFF-NEXT: | | ActualArg -> Expr -> FunctionReference -> Call
!SEMA_OFF-NEXT: | | | ProcedureDesignator -> Name = 'kind'
!SEMA_OFF-NEXT: | | | ActualArgSpec
!SEMA_OFF-NEXT: | | | | ActualArg -> Expr -> LiteralConstant -> IntLiteralConstant = '0'
!SEMA_OFF-NEXT: | ActualArgSpec
!SEMA_OFF-NEXT: | | ActualArg -> Expr -> FunctionReference -> Call
!SEMA_OFF-NEXT: | | | ProcedureDesignator -> Name = 'kind'
!SEMA_OFF-NEXT: | | | ActualArgSpec
!SEMA_OFF-NEXT: | | | | ActualArg -> Expr -> Designator -> DataRef -> Name = 'kind'
!SEMA_OFF:      Name = 'a2'
!SEMA_OFF-NEXT: ComponentArraySpec -> ExplicitShapeSpec
!SEMA_OFF-NEXT: | SpecificationExpr -> Scalar -> Integer -> Expr -> Designator -> DataRef -> Name = 'kind'
!SEMA_OFF-NEXT: Initialization -> Constant -> Expr -> ArrayConstructor -> AcSpec
!SEMA_OFF-NEXT: | AcValue -> Expr -> LiteralConstant -> IntLiteralConstant = '1'
!SEMA_OFF-NEXT: | AcValue -> Expr -> LiteralConstant -> IntLiteralConstant = '2'
!SEMA_OFF:      Name = 'a3'
!SEMA_OFF-NEXT: ComponentArraySpec -> ExplicitShapeSpec
!SEMA_OFF-NEXT: | SpecificationExpr -> Scalar -> Integer -> Expr -> Designator -> DataRef -> Name = 'kind'
!SEMA_OFF-NEXT: Initialization -> Constant -> Expr -> Designator -> DataRef -> Name = 'l'
!SEMA_OFF:      Name = 'a4'
!SEMA_OFF-NEXT: ComponentArraySpec -> ExplicitShapeSpec
!SEMA_OFF-NEXT: | SpecificationExpr -> Scalar -> Integer -> Expr -> FunctionReference -> Call
!SEMA_OFF-NEXT: | | ProcedureDesignator -> Name = 'kind'
!SEMA_OFF-NEXT: | | ActualArgSpec
!SEMA_OFF-NEXT: | | | ActualArg -> Expr -> Designator -> DataRef -> Name = 'kind'
!SEMA_OFF-NEXT: Initialization -> Constant -> Expr -> LiteralConstant -> IntLiteralConstant = '2'
!SEMA_OFF:      Name = 'x'
!SEMA_OFF-NEXT: Initialization -> Constant -> Expr -> LiteralConstant -> IntLiteralConstant = '10'
!SEMA_OFF:      Name = 'll'
!SEMA_OFF-NEXT: Initialization -> Constant -> Expr -> LiteralConstant -> LogicalLiteralConstant
!SEMA_OFF-NEXT: | bool
!SEMA_OFF:      Name = 'r'
!SEMA_OFF-NEXT: Initialization -> Constant -> Expr -> LiteralConstant -> RealLiteralConstant
!SEMA_OFF-NEXT: | Real = '1.0'
!SEMA_OFF:      Name = 'c'
!SEMA_OFF-NEXT: Initialization -> Constant -> Expr -> LiteralConstant -> ComplexLiteralConstant
!SEMA_OFF-NEXT: | ComplexPart -> SignedRealLiteralConstant
!SEMA_OFF-NEXT: | | RealLiteralConstant
!SEMA_OFF-NEXT: | | | Real = '2.0'
!SEMA_OFF-NEXT: | ComplexPart -> SignedRealLiteralConstant
!SEMA_OFF-NEXT: | | RealLiteralConstant
!SEMA_OFF-NEXT: | | | Real = '1.0'
!SEMA_OFF:      Name = 's'
!SEMA_OFF-NEXT: Initialization -> Constant -> Expr -> LiteralConstant -> CharLiteralConstant
!SEMA_OFF-NEXT: | string = 's'

subroutine sub()
  type my_type (kind, l)
    integer, KIND :: kind = 4
    integer, LEN :: l = 4
    integer :: a0 = kind
    integer :: a1(2) = int(kind(0), kind(kind))
    integer :: a2(kind) = [1, 2]
    integer :: a3(kind) = l
    integer :: a4(kind(kind)) = 2
    integer :: x = 10
    logical :: ll = .true.
    real :: r = 1.0
    complex :: c = (2.0, 1.0)
    character :: s = "s"
  end type
end
