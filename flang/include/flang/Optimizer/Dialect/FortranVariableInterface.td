//===- FortranVariableInterface.td -------------------------*- tablegen -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file contains a set of interfaces for Fortran variables.
//
//===----------------------------------------------------------------------===//

#ifndef FORTRANVARIABLEINTERFACE
#define FORTRANVARIABLEINTERFACE

include "mlir/IR/OpBase.td"


def DefineFortranVariableOpInterface : OpInterface<"DefineFortranVariableOpInterface"> {
  let description = [{
    Interface for operation that creates Fortran like variables in order to query about all
    their Fortran properties.
  }];

  let methods = [
    InterfaceMethod<
      /*desc=*/"Get the address produced by the definition",
      /*retTy=*/"mlir::Value",
      /*methodName=*/"getBase",
      /*args=*/(ins),
      /*methodBody=*/[{}],
      /*defaultImplementation=*/[{
        ConcreteOp op = mlir::cast<ConcreteOp>(this->getOperation());
        return op.getResult();
      }]
    >,
    InterfaceMethod<
      /*desc=*/"Get Fortran attributes",
      /*retTy=*/"llvm::Optional<fir::FortranVariableFlagsEnum>",
      /*methodName=*/"getFortranAttrs",
      /*args=*/(ins),
      /*methodBody=*/[{}],
      /*defaultImplementation=*/[{
        ConcreteOp op = mlir::cast<ConcreteOp>(this->getOperation());
        return op.getFortran_attrs();
      }]
    >,
    InterfaceMethod<
      /*desc=*/"Get variable shape",
      /*retTy=*/"llvm::Optional<mlir::Value>",
      /*methodName=*/"getShape",
      /*args=*/(ins),
      /*methodBody=*/[{}],
      /*defaultImplementation=*/[{
        ConcreteOp op = mlir::cast<ConcreteOp>(this->getOperation());
        return op.getShape();
      }]
    >,
    InterfaceMethod<
      /*desc=*/"Get explicit type parameters",
      /*retTy=*/"mlir::OperandRange",
      /*methodName=*/"getExplicitTypeParams",
      /*args=*/(ins),
      /*methodBody=*/[{}],
      /*defaultImplementation=*/[{
        ConcreteOp op = mlir::cast<ConcreteOp>(this->getOperation());
        return op.getTypeparams();
      }]
    >,
  ];

  let extraClassDeclaration = [{

    /// Get the sequence or scalar value type.
    mlir::Type getElementOrSequenceType() {
      return fir::unwrapPassByRefType(getBase().getType());
    }

    /// Get the variable scalar value type.
    mlir::Type getElementType() {
      return fir::unwrapSequenceType(getElementOrSequenceType());
    }

    /// Is this defining an array ?
    bool isArray() {
      return getElementOrSequenceType().isa<fir::SequenceType>();
    }

    bool isPointer() {
      auto attrs = getFortranAttrs();
      return attrs && bitEnumContainsAny(*attrs,
                        fir::FortranVariableFlagsEnum::pointer);
    }

    bool isAllocatable() {
      auto attrs = getFortranAttrs();
      return attrs && bitEnumContainsAny(*attrs,
                        fir::FortranVariableFlagsEnum::allocatable);
    }

    bool isCharacter() {
      return getElementType().isa<fir::CharacterType>();
    }

    bool hasExplicitCharLen() {
      return isCharacter() && !getExplicitTypeParams().empty();
    }

    mlir::Value getExplicitCharLen() {
      assert(hasExplicitCharLen() && "must be an explicit length character");
      return getExplicitTypeParams()[0];
    }

  }];

}

#endif  // FORTRANVARIABLEINTERFACE
