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
      /*desc=*/"Is this variable an array ?",
      /*retTy=*/"bool",
      /*methodName=*/"isArray",
      /*args=*/(ins),
      /*methodBody=*/[{}],
      /*defaultImplementation=*/[{
        ConcreteOp op = mlir::cast<ConcreteOp>(this->getOperation());
        mlir::Type type = op.getResult().getType();
        return fir::unwrapPassByRefType(type).isa<fir::SequenceType>();
      }]
    >,
  ];
}

#endif  // FORTRANVARIABLEINTERFACE
