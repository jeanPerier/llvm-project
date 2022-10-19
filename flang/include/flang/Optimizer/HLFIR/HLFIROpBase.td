//===-- HLFIROpBase.td - HLFIR dialect base definitions ----*- tablegen -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// Definition of the HLFIR dialect and core fir.expr type
///
//===----------------------------------------------------------------------===//

#ifndef FORTRAN_DIALECT_HLFIR_OP_BASE
#define FORTRAN_DIALECT_HLFIR_OP_BASE

include "mlir/IR/OpBase.td"

def HLFIR_Dialect : Dialect {
  let name = "hlfir";

  let summary = "High Level Fortran IR.";

  let description = [{
    This dialect models Fortran expressions and assignments without requiring
    the allocation and manipulation of temporary storage.
    It allows running high level optimization passes and is rather
    straightforward to generate from Fortran expressions and assignments.
  
    It is not a complete implementation of Fortran, for constructs and lower
    level operations, FIR should be used directly.

    A bufferization pass transform hlfir.expr values into FIR temporary in
    memory, and its translation pass to FIR translates high level operations
    into sequence of lower level FIR operations on memory.
  }];

  let useDefaultAttributePrinterParser = 1;
  let cppNamespace = "::mlir::acc";
}


def hlfir_ExprType : TypeDef<hlfir_Dialect, Expr> {
  let mnemonic = "expr";
  let summary = "The type of an array, character, or derived type Fortran expression";

  let description = [{
    Abstract value type for Fortran arrays, characters and derived types.
    The rank cannot be assumed.
  }];


  let parameters = (ins
    ArrayRefParameter<"int64_t", "expression shape">:$shape,
    "mlir::Type":$elementType
    "bool":$polymorphic,
  );

  let extraClassDeclaration = [{
      mlir::Type getEleTy() const {return getElementType();}
    bool isArray() const { return !getShape().empty(); }
  }];

  let hasCustomAssemblyFormat = 1;
  
}

#endif FORTRAN_DIALECT_HLFIR_OP_BASE
