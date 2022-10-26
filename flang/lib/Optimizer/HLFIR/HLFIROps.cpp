//===-- HLFIROps.cpp ------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Coding style: https://mlir.llvm.org/getting_started/DeveloperGuide/
//
//===----------------------------------------------------------------------===//
//
#include "flang/Optimizer/HLFIR/HLFIROps.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/DialectImplementation.h"
#include "mlir/IR/Matchers.h"
#include "mlir/IR/OpImplementation.h"
#include "llvm/ADT/TypeSwitch.h"

#define GET_OP_CLASSES
#include "flang/Optimizer/HLFIR/HLFIROps.cpp.inc"

static mlir::Type getHLFIRVariableTypeFor(mlir::Type inputType,
                                          bool hasExplicitLowerBounds) {
  mlir::Type type = fir::unwrapRefType(inputType);
  if (type.isa<fir::BaseBoxType>())
    return inputType;
  if (auto charType = type.dyn_cast<fir::CharacterType>())
    if (charType.hasDynamicLen())
      return fir::BoxCharType::get(charType.getContext(), charType.getFKind());
  if (fir::hasDynamicSize(type) || hasExplicitLowerBounds)
    return fir::BoxType::get(type);
  return inputType;
}

void hlfir::DeclareOp::build(mlir::OpBuilder &builder,
                             mlir::OperationState &result, mlir::Value memref,
                             llvm::StringRef uniq_name, mlir::Value shape,
                             mlir::ValueRange typeparams,
                             fir::FortranVariableFlagsAttr fortran_attrs) {
  auto nameAttr = builder.getStringAttr(uniq_name);
  mlir::Type inputType = memref.getType();
  bool hasExplicitLowerBounds =
      shape && shape.getType().isa<fir::ShapeShiftType, fir::ShiftType>();
  mlir::Type hlfirVariableType =
      getHLFIRVariableTypeFor(inputType, hasExplicitLowerBounds);
  build(builder, result, {hlfirVariableType, inputType}, memref, shape,
        typeparams, nameAttr, fortran_attrs);
}
