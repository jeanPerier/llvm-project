//===-- HLFIRTools.h -- HLFIR tools       -----------------------*- C++ -*-===//
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

#ifndef FORTRAN_OPTIMIZER_BUILDER_HLFIRTOOLS_H
#define FORTRAN_OPTIMIZER_BUILDER_HLFIRTOOLS_H

#include "flang/Optimizer/Builder/BoxValue.h"
#include "flang/Optimizer/Dialect/FortranVariableInterface.h"
#include "flang/Optimizer/HLFIR/HLFIRDialect.h"

namespace fir {
class FirOpBuilder;
}

namespace hlfir {

inline bool isFortranValueType(mlir::Type type) {
  return type.isa<hlfir::ExprType>() || fir::isa_trivial(type);
}

inline bool isFortranValue(mlir::Value value) {
  return isFortranValueType(value.getType());
}

inline bool isFortranVariable(mlir::Value value) {
  return value.getDefiningOp<fir::FortranVariableOpInterface>();
}

inline bool isFortranEntity(mlir::Value value) {
  return isFortranValue(value) || isFortranVariable(value);
}

class FortranEntity : public mlir::Value {
public:
  FortranEntity(mlir::Value value) : mlir::Value(value) {
    assert(isFortranEntity(value) &&
           "must be a value representing a Fortran value or variable");
  }
  FortranEntity(fir::FortranVariableOpInterface variable)
      : mlir::Value(variable.getBase()) {}
  bool isValue() const { return isFortranValue(*this); }
  bool isVariable() const { return !isValue(); }
  fir::FortranVariableOpInterface getIfVariable() const {
    return this->getDefiningOp<fir::FortranVariableOpInterface>();
  }
  mlir::Value getBase() const { return *this; }
};

/// Functions to translate hlfir::FortranEntity to fir::ExtendedValue.
/// For Fortran arrays, character, and derived type values, this require
/// allocating a storage since these can only be represented in memory in FIR.
/// In that case, a cleanup function is provided to generate the finalization
/// code after the end of the fir::ExtendedValue use.
using CleanupFunction = std::function<void()>;
std::pair<fir::ExtendedValue, std::optional<CleanupFunction>>
translateToExtendedValue(mlir::Location loc, fir::FirOpBuilder &builder,
                         FortranEntity entity);

/// Function to translate FortranVariableOpInterface to fir::ExtendedValue.
/// It does not generate any IR, and is a simple packaging operation.
fir::ExtendedValue
translateToExtendedValue(fir::FortranVariableOpInterface fortranVariable);

std::pair<hlfir::FortranEntity, std::optional<CleanupFunction>>
readHlfirVarToValue(mlir::Location loc, fir::FirOpBuilder &builder,
                    hlfir::FortranEntity entity);

std::pair<hlfir::FortranEntity, std::optional<CleanupFunction>>
copyNonSimplyContiguousIntoTemp(mlir::Location loc, fir::FirOpBuilder &builder,
                                hlfir::FortranEntity entity);

std::pair<hlfir::FortranEntity, std::optional<CleanupFunction>>
storeHlfirValueToTemp(mlir::Location loc, fir::FirOpBuilder &builder,
                      hlfir::FortranEntity entity);
} // namespace hlfir

#endif // FORTRAN_OPTIMIZER_BUILDER_BOXVALUE_H
