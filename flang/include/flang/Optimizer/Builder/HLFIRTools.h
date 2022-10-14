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

namespace fir {

class FirOpBuilder;

class HlfirValue : public details::matcher<HlfirValue> {
public:
  using VT = std::variant<mlir::Value, fir::DefineFortranVariableOpInterface>;
  HlfirValue(mlir::Value val) : valueOrVariable{val} {
    assert(hasFortranValueType(val) && "must be a Fortran value type");
  }
  HlfirValue(fir::DefineFortranVariableOpInterface var)
      : valueOrVariable{var} {}

  static bool hasFortranValueType(mlir::Value value) {
    mlir::Type type = value.getType();
    return type.isa<fir::ExprType>() || fir::isa_trivial(type);
  }

  bool isValue() const {
    return std::holds_alternative<mlir::Value>(valueOrVariable);
  }
  bool isVariable() const { return !isValue(); }

  mlir::Value getBase() {
    return match([](mlir::Value val) { return val; },
                 [](fir::DefineFortranVariableOpInterface var) {
                   return var.getBase();
                 });
  }

  const VT &matchee() const { return valueOrVariable; }

private:
  VT valueOrVariable;
};

fir::ExtendedValue
toExtendedValue(fir::DefineFortranVariableOpInterface varDefinition);

} // namespace fir

namespace fir::factory {

using CleanupFunction = std::function<void()>;

std::pair<fir::ExtendedValue, std::optional<CleanupFunction>>
HlfirValueToExtendedValue(mlir::Location loc, fir::FirOpBuilder &builder,
                          fir::HlfirValue value);

std::pair<fir::HlfirValue, std::optional<CleanupFunction>>
readHlfirVarToValue(mlir::Location loc, fir::FirOpBuilder &builder,
                    fir::HlfirValue hlfirObject);

std::pair<fir::HlfirValue, std::optional<CleanupFunction>>
copyNonSimplyContiguousIntoTemp(mlir::Location loc, fir::FirOpBuilder &builder,
                                fir::HlfirValue hlfirObject);

std::pair<fir::HlfirValue, std::optional<CleanupFunction>>
storeHlfirValueToTemp(mlir::Location loc, fir::FirOpBuilder &builder,
                      fir::HlfirValue hlfirObject);
} // namespace fir::factory

#endif // FORTRAN_OPTIMIZER_BUILDER_BOXVALUE_H
