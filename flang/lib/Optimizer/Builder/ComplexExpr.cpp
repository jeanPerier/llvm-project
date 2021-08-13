//===-- ComplexExpr.cpp ---------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "flang/Lower/ComplexExpr.h"

//===----------------------------------------------------------------------===//
// ComplexExprHelper implementation
//===----------------------------------------------------------------------===//

mlir::Type
Fortran::lower::ComplexExprHelper::getComplexPartType(mlir::Type complexType) {
  return builder.getRealType(complexType.cast<fir::ComplexType>().getFKind());
}

mlir::Type
Fortran::lower::ComplexExprHelper::getComplexPartType(mlir::Value cplx) {
  return getComplexPartType(cplx.getType());
}

mlir::Value Fortran::lower::ComplexExprHelper::createComplex(fir::KindTy kind,
                                                             mlir::Value real,
                                                             mlir::Value imag) {
  auto complexTy = fir::ComplexType::get(builder.getContext(), kind);
  mlir::Value und = builder.create<fir::UndefOp>(loc, complexTy);
  return insert<Part::Imag>(insert<Part::Real>(und, real), imag);
}

mlir::Value Fortran::lower::ComplexExprHelper::createComplex(mlir::Type cplxTy,
                                                             mlir::Value real,
                                                             mlir::Value imag) {
  mlir::Value und = builder.create<fir::UndefOp>(loc, cplxTy);
  return insert<Part::Imag>(insert<Part::Real>(und, real), imag);
}
