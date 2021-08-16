//===-- Lower/NumericRuntime.h -- lower numeric intrinsics --*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef FORTRAN_LOWER_NUMERICRUNTIME_H
#define FORTRAN_LOWER_NUMERICRUNTIME_H

#include "mlir/Dialect/StandardOps/IR/Ops.h"

namespace fir {
class ExtendedValue;
class FirOpBuilder;
} // namespace fir

namespace Fortran::lower {

/// Generate call to Exponent intrinsic runtime routine.
mlir::Value genExponent(fir::FirOpBuilder &builder, mlir::Location loc,
                        mlir::Type resultType, mlir::Value x);

/// Generate call to Fraction intrinsic runtime routine.
mlir::Value genFraction(fir::FirOpBuilder &builder, mlir::Location loc,
                        mlir::Value x);

/// Generate call to RRSpacing intrinsic runtime routine.
mlir::Value genRRSpacing(fir::FirOpBuilder &builder, mlir::Location loc,
                         mlir::Value x);

/// Generate call to Scale intrinsic runtime routine.
mlir::Value genScale(fir::FirOpBuilder &builder, mlir::Location loc,
                     mlir::Value x, mlir::Value i);

/// Generate call to Set_exponent intrinsic runtime routine.
mlir::Value genSetExponent(fir::FirOpBuilder &builder, mlir::Location loc,
                           mlir::Value x, mlir::Value i);

/// Generate call to Spacing intrinsic runtime routine.
mlir::Value genSpacing(fir::FirOpBuilder &builder, mlir::Location loc,
                       mlir::Value x);

} // namespace Fortran::lower
#endif
