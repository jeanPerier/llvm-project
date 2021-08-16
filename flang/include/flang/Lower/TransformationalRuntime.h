//===-- Lower/TransformationalRuntime.h --*- C++ -*-===//
// lower transformational intrinsics
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef FORTRAN_LOWER_TRANSFORMATIONALRUNTIME_H
#define FORTRAN_LOWER_TRANSFORMATIONALRUNTIME_H

#include "mlir/Dialect/StandardOps/IR/Ops.h"

namespace fir {
class ExtendedValue;
class FirOpBuilder;
} // namespace fir

namespace Fortran::lower {

void genCshift(fir::FirOpBuilder &builder, mlir::Location loc,
               mlir::Value resultBox, mlir::Value arrayBox,
               mlir::Value shiftBox, mlir::Value dimBox);

void genCshiftVector(fir::FirOpBuilder &builder, mlir::Location loc,
                     mlir::Value resultBox, mlir::Value arrayBox,
                     mlir::Value shiftBox);

void genMatmul(fir::FirOpBuilder &builder, mlir::Location loc,
               mlir::Value matrixABox, mlir::Value matrixBBox,
               mlir::Value resultBox);

void genPack(fir::FirOpBuilder &builder, mlir::Location loc,
             mlir::Value resultBox, mlir::Value arrayBox, mlir::Value maskBox,
             mlir::Value vectorBox);

void genReshape(fir::FirOpBuilder &builder, mlir::Location loc,
                mlir::Value resultBox, mlir::Value sourceBox,
                mlir::Value shapeBox, mlir::Value padBox, mlir::Value orderBox);

void genSpread(fir::FirOpBuilder &builder, mlir::Location loc,
               mlir::Value resultBox, mlir::Value sourceBox, mlir::Value dim,
               mlir::Value ncopies);

void genTranspose(fir::FirOpBuilder &builder, mlir::Location loc,
                  mlir::Value resultBox, mlir::Value sourceBox);

void genUnpack(fir::FirOpBuilder &builder, mlir::Location loc,
               mlir::Value resultBox, mlir::Value vectorBox,
               mlir::Value maskBox, mlir::Value fieldBox);

} // namespace Fortran::lower

#endif // FORTRAN_LOWER_TRANSFORMATIONALRUNTIME_H
