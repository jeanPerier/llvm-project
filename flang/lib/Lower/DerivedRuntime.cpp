//===-- DerivedRuntime.cpp -- derived type runtime API --------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "flang/Lower/DerivedRuntime.h"
#include "../../runtime/derived-api.h"
#include "RTBuilder.h"
#include "flang/Lower/Todo.h"
#include "flang/Optimizer/Builder/FIRBuilder.h"

using namespace Fortran::runtime;

void Fortran::lower::genDerivedTypeInitialize(fir::FirOpBuilder &builder,
                                              mlir::Location loc,
                                              mlir::Value box) {
  auto func = getRuntimeFunc<mkRTKey(Initialize)>(loc, builder);
  auto fTy = func.getType();
  auto sourceFile = fir::factory::locationToFilename(builder, loc);
  auto sourceLine =
      fir::factory::locationToLineNo(builder, loc, fTy.getInput(2));
  auto args = Fortran::lower::createArguments(builder, loc, fTy, box,
                                              sourceFile, sourceLine);
  builder.create<fir::CallOp>(loc, func, args);
}

void Fortran::lower::genDerivedTypeDestroy(fir::FirOpBuilder &builder,
                                           mlir::Location loc,
                                           mlir::Value box) {
  auto func = getRuntimeFunc<mkRTKey(Destroy)>(loc, builder);
  auto fTy = func.getType();
  auto args = Fortran::lower::createArguments(builder, loc, fTy, box);
  builder.create<fir::CallOp>(loc, func, args);
}

void Fortran::lower::genDerivedTypeAssign(fir::FirOpBuilder &builder,
                                          mlir::Location loc,
                                          mlir::Value destinationBox,
                                          mlir::Value sourceBox) {
  auto func = getRuntimeFunc<mkRTKey(Assign)>(loc, builder);
  auto fTy = func.getType();
  auto sourceFile = fir::factory::locationToFilename(builder, loc);
  auto sourceLine =
      fir::factory::locationToLineNo(builder, loc, fTy.getInput(3));
  auto args = Fortran::lower::createArguments(
      builder, loc, fTy, destinationBox, sourceBox, sourceFile, sourceLine);
  builder.create<fir::CallOp>(loc, func, args);
}
