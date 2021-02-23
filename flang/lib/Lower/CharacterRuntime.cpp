//===-- CharacterRuntime.cpp -- runtime for CHARACTER type entities -------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "flang/Lower/CharacterRuntime.h"
#include "../../runtime/character.h"
#include "RTBuilder.h"
#include "flang/Lower/Bridge.h"
#include "flang/Lower/CharacterExpr.h"
#include "flang/Lower/FIRBuilder.h"
#include "flang/Lower/Support/BoxValue.h"
#include "flang/Lower/Todo.h"
#include "mlir/Dialect/StandardOps/IR/Ops.h"

using namespace Fortran::runtime;

/// Helper function to recover the KIND from the FIR type.
static int discoverKind(mlir::Type ty) {
  if (auto charTy = ty.dyn_cast<fir::CharacterType>())
    return charTy.getFKind();
  if (auto eleTy = fir::dyn_cast_ptrEleTy(ty))
    return discoverKind(eleTy);
  if (auto arrTy = ty.dyn_cast<fir::SequenceType>())
    return discoverKind(arrTy.getEleTy());
  if (auto boxTy = ty.dyn_cast<fir::BoxCharType>())
    return discoverKind(boxTy.getEleTy());
  if (auto boxTy = ty.dyn_cast<fir::BoxType>())
    return discoverKind(boxTy.getEleTy());
  llvm_unreachable("unexpected character type");
}

//===----------------------------------------------------------------------===//
// Lower character operations
//===----------------------------------------------------------------------===//

mlir::Value
Fortran::lower::genRawCharCompare(Fortran::lower::FirOpBuilder &builder,
                                  mlir::Location loc, mlir::CmpIPredicate cmp,
                                  mlir::Value lhsBuff, mlir::Value lhsLen,
                                  mlir::Value rhsBuff, mlir::Value rhsLen) {
  mlir::FuncOp beginFunc;
  switch (discoverKind(lhsBuff.getType())) {
  case 1:
    beginFunc = getRuntimeFunc<mkRTKey(CharacterCompareScalar1)>(loc, builder);
    break;
  case 2:
    beginFunc = getRuntimeFunc<mkRTKey(CharacterCompareScalar2)>(loc, builder);
    break;
  case 4:
    beginFunc = getRuntimeFunc<mkRTKey(CharacterCompareScalar4)>(loc, builder);
    break;
  default:
    llvm_unreachable("runtime does not support CHARACTER KIND");
  }
  auto fTy = beginFunc.getType();
  auto lptr = builder.createConvert(loc, fTy.getInput(0), lhsBuff);
  auto llen = builder.createConvert(loc, fTy.getInput(2), lhsLen);
  auto rptr = builder.createConvert(loc, fTy.getInput(1), rhsBuff);
  auto rlen = builder.createConvert(loc, fTy.getInput(3), rhsLen);
  llvm::SmallVector<mlir::Value, 4> args = {lptr, rptr, llen, rlen};
  auto tri = builder.create<fir::CallOp>(loc, beginFunc, args).getResult(0);
  auto zero = builder.createIntegerConstant(loc, tri.getType(), 0);
  return builder.create<mlir::CmpIOp>(loc, cmp, tri, zero);
}

mlir::Value
Fortran::lower::genCharCompare(Fortran::lower::FirOpBuilder &builder,
                               mlir::Location loc, mlir::CmpIPredicate cmp,
                               const fir::ExtendedValue &lhs,
                               const fir::ExtendedValue &rhs) {
  if (lhs.getBoxOf<fir::BoxValue>() || rhs.getBoxOf<fir::BoxValue>())
    TODO(loc, "character compare from descriptors");
  auto allocateIfNotInMemory = [&](mlir::Value base) -> mlir::Value {
    if (fir::isa_ref_type(base.getType()))
      return base;
    auto mem = builder.create<fir::AllocaOp>(loc, base.getType());
    builder.create<fir::StoreOp>(loc, base, mem);
    return mem;
  };
  auto lhsBuffer = allocateIfNotInMemory(fir::getBase(lhs));
  auto rhsBuffer = allocateIfNotInMemory(fir::getBase(rhs));
  return genRawCharCompare(builder, loc, cmp, lhsBuffer, fir::getLen(lhs),
                           rhsBuffer, fir::getLen(rhs));
}

void Fortran::lower::genTrim(Fortran::lower::FirOpBuilder &builder,
                             mlir::Location loc, const mlir::Value &resultBox,
                             const mlir::Value &stringBox) {
  auto trimFunc = getRuntimeFunc<mkRTKey(Trim)>(loc, builder);
  auto fTy = trimFunc.getType();
  auto sourceFile = builder.locationToFilename(loc);
  auto sourceLine = builder.locationToLineNo(loc, fTy.getInput(3));

  llvm::SmallVector<mlir::Value, 4> args;
  auto i = 0;
  args.emplace_back(builder.createConvert(loc, fTy.getInput(i++), resultBox));
  args.emplace_back(builder.createConvert(loc, fTy.getInput(i++), stringBox));
  args.emplace_back(builder.createConvert(loc, fTy.getInput(i++), sourceFile));
  args.emplace_back(builder.createConvert(loc, fTy.getInput(i++), sourceLine));
  builder.create<fir::CallOp>(loc, trimFunc, args);
}
