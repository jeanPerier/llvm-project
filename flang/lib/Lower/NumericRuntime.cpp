//===-- NumericRuntime.cpp -- runtime for numeric intrinsics -------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "flang/Lower/NumericRuntime.h"
#include "../../runtime/numeric.h"
#include "RTBuilder.h"
#include "flang/Lower/Bridge.h"
#include "flang/Lower/Todo.h"
#include "flang/Optimizer/Builder/BoxValue.h"
#include "flang/Optimizer/Builder/Character.h"
#include "flang/Optimizer/Builder/FIRBuilder.h"
#include "mlir/Dialect/StandardOps/IR/Ops.h"

using namespace Fortran::runtime;

// The real*10 and real*16 placeholders below are used to force the
// compilation of the real*10 and real*16 method names on systems that
// may not have them in their runtime library. This can occur in the
// case of cross compilation, for example.

/// Placeholder for real*10 version of Exponent Intrinsic
struct ForcedExponent10_4 {
  static constexpr const char *name = ExpandAndQuoteKey(RTNAME(Exponent10_4));
  static constexpr Fortran::lower::FuncTypeBuilderFunc getTypeModel() {
    return [](mlir::MLIRContext *ctx) {
      auto fltTy = mlir::FloatType::getF80(ctx);
      auto intTy = mlir::IntegerType::get(ctx, 32);
      return mlir::FunctionType::get(ctx, fltTy, intTy);
    };
  }
};

struct ForcedExponent10_8 {
  static constexpr const char *name = ExpandAndQuoteKey(RTNAME(Exponent10_8));
  static constexpr Fortran::lower::FuncTypeBuilderFunc getTypeModel() {
    return [](mlir::MLIRContext *ctx) {
      auto fltTy = mlir::FloatType::getF80(ctx);
      auto intTy = mlir::IntegerType::get(ctx, 64);
      return mlir::FunctionType::get(ctx, fltTy, intTy);
    };
  }
};

/// Placeholder for real*16 version of Exponent Intrinsic
struct ForcedExponent16_4 {
  static constexpr const char *name = ExpandAndQuoteKey(RTNAME(Exponent16_4));
  static constexpr Fortran::lower::FuncTypeBuilderFunc getTypeModel() {
    return [](mlir::MLIRContext *ctx) {
      auto fltTy = mlir::FloatType::getF128(ctx);
      auto intTy = mlir::IntegerType::get(ctx, 32);
      return mlir::FunctionType::get(ctx, fltTy, intTy);
    };
  }
};

struct ForcedExponent16_8 {
  static constexpr const char *name = ExpandAndQuoteKey(RTNAME(Exponent16_8));
  static constexpr Fortran::lower::FuncTypeBuilderFunc getTypeModel() {
    return [](mlir::MLIRContext *ctx) {
      auto fltTy = mlir::FloatType::getF128(ctx);
      auto intTy = mlir::IntegerType::get(ctx, 64);
      return mlir::FunctionType::get(ctx, fltTy, intTy);
    };
  }
};

/// Placeholder for real*10 version of Fraction Intrinsic
struct ForcedFraction10 {
  static constexpr const char *name = ExpandAndQuoteKey(RTNAME(Fraction10));
  static constexpr Fortran::lower::FuncTypeBuilderFunc getTypeModel() {
    return [](mlir::MLIRContext *ctx) {
      auto ty = mlir::FloatType::getF80(ctx);
      return mlir::FunctionType::get(ctx, {ty}, {ty});
    };
  }
};

/// Placeholder for real*16 version of Fraction Intrinsic
struct ForcedFraction16 {
  static constexpr const char *name = ExpandAndQuoteKey(RTNAME(Fraction16));
  static constexpr Fortran::lower::FuncTypeBuilderFunc getTypeModel() {
    return [](mlir::MLIRContext *ctx) {
      auto ty = mlir::FloatType::getF128(ctx);
      return mlir::FunctionType::get(ctx, {ty}, {ty});
    };
  }
};

/// Placeholder for real*10 version of RRSpacing Intrinsic
struct ForcedRRSpacing10 {
  static constexpr const char *name = ExpandAndQuoteKey(RTNAME(RRSpacing10));
  static constexpr Fortran::lower::FuncTypeBuilderFunc getTypeModel() {
    return [](mlir::MLIRContext *ctx) {
      auto ty = mlir::FloatType::getF80(ctx);
      return mlir::FunctionType::get(ctx, {ty}, {ty});
    };
  }
};

/// Placeholder for real*16 version of RRSpacing Intrinsic
struct ForcedRRSpacing16 {
  static constexpr const char *name = ExpandAndQuoteKey(RTNAME(RRSpacing16));
  static constexpr Fortran::lower::FuncTypeBuilderFunc getTypeModel() {
    return [](mlir::MLIRContext *ctx) {
      auto ty = mlir::FloatType::getF128(ctx);
      return mlir::FunctionType::get(ctx, {ty}, {ty});
    };
  }
};

/// Placeholder for real*10 version of Scale Intrinsic
struct ForcedScale10 {
  static constexpr const char *name = ExpandAndQuoteKey(RTNAME(Scale10));
  static constexpr Fortran::lower::FuncTypeBuilderFunc getTypeModel() {
    return [](mlir::MLIRContext *ctx) {
      auto fltTy = mlir::FloatType::getF80(ctx);
      auto intTy = mlir::IntegerType::get(ctx, 64);
      return mlir::FunctionType::get(ctx, {fltTy, intTy}, {fltTy});
    };
  }
};

/// Placeholder for real*16 version of Scale Intrinsic
struct ForcedScale16 {
  static constexpr const char *name = ExpandAndQuoteKey(RTNAME(Scale16));
  static constexpr Fortran::lower::FuncTypeBuilderFunc getTypeModel() {
    return [](mlir::MLIRContext *ctx) {
      auto fltTy = mlir::FloatType::getF128(ctx);
      auto intTy = mlir::IntegerType::get(ctx, 64);
      return mlir::FunctionType::get(ctx, {fltTy, intTy}, {fltTy});
    };
  }
};

/// Placeholder for real*10 version of RRSpacing Intrinsic
struct ForcedSetExponent10 {
  static constexpr const char *name = ExpandAndQuoteKey(RTNAME(SetExponent10));
  static constexpr Fortran::lower::FuncTypeBuilderFunc getTypeModel() {
    return [](mlir::MLIRContext *ctx) {
      auto fltTy = mlir::FloatType::getF80(ctx);
      auto intTy = mlir::IntegerType::get(ctx, 64);
      return mlir::FunctionType::get(ctx, {fltTy, intTy}, {fltTy});
    };
  }
};

/// Placeholder for real*10 version of RRSpacing Intrinsic
struct ForcedSetExponent16 {
  static constexpr const char *name = ExpandAndQuoteKey(RTNAME(SetExponent16));
  static constexpr Fortran::lower::FuncTypeBuilderFunc getTypeModel() {
    return [](mlir::MLIRContext *ctx) {
      auto fltTy = mlir::FloatType::getF128(ctx);
      auto intTy = mlir::IntegerType::get(ctx, 64);
      return mlir::FunctionType::get(ctx, {fltTy, intTy}, {fltTy});
    };
  }
};

/// Placeholder for real*10 version of Spacing Intrinsic
struct ForcedSpacing10 {
  static constexpr const char *name = ExpandAndQuoteKey(RTNAME(Spacing10));
  static constexpr Fortran::lower::FuncTypeBuilderFunc getTypeModel() {
    return [](mlir::MLIRContext *ctx) {
      auto ty = mlir::FloatType::getF80(ctx);
      return mlir::FunctionType::get(ctx, {ty}, {ty});
    };
  }
};

/// Placeholder for real*16 version of Spacing Intrinsic
struct ForcedSpacing16 {
  static constexpr const char *name = ExpandAndQuoteKey(RTNAME(Spacing16));
  static constexpr Fortran::lower::FuncTypeBuilderFunc getTypeModel() {
    return [](mlir::MLIRContext *ctx) {
      auto ty = mlir::FloatType::getF128(ctx);
      return mlir::FunctionType::get(ctx, {ty}, {ty});
    };
  }
};

/// Generate call to Exponent instrinsic runtime routine.
mlir::Value Fortran::lower::genExponent(fir::FirOpBuilder &builder,
                                        mlir::Location loc,
                                        mlir::Type resultType, mlir::Value x) {
  mlir::FuncOp func;
  mlir::Type fltTy = x.getType();

  if (fltTy.isF32()) {
    if (resultType.isInteger(32))
      func = Fortran::lower::getRuntimeFunc<mkRTKey(Exponent4_4)>(loc, builder);
    else if (resultType.isInteger(64))
      func = Fortran::lower::getRuntimeFunc<mkRTKey(Exponent4_8)>(loc, builder);
  } else if (fltTy.isF64()) {
    if (resultType.isInteger(32))
      func = Fortran::lower::getRuntimeFunc<mkRTKey(Exponent8_4)>(loc, builder);
    else if (resultType.isInteger(64))
      func = Fortran::lower::getRuntimeFunc<mkRTKey(Exponent8_8)>(loc, builder);
  } else if (fltTy.isF80()) {
    if (resultType.isInteger(32))
      func = Fortran::lower::getRuntimeFunc<ForcedExponent10_4>(loc, builder);
    else if (resultType.isInteger(64))
      func = Fortran::lower::getRuntimeFunc<ForcedExponent10_8>(loc, builder);
  } else if (fltTy.isF128()) {
    if (resultType.isInteger(32))
      func = Fortran::lower::getRuntimeFunc<ForcedExponent16_4>(loc, builder);
    else if (resultType.isInteger(64))
      func = Fortran::lower::getRuntimeFunc<ForcedExponent16_8>(loc, builder);
  } else
    TODO(loc, "unsupported real kind in Exponent lowering");

  auto funcTy = func.getType();
  llvm::SmallVector<mlir::Value> args = {
      builder.createConvert(loc, funcTy.getInput(0), x)};

  return builder.create<fir::CallOp>(loc, func, args).getResult(0);
}

/// Generate call to Fraction instrinsic runtime routine.
mlir::Value Fortran::lower::genFraction(fir::FirOpBuilder &builder,
                                        mlir::Location loc, mlir::Value x) {
  mlir::FuncOp func;
  mlir::Type fltTy = x.getType();

  if (fltTy.isF32())
    func = Fortran::lower::getRuntimeFunc<mkRTKey(Fraction4)>(loc, builder);
  else if (fltTy.isF64())
    func = Fortran::lower::getRuntimeFunc<mkRTKey(Fraction8)>(loc, builder);
  else if (fltTy.isF80())
    func = Fortran::lower::getRuntimeFunc<ForcedFraction10>(loc, builder);
  else if (fltTy.isF128())
    func = Fortran::lower::getRuntimeFunc<ForcedFraction16>(loc, builder);
  else
    TODO(loc, "unsupported real kind in Fraction lowering");

  auto funcTy = func.getType();
  llvm::SmallVector<mlir::Value> args = {
      builder.createConvert(loc, funcTy.getInput(0), x)};

  return builder.create<fir::CallOp>(loc, func, args).getResult(0);
}

/// Generate call to RRSpacing intrinsic runtime routine.
mlir::Value Fortran::lower::genRRSpacing(fir::FirOpBuilder &builder,
                                         mlir::Location loc, mlir::Value x) {
  mlir::FuncOp func;
  mlir::Type fltTy = x.getType();

  if (fltTy.isF32())
    func = Fortran::lower::getRuntimeFunc<mkRTKey(RRSpacing4)>(loc, builder);
  else if (fltTy.isF64())
    func = Fortran::lower::getRuntimeFunc<mkRTKey(RRSpacing8)>(loc, builder);
  else if (fltTy.isF80())
    func = Fortran::lower::getRuntimeFunc<ForcedRRSpacing10>(loc, builder);
  else if (fltTy.isF128())
    func = Fortran::lower::getRuntimeFunc<ForcedRRSpacing16>(loc, builder);
  else
    TODO(loc, "unsupported real kind in RRSpacing lowering");

  auto funcTy = func.getType();
  llvm::SmallVector<mlir::Value> args = {
      builder.createConvert(loc, funcTy.getInput(0), x)};

  return builder.create<fir::CallOp>(loc, func, args).getResult(0);
}

/// Generate call to Scale intrinsic runtime routine.
mlir::Value Fortran::lower::genScale(fir::FirOpBuilder &builder,
                                     mlir::Location loc, mlir::Value x,
                                     mlir::Value i) {
  mlir::FuncOp func;
  mlir::Type fltTy = x.getType();

  if (fltTy.isF32())
    func = Fortran::lower::getRuntimeFunc<mkRTKey(Scale4)>(loc, builder);
  else if (fltTy.isF64())
    func = Fortran::lower::getRuntimeFunc<mkRTKey(Scale8)>(loc, builder);
  else if (fltTy.isF80())
    func = Fortran::lower::getRuntimeFunc<ForcedScale10>(loc, builder);
  else if (fltTy.isF128())
    func = Fortran::lower::getRuntimeFunc<ForcedScale16>(loc, builder);
  else
    fir::emitFatalError(loc, "unsupported REAL kind in Scale lowering");

  auto funcTy = func.getType();
  auto args = Fortran::lower::createArguments(builder, loc, funcTy, x, i);

  return builder.create<fir::CallOp>(loc, func, args).getResult(0);
}

/// Generate call to Set_exponent instrinsic runtime routine.
mlir::Value Fortran::lower::genSetExponent(fir::FirOpBuilder &builder,
                                           mlir::Location loc, mlir::Value x,
                                           mlir::Value i) {
  mlir::FuncOp func;
  mlir::Type fltTy = x.getType();

  if (fltTy.isF32())
    func = Fortran::lower::getRuntimeFunc<mkRTKey(SetExponent4)>(loc, builder);
  else if (fltTy.isF64())
    func = Fortran::lower::getRuntimeFunc<mkRTKey(SetExponent8)>(loc, builder);
  else if (fltTy.isF80())
    func = Fortran::lower::getRuntimeFunc<ForcedSetExponent10>(loc, builder);
  else if (fltTy.isF128())
    func = Fortran::lower::getRuntimeFunc<ForcedSetExponent16>(loc, builder);
  else
    TODO(loc, "unsupported real kind in Fraction lowering");

  auto funcTy = func.getType();
  auto args = Fortran::lower::createArguments(builder, loc, funcTy, x, i);

  return builder.create<fir::CallOp>(loc, func, args).getResult(0);
}

/// Generate call to Spacing intrinsic runtime routine.
mlir::Value Fortran::lower::genSpacing(fir::FirOpBuilder &builder,
                                       mlir::Location loc, mlir::Value x) {
  mlir::FuncOp func;
  mlir::Type fltTy = x.getType();

  if (fltTy.isF32())
    func = Fortran::lower::getRuntimeFunc<mkRTKey(Spacing4)>(loc, builder);
  else if (fltTy.isF64())
    func = Fortran::lower::getRuntimeFunc<mkRTKey(Spacing8)>(loc, builder);
  else if (fltTy.isF80())
    func = Fortran::lower::getRuntimeFunc<ForcedSpacing10>(loc, builder);
  else if (fltTy.isF128())
    func = Fortran::lower::getRuntimeFunc<ForcedSpacing16>(loc, builder);
  else
    TODO(loc, "unsupported real kind in Spacing lowering");

  auto funcTy = func.getType();
  llvm::SmallVector<mlir::Value> args = {
      builder.createConvert(loc, funcTy.getInput(0), x)};

  return builder.create<fir::CallOp>(loc, func, args).getResult(0);
}
