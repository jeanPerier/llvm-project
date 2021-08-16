//===-- Runtime.cpp -------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "flang/Lower/Runtime.h"
#include "../../runtime/misc-intrinsic.h"
#include "../runtime/clock.h"
#include "../runtime/pointer.h"
#include "../runtime/random.h"
#include "../runtime/stop.h"
#include "../runtime/time-intrinsic.h"
#include "RTBuilder.h"
#include "StatementContext.h"
#include "flang/Lower/Bridge.h"
#include "flang/Optimizer/Builder/BoxValue.h"
#include "flang/Optimizer/Builder/FIRBuilder.h"
#include "flang/Parser/parse-tree.h"
#include "flang/Semantics/tools.h"
#include "llvm/Support/Debug.h"

#define DEBUG_TYPE "flang-lower-runtime"

using namespace Fortran::runtime;

// TODO: We don't have runtime library support for various features. When they
// are encountered, we emit an error message and exit immediately.
static void noRuntimeSupport(mlir::Location loc, llvm::StringRef stmt) {
  mlir::emitError(loc, "There is no runtime support for ")
      << stmt << " statement.\n";
  std::exit(1);
}

/// Runtime calls that do not return to the caller indicate this condition by
/// terminating the current basic block with an unreachable op.
static void genUnreachable(fir::FirOpBuilder &builder, mlir::Location loc) {
  builder.create<fir::UnreachableOp>(loc);
  auto *newBlock = builder.getBlock()->splitBlock(builder.getInsertionPoint());
  builder.setInsertionPointToStart(newBlock);
}

//===----------------------------------------------------------------------===//
// Misc. Fortran statements that lower to runtime calls
//===----------------------------------------------------------------------===//

void Fortran::lower::genStopStatement(
    Fortran::lower::AbstractConverter &converter,
    const Fortran::parser::StopStmt &stmt) {
  auto &builder = converter.getFirOpBuilder();
  auto loc = converter.getCurrentLocation();
  Fortran::lower::StatementContext stmtCtx;
  llvm::SmallVector<mlir::Value> operands;
  mlir::FuncOp callee;
  mlir::FunctionType calleeType;
  // First operand is stop code (zero if absent)
  if (const auto &code =
          std::get<std::optional<Fortran::parser::StopCode>>(stmt.t)) {
    auto expr =
        converter.genExprValue(*Fortran::semantics::GetExpr(*code), stmtCtx);
    LLVM_DEBUG(llvm::dbgs() << "stop expression: "; expr.dump();
               llvm::dbgs() << '\n');
    expr.match(
        [&](const fir::CharBoxValue &x) {
          callee = getRuntimeFunc<mkRTKey(StopStatementText)>(loc, builder);
          calleeType = callee.getType();
          // Creates a pair of operands for the CHARACTER and its LEN.
          operands.push_back(
              builder.createConvert(loc, calleeType.getInput(0), x.getAddr()));
          operands.push_back(
              builder.createConvert(loc, calleeType.getInput(1), x.getLen()));
        },
        [&](fir::UnboxedValue x) {
          callee = getRuntimeFunc<mkRTKey(StopStatement)>(loc, builder);
          calleeType = callee.getType();
          auto cast = builder.createConvert(loc, calleeType.getInput(0), x);
          operands.push_back(cast);
        },
        [&](auto) {
          mlir::emitError(loc, "unhandled expression in STOP");
          std::exit(1);
        });
  } else {
    callee = getRuntimeFunc<mkRTKey(StopStatement)>(loc, builder);
    calleeType = callee.getType();
    operands.push_back(
        builder.createIntegerConstant(loc, calleeType.getInput(0), 0));
  }

  // Second operand indicates ERROR STOP
  bool isError = std::get<Fortran::parser::StopStmt::Kind>(stmt.t) ==
                 Fortran::parser::StopStmt::Kind::ErrorStop;
  operands.push_back(builder.createIntegerConstant(
      loc, calleeType.getInput(operands.size()), isError));

  // Third operand indicates QUIET (default to false).
  if (const auto &quiet =
          std::get<std::optional<Fortran::parser::ScalarLogicalExpr>>(stmt.t)) {
    auto expr = Fortran::semantics::GetExpr(*quiet);
    assert(expr && "failed getting typed expression");
    auto q = fir::getBase(converter.genExprValue(*expr, stmtCtx));
    operands.push_back(
        builder.createConvert(loc, calleeType.getInput(operands.size()), q));
  } else {
    operands.push_back(builder.createIntegerConstant(
        loc, calleeType.getInput(operands.size()), 0));
  }

  builder.create<fir::CallOp>(loc, callee, operands);
  genUnreachable(builder, loc);
}

void Fortran::lower::genFailImageStatement(
    Fortran::lower::AbstractConverter &converter) {
  auto &builder = converter.getFirOpBuilder();
  auto loc = converter.getCurrentLocation();
  auto callee = getRuntimeFunc<mkRTKey(FailImageStatement)>(loc, builder);
  builder.create<fir::CallOp>(loc, callee, llvm::None);
  genUnreachable(builder, loc);
}

void Fortran::lower::genEventPostStatement(
    Fortran::lower::AbstractConverter &converter,
    const Fortran::parser::EventPostStmt &) {
  // FIXME: There is no runtime call to make for this yet.
  noRuntimeSupport(converter.getCurrentLocation(), "EVENT POST");
}

void Fortran::lower::genEventWaitStatement(
    Fortran::lower::AbstractConverter &converter,
    const Fortran::parser::EventWaitStmt &) {
  // FIXME: There is no runtime call to make for this yet.
  noRuntimeSupport(converter.getCurrentLocation(), "EVENT WAIT");
}

void Fortran::lower::genLockStatement(
    Fortran::lower::AbstractConverter &converter,
    const Fortran::parser::LockStmt &) {
  // FIXME: There is no runtime call to make for this yet.
  noRuntimeSupport(converter.getCurrentLocation(), "LOCK");
}

void Fortran::lower::genUnlockStatement(
    Fortran::lower::AbstractConverter &converter,
    const Fortran::parser::UnlockStmt &) {
  // FIXME: There is no runtime call to make for this yet.
  noRuntimeSupport(converter.getCurrentLocation(), "UNLOCK");
}

void Fortran::lower::genSyncAllStatement(
    Fortran::lower::AbstractConverter &converter,
    const Fortran::parser::SyncAllStmt &) {
  // FIXME: There is no runtime call to make for this yet.
  noRuntimeSupport(converter.getCurrentLocation(), "SYNC ALL");
}

void Fortran::lower::genSyncImagesStatement(
    Fortran::lower::AbstractConverter &converter,
    const Fortran::parser::SyncImagesStmt &) {
  // FIXME: There is no runtime call to make for this yet.
  noRuntimeSupport(converter.getCurrentLocation(), "SYNC IMAGES");
}

void Fortran::lower::genSyncMemoryStatement(
    Fortran::lower::AbstractConverter &converter,
    const Fortran::parser::SyncMemoryStmt &) {
  // FIXME: There is no runtime call to make for this yet.
  noRuntimeSupport(converter.getCurrentLocation(), "SYNC MEMORY");
}

void Fortran::lower::genSyncTeamStatement(
    Fortran::lower::AbstractConverter &converter,
    const Fortran::parser::SyncTeamStmt &) {
  // FIXME: There is no runtime call to make for this yet.
  noRuntimeSupport(converter.getCurrentLocation(), "SYNC TEAM");
}

void Fortran::lower::genPauseStatement(
    Fortran::lower::AbstractConverter &converter,
    const Fortran::parser::PauseStmt &) {
  auto &builder = converter.getFirOpBuilder();
  auto loc = converter.getCurrentLocation();
  auto callee = getRuntimeFunc<mkRTKey(PauseStatement)>(loc, builder);
  builder.create<fir::CallOp>(loc, callee, llvm::None);
}

mlir::Value Fortran::lower::genAssociated(fir::FirOpBuilder &builder,
                                          mlir::Location loc,
                                          mlir::Value pointer,
                                          mlir::Value target) {
  auto func = getRuntimeFunc<mkRTKey(PointerIsAssociatedWith)>(loc, builder);
  auto args = Fortran::lower::createArguments(builder, loc, func.getType(),
                                              pointer, target);
  return builder.create<fir::CallOp>(loc, func, args).getResult(0);
}

mlir::Value Fortran::lower::genCpuTime(fir::FirOpBuilder &builder,
                                       mlir::Location loc) {
  auto func = getRuntimeFunc<mkRTKey(CpuTime)>(loc, builder);
  return builder.create<fir::CallOp>(loc, func, llvm::None).getResult(0);
}

void Fortran::lower::genDateAndTime(fir::FirOpBuilder &builder,
                                    mlir::Location loc,
                                    llvm::Optional<fir::CharBoxValue> date,
                                    llvm::Optional<fir::CharBoxValue> time,
                                    llvm::Optional<fir::CharBoxValue> zone) {
  auto callee = getRuntimeFunc<mkRTKey(DateAndTime)>(loc, builder);
  mlir::Type idxTy = builder.getIndexType();
  mlir::Value zero;
  auto splitArg = [&](llvm::Optional<fir::CharBoxValue> arg,
                      mlir::Value &buffer, mlir::Value &len) {
    if (arg) {
      buffer = arg->getBuffer();
      len = arg->getLen();
    } else {
      if (!zero)
        zero = builder.createIntegerConstant(loc, idxTy, 0);
      buffer = zero;
      len = zero;
    }
  };
  mlir::Value dateBuffer;
  mlir::Value dateLen;
  splitArg(date, dateBuffer, dateLen);
  mlir::Value timeBuffer;
  mlir::Value timeLen;
  splitArg(time, timeBuffer, timeLen);
  mlir::Value zoneBuffer;
  mlir::Value zoneLen;
  splitArg(zone, zoneBuffer, zoneLen);

  llvm::SmallVector<mlir::Value> args{dateBuffer, timeBuffer, zoneBuffer,
                                      dateLen,    timeLen,    zoneLen};
  llvm::SmallVector<mlir::Value> operands;
  for (auto [fst, snd] : llvm::zip(args, callee.getType().getInputs()))
    operands.emplace_back(builder.createConvert(loc, snd, fst));
  builder.create<fir::CallOp>(loc, callee, operands);
}

void Fortran::lower::genRandomInit(fir::FirOpBuilder &builder,
                                   mlir::Location loc, mlir::Value repeatable,
                                   mlir::Value imageDistinct) {
  auto func = getRuntimeFunc<mkRTKey(RandomInit)>(loc, builder);
  auto args = Fortran::lower::createArguments(builder, loc, func.getType(),
                                              repeatable, imageDistinct);
  builder.create<fir::CallOp>(loc, func, args);
}

void Fortran::lower::genRandomNumber(fir::FirOpBuilder &builder,
                                     mlir::Location loc, mlir::Value harvest) {
  auto func = getRuntimeFunc<mkRTKey(RandomNumber)>(loc, builder);
  auto funcTy = func.getType();
  auto sourceFile = fir::factory::locationToFilename(builder, loc);
  auto sourceLine =
      fir::factory::locationToLineNo(builder, loc, funcTy.getInput(2));
  auto args = Fortran::lower::createArguments(builder, loc, funcTy, harvest,
                                              sourceFile, sourceLine);
  builder.create<fir::CallOp>(loc, func, args);
}

void Fortran::lower::genRandomSeed(fir::FirOpBuilder &builder,
                                   mlir::Location loc, int argIndex,
                                   mlir::Value argBox) {
  mlir::FuncOp func;
  // argIndex is the nth (0-origin) argument in declaration order,
  // or -1 if no argument is present.
  switch (argIndex) {
  case -1:
    func = Fortran::lower::getRuntimeFunc<mkRTKey(RandomSeedDefaultPut)>(
        loc, builder);
    builder.create<fir::CallOp>(loc, func);
    return;
  case 0:
    func =
        Fortran::lower::getRuntimeFunc<mkRTKey(RandomSeedSize)>(loc, builder);
    break;
  case 1:
    func = Fortran::lower::getRuntimeFunc<mkRTKey(RandomSeedPut)>(loc, builder);
    break;
  case 2:
    func = Fortran::lower::getRuntimeFunc<mkRTKey(RandomSeedGet)>(loc, builder);
    break;
  default:
    llvm::report_fatal_error("invalid RANDOM_SEED argument index");
  }
  auto funcTy = func.getType();
  auto sourceFile = fir::factory::locationToFilename(builder, loc);
  auto sourceLine =
      fir::factory::locationToLineNo(builder, loc, funcTy.getInput(2));
  auto args = Fortran::lower::createArguments(builder, loc, funcTy, argBox,
                                              sourceFile, sourceLine);
  builder.create<fir::CallOp>(loc, func, args);
}

/// generate runtime call to transfer intrinsic with no size argument
void Fortran::lower::genTransfer(fir::FirOpBuilder &builder, mlir::Location loc,
                                 mlir::Value resultBox, mlir::Value sourceBox,
                                 mlir::Value moldBox) {

  auto func = Fortran::lower::getRuntimeFunc<mkRTKey(Transfer)>(loc, builder);
  auto fTy = func.getType();
  auto sourceFile = fir::factory::locationToFilename(builder, loc);
  auto sourceLine =
      fir::factory::locationToLineNo(builder, loc, fTy.getInput(4));
  auto args = Fortran::lower::createArguments(
      builder, loc, fTy, resultBox, sourceBox, moldBox, sourceFile, sourceLine);
  builder.create<fir::CallOp>(loc, func, args);
}

/// generate runtime call to transfer intrinsic with size argument
void Fortran::lower::genTransferSize(fir::FirOpBuilder &builder,
                                     mlir::Location loc, mlir::Value resultBox,
                                     mlir::Value sourceBox, mlir::Value moldBox,
                                     mlir::Value size) {
  auto func =
      Fortran::lower::getRuntimeFunc<mkRTKey(TransferSize)>(loc, builder);
  auto fTy = func.getType();
  auto sourceFile = fir::factory::locationToFilename(builder, loc);
  auto sourceLine =
      fir::factory::locationToLineNo(builder, loc, fTy.getInput(4));
  auto args =
      Fortran::lower::createArguments(builder, loc, fTy, resultBox, sourceBox,
                                      moldBox, sourceFile, sourceLine, size);
  builder.create<fir::CallOp>(loc, func, args);
}

/// generate system_clock runtime call/s
/// all intrinsic arguments are optional and may appear here as mlir::Value{}
void Fortran::lower::genSystemClock(fir::FirOpBuilder &builder,
                                    mlir::Location loc, mlir::Value count,
                                    mlir::Value rate, mlir::Value max) {
  auto makeCall = [&](mlir::FuncOp func, mlir::Value arg) {
    mlir::Value res = builder.create<fir::CallOp>(loc, func).getResult(0);
    mlir::Value castRes =
        builder.createConvert(loc, fir::dyn_cast_ptrEleTy(arg.getType()), res);
    builder.create<fir::StoreOp>(loc, castRes, arg);
  };
  using Fortran::lower::getRuntimeFunc;
  if (count)
    makeCall(getRuntimeFunc<mkRTKey(SystemClockCount)>(loc, builder), count);
  if (rate)
    makeCall(getRuntimeFunc<mkRTKey(SystemClockCountRate)>(loc, builder), rate);
  if (max)
    makeCall(getRuntimeFunc<mkRTKey(SystemClockCountMax)>(loc, builder), max);
}
