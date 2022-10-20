//===- HLFIRTools.cpp -- HLFIR tools unit tests ---------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "flang/Optimizer/Builder/HLFIRTools.h"
#include "flang/Optimizer/Builder/FIRBuilder.h"
#include "gtest/gtest.h"
#include "flang/Optimizer/Builder/BoxValue.h"
#include "flang/Optimizer/Support/InitFIR.h"
#include "flang/Optimizer/Support/KindMapping.h"

struct HLFIRToolsTest : public testing::Test {
public:
  void SetUp() override {
    fir::support::loadDialects(context);

    llvm::ArrayRef<fir::KindTy> defs;
    fir::KindMapping kindMap(&context, defs);
    mlir::OpBuilder builder(&context);
    auto loc = builder.getUnknownLoc();

    // Set up a Module with a dummy function operation inside.
    // Set the insertion point in the function entry block.
    mlir::ModuleOp mod = builder.create<mlir::ModuleOp>(loc);
    mlir::func::FuncOp func = mlir::func::FuncOp::create(
        loc, "func1", builder.getFunctionType(llvm::None, llvm::None));
    auto *entryBlock = func.addEntryBlock();
    mod.push_back(mod);
    builder.setInsertionPointToStart(entryBlock);

    firBuilder = std::make_unique<fir::FirOpBuilder>(mod, kindMap);
  }

  mlir::Value createDeclare(fir::ExtendedValue exv) {
    // TODO
    return {};
  }

  fir::FirOpBuilder &getBuilder() { return *firBuilder; }

  mlir::MLIRContext context;
  std::unique_ptr<fir::FirOpBuilder> firBuilder;
};


TEST_F(FIRBuilderTest, genIfThen) {
}
