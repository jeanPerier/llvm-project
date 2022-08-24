//===- ShapeInference.cpp - Propagate shape information -===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// TBD
//
//===----------------------------------------------------------------------===//

#include "PassDetail.h"
#include "flang/Optimizer/Dialect/FIRDialect.h"
#include "flang/Optimizer/Dialect/FIROps.h"
#include "flang/Optimizer/Dialect/FIRType.h"
#include "flang/Optimizer/Transforms/Passes.h"
#include "mlir/Dialect/Arithmetic/IR/Arithmetic.h"
//#include "mlir/Interfaces/InferTypeOpInterface.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"


namespace {
} // namespace

namespace {
struct ShapeInferencePass final : public fir::ShapeInferenceBase<ShapeInferencePass> {
  void runOnOperation() override {}
};

} // namespace

std::unique_ptr<mlir::Pass> fir::createShapeInferencePass() {
  return std::make_unique<ShapeInferencePass>();
}
