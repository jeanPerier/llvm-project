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
//#include "mlir/Interfaces/InferTypeOpInterface.h"
//#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "mlir/Transforms/DialectConversion.h"
#include "mlir/IR/BlockAndValueMapping.h"
#include "flang/Optimizer/Builder/Todo.h"


// Note: the patte
bool inline isForall(fir::DoLoopOp loop) {
  return loop->hasAttr("fir.forall");
}

void inline removeForallLabel(fir::DoLoopOp loop) {
  loop->removeAttr("fir.forall");
}

bool inline isTopLevelForall(fir::DoLoopOp loop) {
  if (!isForall(loop))
    return false;
  auto* owner = loop->getParentOp();
  while (owner) {
    if (auto parentLoop = mlir::dyn_cast<fir::DoLoopOp>(owner))
      if (isForall(parentLoop))
        return false;
    owner = owner->getParentOp();
  }
  return true;
}


namespace {


// Note: memory dependencies are dropped on the floor here...

//llvm::SmallPtrSet<mlir::Operation*> gatherOpsToClone(mlir::Operation* leafOp, mlir::Operation* parentOp) {
//  llvm::SmallPtrSet<mlir::Operation*> toBeCloned;
//  llvm::SmallVector<mlir::Operation*> toBeVisited{leafOp};
//  // Start by adding adding all parentOp until parentOp;
//  mlir::Operation* owner = leafOp->getParentOp();
//  while (owner && owner != parentOp) {
//    toBeVisited.push_back(owner);
//    owner = owner->getParentOp();
//  }
//  while (!toBeVisited.empty()) {
//    mlir::Operation* op = toBeVisited.pop_back_val();
//    if (toBeCloned.contains(op))
//     continue;
//    toBeCloned.insert(op);
//    for (mlir::Value operand = op->getOperands())
//      if (mlir::Operation* definingOp = operand.getDefiningOp())
//        if (parentOp->isProperAncestor(definingOp))
//          toBeVisited.push_back(definingOp);
//  }
//}

void cloneAssignmentIntoItsOwnLoopNest(fir::AssignOp assignment, fir::DoLoopOp loop, mlir::PatternRewriter& rewriter) {
  // How can something be cloned while still preserving the order....
  //auto toBeCloned;
  mlir::BlockAndValueMapping mapper;
  auto* newLoop = rewriter.clone(*loop.getOperation(), mapper);
  assert(newLoop && "failed to clone");
  int currentPosition = -1;
  int position = -1;
  loop->walk([&](mlir::Operation *op) {
    if (auto thatAssignment = mlir::dyn_cast<fir::AssignOp>(op)) {
      ++currentPosition;
      if (op == assignment.getOperation())
        position = currentPosition;
    }
  });
  
  llvm::SmallVector<fir::AssignOp> toErase;
  llvm::SmallVector<fir::DoLoopOp> removeLabel;
  currentPosition = -1;
  newLoop->walk([&](mlir::Operation *op) {
    if (auto assignment = mlir::dyn_cast<fir::AssignOp>(op)) {
      ++currentPosition;
      if (currentPosition != position)
        toErase.push_back(assignment);
    } else if (auto doLoop = mlir::dyn_cast<fir::DoLoopOp>(op)) {
      if (isForall(doLoop))
        removeLabel.push_back(doLoop);
    }
  });
  for (fir::AssignOp otherAssignment: toErase)
    rewriter.eraseOp(otherAssignment);
  for (fir::DoLoopOp newNestedLoop : removeLabel)
    removeForallLabel(newNestedLoop);
  removeForallLabel(mlir::cast<fir::DoLoopOp>(newLoop));
  // RUN DCE manually ??
  // walk + clone ?
}

class ForallConversion : public mlir::OpRewritePattern<fir::DoLoopOp> {
public:
  explicit ForallConversion(mlir::MLIRContext *ctx)
      : OpRewritePattern{ctx} {}

  mlir::LogicalResult
  matchAndRewrite(fir::DoLoopOp loop,
                  mlir::PatternRewriter &rewriter) const override {
//    auto *op = amend.getOperation();
//    rewriter.setInsertionPoint(op);
//    auto loc = amend.getLoc();
//    auto undef = rewriter.create<UndefOp>(loc, amend.getType());
//    rewriter.replaceOp(amend, undef.getResult());
    // GATHER assignments:
    llvm::SmallVector<fir::AssignOp> assignments;
    loop->walk([&](mlir::Operation *op) {
      if (auto assignment = mlir::dyn_cast<fir::AssignOp>(op))
        assignments.push_back(assignment);
    });
    rewriter.setInsertionPoint(loop);
    // STUPID TEST: let's split assignments into their own FORALL (note:
    // not valid if where depends on previously assigned values).
    for (fir::AssignOp assignment : assignments)
      cloneAssignmentIntoItsOwnLoopNest(assignment, loop, rewriter);
    // Note: this assumes FORALL do not have results.
    rewriter.eraseOp(loop);
    return mlir::success();
  }
};
}

namespace {
struct ForallSplitPass final : public fir::ForallSplitBase<ForallSplitPass> {
  void runOnOperation() override {
    auto func = getOperation();
    auto *context = &getContext();
    mlir::RewritePatternSet patterns(context);
    patterns.insert<ForallConversion>(context);
    mlir::ConversionTarget target(*context);
    target.addDynamicallyLegalOp<fir::DoLoopOp>(
        [](fir::DoLoopOp loop) { return !isTopLevelForall(loop); });
    target.markUnknownOpDynamicallyLegal([](mlir::Operation*) {return true;});
    if (mlir::failed(
            mlir::applyPartialConversion(func, target, std::move(patterns)))) {
      mlir::emitError(mlir::UnknownLoc::get(context),
                      "failure in forall split pass");
      signalPassFailure();
    }
  }
};

} // namespace

std::unique_ptr<mlir::Pass> fir::createForallSplitPass() {
  return std::make_unique<ForallSplitPass>();
}
