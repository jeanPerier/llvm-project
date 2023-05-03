//===- InlineElementals.cpp - Inline chained hlfir.elemental ops ----------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
// Chained elemental operations like a + b + c can inline the first elemental
// at the hlfir.apply in the body of the second one (as described in
// docs/HighLevelFIR.md). This has to be done in a pass rather than in lowering
// so that it happens after the HLFIR intrinsic simplification pass.
//===----------------------------------------------------------------------===//

#include "flang/Optimizer/Builder/FIRBuilder.h"
#include "flang/Optimizer/Builder/HLFIRTools.h"
#include "flang/Optimizer/Dialect/Support/FIRContext.h"
#include "flang/Optimizer/HLFIR/HLFIROps.h"
#include "flang/Optimizer/HLFIR/Passes.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Support/LLVM.h"
#include "mlir/Transforms/DialectConversion.h"
#include "llvm/ADT/TypeSwitch.h"
#include <iterator>

namespace hlfir {
#define GEN_PASS_DEF_INLINEELEMENTALS
#include "flang/Optimizer/HLFIR/Passes.h.inc"
} // namespace hlfir

// inline sourceBlock into targetBlock - inserting the new operations
// after insertAfterOp
static void inlineBySplice(mlir::Block *sourceBlock, mlir::Block *targetBlock,
                           mlir::Operation *insertAfterOp) {
  assert(insertAfterOp->getBlock() == targetBlock);

  // get an iterator for insertAfterOp
  decltype(targetBlock->begin()) opIterator;
  for (auto it = targetBlock->begin(); it != targetBlock->end(); ++it) {
    mlir::Operation *op = &*it;
    if (op == insertAfterOp) {
      opIterator = it;
      break;
    }
  }

  // inline
  targetBlock->getOperations().splice(opIterator, sourceBlock->getOperations());
}

// apply IRMapping to operation arguments in the given block
static void mapArgumentValues(mlir::Block *block,
                              const mlir::IRMapping &mapper) {
  for (mlir::Operation &op : block->getOperations())
    for (mlir::OpOperand &operand : op.getOpOperands())
      if (mlir::Value mappedVal = mapper.lookupOrNull(operand.get()))
        operand.set(mappedVal);
}

/// If the elemental has only two uses and those two are an apply operation and
/// a destory operation, return those two, otherwise return {}
static std::optional<std::pair<hlfir::ApplyOp, hlfir::DestroyOp>>
getTwoUses(hlfir::ElementalOp elemental) {
  mlir::Operation::user_range users = elemental->getUsers();
  // don't inline anything with more than one use (plus hfir.destroy)
  // We can only inline once because we have to use splice() (see
  // InlineElementalConversion::matchAndRewrite)
  if (std::distance(users.begin(), users.end()) != 2) {
    return std::nullopt;
  }

  hlfir::ApplyOp apply;
  hlfir::DestroyOp destroy;
  for (mlir::Operation *user : users)
    mlir::TypeSwitch<mlir::Operation *, void>(user)
        .Case([&](hlfir::ApplyOp op) { apply = op; })
        .Case([&](hlfir::DestroyOp op) { destroy = op; });

  if (!apply || !destroy)
    return std::nullopt;
  return std::pair{apply, destroy};
}

namespace {
class InlineElementalConversion
    : public mlir::OpRewritePattern<hlfir::ElementalOp> {
public:
  using mlir::OpRewritePattern<hlfir::ElementalOp>::OpRewritePattern;

  mlir::LogicalResult
  matchAndRewrite(hlfir::ElementalOp elemental,
                  mlir::PatternRewriter &rewriter) const override {
    // the option must not be {}, otherwise the op would already be "legal" and
    // MLIR would not try to apply this transformation to "legalise" the op:
    // see target.addDynamicallyLegalOp<hlfir::ElementalOp> in runOnOperaion()
    std::optional<std::pair<hlfir::ApplyOp, hlfir::DestroyOp>> maybeTuple =
        getTwoUses(elemental);
    assert(maybeTuple &&
           "if the tuple is empty, this operation should be marked legal");
    auto [apply, destroy] = *maybeTuple;

    assert(elemental.getRegion().hasOneBlock() &&
           "expect elemental region to have one block");
    mlir::Block *sourceBlock = &elemental.getRegion().back();
    mlir::Block *targetBlock = apply->getBlock();

    // the terminator operation for a hlfir.elemental is always a hlfir.yield
    auto yield = mlir::cast<hlfir::YieldElementOp>(sourceBlock->back());

    // Inline using splice so that the mlir operations are not re-instantiated.
    //
    // When operations are erased, this is recorded and erasures are done in a
    // batch at the end of the pass. If erased operations are cloned, the
    // state saying they are erased is not reproduced, resulting in the clones
    // not being erased. We can't avoid cloning them because there isn't an easy
    // way to keep track of which operations we have erased between multiple
    // applications of this transformation because matchAndRewrite cannot store
    // state.
    //
    // To get around this, we inline by splicing the instructions from the
    // source block's operations list into the target block's operation list.
    // This way the exact same operation instances are used, and all state is
    // preserved.
    inlineBySplice(sourceBlock, targetBlock, apply);

    // map inlined elemental block arguments to the arguments passed to the
    // hlfir.apply
    mlir::IRMapping mapper;
    mapper.map(elemental.getIndices(), apply.getIndices());
    mapArgumentValues(targetBlock, mapper);

    // remove the old elemental and all of the bookkeeping
    rewriter.replaceAllUsesWith(apply.getResult(), yield.getElementValue());
    rewriter.eraseOp(yield);
    rewriter.eraseOp(apply);
    rewriter.eraseOp(destroy);
    rewriter.eraseOp(elemental);

    return mlir::success();
  }
};

class InlineElementalsPass
    : public hlfir::impl::InlineElementalsBase<InlineElementalsPass> {
public:
  void runOnOperation() override {
    mlir::func::FuncOp func = getOperation();
    mlir::MLIRContext *context = &getContext();
    mlir::RewritePatternSet patterns(context);
    patterns.insert<InlineElementalConversion>(context);

    mlir::ConversionTarget target(*context);
    target.markUnknownOpDynamicallyLegal(
        [](mlir::Operation *) { return true; });
    target.addDynamicallyLegalOp<hlfir::ElementalOp>(
        [](hlfir::ElementalOp elemental) { return !getTwoUses(elemental); });

    if (mlir::failed(
            mlir::applyFullConversion(func, target, std::move(patterns)))) {
      mlir::emitError(func->getLoc(), "failure in HLFIR elemental inlining");
      signalPassFailure();
    }
  }
};
} // namespace

std::unique_ptr<mlir::Pass> hlfir::createInlineElementalsPass() {
  return std::make_unique<InlineElementalsPass>();
}
