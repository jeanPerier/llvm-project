//===- ConvertToFIR.cpp - Convert HLFIR to FIR ----------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
// This file defines a pass to lower HLFIR to FIR
//===----------------------------------------------------------------------===//

#include "flang/Optimizer/Builder/Todo.h"
#include "flang/Optimizer/Builder/HLFIRTools.h"
#include "flang/Optimizer/Builder/FIRBuilder.h"
#include "flang/Optimizer/Dialect/FIRDialect.h"
#include "flang/Optimizer/Dialect/FIROps.h"
#include "flang/Optimizer/Dialect/FIRType.h"
#include "flang/Optimizer/HLFIR/HLFIROps.h"
#include "flang/Optimizer/HLFIR/Passes.h"
// TODO: needs to move pass into its own library because of this getKindMapping dependency...
// Otherwise, there is a circular dependency.
#include "flang/Optimizer/Support/FIRContext.h"
#include "mlir/Transforms/DialectConversion.h"

namespace hlfir {
#define GEN_PASS_DEF_CONVERTHLFIRTOFIR
#include "flang/Optimizer/HLFIR/Passes.h.inc"
} // namespace hlfir

using namespace mlir;

namespace {

class AssignOpConversion : public mlir::OpRewritePattern<hlfir::AssignOp> {
public:
  explicit AssignOpConversion(mlir::MLIRContext *ctx) : OpRewritePattern{ctx} {}

  mlir::LogicalResult
  matchAndRewrite(hlfir::AssignOp assignOp,
                  mlir::PatternRewriter &rewriter) const override {
    llvm::errs() << "rewritting assignOp\n";
    mlir::Location loc = assignOp->getLoc();
    hlfir::FortranEntityLike lhs(assignOp.getLhs()); 
    hlfir::FortranEntityLike rhs(assignOp.getRhs()); 
    auto module = assignOp->getParentOfType<mlir::ModuleOp>();
    fir::FirOpBuilder firOpBuilder(rewriter, fir::getKindMapping(module));

    if (rhs.getType().isa<hlfir::ExprType>())
      TODO(loc, "hlfir.expr bufferization");
    auto [lhsExv, lhsCleanUp] = hlfir::translateToExtendedValue(loc, firOpBuilder, lhs);
    auto [rhsExv, rhsCleanUp] = hlfir::translateToExtendedValue(loc, firOpBuilder, rhs);
    llvm::errs() << "reached 1\n";
    if (lhsCleanUp || rhsCleanUp) {
     // This should not be possible outside of the hlfir.expr LHS case. Add a TODO until the hlfir.expr case is dealt with.
     TODO(loc, "cleanup in HLFIR assignment conversion");
    }
    if (lhs.isArray()) {
      TODO(loc, "HLFIR array assignment conversion to FIR");
      // TODO: for derived type assignments where the RHS is in memory and where we anyway call the runtime at the scalar level, Call the runtime at the array level directly instead.
    }
    llvm::errs() << "reached 2\n";
    // Scalar assignment.
    // Assume overlap does not matter for scalar (dealt with memmove for characters).
    // FIXME: may be wrong if type is a derived type with
    // "recursive" allocatable components (but this is what is currently done), in which case an overlap would matter.
    fir::factory::genScalarAssignment(firOpBuilder, loc, lhsExv, rhsExv);
    llvm::errs() << "reached 3\n";
    rewriter.eraseOp(assignOp);
    return mlir::success();
  }
};

class DeclareOpConversion : public mlir::OpRewritePattern<hlfir::DeclareOp> {
public:
  explicit DeclareOpConversion(mlir::MLIRContext *ctx) : OpRewritePattern{ctx} {}

  mlir::LogicalResult
  matchAndRewrite(hlfir::DeclareOp declareOp,
                  mlir::PatternRewriter &rewriter) const override {
    mlir::Location loc = declareOp->getLoc();
    mlir::Value memref = declareOp.getMemref();
    fir::FortranVariableFlagsAttr fortranAttrs;
    if (auto attrs = declareOp.getFortranAttrs())
      fortranAttrs = fir::FortranVariableFlagsAttr::get(rewriter.getContext(), *attrs);
    auto originalBase = rewriter.create<fir::DeclareOp>(loc, memref.getType(), memref, declareOp.getShape(), declareOp.getTypeparams(), declareOp.getUniqName(), fortranAttrs).getResult();
    mlir::Value hlfirBase;
    mlir::Type hlfirBaseType = declareOp.getBase().getType();
    if (hlfirBaseType.isa<fir::BaseBoxType>()) {
      // TODO: optionality ? Need to define what is expected for the created box.
      //
      if (!originalBase.getType().isa<fir::BaseBoxType>()) {
        llvm::SmallVector<mlir::Value> typeParams;
        auto maybeCharType = fir::unwrapSequenceType(fir::unwrapPassByRefType(hlfirBaseType)).dyn_cast<fir::CharacterType>();
        if (!maybeCharType || maybeCharType.hasDynamicLen())
          typeParams.append(declareOp.getTypeparams().begin(), declareOp.getTypeparams().end());
        hlfirBase = rewriter.create<fir::EmboxOp>(loc, hlfirBaseType, originalBase, declareOp.getShape(), /*slice=*/mlir::Value{}, typeParams);
      } else {
        // Rebox so that lower bounds are correct.
        // TODO: ensure "ones" lower bounds are respected here.
        hlfirBase = rewriter.create<fir::ReboxOp>(loc, hlfirBaseType, originalBase, declareOp.getShape(), /*slice=*/mlir::Value{});
      }
    } else if (hlfirBaseType.isa<fir::BoxCharType>()) {
      assert(declareOp.getTypeparams().size() == 1 && "must contain character length");
      hlfirBase = rewriter.create<fir::EmboxCharOp>(loc, hlfirBaseType, originalBase, declareOp.getTypeparams()[0]);
    } else {
      if (hlfirBaseType != originalBase.getType()) {
        declareOp.emitOpError() << "unhandled HLFIR variable type '"<< hlfirBaseType << "'\n";
        return mlir::failure();
      }
      hlfirBase = originalBase;
    }
    rewriter.replaceOp(declareOp, {hlfirBase, originalBase});
    return mlir::success();
  }
};

class ConvertHLFIRtoFIR
    : public hlfir::impl::ConvertHLFIRtoFIRBase<ConvertHLFIRtoFIR> {
public:
  void runOnOperation() override {
    auto func = this->getOperation();
    auto *context = &getContext();
    mlir::RewritePatternSet patterns(context);
    patterns.insert<DeclareOpConversion, AssignOpConversion>(context);
    mlir::ConversionTarget target(*context);
    target.addIllegalDialect<hlfir::hlfirDialect>();
    target.markUnknownOpDynamicallyLegal(
        [](mlir::Operation *) { return true; });
    if (mlir::failed(
            mlir::applyPartialConversion(func, target, std::move(patterns)))) {
      mlir::emitError(mlir::UnknownLoc::get(context),
                      "failure in HLFIR to FIR conversion pass");
      signalPassFailure();
    }
  }
};

} // namespace

std::unique_ptr<mlir::Pass> hlfir::createConvertHLFIRtoFIRPass() {
  return std::make_unique<ConvertHLFIRtoFIR>();
}
