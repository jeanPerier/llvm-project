//===-- PreCGRewrite.cpp --------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Coding style: https://mlir.llvm.org/getting_started/DeveloperGuide/
//
//===----------------------------------------------------------------------===//

#include "CGOps.h"
#include "PassDetail.h"
#include "flang/Optimizer/CodeGen/CodeGen.h"
#include "flang/Optimizer/Dialect/FIRDialect.h"
#include "flang/Optimizer/Dialect/FIROps.h"
#include "flang/Optimizer/Dialect/FIRType.h"
#include "flang/Optimizer/Support/FIRContext.h"
#include "mlir/Transforms/DialectConversion.h"
#include "llvm/ADT/STLExtras.h"

//===----------------------------------------------------------------------===//
// Codegen rewrite: rewriting of subgraphs of ops
//===----------------------------------------------------------------------===//

using namespace fir;

#define DEBUG_TYPE "flang-codegen-rewrite"

static void populateShape(llvm::SmallVectorImpl<mlir::Value> &vec,
                          ShapeOp shape) {
  vec.append(shape.extents().begin(), shape.extents().end());
}

// Operands of fir.shape_shift split into two vectors.
static void populateShapeAndShift(llvm::SmallVectorImpl<mlir::Value> &shapeVec,
                                  llvm::SmallVectorImpl<mlir::Value> &shiftVec,
                                  ShapeShiftOp shift) {
  auto endIter = shift.pairs().end();
  for (auto i = shift.pairs().begin(); i != endIter;) {
    shiftVec.push_back(*i++);
    shapeVec.push_back(*i++);
  }
}

namespace {

/// Convert fir.embox to the extended form where necessary.
class EmboxConversion : public mlir::OpRewritePattern<EmboxOp> {
public:
  using OpRewritePattern::OpRewritePattern;

  mlir::LogicalResult
  matchAndRewrite(EmboxOp embox,
                  mlir::PatternRewriter &rewriter) const override {
    auto shapeVal = embox.getShape();
    // If the embox does not include a shape, then do not convert it
    if (shapeVal)
      return rewriteDynamicShape(embox, rewriter, shapeVal);
    if (auto boxTy = embox.getType().dyn_cast<BoxType>())
      if (auto seqTy = boxTy.getEleTy().dyn_cast<SequenceType>())
        if (seqTy.hasConstantShape())
          return rewriteStaticShape(embox, rewriter, seqTy);
    return mlir::failure();
  }

  mlir::LogicalResult rewriteStaticShape(EmboxOp embox,
                                         mlir::PatternRewriter &rewriter,
                                         SequenceType seqTy) const {
    auto loc = embox.getLoc();
    llvm::SmallVector<mlir::Value, 8> shapeOpers;
    auto idxTy = rewriter.getIndexType();
    for (auto ext : seqTy.getShape()) {
      auto iAttr = rewriter.getIndexAttr(ext);
      auto extVal = rewriter.create<mlir::ConstantOp>(loc, idxTy, iAttr);
      shapeOpers.push_back(extVal);
    }
    auto xbox = rewriter.create<cg::XEmboxOp>(
        loc, embox.getType(), embox.memref(), shapeOpers, llvm::None,
        llvm::None, llvm::None, embox.lenParams());
    LLVM_DEBUG(llvm::dbgs() << "rewriting " << embox << " to " << xbox << '\n');
    rewriter.replaceOp(embox, xbox.getOperation()->getResults());
    return mlir::success();
  }

  mlir::LogicalResult rewriteDynamicShape(EmboxOp embox,
                                          mlir::PatternRewriter &rewriter,
                                          mlir::Value shapeVal) const {
    auto loc = embox.getLoc();
    auto shapeOp = dyn_cast<ShapeOp>(shapeVal.getDefiningOp());
    llvm::SmallVector<mlir::Value, 8> shapeOpers;
    llvm::SmallVector<mlir::Value, 8> shiftOpers;
    if (shapeOp) {
      populateShape(shapeOpers, shapeOp);
    } else {
      auto shiftOp = dyn_cast<ShapeShiftOp>(shapeVal.getDefiningOp());
      assert(shiftOp && "shape is neither fir.shape nor fir.shape_shift");
      populateShapeAndShift(shapeOpers, shiftOpers, shiftOp);
    }
    llvm::SmallVector<mlir::Value, 8> sliceOpers;
    llvm::SmallVector<mlir::Value, 8> subcompOpers;
    if (auto s = embox.getSlice())
      if (auto sliceOp = dyn_cast_or_null<SliceOp>(s.getDefiningOp())) {
        sliceOpers.append(sliceOp.triples().begin(), sliceOp.triples().end());
        subcompOpers.append(sliceOp.fields().begin(), sliceOp.fields().end());
      }
    auto xbox = rewriter.create<cg::XEmboxOp>(
        loc, embox.getType(), embox.memref(), shapeOpers, shiftOpers,
        sliceOpers, subcompOpers, embox.lenParams());
    LLVM_DEBUG(llvm::dbgs() << "rewriting " << embox << " to " << xbox << '\n');
    rewriter.replaceOp(embox, xbox.getOperation()->getResults());
    return mlir::success();
  }
};

/// Convert all fir.array_coor to the extended form.
class ArrayCoorConversion : public mlir::OpRewritePattern<ArrayCoorOp> {
public:
  using OpRewritePattern::OpRewritePattern;

  mlir::LogicalResult
  matchAndRewrite(ArrayCoorOp arrCoor,
                  mlir::PatternRewriter &rewriter) const override {
    auto loc = arrCoor.getLoc();
    auto shapeVal = arrCoor.shape();
    auto shapeOp = dyn_cast<ShapeOp>(shapeVal.getDefiningOp());
    llvm::SmallVector<mlir::Value, 8> shapeOpers;
    llvm::SmallVector<mlir::Value, 8> shiftOpers;
    if (shapeOp) {
      populateShape(shapeOpers, shapeOp);
    } else if (auto shiftOp =
                   dyn_cast<ShapeShiftOp>(shapeVal.getDefiningOp())) {
      populateShapeAndShift(shapeOpers, shiftOpers, shiftOp);
    } else {
      return mlir::failure();
    }
    llvm::SmallVector<mlir::Value, 8> sliceOpers;
    llvm::SmallVector<mlir::Value, 8> subcompOpers;
    if (auto s = arrCoor.slice())
      if (auto sliceOp = dyn_cast_or_null<SliceOp>(s.getDefiningOp())) {
        sliceOpers.append(sliceOp.triples().begin(), sliceOp.triples().end());
        subcompOpers.append(sliceOp.fields().begin(), sliceOp.fields().end());
      }
    mlir::Value sourceBox;
    auto addr = arrCoor.memref();
    if (auto boxTy = addr.getType().dyn_cast<fir::BoxType>()) {
      sourceBox = addr;
      auto refTy = fir::ReferenceType::get(boxTy.getEleTy());
      addr = rewriter.create<fir::BoxAddrOp>(loc, refTy, addr);
    }

    auto xArrCoor = rewriter.create<cg::XArrayCoorOp>(
        loc, arrCoor.getType(), addr, shapeOpers, shiftOpers, sliceOpers,
        subcompOpers, arrCoor.indices(), arrCoor.lenParams(), sourceBox);
    LLVM_DEBUG(llvm::dbgs()
               << "rewriting " << arrCoor << " to " << xArrCoor << '\n');
    rewriter.replaceOp(arrCoor, xArrCoor.getOperation()->getResults());
    return mlir::success();
  }
};

/// Convert FIR structured control flow ops to CFG ops.
class CodeGenRewrite : public CodeGenRewriteBase<CodeGenRewrite> {
public:
  void runOn(mlir::Operation *op, mlir::Region &region) {
    auto &context = getContext();
    mlir::OpBuilder rewriter(&context);
    mlir::ConversionTarget target(context);
    target.addLegalDialect<FIROpsDialect, FIRCodeGenDialect,
                           mlir::StandardOpsDialect>();
    target.addIllegalOp<ArrayCoorOp>();
    target.addDynamicallyLegalOp<EmboxOp>([](EmboxOp embox) {
      return !(embox.getShape() ||
               embox.getType().cast<BoxType>().getEleTy().isa<SequenceType>());
    });
    mlir::OwningRewritePatternList patterns;
    patterns.insert<EmboxConversion, ArrayCoorConversion>(&context);
    if (mlir::failed(
            mlir::applyPartialConversion(op, target, std::move(patterns)))) {
      mlir::emitError(mlir::UnknownLoc::get(&context),
                      "error in running the pre-codegen conversions");
      signalPassFailure();
    }
    // Erase any residual.
    simplifyRegion(region);
  }

  void runOnOperation() override final {
    // Call runOn on all top level regions that may contain emboxOp/arrayCoorOp.
    auto mod = getOperation();
    for (auto func : mod.getOps<mlir::FuncOp>())
      runOn(func, func.getBody());
    for (auto global : mod.getOps<fir::GlobalOp>())
      runOn(global, global.getRegion());
  }

  // Clean up the region.
  void simplifyRegion(mlir::Region &region) {
    for (auto &block : region.getBlocks())
      for (auto &op : block.getOperations()) {
        if (op.getNumRegions() != 0)
          for (auto &reg : op.getRegions())
            simplifyRegion(reg);
        maybeEraseOp(&op);
      }

    for (auto *op : opsToErase)
      op->erase();
    opsToErase.clear();
  }

  void maybeEraseOp(mlir::Operation *op) {
    if (!op)
      return;

    // Erase any embox that was replaced.
    if (auto embox = dyn_cast<EmboxOp>(op))
      if (embox.getShape()) {
        assert(op->use_empty());
        opsToErase.push_back(op);
      }

    // Erase all fir.array_coor.
    if (isa<ArrayCoorOp>(op)) {
      assert(op->use_empty());
      opsToErase.push_back(op);
    }

    // Erase all fir.shape, fir.shape_shift, and fir.slice ops.
    if (isa<ShapeOp>(op)) {
      assert(op->use_empty());
      opsToErase.push_back(op);
    }
    if (isa<ShapeShiftOp>(op)) {
      assert(op->use_empty());
      opsToErase.push_back(op);
    }
    if (isa<SliceOp>(op)) {
      assert(op->use_empty());
      opsToErase.push_back(op);
    }
  }

private:
  std::vector<mlir::Operation *> opsToErase;
};

} // namespace

/// Convert FIR's structured control flow ops to CFG ops.  This conversion
/// enables the `createLowerToCFGPass` to transform these to CFG form.
std::unique_ptr<mlir::Pass> fir::createFirCodeGenRewritePass() {
  return std::make_unique<CodeGenRewrite>();
}
