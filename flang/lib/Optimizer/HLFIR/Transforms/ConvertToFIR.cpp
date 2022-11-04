//===- ConvertToFIR.cpp - Convert HLFIR to FIR ----------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
// This file defines a pass to lower HLFIR to FIR
//===----------------------------------------------------------------------===//

#include "flang/Optimizer/Builder/FIRBuilder.h"
#include "flang/Optimizer/Builder/HLFIRTools.h"
#include "flang/Optimizer/Builder/MutableBox.h"
#include "flang/Optimizer/Builder/Runtime/Assign.h"
#include "flang/Optimizer/Builder/Todo.h"
#include "flang/Optimizer/Dialect/FIRDialect.h"
#include "flang/Optimizer/Dialect/FIROps.h"
#include "flang/Optimizer/Dialect/FIRType.h"
#include "flang/Optimizer/HLFIR/HLFIROps.h"
#include "flang/Optimizer/HLFIR/Passes.h"
#include "flang/Optimizer/Support/FIRContext.h"
#include "mlir/Transforms/DialectConversion.h"

namespace hlfir {
#define GEN_PASS_DEF_CONVERTHLFIRTOFIR
#include "flang/Optimizer/HLFIR/Passes.h.inc"
} // namespace hlfir

using namespace mlir;

namespace {

static bool mayAlias(hlfir::FortranEntityLike lhs,
                     hlfir::FortranEntityLike rhs) {
  return true;
}

class AssignOpConversion : public mlir::OpRewritePattern<hlfir::AssignOp> {
public:
  explicit AssignOpConversion(mlir::MLIRContext *ctx) : OpRewritePattern{ctx} {}

  mlir::LogicalResult
  matchAndRewrite(hlfir::AssignOp assignOp,
                  mlir::PatternRewriter &rewriter) const override {
    // ! Note: this assumes allocatable assignment has been dealt with already.
    mlir::Location loc = assignOp->getLoc();
    hlfir::FortranEntityLike lhs(assignOp.getLhs());
    hlfir::FortranEntityLike rhs(assignOp.getRhs());
    auto module = assignOp->getParentOfType<mlir::ModuleOp>();
    fir::FirOpBuilder builder(rewriter, fir::getKindMapping(module));

    if (rhs.getType().isa<hlfir::ExprType>())
      TODO(loc, "hlfir.expr bufferization or inlining");
    auto [rhsExv, rhsCleanUp] =
        hlfir::translateToExtendedValue(loc, builder, rhs);
    auto [lhsExv, lhsCleanUp] =
        hlfir::translateToExtendedValue(loc, builder, lhs);
    if (lhsCleanUp || rhsCleanUp) {
      // This should not be possible outside of the hlfir.expr LHS case. Add a
      // TODO until the hlfir.expr case is dealt with.
      TODO(loc, "cleanup in HLFIR assignment conversion");
    }

    if (lhs.isArray()) {
      // Just use the runtime
      auto to = fir::getBase(builder.createBox(loc, lhsExv));
      auto from = fir::getBase(builder.createBox(loc, rhsExv));
      bool cleanUpTemp = false;
      mlir::Type fromHeapType = fir::HeapType::get(
          fir::unwrapRefType(from.getType().cast<fir::BoxType>().getEleTy()));
      if (mayAlias(rhs, lhs)) {
        /// Use the runtime to make a quick and dirty temp.
        /// Overkill for scalar rhs that could be done in much more clever ways.
        /// Note that temp descriptor must have the allocatable flag set so that
        /// the runtime will attemp to reallocate.
        mlir::Type fromBoxHeapType = fir::BoxType::get(fromHeapType);
        auto fromMutableBox = builder.createTemporary(loc, fromBoxHeapType);
        mlir::Value unallocatedBox = fir::factory::createUnallocatedBox(
            builder, loc, fromBoxHeapType, {});
        builder.create<fir::StoreOp>(loc, unallocatedBox, fromMutableBox);
        fir::runtime::genAssign(builder, loc, fromMutableBox, from);
        cleanUpTemp = true;
        from = builder.create<fir::LoadOp>(loc, fromMutableBox);
      }
      auto toMutableBox = builder.createTemporary(loc, to.getType());
      // As per 10.2.1.2 point 1 (1) polymorphic variables must be allocatable.
      // It is assumed here that they have been reallocated with the dynamic
      // type and that the mutableBox will not be modified.
      builder.create<fir::StoreOp>(loc, to, toMutableBox);
      fir::runtime::genAssign(builder, loc, toMutableBox, from);
      if (cleanUpTemp) {
        mlir::Value addr =
            builder.create<fir::BoxAddrOp>(loc, fromHeapType, from);
        builder.create<fir::FreeMemOp>(loc, addr);
      }
    } else {
      // Assume overlap does not matter for scalar (dealt with memmove for
      // characters).
      // FIXME: may be wrong if type is a derived type with
      // "recursive" allocatable components (but this is what is currently
      // done), in which case an overlap would matter.
      fir::factory::genScalarAssignment(builder, loc, lhsExv, rhsExv);
    }
    rewriter.eraseOp(assignOp);
    return mlir::success();
  }
};

class DeclareOpConversion : public mlir::OpRewritePattern<hlfir::DeclareOp> {
public:
  explicit DeclareOpConversion(mlir::MLIRContext *ctx)
      : OpRewritePattern{ctx} {}

  mlir::LogicalResult
  matchAndRewrite(hlfir::DeclareOp declareOp,
                  mlir::PatternRewriter &rewriter) const override {
    mlir::Location loc = declareOp->getLoc();
    mlir::Value memref = declareOp.getMemref();
    fir::FortranVariableFlagsAttr fortranAttrs;
    if (auto attrs = declareOp.getFortranAttrs())
      fortranAttrs =
          fir::FortranVariableFlagsAttr::get(rewriter.getContext(), *attrs);
    auto originalBase = rewriter
                            .create<fir::DeclareOp>(
                                loc, memref.getType(), memref,
                                declareOp.getShape(), declareOp.getTypeparams(),
                                declareOp.getUniqName(), fortranAttrs)
                            .getResult();
    mlir::Value hlfirBase;
    mlir::Type hlfirBaseType = declareOp.getBase().getType();
    if (hlfirBaseType.isa<fir::BaseBoxType>()) {
      // TODO: optionality ? Need to define what is expected for the created
      // box.
      //
      if (!originalBase.getType().isa<fir::BaseBoxType>()) {
        llvm::SmallVector<mlir::Value> typeParams;
        auto maybeCharType =
            fir::unwrapSequenceType(fir::unwrapPassByRefType(hlfirBaseType))
                .dyn_cast<fir::CharacterType>();
        if (!maybeCharType || maybeCharType.hasDynamicLen())
          typeParams.append(declareOp.getTypeparams().begin(),
                            declareOp.getTypeparams().end());
        hlfirBase = rewriter.create<fir::EmboxOp>(
            loc, hlfirBaseType, originalBase, declareOp.getShape(),
            /*slice=*/mlir::Value{}, typeParams);
      } else {
        // Rebox so that lower bounds are correct.
        // TODO: ensure "ones" lower bounds are respected here.
        hlfirBase = rewriter.create<fir::ReboxOp>(
            loc, hlfirBaseType, originalBase, declareOp.getShape(),
            /*slice=*/mlir::Value{});
      }
    } else if (hlfirBaseType.isa<fir::BoxCharType>()) {
      assert(declareOp.getTypeparams().size() == 1 &&
             "must contain character length");
      hlfirBase = rewriter.create<fir::EmboxCharOp>(
          loc, hlfirBaseType, originalBase, declareOp.getTypeparams()[0]);
    } else {
      if (hlfirBaseType != originalBase.getType()) {
        declareOp.emitOpError()
            << "unhandled HLFIR variable type '" << hlfirBaseType << "'\n";
        return mlir::failure();
      }
      hlfirBase = originalBase;
    }
    rewriter.replaceOp(declareOp, {hlfirBase, originalBase});
    return mlir::success();
  }
};

class DesignateOpConversion
    : public mlir::OpRewritePattern<hlfir::DesignateOp> {
public:
  explicit DesignateOpConversion(mlir::MLIRContext *ctx)
      : OpRewritePattern{ctx} {}

  mlir::LogicalResult
  matchAndRewrite(hlfir::DesignateOp designate,
                  mlir::PatternRewriter &rewriter) const override {
    mlir::Location loc = designate.getLoc();
    auto module = designate->getParentOfType<mlir::ModuleOp>();
    fir::FirOpBuilder builder(rewriter, fir::getKindMapping(module));

    if (designate.getComponent() || designate.getComplexPart() ||
        !designate.getSubstring().empty()) {
      // build path.
      TODO(loc, "hlfir::designate with complex part or substring or component");
    }

    hlfir::FortranEntityLike baseEntity(designate.getMemref());
    if (baseEntity.isMutableBox())
      TODO(loc, "hlfir::designate load of pointer or allocatable");

    auto [base, shape] = hlfir::genRawBaseAndShape(loc, builder, baseEntity);
    if (designate.getResult().getType().isa<fir::BoxType>()) {
      // Generate embox or rebox.
      if (designate.getIndices().empty())
        TODO(loc, "hlfir::designate whole part");
      // Otherwise, this is an array section with triplets.
      llvm::SmallVector<mlir::Value> triples;
      auto undef = builder.create<fir::UndefOp>(loc, builder.getIndexType());
      auto subscripts = designate.getIndices();
      unsigned i = 0;
      for (auto isTriplet : designate.getIsTriplet()) {
        triples.push_back(subscripts[i++]);
        if (isTriplet) {
          triples.push_back(subscripts[i++]);
          triples.push_back(subscripts[i++]);
        } else {
          triples.push_back(undef);
          triples.push_back(undef);
        }
      }
      mlir::Value slice = builder.create<fir::SliceOp>(
          loc, triples, /*path=*/mlir::ValueRange{});
      if (baseEntity.hasLengthParameters())
        TODO(loc, "hlfir::designate to entity with length parameters");
      llvm::SmallVector<mlir::Type> resultType{designate.getResult().getType()};
      mlir::Value resultBox;
      if (base.getType().isa<fir::BoxType>())
        resultBox =
            builder.create<fir::ReboxOp>(loc, resultType, base, shape, slice);
      else
        resultBox =
            builder.create<fir::EmboxOp>(loc, resultType, base, shape, slice);
      rewriter.replaceOp(designate, resultBox);
      return mlir::success();
    }
    // Indexing a single element (use fir.array_coor of fir.coordinate_of).
    if (designate.getIndices().empty()) {
      // generate fir.coordinate_of.
      TODO(loc, "hlfir::designate to fir.coordinate_of");
    }
    // Generate fir.array_coor
    auto arrayCoor = builder.create<fir::ArrayCoorOp>(
        loc, designate.getResult().getType(), base, shape,
        /*slice=*/mlir::Value{}, designate.getIndices(),
        /*typeParams=*/mlir::ValueRange{});
    rewriter.replaceOp(designate, arrayCoor.getResult());
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
    patterns
        .insert<DeclareOpConversion, AssignOpConversion, DesignateOpConversion>(
            context);
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
