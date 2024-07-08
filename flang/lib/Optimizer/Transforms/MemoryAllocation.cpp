//===- MemoryAllocation.cpp -----------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "flang/Optimizer/Dialect/FIRDialect.h"
#include "flang/Optimizer/Dialect/FIROps.h"
#include "flang/Optimizer/Dialect/FIRType.h"
#include "flang/Optimizer/Transforms/Passes.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/Diagnostics.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/DialectConversion.h"
#include "mlir/Transforms/Passes.h"
#include "llvm/ADT/TypeSwitch.h"

namespace fir {
#define GEN_PASS_DEF_MEMORYALLOCATIONOPT
#include "flang/Optimizer/Transforms/Passes.h.inc"
} // namespace fir

#define DEBUG_TYPE "flang-memory-allocation-opt"

// Number of elements in an array does not determine where it is allocated.
static constexpr std::size_t unlimitedArraySize = ~static_cast<std::size_t>(0);

namespace {
class ReturnAnalysis {
public:
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(ReturnAnalysis)

  ReturnAnalysis(mlir::Operation *op) {
    if (auto func = mlir::dyn_cast<mlir::func::FuncOp>(op))
      for (mlir::Block &block : func)
        for (mlir::Operation &i : block)
          if (mlir::isa<mlir::func::ReturnOp>(i)) {
            returnMap[op].push_back(&i);
            break;
          }
  }

  llvm::SmallVector<mlir::Operation *> getReturns(mlir::Operation *func) const {
    auto iter = returnMap.find(func);
    if (iter != returnMap.end())
      return iter->second;
    return {};
  }

private:
  llvm::DenseMap<mlir::Operation *, llvm::SmallVector<mlir::Operation *>>
      returnMap;
};
} // namespace

/// Return `true` if this allocation is to remain on the stack (`fir.alloca`).
/// Otherwise the allocation should be moved to the heap (`fir.allocmem`).
[[maybe_unused]] static inline bool
keepStackAllocation(fir::AllocaOp alloca, mlir::Block *entry,
                    const fir::MemoryAllocationOptOptions &options) {
  // Limitation: only arrays allocated on the stack in the entry block are
  // considered for now.
  // TODO: Generalize the algorithm and placement of the freemem nodes.
  if (alloca->getBlock() != entry)
    return true;
  if (auto seqTy = mlir::dyn_cast<fir::SequenceType>(alloca.getInType())) {
    if (fir::hasDynamicSize(seqTy)) {
      // Move all arrays with runtime determined size to the heap.
      if (options.dynamicArrayOnHeap)
        return false;
    } else {
      std::int64_t numberOfElements = 1;
      for (std::int64_t i : seqTy.getShape()) {
        numberOfElements *= i;
        // If the count is suspicious, then don't change anything here.
        if (numberOfElements <= 0)
          return true;
      }
      // If the number of elements exceeds the threshold, move the allocation to
      // the heap.
      if (static_cast<std::size_t>(numberOfElements) >
          options.maxStackArraySize) {
        LLVM_DEBUG(llvm::dbgs()
                   << "memory allocation opt: found " << alloca << '\n');
        return false;
      }
    }
  }
  return true;
}

static bool isNonTrivial(fir::AllocaOp alloca) {
  return (!fir::conformsWithPassByRef(alloca.getInType()) &&
          !fir::isa_trivial(alloca.getInType())) ||
         !alloca.getShape().empty() || !alloca.getTypeparams().empty();
}

static fir::AllocMemOp
replaceAllocaByAllocmem(fir::AllocaOp alloca, mlir::PatternRewriter &rewriter) {
  mlir::Type varTy = alloca.getInType();
  auto unpackName = [](std::optional<llvm::StringRef> opt) -> llvm::StringRef {
    if (opt)
      return *opt;
    return {};
  };
  llvm::StringRef uniqName = unpackName(alloca.getUniqName());
  llvm::StringRef bindcName = unpackName(alloca.getBindcName());
  auto heap = rewriter.create<fir::AllocMemOp>(
      alloca.getLoc(), varTy, uniqName, bindcName, alloca.getTypeparams(),
      alloca.getShape());
  rewriter.replaceOpWithNewOp<fir::ConvertOp>(
      alloca, fir::ReferenceType::get(varTy), heap);
  return heap;
}

static void fallback(fir::AllocaOp alloca, mlir::PatternRewriter &rewriter,
                     mlir::Block *entry,
                     llvm::ArrayRef<mlir::Operation *> returnOps) {
  mlir::Location loc = alloca.getLoc();
  rewriter.setInsertionPointToStart(entry);
  mlir::Type heapType = fir::HeapType::get(alloca.getInType());
  //  mlir::Type ptrVarType =
  //  fir::LLVMPointerType::get(fir::HeapType::get(alloca.getInType()));
  mlir::Value ptrVar = rewriter.create<fir::AllocaOp>(loc, heapType);
  mlir::Value nullPtr = rewriter.create<fir::ZeroOp>(loc, heapType);
  rewriter.create<fir::StoreOp>(loc, nullPtr, ptrVar);
  mlir::Type intPtrTy = rewriter.getI64Type();
  mlir::Value c0 = rewriter.create<mlir::arith::ConstantOp>(
      loc, intPtrTy, rewriter.getIntegerAttr(intPtrTy, 0));

  auto genConditionalDealloc = [&]() {
    mlir::Value ptrVal = rewriter.create<fir::LoadOp>(loc, ptrVar);
    mlir::Value ptrToInt =
        rewriter.create<fir::ConvertOp>(loc, intPtrTy, ptrVal);
    mlir::Value isAllocated = rewriter.create<mlir::arith::CmpIOp>(
        loc, mlir::arith::CmpIPredicate::ne, ptrToInt, c0);
    auto ifOp = rewriter.create<fir::IfOp>(loc, std::nullopt, isAllocated,
                                           /*withElseRegion=*/false);
    rewriter.setInsertionPointToStart(&ifOp.getThenRegion().front());
    rewriter.create<fir::FreeMemOp>(loc, ptrVal);
    // Reset ptrVar to avoid ? Not really useful given dealloc positions??
    // rewriter.create<fir::StoreOp>(loc, nullPtr, ptrVar);
    rewriter.setInsertionPointAfter(ifOp);
  };

  rewriter.setInsertionPoint(alloca);
  // In case a back-edge comes back to the alloca, deallocate previously
  // allocated values.
  genConditionalDealloc();
  // Replace alloca by allocmem and store allocated value into "ptrVar"
  // variable.
  mlir::Value allocMem = replaceAllocaByAllocmem(alloca, rewriter);
  rewriter.create<fir::StoreOp>(loc, allocMem, ptrVar);

  // Insert conditional deallocations when leaving the function in case the
  // alloca was reached.
  for (mlir::Operation *retOp : returnOps) {
    rewriter.setInsertionPoint(retOp);
    genConditionalDealloc();
  }
}

namespace {
class AllocaOpConversion : public mlir::OpRewritePattern<fir::AllocaOp> {
public:
  using OpRewritePattern::OpRewritePattern;

  AllocaOpConversion(mlir::MLIRContext *ctx,
                     llvm::ArrayRef<mlir::Operation *> rets, mlir::Block *entry)
      : OpRewritePattern(ctx), returnOps(rets), entry{entry} {}

  llvm::LogicalResult
  matchAndRewrite(fir::AllocaOp alloca,
                  mlir::PatternRewriter &rewriter) const override {
    // mlir::Location loc = alloca.getLoc();
    // fir::AllocMemOp heap = replaceAllocaByAllocmem(alloca, rewriter);
    // for (mlir::Operation *retOp : returnOps) {
    //   rewriter.setInsertionPoint(retOp);
    //   [[maybe_unused]] auto free = rewriter.create<fir::FreeMemOp>(loc,
    //   heap); LLVM_DEBUG(llvm::dbgs() << "memory allocation opt: add free " <<
    //   free
    //                           << " for " << heap << '\n');
    // }
    // LLVM_DEBUG(llvm::dbgs() << "memory allocation opt: replaced " << alloca
    //                         << " with " << heap << '\n');
    fallback(alloca, rewriter, entry, returnOps);
    return mlir::success();
  }

private:
  llvm::ArrayRef<mlir::Operation *> returnOps;
  mlir::Block *entry;
};

/// This pass can reclassify memory allocations (fir.alloca, fir.allocmem) based
/// on heuristics and settings. The intention is to allow better performance and
/// workarounds for conditions such as environments with limited stack space.
///
/// Currently, implements two conversions from stack to heap allocation.
///   1. If a stack allocation is an array larger than some threshold value
///      make it a heap allocation.
///   2. If a stack allocation is an array with a runtime evaluated size make
///      it a heap allocation.
class MemoryAllocationOpt
    : public fir::impl::MemoryAllocationOptBase<MemoryAllocationOpt> {
public:
  MemoryAllocationOpt() {
    // Set options with default values. (See Passes.td.) Note that the
    // command-line options, e.g. dynamicArrayOnHeap,  are not set yet.
    options = {dynamicArrayOnHeap, maxStackArraySize};
  }

  MemoryAllocationOpt(bool dynOnHeap, std::size_t maxStackSize) {
    // Set options with default values. (See Passes.td.)
    options = {dynOnHeap, maxStackSize};
  }

  MemoryAllocationOpt(const fir::MemoryAllocationOptOptions &options)
      : options{options} {}

  /// Override `options` if command-line options have been set.
  inline void useCommandLineOptions() {
    if (dynamicArrayOnHeap)
      options.dynamicArrayOnHeap = dynamicArrayOnHeap;
    if (maxStackArraySize != unlimitedArraySize)
      options.maxStackArraySize = maxStackArraySize;
  }

  void runOnOperation() override {
    auto *context = &getContext();
    auto func = getOperation();
    mlir::RewritePatternSet patterns(context);
    mlir::ConversionTarget target(*context);

    useCommandLineOptions();
    LLVM_DEBUG(llvm::dbgs()
               << "dynamic arrays on heap: " << options.dynamicArrayOnHeap
               << "\nmaximum number of elements of array on stack: "
               << options.maxStackArraySize << '\n');

    // If func is a declaration, skip it.
    if (func.empty())
      return;

    const auto &analysis = getAnalysis<ReturnAnalysis>();

    target.addLegalDialect<fir::FIROpsDialect, mlir::arith::ArithDialect,
                           mlir::func::FuncDialect>();
    target.addDynamicallyLegalOp<fir::AllocaOp>([&](fir::AllocaOp alloca) {
      // return keepStackAllocation(alloca, &func.front(), options);
      return !isNonTrivial(alloca);
    });

    llvm::SmallVector<mlir::Operation *> returnOps = analysis.getReturns(func);
    patterns.insert<AllocaOpConversion>(context, returnOps, &func.front());
    if (mlir::failed(
            mlir::applyPartialConversion(func, target, std::move(patterns)))) {
      mlir::emitError(func.getLoc(),
                      "error in memory allocation optimization\n");
      signalPassFailure();
    }
  }

private:
  fir::MemoryAllocationOptOptions options;
};
} // namespace
