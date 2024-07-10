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
#include "flang/Optimizer/Transforms/MemoryUtils.h"
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

static bool isNonTrivial(fir::AllocaOp alloca) {
  mlir::Type allocType = alloca.getInType();
  return (!fir::conformsWithPassByRef(allocType) &&
          !fir::isa_trivial(allocType) &&
          !fir::isa_builtin_cptr_type(allocType)) ||
         !alloca.getShape().empty() || !alloca.getTypeparams().empty();
}

/// Return `true` if this allocation is to remain on the stack (`fir.alloca`).
/// Otherwise the allocation should be moved to the heap (`fir.allocmem`).
static inline bool
keepStackAllocation(fir::AllocaOp alloca,
                    const fir::MemoryAllocationOptOptions &options) {
  return !isNonTrivial(alloca);
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

static mlir::Value genAllocmem(mlir::RewriterBase &rewriter,
                               fir::AllocaOp alloca,
                               bool deallocPointsDominateAlloc,
                               const fir::MemoryAllocationOptOptions &options) {
  if (keepStackAllocation(alloca, options))
    return nullptr;
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
  return heap;
}

static void genFreemem(mlir::Location loc, mlir::RewriterBase &rewriter,
                       mlir::Value allocmem) {
  rewriter.create<fir::FreeMemOp>(loc, allocmem);
}

/// This pass can reclassify memory allocations (fir.alloca, fir.allocmem) based
/// on heuristics and settings. The intention is to allow better performance and
/// workarounds for conditions such as environments with limited stack space.
///
/// Currently, implements two conversions from stack to heap allocation.
///   1. If a stack allocation is an array larger than some threshold value
///      make it a heap allocation.
///   2. If a stack allocation is an array with a runtime evaluated size make
///      it a heap allocation.
namespace {
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
    auto genAllocmemHelper = [&](mlir::RewriterBase &rewriter,
                                 fir::AllocaOp alloca,
                                 bool deallocPointsDominateAlloc) {
      return genAllocmem(rewriter, alloca, deallocPointsDominateAlloc, options);
    };
    mlir::IRRewriter rewriter(context);
    fir::replaceAllocas(rewriter, func.getOperation(), genAllocmemHelper,
                        genFreemem);
  }

private:
  fir::MemoryAllocationOptOptions options;
};
} // namespace
