//===-- Optimizer/Transforms/MemoryUtils.h ----------------------*- C++ -*-===//
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

#ifndef FORTRAN_OPTIMIZER_TRANSFORMS_MEMORYUTILS_H
#define FORTRAN_OPTIMIZER_TRANSFORMS_MEMORYUTILS_H

#include "flang/Optimizer/Dialect/FIROps.h"

namespace mlir {
class RewriterBase;
}

namespace fir {

using AllocaRewriter =
    llvm::function_ref<mlir::Value(mlir::RewriterBase &, fir::AllocaOp,
                                   bool /*allocaDominatesDeallocLocations*/)>;
using DeallocGenerator =
    llvm::function_ref<void(mlir::Location, mlir::RewriterBase &, mlir::Value)>;

bool replaceAllocas(mlir::RewriterBase &rewriter, mlir::Operation *parentOp,
                    AllocaRewriter, DeallocGenerator);

} // namespace fir

#endif // FORTRAN_OPTIMIZER_TRANSFORMS_MEMORYUTILS_H
