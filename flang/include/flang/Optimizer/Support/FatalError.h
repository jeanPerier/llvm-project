//===-- Optimizer/Support/FatalError.h --------------------------*- C++ -*-===//
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

#ifndef FORTRAN_OPTIMIZER_SUPPORT_FATALERROR_H
#define FORTRAN_OPTIMIZER_SUPPORT_FATALERROR_H

#include "mlir/IR/Diagnostics.h"
#include "llvm/Support/ErrorHandling.h"

namespace fir {

/// Fatal error reporting helper. Report a fatal error with a source location
/// and immediately abort flang.
[[noreturn]] inline void emitFatalError(mlir::Location loc,
                                        const llvm::Twine &message) {
  // Override any handlers that may be registered and buffering messages with
  // a handler that will print the error before aborting.
  loc.getContext()->getDiagEngine().registerHandler([](mlir::Diagnostic& diag) {
    auto &os = llvm::errs();
    if (!diag.getLocation().isa<mlir::UnknownLoc>())
      os << diag.getLocation() << ": ";
    os << "error: ";
    os << diag << '\n';
    os.flush();
    return mlir::success();
  });
  mlir::emitError(loc, message);
  llvm::report_fatal_error("aborting");
}

} // namespace fir

#endif // FORTRAN_OPTIMIZER_SUPPORT_FATALERROR_H
