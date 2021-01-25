//===-- Allocatable.h -- Allocatable statements lowering ------------------===//
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

namespace mlir {
class Value;
class ValueRange;
class Type;
class Location;
} // namespace mlir

namespace fir {
class MutableBoxValue;
class ExtendedValue;
} // namespace fir

namespace Fortran::parser {
struct AllocateStmt;
struct DeallocateStmt;
} // namespace Fortran::parser

namespace Fortran::lower {
struct SymbolBox;
class AbstractConverter;
class FirOpBuilder;

namespace pft {
struct Variable;
}

/// Create a fir.box of type \p boxType that can be used to initialize an
/// allocatable variable. Initialization of such variable has to be done at the
/// beginning of the variable lifetime by storing the created box in the memory
/// for the variable box.
/// \p nonDeferredParams must provide the non deferred length parameters so that
/// they can already be placed in the unallocated box (inquiries about these
/// parameters are legal even in unallocated state).
mlir::Value createUnallocatedBox(Fortran::lower::FirOpBuilder &builder,
                                 mlir::Location loc, mlir::Type boxType,
                                 mlir::ValueRange nonDeferredParams);

/// Lower an allocate statement to fir.
void genAllocateStmt(Fortran::lower::AbstractConverter &,
                     const Fortran::parser::AllocateStmt &, mlir::Location);

/// Lower a deallocate statement to fir.
void genDeallocateStmt(Fortran::lower::AbstractConverter &,
                       const Fortran::parser::DeallocateStmt &, mlir::Location);

/// Create a MutableBoxValue for an allocatable or pointer entity.
/// If the variables is a local variable that is not a dummy, it will be
/// initialized to unallocated/diassociated status.
fir::MutableBoxValue createMutableBox(Fortran::lower::AbstractConverter &,
                                      mlir::Location,
                                      const Fortran::lower::pft::Variable &var,
                                      mlir::Value boxAddr,
                                      mlir::ValueRange nonDeferredParams);

/// Read all mutable properties into a normal symbol box.
/// It is OK to call this on unassociated/unallocated boxes but any use of the
/// resulting values will be undefined (only the base address will be guaranteed
/// to be null).
Fortran::lower::SymbolBox genMutableBoxRead(Fortran::lower::FirOpBuilder &,
                                            mlir::Location,
                                            const fir::MutableBoxValue &);

/// Update a MutableBoxValue to describe entity \p source (that must be in
/// memory). If \lbounds is not empty, it is used to defined the MutableBoxValue
/// lower bounds, otherwise, the lower bounds from \p source are used.
void genMutableBoxWrite(Fortran::lower::FirOpBuilder &, mlir::Location,
                        const fir::MutableBoxValue &,
                        const fir::ExtendedValue &source,
                        mlir::ValueRange lbounds);

/// Update a MutableBoxValue to describe entity \p source (that must be in
/// memory) with a new array layout given by \p lbounds and \p ubounds.
/// \p source must be known to be contiguous at compile time, or it must have
/// rank 1.
void genMutableBoxWriteWithRemap(Fortran::lower::FirOpBuilder &, mlir::Location,
                                 const fir::MutableBoxValue &,
                                 const fir::ExtendedValue &source,
                                 mlir::ValueRange lbounds,
                                 mlir::ValueRange ubounds);

/// Returns the fir.ref<fir.box<T>> of a MutableBoxValue filled with the current
/// association / allocation properties. If the fir.ref<fir.box> already exists
/// and is-up to date, this is a no-op, otherwise, code will be generated to
/// fill the it.
mlir::Value getMutableIRBox(Fortran::lower::FirOpBuilder &, mlir::Location,
                            const fir::MutableBoxValue &);

/// When the MutableBoxValue was passed as a fir.ref<fir.box> to a call that may
/// have modified it, update the MutableBoxValue according to the
/// fir.ref<fir.box> value.
void syncMutableBoxFromIRBox(Fortran::lower::FirOpBuilder &, mlir::Location,
                             const fir::MutableBoxValue &);

} // namespace Fortran::lower
