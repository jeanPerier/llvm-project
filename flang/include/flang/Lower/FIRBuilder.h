//===-- Lower/FirBuilder.h -- FIR operation builder -------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Builder routines for constructing the FIR dialect of MLIR. As FIR is a
// dialect of MLIR, it makes extensive use of MLIR interfaces and MLIR's coding
// style (https://mlir.llvm.org/getting_started/DeveloperGuide/) is used in this
// module.
//
//===----------------------------------------------------------------------===//

#ifndef FORTRAN_LOWER_FIRBUILDER_H
#define FORTRAN_LOWER_FIRBUILDER_H

#include "flang/Common/reference.h"
#include "flang/Optimizer/Dialect/FIROps.h"
#include "flang/Optimizer/Dialect/FIRType.h"
#include "flang/Optimizer/Support/KindMapping.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/Optional.h"

namespace fir {
class AbstractArrayBox;
class ExtendedValue;
} // namespace fir

namespace Fortran::lower {
class AbstractConverter;

//===----------------------------------------------------------------------===//
// FirOpBuilder
//===----------------------------------------------------------------------===//

/// Extends the MLIR OpBuilder to provide methods for building common FIR
/// patterns.
class FirOpBuilder : public mlir::OpBuilder {
public:
  explicit FirOpBuilder(mlir::Operation *op, const fir::KindMapping &kindMap)
      : OpBuilder{op}, kindMap{kindMap} {}

  /// Get the current Region of the insertion point.
  mlir::Region &getRegion() { return *getBlock()->getParent(); }

  /// Get the current Module
  mlir::ModuleOp getModule() {
    return getRegion().getParentOfType<mlir::ModuleOp>();
  }

  /// Get the current Function
  mlir::FuncOp getFunction() {
    return getRegion().getParentOfType<mlir::FuncOp>();
  }

  /// Get a reference to the kind map.
  const fir::KindMapping &getKindMap() { return kindMap; }

  /// The LHS and RHS are not always in agreement in terms of
  /// type. In some cases, the disagreement is between COMPLEX and other scalar
  /// types. In that case, the conversion must insert/extract out of a COMPLEX
  /// value to have the proper semantics and be strongly typed.
  mlir::Value convertWithSemantics(mlir::Location loc, mlir::Type toTy,
                                   mlir::Value val);

  /// Get the entry block of the current Function
  mlir::Block *getEntryBlock() { return &getFunction().front(); }

  /// Safely create a reference type to the type `eleTy`.
  mlir::Type getRefType(mlir::Type eleTy);

  /// Create a 1-dimensional sequence of `eleTy` of unknown size.
  mlir::Type getVarLenSeqTy(mlir::Type eleTy);

  /// Get character length type
  mlir::Type getCharacterLengthType() { return getIndexType(); }

  /// Get the integer type whose bit width corresponds to the width of pointer
  /// types, or is bigger.
  mlir::Type getIntPtrType() {
    // TODO: Delay the need of such type until codegen or find a way to use
    // llvm::DataLayout::getPointerSizeInBits here.
    return getI64Type();
  }

  /// Create a null constant memory reference of type \p ptrType.
  /// If \p ptrType is not provided, !fir.ref<none> type will be used.
  mlir::Value createNullConstant(mlir::Location loc, mlir::Type ptrType = {});

  /// Create an integer constant of type \p type and value \p i.
  mlir::Value createIntegerConstant(mlir::Location loc, mlir::Type integerType,
                                    std::int64_t i);

  /// Create a real constant from an integer value.
  mlir::Value createRealConstant(mlir::Location loc, mlir::Type realType,
                                 llvm::APFloat::integerPart val);

  /// Create a real constant from an APFloat value.
  mlir::Value createRealConstant(mlir::Location loc, mlir::Type realType,
                                 const llvm::APFloat &val);

  /// Create a real constant of type \p realType with a value zero.
  mlir::Value createRealZeroConstant(mlir::Location loc, mlir::Type realType) {
    return createRealConstant(loc, realType, 0u);
  }

  /// Create a slot for a local on the stack. Besides the variable's type and
  /// shape, it may be given name or target attributes.
  mlir::Value allocateLocal(mlir::Location loc, mlir::Type ty,
                            llvm::StringRef nm,
                            llvm::ArrayRef<mlir::Value> shape,
                            llvm::ArrayRef<mlir::Value> lenParams,
                            bool asTarget = false);

  /// Create a temporary. A temp is allocated using `fir.alloca` and can be read
  /// and written using `fir.load` and `fir.store`, resp.  The temporary can be
  /// given a name via a front-end `Symbol` or a `StringRef`.
  mlir::Value createTemporary(mlir::Location loc, mlir::Type type,
                              llvm::StringRef name = {},
                              llvm::ArrayRef<mlir::Value> shape = {});

  /// Create an unnamed and untracked temporary on the stack.
  mlir::Value createTemporary(mlir::Location loc, mlir::Type type,
                              llvm::ArrayRef<mlir::Value> shape) {
    return createTemporary(loc, type, llvm::StringRef{}, shape);
  }

  /// Create a global value.
  fir::GlobalOp createGlobal(mlir::Location loc, mlir::Type type,
                             llvm::StringRef name,
                             mlir::StringAttr linkage = {},
                             mlir::Attribute value = {}, bool isConst = false);

  fir::GlobalOp createGlobal(mlir::Location loc, mlir::Type type,
                             llvm::StringRef name, bool isConst,
                             std::function<void(FirOpBuilder &)> bodyBuilder,
                             mlir::StringAttr linkage = {});

  /// Create a global constant (read-only) value.
  fir::GlobalOp createGlobalConstant(mlir::Location loc, mlir::Type type,
                                     llvm::StringRef name,
                                     mlir::StringAttr linkage = {},
                                     mlir::Attribute value = {}) {
    return createGlobal(loc, type, name, linkage, value, /*isConst=*/true);
  }

  fir::GlobalOp
  createGlobalConstant(mlir::Location loc, mlir::Type type,
                       llvm::StringRef name,
                       std::function<void(FirOpBuilder &)> bodyBuilder,
                       mlir::StringAttr linkage = {}) {
    return createGlobal(loc, type, name, /*isConst=*/true, bodyBuilder,
                        linkage);
  }

  /// Convert a StringRef string into a fir::StringLitOp.
  fir::StringLitOp createStringLitOp(mlir::Location loc,
                                     llvm::StringRef string);

  /// Create a !fir.char<1> string literal global and returns a
  /// fir::CharBoxValue with its address en length.
  fir::ExtendedValue createStringLiteral(mlir::Location loc,
                                         llvm::StringRef string);

  /// Unique a compiler generated identifier. A short prefix should be provided
  /// to hint at the origin of the identifier.
  std::string uniqueCGIdent(llvm::StringRef prefix, llvm::StringRef name);

  //===--------------------------------------------------------------------===//
  // Linkage helpers (inline). The default linkage is external.
  //===--------------------------------------------------------------------===//

  mlir::StringAttr createCommonLinkage() { return getStringAttr("common"); }

  mlir::StringAttr createInternalLinkage() { return getStringAttr("internal"); }

  mlir::StringAttr createLinkOnceLinkage() { return getStringAttr("linkonce"); }

  mlir::StringAttr createWeakLinkage() { return getStringAttr("weak"); }

  /// Get a function by name. If the function exists in the current module, it
  /// is returned. Otherwise, a null FuncOp is returned.
  mlir::FuncOp getNamedFunction(llvm::StringRef name) {
    return getNamedFunction(getModule(), name);
  }

  static mlir::FuncOp getNamedFunction(mlir::ModuleOp module,
                                       llvm::StringRef name);

  fir::GlobalOp getNamedGlobal(llvm::StringRef name) {
    return getNamedGlobal(getModule(), name);
  }

  static fir::GlobalOp getNamedGlobal(mlir::ModuleOp module,
                                      llvm::StringRef name);

  /// Lazy creation of fir.convert op.
  mlir::Value createConvert(mlir::Location loc, mlir::Type toTy,
                            mlir::Value val);

  /// Create a new FuncOp. If the function may have already been created, use
  /// `addNamedFunction` instead.
  mlir::FuncOp createFunction(mlir::Location loc, llvm::StringRef name,
                              mlir::FunctionType ty) {
    return createFunction(loc, getModule(), name, ty);
  }

  static mlir::FuncOp createFunction(mlir::Location loc, mlir::ModuleOp module,
                                     llvm::StringRef name,
                                     mlir::FunctionType ty);

  /// Determine if the named function is already in the module. Return the
  /// instance if found, otherwise add a new named function to the module.
  mlir::FuncOp addNamedFunction(mlir::Location loc, llvm::StringRef name,
                                mlir::FunctionType ty) {
    if (auto func = getNamedFunction(name))
      return func;
    return createFunction(loc, name, ty);
  }

  static mlir::FuncOp addNamedFunction(mlir::Location loc,
                                       mlir::ModuleOp module,
                                       llvm::StringRef name,
                                       mlir::FunctionType ty) {
    if (auto func = getNamedFunction(module, name))
      return func;
    return createFunction(loc, module, name, ty);
  }

  /// Cast the input value to IndexType.
  mlir::Value convertToIndexType(mlir::Location loc, mlir::Value val) {
    return createConvert(loc, getIndexType(), val);
  }

  /// Construct one of the two forms of shape op from an array box.
  mlir::Value consShape(mlir::Location loc, const fir::AbstractArrayBox &arr);

  /// Create one of the shape ops given an extended value.
  mlir::Value createShape(mlir::Location loc, const fir::ExtendedValue &exv);

  /// Create a slice op extended value. The value to be sliced, `exv`, must be
  /// an array.
  mlir::Value createSlice(mlir::Location loc, const fir::ExtendedValue &exv,
                          mlir::ValueRange triples, mlir::ValueRange path);

  /// Create a boxed value (Fortran descriptor) to be passed to the runtime.
  /// \p exv is an extended value holding a memory reference to the object that
  /// must be boxed. This function will crash if provided something that is not
  /// a memory reference type.
  /// Array entities are boxed with a shape and character with their length.
  mlir::Value createBox(mlir::Location loc, const fir::ExtendedValue &exv);

  /// Create constant i1 with value 1. if \p b is true or 0. otherwise
  mlir::Value createBool(mlir::Location loc, bool b) {
    return createIntegerConstant(loc, getIntegerType(1), b ? 1 : 0);
  }

  //===--------------------------------------------------------------------===//
  // Location helpers
  //===--------------------------------------------------------------------===//

  /// Generate a string literal containing the file name and return its address
  mlir::Value locationToFilename(mlir::Location);
  /// Generate a constant of the given type with the location line number
  mlir::Value locationToLineNo(mlir::Location, mlir::Type);

private:
  const fir::KindMapping &kindMap;
};

} // namespace Fortran::lower

#endif // FORTRAN_LOWER_FIRBUILDER_H
