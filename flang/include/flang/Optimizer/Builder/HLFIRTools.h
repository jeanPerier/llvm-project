//===-- HLFIRTools.h -- HLFIR tools       -----------------------*- C++ -*-===//
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

#ifndef FORTRAN_OPTIMIZER_BUILDER_HLFIRTOOLS_H
#define FORTRAN_OPTIMIZER_BUILDER_HLFIRTOOLS_H

#include "flang/Optimizer/Builder/BoxValue.h"
#include "flang/Optimizer/Dialect/FortranVariableInterface.h"
#include "flang/Optimizer/HLFIR/HLFIRDialect.h"
#include "llvm/ADT/TypeSwitch.h"

namespace fir {
class FirOpBuilder;
}

namespace hlfir {

/// Is this an SSA value type for the value of a Fortran expression?
inline bool isFortranValueType(mlir::Type type) {
  return type.isa<hlfir::ExprType>() || fir::isa_trivial(type);
}

/// Is this the value of a Fortran expression in an SSA value form?
inline bool isFortranValue(mlir::Value value) {
  return isFortranValueType(value.getType());
}

inline bool isFortranVariableLike(mlir::Value value) {
  return llvm::TypeSwitch<mlir::Type, bool>(value.getType())
      .Case<fir::ReferenceType, fir::PointerType, fir::HeapType>([](auto p) {
        mlir::Type eleType = p.getEleTy();
        return eleType.isa<fir::BaseBoxType>() || !fir::hasDynamicSize(eleType);
      })
      .Case<fir::BaseBoxType, fir::BoxCharType>([](auto) { return true; })
      .Default([](mlir::Type) { return false; });
}

inline bool isFortranEntityLike(mlir::Value value) {
  return isFortranValue(value) || isFortranVariableLike(value);
}

/// Is this a Fortran variable?
/// Note that by "variable", it must be understood that the mlir::Value is
/// a memory value of a storage that can be reason about as a Fortran object
/// (its bounds, shape, and type parameters, if any, are retrievable).
/// This does not imply that the mlir::Value points to a variable from the
/// original source or can be legally defined: temporaries created to store
/// expression values are considered to be variables, and so are PARAMETERs
/// global constant address.
inline bool isFortranVariable(mlir::Value value) {
  return value.getDefiningOp<fir::FortranVariableOpInterface>();
}

/// Is this a Fortran variable or expression value?
inline bool isFortranEntity(mlir::Value value) {
  return isFortranValue(value) || isFortranVariable(value);
}

class FortranEntityLike : public mlir::Value {
public:
  explicit FortranEntityLike(mlir::Value value) : mlir::Value(value) {
    assert(isFortranEntityLike(value) &&
           "must be a value representing a Fortran value or variable like");
  }
  FortranEntityLike(fir::FortranVariableOpInterface variable)
      : mlir::Value(variable.getBase()) {}
  bool isValue() const { return isFortranValue(*this); }
  bool isVariable() const { return !isValue(); }
  bool isMutableBox() const {
    mlir::Type type = fir::dyn_cast_ptrEleTy(getType());
    return type && type.isa<fir::BaseBoxType>();
  }
  bool isArray() const {
    mlir::Type type = fir::unwrapPassByRefType(fir::unwrapRefType(getType()));
    if (type.isa<fir::SequenceType>())
      return true;
    if (auto exprType = type.dyn_cast<hlfir::ExprType>())
      return exprType.isArray();
    return false;
  }
  bool isScalar() const { return !isArray(); }

  mlir::Type getFortranElementType() const {
    mlir::Type type = fir::unwrapSequenceType(
        fir::unwrapPassByRefType(fir::unwrapRefType(getType())));
    if (auto exprType = type.dyn_cast<hlfir::ExprType>())
      return exprType.getEleTy();
    return type;
  }

  bool hasLengthParameters() const {
    mlir::Type eleTy = getFortranElementType();
    return eleTy.isa<fir::CharacterType>() ||
           fir::isRecordWithTypeParameters(eleTy);
  }

  fir::FortranVariableOpInterface getIfVariableInterface() const {
    return this->getDefiningOp<fir::FortranVariableOpInterface>();
  }
  mlir::Value getBase() const { return *this; }
};

/// Wrapper over an mlir::Value that can be viewed as a Fortran entity.
/// This provides some Fortran specific helpers as well as a guarantee
/// in the compiler source that a certain mlir::Value must be a Fortran
/// entity.
class FortranEntity : public FortranEntityLike {
public:
  explicit FortranEntity(mlir::Value value) : FortranEntityLike(value) {
    assert(isFortranEntity(value) &&
           "must be a value representing a Fortran value or variable");
  }
  FortranEntity(fir::FortranVariableOpInterface variable)
      : FortranEntityLike(variable) {}
  fir::FortranVariableOpInterface getIfVariable() const {
    return getIfVariableInterface();
  }
};

/// Functions to translate hlfir::FortranEntity to fir::ExtendedValue.
/// For Fortran arrays, character, and derived type values, this require
/// allocating a storage since these can only be represented in memory in FIR.
/// In that case, a cleanup function is provided to generate the finalization
/// code after the end of the fir::ExtendedValue use.
using CleanupFunction = std::function<void()>;
std::pair<fir::ExtendedValue, llvm::Optional<CleanupFunction>>
translateToExtendedValue(mlir::Location loc, fir::FirOpBuilder &builder,
                         FortranEntityLike entity);

/// Function to translate FortranVariableOpInterface to fir::ExtendedValue.
/// It does not generate any IR, and is a simple packaging operation.
fir::ExtendedValue
translateToExtendedValue(fir::FortranVariableOpInterface fortranVariable);

/// Translate a FortranVariableOpInterface to a fir::BoxValue.
/// If a fir.box is already available in the FortranVariableOpInterface, this is
/// a simple packaging operation. Otherwise this will create fir.embox
fir::BoxValue translateToBoxValue(mlir::Location loc,
                                  fir::FirOpBuilder &builder,
                                  fir::FortranVariableOpInterface variable);

/// Generate declaration for a fir::ExtendedValue in memory.
FortranEntity genDeclare(mlir::Location loc, fir::FirOpBuilder &builder,
                         const fir::ExtendedValue &exv, llvm::StringRef name,
                         fir::FortranVariableFlagsAttr flags);

llvm::SmallVector<std::pair<mlir::Value, mlir::Value>>
genBounds(mlir::Location loc, fir::FirOpBuilder &builder,
          FortranEntityLike entity);

std::pair<mlir::Value, mlir::Value>
genRawBaseAndShape(mlir::Location loc, fir::FirOpBuilder &builder,
                   FortranEntityLike entity);

/// If the entity is a variable, load its value (dereference pointers and
/// allocatables if needed). Do nothing if the entity os already a variable or
/// if it is not a scalar entity of numerical or logical type.
mlir::Value loadTrivialScalar(mlir::Location loc, fir::FirOpBuilder &builder,
                              FortranEntityLike entity);

} // namespace hlfir

#endif // FORTRAN_OPTIMIZER_BUILDER_HLFIRTOOLS_H
