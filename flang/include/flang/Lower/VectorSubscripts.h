//===-- VectorSubscripts.h -- vector subscripts tools -----------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// \brief Defines a compiler internal representation for lowered designators
///  containing vector subscripts. This representation allows working on such
///  designators in custom ways while ensuring the designator subscripts are
///  only evaluated once. It is mainly intended for cases that do not fit in
///  the array expression lowering framework like input IO in presence of
///  vector subscripts.
///
//===----------------------------------------------------------------------===//

#ifndef FORTRAN_LOWER_VECTORSUBSCRIPTS_H
#define FORTRAN_LOWER_VECTORSUBSCRIPTS_H

#include "flang/Optimizer/Builder/BoxValue.h"
#include "flang/Common/indirection.h"

namespace fir {
class FirOpBuilder;
}

namespace Fortran {

namespace evaluate {
template <typename>
class Expr;
template <typename>
class Designator;
struct SomeType;
} // namespace evaluate

namespace lower {

class AbstractConverter;
class StatementContext;
class ExprLower;

/// VectorSubscriptBox is a lowered representation for any Designator<T> that
/// contain at least one vector subscript.
///
/// A designator `x%a(i,j)%b(1:foo():1, vector, k)%c%d(m)%e1
/// Is lowered into:
///   - an ExtendedValue for ranked base (x%a(i,j)%b)
///   - mlir:Values and ExtendedValues for the triplet, vector subscript and
///     scalar subscripts of the ranked array reference (1:foo():1, vector, k)
///   - a list of fir.field_index and scalar integers mlir::Value for the
///   component
///     path at the right of the ranked array ref (%c%d(m)%e).
///
/// This representation allows later creating loops over the designator elements
/// and fir.array_coor to get the element addresses without re-evaluating any
/// sub-expressions.
class VectorSubscriptBox {
public:
  /// Type of the callbacks that can be passed to work with the element
  /// addresses.
  using ElementalGenerator = std::function<void(const fir::ExtendedValue &)>;
  using ElementalGeneratorWithBoolReturn =
      std::function<mlir::Value(const fir::ExtendedValue &)>;
  struct LoweredVectorSubscript {
    Fortran::lower::ExprLower& getVector();
    const Fortran::lower::ExprLower& getVector() const;
    /// Copy assignments needed to allow lambda capture.
    LoweredVectorSubscript(const LoweredVectorSubscript&);
    LoweredVectorSubscript(LoweredVectorSubscript&&);
    LoweredVectorSubscript &operator=(const LoweredVectorSubscript &);
    LoweredVectorSubscript &operator=(LoweredVectorSubscript &&);
    LoweredVectorSubscript(Fortran::lower::ExprLower&& vector, mlir::Value size);
    ~LoweredVectorSubscript();
    // Lowered vector expression 
    Fortran::common::Indirection<Fortran::lower::ExprLower, /*copy-able*/true> vector;
    // Vector size, guaranteed to be of indexType.
    mlir::Value size;
  };
  struct LoweredTriplet {
    // Triplets value, guaranteed to be of indexType.
    mlir::Value lb;
    mlir::Value ub;
    mlir::Value stride;
  };
  using LoweredSubscript =
      std::variant<mlir::Value, LoweredTriplet, LoweredVectorSubscript>;
  using MaybeSubstring = llvm::SmallVector<mlir::Value, 2>;
  VectorSubscriptBox(
      fir::ExtendedValue &&loweredBase,
      llvm::SmallVector<LoweredSubscript, 16> &&loweredSubscripts,
      llvm::SmallVector<mlir::Value> &&componentPath,
      MaybeSubstring substringBounds, mlir::Type elementType)
      : loweredBase{std::move(loweredBase)}, loweredSubscripts{std::move(
                                                 loweredSubscripts)},
        componentPath{std::move(componentPath)},
        substringBounds{substringBounds}, elementType{elementType} {};

  /// Loop over the elements described by the VectorSubscriptBox, and call
  /// \p elementalGenerator inside the loops with the element addresses.
  void loopOverElements(fir::FirOpBuilder &builder, mlir::Location loc,
                        const ElementalGenerator &elementalGenerator);

  /// Loop over the elements described by the VectorSubscriptBox while a
  /// condition is true, and call \p elementalGenerator inside the loops with
  /// the element addresses. The initial condition value is \p initialCondition,
  /// and then it is the result of \p elementalGenerator. The value of the
  /// condition after the loops is returned.
  mlir::Value loopOverElementsWhile(
      fir::FirOpBuilder &builder, mlir::Location loc,
      const ElementalGeneratorWithBoolReturn &elementalGenerator,
      mlir::Value initialCondition);

  /// Return the type of the elements of the array section.
  mlir::Type getElementType() const { return elementType; }

  /// Create sliceOp for the designator.
  mlir::Value createSlice(fir::FirOpBuilder &builder, mlir::Location loc) const;

  /// Create shapeOp for the designator.
  mlir::Value createShape(fir::FirOpBuilder &builder, mlir::Location loc) const;

  /// Get array base.
  const fir::ExtendedValue& getBase() const;

  /// Get type parameters if this is a character designator or a derived type with length parameters. Return empty vector otherwise.
  llvm::SmallVector<mlir::Value> getTypeParams(fir::FirOpBuilder &builder, mlir::Location loc) const;

  /// Create ExtendedValue the element inside the loop.
  fir::ExtendedValue getElementAt(fir::FirOpBuilder &builder,
                                  mlir::Location loc, mlir::Value shape,
                                  mlir::Value slice,
                                  mlir::ValueRange indices) const;

  bool hasVectorSubscripts() const;

  fir::ExtendedValue asBox(fir::FirOpBuilder& builder, mlir::Location loc) const;

private:
  /// Common implementation for DoLoop and IterWhile loop creations.
  template <typename LoopType, typename Generator>
  mlir::Value loopOverElementsBase(fir::FirOpBuilder &builder,
                                   mlir::Location loc,
                                   const Generator &elementalGenerator,
                                   mlir::Value initialCondition);

  /// Generate the [lb, ub, step] to loop over the section (in loop order, not
  /// Fortran dimension order).
  llvm::SmallVector<std::tuple<mlir::Value, mlir::Value, mlir::Value>>
  genLoopBounds(fir::FirOpBuilder &builder, mlir::Location loc);

  /// Lowered base of the ranked array ref.
  fir::ExtendedValue loweredBase;

  /// Scalar subscripts and components at the left of the ranked
  /// array ref.
  llvm::SmallVector<mlir::Value> preRankedPath;

  /// Subscripts values of the rank arrayRef part.
  llvm::SmallVector<LoweredSubscript, 4> loweredSubscripts;
  /// Scalar subscripts and components at the right of the ranked
  /// array ref part.
  llvm::SmallVector<mlir::Value, 4> componentPath;
  /// List of substring bounds if this is a substring (only the lower bound if
  /// the upper is implicit).
  MaybeSubstring substringBounds;
  /// Type of the elements described by this array section.
  mlir::Type elementType;
};

/// Lower \p expr, that must be an designator containing vector subscripts, to a
/// VectorSubscriptBox representation. This causes evaluation of all the
/// subscripts. Any required clean-ups from subscript expression are added to \p
/// stmtCtx.
VectorSubscriptBox genVectorSubscriptBox(
    mlir::Location loc, Fortran::lower::AbstractConverter &converter,
    Fortran::lower::StatementContext &stmtCtx,
    const Fortran::evaluate::Expr<Fortran::evaluate::SomeType> &expr);

/// Generalized variable representation.
class Variable {
public:
  using ElementalGenerator = std::function<void(fir::FirOpBuilder&, mlir::Location, const fir::ExtendedValue&, llvm::ArrayRef<mlir::Value>)>;
  using ElementalMask = std::function<void(fir::FirOpBuilder&, mlir::Location, llvm::ArrayRef<mlir::Value>)>;

  using ArraySection = VectorSubscriptBox;

  explicit Variable(const fir::ExtendedValue& exv) : var{exv} {}
  explicit Variable(fir::ExtendedValue&& exv) : var{std::move(exv)} {}
  explicit Variable(ArraySection&& arraySection) : var{std::move(arraySection)} {}

  void loopOverElements(fir::FirOpBuilder& builder, mlir::Location loc, const ElementalGenerator& doOnEachElement, const ElementalMask* filter, bool canLoopUnordered);

  void prepareForAddressing(fir::FirOpBuilder& builder, mlir::Location loc);

  fir::ExtendedValue getElementAt(fir::FirOpBuilder& builder, mlir::Location loc, mlir::ValueRange indices) const;

  llvm::SmallVector<mlir::Value> getExtents(fir::FirOpBuilder& builder, mlir::Location loc) const;

  llvm::SmallVector<mlir::Value> getTypeParams(fir::FirOpBuilder& builder, mlir::Location loc) const;

  llvm::SmallVector<mlir::Value> getLBounds(fir::FirOpBuilder& builder, mlir::Location loc) const;

  void reallocate(fir::FirOpBuilder& builder, mlir::Location loc, llvm::ArrayRef<mlir::Value> lbounds, llvm::ArrayRef<mlir::Value> extents, llvm::ArrayRef<mlir::Value> typeParams) const;

  /// Generate code to assign an expression to this variable.
  /// For arrays, if expr overlaps with the variable, expr should have been
  /// temporized before calling this.
  /// This does not perform any allocatable assignment semantics (if the variable is
  /// a whole allocatable, it should be reallocated if needed before).
  void genAssign(fir::FirOpBuilder& builder, mlir::Location loc, const ExprLower& expr, const ElementalMask* filter);

  /// Generate code to assign one variable to another.
  /// This behaves similarly to assign with an expression.
  void genAssign(fir::FirOpBuilder& builder, mlir::Location loc, const Variable& var, const ElementalMask* filter);
  

  /// Returns an fir::ExtendedValue representing the variable without making a temp.
  /// Cannot be called for variable with vector subscripts.
  /// Will generate a fir.embox or fir.rebox for ArraySection.
  fir::ExtendedValue getAsExtendedValue(fir::FirOpBuilder& builder, mlir::Location loc) const;

  bool hasVectorSubscripts() const;

  bool isArray() const;

private:
  // TODO consider using some pointer for ArraySection
  // that is heavy. This is not made easy by the lambda that
  // capture variables by copy in array expression lowering.
  std::variant<fir::ExtendedValue, ArraySection> var;
  mlir::Value shape;
  mlir::Value slice;
  bool readyForAddressing = false;
};

/// Lower an expression that is a variable to a representation that allows
/// modifying the variable.
Variable genVariable(Fortran::lower::AbstractConverter &converter, mlir::Location loc,
  Fortran::lower::StatementContext &stmtCtx,
  const Fortran::evaluate::Expr<Fortran::evaluate::SomeType> &expr);

template<typename T>
struct VariableBuilder {
  Variable gen(Fortran::lower::AbstractConverter &converter, mlir::Location loc,
    Fortran::lower::StatementContext &stmtCtx,
    const Fortran::evaluate::Expr<T> &expr);

  Variable gen(Fortran::lower::AbstractConverter &converter, mlir::Location loc,
    Fortran::lower::StatementContext &stmtCtx,
    const Fortran::evaluate::Designator<T> &expr);
};

} // namespace lower
} // namespace Fortran

#endif // FORTRAN_LOWER_VECTORSUBSCRIPTS_H
