//===-- Lower/Support/BoxValue.h -- internal box values ---------*- C++ -*-===//
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

#ifndef LOWER_SUPPORT_BOXVALUE_H
#define LOWER_SUPPORT_BOXVALUE_H

#include "flang/Optimizer/Dialect/FIRType.h"
#include "flang/Optimizer/Support/Matcher.h"
#include "mlir/IR/OperationSupport.h"
#include "mlir/IR/Value.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/Compiler.h"
#include "llvm/Support/raw_ostream.h"
#include <utility>

namespace fir {
class CharBoxValue;
class ArrayBoxValue;
class CharArrayBoxValue;
class BoxValue;
class ProcBoxValue;
class MutableBoxValue;

llvm::raw_ostream &operator<<(llvm::raw_ostream &, const CharBoxValue &);
llvm::raw_ostream &operator<<(llvm::raw_ostream &, const ArrayBoxValue &);
llvm::raw_ostream &operator<<(llvm::raw_ostream &, const CharArrayBoxValue &);
llvm::raw_ostream &operator<<(llvm::raw_ostream &, const BoxValue &);
llvm::raw_ostream &operator<<(llvm::raw_ostream &, const ProcBoxValue &);
llvm::raw_ostream &operator<<(llvm::raw_ostream &, const MutableBoxValue &);

//===----------------------------------------------------------------------===//
//
// Boxed values
//
// Define a set of containers used internally by the lowering bridge to keep
// track of extended values associated with a Fortran subexpression. These
// associations are maintained during the construction of FIR.
//
//===----------------------------------------------------------------------===//

/// Most expressions of intrinsic type can be passed unboxed. Their properties
/// are known statically.
using UnboxedValue = mlir::Value;

/// Abstract base class.
class AbstractBox {
public:
  AbstractBox() = delete;
  AbstractBox(mlir::Value addr) : addr{addr} {}

  /// FIXME: this comment is not true anymore since genLoad
  /// is loading constant length characters. What is the impact  /// ?
  /// An abstract box always contains a memory reference to a value.
  mlir::Value getAddr() const { return addr; }

protected:
  mlir::Value addr;
};

/// Expressions of CHARACTER type have an associated, possibly dynamic LEN
/// value.
class CharBoxValue : public AbstractBox {
public:
  CharBoxValue(mlir::Value addr, mlir::Value len)
      : AbstractBox{addr}, len{len} {
    if (addr && addr.getType().template isa<fir::BoxCharType>())
      llvm::report_fatal_error("BoxChar should not be in CharBoxValue");
  }

  CharBoxValue clone(mlir::Value newBase) const { return {newBase, len}; }

  /// Convenience alias to get the memory reference to the buffer.
  mlir::Value getBuffer() const { return getAddr(); }

  mlir::Value getLen() const { return len; }
  friend llvm::raw_ostream &operator<<(llvm::raw_ostream &,
                                       const CharBoxValue &);
  LLVM_DUMP_METHOD void dump() const { llvm::errs() << *this; }

protected:
  mlir::Value len;
};

/// Abstract base class.
/// Expressions of type array have at minimum a shape. These expressions may
/// have lbound attributes (dynamic values) that affect the interpretation of
/// indexing expressions.
class AbstractArrayBox {
public:
  AbstractArrayBox() = default;
  AbstractArrayBox(llvm::ArrayRef<mlir::Value> extents,
                   llvm::ArrayRef<mlir::Value> lbounds, mlir::Value sourceBox)
      : extents{extents.begin(), extents.end()},
        lbounds{lbounds.begin(), lbounds.end()}, sourceBox{sourceBox} {}

  // Every array has extents that describe its shape.
  const llvm::SmallVectorImpl<mlir::Value> &getExtents() const {
    return extents;
  }

  // An array expression may have user-defined lower bound values.
  // If this vector is empty, the default in all dimensions in `1`.
  const llvm::SmallVectorImpl<mlir::Value> &getLBounds() const {
    return lbounds;
  }

  /// If the entity is described by a box (e.g. it is a dummy, allocatable, or
  /// pointer) The information in the source box that is also present in other
  /// fields (e.g the extents) is the same, except the lower bounds of the
  /// sourceBox that should be ignored.
  mlir::Value getSourceBox() const { return sourceBox; }

  bool lboundsAllOne() const { return lbounds.empty(); }
  std::size_t rank() const { return extents.size(); }
  /// Is this entity guaranteed to be contiguous at compile time ?
  /// If not, the memory layout is described by sourceBox.
  bool isContiguous() const;

protected:
  llvm::SmallVector<mlir::Value, 4> extents;
  llvm::SmallVector<mlir::Value, 4> lbounds;
  mlir::Value sourceBox;
};

/// Expressions with rank > 0 have extents. They may also have lbounds that are
/// not 1.
class ArrayBoxValue : public AbstractBox, public AbstractArrayBox {
public:
  ArrayBoxValue(mlir::Value addr, llvm::ArrayRef<mlir::Value> extents,
                llvm::ArrayRef<mlir::Value> lbounds = {},
                mlir::Value sourceBox = {})
      : AbstractBox{addr}, AbstractArrayBox{extents, lbounds, sourceBox} {}

  ArrayBoxValue clone(mlir::Value newBase) const {
    // Do not clone sourceBox, its stride and base address information would
    // not describe the new entity.
    return {newBase, extents, lbounds};
  }

  friend llvm::raw_ostream &operator<<(llvm::raw_ostream &,
                                       const ArrayBoxValue &);
  LLVM_DUMP_METHOD void dump() const { llvm::errs() << *this; }
};

/// Expressions of type CHARACTER and with rank > 0.
class CharArrayBoxValue : public CharBoxValue, public AbstractArrayBox {
public:
  CharArrayBoxValue(mlir::Value addr, mlir::Value len,
                    llvm::ArrayRef<mlir::Value> extents,
                    llvm::ArrayRef<mlir::Value> lbounds = {},
                    mlir::Value sourceBox = {})
      : CharBoxValue{addr, len}, AbstractArrayBox{extents, lbounds, sourceBox} {
  }

  CharArrayBoxValue clone(mlir::Value newBase) const {
    return {newBase, len, extents, lbounds};
  }

  friend llvm::raw_ostream &operator<<(llvm::raw_ostream &,
                                       const CharArrayBoxValue &);
  LLVM_DUMP_METHOD void dump() const { llvm::errs() << *this; }
};

/// Expressions that are procedure POINTERs may need a set of references to
/// variables in the host scope.
class ProcBoxValue : public AbstractBox {
public:
  ProcBoxValue(mlir::Value addr, mlir::Value context)
      : AbstractBox{addr}, hostContext{context} {}

  ProcBoxValue clone(mlir::Value newBase) const {
    return {newBase, hostContext};
  }

  mlir::Value getHostContext() const { return hostContext; }

  friend llvm::raw_ostream &operator<<(llvm::raw_ostream &,
                                       const ProcBoxValue &);
  LLVM_DUMP_METHOD void dump() const { llvm::errs() << *this; }

protected:
  mlir::Value hostContext;
};

/// In the generalized form, a boxed value can have a dynamic size, be an array
/// with dynamic extents and lbounds, and take dynamic type parameters.
class BoxValue : public AbstractBox, public AbstractArrayBox {
public:
  BoxValue(mlir::Value addr) : AbstractBox{addr}, AbstractArrayBox{} {}
  BoxValue(mlir::Value addr, mlir::Value len)
      : AbstractBox{addr}, AbstractArrayBox{}, len{len} {}
  BoxValue(mlir::Value addr, llvm::ArrayRef<mlir::Value> extents,
           llvm::ArrayRef<mlir::Value> lbounds = {}, mlir::Value sourceBox = {})
      : AbstractBox{addr}, AbstractArrayBox{extents, lbounds, sourceBox} {}
  BoxValue(mlir::Value addr, mlir::Value len,
           llvm::ArrayRef<mlir::Value> params,
           llvm::ArrayRef<mlir::Value> extents,
           llvm::ArrayRef<mlir::Value> lbounds = {}, mlir::Value sourceBox = {})
      : AbstractBox{addr}, AbstractArrayBox{extents, lbounds, sourceBox},
        len{len}, params{params.begin(), params.end()} {}

  BoxValue clone(mlir::Value newBase) const {
    return {newBase, len, params, extents, lbounds};
  }

  mlir::Value getLen() const { return len; }
  const llvm::SmallVectorImpl<mlir::Value> &getLenTypeParams() const {
    return params;
  }

  friend llvm::raw_ostream &operator<<(llvm::raw_ostream &, const BoxValue &);
  LLVM_DUMP_METHOD void dump() const { llvm::errs() << *this; }

protected:
  mlir::Value len;
  llvm::SmallVector<mlir::Value, 2> params;
};

/// Used for triple notation (array slices)
using RangeBoxValue = std::tuple<mlir::Value, mlir::Value, mlir::Value>;

/// Set of variables (addresses) holding the allocatable properties. These may
/// be empty in case it is not deemed safe to duplicate the descriptor
/// information locally (For instance, a volatile allocatable will always be
/// lowered to a descriptor to preserve the integrity of the entity and its
/// associated properties. As such, all references to the entity and its
/// property will go through the descriptor explicitly.).
class MutableProperties {
public:
  bool isEmpty() const { return !addr; }
  mlir::Value addr;
  llvm::SmallVector<mlir::Value, 2> extents;
  llvm::SmallVector<mlir::Value, 2> lbounds;
  /// Only keep track of the deferred length parameters through variables, since
  /// they are the only ones that can change as per the deferred type parameters
  /// definition in F2018 standard section 3.147.12.2.
  /// Non-deferred values are returned by
  /// MutableBoxValue.nonDeferredLenParams().
  llvm::SmallVector<mlir::Value, 2> deferredParams;
};

/// MutableBoxValue is used for entities that are represented by the address of
/// a box. This is intended to be used for entities whose base address, shape
/// and type are not constant in the entity lifetime (e.g Allocatables and
/// Pointers).
class MutableBoxValue : public AbstractBox {
public:
  /// Create MutableBoxValue given the address \p addr of the box and the non
  /// deferred length parameters \p lenParameters. The non deferred length
  /// parameters must always be provided, even if they are constant and already
  /// reflected in the address type.
  MutableBoxValue(mlir::Value addr, mlir::ValueRange lenParameters,
                  MutableProperties mutableProperties)
      : AbstractBox(addr), lenParams{lenParameters.begin(),
                                     lenParameters.end()},
        mutableProperties{mutableProperties} {
    // Currently only accepts fir.(ref/ptr/heap)<fir.box<type>> mlir::Value for
    // the address. This may change if we accept
    // fir.(ref/ptr/heap)<fir.heap<type>> for scalar without length parameters.
    assert(verify() &&
           "MutableBoxValue requires mem ref to fir.box<fir.[heap|ptr]<type>>");
  }

  /// Get the fir.box<type> part of the address type.
  fir::BoxType getBoxTy() const {
    auto type = getAddr().getType();
    return fir::dyn_cast_ptrEleTy(type).cast<fir::BoxType>();
  }
  /// Return the part of the address type after memory and box types. That is
  /// the element type, maybe wrapped in a fir.array type.
  mlir::Type getBaseTy() const {
    return fir::dyn_cast_ptrEleTy(getBoxTy().getEleTy());
  }
  /// Get the scalar type related to the described entity
  mlir::Type getEleTy() const {
    auto type = getBaseTy();
    if (auto seqTy = type.dyn_cast<fir::SequenceType>())
      return seqTy.getEleTy();
    return type;
  }

  /// Is this a Fortran pointer ?
  bool isPointer() const {
    return getBoxTy().getEleTy().isa<fir::PointerType>();
  }
  /// Is this an allocatable ?
  bool isAllocatable() const {
    return getBoxTy().getEleTy().isa<fir::HeapType>();
  }
  /// Is the entity an array or an assumed rank ?
  bool hasRank() const { return getBaseTy().isa<fir::SequenceType>(); }
  /// Is this an assumed rank ?
  bool hasAssumedRank() const {
    auto seqTy = getBaseTy().dyn_cast<fir::SequenceType>();
    return seqTy && seqTy.hasUnknownShape();
  }
  /// Returns the rank of the entity. Beware that zero will be returned for
  /// both scalars and assumed rank.
  unsigned rank() const {
    auto seqTy = getBaseTy().dyn_cast<fir::SequenceType>();
    if (seqTy)
      return seqTy.getDimension();
    return 0;
  }

  bool isContiguous() const;
  /// Is this a character entity ?
  bool isCharacter() const { return getEleTy().isa<fir::CharacterType>(); };
  /// Is this a derived type entity ?
  bool isDerived() const { return getEleTy().isa<fir::RecordType>(); };
  /// Does this entity has any non deferred length parameters ?
  bool hasNonDeferredLenParams() const { return !lenParams.empty(); }
  /// Return the non deferred length parameters.
  llvm::ArrayRef<mlir::Value> nonDeferredLenParams() const { return lenParams; }
  friend llvm::raw_ostream &operator<<(llvm::raw_ostream &,
                                       const MutableBoxValue &);
  LLVM_DUMP_METHOD void dump() const { llvm::errs() << *this; }

  /// Set of variable is used instead of a descriptor to hold the entity
  /// properties instead of a fir.ref<fir.box<>>.
  bool isDescribedByVariables() const { return !mutableProperties.isEmpty(); }

  const MutableProperties &getMutableProperties() const {
    return mutableProperties;
  }

protected:
  /// Validate the address type form in the constructor.
  bool verify() const;
  /// Hold the non-deferred length parameter values  (both for characters and
  /// derived). Non-deferred length parameters cannot change dynamically, as
  /// opposed to deferred type parameters (3.147.12.2).
  llvm::SmallVector<mlir::Value, 2> lenParams;
  /// Set of variables holding the extents, lower bounds and
  /// base address when it is deemed safe to work with these variables rather
  /// than directly with a descriptor.
  MutableProperties mutableProperties;
};

class ExtendedValue;

mlir::Value getBase(const ExtendedValue &exv);
mlir::Value getLen(const ExtendedValue &exv);
llvm::raw_ostream &operator<<(llvm::raw_ostream &, const ExtendedValue &);
ExtendedValue substBase(const ExtendedValue &exv, mlir::Value base);
bool isArray(const ExtendedValue &exv);

/// An extended value is a box of values pertaining to a discrete entity. It is
/// used in lowering to track all the runtime values related to an entity. For
/// example, an entity may have an address in memory that contains its value(s)
/// as well as various attribute values that describe the shape and starting
/// indices if it is an array entity.
class ExtendedValue : public details::matcher<ExtendedValue> {
public:
  using VT = std::variant<UnboxedValue, CharBoxValue, ArrayBoxValue,
                          CharArrayBoxValue, BoxValue, ProcBoxValue>;

  ExtendedValue() : box{UnboxedValue{}} {}
  ExtendedValue(const ExtendedValue &) = default;
  ExtendedValue(ExtendedValue &&) = default;
  template <typename A, typename = std::enable_if_t<
                            !std::is_same_v<std::decay_t<A>, ExtendedValue>>>
  constexpr ExtendedValue(A &&a) : box{std::forward<A>(a)} {
    if (auto b = getUnboxed()) {
      if (*b) {
        auto type = b->getType();
        if (type.template isa<fir::BoxCharType>())
          llvm::report_fatal_error("BoxChar should be unboxed");
        if (auto refType = type.template dyn_cast<fir::ReferenceType>())
          type = refType.getEleTy();
        if (auto seqType = type.template dyn_cast<fir::SequenceType>())
          type = seqType.getEleTy();
        if (type.template isa<fir::CharacterType>())
          llvm::report_fatal_error(
              "character buffer should be in CharBoxValue");
      }
    }
  }

  template <typename A>
  constexpr const A *getBoxOf() const {
    return std::get_if<A>(&box);
  }

  constexpr const CharBoxValue *getCharBox() const {
    return getBoxOf<CharBoxValue>();
  }

  constexpr const UnboxedValue *getUnboxed() const {
    return getBoxOf<UnboxedValue>();
  }

  /// If the entity in the box does not have a contiguous memory layout,
  /// returns the fir.box describing the memory layout, otherwise, returns
  /// the base.
  mlir::Value getMemRef() const {
    return match(
        [](const ArrayBoxValue &box) {
          return box.isContiguous() ? box.getAddr() : box.getSourceBox();
        },
        [](const CharArrayBoxValue &box) {
          return box.isContiguous() ? box.getAddr() : box.getSourceBox();
        },
        [](const BoxValue &box) {
          return box.isContiguous() ? box.getAddr() : box.getSourceBox();
        },
        [](const UnboxedValue &unboxed) { return unboxed; },
        [](const auto &box) { return box.getAddr(); });
  }

  /// LLVM style debugging of extended values
  LLVM_DUMP_METHOD void dump() const { llvm::errs() << *this << '\n'; }

  friend llvm::raw_ostream &operator<<(llvm::raw_ostream &,
                                       const ExtendedValue &);

  const VT &matchee() const { return box; }

private:
  VT box;
};
} // namespace fir

#endif // LOWER_SUPPORT_BOXVALUE_H
