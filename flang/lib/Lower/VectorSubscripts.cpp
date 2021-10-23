//===-- VectorSubscripts.cpp -- Vector subscripts tools -------------------===//
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

#include "flang/Lower/VectorSubscripts.h"
#include "flang/Lower/AbstractConverter.h"
#include "flang/Lower/Support/Utils.h"
#include "flang/Lower/ConvertExpr.h"
#include "flang/Lower/Todo.h"
#include "flang/Optimizer/Builder/Character.h"
#include "flang/Optimizer/Builder/Complex.h"
#include "flang/Optimizer/Builder/FIRBuilder.h"
#include "flang/Optimizer/Builder/MutableBox.h"
#include "flang/Optimizer/Transforms/Factory.h"
#include "flang/Semantics/expression.h"

namespace {

static fir::ExtendedValue addressComponents(fir::FirOpBuilder& builder, mlir::Location loc, const fir::ExtendedValue& exv, llvm::ArrayRef<fir::FieldIndexOp> fields) {
  TODO(loc, "address components");
}

static constexpr bool mustGenArrayCoor = false;


static fir::ExtendedValue genArrayCoor(fir::FirOpBuilder& builder, mlir::Location loc, const fir::ExtendedValue& array, llvm::ArrayRef<mlir::Value> coordinates) {
  auto addr = fir::getBase(array);
  auto eleTy = fir::unwrapSequenceType(fir::unwrapPassByRefType(addr.getType()));
  auto eleRefTy = builder.getRefType(eleTy);
  auto shape = builder.createShape(loc, array);
  auto elementAddr = builder.create<fir::ArrayCoorOp>(
      loc, eleRefTy, addr, shape, /*slice=*/mlir::Value{}, coordinates,
      fir::getTypeParams(array));
  return fir::factory::arrayElementToExtendedValue(builder, loc, array,
                                                   elementAddr);
}

static llvm::SmallVector<mlir::Value> toZeroBasedIndices(fir::FirOpBuilder& builder, mlir::Location loc, const fir::ExtendedValue& array, llvm::ArrayRef<mlir::Value> coordinates) {
  llvm::SmallVector<mlir::Value> zeroBased;
  auto one = builder.createIntegerConstant(loc, builder.getIndexType(), 1);
  for (auto coor : llvm::enumerate(coordinates)) {
    auto lb = fir::factory::readLowerBound(builder, loc, array, coor.index(), one);
    auto ty = coor.value().getType();
    lb = builder.createConvert(loc, ty, lb);
    zeroBased.push_back(builder.create<mlir::SubIOp>(loc, ty, coor.value(), lb));
  }
  return zeroBased;
}

/// Lower an ArrayRef to a fir.coordinate_of using an element offset instead
/// of array indexes.
/// This generates offset computation from the indexes and length parameters,
/// and use the offset to access the element with a fir.coordinate_of. This
/// must only be used if it is not possible to generate a normal
/// fir.coordinate_of using array indexes (i.e. when the shape information is
/// unavailable in the IR).
fir::ExtendedValue genOffsetAndCoordinateOp(fir::FirOpBuilder& builder, mlir::Location loc, const fir::ExtendedValue& array, llvm::ArrayRef<mlir::Value> coordinates) {
  auto addr = fir::getBase(array);
  auto arrTy = fir::dyn_cast_ptrEleTy(addr.getType());
  auto eleTy = arrTy.cast<fir::SequenceType>().getEleTy();
  auto seqTy = builder.getRefType(builder.getVarLenSeqTy(eleTy));
  auto refTy = builder.getRefType(eleTy);
  auto base = builder.createConvert(loc, seqTy, addr);
  auto idxTy = builder.getIndexType();
  auto one = builder.createIntegerConstant(loc, idxTy, 1);
  auto zero = builder.createIntegerConstant(loc, idxTy, 0);
  auto getLB = [&](const auto &arr, unsigned dim) -> mlir::Value {
    return arr.getLBounds().empty() ? one : arr.getLBounds()[dim];
  };
  auto genFullDim = [&](const auto &arr, mlir::Value delta) -> mlir::Value {
    mlir::Value total = zero;
    assert(arr.getExtents().size() == coordinates.size());
    delta = builder.createConvert(loc, idxTy, delta);
    unsigned dim = 0;
    for (auto [ext, sub] : llvm::zip(arr.getExtents(), coordinates)) {
      auto val = builder.createConvert(loc, idxTy, sub);
      auto lb = builder.createConvert(loc, idxTy, getLB(arr, dim));
      auto diff = builder.create<mlir::SubIOp>(loc, val, lb);
      auto prod = builder.create<mlir::MulIOp>(loc, delta, diff);
      total = builder.create<mlir::AddIOp>(loc, prod, total);
      if (ext)
        delta = builder.create<mlir::MulIOp>(loc, delta, ext);
      ++dim;
    }
    auto origRefTy = refTy;
    if (fir::factory::CharacterExprHelper::isCharacterScalar(refTy)) {
      auto chTy = fir::factory::CharacterExprHelper::getCharacterType(refTy);
      if (fir::characterWithDynamicLen(chTy)) {
        auto ctx = builder.getContext();
        auto kind = fir::factory::CharacterExprHelper::getCharacterKind(chTy);
        auto singleTy = fir::CharacterType::getSingleton(ctx, kind);
        refTy = builder.getRefType(singleTy);
        auto seqRefTy = builder.getRefType(builder.getVarLenSeqTy(singleTy));
        base = builder.createConvert(loc, seqRefTy, base);
      }
    }
    auto coor = builder.create<fir::CoordinateOp>(
        loc, refTy, base, llvm::ArrayRef<mlir::Value>{total});
    // Convert to expected, original type after address arithmetic.
    return builder.createConvert(loc, origRefTy, coor);
  };
  return array.match(
      [&](const fir::ArrayBoxValue &arr) -> fir::ExtendedValue {
        return genFullDim(arr, one);
      },
      [&](const fir::CharArrayBoxValue &arr) -> fir::ExtendedValue {
        auto delta = arr.getLen();
        // If the length is known in the type, fir.coordinate_of will
        // already take the length into account.
        if (fir::factory::CharacterExprHelper::hasConstantLengthInType(arr))
          delta = one;
        return fir::CharBoxValue(genFullDim(arr, delta), arr.getLen());
      },
      [&](const fir::BoxValue &arr) -> fir::ExtendedValue {
        // CoordinateOp for BoxValue is not generated here. The dimensions
        // must be kept in the fir.coordinate_op so that potential fir.box
        // strides can be applied by codegen.
        fir::emitFatalError(
            loc, "internal: BoxValue in dim-collapsed fir.coordinate_of");
      },
      [&](const auto &) -> fir::ExtendedValue {
        fir::emitFatalError(loc, "internal: array lowering failed");
      });
}

/// Address an array with user coordinates (not zero based).
static fir::ExtendedValue addressArray(fir::FirOpBuilder& builder, mlir::Location loc, const fir::ExtendedValue& array, llvm::ArrayRef<mlir::Value> coordinates) {
  if (mustGenArrayCoor)
    return genArrayCoor(builder, loc, array, coordinates);
  auto base = fir::getBase(array);
  auto baseType = fir::unwrapPassByRefType(base.getType());
  if ((array.rank() > 1 && fir::hasDynamicSize(baseType)) ||
      fir::characterWithDynamicLen(fir::unwrapSequenceType(baseType)))
    if (!array.getBoxOf<fir::BoxValue>())
      return genOffsetAndCoordinateOp(builder, loc, array, coordinates);
  auto eleRefTy = builder.getRefType(fir::unwrapSequenceType(baseType));
  // fir::CoordinateOp is zero based.
  auto zeroBasedIndices = toZeroBasedIndices(builder, loc, array, coordinates);
  auto addr = builder.create<fir::CoordinateOp>(loc, eleRefTy, base, zeroBasedIndices);
  return fir::factory::arrayElementToExtendedValue(builder, loc, array, addr);
}

/// Helper class to lower a designator containing vector subscripts into a
/// lowered representation that can be worked with.
class VectorSubscriptBoxBuilder {
public:
  VectorSubscriptBoxBuilder(mlir::Location loc,
                            Fortran::lower::AbstractConverter &converter,
                            Fortran::lower::StatementContext &stmtCtx)
      : converter{converter}, stmtCtx{stmtCtx}, loc{loc} {}

  Fortran::lower::VectorSubscriptBox
  gen(const Fortran::evaluate::Expr<Fortran::evaluate::SomeType> &expr) {
    elementType = genDesignator(expr);
    applyLeftAddressingPart();
    return Fortran::lower::VectorSubscriptBox(
        std::move(loweredBase), std::move(loweredSubscripts),
        std::move(componentPath), substringBounds, elementType);
  }

private:
  using LoweredVectorSubscript =
      Fortran::lower::VectorSubscriptBox::LoweredVectorSubscript;
  using LoweredTriplet = Fortran::lower::VectorSubscriptBox::LoweredTriplet;
  using LoweredSubscript = Fortran::lower::VectorSubscriptBox::LoweredSubscript;
  using MaybeSubstring = Fortran::lower::VectorSubscriptBox::MaybeSubstring;

  /// genDesignator unwraps a Designator<T> and calls `gen` on what the
  /// designator actually contains.
  template <typename A>
  mlir::Type genDesignator(const A &) {
    fir::emitFatalError(loc, "expr must contain a designator");
  }
  template <typename T>
  mlir::Type genDesignator(const Fortran::evaluate::Expr<T> &expr) {
    using ExprVariant = decltype(Fortran::evaluate::Expr<T>::u);
    using Designator = Fortran::evaluate::Designator<T>;
    if constexpr (Fortran::common::HasMember<Designator, ExprVariant>) {
      const auto &designator = std::get<Designator>(expr.u);
      return std::visit([&](const auto &x) { return gen(x); }, designator.u);
    } else {
      return std::visit([&](const auto &x) { return genDesignator(x); },
                        expr.u);
    }
  }

  // The gen(X) methods visit X to lower its base and subscripts and return the
  // type of X elements.

  mlir::Type gen(const Fortran::evaluate::DataRef &dataRef) {
    return std::visit([&](const auto &ref) -> mlir::Type { return gen(ref); },
                      dataRef.u);
  }

  mlir::Type gen(const Fortran::evaluate::SymbolRef &symRef) {
    loweredBase = converter.getSymbolExtendedValue(symRef);
    auto ty = fir::getBase(loweredBase).getType();
    if (symRef->Rank() > 0)
      lastPartWasRanked = true;
    return fir::unwrapPassByRefType(fir::unwrapSequenceType(ty));
  }

  mlir::Type gen(const Fortran::evaluate::Substring &substring) {
    // StaticDataObject::Pointer bases are constants and cannot be
    // subscripted, so the base must be a DataRef here.
    auto baseElementType =
        gen(std::get<Fortran::evaluate::DataRef>(substring.parent()));
    startRightPartIfLastPartWasRanked();
    auto &builder = converter.getFirOpBuilder();
    auto idxTy = builder.getIndexType();
    auto lb = genScalarValue(substring.lower());
    substringBounds.emplace_back(builder.createConvert(loc, idxTy, lb));
    if (const auto &ubExpr = substring.upper()) {
      auto ub = genScalarValue(*ubExpr);
      substringBounds.emplace_back(builder.createConvert(loc, idxTy, ub));
    }
    return baseElementType;
  }

  mlir::Type gen(const Fortran::evaluate::ComplexPart &complexPart) {
    auto complexType = gen(complexPart.complex());
    startRightPartIfLastPartWasRanked();
    auto &builder = converter.getFirOpBuilder();
    auto i32Ty = builder.getI32Type(); // llvm's GEP requires i32
    auto offset = builder.createIntegerConstant(
        loc, i32Ty,
        complexPart.part() == Fortran::evaluate::ComplexPart::Part::RE ? 0 : 1);
    if (isAfterRankedPart)
      componentPath.emplace_back(offset);
    else
      preRankedPath.emplace_back(offset);
    return fir::factory::ComplexExprHelper{builder, loc}.getComplexPartType(
        complexType);
  }

  mlir::Type gen(const Fortran::evaluate::Component &component) {
    auto recTy = gen(component.base()).cast<fir::RecordType>();
    startRightPartIfLastPartWasRanked();
    const auto &componentSymbol = component.GetLastSymbol();
    // Parent components will not be found here, they are not part
    // of the FIR type and cannot be used in the path yet.
    if (componentSymbol.test(Fortran::semantics::Symbol::Flag::ParentComp))
      TODO(loc, "Reference to parent component");
    auto fldTy = fir::FieldType::get(&converter.getMLIRContext());
    auto componentName = toStringRef(componentSymbol.name());
    // Parameters threading in field_index is not yet very clear. We only
    // have the ones of the ranked array ref at hand, but it looks like
    // the fir.field_index expects the one of the direct base.
    if (recTy.getNumLenParams() != 0)
      TODO(loc, "threading length parameters in field index op");
    auto &builder = converter.getFirOpBuilder();
    auto fieldIndex = builder.create<fir::FieldIndexOp>(
        loc, fldTy, componentName, recTy, /*typeParams*/ llvm::None);
    if (isAfterRankedPart)
      componentPath.emplace_back(fieldIndex);
    else
      preRankedPath.emplace_back(fieldIndex);
    if (componentSymbol.Rank() > 0)
      lastPartWasRanked = true;
    return fir::unwrapSequenceType(recTy.getType(componentName));
  }

  mlir::Type gen(const Fortran::evaluate::ArrayRef &arrayRef) {
    auto isTripletOrVector =
        [](const Fortran::evaluate::Subscript &subscript) -> bool {
      return std::visit(
          Fortran::common::visitors{
              [](const Fortran::evaluate::IndirectSubscriptIntegerExpr &expr) {
                return expr.value().Rank() != 0;
              },
              [&](const Fortran::evaluate::Triplet &) { return true; }},
          subscript.u);
    };
    if (llvm::any_of(arrayRef.subscript(), isTripletOrVector))
      return genRankedArrayRefSubscriptAndBase(arrayRef);

    // This is a scalar ArrayRef (only scalar indexes), collect the indexes and
    // visit the base that must contain another arrayRef with the vector
    // subscript.
    auto elementType = gen(namedEntityToDataRef(arrayRef.base()));
    for (const auto &subscript : arrayRef.subscript()) {
      const auto &expr =
          std::get<Fortran::evaluate::IndirectSubscriptIntegerExpr>(
              subscript.u);
      auto subscriptValue = genScalarValue(expr.value());
      if (isAfterRankedPart)
        componentPath.emplace_back(subscriptValue);
      else
        preRankedPath.emplace_back(subscriptValue);
    }
    // The last part rank was "consumed" by the subscripts.
    lastPartWasRanked = false;
    return elementType;
  }

  /// Lower the subscripts and base of the ArrayRef that is an array (there must
  /// be one since there is a vector subscript, and there can only be one
  /// according to C925).
  mlir::Type genRankedArrayRefSubscriptAndBase(
      const Fortran::evaluate::ArrayRef &arrayRef) {
    // Lower and save the base
    auto baseExpr = namedEntityToExpr(arrayRef.base());
    loweredBase = converter.genExprAddr(baseExpr, stmtCtx);
    // Lower and save the subscripts
    auto &builder = converter.getFirOpBuilder();
    auto idxTy = builder.getIndexType();
    auto one = builder.createIntegerConstant(loc, idxTy, 1);
    for (const auto &subscript : llvm::enumerate(arrayRef.subscript())) {
      std::visit(
          Fortran::common::visitors{
              [&](const Fortran::evaluate::IndirectSubscriptIntegerExpr &expr) {
                if (expr.value().Rank() == 0) {
                  // Simple scalar subscript
                  loweredSubscripts.emplace_back(genScalarValue(expr.value()));
                } else {
                  // Vector subscript.
                  // Remove conversion if any to avoid temp creation that may
                  // have been added by the front-end to avoid the creation of a
                  // temp array value.
                  // TODO: ignoreEvConvert is making a copy and breaks forall raising. Consider unwrapping instead, and adding Expr<T> entry points to converter.genExpr.
                  auto vector = converter.genExpr(
                      ignoreEvConvert(expr.value()), stmtCtx, loc);
                  auto extents = vector.getExtents(builder, loc);
                  assert(extents.size() == 1);
                  auto size = builder.createConvert(loc, idxTy, extents[0]);
                  loweredSubscripts.emplace_back(
                      LoweredVectorSubscript{std::move(vector), size});
                }
              },
              [&](const Fortran::evaluate::Triplet &triplet) {
                mlir::Value lb, ub;
                if (const auto &lbExpr = triplet.lower())
                  lb = genScalarValue(*lbExpr);
                else
                  lb = fir::factory::readLowerBound(builder, loc, loweredBase,
                                                    subscript.index(), one);
                lb = builder.createConvert(loc, idxTy, lb);
                if (const auto &ubExpr = triplet.upper()) {
                  ub = genScalarValue(*ubExpr);
                  ub = builder.createConvert(loc, idxTy, ub);
                } else {
                  // ub = lb + extent -1
                  ub = fir::factory::readExtent(builder, loc, loweredBase,
                                                subscript.index());
                  ub = builder.createConvert(loc, idxTy, ub);
                  ub = builder.create<mlir::SubIOp>(loc, ub, one);
                  ub = builder.create<mlir::AddIOp>(loc, lb, ub);
                }
                auto stride = genScalarValue(triplet.stride());
                stride = builder.createConvert(loc, idxTy, stride);
                loweredSubscripts.emplace_back(LoweredTriplet{lb, ub, stride});
              },
          },
          subscript.value().u);
    }
    isAfterRankedPart = true;
    return fir::unwrapSequenceType(
        fir::unwrapPassByRefType(fir::getBase(loweredBase).getType()));
  }

  mlir::Type gen(const Fortran::evaluate::CoarrayRef &) {
    // Is this possible/legal ?
    TODO(loc, "Coarray ref with vector subscript in IO input");
  }

  void applyLeftAddressingPart() {
    auto &builder = converter.getFirOpBuilder();
    if (preRankedPath.empty() && substringBounds.empty())
      return;
    auto prePath = preRankedPath.begin();
    while (prePath != preRankedPath.end()) {
      if (!prePath->getType().isa<fir::FieldType>()) {
        llvm::SmallVector<fir::FieldIndexOp> fields;
        while (prePath != preRankedPath.end()) {
          auto fieldOp = prePath->getDefiningOp<fir::FieldIndexOp>();
          if (!fieldOp)
            break;
          fields.push_back(fieldOp);
          prePath++;
        }
        ++prePath; 
        loweredBase = addressComponents(builder, loc, loweredBase, fields);
      } else {
        auto rank = loweredBase.rank();
        if (rank > 0) {
          llvm::SmallVector<mlir::Value> coors;
          while (prePath != preRankedPath.end() && rank > 0) {
            if (prePath->getType().isa<fir::FieldType>())
              break;
            coors.push_back(*prePath);
            prePath++;
            rank --;
          }
          assert(rank == 0 && "rank mismatch");
          loweredBase = addressArray(builder, loc, loweredBase, coors);
        } else {
          TODO(loc, "scalar complex part");
        }
      }
    }

    // Keep substring info if this is a ranked array section.
    if (!loweredSubscripts.empty())
      return;

    if (!substringBounds.empty())
      TODO(loc, "substring");
  }

  template <typename A>
  mlir::Value genScalarValue(const A &expr) {
    return fir::getBase(converter.genExprValue(toEvExpr(expr), stmtCtx));
  }
  void startRightPartIfLastPartWasRanked() {
    if (lastPartWasRanked)
      isAfterRankedPart = true;
  }

  Fortran::evaluate::DataRef
  namedEntityToDataRef(const Fortran::evaluate::NamedEntity &namedEntity) {
    if (namedEntity.IsSymbol())
      return Fortran::evaluate::DataRef{namedEntity.GetFirstSymbol()};
    return Fortran::evaluate::DataRef{namedEntity.GetComponent()};
  }

  Fortran::evaluate::Expr<Fortran::evaluate::SomeType>
  namedEntityToExpr(const Fortran::evaluate::NamedEntity &namedEntity) {
    return Fortran::evaluate::AsGenericExpr(namedEntityToDataRef(namedEntity))
        .value();
  }

  Fortran::lower::AbstractConverter &converter;
  Fortran::lower::StatementContext &stmtCtx;
  mlir::Location loc;
  /// Elements of VectorSubscriptBox being built.
  fir::ExtendedValue loweredBase;
  llvm::SmallVector<mlir::Value> preRankedPath;
  llvm::SmallVector<LoweredSubscript, 4> loweredSubscripts;
  llvm::SmallVector<mlir::Value> componentPath;
  MaybeSubstring substringBounds;
  bool isAfterRankedPart = false;
  bool lastPartWasRanked = false;
  mlir::Type elementType;
};
} // namespace

Fortran::lower::VectorSubscriptBox Fortran::lower::genVectorSubscriptBox(
    mlir::Location loc, Fortran::lower::AbstractConverter &converter,
    Fortran::lower::StatementContext &stmtCtx,
    const Fortran::evaluate::Expr<Fortran::evaluate::SomeType> &expr) {
  return VectorSubscriptBoxBuilder(loc, converter, stmtCtx).gen(expr);
}

static llvm::SmallVector<mlir::Value> reverseAndSubstractOne(fir::FirOpBuilder& builder, mlir::Location loc, llvm::ArrayRef<mlir::Value> extents, mlir::Value one) {
  llvm::SmallVector<mlir::Value> uppers;
  auto idxTy = builder.getIndexType();
  for (auto extent : llvm::reverse(extents)) {
    extent = builder.createConvert(loc, idxTy, extent);
    uppers.push_back(builder.create<mlir::SubIOp>(loc, extent, one));
  }
  return uppers;
}

void Fortran::lower::VectorSubscriptBox::prepareForAddressing(fir::FirOpBuilder& builder, mlir::Location loc) {
  if (readyForAddressing)
    return;

  if (const auto *mutableBox = loweredBase.getBoxOf<fir::MutableBoxValue>())
    loweredBase = fir::factory::genMutableBoxRead(builder, loc, *mutableBox);

  if (isArray()) {
    shape = createShape(builder, loc);
    slice = createSlice(builder, loc);
  }
  readyForAddressing = true;
}

bool Fortran::lower::VectorSubscriptBox::isArray() const {
  return !loweredSubscripts.empty() || loweredBase.rank() == 0;
}


mlir::Value Fortran::lower::VectorSubscriptBox::loopOverElementsWhile(
    fir::FirOpBuilder &builder, mlir::Location loc,
    const ElementalGeneratorWithBoolReturn &elementalGenerator,
    mlir::Value initialCondition) {

  prepareForAddressing(builder, loc);

  if (!isArray()) {
    auto elem = getElementAt(builder, loc, llvm::None);
    return elementalGenerator(builder, loc, loweredBase, /*inductionVariables*/llvm::None);
  }

  auto idxTy = builder.getIndexType(); 
  auto extents = getExtents(builder, loc);
  auto one = builder.createIntegerConstant(loc, idxTy, 1);
  auto zero = builder.createIntegerConstant(loc, idxTy, 0);
  auto uppers = reverseAndSubstractOne(builder, loc, extents, one);
  llvm::SmallVector<mlir::Value> inductionVariables;
  fir::IterWhileOp outerLoop;
  for (auto ub : uppers) {
    auto loop =
        builder.create<fir::IterWhileOp>(loc, zero, ub, one, initialCondition);
    initialCondition = loop.getIterateVar();
    if (!outerLoop)
      outerLoop = loop;
    else
      builder.create<fir::ResultOp>(loc, loop.getResult(0));
    builder.setInsertionPointToStart(loop.getBody());
    inductionVariables.push_back(loop.getInductionVar());
  }
  assert(outerLoop && !inductionVariables.empty() &&
         "at least one loop should be created");

  auto elem = getElementAt(builder, loc, inductionVariables);

  auto res = elementalGenerator(builder, loc, elem, inductionVariables);
  builder.create<fir::ResultOp>(loc, res);
  builder.setInsertionPointAfter(outerLoop);
  return outerLoop.getResult(0);
}

void Fortran::lower::VectorSubscriptBox::loopOverElements(fir::FirOpBuilder& builder, mlir::Location loc, const ElementalGenerator& doOnEachElement, const ElementalMask* filter, bool canLoopUnordered){
  prepareForAddressing(builder, loc);
  
  if (!isArray()) {
    assert(!filter && "no filter expected for scalars");
    auto elem = getElementAt(builder, loc, llvm::None);
    doOnEachElement(builder, loc, elem, /*inductionVariables*/llvm::None);
    return; 
  }

  auto idxTy = builder.getIndexType(); 
  auto extents = getExtents(builder, loc);

  auto one = builder.createIntegerConstant(loc, idxTy, 1);
  auto zero = builder.createIntegerConstant(loc, idxTy, 0);
  auto uppers = reverseAndSubstractOne(builder, loc, extents, one);
  
  llvm::SmallVector<mlir::Value> inductionVariables;
  fir::DoLoopOp outerLoop;
   
  for (auto ub: uppers) {
    auto loop = builder.create<fir::DoLoopOp>(loc, zero, ub, one, canLoopUnordered);
    if (!outerLoop)
      outerLoop = loop;
    builder.setInsertionPointToStart(loop.getBody());
    inductionVariables.push_back(loop.getInductionVar());
  }

  assert(outerLoop && !inductionVariables.empty() &&
         "at least one loop should be created");

  // Create if-ops nest filter if any.
  if (filter)
    (*filter)(builder, loc, inductionVariables);
  auto elem = getElementAt(builder, loc, inductionVariables);
  doOnEachElement(builder, loc, elem, inductionVariables);
  builder.setInsertionPointAfter(outerLoop);
}

const fir::ExtendedValue&
Fortran::lower::VectorSubscriptBox::getBase() const {
  return loweredBase;
}

mlir::Value
Fortran::lower::VectorSubscriptBox::createShape(fir::FirOpBuilder &builder,
                                                mlir::Location loc) const {
  return builder.createShape(loc, loweredBase);
}

static mlir::Value getLengthFromComponentPath(fir::FirOpBuilder &builder, mlir::Location loc, llvm::ArrayRef<mlir::Value> componentPath) {
  fir::FieldIndexOp lastField;
  for (auto component : llvm::reverse(componentPath)) {
    if (auto field = component.getDefiningOp<fir::FieldIndexOp>()) {
      lastField = field;
      break;
    }
  }
  if (!lastField)
    fir::emitFatalError(loc, "expected component reference in designator");
  auto recTy = lastField.on_type().cast<fir::RecordType>(); 
  auto charType = recTy.getType(lastField.field_id()).cast<fir::CharacterType>();
  // Derived type components with non constant length are F2003.
  if (charType.hasDynamicLen())
    TODO(loc, "designator with derived type length parameters");
  return builder.createIntegerConstant(loc, builder.getCharacterLengthType(), charType.getLen());
}

llvm::SmallVector<mlir::Value>
Fortran::lower::VectorSubscriptBox::getTypeParams(fir::FirOpBuilder &builder, mlir::Location loc) const {
  auto elementType = getElementType();
  if (elementType.isa<fir::CharacterType>()) {
    mlir::Value len = componentPath.empty() ?
      fir::factory::readCharLen(builder, loc, loweredBase) :
      getLengthFromComponentPath(builder, loc, componentPath);
    if (substringBounds.empty())
      return {len};
    auto upper = substringBounds.size() == 2 ? substringBounds[1] : len;
    auto charLenType = builder.getCharacterLengthType();
    upper = builder.createConvert(loc, charLenType, upper); 
    auto lower = builder.createConvert(loc, charLenType, substringBounds[0]); 
    auto zero = builder.createIntegerConstant(loc, charLenType, 0);
    auto one = builder.createIntegerConstant(loc, charLenType, 1);
    auto diff = builder.create<mlir::SubIOp>(loc, upper, lower);
    auto newLen = builder.create<mlir::AddIOp>(loc, diff, one);
    auto cmp = builder.create<mlir::CmpIOp>(loc, mlir::CmpIPredicate::sle, lower, upper);
    auto select = builder.create<mlir::SelectOp>(loc, cmp, newLen, zero);
    return {select.getResult()};
  }
  if (auto recordType = elementType.dyn_cast<fir::RecordType>())
    if (recordType.getNumLenParams() != 0)
      TODO(loc, "derived type designator with length parameters");
  return {};
}

mlir::Value
Fortran::lower::VectorSubscriptBox::createSlice(fir::FirOpBuilder &builder,
                                                mlir::Location loc) const {
  auto idxTy = builder.getIndexType();
  llvm::SmallVector<mlir::Value> triples;
  auto one = builder.createIntegerConstant(loc, idxTy, 1);
  auto undef = builder.create<fir::UndefOp>(loc, idxTy);
  for (const auto &subscript : loweredSubscripts)
    std::visit(Fortran::common::visitors{
                   [&](const LoweredTriplet &triplet) {
                     triples.emplace_back(triplet.lb);
                     triples.emplace_back(triplet.ub);
                     triples.emplace_back(triplet.stride);
                   },
                   [&](const LoweredVectorSubscript &vector) {
                     triples.emplace_back(one);
                     triples.emplace_back(vector.size);
                     triples.emplace_back(one);
                   },
                   [&](const mlir::Value &i) {
                     triples.emplace_back(i);
                     triples.emplace_back(undef);
                     triples.emplace_back(undef);
                   },
               },
               subscript);
  auto sliceTy =
      fir::SliceType::get(builder.getContext(), loweredSubscripts.size());
  return builder.create<fir::SliceOp>(loc, sliceTy, triples, componentPath);
}

/// Compute the shape of a slice (TODO share in fir::factory)
static llvm::SmallVector<mlir::Value> computeSliceShape(fir::FirOpBuilder& builder, mlir::Location loc, mlir::Value slice) {
  llvm::SmallVector<mlir::Value> slicedShape;
  auto slOp = mlir::cast<fir::SliceOp>(slice.getDefiningOp());
  auto triples = slOp.triples();
  auto idxTy = builder.getIndexType();
  for (unsigned i = 0, end = triples.size(); i < end; i += 3) {
    if (!mlir::isa_and_nonnull<fir::UndefOp>(
            triples[i + 1].getDefiningOp())) {
      // (..., lb:ub:step, ...) case:  extent = max((ub-lb+step)/step, 0)
      // See Fortran 2018 9.5.3.3.2 section for more details.
      auto res = builder.genExtentFromTriplet(loc, triples[i], triples[i + 1],
                                              triples[i + 2], idxTy);
      slicedShape.emplace_back(res);
    } else {
      // do nothing. `..., i, ...` case, so dimension is dropped.
    }
  }
  return slicedShape;
}

llvm::SmallVector<mlir::Value>
Fortran::lower::VectorSubscriptBox::getExtents(fir::FirOpBuilder &builder,
                                                  mlir::Location loc) const {
  if (slice)
    return computeSliceShape(builder, loc, slice);
  if (shape && !shape.getType().isa<fir::ShiftType>()) {
    auto extents = fir::factory::getExtents(shape);
    return {extents.begin(), extents.end()};
  }
  
  if (loweredSubscripts.empty())
    return fir::factory::getExtents(builder, loc, loweredBase);
  
  auto idxTy = builder.getIndexType();
  llvm::SmallVector<mlir::Value> extents;
  for (const auto &subscript : loweredSubscripts) {
    if (std::holds_alternative<mlir::Value>(subscript))
      continue;
    if (const auto *triplet = std::get_if<LoweredTriplet>(&subscript)) {
      auto ext = builder.genExtentFromTriplet(loc, triplet->lb, triplet->ub,
                                                 triplet->stride, idxTy);
      extents.push_back(ext);
    } else {
      const auto &vector = std::get<LoweredVectorSubscript>(subscript);
      extents.push_back(vector.size);
    }
  }
  return extents;
}

fir::ExtendedValue Fortran::lower::VectorSubscriptBox::getElementAt(
    fir::FirOpBuilder &builder, mlir::Location loc,
    mlir::ValueRange indices) const {
  auto element = isArray() ? genArrayCoor(builder, loc, indices) : loweredBase;
  if (!substringBounds.empty()) {
    auto *charBox = element.getCharBox();
    assert(charBox && "substring requires CharBox base");
    fir::factory::CharacterExprHelper helper{builder, loc};
    return helper.createSubstring(*charBox, substringBounds);
  }
  return element;
}

fir::ExtendedValue Fortran::lower::VectorSubscriptBox::genArrayCoor(
    fir::FirOpBuilder &builder, mlir::Location loc, mlir::ValueRange indices) const {
  /// Generate the indexes for the array_coor inside the loops.
  llvm::SmallVector<mlir::Value> inductionVariables;
  auto memrefTy = fir::getBase(getBase()).getType();
  auto idx = fir::factory::originateIndices(loc, builder, memrefTy, shape, indices);
  inductionVariables.append(idx.begin(), idx.end());
  auto idxTy = builder.getIndexType();
  llvm::SmallVector<mlir::Value> indexes;
  auto inductionIdx = inductionVariables.size() - 1;
  for (const auto &subscript : loweredSubscripts)
    std::visit(Fortran::common::visitors{
                   [&](const LoweredTriplet &triplet) {
                     indexes.emplace_back(inductionVariables[inductionIdx--]);
                   },
                   [&](const LoweredVectorSubscript &vector) {
                     auto vecIndex = inductionVariables[inductionIdx--];
                     auto vecEltRef = vector.getVector().getElementAt(builder, loc, {vecIndex});
                     auto vecElt = builder.create<fir::LoadOp>(loc, fir::getBase(vecEltRef));
                     indexes.emplace_back(
                         builder.createConvert(loc, idxTy, vecElt));
                   },
                   [&](const mlir::Value &i) {
                     indexes.emplace_back(builder.createConvert(loc, idxTy, i));
                   },
               },
               subscript);
  auto refTy = builder.getRefType(getElementType());
  auto elementAddr = builder.create<fir::ArrayCoorOp>(
      loc, refTy, fir::getBase(loweredBase), shape, slice, indexes,
      fir::getTypeParams(loweredBase));
  return fir::factory::arraySectionElementToExtendedValue(
      builder, loc, loweredBase, elementAddr, slice);
}

bool Fortran::lower::VectorSubscriptBox::hasVectorSubscripts() const {
  for (const auto &subscript : loweredSubscripts)
    if (std::holds_alternative<LoweredVectorSubscript>(subscript))
      return true;
  return false;
}

fir::ExtendedValue Fortran::lower::VectorSubscriptBox::asBox(fir::FirOpBuilder& builder, mlir::Location loc) const {
  auto memref = fir::getBase(loweredBase);
  auto type = fir::unwrapPassByRefType(memref.getType());
  auto boxTy = fir::BoxType::get(type);
  auto shape = createShape(builder, loc);
  auto slice = createSlice(builder, loc);
  if (memref.getType().isa<fir::BoxType>())
    return builder.create<fir::ReboxOp>(loc, boxTy, memref, shape, slice);
  return builder.create<fir::EmboxOp>(loc, boxTy, memref, shape, slice, fir::getTypeParams(loweredBase));
}

fir::ExtendedValue Fortran::lower::VectorSubscriptBox::getAsExtendedValue(fir::FirOpBuilder& builder, mlir::Location loc) const {
  if (isExtendedValue())
    return loweredBase;
  auto memref = fir::getBase(loweredBase);
  auto type = fir::unwrapPassByRefType(memref.getType());
  auto boxTy = fir::BoxType::get(type);
  auto shape = createShape(builder, loc);
  auto slice = createSlice(builder, loc);
  if (memref.getType().isa<fir::BoxType>())
    return builder.create<fir::ReboxOp>(loc, boxTy, memref, shape, slice);
  return builder.create<fir::EmboxOp>(loc, boxTy, memref, shape, slice, fir::getTypeParams(loweredBase));
}

fir::ExtendedValue Fortran::lower::Variable::getElementAt(fir::FirOpBuilder& builder, mlir::Location loc, mlir::ValueRange indices) const {
  assert(readyForAddressing && "array was not prepared for addressing");
  return std::visit(Fortran::common::visitors{
    [&](const fir::ExtendedValue& exv) {
      if (fir::isArray(exv))
        return fir::factory::getElementAt(builder, loc, exv, shape, slice, indices);
      return exv;
    },
    [&](const ArraySection& arraySection) {
      return arraySection.getElementAt(builder, loc, indices);
    }
  }, var);
}

Fortran::lower::Variable Fortran::lower::genVariable(Fortran::lower::AbstractConverter &converter, mlir::Location loc,
  Fortran::lower::StatementContext &stmtCtx,
  const Fortran::evaluate::Expr<Fortran::evaluate::SomeType> &expr) {
  if (const auto* sym = Fortran::evaluate::UnwrapWholeSymbolOrComponentDataRef(expr)) {
    // Whole symbol like `x` of `a(i, j)%p1%...%pn` where non of the parts before
    // pn are ranked.
    if (Fortran::semantics::IsAllocatableOrPointer(*sym))
      return Variable(converter.genExprMutableBox(loc, expr));
    return Variable(converter.genExprAddr(expr, stmtCtx, &loc));
  }
  // TODO: catch pointer function ref

  // Scalar array reference.
  if (expr.Rank() == 0)
    return Variable(converter.genExprAddr(expr, stmtCtx, &loc));

  // Ranked array sections.
  return Variable(Fortran::lower::genVectorSubscriptBox(loc, converter, stmtCtx, expr));
}

void Fortran::lower::Variable::prepareForAddressing(fir::FirOpBuilder& builder, mlir::Location loc) {
  if (readyForAddressing)
    return;
  if (const auto *exv = std::get_if<fir::ExtendedValue>(&var)) {
    if (const auto *mutableBox = exv->getBoxOf<fir::MutableBoxValue>()) {
      var = fir::factory::genMutableBoxRead(builder, loc, *mutableBox);
    }
  }

  if (isArray())
    std::visit(Fortran::common::visitors{
      [&](const fir::ExtendedValue& exv) {
        shape = builder.createShape(loc, exv);
      },
      [&](const ArraySection& arraySection) {
        shape = arraySection.createShape(builder, loc);
        slice = arraySection.createSlice(builder, loc);
      }
      }, var);
  readyForAddressing = true;
}

void Fortran::lower::Variable::loopOverElements(fir::FirOpBuilder& builder, mlir::Location loc, const ElementalGenerator& doOnEachElement, const ElementalMask* filter, bool canLoopUnordered){
  prepareForAddressing(builder, loc);
  
  if (!isArray()) {
    assert(!filter && "no filter expected for scalars");
    const auto& elem = std::get<fir::ExtendedValue>(var);
    doOnEachElement(builder, loc, elem, /*inductionVariables*/llvm::None);
    return; 
  }

  auto idxTy = builder.getIndexType(); 
  auto extents = getExtents(builder, loc);

  auto one = builder.createIntegerConstant(loc, idxTy, 1);
  auto zero = builder.createIntegerConstant(loc, idxTy, 0);
  auto uppers = reverseAndSubstractOne(builder, loc, extents, one);
  
  llvm::SmallVector<mlir::Value> inductionVariables;
  fir::DoLoopOp outerLoop;
   
  for (auto ub: uppers) {
    auto loop = builder.create<fir::DoLoopOp>(loc, zero, ub, one, canLoopUnordered);
    if (!outerLoop)
      outerLoop = loop;
    builder.setInsertionPointToStart(loop.getBody());
    inductionVariables.push_back(loop.getInductionVar());
  }

  assert(outerLoop && !inductionVariables.empty() &&
         "at least one loop should be created");

  // Create if-ops nest filter if any.
  if (filter)
    (*filter)(builder, loc, inductionVariables);
  auto elem = getElementAt(builder, loc, inductionVariables);
  doOnEachElement(builder, loc, elem, inductionVariables);
  builder.setInsertionPointAfter(outerLoop);
}

llvm::SmallVector<mlir::Value> Fortran::lower::Variable::getTypeParams(fir::FirOpBuilder &builder, mlir::Location loc) const {
  return std::visit(Fortran::common::visitors{
    [&](const fir::ExtendedValue& exv) {
      return fir::factory::getTypeParams(builder, loc, exv);
    },
    [&](const ArraySection& arraySection) {
      return arraySection.getTypeParams(builder, loc);
    },
  }, var);
}

llvm::SmallVector<mlir::Value> Fortran::lower::Variable::getExtents(fir::FirOpBuilder &builder, mlir::Location loc) const {
  if (slice)
    return computeSliceShape(builder, loc, slice);
  if (shape && !shape.getType().isa<fir::ShiftType>()) {
    auto extents = fir::factory::getExtents(shape);
    return {extents.begin(), extents.end()};
  }
  
  return std::visit(Fortran::common::visitors{
    [&](const fir::ExtendedValue& exv) {
      return fir::factory::getExtents(builder, loc, exv);
    },
    [&](const ArraySection& arraySection) {
      return fir::factory::getExtents(builder, loc, arraySection.getBase());
    },
  }, var);
}

llvm::SmallVector<mlir::Value> Fortran::lower::Variable::getLBounds(fir::FirOpBuilder &builder, mlir::Location loc) const {
  if (shape) {
    auto origins = fir::factory::getOrigins(shape);
    return {origins.begin(), origins.end()};
  }
  
  return std::visit(Fortran::common::visitors{
    [&](const fir::ExtendedValue& exv) -> llvm::SmallVector<mlir::Value> {
      auto lbs = exv.match(
        [&](fir::CharArrayBoxValue& array) -> llvm::ArrayRef<mlir::Value> {
          return array.getLBounds();
        },
        [&](fir::ArrayBoxValue& array) -> llvm::ArrayRef<mlir::Value> {
          return array.getLBounds();
        },
        [&](fir::BoxValue& array) -> llvm::ArrayRef<mlir::Value> {
          return array.getLBounds();
        },
        [&](auto&) -> llvm::ArrayRef<mlir::Value> {
          return {};
        }
      );
      return {lbs.begin(), lbs.end()};
    },
    [&](const ArraySection&) -> llvm::SmallVector<mlir::Value> {
      // Array section that are not whole symbols or component have
      // no lower bounds.
      return {};
    },
  }, var);
}

bool Fortran::lower::Variable::hasVectorSubscripts() const {
  return std::visit(Fortran::common::visitors{
    [&](const fir::ExtendedValue& exv) {
      return false;
    },
    [&](const ArraySection& arraySection) {
      return arraySection.hasVectorSubscripts();
    },
  }, var);
}

bool Fortran::lower::Variable::isArray() const {
  return std::visit(Fortran::common::visitors{
    [&](const fir::ExtendedValue& exv) {
      return exv.rank() > 0;
    },
    [&](const ArraySection& arraySection) {
      return true;
    },
  }, var);
}

fir::ExtendedValue Fortran::lower::Variable::getAsExtendedValue(fir::FirOpBuilder& builder, mlir::Location loc) const {
  return std::visit(Fortran::common::visitors{
    [&](const fir::ExtendedValue& exv) {
      return exv;
    },
    [&](const ArraySection& arraySection) {
      return arraySection.asBox(builder, loc);
    },
  }, var);
}

void Fortran::lower::Variable::reallocate(fir::FirOpBuilder& builder, mlir::Location loc, llvm::ArrayRef<mlir::Value> lbounds, llvm::ArrayRef<mlir::Value> extents, llvm::ArrayRef<mlir::Value> typeParams) const {
  if (const auto *exv = std::get_if<fir::ExtendedValue>(&var))
    if (const auto *mutableBox = exv->getBoxOf<fir::MutableBoxValue>()) {
      fir::factory::genReallocIfNeeded(builder, loc, *mutableBox, lbounds, extents, typeParams);
      return;
  }
  
  fir::emitFatalError(loc, "trying to reallocate non-allocatable");
}

void Fortran::lower::Variable::genAssign(fir::FirOpBuilder& builder, mlir::Location loc, const Fortran::lower::ExprLower& expr, const ElementalMask* filter) {
  auto elemAssign = [&](fir::FirOpBuilder& b, mlir::Location l, const fir::ExtendedValue& lhsElt, llvm::ArrayRef<mlir::Value> indices) {
    auto rhsElt = expr.getElementAt(b, l, indices);
    fir::factory::assignScalars(b, l, lhsElt, rhsElt);
  };
  loopOverElements(builder, loc, elemAssign, filter, expr.canLoopUnorderedOverElements());
}

void Fortran::lower::Variable::genAssign(fir::FirOpBuilder& builder, mlir::Location loc, const Variable& var, const ElementalMask* filter) {
  Fortran::lower::Variable v(var);
  v.prepareForAddressing(builder, loc);
  Fortran::lower::ExprLower expr(std::move(v));
  genAssign(builder, loc, expr, filter);
}

Fortran::lower::VectorSubscriptBox::LoweredVectorSubscript::LoweredVectorSubscript(const LoweredVectorSubscript&) = default;
Fortran::lower::VectorSubscriptBox::LoweredVectorSubscript::LoweredVectorSubscript(LoweredVectorSubscript&&) = default;
Fortran::lower::VectorSubscriptBox::LoweredVectorSubscript& Fortran::lower::VectorSubscriptBox::LoweredVectorSubscript::operator=(const LoweredVectorSubscript&) = default;
Fortran::lower::VectorSubscriptBox::LoweredVectorSubscript& Fortran::lower::VectorSubscriptBox::LoweredVectorSubscript::operator=(LoweredVectorSubscript&&) = default;
Fortran::lower::VectorSubscriptBox::LoweredVectorSubscript::~LoweredVectorSubscript() = default;

Fortran::lower::VectorSubscriptBox::LoweredVectorSubscript::LoweredVectorSubscript(Fortran::lower::ExprLower&& expr, mlir::Value size) : vector{std::move(expr)}, size{size} {}

const Fortran::lower::ExprLower& Fortran::lower::VectorSubscriptBox::LoweredVectorSubscript::getVector() const {
  return vector.value();
}
Fortran::lower::ExprLower& Fortran::lower::VectorSubscriptBox::LoweredVectorSubscript::getVector() {
  return vector.value();
}
