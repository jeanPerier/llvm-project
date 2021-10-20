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
    // Never visited because expr lowering is used to lowered the ranked
    // ArrayRef.
    fir::emitFatalError(
        loc, "expected at least one ArrayRef with vector susbcripts");
  }

  mlir::Type gen(const Fortran::evaluate::Substring &substring) {
    // StaticDataObject::Pointer bases are constants and cannot be
    // subscripted, so the base must be a DataRef here.
    auto baseElementType =
        gen(std::get<Fortran::evaluate::DataRef>(substring.parent()));
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
    auto &builder = converter.getFirOpBuilder();
    auto i32Ty = builder.getI32Type(); // llvm's GEP requires i32
    auto offset = builder.createIntegerConstant(
        loc, i32Ty,
        complexPart.part() == Fortran::evaluate::ComplexPart::Part::RE ? 0 : 1);
    componentPath.emplace_back(offset);
    return fir::factory::ComplexExprHelper{builder, loc}.getComplexPartType(
        complexType);
  }

  mlir::Type gen(const Fortran::evaluate::Component &component) {
    auto recTy = gen(component.base()).cast<fir::RecordType>();
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
    componentPath.emplace_back(builder.create<fir::FieldIndexOp>(
        loc, fldTy, componentName, recTy, /*typeParams*/ llvm::None));
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
      componentPath.emplace_back(genScalarValue(expr.value()));
    }
    return elementType;
  }

  /// Lower the subscripts and base of the ArrayRef that is an array (there must
  /// be one since there is a vector subscript, and there can only be one
  /// according to C925).
  mlir::Type genRankedArrayRefSubscriptAndBase(
      const Fortran::evaluate::ArrayRef &arrayRef) {
    // Lower the save the base
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
                  auto vector = converter.genExprAddr(
                      ignoreEvConvert(expr.value()), stmtCtx);
                  auto size =
                      fir::factory::readExtent(builder, loc, vector, /*dim=*/0);
                  size = builder.createConvert(loc, idxTy, size);
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
    return fir::unwrapSequenceType(
        fir::unwrapPassByRefType(fir::getBase(loweredBase).getType()));
  }

  mlir::Type gen(const Fortran::evaluate::CoarrayRef &) {
    // Is this possible/legal ?
    TODO(loc, "Coarray ref with vector subscript in IO input");
  }

  template <typename A>
  mlir::Value genScalarValue(const A &expr) {
    return fir::getBase(converter.genExprValue(toEvExpr(expr), stmtCtx));
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
  llvm::SmallVector<LoweredSubscript, 16> loweredSubscripts;
  llvm::SmallVector<mlir::Value> componentPath;
  MaybeSubstring substringBounds;
  mlir::Type elementType;
};
} // namespace

Fortran::lower::VectorSubscriptBox Fortran::lower::genVectorSubscriptBox(
    mlir::Location loc, Fortran::lower::AbstractConverter &converter,
    Fortran::lower::StatementContext &stmtCtx,
    const Fortran::evaluate::Expr<Fortran::evaluate::SomeType> &expr) {
  return VectorSubscriptBoxBuilder(loc, converter, stmtCtx).gen(expr);
}

template <typename LoopType, typename Generator>
mlir::Value Fortran::lower::VectorSubscriptBox::loopOverElementsBase(
    fir::FirOpBuilder &builder, mlir::Location loc,
    const Generator &elementalGenerator,
    [[maybe_unused]] mlir::Value initialCondition) {
  auto shape = createShape(builder, loc);
  auto slice = createSlice(builder, loc);

  // Create loop nest for triplets and vector subscripts in column
  // major order.
  llvm::SmallVector<mlir::Value> inductionVariables;
  LoopType outerLoop;
  for (auto [lb, ub, step] : genLoopBounds(builder, loc)) {
    LoopType loop;
    if constexpr (std::is_same_v<LoopType, fir::IterWhileOp>) {
      loop =
          builder.create<fir::IterWhileOp>(loc, lb, ub, step, initialCondition);
      initialCondition = loop.getIterateVar();
      if (!outerLoop)
        outerLoop = loop;
      else
        builder.create<fir::ResultOp>(loc, loop.getResult(0));
    } else {
      loop =
          builder.create<fir::DoLoopOp>(loc, lb, ub, step, /*unordered=*/false);
      if (!outerLoop)
        outerLoop = loop;
    }
    builder.setInsertionPointToStart(loop.getBody());
    inductionVariables.push_back(loop.getInductionVar());
  }
  assert(outerLoop && !inductionVariables.empty() &&
         "at least one loop should be created");

  auto elem = getElementAt(builder, loc, shape, slice, inductionVariables);

  if constexpr (std::is_same_v<LoopType, fir::IterWhileOp>) {
    auto res = elementalGenerator(elem);
    builder.create<fir::ResultOp>(loc, res);
    builder.setInsertionPointAfter(outerLoop);
    return outerLoop.getResult(0);
  } else {
    elementalGenerator(elem);
    builder.setInsertionPointAfter(outerLoop);
    return {};
  }
}

void Fortran::lower::VectorSubscriptBox::loopOverElements(
    fir::FirOpBuilder &builder, mlir::Location loc,
    const ElementalGenerator &elementalGenerator) {
  mlir::Value initialCondition;
  loopOverElementsBase<fir::DoLoopOp, ElementalGenerator>(
      builder, loc, elementalGenerator, initialCondition);
}

mlir::Value Fortran::lower::VectorSubscriptBox::loopOverElementsWhile(
    fir::FirOpBuilder &builder, mlir::Location loc,
    const ElementalGeneratorWithBoolReturn &elementalGenerator,
    mlir::Value initialCondition) {
  return loopOverElementsBase<fir::IterWhileOp,
                              ElementalGeneratorWithBoolReturn>(
      builder, loc, elementalGenerator, initialCondition);
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

/// Generate zero base loop bounds.
llvm::SmallVector<std::tuple<mlir::Value, mlir::Value, mlir::Value>>
Fortran::lower::VectorSubscriptBox::genLoopBounds(fir::FirOpBuilder &builder,
                                                  mlir::Location loc) {
  auto idxTy = builder.getIndexType();
  auto one = builder.createIntegerConstant(loc, idxTy, 1);
  auto zero = builder.createIntegerConstant(loc, idxTy, 0);
  llvm::SmallVector<std::tuple<mlir::Value, mlir::Value, mlir::Value>> bounds;
  auto dimension = loweredSubscripts.size();
  for (const auto &subscript : llvm::reverse(loweredSubscripts)) {
    --dimension;
    if (std::holds_alternative<mlir::Value>(subscript))
      continue;
    mlir::Value lb, ub, step;
    if (const auto *triplet = std::get_if<LoweredTriplet>(&subscript)) {
      lb = zero;
      ub = builder.genExtentFromTriplet(loc, triplet->lb, triplet->ub,
                                                 triplet->stride, idxTy);
      step = one;
    } else {
      const auto &vector = std::get<LoweredVectorSubscript>(subscript);
      lb = zero;
      ub = builder.create<mlir::SubIOp>(loc, idxTy, vector.size, one);
      step = one;
    }
    bounds.emplace_back(lb, ub, step);
  }
  return bounds;
}

fir::ExtendedValue Fortran::lower::VectorSubscriptBox::getElementAt(
    fir::FirOpBuilder &builder, mlir::Location loc, mlir::Value shape,
    mlir::Value slice, mlir::ValueRange indices) const {
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
                     auto vecBase = fir::getBase(vector.vector);
                     auto vecEleTy = fir::unwrapSequenceType(
                         fir::unwrapPassByRefType(vecBase.getType()));
                     auto refTy = builder.getRefType(vecEleTy);
                     auto vecEltRef = builder.create<fir::CoordinateOp>(
                         loc, refTy, vecBase, vecIndex);
                     auto vecElt =
                         builder.create<fir::LoadOp>(loc, vecEleTy, vecEltRef);
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
  auto element = fir::factory::arraySectionElementToExtendedValue(
      builder, loc, loweredBase, elementAddr, slice);
  if (!substringBounds.empty()) {
    auto *charBox = element.getCharBox();
    assert(charBox && "substring requires CharBox base");
    fir::factory::CharacterExprHelper helper{builder, loc};
    return helper.createSubstring(*charBox, substringBounds);
  }
  return element;
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
  return builder.create<fir::EmboxOp>(loc, boxTy, memref, shape, slice, fir::getTypeParams(memref));
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
      return arraySection.getElementAt(builder, loc, shape, slice, indices);
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
  
  llvm::SmallVector<mlir::Value> uppers;
  for (auto extent : llvm::reverse(extents)) {
    extent = builder.createConvert(loc, idxTy, extent);
    uppers.push_back(builder.create<mlir::SubIOp>(loc, extent, one));
  }

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
