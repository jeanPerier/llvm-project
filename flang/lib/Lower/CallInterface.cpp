//===-- CallInterface.cpp -- Procedure call interface ------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "flang/Lower/CallInterface.h"
#include "StatementContext.h"
#include "flang/Evaluate/characteristics.h"
#include "flang/Evaluate/fold.h"
#include "flang/Lower/Bridge.h"
#include "flang/Lower/Mangler.h"
#include "flang/Lower/PFTBuilder.h"
#include "flang/Lower/Todo.h"
#include "flang/Optimizer/Builder/FIRBuilder.h"
#include "flang/Optimizer/Dialect/FIRDialect.h"
#include "flang/Optimizer/Dialect/FIROpsSupport.h"
#include "flang/Optimizer/Support/InternalNames.h"
#include "flang/Semantics/symbol.h"
#include "flang/Semantics/tools.h"

//===----------------------------------------------------------------------===//
// BIND(C) mangling helpers
//===----------------------------------------------------------------------===//

// Return the binding label (from BIND(C...)) or the mangled name of a symbol.
static std::string getMangledName(const Fortran::semantics::Symbol &symbol) {
  const std::string *bindName = symbol.GetBindName();
  return bindName ? *bindName : Fortran::lower::mangle::mangleName(symbol);
}

//===----------------------------------------------------------------------===//
// Caller side interface implementation
//===----------------------------------------------------------------------===//

bool Fortran::lower::CallerInterface::hasAlternateReturns() const {
  return procRef.hasAlternateReturns();
}

std::string Fortran::lower::CallerInterface::getMangledName() const {
  const auto &proc = procRef.proc();
  if (const auto *symbol = proc.GetSymbol())
    return ::getMangledName(symbol->GetUltimate());
  assert(proc.GetSpecificIntrinsic() &&
         "expected intrinsic procedure in designator");
  return proc.GetName();
}

const Fortran::semantics::Symbol *
Fortran::lower::CallerInterface::getProcedureSymbol() const {
  return procRef.proc().GetSymbol();
}

bool Fortran::lower::CallerInterface::isIndirectCall() const {
  if (const auto *symbol = procRef.proc().GetSymbol())
    return Fortran::semantics::IsPointer(*symbol) ||
           Fortran::semantics::IsDummy(*symbol);
  return false;
}

const Fortran::semantics::Symbol *
Fortran::lower::CallerInterface::getIfIndirectCallSymbol() const {
  if (const auto *symbol = procRef.proc().GetSymbol())
    if (Fortran::semantics::IsPointer(*symbol) ||
        Fortran::semantics::IsDummy(*symbol))
      return symbol;
  return nullptr;
}

mlir::Location Fortran::lower::CallerInterface::getCalleeLocation() const {
  const auto &proc = procRef.proc();
  // FIXME: If the callee is defined in the same file but after the current
  // unit we cannot get its location here and the funcOp is created at the
  // wrong location (i.e, the caller location).
  if (const auto *symbol = proc.GetSymbol())
    return converter.genLocation(symbol->name());
  // Unknown location for intrinsics.
  return converter.genLocation();
}

// Get dummy argument characteristic for a procedure with implicit interface
// from the actual argument characteristic. The actual argument may not be a F77
// entity. The attribute must be dropped and the shape, if any, must be made
// explicit.
static Fortran::evaluate::characteristics::DummyDataObject
asImplicitArg(Fortran::evaluate::characteristics::DummyDataObject &&dummy) {
  Fortran::evaluate::Shape shape =
      dummy.type.attrs().none() ? dummy.type.shape()
                                : Fortran::evaluate::Shape(dummy.type.Rank());
  return Fortran::evaluate::characteristics::DummyDataObject(
      Fortran::evaluate::characteristics::TypeAndShape(dummy.type.type(),
                                                       std::move(shape)));
}

static Fortran::evaluate::characteristics::DummyArgument
asImplicitArg(Fortran::evaluate::characteristics::DummyArgument &&dummy) {
  return std::visit(
      Fortran::common::visitors{
          [&](Fortran::evaluate::characteristics::DummyDataObject &obj) {
            return Fortran::evaluate::characteristics::DummyArgument(
                std::move(dummy.name), asImplicitArg(std::move(obj)));
          },
          [&](Fortran::evaluate::characteristics::DummyProcedure &proc) {
            return Fortran::evaluate::characteristics::DummyArgument(
                std::move(dummy.name), std::move(proc));
          },
          [](Fortran::evaluate::characteristics::AlternateReturn &x) {
            return Fortran::evaluate::characteristics::DummyArgument(
                std::move(x));
          }},
      dummy.u);
}

Fortran::evaluate::characteristics::Procedure
Fortran::lower::CallerInterface::characterize() const {
  auto &foldingContext = converter.getFoldingContext();
  auto characteristic =
      Fortran::evaluate::characteristics::Procedure::Characterize(
          procRef.proc(), foldingContext);
  assert(characteristic && "Failed to get characteristic from procRef");
  // The characteristic may not contain the argument characteristic if the
  // ProcedureDesignator has no interface.
  if (!characteristic->HasExplicitInterface()) {
    for (const auto &arg : procRef.arguments()) {
      if (arg.value().isAlternateReturn()) {
        characteristic->dummyArguments.emplace_back(
            Fortran::evaluate::characteristics::AlternateReturn{});
      } else {
        // Argument cannot be optional with implicit interface
        const auto *expr = arg.value().UnwrapExpr();
        assert(
            expr &&
            "argument in call with implicit interface cannot be assumed type");
        auto argCharacteristic =
            Fortran::evaluate::characteristics::DummyArgument::FromActual(
                "actual", *expr, foldingContext);
        assert(argCharacteristic &&
               "failed to characterize argument in implicit call");
        characteristic->dummyArguments.emplace_back(
            asImplicitArg(std::move(*argCharacteristic)));
      }
    }
  }
  return *characteristic;
}

void Fortran::lower::CallerInterface::placeInput(
    const PassedEntity &passedEntity, mlir::Value arg) {
  assert(static_cast<int>(actualInputs.size()) > passedEntity.firArgument &&
         passedEntity.firArgument >= 0 &&
         passedEntity.passBy != CallInterface::PassEntityBy::AddressAndLength &&
         "bad arg position");
  actualInputs[passedEntity.firArgument] = arg;
}

void Fortran::lower::CallerInterface::placeAddressAndLengthInput(
    const PassedEntity &passedEntity, mlir::Value addr, mlir::Value len) {
  assert(static_cast<int>(actualInputs.size()) > passedEntity.firArgument &&
         static_cast<int>(actualInputs.size()) > passedEntity.firLength &&
         passedEntity.firArgument >= 0 && passedEntity.firLength >= 0 &&
         passedEntity.passBy == CallInterface::PassEntityBy::AddressAndLength &&
         "bad arg position");
  actualInputs[passedEntity.firArgument] = addr;
  actualInputs[passedEntity.firLength] = len;
}

bool Fortran::lower::CallerInterface::verifyActualInputs() const {
  if (getNumFIRArguments() != actualInputs.size())
    return false;
  for (auto arg : actualInputs) {
    if (!arg)
      return false;
  }
  return true;
}

template <typename T>
static inline auto AsGenericExpr(T e) {
  return Fortran::evaluate::AsGenericExpr(Fortran::common::Clone(e));
}

void Fortran::lower::CallerInterface::walkResultLengths(
    ExprVisitor visitor) const {
  assert(characteristic && "characteristic was not computed");
  const auto &result = characteristic->functionResult.value();
  const auto *typeAndShape = result.GetTypeAndShape();
  assert(typeAndShape && "no result type");
  auto dynamicType = typeAndShape->type();
  // Visit result length specification expressions that are explicit.
  if (dynamicType.category() == Fortran::common::TypeCategory::Character) {
    if (auto length = dynamicType.GetCharLength())
      visitor(AsGenericExpr(*length));
  } else if (dynamicType.category() == common::TypeCategory::Derived) {
    const auto &derivedTypeSpec = dynamicType.GetDerivedTypeSpec();
    if (Fortran::semantics::CountLenParameters(derivedTypeSpec) > 0)
      TODO(converter.genLocation(),
           "function result with derived type length parameters");
  }
}

void Fortran::lower::CallerInterface::walkResultExtents(
    ExprVisitor visitor) const {
  using Attr = Fortran::evaluate::characteristics::TypeAndShape::Attr;
  assert(characteristic && "characteristic was not computed");
  const auto &result = characteristic->functionResult.value();
  const auto *typeAndShape = result.GetTypeAndShape();
  assert(typeAndShape && "no result type");
  if (typeAndShape->attrs().test(Attr::DeferredShape))
    return;
  for (auto extent : typeAndShape->shape())
    visitor(AsGenericExpr(extent.value()));
}

bool Fortran::lower::CallerInterface::mustMapInterfaceSymbols() const {
  assert(characteristic && "characteristic was not computed");
  const auto &result = characteristic->functionResult;
  if (!result || result->CanBeReturnedViaImplicitInterface())
    return false;
  bool allResultSpecExprConstant = true;
  auto visitor = [&](const Fortran::lower::SomeExpr &e) {
    allResultSpecExprConstant &= Fortran::evaluate::IsConstantExpr(e);
  };
  walkResultLengths(visitor);
  walkResultExtents(visitor);
  return !allResultSpecExprConstant;
}

mlir::Value Fortran::lower::CallerInterface::getArgumentValue(
    const semantics::Symbol &sym) const {
  auto loc = converter.genLocation();
  const auto *iface = procRef.proc().GetInterfaceSymbol();

  if (!iface)
    fir::emitFatalError(
        loc, "mapping actual and dummy arguments requires an interface");
  const auto &dummies = iface->get<semantics::SubprogramDetails>().dummyArgs();
  auto it = std::find(dummies.begin(), dummies.end(), &sym);
  if (it == dummies.end())
    fir::emitFatalError(loc, "symbol is not a dummy in this call");
  auto mlirArgIndex = passedArguments[it - dummies.begin()].firArgument;
  return actualInputs[mlirArgIndex];
}

mlir::Type Fortran::lower::CallerInterface::getResultStorageType() const {
  if (passedResult)
    return fir::dyn_cast_ptrEleTy(inputs[passedResult->firArgument].type);
  assert(saveResult && !outputs.empty());
  return outputs[0].type;
}

const Fortran::semantics::Symbol &
Fortran::lower::CallerInterface::getResultSymbol() const {
  auto loc = converter.genLocation();
  const auto *iface = procRef.proc().GetInterfaceSymbol();
  if (!iface)
    fir::emitFatalError(
        loc, "mapping actual and dummy arguments requires an interface");
  return iface->get<semantics::SubprogramDetails>().result();
}

//===----------------------------------------------------------------------===//
// Callee side interface implementation
//===----------------------------------------------------------------------===//

bool Fortran::lower::CalleeInterface::hasAlternateReturns() const {
  return !funit.isMainProgram() &&
         Fortran::semantics::HasAlternateReturns(funit.getSubprogramSymbol());
}

std::string Fortran::lower::CalleeInterface::getMangledName() const {
  if (funit.isMainProgram())
    return fir::NameUniquer::doProgramEntry().str();
  return ::getMangledName(funit.getSubprogramSymbol());
}

const Fortran::semantics::Symbol *
Fortran::lower::CalleeInterface::getProcedureSymbol() const {
  if (funit.isMainProgram())
    return nullptr;
  return &funit.getSubprogramSymbol();
}

mlir::Location Fortran::lower::CalleeInterface::getCalleeLocation() const {
  // FIXME: do NOT use unknown for the anonymous PROGRAM case. We probably
  // should just stash the location in the funit regardless.
  return converter.genLocation(funit.getStartingSourceLoc());
}

Fortran::evaluate::characteristics::Procedure
Fortran::lower::CalleeInterface::characterize() const {
  auto &foldingContext = converter.getFoldingContext();
  auto characteristic =
      Fortran::evaluate::characteristics::Procedure::Characterize(
          funit.getSubprogramSymbol(), foldingContext);
  assert(characteristic && "Fail to get characteristic from symbol");
  return *characteristic;
}

bool Fortran::lower::CalleeInterface::isMainProgram() const {
  return funit.isMainProgram();
}

mlir::FuncOp Fortran::lower::CalleeInterface::addEntryBlockAndMapArguments() {
  // On the callee side, directly map the mlir::value argument of
  // the function block to the Fortran symbols.
  func.addEntryBlock();
  mapPassedEntities();
  return func;
}

bool Fortran::lower::CalleeInterface::hasHostAssociated() const {
  return funit.parentHasHostAssoc();
}

mlir::Type Fortran::lower::CalleeInterface::getHostAssociatedTy() const {
  assert(hasHostAssociated());
  return funit.parentHostAssoc().getArgumentType(converter);
}

mlir::Value Fortran::lower::CalleeInterface::getHostAssociatedTuple() const {
  assert(hasHostAssociated() || !funit.getHostAssoc().empty());
  return converter.hostAssocTupleValue();
}

//===----------------------------------------------------------------------===//
// CallInterface implementation: this part is common to both caller and caller
// sides.
//===----------------------------------------------------------------------===//

static void addSymbolAttribute(mlir::FuncOp func,
                               const Fortran::semantics::Symbol &sym,
                               mlir::MLIRContext &mlirContext) {
  // Only add this on bind(C) functions for which the symbol is not reflected in
  // the current context.
  if (!Fortran::semantics::IsBindCProcedure(sym))
    return;
  auto name =
      Fortran::lower::mangle::mangleName(sym, /*keepExternalInScope=*/true);
  auto strAttr = mlir::StringAttr::get(&mlirContext, name);
  func->setAttr(fir::getSymbolAttrName(), strAttr);
}

/// Declare drives the different actions to be performed while analyzing the
/// signature and building/finding the mlir::FuncOp.
template <typename T>
void Fortran::lower::CallInterface<T>::declare() {
  if (!side().isMainProgram()) {
    characteristic =
        std::make_unique<Fortran::evaluate::characteristics::Procedure>(
            side().characterize());
    auto isImplicit = characteristic->CanBeCalledViaImplicitInterface();
    determineInterface(isImplicit, *characteristic);
  }
  // No input/output for main program

  // Create / get funcOp for direct calls. For indirect calls (only meaningful
  // on the caller side), no funcOp has to be created here. The mlir::Value
  // holding the indirection is used when creating the fir::CallOp.
  if (!side().isIndirectCall()) {
    auto name = side().getMangledName();
    auto module = converter.getModuleOp();
    func = fir::FirOpBuilder::getNamedFunction(module, name);
    if (!func) {
      mlir::Location loc = side().getCalleeLocation();
      mlir::FunctionType ty = genFunctionType();
      func = fir::FirOpBuilder::createFunction(loc, module, name, ty);
      if (const auto *sym = side().getProcedureSymbol())
        addSymbolAttribute(func, *sym, converter.getMLIRContext());
      for (const auto &placeHolder : llvm::enumerate(inputs))
        if (!placeHolder.value().attributes.empty())
          func.setArgAttrs(placeHolder.index(), placeHolder.value().attributes);
    }
  }
}

/// Once the signature has been analyzed and the mlir::FuncOp was built/found,
/// map the fir inputs to Fortran entities (the symbols or expressions).
template <typename T>
void Fortran::lower::CallInterface<T>::mapPassedEntities() {
  // map back fir inputs to passed entities
  if constexpr (std::is_same_v<T, Fortran::lower::CalleeInterface>) {
    assert(inputs.size() == func.front().getArguments().size() &&
           "function previously created with different number of arguments");
    for (auto [fst, snd] : llvm::zip(inputs, func.front().getArguments()))
      mapBackInputToPassedEntity(fst, snd);
  } else {
    // On the caller side, map the index of the mlir argument position
    // to Fortran ActualArguments.
    auto firPosition = 0;
    for (const auto &placeHolder : inputs)
      mapBackInputToPassedEntity(placeHolder, firPosition++);
  }
}

template <typename T>
void Fortran::lower::CallInterface<T>::mapBackInputToPassedEntity(
    const FirPlaceHolder &placeHolder, FirValue firValue) {
  auto &passedEntity =
      placeHolder.passedEntityPosition == FirPlaceHolder::resultEntityPosition
          ? passedResult.value()
          : passedArguments[placeHolder.passedEntityPosition];
  if (placeHolder.property == Property::CharLength)
    passedEntity.firLength = firValue;
  else
    passedEntity.firArgument = firValue;
}

/// Helpers to access ActualArgument/Symbols
static const Fortran::evaluate::ActualArguments &
getEntityContainer(const Fortran::evaluate::ProcedureRef &proc) {
  return proc.arguments();
}

static const std::vector<Fortran::semantics::Symbol *> &
getEntityContainer(Fortran::lower::pft::FunctionLikeUnit &funit) {
  return funit.getSubprogramSymbol()
      .get<Fortran::semantics::SubprogramDetails>()
      .dummyArgs();
}

static const Fortran::evaluate::ActualArgument *getDataObjectEntity(
    const std::optional<Fortran::evaluate::ActualArgument> &arg) {
  if (arg)
    return &*arg;
  return nullptr;
}

static const Fortran::semantics::Symbol &
getDataObjectEntity(const Fortran::semantics::Symbol *arg) {
  assert(arg && "expect symbol for data object entity");
  return *arg;
}

static const Fortran::evaluate::ActualArgument *
getResultEntity(const Fortran::evaluate::ProcedureRef &) {
  return nullptr;
}

static const Fortran::semantics::Symbol &
getResultEntity(Fortran::lower::pft::FunctionLikeUnit &funit) {
  const auto &details =
      funit.getSubprogramSymbol().get<Fortran::semantics::SubprogramDetails>();
  return details.result();
}

/// Bypass helpers to manipulate entities since they are not any symbol/actual
/// argument to associate. See SignatureBuilder below.
using FakeEntity = bool;
using FakeEntities = llvm::SmallVector<FakeEntity>;
static FakeEntities
getEntityContainer(const Fortran::evaluate::characteristics::Procedure &proc) {
  FakeEntities enities(proc.dummyArguments.size());
  return enities;
}
static const FakeEntity &getDataObjectEntity(const FakeEntity &e) { return e; }
static FakeEntity
getResultEntity(const Fortran::evaluate::characteristics::Procedure &proc) {
  return false;
}

/// This is the actual part that defines the FIR interface based on the
/// characteristic. It directly mutates the CallInterface members.
template <typename T>
class Fortran::lower::CallInterfaceImpl {
  using CallInterface = Fortran::lower::CallInterface<T>;
  using PassEntityBy = typename CallInterface::PassEntityBy;
  using PassedEntity = typename CallInterface::PassedEntity;
  using FirValue = typename CallInterface::FirValue;
  using FortranEntity = typename CallInterface::FortranEntity;
  using FirPlaceHolder = typename CallInterface::FirPlaceHolder;
  using Property = typename CallInterface::Property;
  using TypeAndShape = Fortran::evaluate::characteristics::TypeAndShape;
  using DummyCharacteristics =
      Fortran::evaluate::characteristics::DummyArgument;

public:
  CallInterfaceImpl(CallInterface &i)
      : interface(i), mlirContext{i.converter.getMLIRContext()} {}

  void buildImplicitInterface(
      const Fortran::evaluate::characteristics::Procedure &procedure) {
    // Handle result
    if (const auto &result = procedure.functionResult)
      handleImplicitResult(*result);
    else if (interface.side().hasAlternateReturns())
      addFirResult(mlir::IndexType::get(&mlirContext),
                   FirPlaceHolder::resultEntityPosition, Property::Value);
    // Handle arguments
    const auto &argumentEntities =
        getEntityContainer(interface.side().getCallDescription());
    for (auto pair : llvm::zip(procedure.dummyArguments, argumentEntities)) {
      const auto &argCharacteristics = std::get<0>(pair);
      std::visit(
          Fortran::common::visitors{
              [&](const auto &dummy) {
                const auto &entity = getDataObjectEntity(std::get<1>(pair));
                handleImplicitDummy(&argCharacteristics, dummy, entity);
              },
              [&](const Fortran::evaluate::characteristics::AlternateReturn &) {
                // nothing to do
              },
          },
          argCharacteristics.u);
    }
  }

  void buildExplicitInterface(
      const Fortran::evaluate::characteristics::Procedure &procedure) {
    // Handle result
    if (const auto &result = procedure.functionResult) {
      if (result->CanBeReturnedViaImplicitInterface())
        handleImplicitResult(*result);
      else
        handleExplicitResult(*result);
    } else if (interface.side().hasAlternateReturns()) {
      addFirResult(mlir::IndexType::get(&mlirContext),
                   FirPlaceHolder::resultEntityPosition, Property::Value);
    }
    // Handle arguments
    const auto &argumentEntities =
        getEntityContainer(interface.side().getCallDescription());
    for (auto pair : llvm::zip(procedure.dummyArguments, argumentEntities)) {
      const auto &argCharacteristics = std::get<0>(pair);
      std::visit(
          Fortran::common::visitors{
              [&](const Fortran::evaluate::characteristics::DummyDataObject
                      &dummy) {
                const auto &entity = getDataObjectEntity(std::get<1>(pair));
                if (dummy.CanBePassedViaImplicitInterface())
                  handleImplicitDummy(&argCharacteristics, dummy, entity);
                else
                  handleExplicitDummy(&argCharacteristics, dummy, entity);
              },
              [&](const Fortran::evaluate::characteristics::DummyProcedure
                      &dummy) {
                const auto &entity = getDataObjectEntity(std::get<1>(pair));
                handleImplicitDummy(&argCharacteristics, dummy, entity);
              },
              [&](const Fortran::evaluate::characteristics::AlternateReturn &) {
                // nothing to do
              },
          },
          argCharacteristics.u);
    }
  }

  void appendHostAssocTupleArg(mlir::Type tupTy) {
    auto *ctxt = tupTy.getContext();
    addFirOperand(tupTy, nextPassedArgPosition(), Property::BaseAddress,
                  {mlir::NamedAttribute{
                      mlir::Identifier::get(fir::getHostAssocAttrName(), ctxt),
                      mlir::UnitAttr::get(ctxt)}});
    interface.passedArguments.emplace_back(
        PassedEntity{PassEntityBy::BaseAddress, std::nullopt,
                     interface.side().getHostAssociatedTuple(), emptyValue()});
  }

private:
  void handleImplicitResult(
      const Fortran::evaluate::characteristics::FunctionResult &result) {
    if (result.IsProcedurePointer())
      TODO(interface.converter.genLocation(),
           "procedure pointer result not yet handled");
    const auto *typeAndShape = result.GetTypeAndShape();
    assert(typeAndShape && "expect type for non proc pointer result");
    auto dynamicType = typeAndShape->type();
    // Character result allocated by caller and passed as hidden arguments
    if (dynamicType.category() == Fortran::common::TypeCategory::Character) {
      handleImplicitCharacterResult(dynamicType);
    } else if (dynamicType.category() ==
               Fortran::common::TypeCategory::Derived) {
      // Derived result need to be allocated by the caller and the result value
      // must be saved. Derived type in implicit interface cannot have length
      // parameters.
      setSaveResult();
      auto mlirType = translateDynamicType(dynamicType);
      addFirResult(mlirType, FirPlaceHolder::resultEntityPosition,
                   Property::Value);
    } else {
      // All result other than characters/derived are simply returned by value
      // in implicit interfaces
      auto mlirType =
          getConverter().genType(dynamicType.category(), dynamicType.kind());
      addFirResult(mlirType, FirPlaceHolder::resultEntityPosition,
                   Property::Value);
    }
  }
  void
  handleImplicitCharacterResult(const Fortran::evaluate::DynamicType &type) {
    auto resultPosition = FirPlaceHolder::resultEntityPosition;
    setPassedResult(PassEntityBy::AddressAndLength,
                    getResultEntity(interface.side().getCallDescription()));
    auto lenTy = mlir::IndexType::get(&mlirContext);
    auto charRefTy = fir::ReferenceType::get(
        fir::CharacterType::getUnknownLen(&mlirContext, type.kind()));
    auto boxCharTy = fir::BoxCharType::get(&mlirContext, type.kind());
    addFirOperand(charRefTy, resultPosition, Property::CharAddress);
    addFirOperand(lenTy, resultPosition, Property::CharLength);
    /// For now, also return it by boxchar
    addFirResult(boxCharTy, resultPosition, Property::BoxChar);
  }

  void handleImplicitDummy(
      const DummyCharacteristics *characteristics,
      const Fortran::evaluate::characteristics::DummyDataObject &obj,
      const FortranEntity &entity) {
    auto dynamicType = obj.type.type();
    if (dynamicType.category() == Fortran::common::TypeCategory::Character) {
      auto boxCharTy = fir::BoxCharType::get(&mlirContext, dynamicType.kind());
      addFirOperand(boxCharTy, nextPassedArgPosition(), Property::BoxChar);
      addPassedArg(PassEntityBy::BoxChar, entity, characteristics);
    } else {
      // non-PDT derived type allowed in implicit interface.
      auto type = translateDynamicType(dynamicType);
      fir::SequenceType::Shape bounds = getBounds(obj.type.shape());
      if (!bounds.empty())
        type = fir::SequenceType::get(bounds, type);
      auto refType = fir::ReferenceType::get(type);

      addFirOperand(refType, nextPassedArgPosition(), Property::BaseAddress);
      addPassedArg(PassEntityBy::BaseAddress, entity, characteristics);
    }
  }

  // Define when an explicit argument must be passed in a fir.box.
  bool dummyRequiresBox(
      const Fortran::evaluate::characteristics::DummyDataObject &obj) {
    using ShapeAttr = Fortran::evaluate::characteristics::TypeAndShape::Attr;
    using ShapeAttrs = Fortran::evaluate::characteristics::TypeAndShape::Attrs;
    constexpr ShapeAttrs shapeRequiringBox = {
        ShapeAttr::AssumedShape, ShapeAttr::DeferredShape,
        ShapeAttr::AssumedRank, ShapeAttr::Coarray};
    if ((obj.type.attrs() & shapeRequiringBox).any())
      // Need to pass shape/coshape info in fir.box.
      return true;
    if (obj.type.type().IsPolymorphic())
      // Need to pass dynamic type info in fir.box.
      return true;
    if (const auto *derived =
            Fortran::evaluate::GetDerivedTypeSpec(obj.type.type()))
      // Need to pass type parameters in fir.box if any.
      return derived->parameters().empty();
    return false;
  }

  mlir::Type
  translateDynamicType(const Fortran::evaluate::DynamicType &dynamicType) {
    auto cat = dynamicType.category();
    // DERIVED
    if (cat == Fortran::common::TypeCategory::Derived)
      return getConverter().genType(dynamicType.GetDerivedTypeSpec());
    // CHARACTER with compile time constant length.
    if (cat == Fortran::common::TypeCategory::Character)
      if (auto constantLen = toInt64(dynamicType.GetCharLength()))
        return getConverter().genType(cat, dynamicType.kind(), {*constantLen});
    // INTEGER, REAL, LOGICAL, COMPLEX, and CHARACTER with dynamic length.
    return getConverter().genType(cat, dynamicType.kind());
  }

  void handleExplicitDummy(
      const DummyCharacteristics *characteristics,
      const Fortran::evaluate::characteristics::DummyDataObject &obj,
      const FortranEntity &entity) {
    using Attrs = Fortran::evaluate::characteristics::DummyDataObject::Attr;

    bool isValueAttr = false;
    [[maybe_unused]] auto loc = interface.converter.genLocation();
    llvm::SmallVector<mlir::NamedAttribute> attrs;
    auto addMLIRAttr = [&](llvm::StringRef attr) {
      attrs.emplace_back(mlir::Identifier::get(attr, &mlirContext),
                         UnitAttr::get(&mlirContext));
    };
    if (obj.attrs.test(Attrs::Optional))
      addMLIRAttr(fir::getOptionalAttrName());
    if (obj.attrs.test(Attrs::Asynchronous))
      TODO(loc, "Asynchronous in procedure interface");
    if (obj.attrs.test(Attrs::Contiguous))
      addMLIRAttr(fir::getContiguousAttrName());
    if (obj.attrs.test(Attrs::Value))
      isValueAttr = true; // TODO: do we want an mlir::Attribute as well?
    if (obj.attrs.test(Attrs::Volatile))
      TODO(loc, "Volatile in procedure interface");
    if (obj.attrs.test(Attrs::Target))
      addMLIRAttr(fir::getTargetAttrName());

    // TODO: intents that require special care (e.g finalization)

    using ShapeAttrs = Fortran::evaluate::characteristics::TypeAndShape::Attr;
    const auto &shapeAttrs = obj.type.attrs();
    if (shapeAttrs.test(ShapeAttrs::AssumedRank))
      TODO(loc, "Assumed Rank in procedure interface");
    if (shapeAttrs.test(ShapeAttrs::Coarray))
      TODO(loc, "Coarray in procedure interface");

    // So far assume that if the argument cannot be passed by implicit interface
    // it must be by box. That may no be always true (e.g for simple optionals)

    auto dynamicType = obj.type.type();
    auto type = translateDynamicType(dynamicType);
    fir::SequenceType::Shape bounds = getBounds(obj.type.shape());
    if (!bounds.empty())
      type = fir::SequenceType::get(bounds, type);
    if (obj.attrs.test(Attrs::Allocatable))
      type = fir::HeapType::get(type);
    if (obj.attrs.test(Attrs::Pointer))
      type = fir::PointerType::get(type);
    auto boxType = fir::BoxType::get(type);

    if (obj.attrs.test(Attrs::Allocatable) || obj.attrs.test(Attrs::Pointer)) {
      // Pass as fir.ref<fir.box>
      auto boxRefType = fir::ReferenceType::get(boxType);
      addFirOperand(boxRefType, nextPassedArgPosition(), Property::MutableBox,
                    attrs);
      addPassedArg(PassEntityBy::MutableBox, entity, characteristics);
    } else if (dummyRequiresBox(obj)) {
      // Pass as fir.box
      addFirOperand(boxType, nextPassedArgPosition(), Property::Box, attrs);
      addPassedArg(PassEntityBy::Box, entity, characteristics);
    } else if (dynamicType.category() ==
               Fortran::common::TypeCategory::Character) {
      // Pass as fir.box_char
      auto boxCharTy = fir::BoxCharType::get(&mlirContext, dynamicType.kind());
      addFirOperand(boxCharTy, nextPassedArgPosition(), Property::BoxChar,
                    attrs);
      addPassedArg(isValueAttr ? PassEntityBy::CharBoxValueAttribute
                               : PassEntityBy::BoxChar,
                   entity, characteristics);
    } else {
      // Pass as fir.ref
      auto refType = fir::ReferenceType::get(type);
      addFirOperand(refType, nextPassedArgPosition(), Property::BaseAddress,
                    attrs);
      addPassedArg(isValueAttr ? PassEntityBy::BaseAddressValueAttribute
                               : PassEntityBy::BaseAddress,
                   entity, characteristics);
    }
  }

  void handleImplicitDummy(
      const DummyCharacteristics *characteristics,
      const Fortran::evaluate::characteristics::DummyProcedure &proc,
      const FortranEntity &entity) {
    if (proc.attrs.test(
            Fortran::evaluate::characteristics::DummyProcedure::Attr::Pointer))
      llvm_unreachable("TODO: procedure pointer arguments");
    // Otherwise, it is a dummy procedure

    // TODO: Get actual function type of the dummy procedure, at least when an
    // interface is given.
    // In general, that is a nice to have but we cannot guarantee to find the
    // function type that will match the one of the calls, we may not even know
    // how many arguments the dummy procedure accepts (e.g. if a procedure
    // pointer is only transiting through the current procedure without being
    // called), so a function type cast must always be inserted.
    auto funcType =
        mlir::FunctionType::get(&mlirContext, llvm::None, llvm::None);
    addFirOperand(funcType, nextPassedArgPosition(), Property::BaseAddress);
    addPassedArg(PassEntityBy::BaseAddress, entity, characteristics);
  }

  void handleExplicitResult(
      const Fortran::evaluate::characteristics::FunctionResult &result) {
    using Attr = Fortran::evaluate::characteristics::FunctionResult::Attr;

    if (result.IsProcedurePointer())
      TODO(interface.converter.genLocation(),
           "procedure pointer result not yet handled");
    const auto *typeAndShape = result.GetTypeAndShape();
    assert(typeAndShape && "expect type for non proc pointer result");
    auto mlirType = translateDynamicType(typeAndShape->type());
    fir::SequenceType::Shape bounds = getBounds(typeAndShape->shape());
    if (!bounds.empty())
      mlirType = fir::SequenceType::get(bounds, mlirType);
    if (result.attrs.test(Attr::Allocatable))
      mlirType = fir::BoxType::get(fir::HeapType::get(mlirType));
    if (result.attrs.test(Attr::Pointer))
      mlirType = fir::BoxType::get(fir::PointerType::get(mlirType));

    if (fir::isa_char(mlirType)) {
      // Character scalar results must be passed as arguments in lowering so
      // that an assumed length character function callee can access the result
      // length. A function with a result requiring an explicit interface does
      // not have to be compatible with assumed length function, but most
      // compilers supports it.
      handleImplicitCharacterResult(typeAndShape->type());
      return;
    }

    addFirResult(mlirType, FirPlaceHolder::resultEntityPosition,
                 Property::Value);
    // Explicit results require the caller to allocate the storage and save the
    // function result in the storage with a fir.save_result.
    setSaveResult();
  }

  fir::SequenceType::Shape getBounds(const Fortran::evaluate::Shape &shape) {
    fir::SequenceType::Shape bounds;
    for (const auto &extent : shape) {
      auto bound = fir::SequenceType::getUnknownExtent();
      if (auto i = toInt64(extent))
        bound = *i;
      bounds.emplace_back(bound);
    }
    return bounds;
  }
  std::optional<std::int64_t>
  toInt64(std::optional<
          Fortran::evaluate::Expr<Fortran::evaluate::SubscriptInteger>>
              expr) {
    if (expr)
      return Fortran::evaluate::ToInt64(Fortran::evaluate::Fold(
          getConverter().getFoldingContext(), AsGenericExpr(*expr)));
    return std::nullopt;
  }
  void
  addFirOperand(mlir::Type type, int entityPosition, Property p,
                llvm::ArrayRef<mlir::NamedAttribute> attributes = llvm::None) {
    interface.inputs.emplace_back(
        FirPlaceHolder{type, entityPosition, p, attributes});
  }
  void
  addFirResult(mlir::Type type, int entityPosition, Property p,
               llvm::ArrayRef<mlir::NamedAttribute> attributes = llvm::None) {
    interface.outputs.emplace_back(
        FirPlaceHolder{type, entityPosition, p, attributes});
  }
  void addPassedArg(PassEntityBy p, FortranEntity entity,
                    const DummyCharacteristics *characteristics) {
    interface.passedArguments.emplace_back(
        PassedEntity{p, entity, emptyValue(), emptyValue(), characteristics});
  }
  void setPassedResult(PassEntityBy p, FortranEntity entity) {
    interface.passedResult =
        PassedEntity{p, entity, emptyValue(), emptyValue()};
  }
  void setSaveResult() { interface.saveResult = true; }
  int nextPassedArgPosition() { return interface.passedArguments.size(); }

  static FirValue emptyValue() {
    if constexpr (std::is_same_v<Fortran::lower::CalleeInterface, T>) {
      return {};
    } else {
      return -1;
    }
  }

  Fortran::lower::AbstractConverter &getConverter() {
    return interface.converter;
  }
  CallInterface &interface;
  mlir::MLIRContext &mlirContext;
};

template <typename T>
bool Fortran::lower::CallInterface<T>::PassedEntity::isOptional() const {
  if (!characteristics)
    return false;
  return characteristics->IsOptional();
}
template <typename T>
bool Fortran::lower::CallInterface<T>::PassedEntity::mayBeModifiedByCall()
    const {
  if (!characteristics)
    return true;
  return characteristics->GetIntent() != Fortran::common::Intent::In;
}
template <typename T>
bool Fortran::lower::CallInterface<T>::PassedEntity::mayBeReadByCall() const {
  if (!characteristics)
    return true;
  return characteristics->GetIntent() != Fortran::common::Intent::Out;
}

template <typename T>
void Fortran::lower::CallInterface<T>::determineInterface(
    bool isImplicit,
    const Fortran::evaluate::characteristics::Procedure &procedure) {
  CallInterfaceImpl<T> impl(*this);
  if (isImplicit)
    impl.buildImplicitInterface(procedure);
  else
    impl.buildExplicitInterface(procedure);
  // We only expect the extra host asspciations argument from the callee side as
  // the definition of internal procedures will be present, and we'll always
  // have a FuncOp definition in the ModuleOp, when lowering.
  if constexpr (std::is_same_v<T, Fortran::lower::CalleeInterface>) {
    if (side().hasHostAssociated())
      impl.appendHostAssocTupleArg(side().getHostAssociatedTy());
  }
}

template <typename T>
mlir::FunctionType Fortran::lower::CallInterface<T>::genFunctionType() {
  llvm::SmallVector<mlir::Type> returnTys;
  llvm::SmallVector<mlir::Type> inputTys;
  for (const auto &placeHolder : outputs)
    returnTys.emplace_back(placeHolder.type);
  for (const auto &placeHolder : inputs)
    inputTys.emplace_back(placeHolder.type);
  return mlir::FunctionType::get(&converter.getMLIRContext(), inputTys,
                                 returnTys);
}

template <typename T>
llvm::SmallVector<mlir::Type>
Fortran::lower::CallInterface<T>::getResultType() const {
  llvm::SmallVector<mlir::Type> types;
  for (const auto &out : outputs)
    types.emplace_back(out.type);
  return types;
}

template class Fortran::lower::CallInterface<Fortran::lower::CalleeInterface>;
template class Fortran::lower::CallInterface<Fortran::lower::CallerInterface>;

//===----------------------------------------------------------------------===//
// Function Type Translation
//===----------------------------------------------------------------------===//

/// Build signature from characteristics when there is no Fortran entity to
/// associate with the arguments (i.e, this is not a call site or a procedure
/// declaration. This is needed when dealing with function pointers/dummy
/// arguments.

class SignatureBuilder;
template <>
struct Fortran::lower::PassedEntityTypes<SignatureBuilder> {
  using FortranEntity = FakeEntity;
  using FirValue = int;
};

/// SignatureBuilder is a CRTP implementation of CallInterface intended to
/// help translating characteristics::Procedure to mlir::FunctionType using
/// the CallInterface translation.
class SignatureBuilder
    : public Fortran::lower::CallInterface<SignatureBuilder> {
public:
  SignatureBuilder(const Fortran::evaluate::characteristics::Procedure &p,
                   Fortran::lower::AbstractConverter &c, bool forceImplicit)
      : CallInterface{c}, proc{p} {
    auto isImplicit = forceImplicit || proc.CanBeCalledViaImplicitInterface();
    determineInterface(isImplicit, proc);
  }
  /// Does the procedure characteristics being translated have alternate
  /// returns ?
  bool hasAlternateReturns() const {
    for (const auto &dummy : proc.dummyArguments)
      if (std::holds_alternative<
              Fortran::evaluate::characteristics::AlternateReturn>(dummy.u))
        return true;
    return false;
  };

  /// This is only here to fulfill CRTP dependencies and should not be called.
  std::string getMangledName() const {
    llvm_unreachable("trying to get name from SignatureBuilder");
  }

  /// This is only here to fulfill CRTP dependencies and should not be called.
  mlir::Location getCalleeLocation() const {
    llvm_unreachable("trying to get callee location from SignatureBuilder");
  }

  /// This is only here to fulfill CRTP dependencies and should not be called.
  const Fortran::semantics::Symbol *getProcedureSymbol() const {
    llvm_unreachable("trying to get callee symbol from SignatureBuilder");
  };

  Fortran::evaluate::characteristics::Procedure characterize() const {
    return proc;
  }
  /// SignatureBuilder cannot be used on main program.
  static constexpr bool isMainProgram() { return false; }

  /// Return the characteristics::Procedure that is being translated to
  /// mlir::FunctionType.
  const Fortran::evaluate::characteristics::Procedure &
  getCallDescription() const {
    return proc;
  }

  /// This is not the description of an indirect call.
  static constexpr bool isIndirectCall() { return false; }

  /// Return the translated signature.
  mlir::FunctionType getFunctionType() { return genFunctionType(); }

  // Copy of base implementation.
  static constexpr bool hasHostAssociated() { return false; }
  mlir::Type getHostAssociatedTy() const {
    llvm_unreachable("getting host associated type in SignatureBuilder");
  }

private:
  const Fortran::evaluate::characteristics::Procedure &proc;
};

mlir::FunctionType Fortran::lower::translateSignature(
    const Fortran::evaluate::ProcedureDesignator &proc,
    Fortran::lower::AbstractConverter &converter) {
  auto characteristics =
      Fortran::evaluate::characteristics::Procedure::Characterize(
          proc, converter.getFoldingContext());
  // Most unrestricted intrinsic characteristic has the Elemental attribute
  // which triggers CanBeCalledViaImplicitInterface to return false. However,
  // using implicit interface rules is just fine here.
  bool forceImplicit = proc.GetSpecificIntrinsic();
  return SignatureBuilder{characteristics.value(), converter, forceImplicit}
      .getFunctionType();
}

mlir::FuncOp Fortran::lower::getOrDeclareFunction(
    llvm::StringRef name, const Fortran::evaluate::ProcedureDesignator &proc,
    Fortran::lower::AbstractConverter &converter) {
  auto module = converter.getModuleOp();
  mlir::FuncOp func = fir::FirOpBuilder::getNamedFunction(module, name);
  if (func)
    return func;

  const auto *symbol = proc.GetSymbol();
  assert(symbol && "non user function in getOrDeclareFunction");
  // getOrDeclareFunction is only used for functions not defined in the current
  // program unit, so use the location of the procedure designator symbol, which
  // is the first occurrence of the procedure in the program unit.
  auto loc = converter.genLocation(symbol->name());
  auto characteristics =
      Fortran::evaluate::characteristics::Procedure::Characterize(
          proc, converter.getFoldingContext());
  auto ty = SignatureBuilder{characteristics.value(), converter,
                             /*forceImplicit=*/false}
                .getFunctionType();
  auto newFunc = fir::FirOpBuilder::createFunction(loc, module, name, ty);
  addSymbolAttribute(newFunc, *symbol, converter.getMLIRContext());
  return newFunc;
}
