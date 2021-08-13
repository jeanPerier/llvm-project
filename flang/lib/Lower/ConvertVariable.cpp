//===-- ConvertVariable.cpp -- bridge to lower to MLIR --------------------===//
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

#include "ConvertVariable.h"
#include "BoxAnalyzer.h"
#include "StatementContext.h"
#include "SymbolMap.h"
#include "flang/Lower/AbstractConverter.h"
#include "flang/Lower/Allocatable.h"
#include "flang/Lower/CallInterface.h"
#include "flang/Lower/CharacterExpr.h"
#include "flang/Lower/ConvertExpr.h"
#include "flang/Lower/DerivedRuntime.h"
#include "flang/Lower/FIRBuilder.h"
#include "flang/Lower/Mangler.h"
#include "flang/Lower/PFTBuilder.h"
#include "flang/Lower/Support/Utils.h"
#include "flang/Lower/Todo.h"
#include "flang/Optimizer/Dialect/FIRAttr.h"
#include "flang/Optimizer/Dialect/FIRDialect.h"
#include "flang/Optimizer/Dialect/FIROps.h"
#include "flang/Optimizer/Support/FIRContext.h"
#include "flang/Optimizer/Support/FatalError.h"
#include "flang/Semantics/tools.h"
#include "llvm/Support/Debug.h"

#define DEBUG_TYPE "flang-lower-variable"

/// Helper to retrieve a copy of a character literal string from a SomeExpr.
/// Required to build character global initializers.
template <int KIND>
static llvm::Optional<std::tuple<std::string, std::size_t>>
getCharacterLiteralCopy(
    const Fortran::evaluate::Expr<
        Fortran::evaluate::Type<Fortran::common::TypeCategory::Character, KIND>>
        &x) {
  if (const auto *con =
          Fortran::evaluate::UnwrapConstantValue<Fortran::evaluate::Type<
              Fortran::common::TypeCategory::Character, KIND>>(x))
    if (auto val = con->GetScalarValue())
      return std::tuple<std::string, std::size_t>{
          std::string{(const char *)val->c_str(),
                      KIND * (std::size_t)con->LEN()},
          (std::size_t)con->LEN()};
  return llvm::None;
}
static llvm::Optional<std::tuple<std::string, std::size_t>>
getCharacterLiteralCopy(
    const Fortran::evaluate::Expr<Fortran::evaluate::SomeCharacter> &x) {
  return std::visit([](const auto &e) { return getCharacterLiteralCopy(e); },
                    x.u);
}
static llvm::Optional<std::tuple<std::string, std::size_t>>
getCharacterLiteralCopy(
    const Fortran::evaluate::Expr<Fortran::evaluate::SomeType> &x) {
  if (const auto *e = Fortran::evaluate::UnwrapExpr<
          Fortran::evaluate::Expr<Fortran::evaluate::SomeCharacter>>(x))
    return getCharacterLiteralCopy(*e);
  return llvm::None;
}
template <typename A>
static llvm::Optional<std::tuple<std::string, std::size_t>>
getCharacterLiteralCopy(const std::optional<A> &x) {
  if (x)
    return getCharacterLiteralCopy(*x);
  return llvm::None;
}

/// Helper to lower a scalar expression using a specific symbol mapping.
static mlir::Value genScalarValue(Fortran::lower::AbstractConverter &converter,
                                  mlir::Location loc,
                                  const Fortran::lower::SomeExpr &expr,
                                  Fortran::lower::SymMap &symMap,
                                  Fortran::lower::StatementContext &context) {
  // This does not use the AbstractConverter member function to override the
  // symbol mapping to be used expression lowering.
  return fir::getBase(Fortran::lower::createSomeExtendedExpression(
      loc, converter, expr, symMap, context));
}

/// Does this variable has a default initialization ?
static bool hasDefaultInitialization(const Fortran::semantics::Symbol &sym) {
  if (sym.has<Fortran::semantics::ObjectEntityDetails>())
    if (!Fortran::semantics::IsAllocatableOrPointer(sym))
      if (const auto *declTypeSpec = sym.GetType())
        if (const auto *derivedTypeSpec = declTypeSpec->AsDerived())
          return derivedTypeSpec->HasDefaultInitialization();
  return false;
}

//===----------------------------------------------------------------===//
// Global variables instantiation (not for alias and common)
//===----------------------------------------------------------------===//

/// Helper to generate expression value inside global initializer.
static fir::ExtendedValue
genInitializerExprValue(Fortran::lower::AbstractConverter &converter,
                        mlir::Location loc,
                        const Fortran::lower::SomeExpr &expr,
                        Fortran::lower::StatementContext &stmtCtx) {
  // Data initializer are constant value and should not depend on other symbols
  // given the front-end fold parameter references. In any case, the "current"
  // map of the converter should not be used since it holds mapping to
  // mlir::Value from another mlir region. If these value are used by accident
  // in the initializer, this will lead to segfaults in mlir code.
  Fortran::lower::SymMap emptyMap;
  return Fortran::lower::createSomeInitializerExpression(loc, converter, expr,
                                                         emptyMap, stmtCtx);
}

/// Create the global op declaration without any initializer
static fir::GlobalOp declareGlobal(Fortran::lower::AbstractConverter &converter,
                                   const Fortran::lower::pft::Variable &var,
                                   llvm::StringRef globalName,
                                   mlir::StringAttr linkage) {
  auto &builder = converter.getFirOpBuilder();
  const auto &sym = var.getSymbol();
  auto loc = converter.genLocation(sym.name());
  // Resolve potential host and module association before checking that this
  // symbol is an object of a function pointer.
  const auto &ultimate = sym.GetUltimate();
  if (!ultimate.has<Fortran::semantics::ObjectEntityDetails>() &&
      !ultimate.has<Fortran::semantics::ProcEntityDetails>())
    mlir::emitError(loc, "lowering global declaration: symbol '")
        << toStringRef(sym.name()) << "' has unexpected details\n";
  return builder.createGlobal(loc, converter.genType(var), globalName, linkage);
}

/// Temporary helper to catch todos in initial data target lowering.
static bool
hasDerivedTypeWithLengthParameters(const Fortran::semantics::Symbol &sym) {
  if (const auto *declTy = sym.GetType())
    if (const auto *derived = declTy->AsDerived())
      return Fortran::semantics::CountLenParameters(*derived) > 0;
  return false;
}

static mlir::Type unwrapElementType(mlir::Type type) {
  if (auto ty = fir::dyn_cast_ptrOrBoxEleTy(type))
    type = ty;
  if (auto seqType = type.dyn_cast<fir::SequenceType>())
    type = seqType.getEleTy();
  return type;
}

fir::ExtendedValue Fortran::lower::genExtAddrInInitializer(
    Fortran::lower::AbstractConverter &converter, mlir::Location loc,
    const Fortran::lower::SomeExpr &addr) {
  Fortran::lower::SymMap globalOpSymMap;
  Fortran::lower::AggregateStoreMap storeMap;
  Fortran::lower::StatementContext stmtCtx;
  if (const auto *sym = Fortran::evaluate::GetFirstSymbol(addr)) {
    // Length parameters processing will need care in global initializer
    // context.
    if (hasDerivedTypeWithLengthParameters(*sym))
      TODO(loc, "initial-data-target with derived type length parameters");

    auto var = Fortran::lower::pft::Variable(*sym, /*global=*/true);
    Fortran::lower::instantiateVariable(converter, var, globalOpSymMap,
                                        storeMap);
  }
  return Fortran::lower::createSomeExtendedAddress(loc, converter, addr,
                                                   globalOpSymMap, stmtCtx);
}

/// create initial-data-target fir.box in a global initializer region.
mlir::Value Fortran::lower::genInitialDataTarget(
    Fortran::lower::AbstractConverter &converter, mlir::Location loc,
    mlir::Type boxType, const Fortran::lower::SomeExpr &initialTarget) {
  Fortran::lower::SymMap globalOpSymMap;
  Fortran::lower::AggregateStoreMap storeMap;
  Fortran::lower::StatementContext stmtCtx;
  auto &builder = converter.getFirOpBuilder();
  if (Fortran::evaluate::UnwrapExpr<Fortran::evaluate::NullPointer>(
          initialTarget))
    return fir::factory::createUnallocatedBox(
        builder, loc, boxType, /*nonDeferredParams=*/llvm::None);
  // Pointer initial data target, and NULL(mold).
  if (const auto *sym = Fortran::evaluate::GetFirstSymbol(initialTarget)) {
    // Length parameters processing will need care in global initializer
    // context.
    if (hasDerivedTypeWithLengthParameters(*sym))
      TODO(loc, "initial-data-target with derived type length parameters");

    auto var = Fortran::lower::pft::Variable(*sym, /*global=*/true);
    Fortran::lower::instantiateVariable(converter, var, globalOpSymMap,
                                        storeMap);
  }
  mlir::Value box;
  if (initialTarget.Rank() > 0) {
    box = fir::getBase(Fortran::lower::createSomeArrayBox(
        converter, initialTarget, globalOpSymMap, stmtCtx));
  } else {
    auto addr = Fortran::lower::createSomeExtendedAddress(
        loc, converter, initialTarget, globalOpSymMap, stmtCtx);
    box = builder.createBox(loc, addr);
  }
  // box is a fir.box<T>, not a fir.box<fir.ptr<T>> as it should to be used
  // for pointers. A fir.convert should not be used here, because it would
  // not actually set the pointer attribute in the descriptor.
  // In a normal context, fir.rebox would be used to set the pointer attribute
  // while copying the projection from another fir.box. But fir.rebox cannot be
  // used in initializer because its current codegen expects that the input
  // fir.box is in memory, which is not the case in initializers.
  // So, just replace the fir.embox that created addr with one with
  // fir.box<fir.ptr<T>> result type.
  // Note that the descriptor cannot have been created with fir.rebox because
  // the initial-data-target cannot be a fir.box itself (it cannot be
  // assumed-shape, deferred-shape, or polymorphic as per C765). However the
  // case where the initial data target is a derived type with length parameters
  // will most likely be a bit trickier, hence the TODO above.

  auto *op = box.getDefiningOp();
  if (!op || !mlir::isa<fir::EmboxOp>(*op))
    fir::emitFatalError(
        loc, "fir.box must be created with embox in global initializers");
  auto targetEleTy = unwrapElementType(box.getType());
  if (!fir::isa_char(targetEleTy))
    return builder.create<fir::EmboxOp>(loc, boxType, op->getOperands(),
                                        op->getAttrs());

  // Handle the character case length particularities: embox takes a length
  // value argument when the result type has unknown length, but not when the
  // result type has constant length. The type of the initial target must be
  // constant length, but the one of the pointer may not be. In this case, a
  // length operand must be added.
  auto targetLen = targetEleTy.cast<fir::CharacterType>().getLen();
  auto ptrLen = unwrapElementType(boxType).cast<fir::CharacterType>().getLen();
  if (ptrLen == targetLen)
    // Nothing to do
    return builder.create<fir::EmboxOp>(loc, boxType, op->getOperands(),
                                        op->getAttrs());
  auto embox = mlir::cast<fir::EmboxOp>(*op);
  auto ptrType = boxType.cast<fir::BoxType>().getEleTy();
  auto memref = builder.createConvert(loc, ptrType, embox.memref());
  if (targetLen == fir::CharacterType::unknownLen())
    // Drop the length argument.
    return builder.create<fir::EmboxOp>(loc, boxType, memref, embox.shape(),
                                        embox.slice());
  // targetLen is constant and ptrLen is unknown. Add a length argument.
  auto targetLenValue =
      builder.createIntegerConstant(loc, builder.getIndexType(), targetLen);
  return builder.create<fir::EmboxOp>(loc, boxType, memref, embox.shape(),
                                      embox.slice(),
                                      mlir::ValueRange{targetLenValue});
}

static mlir::Value genDefaultInitializerValue(
    Fortran::lower::AbstractConverter &converter, mlir::Location loc,
    const Fortran::semantics::Symbol &sym, mlir::Type symTy,
    Fortran::lower::StatementContext &stmtCtx) {
  auto &builder = converter.getFirOpBuilder();
  auto scalarType = symTy;
  fir::SequenceType sequenceType;
  if (auto ty = symTy.dyn_cast<fir::SequenceType>()) {
    sequenceType = ty;
    scalarType = ty.getEleTy();
  }
  // Build a scalar default value of the symbol type, looping through the
  // components to build each component initial value.
  auto recTy = scalarType.cast<fir::RecordType>();
  auto fieldTy = fir::FieldType::get(scalarType.getContext());
  mlir::Value initialValue = builder.create<fir::UndefOp>(loc, scalarType);
  const auto *declTy = sym.GetType();
  assert(declTy && "var with default initialization must have a type");
  Fortran::semantics::OrderedComponentIterator components(
      declTy->derivedTypeSpec());
  for (const auto &component : components) {
    // Skip parent components, the sub-components of parent types are part of
    // components and will be looped through right after.
    if (component.test(Fortran::semantics::Symbol::Flag::ParentComp))
      continue;
    mlir::Value componentValue;
    auto name = toStringRef(component.name());
    auto componentTy = recTy.getType(name);
    assert(componentTy && "component not found in type");
    if (const auto *object{
            component.detailsIf<Fortran::semantics::ObjectEntityDetails>()}) {
      if (const auto &init = object->init()) {
        // Component has explicit initialization.
        if (Fortran::semantics::IsPointer(component))
          // Initial data target.
          componentValue =
              genInitialDataTarget(converter, loc, componentTy, *init);
        else
          // Initial value.
          componentValue = fir::getBase(
              genInitializerExprValue(converter, loc, *init, stmtCtx));
      } else if (Fortran::semantics::IsAllocatableOrPointer(component)) {
        // Pointer or allocatable without initialization.
        // Create deallocated/disassociated value.
        // From a standard point of view, pointer without initialization do not
        // need to be disassociated, but for sanity and simplicity, do it in
        // global constructor since this has no runtime cost.
        componentValue = fir::factory::createUnallocatedBox(
            builder, loc, componentTy, llvm::None);
      } else if (hasDefaultInitialization(component)) {
        // Component type has default initialization.
        componentValue = genDefaultInitializerValue(converter, loc, component,
                                                    componentTy, stmtCtx);
      } else {
        // Component has no initial value.
        componentValue = builder.create<fir::UndefOp>(loc, componentTy);
      }
    } else if (const auto *proc{
                   component
                       .detailsIf<Fortran::semantics::ProcEntityDetails>()}) {
      if (proc->init().has_value())
        TODO(loc, "procedure pointer component default initialization");
      else
        componentValue = builder.create<fir::UndefOp>(loc, componentTy);
    }
    assert(componentValue && "must have been computed");
    componentValue = builder.createConvert(loc, componentTy, componentValue);
    // FIXME: type parameters must come from the derived-type-spec
    mlir::Value field = builder.create<fir::FieldIndexOp>(
        loc, fieldTy, name, scalarType,
        /*typeParams=*/mlir::ValueRange{} /*TODO*/);
    initialValue = builder.create<fir::InsertValueOp>(loc, recTy, initialValue,
                                                      componentValue, field);
  }

  if (sequenceType) {
    // For arrays, duplicate the scalar value to all elements with an
    // fir.insert_range covering the whole array.
    auto arrayInitialValue = builder.create<fir::UndefOp>(loc, sequenceType);
    llvm::SmallVector<mlir::Value> rangeBounds;
    auto idxTy = builder.getIndexType();
    auto zero = builder.createIntegerConstant(loc, idxTy, 0);
    for (auto extent : sequenceType.getShape()) {
      if (extent == fir::SequenceType::getUnknownExtent())
        TODO(loc,
             "default initial value of array component with length parameters");
      rangeBounds.push_back(zero);
      rangeBounds.push_back(
          builder.createIntegerConstant(loc, idxTy, extent - 1));
    }
    return builder.create<fir::InsertOnRangeOp>(
        loc, sequenceType, arrayInitialValue, initialValue, rangeBounds);
  }
  return initialValue;
}

/// Create the global op and its init if it has one
static fir::GlobalOp defineGlobal(Fortran::lower::AbstractConverter &converter,
                                  const Fortran::lower::pft::Variable &var,
                                  llvm::StringRef globalName,
                                  mlir::StringAttr linkage) {
  auto &builder = converter.getFirOpBuilder();
  const auto &sym = var.getSymbol();
  auto loc = converter.genLocation(sym.name());
  bool isConst = sym.attrs().test(Fortran::semantics::Attr::PARAMETER);
  fir::GlobalOp global;
  if (Fortran::semantics::IsAllocatableOrPointer(sym)) {
    auto symTy = converter.genType(var);
    const auto *details =
        sym.detailsIf<Fortran::semantics::ObjectEntityDetails>();
    if (details && details->init()) {
      auto expr = *details->init();
      auto init = [&](Fortran::lower::FirOpBuilder &b) {
        auto box =
            Fortran::lower::genInitialDataTarget(converter, loc, symTy, expr);
        b.create<fir::HasValueOp>(loc, box);
      };
      global =
          builder.createGlobal(loc, symTy, globalName, isConst, init, linkage);
    } else {
      // Create unallocated/disassociated descriptor if no explicit init
      auto init = [&](Fortran::lower::FirOpBuilder &b) {
        auto box =
            fir::factory::createUnallocatedBox(b, loc, symTy, llvm::None);
        b.create<fir::HasValueOp>(loc, box);
      };
      global =
          builder.createGlobal(loc, symTy, globalName, isConst, init, linkage);
    }

  } else if (const auto *details =
                 sym.detailsIf<Fortran::semantics::ObjectEntityDetails>()) {
    if (details->init()) {
      auto symTy = converter.genType(var);
      if (fir::isa_char(symTy)) {
        // CHARACTER literal
        if (auto chLit = getCharacterLiteralCopy(details->init().value())) {
          auto init = builder.getStringAttr(std::get<std::string>(*chLit));
          global = builder.createGlobal(loc, symTy, globalName, linkage, init,
                                        isConst);
        } else {
          fir::emitFatalError(loc, "CHARACTER has unexpected initial value");
        }
      } else {
        global = builder.createGlobal(
            loc, symTy, globalName, isConst,
            [&](Fortran::lower::FirOpBuilder &builder) {
              Fortran::lower::StatementContext stmtCtx(/*prohibited=*/true);
              auto initVal = genInitializerExprValue(
                  converter, loc, details->init().value(), stmtCtx);
              auto castTo =
                  builder.createConvert(loc, symTy, fir::getBase(initVal));
              stmtCtx.finalize();
              builder.create<fir::HasValueOp>(loc, castTo);
            },
            linkage);
      }
    } else if (hasDefaultInitialization(sym)) {
      auto symTy = converter.genType(var);
      global = builder.createGlobal(
          loc, symTy, globalName, isConst,
          [&](Fortran::lower::FirOpBuilder &builder) {
            Fortran::lower::StatementContext stmtCtx(/*prohibited=*/true);
            auto initVal =
                genDefaultInitializerValue(converter, loc, sym, symTy, stmtCtx);
            auto castTo =
                builder.createConvert(loc, symTy, fir::getBase(initVal));
            stmtCtx.finalize();
            builder.create<fir::HasValueOp>(loc, castTo);
          },
          linkage);
    }
  } else if (sym.has<Fortran::semantics::CommonBlockDetails>()) {
    mlir::emitError(loc, "COMMON symbol processed elsewhere");
  } else {
    TODO(loc, "global"); // Procedure pointer or something else
  }
  // Creates undefined initializer for globals without initialziers
  if (!global) {
    auto symTy = converter.genType(var);
    global = builder.createGlobal(
        loc, symTy, globalName, isConst,
        [&](Fortran::lower::FirOpBuilder &builder) {
          builder.create<fir::HasValueOp>(
              loc, builder.create<fir::UndefOp>(loc, symTy));
        },
        linkage);
  }
  // Set public visibility to prevent global definition to be optimized out
  // even if they have no initializer and are unused in this compilation unit.
  global.setVisibility(mlir::SymbolTable::Visibility::Public);
  return global;
}

/// Instantiate a global variable. If it hasn't already been processed, add
/// the global to the ModuleOp as a new uniqued symbol and initialize it with
/// the correct value. It will be referenced on demand using `fir.addr_of`.
static void instantiateGlobal(Fortran::lower::AbstractConverter &converter,
                              const Fortran::lower::pft::Variable &var,
                              Fortran::lower::SymMap &symMap) {
  const auto &sym = var.getSymbol();
  assert(!var.isAlias() && "must be handled in instantiateAlias");
  auto &builder = converter.getFirOpBuilder();
  auto globalName = Fortran::lower::mangle::mangleName(sym);
  auto loc = converter.genLocation(sym.name());
  fir::GlobalOp global = builder.getNamedGlobal(globalName);
  if (!global) {
    if (var.isDeclaration()) {
      // Using a global from a module not defined in this compilation unit.
      mlir::StringAttr externalLinkage;
      global = declareGlobal(converter, var, globalName, externalLinkage);
    } else {
      // Module and common globals are defined elsewhere.
      // The only globals defined here are the globals owned by procedures
      // and they do not need to be visible in other compilation unit.
      auto internalLinkage = builder.createInternalLinkage();
      global = defineGlobal(converter, var, globalName, internalLinkage);
    }
  }
  auto addrOf = builder.create<fir::AddrOfOp>(loc, global.resultType(),
                                              global.getSymbol());
  Fortran::lower::StatementContext stmtCtx;
  mapSymbolAttributes(converter, var, symMap, stmtCtx, addrOf);
}

//===----------------------------------------------------------------===//
// Local variables instantiation (not for alias)
//===----------------------------------------------------------------===//

/// Create a stack slot for a local variable. Precondition: the insertion
/// point of the builder must be in the entry block, which is currently being
/// constructed.
static mlir::Value createNewLocal(Fortran::lower::AbstractConverter &converter,
                                  mlir::Location loc,
                                  const Fortran::lower::pft::Variable &var,
                                  mlir::Value preAlloc,
                                  llvm::ArrayRef<mlir::Value> shape = {},
                                  llvm::ArrayRef<mlir::Value> lenParams = {}) {
  if (preAlloc)
    return preAlloc;
  auto &builder = converter.getFirOpBuilder();
  auto nm = Fortran::lower::mangle::mangleName(var.getSymbol());
  auto ty = converter.genType(var);
  const auto &ultimateSymbol = var.getSymbol().GetUltimate();
  auto symNm = toStringRef(ultimateSymbol.name());
  auto isTarg = var.isTarget();
  // Let the builder do all the heavy lifting.
  return builder.allocateLocal(loc, ty, nm, symNm, shape, lenParams, isTarg);
}

/// Must \p var be default initialized at runtime when entering its scope.
static bool
mustBeDefaultInitializedAtRuntime(const Fortran::lower::pft::Variable &var) {
  if (!var.hasSymbol())
    return false;
  const auto &sym = var.getSymbol();
  if (var.isGlobal())
    // Global variables are statically initialized.
    return false;
  if (Fortran::semantics::IsDummy(sym) && !Fortran::semantics::IsIntentOut(sym))
    return false;
  // Local variables (including function results), and intent(out) dummies must
  // be default initialized at runtime if their type has default initialization.
  return hasDefaultInitialization(sym);
}

/// Call default initialization runtime routine to initialize \p var.
static void
defaultInitializeAtRuntime(Fortran::lower::AbstractConverter &converter,
                           const Fortran::lower::pft::Variable &var,
                           Fortran::lower::SymMap &symMap) {
  auto &builder = converter.getFirOpBuilder();
  auto loc = converter.getCurrentLocation();
  const auto &sym = var.getSymbol();
  auto exv = symMap.lookupSymbol(sym).toExtendedValue();
  if (Fortran::semantics::IsOptional(sym)) {
    // 15.5.2.12 point 3, absent optional dummies are not initialized.
    // Creating descriptor/passing null descriptor to the runtime would
    // create runtime crashes.
    auto isPresent = builder.create<fir::IsPresentOp>(loc, builder.getI1Type(),
                                                      fir::getBase(exv));
    builder.genIfThen(loc, isPresent)
        .genThen([&]() {
          auto box = builder.createBox(loc, exv);
          Fortran::lower::genDerivedTypeInitialize(builder, loc, box);
        })
        .end();
  } else {
    auto box = builder.createBox(loc, exv);
    Fortran::lower::genDerivedTypeInitialize(builder, loc, box);
  }
}

/// Instantiate a local variable. Precondition: Each variable will be visited
/// such that if its properties depend on other variables, the variables upon
/// which its properties depend will already have been visited.
static void instantiateLocal(Fortran::lower::AbstractConverter &converter,
                             const Fortran::lower::pft::Variable &var,
                             Fortran::lower::SymMap &symMap) {
  assert(!var.isAlias());
  Fortran::lower::StatementContext stmtCtx;
  mapSymbolAttributes(converter, var, symMap, stmtCtx);
  if (mustBeDefaultInitializedAtRuntime(var))
    defaultInitializeAtRuntime(converter, var, symMap);
}

//===----------------------------------------------------------------===//
// Aliased (EQUIVALENCE) variables instantiation
//===----------------------------------------------------------------===//

/// Insert \p aggregateStore instance into an AggregateStoreMap.
static void insertAggregateStore(Fortran::lower::AggregateStoreMap &storeMap,
                                 const Fortran::lower::pft::Variable &var,
                                 mlir::Value aggregateStore) {
  auto off = var.getAggregateStore().getOffset();
  Fortran::lower::AggregateStoreKey key = {var.getOwningScope(), off};
  storeMap[key] = aggregateStore;
}

/// Retrieve the aggregate store instance of \p alias from an
/// AggregateStoreMap.
static mlir::Value
getAggregateStore(Fortran::lower::AggregateStoreMap &storeMap,
                  const Fortran::lower::pft::Variable &alias) {
  Fortran::lower::AggregateStoreKey key = {alias.getOwningScope(),
                                           alias.getAlias()};
  auto iter = storeMap.find(key);
  assert(iter != storeMap.end());
  return iter->second;
}

/// Build the name for the storage of a global equivalence.
static std::string mangleGlobalAggregateStore(
    const Fortran::lower::pft::Variable::AggregateStore &st) {
  assert(st.isGlobal() && "cannot name local aggregate");
  return Fortran::lower::mangle::mangleName(*st.vars[0]);
}

// In equivalences, 19.5.3.4 point 10 gives rules regarding explicit and default
// initialization interaction. The interpretation of this point is not
// ubiquitous among compilers, but the majority (nag, ifort, nvfortran) accept
// full or partial overlap of explicit init over default init. In case of
// partial overlap, these compiler still apply the default init for the
// components for which the storage is not explicitly initialized.
//
// In the example below, the equivalence storage must be initialized to [3 , 2].
// 3 comes from the explicit init of k, and 2 from part of the deafult init of
// a.
//
//  type t
//    i = 1
//    j = 2
//  end type
//  type(t), save :: a
//  integer, save :: k = 3
//  equivalence (a, k)
//

/// Helper to analyze overlaps between default and explicit initialization in
/// equivalences. \p offset is the current equivalence offset, memIter is the
/// current symbol iterator over the equivalence members. It must point to a
/// member that has default initialization. \p endIter must be the end of the
/// equivalence members iterators. This will tell if the equivalence member \p
/// memIter is fully, partially, or not overlapped by members with explicit
/// initialization.
enum class OverlapWithExplicitInit { None, Partial, Full };
template <typename SymIter>
static OverlapWithExplicitInit analyzeDefaultInitOverlap(std::size_t offset,
                                                         SymIter memIter,
                                                         SymIter endIter) {
  const auto *mem = *memIter;
  auto isFullyEquivalencedByExplicitInit = false;
  auto overlapsWithExplicitInit = false;
  if (mem->offset() < offset)
    // Explicit initialization was already performed for part of the storage
    // that also have default initialization.
    overlapsWithExplicitInit = true;
  auto memEnd = mem->offset() + mem->size();
  if (memEnd > offset) {
    // The default initialization overlaps storage that has no initialization
    // yet. It may have to be taken into account if no further explicit
    // initialization overrides it.
    auto fullExplicitInitUntil = offset;
    for (auto peek = memIter; peek < endIter; ++peek) {
      const auto *nextSym = *peek;
      if (nextSym->offset() > memEnd)
        // Reached the end of the storage that may be default initialized.
        break;
      if (const auto *nextDet = nextSym->template detailsIf<
                                Fortran::semantics::ObjectEntityDetails>())
        if (nextDet->init()) {
          // Test there is no gap between the previous explicit init and this
          // one.
          overlapsWithExplicitInit = true;
          if (nextSym->offset() == fullExplicitInitUntil)
            fullExplicitInitUntil = nextSym->offset() + nextSym->size();
        }
    }
    isFullyEquivalencedByExplicitInit = fullExplicitInitUntil >= memEnd;
  } else {
    isFullyEquivalencedByExplicitInit = overlapsWithExplicitInit;
  }
  if (!overlapsWithExplicitInit)
    return OverlapWithExplicitInit::None;
  if (isFullyEquivalencedByExplicitInit)
    return OverlapWithExplicitInit::Full;
  return OverlapWithExplicitInit::Partial;
}

/// Build the type for the storage of an equivalence.
static mlir::TupleType
getAggregateType(Fortran::lower::AbstractConverter &converter,
                 const Fortran::lower::pft::Variable::AggregateStore &st) {
  auto &builder = converter.getFirOpBuilder();
  auto i8Ty = builder.getIntegerType(8);
  llvm::SmallVector<mlir::Type> members;
  std::size_t counter = std::get<0>(st.interval);

  for (auto iter = st.vars.begin(); iter != st.vars.end(); ++iter) {
    const auto *mem = *iter;
    if (const auto *memDet =
            mem->detailsIf<Fortran::semantics::ObjectEntityDetails>()) {
      if (mem->offset() > counter) {
        fir::SequenceType::Shape len = {
            static_cast<fir::SequenceType::Extent>(mem->offset() - counter)};
        auto byteTy = builder.getIntegerType(8);
        auto memTy = fir::SequenceType::get(len, byteTy);
        members.push_back(memTy);
        counter = mem->offset();
      }
      if (memDet->init()) {
        auto memTy = converter.genType(*mem);
        members.push_back(memTy);
        counter = mem->offset() + mem->size();
      } else if (hasDefaultInitialization(*mem)) {
        auto overlap = analyzeDefaultInitOverlap(counter, iter, st.vars.end());
        if (overlap == OverlapWithExplicitInit::None) {
          if (mem->offset() == counter) {
            // No overlap with explicit init on the storage and default init
            // not yet performed for this storage. Apply default initialization.
            auto memTy = converter.genType(*mem);
            members.push_back(memTy);
            counter = mem->offset() + mem->size();
          } else {
            // It is possible the storage was already default initialized by an
            // entity of the same type. The standard mandates in 19.5.3.4 point
            // 10 that only one type must provide default initialization for a
            // storage. Simply ensure the already default initialized storage
            // has the same size.
            assert(counter == mem->offset() + mem->size() &&
                   "bad default init overlap");
          }
        } else if (overlap == OverlapWithExplicitInit::Partial) {
          // Must interleave pieces of default initialization with explicit init
          // to achieve ifort, nvfortran and nag semantics.
          TODO(converter.genLocation(mem->name()),
               "overlapping default and explicit initialization in equivalence "
               "storage");
        } else {
          // The storage if fully covered with explicit initialization, simply
          // ignore the default initialization and let explicit initialization
          // happen.
          assert(overlap == OverlapWithExplicitInit::Full);
        }
      }
    }
  }
  if (counter < std::get<0>(st.interval) + std::get<1>(st.interval)) {
    fir::SequenceType::Shape len = {static_cast<fir::SequenceType::Extent>(
        std::get<0>(st.interval) + std::get<1>(st.interval) - counter)};
    auto memTy = fir::SequenceType::get(len, i8Ty);
    members.push_back(memTy);
  }
  return mlir::TupleType::get(builder.getContext(), members);
}
/// Define a GlobalOp for the storage of a global equivalence described
/// by \p aggregate. The global is named \p aggName and is created with
/// the provided \p linkage.
/// If any of the equivalence members are initialized, an initializer is
/// created for the equivalence.
/// This is to be used when lowering the scope that owns the equivalence
/// (as opposed to simply using it through host or use association).
/// This is not to be used for equivalence of common block members (they
/// already have the common block GlobalOp for them, see defineCommonBlock).
static fir::GlobalOp defineGlobalAggregateStore(
    Fortran::lower::AbstractConverter &converter,
    const Fortran::lower::pft::Variable::AggregateStore &aggregate,
    StringRef aggName, mlir::StringAttr linkage) {
  assert(aggregate.isGlobal() && "not a global interval");
  auto &builder = converter.getFirOpBuilder();
  auto loc = converter.getCurrentLocation();
  auto idxTy = builder.getIndexType();
  mlir::TupleType aggTy = getAggregateType(converter, aggregate);
  auto initFunc = [&](Fortran::lower::FirOpBuilder &builder) {
    mlir::Value cb = builder.create<fir::UndefOp>(loc, aggTy);
    unsigned tupIdx = 0;
    std::size_t offset = std::get<0>(aggregate.interval);
    LLVM_DEBUG(llvm::dbgs() << "equivalence {\n");
    for (auto iter = aggregate.vars.begin(); iter != aggregate.vars.end();
         ++iter) {
      const auto *mem = *iter;
      if (const auto *memDet =
              mem->detailsIf<Fortran::semantics::ObjectEntityDetails>()) {
        if (mem->offset() > offset) {
          ++tupIdx;
          offset = mem->offset();
        }
        bool applyDefaultInit = false;
        if (hasDefaultInitialization(*mem)) {
          auto overlap =
              analyzeDefaultInitOverlap(offset, iter, aggregate.vars.end());
          if (overlap == OverlapWithExplicitInit::None) {
            if (mem->offset() == offset)
              // No explicit init. Not yet default initialized.
              applyDefaultInit = true;
            else
              // Already default initialized.
              assert(offset == mem->offset() + mem->size() &&
                     "bad default init overlap");
          } else if (overlap == OverlapWithExplicitInit::Partial) {
            TODO(converter.genLocation(mem->name()),
                 "overlapping default and explicit initialization in "
                 "equivalence storage");
          } else {
            // Explicit inits completely override default init.
            assert(overlap == OverlapWithExplicitInit::Full);
          }
        }
        if (memDet->init() || applyDefaultInit) {
          LLVM_DEBUG(llvm::dbgs()
                     << "offset: " << mem->offset() << " is " << *mem << '\n');
          Fortran::lower::StatementContext stmtCtx;
          mlir::Value initVal;
          if (memDet->init())
            // Explicit initialization.
            initVal = fir::getBase(genInitializerExprValue(
                converter, loc, memDet->init().value(), stmtCtx));
          else
            initVal = genDefaultInitializerValue(
                converter, loc, *mem, converter.genType(*mem), stmtCtx);
          auto offVal = builder.createIntegerConstant(loc, idxTy, tupIdx);
          auto castVal =
              builder.createConvert(loc, aggTy.getType(tupIdx), initVal);
          cb = builder.create<fir::InsertValueOp>(loc, aggTy, cb, castVal,
                                                  offVal);
          ++tupIdx;
          offset = mem->offset() + mem->size();
        }
      }
    }
    LLVM_DEBUG(llvm::dbgs() << "}\n");
    builder.create<fir::HasValueOp>(loc, cb);
  };
  return builder.createGlobal(loc, aggTy, aggName,
                              /*isConstant=*/false, initFunc, linkage);
}

/// Declare a GlobalOp for the storage of a global equivalence described
/// by \p aggregate. The global is named \p aggName and is created with
/// the provided \p linkage.
/// No initializer is built for the created GlobalOp.
/// This is to be used when lowering the scope that uses members of an
/// equivalence it through host or use association.
/// This is not to be used for equivalence of common block members (they
/// already have the common block GlobalOp for them, see defineCommonBlock).
static fir::GlobalOp declareGlobalAggregateStore(
    Fortran::lower::AbstractConverter &converter, mlir::Location loc,
    const Fortran::lower::pft::Variable::AggregateStore &aggregate,
    StringRef aggName, mlir::StringAttr linkage) {
  assert(aggregate.isGlobal() && "not a global interval");
  auto aggTy = getAggregateType(converter, aggregate);
  return converter.getFirOpBuilder().createGlobal(loc, aggTy, aggName, linkage);
}

/// This is an aggregate store for a set of EQUIVALENCED variables. Create the
/// storage on the stack or global memory and add it to the map.
static void
instantiateAggregateStore(Fortran::lower::AbstractConverter &converter,
                          const Fortran::lower::pft::Variable &var,
                          Fortran::lower::AggregateStoreMap &storeMap) {
  assert(var.isAggregateStore() && "not an interval");
  auto &builder = converter.getFirOpBuilder();
  auto i8Ty = builder.getIntegerType(8);
  auto loc = converter.getCurrentLocation();
  if (var.isGlobal()) {
    // The scope of this aggregate is this procedure.
    auto aggName = mangleGlobalAggregateStore(var.getAggregateStore());
    fir::GlobalOp global = builder.getNamedGlobal(aggName);
    if (!global) {
      auto &aggregate = var.getAggregateStore();
      if (var.isDeclaration()) {
        // Using aggregate from a module not defined in the current
        // compilation unit.
        mlir::StringAttr externalLinkage;
        global = declareGlobalAggregateStore(converter, loc, aggregate, aggName,
                                             externalLinkage);
      } else {
        // The aggregate is owned by a procedure and must not be
        // visible in other compilation units.
        auto internalLinkage = builder.createInternalLinkage();
        global = defineGlobalAggregateStore(converter, aggregate, aggName,
                                            internalLinkage);
      }
    }
    auto addr = builder.create<fir::AddrOfOp>(loc, global.resultType(),
                                              global.getSymbol());
    auto size = std::get<1>(var.getInterval());
    fir::SequenceType::Shape shape(1, size);
    auto seqTy = fir::SequenceType::get(shape, i8Ty);
    auto refTy = builder.getRefType(seqTy);
    auto aggregateStore = builder.createConvert(loc, refTy, addr);
    insertAggregateStore(storeMap, var, aggregateStore);
    return;
  }
  // This is a local aggregate, allocate an anonymous block of memory.
  auto size = std::get<1>(var.getInterval());
  fir::SequenceType::Shape shape(1, size);
  auto seqTy = fir::SequenceType::get(shape, i8Ty);
  auto local =
      builder.allocateLocal(loc, seqTy, ".aggtmp", "", llvm::None, llvm::None,
                            /*target=*/false);
  insertAggregateStore(storeMap, var, local);
}

/// Instantiate a member of an equivalence. Compute its address in its
/// aggregate storage and lower its attributes.
static void instantiateAlias(Fortran::lower::AbstractConverter &converter,
                             const Fortran::lower::pft::Variable &var,
                             Fortran::lower::SymMap &symMap,
                             Fortran::lower::AggregateStoreMap &storeMap) {
  auto &builder = converter.getFirOpBuilder();
  assert(var.isAlias());
  const auto &sym = var.getSymbol();
  const auto loc = converter.genLocation(sym.name());
  auto idxTy = builder.getIndexType();
  auto aliasOffset = var.getAlias();
  auto store = getAggregateStore(storeMap, var);
  auto i8Ty = builder.getIntegerType(8);
  auto i8Ptr = builder.getRefType(i8Ty);
  auto offset =
      builder.createIntegerConstant(loc, idxTy, sym.offset() - aliasOffset);
  auto ptr = builder.create<fir::CoordinateOp>(loc, i8Ptr, store,
                                               mlir::ValueRange{offset});
  auto preAlloc = builder.createConvert(
      loc, builder.getRefType(converter.genType(sym)), ptr);
  Fortran::lower::StatementContext stmtCtx;
  mapSymbolAttributes(converter, var, symMap, stmtCtx, preAlloc);
  // Default initialization is possible for equivalence members: see
  // F2018 19.5.3.4. Note that if several equivalenced entities have
  // default initialization, they must have the same type, and the standard
  // allows the storage to be default initialized several times (this has
  // no consequences other than wasting some execution time). For now,
  // do not try optimizing this to single default initializations of
  // the equivalenced storages. Keep lowering simple.
  if (mustBeDefaultInitializedAtRuntime(var))
    defaultInitializeAtRuntime(converter, var, symMap);
}

//===--------------------------------------------------------------===//
// COMMON blocks instantiation
//===--------------------------------------------------------------===//

/// Does any member of the common block has an initializer ?
static bool
commonBlockHasInit(const Fortran::semantics::MutableSymbolVector &cmnBlkMems) {
  for (const auto &mem : cmnBlkMems) {
    if (const auto *memDet =
            mem->detailsIf<Fortran::semantics::ObjectEntityDetails>())
      if (memDet->init())
        return true;
  }
  return false;
}

/// Build a tuple type for a common block based on the common block
/// members and the common block size.
/// This type is only needed to build common block initializers where
/// the initial value is the collection of the member initial values.
static mlir::TupleType getTypeOfCommonWithInit(
    Fortran::lower::AbstractConverter &converter,
    const Fortran::semantics::MutableSymbolVector &cmnBlkMems,
    std::size_t commonSize) {
  auto &builder = converter.getFirOpBuilder();
  llvm::SmallVector<mlir::Type> members;
  std::size_t counter = 0;
  for (const auto &mem : cmnBlkMems) {
    if (const auto *memDet =
            mem->detailsIf<Fortran::semantics::ObjectEntityDetails>()) {
      if (mem->offset() > counter) {
        fir::SequenceType::Shape len = {
            static_cast<fir::SequenceType::Extent>(mem->offset() - counter)};
        auto byteTy = builder.getIntegerType(8);
        auto memTy = fir::SequenceType::get(len, byteTy);
        members.push_back(memTy);
        counter = mem->offset();
      }
      if (memDet->init()) {
        auto memTy = converter.genType(*mem);
        members.push_back(memTy);
        counter = mem->offset() + mem->size();
      }
    }
  }
  if (counter < commonSize) {
    fir::SequenceType::Shape len = {
        static_cast<fir::SequenceType::Extent>(commonSize - counter)};
    auto byteTy = builder.getIntegerType(8);
    auto memTy = fir::SequenceType::get(len, byteTy);
    members.push_back(memTy);
  }
  return mlir::TupleType::get(builder.getContext(), members);
}

/// Common block members may have aliases. They are not in the common block
/// member list from the symbol. We need to know about these aliases if they
/// have initializer to generate the common initializer.
/// This function takes care of adding aliases with initializer to the member
/// list.
static Fortran::semantics::MutableSymbolVector
getCommonMembersWithInitAliases(const Fortran::semantics::Symbol &common) {
  const auto &commonDetails =
      common.get<Fortran::semantics::CommonBlockDetails>();
  auto members = commonDetails.objects();

  // The number and size of equivalence and common is expected to be small, so
  // no effort is given to optimize this loop of complexity equivalenced
  // common members * common members
  for (const auto &set : common.owner().equivalenceSets())
    for (const auto &obj : set) {
      if (const auto &details =
              obj.symbol.detailsIf<Fortran::semantics::ObjectEntityDetails>()) {
        const auto *com = FindCommonBlockContaining(obj.symbol);
        if (!details->init() || com != &common)
          continue;
        // This is an alias with an init that belongs to the list
        if (std::find(members.begin(), members.end(), obj.symbol) ==
            members.end())
          members.emplace_back(obj.symbol);
      }
    }
  return members;
}

/// Define a global for a common block if it does not already exist in the
/// mlir module.
/// There is no "declare" version since there is not a
/// scope that owns common blocks more that the others. All scopes using
/// a common block attempts to define it with common linkage.
static fir::GlobalOp
defineCommonBlock(Fortran::lower::AbstractConverter &converter,
                  const Fortran::semantics::Symbol &common) {
  auto &builder = converter.getFirOpBuilder();
  auto commonName = Fortran::lower::mangle::mangleName(common);
  auto global = builder.getNamedGlobal(commonName);
  if (global)
    return global;
  auto cmnBlkMems = getCommonMembersWithInitAliases(common);
  auto loc = converter.genLocation(common.name());
  auto idxTy = builder.getIndexType();
  auto linkage = builder.createCommonLinkage();
  if (!common.name().size() || !commonBlockHasInit(cmnBlkMems)) {
    // A blank (anonymous) COMMON block must always be initialized to zero.
    // A named COMMON block sans initializers is also initialized to zero.
    // mlir::Vector types must have a strictly positive size, so at least
    // temporarily, force a zero size COMMON block to have one byte.
    const auto sz = static_cast<fir::SequenceType::Extent>(
        common.size() > 0 ? common.size() : 1);
    fir::SequenceType::Shape shape = {sz};
    auto i8Ty = builder.getIntegerType(8);
    auto commonTy = fir::SequenceType::get(shape, i8Ty);
    auto vecTy = mlir::VectorType::get(sz, i8Ty);
    mlir::Attribute zero = builder.getIntegerAttr(i8Ty, 0);
    auto init = mlir::DenseElementsAttr::get(vecTy, llvm::makeArrayRef(zero));
    return builder.createGlobal(loc, commonTy, commonName, linkage, init);
  }

  // Named common with initializer, sort members by offset before generating
  // the type and initializer.
  std::sort(cmnBlkMems.begin(), cmnBlkMems.end(),
            [](auto &s1, auto &s2) { return s1->offset() < s2->offset(); });
  auto commonTy = getTypeOfCommonWithInit(converter, cmnBlkMems, common.size());
  auto initFunc = [&](Fortran::lower::FirOpBuilder &builder) {
    mlir::Value cb = builder.create<fir::UndefOp>(loc, commonTy);
    unsigned tupIdx = 0;
    std::size_t offset = 0;
    LLVM_DEBUG(llvm::dbgs() << "block {\n");
    for (const auto &mem : cmnBlkMems) {
      if (const auto *memDet =
              mem->detailsIf<Fortran::semantics::ObjectEntityDetails>()) {
        if (mem->offset() > offset) {
          ++tupIdx;
          offset = mem->offset();
        }
        if (memDet->init()) {
          LLVM_DEBUG(llvm::dbgs()
                     << "offset: " << mem->offset() << " is " << *mem << '\n');
          Fortran::lower::StatementContext stmtCtx;
          auto initExpr = memDet->init().value();
          auto initVal =
              Fortran::semantics::IsPointer(*mem)
                  ? Fortran::lower::genInitialDataTarget(
                        converter, loc, converter.genType(*mem), initExpr)
                  : genInitializerExprValue(converter, loc, initExpr, stmtCtx);
          auto offVal = builder.createIntegerConstant(loc, idxTy, tupIdx);
          auto castVal = builder.createConvert(loc, commonTy.getType(tupIdx),
                                               fir::getBase(initVal));
          cb = builder.create<fir::InsertValueOp>(loc, commonTy, cb, castVal,
                                                  offVal);
          ++tupIdx;
          offset = mem->offset() + mem->size();
        }
      }
    }
    LLVM_DEBUG(llvm::dbgs() << "}\n");
    builder.create<fir::HasValueOp>(loc, cb);
  };
  // create the global object
  return builder.createGlobal(loc, commonTy, commonName,
                              /*isConstant=*/false, initFunc);
}
/// The COMMON block is a global structure. `var` will be at some offset
/// within the COMMON block. Adds the address of `var` (COMMON + offset) to
/// the symbol map.
static void instantiateCommon(Fortran::lower::AbstractConverter &converter,
                              const Fortran::semantics::Symbol &common,
                              const Fortran::lower::pft::Variable &var,
                              Fortran::lower::SymMap &symMap) {
  auto &builder = converter.getFirOpBuilder();
  const auto &varSym = var.getSymbol();
  auto loc = converter.genLocation(varSym.name());

  mlir::Value commonAddr;
  if (auto symBox = symMap.lookupSymbol(common))
    commonAddr = symBox.getAddr();
  if (!commonAddr) {
    // introduce a local AddrOf and add it to the map
    auto global = defineCommonBlock(converter, common);
    commonAddr = builder.create<fir::AddrOfOp>(loc, global.resultType(),
                                               global.getSymbol());

    symMap.addSymbol(common, commonAddr);
  }
  auto byteOffset = varSym.GetUltimate().offset();
  auto i8Ty = builder.getIntegerType(8);
  auto i8Ptr = builder.getRefType(i8Ty);
  auto seqTy = builder.getRefType(builder.getVarLenSeqTy(i8Ty));
  auto base = builder.createConvert(loc, seqTy, commonAddr);
  auto offs =
      builder.createIntegerConstant(loc, builder.getIndexType(), byteOffset);
  auto varAddr = builder.create<fir::CoordinateOp>(loc, i8Ptr, base,
                                                   mlir::ValueRange{offs});
  auto localTy = builder.getRefType(converter.genType(var.getSymbol()));
  mlir::Value local = builder.createConvert(loc, localTy, varAddr);
  Fortran::lower::StatementContext stmtCtx;
  mapSymbolAttributes(converter, var, symMap, stmtCtx, local);
}

//===--------------------------------------------------------------===//
// Lower Variables specification expressions and attributes
//===--------------------------------------------------------------===//

/// Helper to decide if a dummy argument must be tracked in an BoxValue.
static bool lowerToBoxValue(const Fortran::semantics::Symbol &sym,
                            mlir::Value dummyArg) {
  // Only dummy arguments coming as fir.box can be tracked in an BoxValue.
  if (!dummyArg || !dummyArg.getType().isa<fir::BoxType>())
    return false;
  // Non contiguous arrays must be tracked in an BoxValue.
  if (sym.Rank() > 0 && !sym.attrs().test(Fortran::semantics::Attr::CONTIGUOUS))
    return true;
  // Assumed rank and optional fir.box cannot yet be read while lowering the
  // specifications.
  if (Fortran::semantics::IsAssumedRankArray(sym) ||
      Fortran::semantics::IsOptional(sym))
    return true;
  // Polymorphic entity should be tracked through a fir.box that has the
  // dynamic type info.
  if (const auto *type = sym.GetType())
    if (type->IsPolymorphic())
      return true;
  return false;
}

/// Compute extent from lower and upper bound.
static mlir::Value computeExtent(Fortran::lower::FirOpBuilder &builder,
                                 mlir::Location loc, mlir::Value lb,
                                 mlir::Value ub) {
  auto idxTy = builder.getIndexType();
  // Let the folder deal with the common `ub - <const> + 1` case.
  auto diff = builder.create<mlir::SubIOp>(loc, idxTy, ub, lb);
  auto one = builder.createIntegerConstant(loc, idxTy, 1);
  return builder.create<mlir::AddIOp>(loc, idxTy, diff, one);
}

/// Lower explicit lower bounds into \p result. Does nothing if this is not an
/// array, or if the lower bounds are deferred, or all implicit or one.
static void lowerExplicitLowerBounds(
    Fortran::lower::AbstractConverter &converter, mlir::Location loc,
    const Fortran::lower::BoxAnalyzer &box,
    llvm::SmallVectorImpl<mlir::Value> &result, Fortran::lower::SymMap &symMap,
    Fortran::lower::StatementContext &stmtCtx) {
  if (!box.isArray() || box.lboundIsAllOnes())
    return;
  auto &builder = converter.getFirOpBuilder();
  auto idxTy = builder.getIndexType();
  if (box.isStaticArray()) {
    for (auto lb : box.staticLBound())
      result.emplace_back(builder.createIntegerConstant(loc, idxTy, lb));
    return;
  }
  for (const auto *spec : box.dynamicBound()) {
    if (auto low = spec->lbound().GetExplicit()) {
      auto expr = Fortran::semantics::SomeExpr{*low};
      auto lb = builder.createConvert(
          loc, idxTy, genScalarValue(converter, loc, expr, symMap, stmtCtx));
      result.emplace_back(lb);
    } else if (!spec->lbound().isDeferred()) {
      // Implicit lower bound is 1 (Fortran 2018 section 8.5.8.3 point 3.)
      result.emplace_back(builder.createIntegerConstant(loc, idxTy, 1));
    }
  }
  assert(result.empty() || result.size() == box.dynamicBound().size());
}

/// Lower explicit extents into \p result if this is an explicit-shape or
/// assumed-size array. Does nothing if this is not an explicit-shape or
/// assumed-size array.
static void lowerExplicitExtents(Fortran::lower::AbstractConverter &converter,
                                 mlir::Location loc,
                                 const Fortran::lower::BoxAnalyzer &box,
                                 llvm::ArrayRef<mlir::Value> lowerBounds,
                                 llvm::SmallVectorImpl<mlir::Value> &result,
                                 Fortran::lower::SymMap &symMap,
                                 Fortran::lower::StatementContext &stmtCtx) {
  if (!box.isArray())
    return;
  auto &builder = converter.getFirOpBuilder();
  auto idxTy = builder.getIndexType();
  if (box.isStaticArray()) {
    for (auto extent : box.staticShape())
      result.emplace_back(builder.createIntegerConstant(loc, idxTy, extent));
    return;
  }
  for (const auto &spec : llvm::enumerate(box.dynamicBound())) {
    if (auto up = spec.value()->ubound().GetExplicit()) {
      auto expr = Fortran::semantics::SomeExpr{*up};
      auto ub = builder.createConvert(
          loc, idxTy, genScalarValue(converter, loc, expr, symMap, stmtCtx));
      if (lowerBounds.empty())
        result.emplace_back(ub);
      else
        result.emplace_back(
            computeExtent(builder, loc, lowerBounds[spec.index()], ub));
    } else if (spec.value()->ubound().isAssumed()) {
      // Column extent is undefined. Must be provided by user's code.
      result.emplace_back(builder.create<fir::UndefOp>(loc, idxTy));
    }
  }
  assert(result.empty() || result.size() == box.dynamicBound().size());
}

/// Lower explicit character length if any. Return empty mlir::Value if no
/// explicit length.
static mlir::Value
lowerExplicitCharLen(Fortran::lower::AbstractConverter &converter,
                     mlir::Location loc, const Fortran::lower::BoxAnalyzer &box,
                     Fortran::lower::SymMap &symMap,
                     Fortran::lower::StatementContext &stmtCtx) {
  if (!box.isChar())
    return mlir::Value{};
  auto &builder = converter.getFirOpBuilder();
  auto lenTy = builder.getCharacterLengthType();
  if (auto len = box.getCharLenConst())
    return builder.createIntegerConstant(loc, lenTy, *len);
  if (auto lenExpr = box.getCharLenExpr())
    return genScalarValue(converter, loc, *lenExpr, symMap, stmtCtx);
  return mlir::Value{};
}

/// Lower specification expressions and attributes of variable \p var and
/// add it to the symbol map.
/// For global and aliases, the address must be pre-computed and provided
/// in \p preAlloc.
/// Dummy arguments must have already been mapped to mlir block arguments
/// their mapping may be updated here.
void Fortran::lower::mapSymbolAttributes(
    AbstractConverter &converter, const Fortran::lower::pft::Variable &var,
    Fortran::lower::SymMap &symMap, Fortran::lower::StatementContext &stmtCtx,
    mlir::Value preAlloc) {
  auto &builder = converter.getFirOpBuilder();
  const auto &sym = var.getSymbol();
  const auto loc = converter.genLocation(sym.name());
  auto idxTy = builder.getIndexType();
  const auto isDummy = Fortran::semantics::IsDummy(sym);
  const auto isResult = Fortran::semantics::IsFunctionResult(sym);
  const auto replace = isDummy || isResult;
  Fortran::lower::CharacterExprHelper charHelp{builder, loc};
  Fortran::lower::BoxAnalyzer ba;
  ba.analyze(sym);

  // First deal with pointers an allocatables, because their handling here
  // is the same regardless of their rank.
  if (Fortran::semantics::IsAllocatableOrPointer(sym)) {
    // Get address of fir.box describing the entity.
    // global
    auto boxAlloc = preAlloc;
    // dummy or passed result
    if (!boxAlloc)
      if (auto symbox = symMap.lookupSymbol(sym))
        boxAlloc = symbox.getAddr();
    // local
    if (!boxAlloc)
      boxAlloc = createNewLocal(converter, loc, var, preAlloc);
    // Lower non deferred parameters.
    llvm::SmallVector<mlir::Value> nonDeferredLenParams;
    if (ba.isChar()) {
      if (auto len = lowerExplicitCharLen(converter, loc, ba, symMap, stmtCtx))
        nonDeferredLenParams.push_back(len);
      else if (Fortran::semantics::IsAssumedLengthCharacter(sym))
        TODO(loc, "assumed length character allocatable");
    } else if (const auto *declTy = sym.GetType()) {
      if (const auto *derived = declTy->AsDerived())
        if (Fortran::semantics::CountLenParameters(*derived) != 0)
          TODO(loc,
               "derived type allocatable or pointer with length parameters");
    }
    auto box = Fortran::lower::createMutableBox(converter, loc, var, boxAlloc,
                                                nonDeferredLenParams);
    symMap.addAllocatableOrPointer(var.getSymbol(), box, replace);
    return;
  }

  if (isDummy) {
    auto dummyArg = symMap.lookupSymbol(sym).getAddr();
    if (lowerToBoxValue(sym, dummyArg)) {
      llvm::SmallVector<mlir::Value> lbounds;
      llvm::SmallVector<mlir::Value> extents;
      llvm::SmallVector<mlir::Value> explicitParams;
      // Lower lower bounds, explicit type parameters and explicit
      // extents if any.
      if (ba.isChar())
        if (auto len =
                lowerExplicitCharLen(converter, loc, ba, symMap, stmtCtx))
          explicitParams.push_back(len);
      // TODO: derived type length parameters.
      lowerExplicitLowerBounds(converter, loc, ba, lbounds, symMap, stmtCtx);
      lowerExplicitExtents(converter, loc, ba, lbounds, extents, symMap,
                           stmtCtx);
      symMap.addBoxSymbol(sym, dummyArg, lbounds, explicitParams, extents,
                          replace);
      return;
    }
  }

  // Helper to generate scalars for the symbol properties.
  auto genValue = [&](const Fortran::lower::SomeExpr &expr) {
    return genScalarValue(converter, loc, expr, symMap, stmtCtx);
  };

  // For symbols reaching this point, all properties are constant and can be
  // read/computed already into ssa values.

  // The origin must be \vec{1}.
  auto populateShape = [&](auto &shapes, const auto &bounds, mlir::Value box) {
    for (auto iter : llvm::enumerate(bounds)) {
      auto *spec = iter.value();
      assert(spec->lbound().GetExplicit() &&
             "lbound must be explicit with constant value 1");
      if (auto high = spec->ubound().GetExplicit()) {
        Fortran::semantics::SomeExpr highEx{*high};
        auto ub = genValue(highEx);
        shapes.emplace_back(builder.createConvert(loc, idxTy, ub));
      } else if (spec->ubound().isDeferred()) {
        assert(box && "deferred bounds require a descriptor");
        auto dim = builder.createIntegerConstant(loc, idxTy, iter.index());
        auto dimInfo =
            builder.create<fir::BoxDimsOp>(loc, idxTy, idxTy, idxTy, box, dim);
        shapes.emplace_back(dimInfo.getResult(1));
      } else if (spec->ubound().isAssumed()) {
        shapes.emplace_back(builder.create<fir::UndefOp>(loc, idxTy));
      } else {
        llvm::report_fatal_error("unknown bound category");
      }
    }
  };

  // The origin is not \vec{1}.
  auto populateLBoundsExtents = [&](auto &lbounds, auto &extents,
                                    const auto &bounds, mlir::Value box) {
    for (auto iter : llvm::enumerate(bounds)) {
      auto *spec = iter.value();
      fir::BoxDimsOp dimInfo;
      mlir::Value ub, lb;
      if (spec->lbound().isDeferred() || spec->ubound().isDeferred()) {
        // This is an assumed shape because allocatables and pointers extents
        // are not constant in the scope and are not read here.
        assert(box && "deferred bounds require a descriptor");
        auto dim = builder.createIntegerConstant(loc, idxTy, iter.index());
        dimInfo =
            builder.create<fir::BoxDimsOp>(loc, idxTy, idxTy, idxTy, box, dim);
        extents.emplace_back(dimInfo.getResult(1));
        if (auto low = spec->lbound().GetExplicit()) {
          auto expr = Fortran::semantics::SomeExpr{*low};
          auto lb = builder.createConvert(loc, idxTy, genValue(expr));
          lbounds.emplace_back(lb);
        } else {
          // Implicit lower bound is 1 (Fortran 2018 section 8.5.8.3 point 3.)
          lbounds.emplace_back(builder.createIntegerConstant(loc, idxTy, 1));
        }
      } else {
        if (auto low = spec->lbound().GetExplicit()) {
          auto expr = Fortran::semantics::SomeExpr{*low};
          lb = builder.createConvert(loc, idxTy, genValue(expr));
        } else {
          TODO(loc, "assumed rank lowering");
        }

        if (auto high = spec->ubound().GetExplicit()) {
          auto expr = Fortran::semantics::SomeExpr{*high};
          ub = builder.createConvert(loc, idxTy, genValue(expr));
          lbounds.emplace_back(lb);
          extents.emplace_back(computeExtent(builder, loc, lb, ub));
        } else {
          // An assumed size array. The extent is not computed.
          assert(spec->ubound().isAssumed() && "expected assumed size");
          lbounds.emplace_back(lb);
          extents.emplace_back(builder.create<fir::UndefOp>(loc, idxTy));
        }
      }
    }
  };

  ba.match(
      //===--------------------------------------------------------------===//
      // Trivial case.
      //===--------------------------------------------------------------===//
      [&](const Fortran::lower::details::ScalarSym &) {
        if (isDummy) {
          // This is an argument.
          if (!symMap.lookupSymbol(sym))
            mlir::emitError(loc, "symbol \"")
                << toStringRef(sym.name()) << "\" must already be in map";
          return;
        } else if (isResult) {
          // Some Fortran results may be passed by argument (e.g. derived
          // types)
          if (symMap.lookupSymbol(sym))
            return;
        }
        // Otherwise, it's a local variable or function result.
        auto local = createNewLocal(converter, loc, var, preAlloc);
        symMap.addSymbol(sym, local);
      },

      //===--------------------------------------------------------------===//
      // The non-trivial cases are when we have an argument or local that has
      // a repetition value. Arguments might be passed as simple pointers and
      // need to be cast to a multi-dimensional array with constant bounds
      // (possibly with a missing column), bounds computed in the callee
      // (here), or with bounds from the caller (boxed somewhere else). Locals
      // have the same properties except they are never boxed arguments from
      // the caller and never having a missing column size.
      //===--------------------------------------------------------------===//

      [&](const Fortran::lower::details::ScalarStaticChar &x) {
        // type is a CHARACTER, determine the LEN value
        auto charLen = x.charLen();
        if (replace) {
          auto symBox = symMap.lookupSymbol(sym);
          auto unboxchar = charHelp.createUnboxChar(symBox.getAddr());
          auto boxAddr = unboxchar.first;
          // Set/override LEN with a constant
          auto len = builder.createIntegerConstant(loc, idxTy, charLen);
          symMap.addCharSymbol(sym, boxAddr, len, true);
          return;
        }
        auto len = builder.createIntegerConstant(loc, idxTy, charLen);
        if (preAlloc) {
          symMap.addCharSymbol(sym, preAlloc, len);
          return;
        }
        auto local = createNewLocal(converter, loc, var, preAlloc);
        symMap.addCharSymbol(sym, local, len);
      },

      //===--------------------------------------------------------------===//

      [&](const Fortran::lower::details::ScalarDynamicChar &x) {
        // type is a CHARACTER, determine the LEN value
        auto charLen = x.charLen();
        if (replace) {
          auto symBox = symMap.lookupSymbol(sym);
          auto boxAddr = symBox.getAddr();
          mlir::Value len;
          auto addrTy = boxAddr.getType();
          if (addrTy.isa<fir::BoxCharType>() || addrTy.isa<fir::BoxType>()) {
            std::tie(boxAddr, len) = charHelp.createUnboxChar(symBox.getAddr());
          } else {
            // dummy from an other entry case: we cannot get a dynamic length
            // for it, it's illegal for the user program to use it. However,
            // since we are lowering all function unit statements regardless
            // of whether the execution will reach them or not, we need to
            // fill a value for the length here.
            len = builder.createIntegerConstant(
                loc, builder.getCharacterLengthType(), 1);
          }
          // Override LEN with an expression
          if (charLen)
            len = genValue(*charLen);
          symMap.addCharSymbol(sym, boxAddr, len, true);
          return;
        }
        // local CHARACTER variable
        mlir::Value len;
        if (charLen)
          len = genValue(*charLen);
        else
          len = builder.createIntegerConstant(loc, idxTy, sym.size());
        if (preAlloc) {
          symMap.addCharSymbol(sym, preAlloc, len);
          return;
        }
        llvm::SmallVector<mlir::Value> lengths = {len};
        auto local =
            createNewLocal(converter, loc, var, preAlloc, llvm::None, lengths);
        symMap.addCharSymbol(sym, local, len);
      },

      //===--------------------------------------------------------------===//

      [&](const Fortran::lower::details::StaticArray &x) {
        // object shape is constant, not a character
        auto castTy = builder.getRefType(converter.genType(var));
        mlir::Value addr = symMap.lookupSymbol(sym).getAddr();
        if (addr)
          addr = builder.createConvert(loc, castTy, addr);
        if (x.lboundAllOnes()) {
          // if lower bounds are all ones, build simple shaped object
          llvm::SmallVector<mlir::Value> shape;
          for (auto i : x.shapes)
            shape.push_back(builder.createIntegerConstant(loc, idxTy, i));
          mlir::Value local =
              isDummy ? addr : createNewLocal(converter, loc, var, preAlloc);
          symMap.addSymbolWithShape(sym, local, shape, isDummy);
          return;
        }
        // If object is an array process the lower bound and extent values by
        // constructing constants and populating the lbounds and extents.
        llvm::SmallVector<mlir::Value> extents;
        llvm::SmallVector<mlir::Value> lbounds;
        for (auto [fst, snd] : llvm::zip(x.lbounds, x.shapes)) {
          lbounds.emplace_back(builder.createIntegerConstant(loc, idxTy, fst));
          extents.emplace_back(builder.createIntegerConstant(loc, idxTy, snd));
        }
        mlir::Value local =
            isDummy ? addr
                    : createNewLocal(converter, loc, var, preAlloc, extents);
        assert(isDummy || Fortran::lower::isExplicitShape(sym));
        symMap.addSymbolWithBounds(sym, local, extents, lbounds, isDummy);
      },

      //===--------------------------------------------------------------===//

      [&](const Fortran::lower::details::DynamicArray &x) {
        // cast to the known constant parts from the declaration
        auto varType = converter.genType(var);
        mlir::Value addr = symMap.lookupSymbol(sym).getAddr();
        mlir::Value argBox;
        auto castTy = builder.getRefType(varType);
        if (addr) {
          if (auto boxTy = addr.getType().dyn_cast<fir::BoxType>()) {
            argBox = addr;
            auto refTy = builder.getRefType(boxTy.getEleTy());
            addr = builder.create<fir::BoxAddrOp>(loc, refTy, argBox);
          }
          addr = builder.createConvert(loc, castTy, addr);
        }
        if (x.lboundAllOnes()) {
          // if lower bounds are all ones, build simple shaped object
          llvm::SmallVector<mlir::Value> shapes;
          populateShape(shapes, x.bounds, argBox);
          if (isDummy) {
            symMap.addSymbolWithShape(sym, addr, shapes, true);
            return;
          }
          // local array with computed bounds
          assert(Fortran::lower::isExplicitShape(sym) ||
                 Fortran::semantics::IsAllocatableOrPointer(sym));
          auto local = createNewLocal(converter, loc, var, preAlloc, shapes);
          symMap.addSymbolWithShape(sym, local, shapes);
          return;
        }
        // if object is an array process the lower bound and extent values
        llvm::SmallVector<mlir::Value> extents;
        llvm::SmallVector<mlir::Value> lbounds;
        populateLBoundsExtents(lbounds, extents, x.bounds, argBox);
        if (isDummy) {
          symMap.addSymbolWithBounds(sym, addr, extents, lbounds, true);
          return;
        }
        // local array with computed bounds
        assert(Fortran::lower::isExplicitShape(sym));
        auto local = createNewLocal(converter, loc, var, preAlloc, extents);
        symMap.addSymbolWithBounds(sym, local, extents, lbounds);
      },

      //===--------------------------------------------------------------===//

      [&](const Fortran::lower::details::StaticArrayStaticChar &x) {
        // if element type is a CHARACTER, determine the LEN value
        auto charLen = x.charLen();
        mlir::Value addr;
        mlir::Value len;
        if (isDummy) {
          auto symBox = symMap.lookupSymbol(sym);
          auto unboxchar = charHelp.createUnboxChar(symBox.getAddr());
          addr = unboxchar.first;
          // Set/override LEN with a constant
          len = builder.createIntegerConstant(loc, idxTy, charLen);
        } else {
          // local CHARACTER variable
          len = builder.createIntegerConstant(loc, idxTy, charLen);
        }

        // object shape is constant
        auto castTy = builder.getRefType(converter.genType(var));
        if (addr)
          addr = builder.createConvert(loc, castTy, addr);

        if (x.lboundAllOnes()) {
          // if lower bounds are all ones, build simple shaped object
          llvm::SmallVector<mlir::Value> shape;
          for (auto i : x.shapes)
            shape.push_back(builder.createIntegerConstant(loc, idxTy, i));
          mlir::Value local =
              isDummy ? addr : createNewLocal(converter, loc, var, preAlloc);
          symMap.addCharSymbolWithShape(sym, local, len, shape, isDummy);
          return;
        }

        // if object is an array process the lower bound and extent values
        llvm::SmallVector<mlir::Value> extents;
        llvm::SmallVector<mlir::Value> lbounds;
        // construct constants and populate `bounds`
        for (auto [fst, snd] : llvm::zip(x.lbounds, x.shapes)) {
          lbounds.emplace_back(builder.createIntegerConstant(loc, idxTy, fst));
          extents.emplace_back(builder.createIntegerConstant(loc, idxTy, snd));
        }

        if (isDummy) {
          symMap.addCharSymbolWithBounds(sym, addr, len, extents, lbounds,
                                         true);
          return;
        }
        // local CHARACTER array with computed bounds
        assert(Fortran::lower::isExplicitShape(sym));
        auto local = createNewLocal(converter, loc, var, preAlloc, extents);
        symMap.addCharSymbolWithBounds(sym, local, len, extents, lbounds);
      },

      //===--------------------------------------------------------------===//

      [&](const Fortran::lower::details::StaticArrayDynamicChar &x) {
        mlir::Value addr;
        mlir::Value len;
        [[maybe_unused]] bool mustBeDummy = false;
        auto charLen = x.charLen();
        // if element type is a CHARACTER, determine the LEN value
        if (isDummy) {
          auto symBox = symMap.lookupSymbol(sym);
          auto unboxchar = charHelp.createUnboxChar(symBox.getAddr());
          addr = unboxchar.first;
          if (charLen) {
            // Set/override LEN with an expression
            len = genValue(*charLen);
          } else {
            // LEN is from the boxchar
            len = unboxchar.second;
            mustBeDummy = true;
          }
        } else {
          // local CHARACTER variable
          if (charLen)
            len = genValue(*charLen);
          else
            len = builder.createIntegerConstant(loc, idxTy, sym.size());
        }
        llvm::SmallVector<mlir::Value> lengths = {len};

        // cast to the known constant parts from the declaration
        auto castTy = builder.getRefType(converter.genType(var));
        if (addr)
          addr = builder.createConvert(loc, castTy, addr);

        if (x.lboundAllOnes()) {
          // if lower bounds are all ones, build simple shaped object
          llvm::SmallVector<mlir::Value> shape;
          for (auto i : x.shapes)
            shape.push_back(builder.createIntegerConstant(loc, idxTy, i));
          if (isDummy) {
            symMap.addCharSymbolWithShape(sym, addr, len, shape, true);
            return;
          }
          // local CHARACTER array with constant size
          auto local = createNewLocal(converter, loc, var, preAlloc, llvm::None,
                                      lengths);
          symMap.addCharSymbolWithShape(sym, local, len, shape);
          return;
        }

        // if object is an array process the lower bound and extent values
        llvm::SmallVector<mlir::Value> extents;
        llvm::SmallVector<mlir::Value> lbounds;

        // construct constants and populate `bounds`
        for (auto [fst, snd] : llvm::zip(x.lbounds, x.shapes)) {
          lbounds.emplace_back(builder.createIntegerConstant(loc, idxTy, fst));
          extents.emplace_back(builder.createIntegerConstant(loc, idxTy, snd));
        }
        if (isDummy) {
          symMap.addCharSymbolWithBounds(sym, addr, len, extents, lbounds,
                                         true);
          return;
        }
        // local CHARACTER array with computed bounds
        assert((!mustBeDummy) && (Fortran::lower::isExplicitShape(sym)));
        auto local =
            createNewLocal(converter, loc, var, preAlloc, llvm::None, lengths);
        symMap.addCharSymbolWithBounds(sym, local, len, extents, lbounds);
      },

      //===--------------------------------------------------------------===//

      [&](const Fortran::lower::details::DynamicArrayStaticChar &x) {
        mlir::Value addr;
        mlir::Value len;
        mlir::Value argBox;
        auto charLen = x.charLen();
        // if element type is a CHARACTER, determine the LEN value
        if (isDummy) {
          auto actualArg = symMap.lookupSymbol(sym).getAddr();
          if (auto boxTy = actualArg.getType().dyn_cast<fir::BoxType>()) {
            argBox = actualArg;
            auto refTy = builder.getRefType(boxTy.getEleTy());
            addr = builder.create<fir::BoxAddrOp>(loc, refTy, argBox);
          } else {
            addr = charHelp.createUnboxChar(actualArg).first;
          }
          // Set/override LEN with a constant
          len = builder.createIntegerConstant(loc, idxTy, charLen);
        } else {
          // local CHARACTER variable
          len = builder.createIntegerConstant(loc, idxTy, charLen);
        }

        // cast to the known constant parts from the declaration
        auto castTy = builder.getRefType(converter.genType(var));
        if (addr)
          addr = builder.createConvert(loc, castTy, addr);
        if (x.lboundAllOnes()) {
          // if lower bounds are all ones, build simple shaped object
          llvm::SmallVector<mlir::Value> shape;
          populateShape(shape, x.bounds, argBox);
          if (isDummy) {
            symMap.addCharSymbolWithShape(sym, addr, len, shape, true);
            return;
          }
          // local CHARACTER array
          auto local = createNewLocal(converter, loc, var, preAlloc, shape);
          symMap.addCharSymbolWithShape(sym, local, len, shape);
          return;
        }
        // if object is an array process the lower bound and extent values
        llvm::SmallVector<mlir::Value> extents;
        llvm::SmallVector<mlir::Value> lbounds;
        populateLBoundsExtents(lbounds, extents, x.bounds, argBox);
        if (isDummy) {
          symMap.addCharSymbolWithBounds(sym, addr, len, extents, lbounds,
                                         true);
          return;
        }
        // local CHARACTER array with computed bounds
        assert(Fortran::lower::isExplicitShape(sym));
        auto local = createNewLocal(converter, loc, var, preAlloc, extents);
        symMap.addCharSymbolWithBounds(sym, local, len, extents, lbounds);
      },

      //===--------------------------------------------------------------===//

      [&](const Fortran::lower::details::DynamicArrayDynamicChar &x) {
        mlir::Value addr;
        mlir::Value len;
        mlir::Value argBox;
        auto charLen = x.charLen();
        // if element type is a CHARACTER, determine the LEN value
        if (isDummy) {
          auto actualArg = symMap.lookupSymbol(sym).getAddr();
          if (auto boxTy = actualArg.getType().dyn_cast<fir::BoxType>()) {
            argBox = actualArg;
            auto refTy = builder.getRefType(boxTy.getEleTy());
            addr = builder.create<fir::BoxAddrOp>(loc, refTy, argBox);
            if (charLen)
              // Set/override LEN with an expression.
              len = genValue(*charLen);
            else
              // Get the length from the actual arguments.
              len = charHelp.readLengthFromBox(argBox);
          } else {
            auto unboxchar = charHelp.createUnboxChar(actualArg);
            addr = unboxchar.first;
            if (charLen) {
              // Set/override LEN with an expression
              len = genValue(*charLen);
            } else {
              // Get the length from the actual arguments.
              len = unboxchar.second;
            }
          }
        } else {
          // local CHARACTER variable
          if (charLen)
            len = genValue(*charLen);
          else
            len = builder.createIntegerConstant(loc, idxTy, sym.size());
        }
        llvm::SmallVector<mlir::Value> lengths = {len};

        // cast to the known constant parts from the declaration
        auto castTy = builder.getRefType(converter.genType(var));
        if (addr)
          addr = builder.createConvert(loc, castTy, addr);
        if (x.lboundAllOnes()) {
          // if lower bounds are all ones, build simple shaped object
          llvm::SmallVector<mlir::Value> shape;
          populateShape(shape, x.bounds, argBox);
          if (isDummy) {
            symMap.addCharSymbolWithShape(sym, addr, len, shape, true);
            return;
          }
          // local CHARACTER array
          auto local =
              createNewLocal(converter, loc, var, preAlloc, shape, lengths);
          symMap.addCharSymbolWithShape(sym, local, len, shape);
          return;
        }
        // Process the lower bound and extent values.
        llvm::SmallVector<mlir::Value> extents;
        llvm::SmallVector<mlir::Value> lbounds;
        populateLBoundsExtents(lbounds, extents, x.bounds, argBox);
        if (isDummy) {
          symMap.addCharSymbolWithBounds(sym, addr, len, extents, lbounds,
                                         true);
          return;
        }
        // local CHARACTER array with computed bounds
        assert(Fortran::lower::isExplicitShape(sym));
        auto local =
            createNewLocal(converter, loc, var, preAlloc, extents, lengths);
        symMap.addCharSymbolWithBounds(sym, local, len, extents, lbounds);
      },

      //===--------------------------------------------------------------===//

      [&](const Fortran::lower::BoxAnalyzer::None &) {
        mlir::emitError(loc, "symbol analysis failed on ")
            << toStringRef(sym.name());
      });
}

void Fortran::lower::defineModuleVariable(
    AbstractConverter &converter, const Fortran::lower::pft::Variable &var) {
  // linkOnce linkage allows the definition to be kept by LLVM
  // even if it is unused and there is not init (other than undef).
  // Emitting llvm :'@glob = global type undef' would also work, but mlir
  // always insert "external" and removes the undef init when lowering a
  // global without explicit linkage. This ends-up in llvm removing the
  // symbol if unsued potentially creating linking issues. Hence the use
  // of linkOnce.
  auto linkOnce = converter.getFirOpBuilder().createLinkOnceLinkage();
  // Only define variable owned by this module
  if (var.isDeclaration())
    return;
  if (!var.isGlobal())
    fir::emitFatalError(converter.genLocation(),
                        "attempting to lower module variable as local");
  // Define aggregate storages for equivalenced objects.
  if (var.isAggregateStore()) {
    auto &aggregate = var.getAggregateStore();
    auto aggName = mangleGlobalAggregateStore(aggregate);
    defineGlobalAggregateStore(converter, aggregate, aggName, linkOnce);
    return;
  }
  const auto &sym = var.getSymbol();
  if (const auto *common =
          Fortran::semantics::FindCommonBlockContaining(var.getSymbol())) {
    // Define common block containing the variable.
    defineCommonBlock(converter, *common);
  } else if (var.isAlias()) {
    // Do nothing. Mapping will be done on user side.
  } else {
    auto globalName = Fortran::lower::mangle::mangleName(sym);
    defineGlobal(converter, var, globalName, linkOnce);
  }
}

void Fortran::lower::instantiateVariable(AbstractConverter &converter,
                                         const pft::Variable &var,
                                         Fortran::lower::SymMap &symMap,
                                         AggregateStoreMap &storeMap) {
  if (var.isAggregateStore()) {
    instantiateAggregateStore(converter, var, storeMap);
  } else if (const auto *common = Fortran::semantics::FindCommonBlockContaining(
                 var.getSymbol().GetUltimate())) {
    instantiateCommon(converter, *common, var, symMap);
  } else if (var.isAlias()) {
    instantiateAlias(converter, var, symMap, storeMap);
  } else if (var.isGlobal()) {
    instantiateGlobal(converter, var, symMap);
  } else {
    instantiateLocal(converter, var, symMap);
  }
}

void Fortran::lower::mapCallInterfaceSymbols(
    AbstractConverter &converter, const Fortran::lower::CallerInterface &caller,
    SymMap &symMap) {
  Fortran::lower::AggregateStoreMap storeMap;
  const auto &result = caller.getResultSymbol();
  for (auto var : Fortran::lower::pft::buildFuncResultDependencyList(result)) {
    if (var.isAggregateStore()) {
      instantiateVariable(converter, var, symMap, storeMap);
    } else {
      const Fortran::semantics::Symbol &sym = var.getSymbol();
      // Get the argument for the dummy argument symbols of the current call.
      if (Fortran::semantics::IsDummy(sym) && sym.owner() == result.owner())
        symMap.addSymbol(sym, caller.getArgumentValue(sym));
      instantiateVariable(converter, var, symMap, storeMap);
    }
  }
}
