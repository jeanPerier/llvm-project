//===-- ConvertExprToHLFIR.cpp
//---------------------------------------------------===//
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

#include "flang/Lower/ConvertExprToHLFIR.h"
#include "flang/Lower/AbstractConverter.h"
#include "flang/Lower/StatementContext.h"
#include "flang/Lower/SymbolMap.h"
#include "flang/Optimizer/Builder/Todo.h"

namespace {

/// Lower Designators to HLFIR.
class HlfirDesignatorBuilder {
public:
  HlfirDesignatorBuilder(mlir::Location loc,
                         Fortran::lower::AbstractConverter &converter,
                         Fortran::lower::SymMap &symMap,
                         Fortran::lower::StatementContext &stmtCtx)
      : converter{converter}, symMap{symMap}, stmtCtx{stmtCtx}, loc{loc} {}

  // Character designators variant contains substrings
  using CharacterDesignators =
      decltype(Fortran::evaluate::Designator<Fortran::evaluate::Type<
                   Fortran::evaluate::TypeCategory::Character, 1>>::u);
  fir::HlfirValue gen(const CharacterDesignators &designatorVariant) {
    return std::visit([&](const auto &x) { return gen(x); }, designatorVariant);
  }
  // Character designators variant contains complex parts
  using RealDesignators =
      decltype(Fortran::evaluate::Designator<Fortran::evaluate::Type<
                   Fortran::evaluate::TypeCategory::Real, 4>>::u);
  fir::HlfirValue gen(const RealDesignators &designatorVariant) {
    return std::visit([&](const auto &x) { return gen(x); }, designatorVariant);
  }
  // All other designators are similar
  using OtherDesignators =
      decltype(Fortran::evaluate::Designator<Fortran::evaluate::Type<
                   Fortran::evaluate::TypeCategory::Integer, 4>>::u);
  fir::HlfirValue gen(const OtherDesignators &designatorVariant) {
    return std::visit([&](const auto &x) { return gen(x); }, designatorVariant);
  }

private:
  fir::HlfirValue gen(const Fortran::evaluate::SymbolRef &symbolRef) {
    if (llvm::Optional<fir::DefineFortranVariableOpInterface> varDef =
            getSymMap().lookupVariableDefinition(symbolRef))
      return *varDef;
    TODO(getLoc(), "symbol");
  }
  fir::HlfirValue gen(const Fortran::evaluate::Component &component) {
    TODO(getLoc(), "component");
  }
  fir::HlfirValue gen(const Fortran::evaluate::ArrayRef &arrayRef) {
    TODO(getLoc(), "ArrayRef");
  }
  fir::HlfirValue gen(const Fortran::evaluate::CoarrayRef &coarrayRef) {
    TODO(getLoc(), "CoarrayRef");
  }
  fir::HlfirValue gen(const Fortran::evaluate::ComplexPart &complexPart) {
    TODO(getLoc(), "complex part");
  }
  fir::HlfirValue gen(const Fortran::evaluate::Substring &substring) {
    TODO(getLoc(), "substrings");
  }

  mlir::Location getLoc() const { return loc; }
  Fortran::lower::AbstractConverter &getConverter() { return converter; }
  fir::FirOpBuilder &getBuilder() { return converter.getFirOpBuilder(); }
  Fortran::lower::SymMap &getSymMap() { return symMap; }
  Fortran::lower::StatementContext &getStmtCtx() { return stmtCtx; }

  Fortran::lower::AbstractConverter &converter;
  Fortran::lower::SymMap &symMap;
  Fortran::lower::StatementContext &stmtCtx;
  mlir::Location loc;
};

class HlfirConstanBuilder {
}

/// Lower Expr to HLFIR.
class HlfirBuilder {
public:
  HlfirBuilder(mlir::Location loc, Fortran::lower::AbstractConverter &converter,
               Fortran::lower::SymMap &symMap,
               Fortran::lower::StatementContext &stmtCtx)
      : converter{converter}, symMap{symMap}, stmtCtx{stmtCtx}, loc{loc} {}

  template <typename T>
  fir::HlfirValue gen(const Fortran::evaluate::Expr<T> &expr) {
    return std::visit([&](const auto &x) { return gen(x); }, expr.u);
  }

private:
  fir::HlfirValue gen(const Fortran::evaluate::BOZLiteralConstant &expr) {
    fir::emitFatalError(loc, "BOZ literal must be replaced by semantics");
  }
  fir::HlfirValue gen(const Fortran::evaluate::NullPointer &expr) {
    TODO(getLoc(), "NullPointer");
  }
  fir::HlfirValue gen(const Fortran::evaluate::ProcedureDesignator &expr) {
    TODO(getLoc(), "ProcDes");
  }
  fir::HlfirValue gen(const Fortran::evaluate::ProcedureRef &expr) {
    TODO(getLoc(), "ProcRef");
  }

  template <typename T>
  fir::HlfirValue gen(const Fortran::evaluate::Designator<T> &designator) {
    return HlfirDesignatorBuilder(getLoc(), getConverter(), getSymMap(),
                                  getStmtCtx())
        .gen(designator.u);
  }

  template <typename T>
  fir::HlfirValue gen(const Fortran::evaluate::FunctionRef<T> &expr) {
    TODO(getLoc(), "funcRef");
  }

  template <typename T>
  fir::HlfirValue gen(const Fortran::evaluate::Constant<T> &expr) {
    TODO(getLoc(), "constant");
  }

  template <typename T>
  fir::HlfirValue gen(const Fortran::evaluate::ArrayConstructor<T> &expr) {
    TODO(getLoc(), "ArrayCtor");
  }

  template <Fortran::common::TypeCategory TC1, int KIND,
            Fortran::common::TypeCategory TC2>
  fir::HlfirValue
  gen(const Fortran::evaluate::Convert<Fortran::evaluate::Type<TC1, KIND>, TC2>
          &convert) {
    TODO(getLoc(), "convert");
  }

  template <typename D, typename R, typename O>
  fir::HlfirValue gen(const Fortran::evaluate::Operation<D, R, O> &op) {
    TODO(getLoc(), "unary op");
  }

  template <typename D, typename R, typename LO, typename RO>
  fir::HlfirValue gen(const Fortran::evaluate::Operation<D, R, LO, RO> &op) {
    TODO(getLoc(), "binary op");
  }

  fir::HlfirValue
  gen(const Fortran::evaluate::Relational<Fortran::evaluate::SomeType> &op) {
    return std::visit([&](const auto &x) { return gen(x); }, op.u);
  }

  fir::HlfirValue gen(const Fortran::evaluate::TypeParamInquiry &) {
    TODO(getLoc(), "type parameter inquiry");
  }

  fir::HlfirValue gen(const Fortran::evaluate::DescriptorInquiry &desc) {
    TODO(getLoc(), "descriptor inquiry");
  }

  fir::HlfirValue gen(const Fortran::evaluate::ImpliedDoIndex &var) {
    TODO(getLoc(), "implied do index");
  }

  fir::HlfirValue gen(const Fortran::evaluate::StructureConstructor &var) {
    TODO(getLoc(), "structure constructor");
  }

  mlir::Location getLoc() const { return loc; }
  Fortran::lower::AbstractConverter &getConverter() { return converter; }
  fir::FirOpBuilder &getBuilder() { return converter.getFirOpBuilder(); }
  Fortran::lower::SymMap &getSymMap() { return symMap; }
  Fortran::lower::StatementContext &getStmtCtx() { return stmtCtx; }

  Fortran::lower::AbstractConverter &converter;
  Fortran::lower::SymMap &symMap;
  Fortran::lower::StatementContext &stmtCtx;
  mlir::Location loc;
};

} // namespace

fir::HlfirValue Fortran::lower::convertExprToHLFIR(
    mlir::Location loc, Fortran::lower::AbstractConverter &converter,
    const Fortran::lower::SomeExpr &expr, Fortran::lower::SymMap &symMap,
    Fortran::lower::StatementContext &stmtCtx) {
  return HlfirBuilder(loc, converter, symMap, stmtCtx).gen(expr);
}
