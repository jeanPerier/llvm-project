//===-- Coarray.cpp -------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// Implementation of the lowering of image related constructs and expressions.
/// Fortran images can form teams, communicate via coarrays, etc.
///
//===----------------------------------------------------------------------===//

#include "flang/Lower/Coarray.h"
#include "SymbolMap.h"
#include "flang/Lower/AbstractConverter.h"
#include "flang/Lower/Todo.h"
#include "flang/Optimizer/Builder/FIRBuilder.h"
#include "flang/Parser/parse-tree.h"
#include "flang/Semantics/expression.h"

//===----------------------------------------------------------------------===//
// TEAM statements and constructs
//===----------------------------------------------------------------------===//

void Fortran::lower::genChangeTeamConstruct(
    Fortran::lower::AbstractConverter &converter,
    Fortran::lower::pft::Evaluation &,
    const Fortran::parser::ChangeTeamConstruct &) {
  TODO(converter.genLocation(), "CHANGE TEAM construct");
}

void Fortran::lower::genChangeTeamStmt(
    Fortran::lower::AbstractConverter &converter,
    Fortran::lower::pft::Evaluation &,
    const Fortran::parser::ChangeTeamStmt &) {
  TODO(converter.genLocation(), "CHANGE TEAM stmt");
}

void Fortran::lower::genEndChangeTeamStmt(
    Fortran::lower::AbstractConverter &converter,
    Fortran::lower::pft::Evaluation &,
    const Fortran::parser::EndChangeTeamStmt &) {
  TODO(converter.genLocation(), "END CHANGE TEAM");
}

void Fortran::lower::genFormTeamStatement(
    Fortran::lower::AbstractConverter &converter,
    Fortran::lower::pft::Evaluation &, const Fortran::parser::FormTeamStmt &) {
  TODO(converter.genLocation(), "FORM TEAM");
}

//===----------------------------------------------------------------------===//
// COARRAY expressions
//===----------------------------------------------------------------------===//

fir::ExtendedValue Fortran::lower::CoarrayExprHelper::genAddr(
    const Fortran::evaluate::CoarrayRef &expr) {
  (void)converter;
  (void)symMap;
  (void)loc;
  TODO(converter.genLocation(), "co-array address");
}

fir::ExtendedValue Fortran::lower::CoarrayExprHelper::genValue(
    const Fortran::evaluate::CoarrayRef &expr) {
  TODO(converter.genLocation(), "co-array value");
}
