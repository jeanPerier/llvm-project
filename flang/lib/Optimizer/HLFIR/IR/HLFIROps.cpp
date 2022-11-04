//===-- HLFIROps.cpp ------------------------------------------------------===//
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
//
#include "flang/Optimizer/HLFIR/HLFIROps.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/DialectImplementation.h"
#include "mlir/IR/Matchers.h"
#include "mlir/IR/OpImplementation.h"
#include "llvm/ADT/TypeSwitch.h"
#include <tuple>

static mlir::Type getHLFIRVariableTypeFor(mlir::Type inputType,
                                          bool hasExplicitLowerBounds) {
  mlir::Type type = fir::unwrapRefType(inputType);
  if (type.isa<fir::BaseBoxType>())
    return inputType;
  if (auto charType = type.dyn_cast<fir::CharacterType>())
    if (charType.hasDynamicLen())
      return fir::BoxCharType::get(charType.getContext(), charType.getFKind());
  if (fir::hasDynamicSize(type) || hasExplicitLowerBounds)
    return fir::BoxType::get(type);
  return inputType;
}

//===----------------------------------------------------------------------===//
// DeclareOp
//===----------------------------------------------------------------------===//

void hlfir::DeclareOp::build(mlir::OpBuilder &builder,
                             mlir::OperationState &result, mlir::Value memref,
                             llvm::StringRef uniq_name, mlir::Value shape,
                             mlir::ValueRange typeparams,
                             fir::FortranVariableFlagsAttr fortran_attrs) {
  auto nameAttr = builder.getStringAttr(uniq_name);
  mlir::Type inputType = memref.getType();
  bool hasExplicitLowerBounds =
      shape && shape.getType().isa<fir::ShapeShiftType, fir::ShiftType>();
  mlir::Type hlfirVariableType =
      getHLFIRVariableTypeFor(inputType, hasExplicitLowerBounds);
  build(builder, result, {hlfirVariableType, inputType}, memref, shape,
        typeparams, nameAttr, fortran_attrs);
}

//===----------------------------------------------------------------------===//
// DesignateOp
//===----------------------------------------------------------------------===//

void hlfir::DesignateOp::build(
    mlir::OpBuilder &builder, mlir::OperationState &result,
    mlir::Type result_type, mlir::Value memref, llvm::StringRef component,
    mlir::Value component_shape, llvm::ArrayRef<Subscript> subscripts,
    mlir::ValueRange substring, llvm::Optional<bool> complex_part,
    mlir::Value shape, mlir::ValueRange typeparams,
    fir::FortranVariableFlagsAttr fortran_attrs) {
  auto componentAttr =
      component.empty() ? mlir::StringAttr{} : builder.getStringAttr(component);
  llvm::SmallVector<mlir::Value> indices;
  llvm::SmallVector<bool> isTriplet;
  for (auto subscript : subscripts) {
    if (auto *triplet = std::get_if<Triplet>(&subscript)) {
      isTriplet.push_back(true);
      indices.push_back(std::get<0>(*triplet));
      indices.push_back(std::get<1>(*triplet));
      indices.push_back(std::get<2>(*triplet));
    } else {
      isTriplet.push_back(false);
      indices.push_back(std::get<mlir::Value>(subscript));
    }
  }
  auto isTripletAttr =
      mlir::DenseBoolArrayAttr::get(builder.getContext(), isTriplet);
  auto complexPartAttr =
      complex_part.has_value()
          ? mlir::BoolAttr::get(builder.getContext(), *complex_part)
          : mlir::BoolAttr{};
  build(builder, result, result_type, memref, componentAttr, component_shape,
        indices, isTripletAttr, substring, complexPartAttr, shape, typeparams,
        fortran_attrs);
}

static mlir::ParseResult parseDesignatorIndices(
    mlir::OpAsmParser &parser,
    llvm::SmallVectorImpl<mlir::OpAsmParser::UnresolvedOperand> &indices,
    mlir::DenseBoolArrayAttr &isTripletAttr) {
  if (mlir::succeeded(parser.parseOptionalLParen())) {
    llvm::SmallVector<bool> isTriplet;
    do {
      mlir::OpAsmParser::UnresolvedOperand i1, i2, i3;
      if (parser.parseOperand(i1))
        return mlir::failure();
      indices.push_back(i1);
      if (mlir::succeeded(parser.parseOptionalColon())) {
        if (parser.parseOperand(i2) || parser.parseColon() ||
            parser.parseOperand(i3))
          return mlir::failure();
        indices.push_back(i2);
        indices.push_back(i3);
        isTriplet.push_back(true);
      } else {
        isTriplet.push_back(false);
      }
    } while (mlir::succeeded(parser.parseOptionalComma()));
    if (parser.parseRParen())
      return mlir::failure();
    isTripletAttr =
        mlir::DenseBoolArrayAttr::get(parser.getContext(), isTriplet);
  }
  return mlir::success();
}

static void
printDesignatorIndices(mlir::OpAsmPrinter &p, hlfir::DesignateOp designateOp,
                       mlir::OperandRange indices,
                       const mlir::DenseBoolArrayAttr &isTripletAttr) {
  if (!indices.empty()) {
    p << '(';
    unsigned i = 0;
    for (auto isTriplet : isTripletAttr.asArrayRef()) {
      if (isTriplet) {
        assert(i + 2 < indices.size() && "ill-formed indices");
        p << indices[i] << ":" << indices[i + 1] << ":" << indices[i + 2];
        i += 3;
      } else {
        p << indices[i++];
      }
      if (i != indices.size())
        p << ", ";
    }
    p << ')';
  }
}

static mlir::ParseResult
parseDesignatorComplexPart(mlir::OpAsmParser &parser,
                           mlir::BoolAttr &complexPart) {
  if (mlir::succeeded(parser.parseOptionalKeyword("imag")))
    complexPart = mlir::BoolAttr::get(parser.getContext(), true);
  else if (mlir::succeeded(parser.parseOptionalKeyword("real")))
    complexPart = mlir::BoolAttr::get(parser.getContext(), false);
  return mlir::success();
}

static void printDesignatorComplexPart(mlir::OpAsmPrinter &p,
                                       hlfir::DesignateOp designateOp,
                                       mlir::BoolAttr complexPartAttr) {
  if (complexPartAttr) {
    if (complexPartAttr.getValue())
      p << "imag";
    else
      p << "real";
  }
}

mlir::LogicalResult hlfir::DesignateOp::verify() {
  // TODO.
  return mlir::success();
}

#define GET_OP_CLASSES
#include "flang/Optimizer/HLFIR/HLFIROps.cpp.inc"
