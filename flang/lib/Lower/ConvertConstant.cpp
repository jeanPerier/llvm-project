//===-- ConvertConstant.cpp
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

#include "flang/Lower/ConvertConstant.h"
#include "flang/Lower/ConvertType.h"
#include "flang/Optimizer/Builder/Complex.h"

/// Convert string, \p s, to an APFloat value. Recognize and handle Inf and
/// NaN strings as well. \p s is assumed to not contain any spaces.
static llvm::APFloat consAPFloat(const llvm::fltSemantics &fsem,
                                 llvm::StringRef s) {
  assert(!s.contains(' '));
  if (s.compare_insensitive("-inf") == 0)
    return llvm::APFloat::getInf(fsem, /*negative=*/true);
  if (s.compare_insensitive("inf") == 0 || s.compare_insensitive("+inf") == 0)
    return llvm::APFloat::getInf(fsem);
  // TODO: Add support for quiet and signaling NaNs.
  if (s.compare_insensitive("-nan") == 0)
    return llvm::APFloat::getNaN(fsem, /*negative=*/true);
  if (s.compare_insensitive("nan") == 0 || s.compare_insensitive("+nan") == 0)
    return llvm::APFloat::getNaN(fsem);
  return {fsem, s};
}

/// Generate a real constant with a value `value`.
template <int KIND>
static mlir::Value genRealConstant(fir::FirOpBuilder &builder,
                                   mlir::Location loc,
                                   const llvm::APFloat &value) {
  mlir::Type fltTy = Fortran::lower::convertReal(builder.getContext(), KIND);
  return builder.createRealConstant(loc, fltTy, value);
}

/// Convert a scalar literal constant to IR.
template <Fortran::common::TypeCategory TC, int KIND>
fir::ExtendedValue Fortran::lower::
    ConstantBuilderImpl<Fortran::evaluate::Type<TC, KIND>>::genScalarLit(
        fir::FirOpBuilder &builder, mlir::Location loc,
        const Fortran::evaluate::Scalar<Fortran::evaluate::Type<TC, KIND>>
            &value) {
  if constexpr (TC == Fortran::common::TypeCategory::Integer) {
    mlir::Type ty =
        Fortran::lower::getFIRType(builder.getContext(), TC, KIND, llvm::None);
    if (KIND == 16) {
      auto bigInt =
          llvm::APInt(ty.getIntOrFloatBitWidth(), value.SignedDecimal(), 10);
      return builder.create<mlir::arith::ConstantOp>(
          loc, ty, mlir::IntegerAttr::get(ty, bigInt));
    }
    return builder.createIntegerConstant(loc, ty, value.ToInt64());
  } else if constexpr (TC == Fortran::common::TypeCategory::Logical) {
    return builder.createBool(loc, value.IsTrue());
  } else if constexpr (TC == Fortran::common::TypeCategory::Real) {
    std::string str = value.DumpHexadecimal();
    if constexpr (KIND == 2) {
      auto floatVal = consAPFloat(llvm::APFloatBase::IEEEhalf(), str);
      return genRealConstant<KIND>(builder, loc, floatVal);
    } else if constexpr (KIND == 3) {
      auto floatVal = consAPFloat(llvm::APFloatBase::BFloat(), str);
      return genRealConstant<KIND>(builder, loc, floatVal);
    } else if constexpr (KIND == 4) {
      auto floatVal = consAPFloat(llvm::APFloatBase::IEEEsingle(), str);
      return genRealConstant<KIND>(builder, loc, floatVal);
    } else if constexpr (KIND == 10) {
      auto floatVal = consAPFloat(llvm::APFloatBase::x87DoubleExtended(), str);
      return genRealConstant<KIND>(builder, loc, floatVal);
    } else if constexpr (KIND == 16) {
      auto floatVal = consAPFloat(llvm::APFloatBase::IEEEquad(), str);
      return genRealConstant<KIND>(builder, loc, floatVal);
    } else {
      // convert everything else to double
      auto floatVal = consAPFloat(llvm::APFloatBase::IEEEdouble(), str);
      return genRealConstant<KIND>(builder, loc, floatVal);
    }
  } else if constexpr (TC == Fortran::common::TypeCategory::Complex) {
    using TR =
        Fortran::evaluate::Type<Fortran::common::TypeCategory::Real, KIND>;
    mlir::Value realPart = fir::getBase(
        ConstantBuilderImpl<TR>::genScalarLit(builder, loc, value.REAL()));
    mlir::Value imagPart = fir::getBase(
        ConstantBuilderImpl<TR>::genScalarLit(builder, loc, value.AIMAG()));
    return fir::factory::Complex{builder, loc}.createComplex(KIND, realPart,
                                                             imagPart);
  } else /*constexpr*/ {
    llvm_unreachable("unhandled constant");
  }
}

/// Generate a raw literal value and store it in the rawVals vector.
template <Fortran::common::TypeCategory TC, int KIND>
mlir::Type Fortran::lower::
    ConstantBuilderImpl<Fortran::evaluate::Type<TC, KIND>>::convertToAttribute(
        fir::FirOpBuilder &builder,
        const Fortran::evaluate::Scalar<Fortran::evaluate::Type<TC, KIND>>
            &value,
        llvm::SmallVectorImpl<mlir::Attribute> &outputAttributes) {
  mlir::Attribute val;
  auto attrTc = TC == Fortran::common::TypeCategory::Logical
                    ? Fortran::common::TypeCategory::Integer
                    : TC;
  mlir::Type type = Fortran::lower::getFIRType(builder.getContext(), attrTc,
                                               KIND, llvm::None);
  if constexpr (TC == Fortran::common::TypeCategory::Integer) {
    val = builder.getIntegerAttr(type, value.ToInt64());
  } else if constexpr (TC == Fortran::common::TypeCategory::Logical) {
    val = builder.getIntegerAttr(type, value.IsTrue());
  } else if constexpr (TC == Fortran::common::TypeCategory::Real) {
    std::string str = value.DumpHexadecimal();
    auto floatVal =
        consAPFloat(builder.getKindMap().getFloatSemantics(KIND), str);
    val = builder.getFloatAttr(type, floatVal);
  } else if constexpr (TC == Fortran::common::TypeCategory::Complex) {
    std::string strReal = value.REAL().DumpHexadecimal();
    std::string strImg = value.AIMAG().DumpHexadecimal();
    auto realVal =
        consAPFloat(builder.getKindMap().getFloatSemantics(KIND), strReal);
    outputAttributes.push_back(builder.getFloatAttr(type, realVal));
    auto imgVal =
        consAPFloat(builder.getKindMap().getFloatSemantics(KIND), strImg);
    val = builder.getFloatAttr(type, imgVal);
  }
  outputAttributes.push_back(val);
  return type;
}

template <int KIND>
static fir::StringLitOp
createStringLitOp(fir::FirOpBuilder &builder, mlir::Location loc,
                  const Fortran::evaluate::Scalar<Fortran::evaluate::Type<
                      Fortran::common::TypeCategory::Character, KIND>> &value,
                  [[maybe_unused]] int64_t len) {
  if constexpr (KIND == 1) {
    assert(value.size() == static_cast<std::uint64_t>(len));
    return builder.createStringLitOp(loc, value);
  } else {
    using ET = typename std::decay_t<decltype(value)>::value_type;
    fir::CharacterType type =
        fir::CharacterType::get(builder.getContext(), KIND, len);
    mlir::MLIRContext *context = builder.getContext();
    std::int64_t size = static_cast<std::int64_t>(value.size());
    mlir::ShapedType shape = mlir::RankedTensorType::get(
        llvm::ArrayRef<std::int64_t>{size},
        mlir::IntegerType::get(builder.getContext(), sizeof(ET) * 8));
    auto denseAttr = mlir::DenseElementsAttr::get(
        shape, llvm::ArrayRef<ET>{value.data(), value.size()});
    auto denseTag = mlir::StringAttr::get(context, fir::StringLitOp::xlist());
    mlir::NamedAttribute dataAttr(denseTag, denseAttr);
    auto sizeTag = mlir::StringAttr::get(context, fir::StringLitOp::size());
    mlir::NamedAttribute sizeAttr(sizeTag, builder.getI64IntegerAttr(len));
    llvm::SmallVector<mlir::NamedAttribute> attrs = {dataAttr, sizeAttr};
    return builder.create<fir::StringLitOp>(
        loc, llvm::ArrayRef<mlir::Type>{type}, llvm::None, attrs);
  }
}

/// Convert a scalar literal CHARACTER to IR.
template <int KIND>
fir::ExtendedValue Fortran::lower::ConstantBuilderImpl<
    Fortran::evaluate::Type<Fortran::common::TypeCategory::Character, KIND>>::
    genScalarLit(fir::FirOpBuilder &builder, mlir::Location loc,
                 const Fortran::evaluate::Scalar<Fortran::evaluate::Type<
                     Fortran::common::TypeCategory::Character, KIND>> &value,
                 int64_t len, bool outlineInReadOnlyMemory) {
  if (!outlineInReadOnlyMemory) {
    // When in an initializer context, construct the literal op itself and do
    // not construct another constant object in rodata.
    mlir::Value stringLit = createStringLitOp<KIND>(builder, loc, value, len);
    mlir::Value lenp = builder.createIntegerConstant(
        loc, builder.getCharacterLengthType(), len);
    return fir::CharBoxValue{stringLit, lenp};
  }
  // Otherwise, the string is in a plain old expression so "outline" the value
  // in read only data  by hashconsing it to a constant literal object.
  if constexpr (KIND == 1) {
    // ASCII global constants are created using an mlir string attribute.
    return fir::factory::createStringLiteral(builder, loc, value);
  }

  mlir::Value lenp =
      builder.createIntegerConstant(loc, builder.getCharacterLengthType(), len);

  auto size = builder.getKindMap().getCharacterBitsize(KIND) / 8 * value.size();
  llvm::StringRef strVal(reinterpret_cast<const char *>(value.c_str()), size);
  std::string globalName = fir::factory::uniqueCGIdent("cl", strVal);
  fir::GlobalOp global = builder.getNamedGlobal(globalName);
  fir::CharacterType type =
      fir::CharacterType::get(builder.getContext(), KIND, len);
  if (!global)
    global = builder.createGlobalConstant(
        loc, type, globalName,
        [&](fir::FirOpBuilder &builder) {
          fir::StringLitOp str =
              createStringLitOp<KIND>(builder, loc, value, len);
          builder.create<fir::HasValueOp>(loc, str);
        },
        builder.createLinkOnceLinkage());
  auto addr = builder.create<fir::AddrOfOp>(loc, global.resultType(),
                                            global.getSymbol());
  return fir::CharBoxValue{addr, lenp};
}

using namespace Fortran::evaluate;
FOR_EACH_INTRINSIC_KIND(template class Fortran::lower::ConstantBuilderImpl, )
