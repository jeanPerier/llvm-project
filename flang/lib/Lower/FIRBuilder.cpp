//===-- FIRBuilder.cpp ----------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "flang/Lower/FIRBuilder.h"
#include "SymbolMap.h"
#include "flang/Lower/Bridge.h"
#include "flang/Lower/CharacterExpr.h"
#include "flang/Lower/ComplexExpr.h"
#include "flang/Lower/ConvertType.h"
#include "flang/Lower/Support/BoxValue.h"
#include "flang/Optimizer/Dialect/FIRAttr.h"
#include "flang/Optimizer/Dialect/FIROpsSupport.h"
#include "flang/Optimizer/Support/FatalError.h"
#include "flang/Optimizer/Support/InternalNames.h"
#include "flang/Semantics/symbol.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/MD5.h"

static llvm::cl::opt<std::size_t>
    nameLengthHashSize("length-to-hash-string-literal",
                       llvm::cl::desc("string literals that exceed this length"
                                      " will use a hash value as their symbol "
                                      "name"),
                       llvm::cl::init(32));

mlir::FuncOp Fortran::lower::FirOpBuilder::createFunction(
    mlir::Location loc, mlir::ModuleOp module, llvm::StringRef name,
    mlir::FunctionType ty) {
  return fir::createFuncOp(loc, module, name, ty);
}

mlir::FuncOp
Fortran::lower::FirOpBuilder::getNamedFunction(mlir::ModuleOp modOp,
                                               llvm::StringRef name) {
  return modOp.lookupSymbol<mlir::FuncOp>(name);
}

fir::GlobalOp
Fortran::lower::FirOpBuilder::getNamedGlobal(mlir::ModuleOp modOp,
                                             llvm::StringRef name) {
  return modOp.lookupSymbol<fir::GlobalOp>(name);
}

mlir::Type Fortran::lower::FirOpBuilder::getRefType(mlir::Type eleTy) {
  assert(!eleTy.isa<fir::ReferenceType>() && "cannot be a reference type");
  return fir::ReferenceType::get(eleTy);
}

mlir::Type Fortran::lower::FirOpBuilder::getVarLenSeqTy(mlir::Type eleTy) {
  fir::SequenceType::Shape shape = {fir::SequenceType::getUnknownExtent()};
  return fir::SequenceType::get(shape, eleTy);
}

mlir::Value
Fortran::lower::FirOpBuilder::createNullConstant(mlir::Location loc,
                                                 mlir::Type ptrType) {
  auto indexType = getIndexType();
  auto zero = createIntegerConstant(loc, indexType, 0);
  if (!ptrType)
    ptrType = getRefType(getNoneType());
  return createConvert(loc, ptrType, zero);
}

mlir::Value Fortran::lower::FirOpBuilder::createIntegerConstant(
    mlir::Location loc, mlir::Type ty, std::int64_t cst) {
  return create<mlir::ConstantOp>(loc, ty, getIntegerAttr(ty, cst));
}

mlir::Value Fortran::lower::FirOpBuilder::createRealConstant(
    mlir::Location loc, mlir::Type fltTy, llvm::APFloat::integerPart val) {
  auto apf = [&]() -> llvm::APFloat {
    if (auto ty = fltTy.dyn_cast<fir::RealType>()) {
      fltTy = Fortran::lower::convertReal(ty.getContext(), ty.getFKind());
      return llvm::APFloat(kindMap.getFloatSemantics(ty.getFKind()), val);
    } else {
      if (fltTy.isF16())
        return llvm::APFloat(llvm::APFloat::IEEEhalf(), val);
      if (fltTy.isBF16())
        return llvm::APFloat(llvm::APFloat::BFloat(), val);
      if (fltTy.isF32())
        return llvm::APFloat(llvm::APFloat::IEEEsingle(), val);
      if (fltTy.isF64())
        return llvm::APFloat(llvm::APFloat::IEEEdouble(), val);
      if (fltTy.isF80())
        return llvm::APFloat(llvm::APFloat::x87DoubleExtended(), val);
      if (fltTy.isF128())
        return llvm::APFloat(llvm::APFloat::IEEEquad(), val);
      llvm_unreachable("unhandled MLIR floating-point type");
    }
  };
  return createRealConstant(loc, fltTy, apf());
}

mlir::Value Fortran::lower::FirOpBuilder::createRealConstant(
    mlir::Location loc, mlir::Type fltTy, const llvm::APFloat &value) {
  if (fltTy.isa<mlir::FloatType>()) {
    auto attr = getFloatAttr(fltTy, value);
    return create<mlir::ConstantOp>(loc, fltTy, attr);
  }
  llvm_unreachable("should use builtin floating-point type");
}

mlir::Value Fortran::lower::FirOpBuilder::allocateLocal(
    mlir::Location loc, mlir::Type ty, llvm::StringRef nm,
    llvm::ArrayRef<mlir::Value> shape, llvm::ArrayRef<mlir::Value> lenParams,
    bool asTarget) {
  llvm::SmallVector<mlir::Value, 8> indices;
  auto idxTy = getIndexType();
  // FIXME: AllocaOp has a lenParams argument, but it is ignored, so add lengths
  // into the index so far (for characters, that works OK).
  llvm::for_each(lenParams, [&](mlir::Value sh) {
    indices.push_back(createConvert(loc, idxTy, sh));
  });
  llvm::for_each(shape, [&](mlir::Value sh) {
    indices.push_back(createConvert(loc, idxTy, sh));
  });
  llvm::SmallVector<mlir::NamedAttribute, 2> attrs;
  if (asTarget)
    attrs.emplace_back(mlir::Identifier::get("target", getContext()),
                       getUnitAttr());
  return create<fir::AllocaOp>(loc, ty, nm, llvm::None, indices, attrs);
}

/// Create a temporary variable on the stack. Anonymous temporaries have no
/// `name` value.
mlir::Value Fortran::lower::FirOpBuilder::createTemporary(
    mlir::Location loc, mlir::Type type, llvm::StringRef name,
    llvm::ArrayRef<mlir::Value> shape) {
  auto insPt = saveInsertionPoint();
  if (shape.empty())
    setInsertionPointToStart(getEntryBlock());
  else
    setInsertionPointAfter(shape.back().getDefiningOp());
  assert(!type.isa<fir::ReferenceType>() && "cannot be a reference");
  auto ae = create<fir::AllocaOp>(loc, type, name, llvm::None, shape);
  restoreInsertionPoint(insPt);
  return ae;
}

/// Create a global variable in the (read-only) data section. A global variable
/// must have a unique name to identify and reference it.
fir::GlobalOp Fortran::lower::FirOpBuilder::createGlobal(
    mlir::Location loc, mlir::Type type, llvm::StringRef name,
    mlir::StringAttr linkage, mlir::Attribute value, bool isConst) {
  auto module = getModule();
  auto insertPt = saveInsertionPoint();
  if (auto glob = module.lookupSymbol<fir::GlobalOp>(name))
    return glob;
  setInsertionPoint(module.getBody()->getTerminator());
  auto glob = create<fir::GlobalOp>(loc, name, isConst, type, value, linkage);
  restoreInsertionPoint(insertPt);
  return glob;
}

fir::GlobalOp Fortran::lower::FirOpBuilder::createGlobal(
    mlir::Location loc, mlir::Type type, llvm::StringRef name, bool isConst,
    std::function<void(FirOpBuilder &)> bodyBuilder, mlir::StringAttr linkage) {
  auto module = getModule();
  auto insertPt = saveInsertionPoint();
  if (auto glob = module.lookupSymbol<fir::GlobalOp>(name))
    return glob;
  setInsertionPoint(module.getBody()->getTerminator());
  auto glob = create<fir::GlobalOp>(loc, name, isConst, type, mlir::Attribute{},
                                    linkage);
  auto &region = glob.getRegion();
  region.push_back(new mlir::Block);
  auto &block = glob.getRegion().back();
  setInsertionPointToStart(&block);
  bodyBuilder(*this);
  restoreInsertionPoint(insertPt);
  return glob;
}

mlir::Value Fortran::lower::FirOpBuilder::convertWithSemantics(
    mlir::Location loc, mlir::Type toTy, mlir::Value val) {
  assert(toTy && "store location must be typed");
  auto fromTy = val.getType();
  if (fromTy == toTy)
    return val;
  ComplexExprHelper helper{*this, loc};
  if ((fir::isa_real(fromTy) || fir::isa_integer(fromTy)) &&
      fir::isa_complex(toTy)) {
    // imaginary part is zero
    auto eleTy = helper.getComplexPartType(toTy);
    auto cast = createConvert(loc, eleTy, val);
    llvm::APFloat zero{
        kindMap.getFloatSemantics(toTy.cast<fir::ComplexType>().getFKind()), 0};
    auto imag = createRealConstant(loc, eleTy, zero);
    return helper.createComplex(toTy, cast, imag);
  }
  if (fir::isa_complex(fromTy) &&
      (fir::isa_integer(toTy) || fir::isa_real(toTy))) {
    // drop the imaginary part
    auto rp = helper.extractComplexPart(val, /*isImagPart=*/false);
    return createConvert(loc, toTy, rp);
  }
  return createConvert(loc, toTy, val);
}

mlir::Value Fortran::lower::FirOpBuilder::createConvert(mlir::Location loc,
                                                        mlir::Type toTy,
                                                        mlir::Value val) {
  if (val.getType() != toTy)
    return create<fir::ConvertOp>(loc, toTy, val);
  return val;
}

fir::StringLitOp
Fortran::lower::FirOpBuilder::createStringLitOp(mlir::Location loc,
                                                llvm::StringRef data) {
  auto type = fir::CharacterType::get(getContext(), 1, data.size());
  auto strAttr = mlir::StringAttr::get(getContext(), data);
  auto valTag = mlir::Identifier::get(fir::StringLitOp::value(), getContext());
  mlir::NamedAttribute dataAttr(valTag, strAttr);
  auto sizeTag = mlir::Identifier::get(fir::StringLitOp::size(), getContext());
  mlir::NamedAttribute sizeAttr(sizeTag, getI64IntegerAttr(data.size()));
  llvm::SmallVector<mlir::NamedAttribute, 2> attrs{dataAttr, sizeAttr};
  return create<fir::StringLitOp>(loc, llvm::ArrayRef<mlir::Type>{type},
                                  llvm::None, attrs);
}

mlir::Value
Fortran::lower::FirOpBuilder::consShape(mlir::Location loc,
                                        const fir::AbstractArrayBox &arr) {
  if (arr.lboundsAllOne()) {
    auto shapeType = fir::ShapeType::get(getContext(), arr.getExtents().size());
    return create<fir::ShapeOp>(loc, shapeType, arr.getExtents());
  }
  auto shapeType =
      fir::ShapeShiftType::get(getContext(), arr.getExtents().size());
  SmallVector<mlir::Value, 8> shapeArgs;
  auto idxTy = getIndexType();
  for (auto [lbnd, ext] : llvm::zip(arr.getLBounds(), arr.getExtents())) {
    auto lb = createConvert(loc, idxTy, lbnd);
    shapeArgs.push_back(lb);
    shapeArgs.push_back(ext);
  }
  return create<fir::ShapeShiftOp>(loc, shapeType, shapeArgs);
}

mlir::Value
Fortran::lower::FirOpBuilder::createShape(mlir::Location loc,
                                          const fir::ExtendedValue &exv) {
  return exv.match(
      [&](const fir::ArrayBoxValue &box) { return consShape(loc, box); },
      [&](const fir::CharArrayBoxValue &box) { return consShape(loc, box); },
      [&](const fir::BoxValue &box) { return consShape(loc, box); },
      [&](auto) -> mlir::Value { fir::emitFatalError(loc, "not an array"); });
}

mlir::Value Fortran::lower::FirOpBuilder::createSlice(
    mlir::Location loc, const fir::ExtendedValue &exv, mlir::ValueRange triples,
    mlir::ValueRange path) {
  if (triples.empty()) {
    // If there is no slicing by triple notation, then take the whole array.
    auto fullShape = [&](const fir::AbstractArrayBox &arr) -> mlir::Value {
      llvm::SmallVector<mlir::Value, 8> trips;
      auto idxTy = getIndexType();
      auto one = createIntegerConstant(loc, idxTy, 1);
      auto sliceTy = fir::SliceType::get(getContext(), arr.rank());
      if (arr.lboundsAllOne()) {
        for (auto v : arr.getExtents()) {
          trips.push_back(one);
          trips.push_back(v);
          trips.push_back(one);
        }
        return create<fir::SliceOp>(loc, sliceTy, trips, path);
      }
      for (auto [lbnd, ext] : llvm::zip(arr.getLBounds(), arr.getExtents())) {
        auto lb = createConvert(loc, idxTy, lbnd);
        trips.push_back(lb);
        trips.push_back(ext);
        trips.push_back(one);
      }
      return create<fir::SliceOp>(loc, sliceTy, trips, path);
    };
    return exv.match(
        [&](const fir::ArrayBoxValue &box) { return fullShape(box); },
        [&](const fir::CharArrayBoxValue &box) { return fullShape(box); },
        [&](const fir::BoxValue &box) { return fullShape(box); },
        [&](auto) -> mlir::Value { fir::emitFatalError(loc, "not an array"); });
  }
  auto sf = [&](const fir::AbstractArrayBox &arr) -> mlir::Value {
    auto sliceTy = fir::SliceType::get(getContext(), arr.rank());
    return create<fir::SliceOp>(loc, sliceTy, triples, path);
  };
  return exv.match(
      [&](const fir::ArrayBoxValue &box) { return sf(box); },
      [&](const fir::CharArrayBoxValue &box) { return sf(box); },
      [&](const fir::BoxValue &box) { return sf(box); },
      [&](auto) -> mlir::Value { fir::emitFatalError(loc, "not an array"); });
}

mlir::Value
Fortran::lower::FirOpBuilder::createBox(mlir::Location loc,
                                        const fir::ExtendedValue &exv) {
  auto itemAddr = fir::getBase(exv);
  if (itemAddr.getType().isa<fir::BoxType>())
    return itemAddr;
  auto elementType = fir::dyn_cast_ptrEleTy(itemAddr.getType());
  if (!elementType)
    mlir::emitError(loc, "internal: expected a memory reference type ")
        << itemAddr.getType();
  auto boxTy = fir::BoxType::get(elementType);
  return exv.match(
      [&](const fir::ArrayBoxValue &box) -> mlir::Value {
        auto s = createShape(loc, exv);
        return create<fir::EmboxOp>(loc, boxTy, itemAddr, s);
      },
      [&](const fir::CharArrayBoxValue &box) -> mlir::Value {
        auto s = createShape(loc, exv);
        if (Fortran::lower::CharacterExprHelper::hasConstantLengthInType(exv))
          return create<fir::EmboxOp>(loc, boxTy, itemAddr, s);

        mlir::Value emptySlice;
        llvm::SmallVector<mlir::Value, 1> lenParams{box.getLen()};
        return create<fir::EmboxOp>(loc, boxTy, itemAddr, s, emptySlice,
                                    lenParams);
      },
      [&](const fir::BoxValue &box) -> mlir::Value {
        auto s = createShape(loc, exv);
        return create<fir::EmboxOp>(loc, boxTy, itemAddr, s);
      },
      [&](const fir::CharBoxValue &box) -> mlir::Value {
        if (Fortran::lower::CharacterExprHelper::hasConstantLengthInType(exv))
          return create<fir::EmboxOp>(loc, boxTy, itemAddr);
        mlir::Value emptyShape, emptySlice;
        llvm::SmallVector<mlir::Value, 1> lenParams{box.getLen()};
        return create<fir::EmboxOp>(loc, boxTy, itemAddr, emptyShape,
                                    emptySlice, lenParams);
      },
      [&](const auto &) -> mlir::Value {
        return create<fir::EmboxOp>(loc, boxTy, itemAddr);
      });
}

std::string Fortran::lower::FirOpBuilder::uniqueCGIdent(llvm::StringRef prefix,
                                                        llvm::StringRef name) {
  // For "long" identifiers use a hash value
  if (name.size() > nameLengthHashSize) {
    llvm::MD5 hash;
    hash.update(name);
    llvm::MD5::MD5Result result;
    hash.final(result);
    llvm::SmallString<32> str;
    llvm::MD5::stringifyResult(result, str);
    std::string hashName = prefix.str();
    hashName.append(".").append(str.c_str());
    return fir::NameUniquer::doGenerated(hashName);
  }
  // "Short" identifiers use a reversible hex string
  std::string nm = prefix.str();
  return fir::NameUniquer::doGenerated(
      nm.append(".").append(llvm::toHex(name)));
}

mlir::Value
Fortran::lower::FirOpBuilder::locationToFilename(mlir::Location loc) {
  if (auto flc = loc.dyn_cast<mlir::FileLineColLoc>()) {
    // must be encoded as asciiz, C string
    auto fn = flc.getFilename().str() + '\0';
    return fir::getBase(createStringLiteral(loc, fn));
  }
  return createNullConstant(loc);
}

mlir::Value Fortran::lower::FirOpBuilder::locationToLineNo(mlir::Location loc,
                                                           mlir::Type type) {
  if (auto flc = loc.dyn_cast<mlir::FileLineColLoc>())
    return createIntegerConstant(loc, type, flc.getLine());
  return createIntegerConstant(loc, type, 0);
}

fir::ExtendedValue
Fortran::lower::FirOpBuilder::createStringLiteral(mlir::Location loc,
                                                  llvm::StringRef str) {
  std::string globalName = uniqueCGIdent("cl", str);
  auto type = fir::CharacterType::get(getContext(), 1, str.size());
  auto global = getNamedGlobal(globalName);
  if (!global)
    global = createGlobalConstant(
        loc, type, globalName,
        [&](Fortran::lower::FirOpBuilder &builder) {
          auto stringLitOp = builder.createStringLitOp(loc, str);
          builder.create<fir::HasValueOp>(loc, stringLitOp);
        },
        createLinkOnceLinkage());
  auto addr =
      create<fir::AddrOfOp>(loc, global.resultType(), global.getSymbol());
  auto len = createIntegerConstant(loc, getCharacterLengthType(), str.size());
  return fir::CharBoxValue{addr, len};
}
