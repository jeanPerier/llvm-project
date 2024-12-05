//===- BuiltinTypeInterfaces.cpp ------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Diagnostics.h"
#include "llvm/ADT/APFloat.h"
#include "llvm/ADT/Sequence.h"

using namespace mlir;
using namespace mlir::detail;

//===----------------------------------------------------------------------===//
/// Tablegen Interface Definitions
//===----------------------------------------------------------------------===//

#include "mlir/IR/BuiltinTypeInterfaces.cpp.inc"

//===----------------------------------------------------------------------===//
// FloatType
//===----------------------------------------------------------------------===//

unsigned FloatType::getWidth() {
  return APFloat::semanticsSizeInBits(getFloatSemantics());
}

FloatType FloatType::scaleElementBitwidth(unsigned scale) {
  if (!scale)
    return FloatType();
  MLIRContext *ctx = getContext();
  if (isF16() || isBF16()) {
    if (scale == 2)
      return FloatType::getF32(ctx);
    if (scale == 4)
      return FloatType::getF64(ctx);
  }
  if (isF32())
    if (scale == 2)
      return FloatType::getF64(ctx);
  return FloatType();
}

unsigned FloatType::getFPMantissaWidth() {
  return APFloat::semanticsPrecision(getFloatSemantics());
}

//===----------------------------------------------------------------------===//
// ShapedType
//===----------------------------------------------------------------------===//

constexpr int64_t ShapedType::kDynamic;

int64_t ShapedType::getNumElements(ArrayRef<int64_t> shape) {
  int64_t num = 1;
  for (int64_t dim : shape) {
    num *= dim;
    assert(num >= 0 && "integer overflow in element count computation");
  }
  return num;
}
