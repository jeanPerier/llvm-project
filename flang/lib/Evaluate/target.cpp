//===-- lib/Semantics/target.cpp ------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "flang/Evaluate/target.h"
#include "flang/Evaluate/common.h"
#include "flang/Evaluate/type.h"

namespace Fortran::evaluate {

Rounding TargetCharacteristics::defaultRounding;

TargetCharacteristics::TargetCharacteristics() {
  // TODO: Fill in the type information from command-line targeting information.
  auto enableCategoryKinds{[this](TypeCategory category) {
    for (int kind{0}; kind < maxKind; ++kind) {
      if (CanSupportType(category, kind)) {
        auto byteSize{static_cast<std::size_t>(kind)};
        if (category == TypeCategory::Real ||
            category == TypeCategory::Complex) {
          if (kind == 3) {
            // non-IEEE 16-bit format (truncated 32-bit)
            byteSize = 2;
          } else if (kind == 10) {
            // x87 floating-point -- follow gcc precedent for "long double"
            byteSize = 16;
          }
        }
        std::size_t align{byteSize};
        if (category == TypeCategory::Complex) {
          byteSize = 2 * byteSize;
        }
        EnableType(category, kind, byteSize, align);
      }
    }
  }};
  enableCategoryKinds(TypeCategory::Integer);
  enableCategoryKinds(TypeCategory::Real);
  enableCategoryKinds(TypeCategory::Complex);
  enableCategoryKinds(TypeCategory::Character);
  enableCategoryKinds(TypeCategory::Logical);

  isBigEndian_ = !isHostLittleEndian;

  areSubnormalsFlushedToZero_ = false;
}

bool TargetCharacteristics::CanSupportType(
    TypeCategory category, std::int64_t kind) {
#if !__x86_64__
  if ((category == TypeCategory::Real || category == TypeCategory::Complex) &&
      kind == 10) {
    return false;
  }
#endif
  return IsValidKindOfIntrinsicType(category, kind);
}

bool TargetCharacteristics::EnableType(common::TypeCategory category,
    std::int64_t kind, std::size_t byteSize, std::size_t align) {
  if (CanSupportType(category, kind)) {
    byteSize_[static_cast<int>(category)][kind] = byteSize;
    align_[static_cast<int>(category)][kind] = align;
    maxByteSize_ = std::max(maxByteSize_, byteSize);
    maxAlignment_ = std::max(maxAlignment_, align);
    return true;
  } else {
    return false;
  }
}

void TargetCharacteristics::DisableType(
    common::TypeCategory category, std::int64_t kind) {
  if (kind >= 0 && kind < maxKind) {
    align_[static_cast<int>(category)][kind] = 0;
  }
}

std::size_t TargetCharacteristics::GetByteSize(
    common::TypeCategory category, std::int64_t kind) const {
  if (kind >= 0 && kind < maxKind) {
    return byteSize_[static_cast<int>(category)][kind];
  } else {
    return 0;
  }
}

std::size_t TargetCharacteristics::GetAlignment(
    common::TypeCategory category, std::int64_t kind) const {
  if (kind >= 0 && kind < maxKind) {
    return align_[static_cast<int>(category)][kind];
  } else {
    return 0;
  }
}

bool TargetCharacteristics::IsTypeEnabled(
    common::TypeCategory category, std::int64_t kind) const {
  return GetAlignment(category, kind) > 0;
}

void TargetCharacteristics::set_isBigEndian(bool isBig) {
  isBigEndian_ = isBig;
}

void TargetCharacteristics::set_areSubnormalsFlushedToZero(bool yes) {
  areSubnormalsFlushedToZero_ = yes;
}

void TargetCharacteristics::set_roundingMode(Rounding rounding) {
  roundingMode_ = rounding;
}

} // namespace Fortran::evaluate
