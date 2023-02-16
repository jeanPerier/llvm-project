//===-- include/flang/Runtime/array-constructor.h ---------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// External APIs to create temporary array constructor when the final extents or
// length parameters cannot be pre-computed.

#ifndef FORTRAN_RUNTIME_ARRAYCONSTRUCTOR_H_
#define FORTRAN_RUNTIME_ARRAYCONSTRUCTOR_H_

#include "flang/Runtime/entry-names.h"
#include "flang/Runtime/runtime-struct-declare.h"
#include <cstdint>

namespace Fortran::runtime {
class Descriptor;

#define ARRAY_CONSTRUCTOR_TEMP_STRUCT(STRUCT_NAME, MEMBER) \
  STRUCT_NAME(ArrayConstructorTemporary) \
  MEMBER(Descriptor*, descriptor) \
  MEMBER(std::int64_t, nextValuePosition) \
  MEMBER(std::int64_t, allocationSize)

FLANG_DECLARE_RUNTIME_STRUCT(ARRAY_CONSTRUCTOR_TEMP_STRUCT)

extern "C" {
void RTNAME(PushArrayConstructorValue)(ArrayConstructorTemporary& to, const Descriptor &value,
    const char *sourceFile = nullptr, int sourceLine = 0);
} // extern "C"
} // namespace Fortran::runtime
#endif // FORTRAN_RUNTIME_ARRAYCONSTRUCTOR_H_
