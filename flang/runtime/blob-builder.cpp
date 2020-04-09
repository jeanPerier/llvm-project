//===-- runtime/blob-builder.cpp --------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "blob-builder.h"

namespace Fortran::runtime {

std::size_t TypedBlobBuilder<const char *>::Measurements(const char *str) {
  return str ? 1 : 0;
}

std::size_t TypedBlobBuilder<const char *>::Measure(
    std::size_t *&p, const char *str) {
  std::size_t content{str ? (*p++ = std::strlen(str) + 1) : 0};
  return RawSizedBlob::ComputeBytesFor(content);
}

void TypedBlobBuilder<const char *>::Fill(std::size_t *&p, const char *str) {
  RawSizedBlob::Fill(str ? *p++ : 0, str);
}
} // namespace Fortran::runtime
