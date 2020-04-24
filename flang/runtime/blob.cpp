//===-- runtime/blob.cpp ----------------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "blob.h"

namespace Fortran::runtime {

void RawSizedBlob::Init(std::size_t bytes) {
  RawBlob::template Get<Prefix>().contentBytes = bytes;
}

void RawSizedBlob::Fill(std::size_t bytes, const char *p) {
  Init(bytes);
  std::memcpy(&Get<char>(0), p, bytes);
}

constexpr std::size_t IndexBytesFor(std::size_t elements) {
  return PaddedForAlignment((1 + elements) * sizeof(std::size_t));
}

std::size_t RawIndexedBlob::ComputeBytesFor(
    std::size_t elements, const std::size_t sizes[]) {
  std::size_t cumulative{IndexBytesFor(elements)};
  for (std::size_t j{0}; j < elements; ++j) {
    cumulative += PaddedForAlignment(sizes[j]);
  }
  return RawSizedBlob::ComputeBytesFor(cumulative);
}

void RawIndexedBlob::Init(std::size_t elements, const std::size_t sizes[]) {
  Index &index{RawSizedBlob::template Get<Index>()};
  index.elements = elements;
  std::size_t offset{IndexBytesFor(elements)};
  for (std::size_t j{0}; j < elements; ++j) {
    index.offset[j] = offset;
    offset += PaddedForAlignment(sizes[j]);
  }
  RawSizedBlob::Init(offset);
}

const char *TypedBlob<const char *>::Get() const {
  return contentBytes() ? &RawSizedBlob::Get<const char>() : nullptr;
}
} // namespace Fortran::runtime
