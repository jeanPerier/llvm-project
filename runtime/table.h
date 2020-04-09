//===-- runtime/table.h -----------------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef FORTRAN_RUNTIME_TABLE_H_
#define FORTRAN_RUNTIME_TABLE_H_

#include <cstddef>

// Utility framework for constructing a collection of related tables
// at compilation time (or in a unit test) that are packed together
// in such a way as to facilitate their passage through code generation
// as an untyped blob of bits, avoid the need to relocate interior pointers
// at link time, and enable immediate fast random access to their contents
// in read-only storage in the runtime support library without preparatory
// indexing or unpacking or modification.
// The important use cases are expected to be derived type descriptions
// and NAMELIST groups.

namespace Fortran::runtime {

class alignas(8) TableEntry {
public:
  std::size_t totalBytes() const {
    return staic_cast<std::size_t>(totalBytes_);
  }
  std::size_t ContentBytes() const { return totalBytes_ - sizeof totalBytes_; }
  template <typename A> const A &at(int n) const {
    return *reinterpret_cast<const A *>(raw_ + n);
  }

private:
  int totalBytes_;
  char raw_[1];
};

class alignas(8) Table : public TableEntry {
public:
  int entries() const { return at<int>(offsetof(Schema, entries)); }
  const TableEntry &entry(int n) { return at<TableEntry>(entryOffset()[n]); }

private:
  const int *entryOffset() const {
    return &at<int>(offsetof(Schema, entryOffset));
  }
  struct Schema { // for offsetof()
    int entries;
    int entryOffset[1];
  };
};
} // namespace Fortran::runtime

#endif // FORTRAN_RUNTIME_TABLE_H_
