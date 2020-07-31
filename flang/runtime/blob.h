//===-- runtime/blob.h ------------------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef FORTRAN_RUNTIME_BLOB_H_
#define FORTRAN_RUNTIME_BLOB_H_

// Utility framework for constructing a collection of related tables
// at compilation time (or in a unit test) that are packed together
// in such a way as to facilitate their passage through code generation
// as an untyped blob of bits, avoid the need to relocate interior data
// pointers at link time, and enable immediate fast random access to their
// contents in read-only storage in the runtime support library without
// preparatory indexing or unpacking or other modification.
// The important use cases are expected to be derived type descriptions
// and NAMELIST groups.
//
// This header defines three classes of "raw" blobs: one with no metadata
// at all, one with a size, and an indexed one that provides random access
// to constituents, which need not have the same size.
//
// The raw blobs are used to implement various specializations of the
// TypedBlob<> template class.  Typed blobs support plain old data,
// NUL-terminated C-style strings, and the STL templates unique_ptr<>,
// optional<>, tuple<>, and vector<>, where the base type(s) of the template
// must also be supported as typed blobs.
//
// See blob-builder.h for CreateBlob(), which builds TypedBlob<> instances
// directly from instances of the supported types.
//
// Blobs don't supply versioning on their own; take care to pass and check
// version codes in the payload data.

#include "memory.h"
#include "terminator.h"
#include "flang/Common/template.h"
#include <cstddef>
#include <cstring>
#include <memory>
#include <optional>
#include <tuple>
#include <vector>

namespace Fortran::runtime {

static constexpr std::size_t alignment{sizeof(std::int64_t)};

// Rounds the size of an object up to a multiple of the alignment.
constexpr std::size_t PaddedForAlignment(std::size_t bytes) {
  return (bytes + alignment - 1) & -alignment;
}

// Raw blob classes

class alignas(alignment) RawBlob {
public:
  template <typename A> const A &Get(std::size_t n = 0) const {
    return *reinterpret_cast<const A *>(raw_ + n);
  }
  template <typename A> A &Get(std::size_t n = 0) {
    return *reinterpret_cast<A *>(raw_ + n);
  }

private:
  char raw_[1];
};

class RawSizedBlob : public RawBlob {
public:
  void Init(std::size_t);
  void Fill(std::size_t, const char *);
  std::size_t contentBytes() const {
    return RawBlob::Get<Prefix>().contentBytes;
  }
  static constexpr std::size_t ComputeBytesFor(std::size_t n) {
    return PaddedForAlignment(n + prefixBytes);
  }

  template <typename A> const A &Get(std::size_t n = 0) const {
    return RawBlob::template Get<A>(n + prefixBytes);
  }
  template <typename A> A &Get(std::size_t n = 0) {
    return RawBlob::template Get<A>(n + prefixBytes);
  }

private:
  struct alignas(alignment) Prefix {
    std::size_t contentBytes;
  };
  static constexpr std::size_t prefixBytes{PaddedForAlignment(sizeof(Prefix))};
};

class RawIndexedBlob : public RawSizedBlob {
public:
  void Init(std::size_t elements, const std::size_t[]);
  std::size_t Elements() const { return GetIndex().elements; }
  template <typename A = RawBlob> const A &Get(std::size_t j) const {
    return RawSizedBlob::Get<const A>(ItemOffset(j));
  }
  template <typename A = RawBlob> A &Get(std::size_t j) {
    return RawSizedBlob::Get<A>(ItemOffset(j));
  }

protected:
  static std::size_t ComputeBytesFor(std::size_t elements, const std::size_t[]);

private:
  struct Index {
    std::size_t elements;
    std::size_t offset[1]; // overindexed; offsets are from start of Index
  };

  const Index &GetIndex() const {
    return RawSizedBlob::template Get<const Index>();
  }

  std::size_t ItemOffset(std::size_t j) const { return GetIndex().offset[j]; }
};

// Typed blob classes

// The default typed blob is for fixed-size plain old data.
template <typename A> class TypedBlob : public RawBlob {
public:
  static_assert(std::is_pod_v<A>);
  const A &Get() const { return RawBlob::template Get<const A>(); }
};

// Specializations of TypedBlob follow

template <> class TypedBlob<const char *> : public RawSizedBlob {
public:
  const char *Get() const;
};

// Forward declarations of specializations for mutual reference
template <typename A> class TypedBlob<std::unique_ptr<A>>;
template <typename A> class TypedBlob<std::optional<A>>;
template <typename A> class TypedBlob<std::vector<A>>;

template <typename A> struct TypedBlobValue { using type = const A &; };

template <typename A>
using TypedBlobValueType = typename TypedBlobValue<A>::type;

template <> struct TypedBlobValue<const char *> { using type = const char *; };
template <typename... As> struct TypedBlobValue<std::tuple<As...>> {
  using type =
      common::MapTemplate<TypedBlobValueType, std::tuple<As...>, std::tuple>;
};
template <typename A> struct TypedBlobValue<std::unique_ptr<A>> {
  using type = const TypedBlob<std::unique_ptr<A>> &;
};
template <typename A> struct TypedBlobValue<std::optional<A>> {
  using type = const TypedBlob<std::optional<A>> &;
};
template <typename A> struct TypedBlobValue<std::vector<A>> {
  using type = const TypedBlob<std::vector<A>> &;
};

template <typename A>
class TypedBlob<std::unique_ptr<A>> : public RawIndexedBlob {
public:
  using ElementValue = TypedBlobValueType<A>;
  operator bool() const { return Elements() > 0; }
  ElementValue operator*() const {
    return RawIndexedBlob::template Get<TypedBlob<A>>(0).Get();
  }
  const TypedBlob &Get() const { return *this; }
  const TypedBlob<A> &GetElementTypedBlob() const {
    return RawIndexedBlob::template Get<TypedBlob<A>>(0);
  }
};

template <typename A>
class TypedBlob<std::optional<A>> : public RawIndexedBlob {
public:
  using ElementValue = TypedBlobValueType<A>;
  bool has_value() const { return Elements() > 0; }
  operator bool() const { return has_value(); }
  ElementValue operator*() const {
    return RawIndexedBlob::template Get<TypedBlob<A>>(0).Get();
  }
  ElementValue value() const { return **this; }
  ElementValue value_or(A x) const { return *this ? **this : x; }

  // Because TypedBlob<std::optional<>> supports most of the API
  // of std::optional<>, it can be its own entire value.
  const TypedBlob &Get() const { return *this; }

  const TypedBlob<A> &GetElementTypedBlob() const {
    return RawIndexedBlob::template Get<TypedBlob<A>>(0);
  }
};

template <typename... As>
class TypedBlob<std::tuple<As...>> : public RawIndexedBlob {
public:
  using Tuple = std::tuple<As...>;
  template <std::size_t J> using ElementType = std::tuple_element_t<J, Tuple>;
  using ValueType = TypedBlobValueType<Tuple>;
  template <std::size_t J>
  using ElementValueType = std::tuple_element_t<J, ValueType>;
  ValueType Get() const { return GetHelper(std::index_sequence_for<As...>()); }

  template <std::size_t J>
  const TypedBlob<ElementType<J>> &GetElementTypedBlob() const {
    return RawIndexedBlob::template Get<TypedBlob<ElementType<J>>>(J);
  }

  template <typename A> const TypedBlob<A> &GetElementTypedBlob() const {
    return this->template get<ElementTypeIndex<A>>();
  }

  template <std::size_t J> ElementValueType<J> get() const {
    return this->template GetElementTypedBlob<J>().Get();
  }

  template <typename A> TypedBlobValueType<A> get() const {
    return this->template GetElementTypedBlob<A>().Get();
  }

private:
  template <std::size_t... J>
  ValueType GetHelper(std::index_sequence<J...>) const {
    return {get<J>()...};
  }

  template <typename B>
  static constexpr std::size_t ElementTypeIndex{common::TypeIndex<B, As...>};
};

template <typename A> class TypedBlob<std::vector<A>> : public RawIndexedBlob {
public:
  using ElementValue = TypedBlobValueType<A>;

  class Iterator {
  public:
    Iterator(const TypedBlob &a, std::size_t j) : blob_{a}, j_{j} {}
    ElementValue operator*() const { return blob_.at(j_); }
    ElementValue *operator->() const { return &**this; }
    Iterator &operator++() {
      ++j_;
      return *this;
    }
    Iterator operator++(int) {
      Iterator result{blob_, j_};
      ++j_;
      return result;
    }
    bool operator==(const Iterator &that) const {
      return &blob_ == &that.blob_ && j_ == that.j_;
    }
    bool operator!=(const Iterator &that) const { return !(*this == that); }

  private:
    const TypedBlob &blob_;
    std::size_t j_;
  };

  std::size_t size() const { return Elements(); }
  ElementValue at(std::size_t j) const {
    return RawIndexedBlob::template Get<TypedBlob<A>>(j).Get();
  }
  Iterator cbegin() const { return {*this, 0}; }
  Iterator begin() const { return cbegin(); }
  Iterator cend() const { return {*this, size()}; }
  Iterator end() const { return cend(); }

  // The TypedBlob for a std::vector is its own value, as it supports the
  // necessary subset of std::vector's API.
  const TypedBlob &Get() const { return *this; }

  const TypedBlob<A> &GetElementTypedBlob(std::size_t j) const {
    return RawIndexedBlob::template Get<TypedBlob<A>>(j);
  }
};

// Relative pointers for use in static tables, specifically for
// pointers to derived types in the addenda of static descriptors.

template <typename A> class RelativePointer {
public:
  template <typename CONTAINER>
  void Set(const CONTAINER &container, const A &object) {
    offset_ = reinterpret_cast<const char *>(&object) -
        reinterpret_cast<const char *>(&container);
  }
  void Nullify() { offset_ = 0; }
  template <typename CONTAINER> const A *Get(const CONTAINER &container) const {
    if (offset_ == 0) {
      return nullptr;
    }
    return reinterpret_cast<const A *>(
        reinterpret_cast<const char *>(&container) + offset_);
  }

private:
  std::ptrdiff_t offset_{0};
};

} // namespace Fortran::runtime

#endif // FORTRAN_RUNTIME_BLOB_H_
