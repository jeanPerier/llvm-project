//===-- runtime/blob-builder.h ----------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef FORTRAN_RUNTIME_BLOB_BUILDER_H_
#define FORTRAN_RUNTIME_BLOB_BUILDER_H_

// CreateBlob() constructs blobs directly from instances of the types
// supported by TypedBlob<> in blob.h.
//
// To construct a table in the compiler for use in the runtime, design
// the table with the supported containers, run CreateBlob(),
// and pass the blob through the bridge (TBD).  To access the tables from
// the blobs in the runtime support library, view their content as a
// TypedBlob<> with the original base type, which should of course be
// defined from the same header file that was used to build the blob in
// the compiler.

#include "blob.h"

namespace Fortran::runtime {

template <typename A> struct TypedBlobBuilder : public TypedBlob<A> {
  static constexpr std::size_t Measurements(const A &x) { return 0; }
  static constexpr std::size_t Measure(std::size_t *&, const A &x) {
    return sizeof x;
  }
  void Fill(std::size_t *&, const A &x) { RawBlob::template Get<A>() = x; }
};

template <>
struct TypedBlobBuilder<const char *> : public TypedBlob<const char *> {
  static std::size_t Measurements(const char *);
  static std::size_t Measure(std::size_t *&, const char *);
  void Fill(std::size_t *&, const char *);
};

template <typename A>
struct TypedBlobBuilder<std::unique_ptr<A>>
    : public TypedBlob<std::unique_ptr<A>> {
  static std::size_t Measurements(const std::unique_ptr<A> &x) {
    return x.get() ? 1 + TypedBlobBuilder<A>::Measurements(*x) : 0;
  }

  static std::size_t Measure(std::size_t *&p, const std::unique_ptr<A> &x) {
    if (const auto *value{x.get()}) {
      std::size_t *size{p++};
      *size = TypedBlobBuilder<A>::Measure(p, *value);
      return RawIndexedBlob::ComputeBytesFor(1, size);
    } else {
      return RawIndexedBlob::ComputeBytesFor(0, nullptr);
    }
  }

  void Fill(std::size_t *&p, const std::unique_ptr<A> &x) {
    if (x.get()) {
      RawIndexedBlob::Init(1, p++);
      RawIndexedBlob::template Get<TypedBlobBuilder<A>>(0).Fill(p, *x);
    } else {
      RawIndexedBlob::Init(0, nullptr);
    }
  }
};

template <typename A>
struct TypedBlobBuilder<std::optional<A>> : public TypedBlob<std::optional<A>> {
  static std::size_t Measurements(const std::optional<A> &x) {
    return x.has_value() ? 1 + TypedBlobBuilder<A>::Measurements(*x) : 0;
  }

  static std::size_t Measure(std::size_t *&p, const std::optional<A> &x) {
    if (x.has_value()) {
      std::size_t *size{p++};
      *size = TypedBlobBuilder<A>::Measure(p, *x);
      return RawIndexedBlob::ComputeBytesFor(1, size);
    } else {
      return RawIndexedBlob::ComputeBytesFor(0, nullptr);
    }
  }

  void Fill(std::size_t *&p, const std::optional<A> &x) {
    if (x.has_value()) {
      RawIndexedBlob::Init(1, p++);
      RawIndexedBlob::template Get<TypedBlobBuilder<A>>(0).Fill(p, *x);
    } else {
      RawIndexedBlob::Init(0, nullptr);
    }
  }
};

template <typename... As>
class TypedBlobBuilder<std::tuple<As...>>
    : public TypedBlob<std::tuple<As...>> {
public:
  using Base = TypedBlob<std::tuple<As...>>;
  using typename Base::Tuple;

  static std::size_t Measurements(const Tuple &x) {
    return MeasurementsHelper(x, std::index_sequence_for<As...>());
  }

  static std::size_t Measure(std::size_t *&p, const Tuple &x) {
    return MeasureHelper(p, x, std::index_sequence_for<As...>());
  }

  void Fill(std::size_t *&p, const Tuple &x) {
    std::size_t *sizes{p};
    p += sizeof...(As);
    RawIndexedBlob::Init(sizeof...(As), sizes);
    FillElements(p, x, std::index_sequence_for<As...>());
  }

private:
  template <std::size_t... J>
  static std::size_t MeasurementsHelper(
      const Tuple &x, std::index_sequence<J...>) {
    return (sizeof...(As) + ... +
        TypedBlobBuilder<As>::Measurements(std::get<J>(x)));
  }

  template <std::size_t... J>
  static std::size_t MeasureHelper(
      std::size_t *&p, const Tuple &x, std::index_sequence<J...>) {
    std::size_t *size{p};
    p += sizeof...(As);
    std::size_t *q{size};
    ((*q++ = TypedBlobBuilder<As>::Measure(p, std::get<J>(x))), ...);
    return RawIndexedBlob::ComputeBytesFor(sizeof...(As), size);
  }

  template <std::size_t J>
  TypedBlobBuilder<typename Base::template ElementType<J>> &
  GetElementTypedBlobBuilder() {
    return RawIndexedBlob::template Get<
        TypedBlobBuilder<typename Base::template ElementType<J>>>(J);
  }

  template <std::size_t... J>
  void FillElements(
      std::size_t *&p, const Tuple &x, std::index_sequence<J...>) {
    (GetElementTypedBlobBuilder<J>().Fill(p, std::get<J>(x)), ...);
  }
};

template <typename A>
class TypedBlobBuilder<std::vector<A>> : public TypedBlob<std::vector<A>> {
public:
  static std::size_t Measurements(const std::vector<A> &x) {
    std::size_t measurements{x.size()};
    for (const A &element : x) {
      measurements += TypedBlobBuilder<A>::Measurements(element);
    }
    return measurements;
  }

  static std::size_t Measure(std::size_t *&p, const std::vector<A> &x) {
    std::size_t *size{p};
    p += x.size();
    std::size_t *q{size};
    for (const A &element : x) {
      *q++ = TypedBlobBuilder<A>::Measure(p, element);
    }
    return RawIndexedBlob::ComputeBytesFor(x.size(), size);
  }

  void Fill(std::size_t *&p, const std::vector<A> &x) {
    std::size_t *sizes{p};
    std::size_t n{x.size()};
    p += n;
    RawIndexedBlob::Init(n, sizes);
    for (std::size_t j{0}; j < n; ++j) {
      GetElementTypedBlobBuilder(j).Fill(p, x[j]);
    }
  }

private:
  TypedBlobBuilder<A> &GetElementTypedBlobBuilder(std::size_t j) {
    return RawIndexedBlob::template Get<TypedBlobBuilder<A>>(j);
  }
};

// The CreateBlob() template function builds a typed blob holding
// all of the contents of an object.  The implementation member functions
// of TypedBlob<> specialization that are used by CreateBlob()
// -- Measurements(), Measure(), and Fill() -- use a two-pass approach to
// calculate the size of the blob and then a third pass to populate it.
template <typename A>
OwningPtr<TypedBlob<A>> CreateBlob(const Terminator &terminator, const A &x) {
  std::size_t measurements{TypedBlobBuilder<A>::Measurements(x)};
  OwningPtr<RawBlob> measureBlob{
      SizedNew<RawBlob>{terminator}(measurements * sizeof(std::size_t))};
  std::size_t *sizes{&measureBlob->Get<std::size_t>()};
  std::size_t *p{sizes};
  std::size_t total{TypedBlobBuilder<A>::Measure(p, x)};
  OwningPtr<TypedBlobBuilder<A>> blob{
      SizedNew<TypedBlobBuilder<A>>{terminator}(total)};
  p = sizes;
  blob->Fill(p, x);
  return OwningPtr<TypedBlob<A>>{blob.release()};
}
} // namespace Fortran::runtime
#endif // FORTRAN_RUNTIME_BLOB_BUILDER_H_
