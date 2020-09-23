//===-- include/flang/Common/static-multimap-view.h -------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef FORTRAN_COMMON_STATIC_MULTIMAP_VIEW_H_
#define FORTRAN_COMMON_STATIC_MULTIMAP_VIEW_H_
#include <utility>

/// StaticMultimapView is a constexpr friendly multimap
/// implementation over sorted constexpr arrays.
/// As the View name suggests, it does not duplicate the
/// sorted array but only brings range and search concepts
/// over it. It provides compile time search and can also
/// provide dynamic search (currently linear, can be improved to
/// log(n) due to the sorted array property).

namespace Fortran::common {

template <typename V> class StaticMultimapView {
public:
  using Key = typename V::Key;
  struct Range {
    using const_iterator = const V *;
    constexpr const_iterator begin() const { return startPtr; }
    constexpr const_iterator end() const { return endPtr; }
    constexpr bool empty() const {
      return startPtr == nullptr || endPtr == nullptr || endPtr <= startPtr;
    }
    constexpr std::size_t size() const {
      return empty() ? 0 : static_cast<std::size_t>(endPtr - startPtr);
    }
    const V *startPtr{nullptr};
    const V *endPtr{nullptr};
  };
  using const_iterator = typename Range::const_iterator;

  template <std::size_t N>
  constexpr StaticMultimapView(const V (&array)[N])
      : range_{&array[0], &array[0] + N} {}
  constexpr const_iterator begin() const { return range_.begin(); }
  constexpr const_iterator end() const { return range_.end(); }

  // Assume array is sorted.
  // TODO make it a log(n) search based on sorted property
  // std::equal_range will be constexpr in C++20 only.
  constexpr Range getRange(const Key &key) const {
    bool matched{false};
    const V *start{nullptr}, *end{nullptr};
    for (const auto &desc : range_) {
      if (desc.key == key) {
        if (!matched) {
          start = &desc;
          matched = true;
        }
      } else if (matched) {
        end = &desc;
        matched = false;
      }
    }
    if (matched) {
      end = range_.end();
    }
    return Range{start, end};
  }

  constexpr std::pair<const_iterator, const_iterator> equal_range(
      const Key &key) const {
    Range range{getRange(key)};
    return {range.begin(), range.end()};
  }

  constexpr typename Range::const_iterator find(Key key) const {
    const Range subRange{getRange(key)};
    return subRange.size() == 1 ? subRange.begin() : end();
  }

private:
  Range range_{nullptr, nullptr};
};
} // namespace Fortran::common
#endif // FORTRAN_COMMON_STATIC_MULTIMAP_VIEW_H_
