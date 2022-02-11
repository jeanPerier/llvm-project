//===-- include/flang/Common/builtin-modules.h ------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef FORTRAN_COMMON_BUILTIN_MODULES_H_
#define FORTRAN_COMMON_BUILTIN_MODULES_H_

// Define information about flang builtin modules.

namespace Fortran::common {
// Name of the builtin module that defines builtin derived types meant
// to describe other derived types at runtime in flang descriptor.
constexpr char typeInfoBuiltinModule[]{"__fortran_type_info"};
} // namespace Fortran::common
#endif // FORTRAN_COMMON_BUILTIN_MODULES_H_
