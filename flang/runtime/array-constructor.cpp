//===-- runtime/array-constructor.cpp -------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "flang/Runtime/array-constructor.h"
#include "flang/Runtime/assign.h"
#include "flang/Runtime/descriptor.h"

namespace Fortran::runtime {

extern "C" {
void RTNAME(PushArrayConstructorValue)(ArrayConstructorTemporary& to, const Descriptor &value,
    const char *sourceFile, int sourceLine) {
  to.nextValuePosition = to.allocationSize;
  
 // if (to->descriptor.)
 // If is not allocated
 //  -> initial allocation based on value.
 //  -> update allocationSize
 // else if storage is too small
 //  -> realloc
 //  -> update allocationSize
 //  -> update descriptor address.
 // Get element or section for value.
 // Copy value to temp.
 // Update descriptor extent if nextValuePosition > extent.
 //  -> what if descriptor was already finished ?
}
} // extern "C"
} // namespace Fortran::runtime
