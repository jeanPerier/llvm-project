//===-- include/flang/Runtime/struct-declare-start.h ---------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// Note: this header does not have ifndef/defined/endif on purpose: it really should be included every time it appears in an include.

// Define a set of macros that allow declaring structs in the runtime header such that both the runtime code and compiler code share a common representation and can safely interact with it.
//
// The following struct:
// ```
//   struct SomeStruct {
//      type1 member1;
//      type2 member2;
//      ...
//   };
// ```
// 
// Should be defined in the runtime header as:
//
// ```
//  #include "flang/Runtime/runtime-struct-declare.h"
//  #define SOME_STRUCT_STRUCT(STRUCT_NAME, MEMBER)    \
//    STRUCT_NAME(SomeStruct) \
//    MEMBER(type1, member1)  \
//    MEMBER(type2, member2)  \
//    ....
//
//    FLANG_DECLARE_RUNTIME_STRUCT(SOME_STRUCT_STRUCT)
//
// ```
// 
// And it can be used in the flang compiler code as:
//
// ```
//  #include "runtime-header-that-declared-the-struct.h"
//  #include "flang/Optimizer/Builder/Runtime/runtime-struct.h"
//  ...
//  DECLARE_RUNTIME_STRUCT_BUILDER(SOME_STRUCT_STRUCT)
// ```
//
// Which will generate a class which allows lowering to safely interact with the runtime struct:
// ```
//   struct SomeStructRuntimeBuilder {
//      mlir::Type member1Type(mlir::Context*);
//      mlir::Type member2Type(mlir::Context*);
//      ....
//      mlir::Type getStructType(mlir::Context*);
//      mlir::Type member1Address(mlir::Location, fir::FirOpBuilder&, mlir::Value structAddr);
//      mlir::Type member2Address(mlir::Location, fir::FirOpBuilder&, mlir::Value structAddr);
//      ...
//   };
// ```

#define FLANG_DECLARE_RUNTIME_STRUCT_NAME(name) struct name {
#define FLANG_DECLARE_RUNTIME_STRUCT_MEMBER(type, name) type name;
#define FLANG_DECLARE_RUNTIME_STRUCT(SOME_STRUCT) \
  SOME_STRUCT(FLANG_DECLARE_RUNTIME_STRUCT_NAME, FLANG_DECLARE_RUNTIME_STRUCT_MEMBER) };
