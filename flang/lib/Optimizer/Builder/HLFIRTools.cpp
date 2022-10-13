//===-- HLFIRTools.cpp
//------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Tools to manipulate HLFIR variable and expressions
//
//===----------------------------------------------------------------------===//

#include "flang/Optimizer/Builder/HLFIRTools.h"
#include "flang/Optimizer/Builder/FIRBuilder.h"
#include "flang/Optimizer/Builder/Todo.h"

// Return explicit extents. If the base is a fir.box, this won't read it to
// return the extents and will instead return an empty vector.
static llvm::SmallVector<mlir::Value>
getExplicitExtents(fir::DefineFortranVariableOpInterface var) {
  llvm::SmallVector<mlir::Value> result;
  if (llvm::Optional<mlir::Value> shape = var.getShape()) {
    auto *shapeOp = shape->getDefiningOp();
    if (auto s = mlir::dyn_cast_or_null<fir::ShapeOp>(shapeOp)) {
      auto e = s.getExtents();
      result.append(e.begin(), e.end());
    } else if (auto s = mlir::dyn_cast_or_null<fir::ShapeShiftOp>(shapeOp)) {
      auto e = s.getExtents();
      result.append(e.begin(), e.end());
    } else if (mlir::dyn_cast_or_null<fir::ShiftOp>(shapeOp)) {
      return {};
    } else {
      TODO(var->getLoc(), "read fir.shape to get extents");
    }
  }
  return {};
}

// Return explicit lower bounds. For pointers and allocatables, this will not
// read the lower bounds and instead return an empty vector.
static llvm::SmallVector<mlir::Value>
getExplicitLbounds(fir::DefineFortranVariableOpInterface var) {
  llvm::SmallVector<mlir::Value> result;
  if (llvm::Optional<mlir::Value> shape = var.getShape()) {
    auto *shapeOp = shape->getDefiningOp();
    if (auto s = mlir::dyn_cast_or_null<fir::ShapeOp>(shapeOp)) {
      return {};
    } else if (auto s = mlir::dyn_cast_or_null<fir::ShapeShiftOp>(shapeOp)) {
      auto e = s.getOrigins();
      result.append(e.begin(), e.end());
    } else if (mlir::dyn_cast_or_null<fir::ShiftOp>(shapeOp)) {
      auto e = s.getOrigins();
      result.append(e.begin(), e.end());
    } else {
      TODO(var->getLoc(), "read fir.shape to get lower bounds");
    }
  }
  return {};
}

static llvm::SmallVector<mlir::Value>
getExplicitTypeParams(fir::DefineFortranVariableOpInterface var) {
  llvm::SmallVector<mlir::Value> res;
  mlir::OperandRange range = var.getExplicitTypeParams();
  res.append(range.begin(), range.end());
  return res;
}

fir::ExtendedValue
fir::toExtendedValue(fir::DefineFortranVariableOpInterface var) {
  if (var.isPointer() || var.isAllocatable())
    TODO(var->getLoc(), "pointer or allocatable "
                        "DefineFortranVariableOpInterface to extendedValue");
  if (var.getBase().getType().isa<fir::BaseBoxType>())
    return fir::BoxValue(var.getBase(), getExplicitLbounds(var),
                         getExplicitTypeParams(var), getExplicitExtents(var));
  if (var.isCharacter()) {
    if (var.isArray())
      return fir::CharArrayBoxValue(var.getBase(), var.getExplicitCharLen(),
                                    getExplicitExtents(var),
                                    getExplicitLbounds(var));
    return fir::CharBoxValue(var.getBase(), var.getExplicitCharLen());
  }
  if (var.isArray())
    return fir::ArrayBoxValue(var.getBase(), getExplicitExtents(var),
                              getExplicitLbounds(var));
  return var.getBase();
}
