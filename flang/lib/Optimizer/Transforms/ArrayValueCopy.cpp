//===-- ArrayValueCopy.cpp ------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "PassDetail.h"
#include "flang/Lower/Todo.h" // delete!
#include "flang/Optimizer/Builder/BoxValue.h"
#include "flang/Optimizer/Builder/Character.h"
#include "flang/Optimizer/Builder/FIRBuilder.h"
#include "flang/Optimizer/Builder/Factory.h"
#include "flang/Optimizer/Dialect/FIRDialect.h"
#include "flang/Optimizer/Dialect/FIROpsSupport.h"
#include "flang/Optimizer/Support/FIRContext.h"
#include "flang/Optimizer/Transforms/Passes.h"
#include "mlir/Dialect/SCF/SCF.h"
#include "mlir/Transforms/DialectConversion.h"
#include "llvm/Support/Debug.h"

#define DEBUG_TYPE "flang-array-value-copy"

using namespace fir;

using OperationUseMapT = llvm::DenseMap<mlir::Operation *, mlir::Operation *>;

namespace {
/// Array copy analysis.
/// Perform an interference analysis between array values.
///
/// Lowering will generate a sequence of the following form.
/// ```mlir
///   %a_1 = fir.array_load %array_1(%shape) : ...
///   ...
///   %a_j = fir.array_load %array_j(%shape) : ...
///   ...
///   %a_n = fir.array_load %array_n(%shape) : ...
///     ...
///     %v_i = fir.array_fetch %a_i, ...
///     %a_j1 = fir.array_update %a_j, ...
///     ...
///   fir.array_merge_store %a_j, %a_jn to %array_j : ...
/// ```
///
/// The analysis is to determine if there are any conflicts. A conflict is when
/// one the following cases occurs.
///
/// 1. There is an `array_update` to an array value, a_j, such that a_j was
/// loaded from the same array memory reference (array_j) but with a different
/// shape as the other array values a_i, where i != j. [Possible overlapping
/// arrays.]
///
/// 2. There is either an array_fetch or array_update of a_j with a different
/// set of index values. [Possible loop-carried dependence.]
///
/// If none of the array values overlap in storage and the accesses are not
/// loop-carried, then the arrays are conflict-free and no copies are required.
class ArrayCopyAnalysis {
public:
  using ConflictSetT = llvm::SmallPtrSet<mlir::Operation *, 16>;
  using UseSetT = llvm::SmallPtrSet<mlir::OpOperand *, 8>;
  using LoadMapSetsT = llvm::DenseMap<mlir::Operation *, UseSetT>;
  using AmendAccessSetT = llvm::SmallPtrSet<mlir::Operation *, 4>;

  ArrayCopyAnalysis(mlir::Operation *op) : operation{op} {
    construct(op->getRegions());
  }

  mlir::Operation *getOperation() const { return operation; }

  /// Return true iff the `array_merge_store` has potential conflicts.
  bool hasPotentialConflict(mlir::Operation *op) const {
    LLVM_DEBUG(llvm::dbgs()
               << "looking for a conflict on " << *op
               << " and the set has a total of " << conflicts.size() << '\n');
    return conflicts.contains(op);
  }

  /// Return the use map.
  /// The use map maps array access, amend, fetch and update operations back to
  /// the array load that is the original source of the array value.
  /// It maps an array_load to an array_merge_store, if and only if the loaded
  /// array value has pending modifications to be merged.
  const OperationUseMapT &getUseMap() const { return useMap; }

  /// Return the set of array_access ops directly associated with array_amend
  /// ops.
  bool inAmendAccessSet(mlir::Operation *op) const {
    return amendAccesses.count(op);
  }

  /// For ArrayLoad `load`, return the transitive set of all OpOperands.
  UseSetT getLoadUseSet(mlir::Operation *load) const {
    assert(loadMapSets.count(load) && "analysis missed an array load?");
    return loadMapSets.lookup(load);
  }

  void arrayMentions(llvm::SmallVectorImpl<mlir::Operation *> &mentions,
                     ArrayLoadOp load);

private:
  void construct(mlir::MutableArrayRef<mlir::Region> regions);

  mlir::Operation *operation; // operation that analysis ran upon
  ConflictSetT conflicts;     // set of conflicts (loads and merge stores)
  OperationUseMapT useMap;
  LoadMapSetsT loadMapSets;
  // Set of array_access ops associated with array_amend ops.
  AmendAccessSetT amendAccesses;
};

/// Helper class to collect all array operations that produced an array value.
class ReachCollector {
public:
  ReachCollector(llvm::SmallVectorImpl<mlir::Operation *> &reach,
                 mlir::Region *loopRegion)
      : reach{reach}, loopRegion{loopRegion} {}

  void collectArrayMentionFrom(mlir::Operation *op, mlir::ValueRange range) {
    if (range.empty()) {
      collectArrayMentionFrom(op, mlir::Value{});
      return;
    }
    for (auto v : range)
      collectArrayMentionFrom(v);
  }

  // Collect all the array_access ops in `block`. This recursively looks into
  // blocks in ops with regions.
  // FIXME: This is temporarily relying on the array_amend appearing in a
  // do_loop Region.  This phase ordering assumption can be eliminated by using
  // dominance information to find the array_access ops or by scanning the
  // transitive closure of the amending array_access's users and the defs that
  // reach them.
  void collectAccesses(llvm::SmallVector<fir::ArrayAccessOp> &result,
                       mlir::Block *block) {
    for (auto &op : *block) {
      if (auto access = mlir::dyn_cast<ArrayAccessOp>(op)) {
        LLVM_DEBUG(llvm::dbgs() << "adding access: " << access << '\n');
        result.push_back(access);
        continue;
      }
      for (auto &region : op.getRegions())
        for (auto &bb : region.getBlocks())
          collectAccesses(result, &bb);
    }
  }

  void collectArrayMentionFrom(mlir::Operation *op, mlir::Value val) {
    // `val` is defined by an Op, process the defining Op.
    // If `val` is defined by a region containing Op, we want to drill down
    // and through that Op's region(s).
    LLVM_DEBUG(llvm::dbgs() << "popset: " << *op << '\n');
    auto popFn = [&](auto rop) {
      assert(val && "op must have a result value");
      auto resNum = val.cast<mlir::OpResult>().getResultNumber();
      llvm::SmallVector<mlir::Value> results;
      rop.resultToSourceOps(results, resNum);
      for (auto u : results)
        collectArrayMentionFrom(u);
    };
    if (auto rop = mlir::dyn_cast<DoLoopOp>(op)) {
      popFn(rop);
      return;
    }
    if (auto rop = mlir::dyn_cast<IterWhileOp>(op)) {
      popFn(rop);
      return;
    }
    if (auto rop = mlir::dyn_cast<fir::IfOp>(op)) {
      popFn(rop);
      return;
    }
    if (auto box = mlir::dyn_cast<EmboxOp>(op)) {
      for (auto *user : box.memref().getUsers())
        if (user != op)
          collectArrayMentionFrom(user, user->getResults());
      return;
    }
    if (auto mergeStore = mlir::dyn_cast<ArrayMergeStoreOp>(op)) {
      if (opIsInsideLoops(mergeStore))
        collectArrayMentionFrom(mergeStore.sequence());
      return;
    }

    if (mlir::isa<AllocaOp, AllocMemOp>(op)) {
      // Look for any stores inside the loops, and collect an array operation
      // that produced the value being stored to it.
      for (auto *user : op->getUsers())
        if (auto store = mlir::dyn_cast<fir::StoreOp>(user))
          if (opIsInsideLoops(store))
            collectArrayMentionFrom(store.value());
      return;
    }

    // Scan the uses of amend's memref
    if (auto amend = mlir::dyn_cast<ArrayAmendOp>(op)) {
      reach.push_back(op);
      llvm::SmallVector<ArrayAccessOp> accesses;
      collectAccesses(accesses, op->getBlock());
      for (auto access : accesses)
        collectArrayMentionFrom(access.getResult());
    }

    // Otherwise, Op does not contain a region so just chase its operands.
    if (mlir::isa<ArrayAccessOp, ArrayLoadOp, ArrayUpdateOp, ArrayModifyOp,
                  ArrayFetchOp>(op)) {
      LLVM_DEBUG(llvm::dbgs() << "add " << *op << " to reachable set\n");
      reach.push_back(op);
    }

    // Include all array_access ops using an array_load.
    if (auto arrLd = mlir::dyn_cast<ArrayLoadOp>(op))
      for (auto *user : arrLd.getResult().getUsers())
        if (mlir::isa<ArrayAccessOp>(user)) {
          LLVM_DEBUG(llvm::dbgs() << "add " << *user << " to reachable set\n");
          reach.push_back(user);
        }

    // Array modify assignment is performed on the result. So the analysis must
    // look at the what is done with the result.
    if (mlir::isa<ArrayModifyOp>(op))
      for (auto *user : op->getResult(0).getUsers())
        followUsers(user);

    if (mlir::isa<fir::CallOp>(op)) {
      LLVM_DEBUG(llvm::dbgs() << "add " << *op << " to reachable set\n");
      reach.push_back(op);
    }

    for (auto u : op->getOperands())
      collectArrayMentionFrom(u);
  }

  void collectArrayMentionFrom(mlir::BlockArgument ba) {
    auto *parent = ba.getOwner()->getParentOp();
    // If inside an Op holding a region, the block argument corresponds to an
    // argument passed to the containing Op.
    auto popFn = [&](auto rop) {
      collectArrayMentionFrom(rop.blockArgToSourceOp(ba.getArgNumber()));
    };
    if (auto rop = mlir::dyn_cast<DoLoopOp>(parent)) {
      popFn(rop);
      return;
    }
    if (auto rop = mlir::dyn_cast<IterWhileOp>(parent)) {
      popFn(rop);
      return;
    }
    // Otherwise, a block argument is provided via the pred blocks.
    for (auto *pred : ba.getOwner()->getPredecessors()) {
      auto u = pred->getTerminator()->getOperand(ba.getArgNumber());
      collectArrayMentionFrom(u);
    }
  }

  // Recursively trace operands to find all array operations relating to the
  // values merged.
  void collectArrayMentionFrom(mlir::Value val) {
    if (!val || visited.contains(val))
      return;
    visited.insert(val);

    // Process a block argument.
    if (auto ba = val.dyn_cast<mlir::BlockArgument>()) {
      collectArrayMentionFrom(ba);
      return;
    }

    // Process an Op.
    if (auto *op = val.getDefiningOp()) {
      collectArrayMentionFrom(op, val);
      return;
    }

    emitFatalError(val.getLoc(), "unhandled value");
  }

  /// Return all ops that produce the array value that is stored into the
  /// `array_merge_store`.
  static void reachingValues(llvm::SmallVectorImpl<mlir::Operation *> &reach,
                             mlir::Value seq) {
    reach.clear();
    mlir::Region *loopRegion = nullptr;
    if (auto doLoop = mlir::dyn_cast_or_null<DoLoopOp>(seq.getDefiningOp()))
      loopRegion = &doLoop->getRegion(0);
    ReachCollector collector(reach, loopRegion);
    collector.collectArrayMentionFrom(seq);
  }

private:
  /// Is \op inside the loop nest region ?
  /// FIXME: replace this structural dependence with graph properties.
  bool opIsInsideLoops(mlir::Operation *op) const {
    auto *region = op->getParentRegion();
    while (region) {
      if (region == loopRegion)
        return true;
      region = region->getParentRegion();
    }
    return false;
  }

  /// Recursively trace the use of an operation results, calling
  /// collectArrayMentionFrom on the direct and indirect user operands.
  void followUsers(mlir::Operation *op) {
    for (auto userOperand : op->getOperands())
      collectArrayMentionFrom(userOperand);
    // Go through potential converts/coordinate_op.
    for (auto indirectUser : op->getUsers())
      followUsers(indirectUser);
  }

  llvm::SmallVectorImpl<mlir::Operation *> &reach;
  llvm::SmallPtrSet<mlir::Value, 16> visited;
  /// Region of the loops nest that produced the array value.
  mlir::Region *loopRegion;
};
} // namespace

/// Find all the array operations that access the array value that is loaded by
/// the array load operation, `load`.
void ArrayCopyAnalysis::arrayMentions(
    llvm::SmallVectorImpl<mlir::Operation *> &mentions, ArrayLoadOp load) {
  mentions.clear();
  auto lmIter = loadMapSets.find(load);
  if (lmIter != loadMapSets.end()) {
    for (auto *opnd : lmIter->second) {
      auto *owner = opnd->getOwner();
      if (mlir::isa<ArrayAccessOp, ArrayAmendOp, ArrayFetchOp, ArrayUpdateOp,
                    ArrayModifyOp>(owner))
        mentions.push_back(owner);
    }
    return;
  }

  UseSetT visited;
  llvm::SmallVector<mlir::OpOperand *> queue; // uses of ArrayLoad[orig]

  auto appendToQueue = [&](mlir::Value val) {
    for (auto &use : val.getUses())
      if (!visited.count(&use)) {
        visited.insert(&use);
        queue.push_back(&use);
      }
  };

  // Build the set of uses of `original`.
  // let USES = { uses of original fir.load }
  appendToQueue(load);

  // Process the worklist until done.
  while (!queue.empty()) {
    auto *operand = queue.pop_back_val();
    auto *owner = operand->getOwner();
    if (!owner)
      continue;
    auto structuredLoop = [&](auto ro) {
      if (auto blockArg = ro.iterArgToBlockArg(operand->get())) {
        auto arg = blockArg.getArgNumber();
        auto output = ro.getResult(ro.finalValue() ? arg : arg - 1);
        appendToQueue(output);
        appendToQueue(blockArg);
      }
    };
    auto branchOp = [&](mlir::Block *dest, auto operands) {
      for (auto i : llvm::enumerate(operands))
        if (operand->get() == i.value()) {
          auto blockArg = dest->getArgument(i.index());
          appendToQueue(blockArg);
        }
    };
    // Thread uses into structured loop bodies and return value uses.
    if (auto ro = mlir::dyn_cast<DoLoopOp>(owner)) {
      structuredLoop(ro);
    } else if (auto ro = mlir::dyn_cast<IterWhileOp>(owner)) {
      structuredLoop(ro);
    } else if (auto rs = mlir::dyn_cast<ResultOp>(owner)) {
      // Thread any uses of fir.if that return the marked array value.
      auto *parent = rs->getParentRegion()->getParentOp();
      if (auto ifOp = mlir::dyn_cast<fir::IfOp>(parent))
        appendToQueue(ifOp.getResult(operand->getOperandNumber()));
    } else if (mlir::isa<ArrayFetchOp>(owner)) {
      // Keep track of array value fetches.
      LLVM_DEBUG(llvm::dbgs()
                 << "add fetch {" << *owner << "} to array value set\n");
      mentions.push_back(owner);
    } else if (auto update = mlir::dyn_cast<ArrayUpdateOp>(owner)) {
      // Keep track of array value updates and thread the return value uses.
      LLVM_DEBUG(llvm::dbgs()
                 << "add update {" << *owner << "} to array value set\n");
      mentions.push_back(owner);
      appendToQueue(update.getResult());
    } else if (auto update = mlir::dyn_cast<ArrayModifyOp>(owner)) {
      // Keep track of array value modification and thread the return value
      // uses.
      LLVM_DEBUG(llvm::dbgs()
                 << "add modify {" << *owner << "} to array value set\n");
      mentions.push_back(owner);
      appendToQueue(update.getResult(1));
    } else if (auto mention = mlir::dyn_cast<ArrayAccessOp>(owner)) {
      mentions.push_back(owner);
    } else if (auto amend = mlir::dyn_cast<ArrayAmendOp>(owner)) {
      mentions.push_back(owner);
      appendToQueue(amend.getResult());
    } else if (auto br = mlir::dyn_cast<mlir::BranchOp>(owner)) {
      branchOp(br.getDest(), br.destOperands());
    } else if (auto br = mlir::dyn_cast<mlir::CondBranchOp>(owner)) {
      branchOp(br.getTrueDest(), br.getTrueOperands());
      branchOp(br.getFalseDest(), br.getFalseOperands());
    } else if (mlir::isa<ArrayMergeStoreOp>(owner)) {
      // do nothing
    } else {
      llvm::report_fatal_error("array value reached unexpected op");
    }
  }
  loadMapSets.insert({load, visited});
}

/// Is there a conflict between the array value that was updated and to be
/// stored to `st` and the set of arrays loaded (`reach`) and used to compute
/// the updated value?
static bool conflictOnLoad(llvm::ArrayRef<mlir::Operation *> reach,
                           ArrayMergeStoreOp st) {
  mlir::Value load;
  auto addr = st.memref();
  auto stEleTy = dyn_cast_ptrOrBoxEleTy(addr.getType());
  for (auto *op : reach)
    if (auto ld = mlir::dyn_cast<ArrayLoadOp>(op)) {
      auto ldTy = ld.memref().getType();
      if (auto boxTy = ldTy.dyn_cast<BoxType>())
        ldTy = boxTy.getEleTy();
      if (ldTy.isa<fir::PointerType>() && stEleTy == dyn_cast_ptrEleTy(ldTy))
        return true;
      if (ld.memref() == addr) {
        if (ld.getResult() != st.original())
          return true;
        if (load)
          return true;
        load = ld;
      }
    }
  return false;
}

/// Is there a conflict on the array being merged into?
static bool conflictOnMerge(llvm::ArrayRef<mlir::Operation *> mentions) {
  if (mentions.size() < 2)
    return false;
  llvm::SmallVector<mlir::Value> indices;
  LLVM_DEBUG(llvm::dbgs() << "check merge conflict on with " << mentions.size()
                          << " mentions on the list\n");
  bool valSeen = false;
  bool refSeen = false;
  for (auto *op : mentions) {
    llvm::SmallVector<mlir::Value> compareVector;
    if (auto u = mlir::dyn_cast<ArrayUpdateOp>(op)) {
      valSeen = true;
      if (indices.empty()) {
        indices = u.indices();
        continue;
      }
      compareVector = u.indices();
    } else if (auto f = mlir::dyn_cast<ArrayModifyOp>(op)) {
      valSeen = true;
      if (indices.empty()) {
        indices = f.indices();
        continue;
      }
      compareVector = f.indices();
    } else if (auto f = mlir::dyn_cast<ArrayFetchOp>(op)) {
      valSeen = true;
      if (indices.empty()) {
        indices = f.indices();
        continue;
      }
      compareVector = f.indices();
    } else if (auto f = mlir::dyn_cast<ArrayAccessOp>(op)) {
      refSeen = true;
      if (indices.empty()) {
        indices = f.indices();
        continue;
      }
      compareVector = f.indices();
    } else if (mlir::isa<ArrayAmendOp>(op)) {
      refSeen = true;
      continue;
    } else {
      mlir::emitError(op->getLoc(), "unexpected operation in analysis");
    }
    if (compareVector.size() != indices.size() ||
        llvm::any_of(llvm::zip(compareVector, indices), [&](auto pair) {
          return std::get<0>(pair) != std::get<1>(pair);
        }))
      return true;
    LLVM_DEBUG(llvm::dbgs() << "vectors compare equal\n");
  }
  return valSeen && refSeen;
}

/// With element-by-reference semantics, an amended array with more than once
/// access to the same loaded array are conservatively considered a conflict.
/// Note: the array copy can still be eliminated in subsequent optimizations.
static bool conflictOnReference(llvm::ArrayRef<mlir::Operation *> mentions) {
  LLVM_DEBUG(llvm::dbgs() << "checking reference semantics " << mentions.size()
                          << '\n');
  if (mentions.size() < 3)
    return false;
  unsigned amendCount = 0;
  unsigned accessCount = 0;
  for (auto *op : mentions) {
    if (mlir::isa<ArrayAmendOp>(op) && ++amendCount > 1) {
      LLVM_DEBUG(llvm::dbgs() << "conflict: multiple amends of array value\n");
      return true;
    }
    if (mlir::isa<ArrayAccessOp>(op) && ++accessCount > 1) {
      LLVM_DEBUG(llvm::dbgs()
                 << "conflict: multiple accesses of array value\n");
      return true;
    }
    if (mlir::isa<ArrayFetchOp, ArrayUpdateOp, ArrayModifyOp>(op)) {
      LLVM_DEBUG(llvm::dbgs()
                 << "conflict: array value has both uses by-value and uses "
                    "by-reference. conservative assumption.\n");
      return true;
    }
  }
  return false;
}

static mlir::Operation *
amendingAccess(llvm::ArrayRef<mlir::Operation *> mentions) {
  for (auto *op : mentions)
    if (auto amend = mlir::dyn_cast<ArrayAmendOp>(op))
      return amend.memref().getDefiningOp();
  return {};
}

// Are any conflicts present? The conflicts detected here are described above.
static bool conflictDetected(llvm::ArrayRef<mlir::Operation *> reach,
                             llvm::ArrayRef<mlir::Operation *> mentions,
                             ArrayMergeStoreOp st) {
  return conflictOnLoad(reach, st) || conflictOnMerge(mentions);
}

// Assume that any call to a function that uses host-associations will be
// modifying the output array.
static bool
conservativeCallConflict(llvm::ArrayRef<mlir::Operation *> reaches) {
  return llvm::any_of(reaches, [](mlir::Operation *op) {
    if (auto call = mlir::dyn_cast<fir::CallOp>(op))
      if (auto callee =
              call.getCallableForCallee().dyn_cast<mlir::SymbolRefAttr>()) {
        auto module = op->getParentOfType<mlir::ModuleOp>();
        return fir::hasHostAssociationArgument(
            module.lookupSymbol<mlir::FuncOp>(callee));
      }
    return false;
  });
}

/// Constructor of the array copy analysis.
/// This performs the analysis and saves the intermediate results.
void ArrayCopyAnalysis::construct(mlir::MutableArrayRef<mlir::Region> regions) {
  for (auto &region : regions)
    for (auto &block : region.getBlocks())
      for (auto &op : block.getOperations()) {
        if (op.getNumRegions())
          construct(op.getRegions());
        if (auto st = mlir::dyn_cast<ArrayMergeStoreOp>(op)) {
          llvm::SmallVector<Operation *> values;
          ReachCollector::reachingValues(values, st.sequence());
          auto callConflict = conservativeCallConflict(values);
          llvm::SmallVector<Operation *> mentions;
          arrayMentions(mentions,
                        mlir::cast<ArrayLoadOp>(st.original().getDefiningOp()));
          auto conflict = conflictDetected(values, mentions, st);
          auto refConflict = conflictOnReference(mentions);
          if (callConflict || conflict || refConflict) {
            LLVM_DEBUG(llvm::dbgs()
                       << "CONFLICT: copies required for " << st << '\n'
                       << "   adding conflicts on: " << op << " and "
                       << st.original() << '\n');
            conflicts.insert(&op);
            conflicts.insert(st.original().getDefiningOp());
            if (auto *access = amendingAccess(mentions))
              amendAccesses.insert(access);
          }
          auto *ld = st.original().getDefiningOp();
          LLVM_DEBUG(llvm::dbgs()
                     << "map: adding {" << *ld << " -> " << st << "}\n");
          useMap.insert({ld, &op});
        } else if (auto load = mlir::dyn_cast<ArrayLoadOp>(op)) {
          llvm::SmallVector<mlir::Operation *> mentions;
          arrayMentions(mentions, load);
          LLVM_DEBUG(llvm::dbgs() << "process load: " << load
                                  << ", mentions: " << mentions.size() << '\n');
          for (auto *acc : mentions) {
            LLVM_DEBUG(llvm::dbgs() << " mention: " << *acc << '\n');
            if (mlir::isa<ArrayAccessOp, ArrayAmendOp, ArrayFetchOp,
                          ArrayUpdateOp, ArrayModifyOp>(acc)) {
              if (useMap.count(acc)) {
                mlir::emitError(
                    load.getLoc(),
                    "The parallel semantics of multiple array_merge_stores per "
                    "array_load are not supported.");
                continue;
              }
              LLVM_DEBUG(llvm::dbgs() << "map: adding {" << *acc << "} -> {"
                                      << load << "}\n");
              useMap.insert({acc, &op});
            }
          }
        }
      }
}

//===----------------------------------------------------------------------===//
// Conversions for converting out of array value form.
//===----------------------------------------------------------------------===//

namespace {
class ArrayLoadConversion : public mlir::OpRewritePattern<ArrayLoadOp> {
public:
  using OpRewritePattern::OpRewritePattern;

  mlir::LogicalResult
  matchAndRewrite(ArrayLoadOp load,
                  mlir::PatternRewriter &rewriter) const override {
    LLVM_DEBUG(llvm::dbgs() << "replace load " << load << " with undef.\n");
    rewriter.replaceOpWithNewOp<UndefOp>(load, load.getType());
    return mlir::success();
  }
};

class ArrayMergeStoreConversion
    : public mlir::OpRewritePattern<ArrayMergeStoreOp> {
public:
  using OpRewritePattern::OpRewritePattern;

  mlir::LogicalResult
  matchAndRewrite(ArrayMergeStoreOp store,
                  mlir::PatternRewriter &rewriter) const override {
    LLVM_DEBUG(llvm::dbgs() << "marking store " << store << " as dead.\n");
    rewriter.eraseOp(store);
    return mlir::success();
  }
};
} // namespace

static mlir::Type getEleTy(mlir::Type ty) {
  auto eleTy = unwrapSequenceType(unwrapPassByRefType(ty));
  // FIXME: keep ptr/heap/ref information.
  return ReferenceType::get(eleTy);
}

// Extract extents from the ShapeOp/ShapeShiftOp into the result vector.
static void getExtents(llvm::SmallVectorImpl<mlir::Value> &result,
                       mlir::Value shape) {
  auto *shapeOp = shape.getDefiningOp();
  if (auto s = mlir::dyn_cast<ShapeOp>(shapeOp)) {
    auto e = s.getExtents();
    result.insert(result.end(), e.begin(), e.end());
    return;
  }
  if (auto s = mlir::dyn_cast<ShapeShiftOp>(shapeOp)) {
    auto e = s.getExtents();
    result.insert(result.end(), e.begin(), e.end());
    return;
  }
  llvm::report_fatal_error("not a fir.shape/fir.shape_shift op");
}

// Place the extents of the array loaded by an ArrayLoadOp into the result
// vector and return a ShapeOp/ShapeShiftOp with the corresponding extents. If
// the ArrayLoadOp is loading a fir.box, code will be generated to read the
// extents from the fir.box, and a the retunred ShapeOp is built with the read
// extents.
// Otherwise, the extents will be extracted from the ShapeOp/ShapeShiftOp
// argument of the ArrayLoadOp that is returned.
static mlir::Value
getOrReadExtentsAndShapeOp(mlir::Location loc, mlir::PatternRewriter &rewriter,
                           ArrayLoadOp loadOp,
                           llvm::SmallVectorImpl<mlir::Value> &result) {
  assert(result.empty());
  if (auto boxTy = loadOp.memref().getType().dyn_cast<BoxType>()) {
    auto rank =
        dyn_cast_ptrOrBoxEleTy(boxTy).cast<SequenceType>().getDimension();
    auto idxTy = rewriter.getIndexType();
    for (decltype(rank) dim = 0; dim < rank; ++dim) {
      auto dimVal = rewriter.create<mlir::arith::ConstantIndexOp>(loc, dim);
      auto dimInfo = rewriter.create<BoxDimsOp>(loc, idxTy, idxTy, idxTy,
                                                loadOp.memref(), dimVal);
      result.emplace_back(dimInfo.getResult(1));
    }
    auto shapeType = ShapeType::get(rewriter.getContext(), rank);
    return rewriter.create<ShapeOp>(loc, shapeType, result);
  }
  getExtents(result, loadOp.shape());
  return loadOp.shape();
}

static mlir::Type toRefType(mlir::Type ty) {
  if (isa_ref_type(ty))
    return ty;
  return ReferenceType::get(ty);
}

static mlir::Value
genCoorOp(mlir::PatternRewriter &rewriter, mlir::Location loc, mlir::Type eleTy,
          mlir::Type resTy, mlir::Value alloc, mlir::Value shape,
          mlir::Value slice, mlir::ValueRange indices,
          mlir::ValueRange typeparams, bool skipOrig = false) {
  llvm::SmallVector<mlir::Value> originated;
  if (skipOrig)
    originated.assign(indices.begin(), indices.end());
  else
    originated = factory::originateIndices(loc, rewriter, alloc.getType(),
                                           shape, indices);
  auto seqTy = dyn_cast_ptrOrBoxEleTy(alloc.getType());
  assert(seqTy && seqTy.isa<SequenceType>());
  const auto dimension = seqTy.cast<SequenceType>().getDimension();
  mlir::Value result = rewriter.create<ArrayCoorOp>(
      loc, eleTy, alloc, shape, slice,
      llvm::ArrayRef<mlir::Value>{originated}.take_front(dimension),
      typeparams);
  if (dimension < originated.size())
    result = rewriter.create<CoordinateOp>(
        loc, resTy, result,
        llvm::ArrayRef<mlir::Value>{originated}.drop_front(dimension));
  return result;
}

/// Generate an array copy. This is used for both copy-in and copy-out.
static void genArrayCopy(mlir::Location loc, mlir::PatternRewriter &rewriter,
                         mlir::Value dst, mlir::Value src, mlir::Value shapeOp,
                         ArrayLoadOp arrLoad) {
  auto insPt = rewriter.saveInsertionPoint();
  llvm::SmallVector<mlir::Value> indices;
  llvm::SmallVector<mlir::Value> extents;
  getExtents(extents, shapeOp);
  // Build loop nest from column to row.
  for (auto sh : llvm::reverse(extents)) {
    auto idxTy = rewriter.getIndexType();
    auto ubi = rewriter.create<ConvertOp>(loc, idxTy, sh);
    auto zero = rewriter.create<mlir::arith::ConstantIndexOp>(loc, 0);
    auto one = rewriter.create<mlir::arith::ConstantIndexOp>(loc, 1);
    auto ub = rewriter.create<mlir::arith::SubIOp>(loc, idxTy, ubi, one);
    auto loop = rewriter.create<DoLoopOp>(loc, zero, ub, one);
    rewriter.setInsertionPointToStart(loop.getBody());
    indices.push_back(loop.getInductionVar());
  }
  // Reverse the indices so they are in column-major order.
  std::reverse(indices.begin(), indices.end());
  auto typeparams = arrLoad.typeparams();
  auto fromAddr = rewriter.create<ArrayCoorOp>(
      loc, getEleTy(src.getType()), src, shapeOp, mlir::Value{},
      factory::originateIndices(loc, rewriter, src.getType(), shapeOp, indices),
      typeparams);
  auto toAddr = rewriter.create<ArrayCoorOp>(
      loc, getEleTy(dst.getType()), dst, shapeOp, mlir::Value{},
      factory::originateIndices(loc, rewriter, dst.getType(), shapeOp, indices),
      typeparams);
  auto eleTy = unwrapSequenceType(unwrapPassByRefType(dst.getType()));
  if (hasDynamicSize(eleTy)) {
    if (auto charTy = eleTy.dyn_cast<fir::CharacterType>()) {
      assert(charTy.hasDynamicLen() && "dynamic size and constant length");
      // Copy from (to) object to (from) temp copy of same object.
      auto len = typeparams.back();
      CharBoxValue toChar(toAddr, len);
      CharBoxValue fromChar(fromAddr, len);
      auto module = toAddr->getParentOfType<mlir::ModuleOp>();
      FirOpBuilder builder{rewriter, getKindMapping(module)};
      factory::CharacterExprHelper helper{builder, loc};
      helper.createAssign(ExtendedValue{toChar}, ExtendedValue{fromChar});
    } else {
      TODO(loc, "copy element of dynamic size");
    }
  } else {
    auto load = rewriter.create<fir::LoadOp>(loc, fromAddr);
    rewriter.create<fir::StoreOp>(loc, load, toAddr);
  }
  rewriter.restoreInsertionPoint(insPt);
}

namespace {
/// Conversion of fir.array_update and fir.array_modify Ops.
/// If there is a conflict for the update, then we need to perform a
/// copy-in/copy-out to preserve the original values of the array. If there is
/// no conflict, then it is save to eschew making any copies.
template <typename ArrayOp>
class ArrayUpdateConversionBase : public mlir::OpRewritePattern<ArrayOp> {
public:
  // TODO: Implement copy/swap semantics?
  explicit ArrayUpdateConversionBase(mlir::MLIRContext *ctx,
                                     const ArrayCopyAnalysis &a,
                                     const OperationUseMapT &m)
      : mlir::OpRewritePattern<ArrayOp>{ctx}, analysis{a}, useMap{m} {}

  /// The array_access, \p access, is to be to a cloned copy due to a potential
  /// conflict. Uses copy-in/copy-out semantics and not copy/swap.
  mlir::Value referenceToClone(mlir::Location loc,
                               mlir::PatternRewriter &rewriter,
                               ArrayOp access) const {
    LLVM_DEBUG(llvm::dbgs()
               << "generating copy-in/copy-out loops for " << access << '\n');
    auto *op = access.getOperation();
    auto *loadOp = useMap.lookup(op);
    auto load = mlir::cast<ArrayLoadOp>(loadOp);
    auto eleTy = access.getType();
    rewriter.setInsertionPoint(loadOp);
    // Copy in.
    llvm::SmallVector<mlir::Value> extents;
    auto shapeOp = getOrReadExtentsAndShapeOp(loc, rewriter, load, extents);
    auto allocmem = rewriter.create<AllocMemOp>(
        loc, dyn_cast_ptrOrBoxEleTy(load.memref().getType()), load.typeparams(),
        extents);
    genArrayCopy(load.getLoc(), rewriter, allocmem, load.memref(), shapeOp,
                 load);
    // Generate the reference for the access.
    rewriter.setInsertionPoint(op);
    auto coor =
        genCoorOp(rewriter, loc, getEleTy(load.getType()), eleTy, allocmem,
                  shapeOp, load.slice(), access.indices(), load.typeparams(),
                  access->hasAttr(factory::attrFortranArrayOffsets()));
    // Copy out.
    auto *storeOp = useMap.lookup(loadOp);
    auto store = mlir::cast<ArrayMergeStoreOp>(storeOp);
    rewriter.setInsertionPoint(storeOp);
    // Copy out.
    genArrayCopy(store.getLoc(), rewriter, store.memref(), allocmem, shapeOp,
                 load);
    rewriter.create<FreeMemOp>(loc, allocmem);
    return coor;
  }

  /// Copy the RHS element into the LHS and insert copy-in/copy-out between a
  /// temp and the LHS if the analysis found potential overlaps between the RHS
  /// and LHS arrays. The element copy generator must be provided through \p
  /// assignElement. \p update must be the ArrayUpdateOp or the ArrayModifyOp.
  /// Returns the address of the LHS element inside the loop and the LHS
  /// ArrayLoad result.
  std::pair<mlir::Value, mlir::Value>
  materializeAssignment(mlir::Location loc, mlir::PatternRewriter &rewriter,
                        ArrayOp update,
                        const std::function<void(mlir::Value)> &assignElement,
                        mlir::Type lhsEltRefType) const {
    auto *op = update.getOperation();
    auto *loadOp = useMap.lookup(op);
    auto load = mlir::cast<ArrayLoadOp>(loadOp);
    LLVM_DEBUG(llvm::outs() << "does " << load << " have a conflict?\n");
    if (analysis.hasPotentialConflict(loadOp)) {
      // If there is a conflict between the arrays, then we copy the lhs array
      // to a temporary, update the temporary, and copy the temporary back to
      // the lhs array. This yields Fortran's copy-in copy-out array semantics.
      LLVM_DEBUG(llvm::outs() << "Yes, conflict was found\n");
      rewriter.setInsertionPoint(loadOp);
      // Copy in.
      llvm::SmallVector<mlir::Value> extents;
      llvm::SmallVector<mlir::Value> nonconstantExtents;
      auto shapeOp = getOrReadExtentsAndShapeOp(loc, rewriter, load, extents);
      auto arrTy = fir::unwrapPassByRefType(load.memref().getType());
      auto seqTy = arrTy.template dyn_cast<fir::SequenceType>();
      assert(seqTy && "expecting sequence type");
      for (auto [s, x] : llvm::zip(seqTy.getShape(), extents))
        if (s == fir::SequenceType::getUnknownExtent())
          nonconstantExtents.emplace_back(x);
      auto allocmem = rewriter.create<AllocMemOp>(
          loc, dyn_cast_ptrOrBoxEleTy(load.memref().getType()),
          load.typeparams(), nonconstantExtents);
      genArrayCopy(load.getLoc(), rewriter, allocmem, load.memref(), shapeOp,
                   load);
      rewriter.setInsertionPoint(op);
      auto coor = genCoorOp(
          rewriter, loc, getEleTy(load.getType()), lhsEltRefType, allocmem,
          shapeOp, load.slice(), update.indices(), load.typeparams(),
          update->hasAttr(factory::attrFortranArrayOffsets()));
      assignElement(coor);
      auto *storeOp = useMap.lookup(loadOp);
      auto store = mlir::cast<ArrayMergeStoreOp>(storeOp);
      rewriter.setInsertionPoint(storeOp);
      // Copy out.
      genArrayCopy(store.getLoc(), rewriter, store.memref(), allocmem, shapeOp,
                   load);
      rewriter.create<FreeMemOp>(loc, allocmem);
      return {coor, load.getResult()};
    }
    // Otherwise, when there is no conflict (a possible loop-carried
    // dependence), the lhs array can be updated in place.
    LLVM_DEBUG(llvm::outs() << "No, conflict wasn't found\n");
    rewriter.setInsertionPoint(op);
    auto coorTy = getEleTy(load.getType());
    auto coor = genCoorOp(rewriter, loc, coorTy, lhsEltRefType, load.memref(),
                          load.shape(), load.slice(), update.indices(),
                          load.typeparams(),
                          update->hasAttr(factory::attrFortranArrayOffsets()));
    assignElement(coor);
    return {coor, load.getResult()};
  }

protected:
  const ArrayCopyAnalysis &analysis;
  const OperationUseMapT &useMap;
};

class ArrayUpdateConversion : public ArrayUpdateConversionBase<ArrayUpdateOp> {
public:
  explicit ArrayUpdateConversion(mlir::MLIRContext *ctx,
                                 const ArrayCopyAnalysis &a,
                                 const OperationUseMapT &m)
      : ArrayUpdateConversionBase{ctx, a, m} {}

  mlir::LogicalResult
  matchAndRewrite(ArrayUpdateOp update,
                  mlir::PatternRewriter &rewriter) const override {
    auto loc = update.getLoc();
    auto assignElement = [&](mlir::Value coor) {
      auto input = update.merge();
      if (auto inEleTy = dyn_cast_ptrEleTy(input.getType())) {
        emitFatalError(loc, "array_update on references not supported");
      } else {
        rewriter.create<fir::StoreOp>(loc, input, coor);
      }
    };
    auto lhsEltRefType = toRefType(update.merge().getType());
    auto [_, lhsLoadResult] = materializeAssignment(
        loc, rewriter, update, assignElement, lhsEltRefType);
    update.replaceAllUsesWith(lhsLoadResult);
    rewriter.replaceOp(update, lhsLoadResult);
    return mlir::success();
  }
};

class ArrayModifyConversion : public ArrayUpdateConversionBase<ArrayModifyOp> {
public:
  explicit ArrayModifyConversion(mlir::MLIRContext *ctx,
                                 const ArrayCopyAnalysis &a,
                                 const OperationUseMapT &m)
      : ArrayUpdateConversionBase{ctx, a, m} {}

  mlir::LogicalResult
  matchAndRewrite(ArrayModifyOp modify,
                  mlir::PatternRewriter &rewriter) const override {
    auto loc = modify.getLoc();
    auto assignElement = [](mlir::Value) {
      // Assignment already materialized by lowering using lhs element address.
    };
    auto lhsEltRefType = modify.getResult(0).getType();
    auto [lhsEltCoor, lhsLoadResult] = materializeAssignment(
        loc, rewriter, modify, assignElement, lhsEltRefType);
    modify.replaceAllUsesWith(mlir::ValueRange{lhsEltCoor, lhsLoadResult});
    rewriter.replaceOp(modify, mlir::ValueRange{lhsEltCoor, lhsLoadResult});
    return mlir::success();
  }
};

class ArrayFetchConversion : public mlir::OpRewritePattern<ArrayFetchOp> {
public:
  explicit ArrayFetchConversion(mlir::MLIRContext *ctx,
                                const OperationUseMapT &m)
      : OpRewritePattern{ctx}, useMap{m} {}

  mlir::LogicalResult
  matchAndRewrite(ArrayFetchOp fetch,
                  mlir::PatternRewriter &rewriter) const override {
    auto *op = fetch.getOperation();
    rewriter.setInsertionPoint(op);
    auto load = mlir::cast<ArrayLoadOp>(useMap.lookup(op));
    auto loc = fetch.getLoc();
    auto coor = genCoorOp(
        rewriter, loc, getEleTy(load.getType()), toRefType(fetch.getType()),
        load.memref(), load.shape(), load.slice(), fetch.indices(),
        load.typeparams(), fetch->hasAttr(factory::attrFortranArrayOffsets()));
    if (isa_ref_type(fetch.getType()))
      rewriter.replaceOp(fetch, coor);
    else
      rewriter.replaceOpWithNewOp<fir::LoadOp>(fetch, coor);
    return mlir::success();
  }

private:
  const OperationUseMapT &useMap;
};

/// As array_access op is like an array_fetch op, except that it does not imply
/// a load op. (It operates in the reference domain.)
class ArrayAccessConversion : public ArrayUpdateConversionBase<ArrayAccessOp> {
public:
  explicit ArrayAccessConversion(mlir::MLIRContext *ctx,
                                 const ArrayCopyAnalysis &a,
                                 const OperationUseMapT &m)
      : ArrayUpdateConversionBase{ctx, a, m} {}

  mlir::LogicalResult
  matchAndRewrite(ArrayAccessOp access,
                  mlir::PatternRewriter &rewriter) const override {
    auto *op = access.getOperation();
    auto loc = access.getLoc();
    if (analysis.inAmendAccessSet(op)) {
      // This array_access is associated with an array_amend and there is a
      // conflict. Make a copy to store into.
      auto result = referenceToClone(loc, rewriter, access);
      access.replaceAllUsesWith(result);
      rewriter.replaceOp(access, result);
      return mlir::success();
    }
    rewriter.setInsertionPoint(op);
    auto load = mlir::cast<ArrayLoadOp>(useMap.lookup(op));
    auto coor = genCoorOp(
        rewriter, loc, getEleTy(load.getType()), toRefType(access.getType()),
        load.memref(), load.shape(), load.slice(), access.indices(),
        load.typeparams(), access->hasAttr(factory::attrFortranArrayOffsets()));
    rewriter.replaceOp(access, coor);
    return mlir::success();
  }
};

/// An array_amend op is a marker to record which array access is being used to
/// update an array value. After this pass runs, an array_amend has no
/// semantics. We rewrite these to undefined values here to remove them while
/// preserving SSA form.
class ArrayAmendConversion : public mlir::OpRewritePattern<ArrayAmendOp> {
public:
  explicit ArrayAmendConversion(mlir::MLIRContext *ctx)
      : OpRewritePattern{ctx} {}

  mlir::LogicalResult
  matchAndRewrite(ArrayAmendOp amend,
                  mlir::PatternRewriter &rewriter) const override {
    auto *op = amend.getOperation();
    rewriter.setInsertionPoint(op);
    auto loc = amend.getLoc();
    auto undef = rewriter.create<UndefOp>(loc, amend.getType());
    rewriter.replaceOp(amend, undef.getResult());
    return mlir::success();
  }
};

class ArrayValueCopyConverter
    : public ArrayValueCopyBase<ArrayValueCopyConverter> {
public:
  void runOnFunction() override {
    auto func = getFunction();
    LLVM_DEBUG(llvm::dbgs() << "\n\narray-value-copy pass on function '"
                            << func.getName() << "'\n");
    auto *context = &getContext();

    // Perform the conflict analysis.
    const auto &analysis = getAnalysis<ArrayCopyAnalysis>();
    const auto &useMap = analysis.getUseMap();

    mlir::OwningRewritePatternList patterns1(context);
    patterns1.insert<ArrayFetchConversion>(context, useMap);
    patterns1.insert<ArrayUpdateConversion>(context, analysis, useMap);
    patterns1.insert<ArrayModifyConversion>(context, analysis, useMap);
    patterns1.insert<ArrayAccessConversion>(context, analysis, useMap);
    patterns1.insert<ArrayAmendConversion>(context);
    mlir::ConversionTarget target(*context);
    target.addLegalDialect<FIROpsDialect, mlir::scf::SCFDialect,
                           mlir::arith::ArithmeticDialect,
                           mlir::StandardOpsDialect>();
    target.addIllegalOp<ArrayAccessOp, ArrayAmendOp, ArrayFetchOp,
                        ArrayUpdateOp, ArrayModifyOp>();
    // Rewrite the array fetch and array update ops.
    if (mlir::failed(
            mlir::applyPartialConversion(func, target, std::move(patterns1)))) {
      mlir::emitError(mlir::UnknownLoc::get(context),
                      "failure in array-value-copy pass, phase 1");
      signalPassFailure();
    }

    mlir::OwningRewritePatternList patterns2(context);
    patterns2.insert<ArrayLoadConversion>(context);
    patterns2.insert<ArrayMergeStoreConversion>(context);
    target.addIllegalOp<ArrayLoadOp, ArrayMergeStoreOp>();
    if (mlir::failed(
            mlir::applyPartialConversion(func, target, std::move(patterns2)))) {
      mlir::emitError(mlir::UnknownLoc::get(context),
                      "failure in array-value-copy pass, phase 2");
      signalPassFailure();
    }
  }
};
} // namespace

std::unique_ptr<mlir::Pass> fir::createArrayValueCopyPass() {
  return std::make_unique<ArrayValueCopyConverter>();
}
