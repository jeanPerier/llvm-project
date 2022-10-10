//===- ShapeInference.cpp - Propagate shape information -===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// TBD
//
//===----------------------------------------------------------------------===//

#include "flang/Optimizer/Builder/FIRBuilder.h"
#include "flang/Optimizer/Dialect/FIRDialect.h"
#include "flang/Optimizer/Dialect/FIROps.h"
#include "flang/Optimizer/Dialect/FIRType.h"
#include "flang/Optimizer/Transforms/Passes.h"
// #include "mlir/Interfaces/InferTypeOpInterface.h"
// #include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "flang/Optimizer/Builder/Todo.h"
#include "flang/Optimizer/Support/FIRContext.h"
#include "flang/Optimizer/Support/Matcher.h"
#include "mlir/IR/BlockAndValueMapping.h"
#include "mlir/Transforms/DialectConversion.h"
#include "llvm/ADT/SmallSet.h"

namespace fir {
#define GEN_PASS_DEF_FORALLSPLIT
#include "flang/Optimizer/Transforms/Passes.h.inc"
} // namespace fir

// Note: the patte
bool inline isForall(fir::DoLoopOp loop) { return loop->hasAttr("fir.forall"); }

void inline removeForallLabel(fir::DoLoopOp loop) {
  loop->removeAttr("fir.forall");
}

bool inline isTopLevelForall(fir::DoLoopOp loop) {
  if (!isForall(loop))
    return false;
  auto *owner = loop->getParentOp();
  while (owner) {
    if (auto parentLoop = mlir::dyn_cast<fir::DoLoopOp>(owner))
      if (isForall(parentLoop))
        return false;
    owner = owner->getParentOp();
  }
  return true;
}

namespace {

// Note: memory dependencies are dropped on the floor here...

// llvm::SmallPtrSet<mlir::Operation*> gatherOpsToClone(mlir::Operation* leafOp,
// mlir::Operation* parentOp) {
//   llvm::SmallPtrSet<mlir::Operation*> toBeCloned;
//   llvm::SmallVector<mlir::Operation*> toBeVisited{leafOp};
//   // Start by adding adding all parentOp until parentOp;
//   mlir::Operation* owner = leafOp->getParentOp();
//   while (owner && owner != parentOp) {
//     toBeVisited.push_back(owner);
//     owner = owner->getParentOp();
//   }
//   while (!toBeVisited.empty()) {
//     mlir::Operation* op = toBeVisited.pop_back_val();
//     if (toBeCloned.contains(op))
//      continue;
//     toBeCloned.insert(op);
//     for (mlir::Value operand = op->getOperands())
//       if (mlir::Operation* definingOp = operand.getDefiningOp())
//         if (parentOp->isProperAncestor(definingOp))
//           toBeVisited.push_back(definingOp);
//   }
// }

void cloneAssignmentIntoItsOwnLoopNest(fir::AssignOp assignment,
                                       fir::DoLoopOp loop,
                                       mlir::PatternRewriter &rewriter) {
  // How can something be cloned while still preserving the order....
  // auto toBeCloned;
  mlir::BlockAndValueMapping mapper;
  auto *newLoop = rewriter.clone(*loop.getOperation(), mapper);
  assert(newLoop && "failed to clone");
  int currentPosition = -1;
  int position = -1;
  loop->walk([&](mlir::Operation *op) {
    if (auto thatAssignment = mlir::dyn_cast<fir::AssignOp>(op)) {
      ++currentPosition;
      if (op == assignment.getOperation())
        position = currentPosition;
    }
  });

  llvm::SmallVector<fir::AssignOp> toErase;
  llvm::SmallVector<fir::DoLoopOp> removeLabel;
  currentPosition = -1;
  newLoop->walk([&](mlir::Operation *op) {
    if (auto assignment = mlir::dyn_cast<fir::AssignOp>(op)) {
      ++currentPosition;
      if (currentPosition != position)
        toErase.push_back(assignment);
    } else if (auto doLoop = mlir::dyn_cast<fir::DoLoopOp>(op)) {
      if (isForall(doLoop))
        removeLabel.push_back(doLoop);
    }
  });
  for (fir::AssignOp otherAssignment : toErase)
    rewriter.eraseOp(otherAssignment);
  for (fir::DoLoopOp newNestedLoop : removeLabel)
    removeForallLabel(newNestedLoop);
  removeForallLabel(mlir::cast<fir::DoLoopOp>(newLoop));
  // RUN DCE manually ??
  // walk + clone ?
}

void cloneAssignmentsIntoItsOwnLoopNest(
    llvm::SmallVector<fir::AssignOp> assignments, fir::DoLoopOp loop,
    mlir::PatternRewriter &rewriter) {
  // How can something be cloned while still preserving the order....
  // auto toBeCloned;
  mlir::BlockAndValueMapping mapper;
  auto *newLoop = rewriter.clone(*loop.getOperation(), mapper);
  assert(newLoop && "failed to clone");
  int currentPosition = -1;
  llvm::SmallSet<int, 8> assignmentPositions;
  llvm::SmallVector<fir::AssignOp> toEraseInSecondLoop;
  loop->walk([&](mlir::Operation *op) {
    if (auto thatAssignment = mlir::dyn_cast<fir::AssignOp>(op)) {
      ++currentPosition;
      for (auto assignment : assignments)
        if (op == assignment.getOperation()) {
          assignmentPositions.insert(currentPosition);
          toEraseInSecondLoop.push_back(thatAssignment);
          break;
        }
    }
  });

  llvm::SmallVector<fir::AssignOp> toErase;
  llvm::SmallVector<fir::DoLoopOp> removeLabel;
  currentPosition = -1;
  newLoop->walk([&](mlir::Operation *op) {
    if (auto assignment = mlir::dyn_cast<fir::AssignOp>(op)) {
      ++currentPosition;
      if (!assignmentPositions.contains(currentPosition))
        toErase.push_back(assignment);
    } else if (auto doLoop = mlir::dyn_cast<fir::DoLoopOp>(op)) {
      if (isForall(doLoop))
        removeLabel.push_back(doLoop);
    }
  });
  for (fir::AssignOp otherAssignment : toErase)
    rewriter.eraseOp(otherAssignment);
  for (fir::AssignOp otherAssignment : toEraseInSecondLoop)
    rewriter.eraseOp(otherAssignment);
  for (fir::DoLoopOp newNestedLoop : removeLabel)
    removeForallLabel(newNestedLoop);
  removeForallLabel(mlir::cast<fir::DoLoopOp>(newLoop));
  // RUN DCE manually ??
  // walk + clone ?
}

/// Assignment analysis
/// 1. Get base and all dependency "roots"
/// 2. Overlap analysis between bases and dependencies.
/// 3. Temp introductions.

struct VariableDirectReference {
  bool mayAccess(const VariableDirectReference &) const { return true; }
  mlir::Operation *varDeclaration;
  // TBD accesses like indices ?
  // TBD opaque function reference may access non local variables.
};

struct FunctionCall {
  bool mayAccess(const VariableDirectReference &) const { return true; }
  fir::CallOp call;
};

struct UnknownAccess {
  bool mayAccess(const VariableDirectReference &) const { return true; }
  mlir::Value accessResult;
};

class VariableAccess : public fir::details::matcher<VariableAccess> {
public:
  using AccessType =
      std::variant<VariableDirectReference, FunctionCall, UnknownAccess>;
  bool mayAccess(const VariableDirectReference &var) const {
    return match([&](const auto &access) { return access.mayAccess(var); });
  }

  template <typename A>
  VariableAccess(A &&a) : variableAccess{std::forward<A>(a)} {}
  const AccessType &matchee() const { return variableAccess; }

private:
  AccessType variableAccess;
};

struct AssignmentAliasing {
  VariableDirectReference lhsRef;
  llvm::SmallVector<std::pair<mlir::Value, llvm::SmallVector<VariableAccess>>>
      variableAccesses;
};

AssignmentAliasing analyzeAssignment(fir::AssignOp assignment,
                                     mlir::Operation *outterForall) {
  llvm::SmallVector<mlir::Value> primaryOperandsToBeAnalyzed = {
      assignment.getValue()};
  mlir::Operation *lhs = assignment.getVar().getDefiningOp();
  while (lhs) {
    if (auto designateOp = mlir::dyn_cast<fir::DesignateOp>(lhs)) {
      // TODO: stop here if LHS is a pointer component.
      lhs = designateOp.getVar().getDefiningOp();
      primaryOperandsToBeAnalyzed.append(designateOp.getIndices().begin(),
                                         designateOp.getIndices().end());
    } else if (auto declareOp = mlir::dyn_cast<fir::DeclareOp>(lhs)) {
      break;
    } else {
      lhs = nullptr;
    }
  }
  if (!lhs)
    fir::emitFatalError(assignment.getLoc(), "unexpected LHS source");

  // Add forall/if op operands.
  if (outterForall) {
    mlir::Operation *ownerOp = assignment->getParentOp();
    while (ownerOp && outterForall->isProperAncestor(ownerOp)) {
      primaryOperandsToBeAnalyzed.append(ownerOp->getOperands().begin(),
                                         ownerOp->getOperands().end());
      ownerOp = ownerOp->getParentOp();
    }
  }

  llvm::SmallVector<std::pair<mlir::Value, llvm::SmallVector<VariableAccess>>>
      variableAccesses;
  for (mlir::Value primaryOperand : primaryOperandsToBeAnalyzed) {
    llvm::SmallVector<mlir::Value> operandsToBeAnalyzed = {primaryOperand};
    variableAccesses.emplace_back(
        std::make_pair(primaryOperand, llvm::SmallVector<VariableAccess>{}));
    while (!operandsToBeAnalyzed.empty()) {
      auto operand = operandsToBeAnalyzed.pop_back_val();
      if (mlir::Operation *parentOp = operand.getDefiningOp()) {
        const bool opIsInsideForall =
            outterForall && outterForall->isProperAncestor(parentOp);
        // If the operand is not an address and is defined above the outer
        // forall, its value is not impacted by the assignment.
        if (!opIsInsideForall && fir::isa_trivial(operand.getType()))
          continue;
        if (auto designateOp = mlir::dyn_cast<fir::DesignateOp>(parentOp)) {
          // TODO: Add access if LHS is a pointer component.
        } else if (auto declareOp = mlir::dyn_cast<fir::DeclareOp>(parentOp)) {
          variableAccesses.back().second.emplace_back(
              VariableDirectReference{parentOp});
          if (opIsInsideForall)
            TODO(parentOp->getLoc(), "handle declare_op inside Forall");
          continue;
        } else if (auto callOp = mlir::dyn_cast<fir::CallOp>(parentOp)) {
          if (opIsInsideForall)
            variableAccesses.back().second.emplace_back(FunctionCall{callOp});
        }
        if (parentOp->getNumRegions() > 0)
          TODO(parentOp->getLoc(), "alias analysis of operations with regions");
        // TODO: deal with loads/stores ? What if op has side effects like calls
        // ?
        operandsToBeAnalyzed.append(parentOp->getOperands().begin(),
                                    parentOp->getOperands().end());
      } else {
        auto blockArgument = operand.cast<mlir::BlockArgument>();
        mlir::Block *ownerBlock = blockArgument.getOwner();
        mlir::Operation *parentOp2 = ownerBlock->getParentOp();
        const bool opIsInsideForall =
            outterForall && outterForall->isProperAncestor(parentOp2);
        if (!opIsInsideForall && fir::isa_trivial(operand.getType()))
          continue;
        if (ownerBlock->isEntryBlock()) {
          if (auto doLoop = mlir::dyn_cast<fir::DoLoopOp>(parentOp2))
            if (isForall(doLoop))
              continue;
        }
        TODO(operand.getLoc(), "unknown block argument used in assignment");
        // TODO: block operation, check if forall index, otherwise
        // temporize/complain ?
      }
    }
  }
  return AssignmentAliasing{VariableDirectReference{lhs},
                            std::move(variableAccesses)};
}

bool requiresTemporaries(const AssignmentAliasing &assignmentAnalysis) {
  for (const auto &primary : assignmentAnalysis.variableAccesses)
    if (!primary.second.empty())
      return true;
  return false;
}

mlir::Operation *getParentOrDefinigOp(mlir::Value value) {
  if (mlir::Operation *definingOp = value.getDefiningOp())
    return definingOp;
  return value.getParentBlock()->getParentOp();
}

bool isDefinedIn(mlir::Value value, mlir::Operation *op) {
  return op && op->isAncestor(getParentOrDefinigOp(value));
}

// Temp: Options:
//  - Perfect match (Forall bounds and primary shape can be deduced)
//  - Maximizing: Forall bounds and primary shape can be maximized
//  - LHS temporization
//  - Allocatable temps model.
//  - Pre-run (compute number of iterations)
//      - shape/typeparams loop independent.
//      - shape/type.

// Let's do the allocatable temp model with a pre-count of the iteration number.

// Step 1. pre-count:
// - Duplicate loop and remove all assignments/extra loop.
// - alloca index and increment it.

/// True IFF all data about this value can be inferred:
/// -> type (dynamic type)
/// -> rank (or assumed rank)
/// -> shape (if not assumed rank, and last extent might be undefined for
/// assumed size arrays)
/// -> type parameters
// bool isFortranObject(mlir::Value) {
// };
//
// bool isArray(mlir::Value) {
// };
//
// bool hasLengthParameters(mlir::Value) {
// };
//
// bool computeShape(mlir::Value);

/// (i:i+2) -> shape is obviously two.... How can we compute that in the general
/// case and hoist that out of the forall if needed. compute + fold inside the
/// loop, then "hoist" ? Imagine we can hoist that. How do we then make a link
/// with this value ? Need to use it somewhere.... in the designate ?

// Step 2: create type:
// Step 3: allocate + index

// class ForallAllocatableBasedTemp {
//   void create(mlir::Value, mlir::PatternRewriter&);
//   void resetAddressing(mlir::PatternRewriter&);
//   SmallPtrSetImpl<mlir::Operation *> storeValueAt(mlir::Value,
//   mlir::ValueRange forallIndices, mlir::PatternRewriter&); mlir::Value
//   loadValueAt(mlir::ValueRange forallIndices, mlir::PatternRewriter&); void
//   cleanUpAt(mlir::ValueRange forallIndices); void
//   cleanUp(mlir::PatternRewriter&);
// private:
//   mlir::Value index;
//   mlir::Value boxArrayStorage;
// };

// class ForallArrayTemp {
// };
//
//// TODO:
// class ForallPointerTemp {
// };
//
// class ForallTemp {
// public:
//   static ForallTemp createForallTemp(mlir::Value value);
// private:
//   std::variant<>
// };

llvm::SmallVector<fir::AssignOp>
insertTemporaries(const AssignmentAliasing &assignmentAnalysis,
                  mlir::Operation *outterForall,
                  mlir::PatternRewriter &rewriter) {
  auto module = assignmentAnalysis.lhsRef.varDeclaration
                    ->getParentOfType<mlir::ModuleOp>();
  fir::FirOpBuilder builder(rewriter, fir::getKindMapping(module));
  mlir::Type idxTy = builder.getIndexType();
  llvm::SmallVector<fir::AssignOp> tempAssignments;

  for (const auto &primary : assignmentAnalysis.variableAccesses)
    if (!primary.second.empty()) {
      mlir::Value value = primary.first;
      mlir::Location loc = value.getLoc();
      // For now, temporize the primary. More clever analysis could allow making
      // smaller/less temporaries by trying to find smaller SSA values that are
      // in the DAG between the actual dependency and the resulting SSA value
      // used in the assignment (called the "primary" here). Several primaries
      // may also depends on a common SSA value.

      // Step 1: compute the primary size.
      mlir::Type elementType;
      if (fir::conformsWithPassByRef(value.getType()))
        TODO(loc, "temporize from memory");
      if (!fir::isa_trivial(value.getType()))
        TODO(loc, "temporize from variable or expression, or weird type");
      elementType = value.getType();
      // Step 2: compute the iteration size at the point of definition of the
      // primary.
      llvm::SmallVector<mlir::Value> temporaryShape;
      mlir::Operation *owner = getParentOrDefinigOp(value);
      while (owner && outterForall && outterForall->isAncestor(owner)) {
        if (auto doLoop = mlir::dyn_cast<fir::DoLoopOp>(owner))
          if (isForall(doLoop)) {
            mlir::Value lb, ub, step;
            lb = doLoop.getLowerBound();
            ub = doLoop.getUpperBound();
            step = doLoop.getStep();
            if (!isDefinedIn(lb, outterForall) &&
                !isDefinedIn(ub, outterForall) &&
                !isDefinedIn(step, outterForall)) {
              mlir::Value extent =
                  builder.genExtentFromTriplet(loc, lb, ub, step, idxTy);
              temporaryShape.push_back(extent);
            } else {
              TODO(value.getLoc(), "Forall temp with forall bounds depending "
                                   "on outter Forall indices");
            }
          }
        owner = owner->getParentOp();
      }
      // Step 3: insert temp creation + clean-up.
      rewriter.setInsertionPoint(outterForall);
      if (temporaryShape.empty())
        TODO(loc, "temporize outside of forall");
      mlir::Type tempType =
          builder.getVarLenSeqTy(elementType, temporaryShape.size());
      auto temp = builder.create<fir::AllocMemOp>(
          loc, tempType, /*typeParams=*/llvm::None, temporaryShape);
      auto shapeType =
          fir::ShapeType::get(rewriter.getContext(), temporaryShape.size());
      auto shape = builder.create<fir::ShapeOp>(loc, shapeType, temporaryShape);
      mlir::Type varType = fir::VarType::get(tempType);
      auto tempVar = builder.create<fir::DeclareOp>(
          loc, varType, temp, shape, /*typeParams*/ llvm::None,
          /*fortran_attrs=*/fir::FortranVariableFlagsAttr{});
      // This assumes this is the latest Forall. Is this true ?
      builder.setInsertionPointAfter(outterForall);
      // TODO: proper finalization if needed ?
      builder.create<fir::FreeMemOp>(loc, temp);
      // Step 4: assign primary to temp, and replace primary usages by load from
      // the temp. Address temp. TODO: Needs to account for Forall bounds. Or
      // use Forall zero/one base index..
      llvm::SmallVector<mlir::Value> inductionValues;
      owner = getParentOrDefinigOp(value);
      while (owner && outterForall && outterForall->isAncestor(owner)) {
        if (auto doLoop = mlir::dyn_cast<fir::DoLoopOp>(owner))
          if (isForall(doLoop))
            inductionValues.push_back(doLoop.getInductionVar());
        owner = owner->getParentOp();
      }
      if (mlir::Operation *definingOp = value.getDefiningOp())
        builder.setInsertionPointAfter(definingOp);
      else
        builder.setInsertionPointToStart(value.getParentBlock());

      mlir::Type varEleType = fir::VarType::get(elementType);
      auto tempEltWrite = builder.create<fir::DesignateOp>(
          loc, varEleType, tempVar, inductionValues);
      auto tempAssignment =
          builder.create<fir::AssignOp>(loc, value, tempEltWrite);
      tempAssignments.push_back(tempAssignment);
      auto tempEltRead = builder.create<fir::DesignateOp>(
          loc, varEleType, tempVar, inductionValues);
      auto newValue =
          builder.create<fir::AsValueOp>(loc, elementType, tempEltRead);
      value.replaceAllUsesExcept(newValue, tempAssignment.getOperation());
    }
  return tempAssignments;
}

class ForallConversion : public mlir::OpRewritePattern<fir::DoLoopOp> {
public:
  explicit ForallConversion(mlir::MLIRContext *ctx) : OpRewritePattern{ctx} {}

  mlir::LogicalResult
  matchAndRewrite(fir::DoLoopOp loop,
                  mlir::PatternRewriter &rewriter) const override {
    // Gather forall assignments assignments:
    llvm::SmallVector<fir::AssignOp> assignments;
    loop->walk([&](mlir::Operation *op) {
      if (auto assignment = mlir::dyn_cast<fir::AssignOp>(op))
        assignments.push_back(assignment);
    });

    rewriter.startRootUpdate(loop);
    for (auto iter : llvm::enumerate(assignments)) {
      fir::AssignOp assignment = iter.value();
      auto analysis = analyzeAssignment(assignment, loop.getOperation());
      if (requiresTemporaries(analysis)) {
        auto tempAssignments =
            insertTemporaries(analysis, loop.getOperation(), rewriter);
        cloneAssignmentsIntoItsOwnLoopNest(tempAssignments, loop, rewriter);
      }
      if (iter.index() + 1 < assignments.size())
        cloneAssignmentsIntoItsOwnLoopNest({assignment}, loop, rewriter);
      // llvm::errs() << "listing conflicts with: " <<
      // *analysis.lhsRef.varDeclaration << "\n"; for (const auto& primary :
      // analysis.variableAccesses) {
      //   llvm::errs() << "  from: "<< primary.first << "\n";
      //   for (const auto& access : primary.second) {
      //     access.match(
      //         [](const VariableDirectReference& ref){
      //           llvm::errs() << "    ref: "<< *ref.varDeclaration << "\n";
      //         },
      //         [](const FunctionCall& call){
      //           llvm::errs() << "    call: "<< call.call << "\n";
      //         },
      //         [](const UnknownAccess& unknown){
      //           llvm::errs() << "    unknown: "<< unknown.accessResult <<
      //           "\n";
      //         }
      //     );
      //   }
      // }
    }
    loop->walk([&](mlir::Operation *op) {
      if (auto doLoop = mlir::dyn_cast<fir::DoLoopOp>(op))
        if (isForall(doLoop))
          removeForallLabel(doLoop);
    });
    rewriter.finalizeRootUpdate(loop);
    //    rewriter.setInsertionPoint(loop);
    //    // STUPID TEST: let's split assignments into their own FORALL (note:
    //    // not valid if where depends on previously assigned values).
    //    for (fir::AssignOp assignment : assignments)
    //      cloneAssignmentIntoItsOwnLoopNest(assignment, loop, rewriter);
    //    // Note: this assumes FORALL do not have results.
    //    rewriter.eraseOp(loop);
    return mlir::success();
  }
};
} // namespace

namespace {
struct ForallSplitPass final
    : public fir::impl::ForallSplitBase<ForallSplitPass> {
  void runOnOperation() override {
    auto func = getOperation();
    auto *context = &getContext();
    mlir::RewritePatternSet patterns(context);
    patterns.insert<ForallConversion>(context);
    mlir::ConversionTarget target(*context);
    target.addDynamicallyLegalOp<fir::DoLoopOp>(
        [](fir::DoLoopOp loop) { return !isTopLevelForall(loop); });
    target.markUnknownOpDynamicallyLegal(
        [](mlir::Operation *) { return true; });
    if (mlir::failed(
            mlir::applyPartialConversion(func, target, std::move(patterns)))) {
      mlir::emitError(mlir::UnknownLoc::get(context),
                      "failure in forall split pass");
      signalPassFailure();
    }
  }
};

} // namespace

std::unique_ptr<mlir::Pass> fir::createForallSplitPass() {
  return std::make_unique<ForallSplitPass>();
}
