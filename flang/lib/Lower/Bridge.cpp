//===-- Bridge.cc -- bridge to lower to MLIR ------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Coding style: https://mlir.llvm.org/getting_started/DeveloperGuide/
//
//===----------------------------------------------------------------------===//

#include "flang/Lower/Bridge.h"
#include "../../runtime/iostat.h"
#include "BoxAnalyzer.h"
#include "StatementContext.h"
#include "flang/Lower/Allocatable.h"
#include "flang/Lower/CallInterface.h"
#include "flang/Lower/CharacterExpr.h"
#include "flang/Lower/Coarray.h"
#include "flang/Lower/ConvertExpr.h"
#include "flang/Lower/ConvertType.h"
#include "flang/Lower/FIRBuilder.h"
#include "flang/Lower/IO.h"
#include "flang/Lower/Mangler.h"
#include "flang/Lower/OpenACC.h"
#include "flang/Lower/OpenMP.h"
#include "flang/Lower/PFTBuilder.h"
#include "flang/Lower/Runtime.h"
#include "flang/Lower/Support/BoxValue.h"
#include "flang/Lower/Support/Utils.h"
#include "flang/Lower/Todo.h"
#include "flang/Optimizer/Dialect/FIRAttr.h"
#include "flang/Optimizer/Dialect/FIRDialect.h"
#include "flang/Optimizer/Dialect/FIROps.h"
#include "flang/Optimizer/Support/FIRContext.h"
#include "flang/Optimizer/Support/FatalError.h"
#include "flang/Optimizer/Support/InternalNames.h"
#include "flang/Optimizer/Transforms/Passes.h"
#include "flang/Parser/parse-tree.h"
#include "flang/Semantics/tools.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/StandardOps/IR/Ops.h"
#include "mlir/Parser.h"
#include "mlir/Target/LLVMIR.h"
#include "mlir/Transforms/RegionUtils.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/MD5.h"

#define DEBUG_TYPE "flang-lower-bridge"

static llvm::cl::opt<bool> dumpBeforeFir(
    "fdebug-dump-pre-fir", llvm::cl::init(false),
    llvm::cl::desc("dump the Pre-FIR tree prior to FIR generation"));

static llvm::cl::opt<std::size_t>
    nameLengthHashSize("length-to-hash-string-literal",
                       llvm::cl::desc("string literals that exceed this length"
                                      " will use a hash value as their symbol "
                                      "name"),
                       llvm::cl::init(32));

namespace {
/// Information for generating a structured or unstructured increment loop.
struct IncrementLoopInfo {
  template <typename T>
  explicit IncrementLoopInfo(Fortran::semantics::Symbol &sym, const T &lower,
                             const T &upper, const std::optional<T> &step,
                             bool isUnordered = false)
      : loopVariableSym{sym}, lowerExpr{Fortran::semantics::GetExpr(lower)},
        upperExpr{Fortran::semantics::GetExpr(upper)},
        stepExpr{Fortran::semantics::GetExpr(step)}, isUnordered{isUnordered} {}

  IncrementLoopInfo(IncrementLoopInfo &&) = default;
  IncrementLoopInfo &operator=(IncrementLoopInfo &&x) { return x; }

  bool isStructured() const { return !headerBlock; }

  // Data members common to both structured and unstructured loops.
  const Fortran::semantics::Symbol &loopVariableSym;
  const Fortran::semantics::SomeExpr *lowerExpr;
  const Fortran::semantics::SomeExpr *upperExpr;
  const Fortran::semantics::SomeExpr *stepExpr;
  const Fortran::semantics::SomeExpr *maskExpr = nullptr;
  bool isUnordered; // do concurrent, forall
  llvm::SmallVector<const Fortran::semantics::Symbol *, 4> localInitSymList;
  mlir::Value loopVariable = nullptr;
  mlir::Value stepValue = nullptr; // possible uses in multiple blocks

  // Data members for structured loops.
  fir::DoLoopOp doLoop = nullptr;

  // Data members for unstructured loops.
  bool hasRealControl = false;
  mlir::Value tripVariable = nullptr;
  mlir::Block *headerBlock = nullptr; // loop entry and test block
  mlir::Block *maskBlock = nullptr;   // concurrent loop mask block
  mlir::Block *bodyBlock = nullptr;   // first loop body block
  mlir::Block *exitBlock = nullptr;   // loop exit target block
};

using IncrementLoopNestInfo = llvm::SmallVector<IncrementLoopInfo, 4>;
} // namespace

// Retrieve a copy of a character literal string from a SomeExpr.
template <int KIND>
static llvm::Optional<std::tuple<std::string, std::size_t>>
getCharacterLiteralCopy(
    const Fortran::evaluate::Expr<
        Fortran::evaluate::Type<Fortran::common::TypeCategory::Character, KIND>>
        &x) {
  if (const auto *con =
          Fortran::evaluate::UnwrapConstantValue<Fortran::evaluate::Type<
              Fortran::common::TypeCategory::Character, KIND>>(x))
    if (auto val = con->GetScalarValue())
      return std::tuple<std::string, std::size_t>{
          std::string{(const char *)val->c_str(),
                      KIND * (std::size_t)con->LEN()},
          (std::size_t)con->LEN()};
  return llvm::None;
}
static llvm::Optional<std::tuple<std::string, std::size_t>>
getCharacterLiteralCopy(
    const Fortran::evaluate::Expr<Fortran::evaluate::SomeCharacter> &x) {
  return std::visit([](const auto &e) { return getCharacterLiteralCopy(e); },
                    x.u);
}
static llvm::Optional<std::tuple<std::string, std::size_t>>
getCharacterLiteralCopy(
    const Fortran::evaluate::Expr<Fortran::evaluate::SomeType> &x) {
  if (const auto *e = Fortran::evaluate::UnwrapExpr<
          Fortran::evaluate::Expr<Fortran::evaluate::SomeCharacter>>(x))
    return getCharacterLiteralCopy(*e);
  return llvm::None;
}
template <typename A>
static llvm::Optional<std::tuple<std::string, std::size_t>>
getCharacterLiteralCopy(const std::optional<A> &x) {
  if (x)
    return getCharacterLiteralCopy(*x);
  return llvm::None;
}

//===----------------------------------------------------------------------===//
// FirConverter
//===----------------------------------------------------------------------===//

namespace {
/// Walk over the pre-FIR tree (PFT) and lower it to the FIR dialect of MLIR.
///
/// After building the PFT, the FirConverter processes that representation
/// and lowers it to the FIR executable representation.
class FirConverter : public Fortran::lower::AbstractConverter {
public:
  explicit FirConverter(Fortran::lower::LoweringBridge &bridge,
                        fir::NameUniquer &uniquer)
      : bridge{bridge}, uniquer{uniquer}, foldingContext{
                                              bridge.createFoldingContext()} {}
  virtual ~FirConverter() = default;

  /// Convert the PFT to FIR
  void run(Fortran::lower::pft::Program &pft) {
    // Declare mlir::FuncOp for all the FunctionLikeUnit defined in the PFT
    // before lowering any function bodies so that the definition signatures
    // prevail on call spot signatures.
    declareFunctions(pft);

    // Define variables of the modules defined in this program. This is done
    // first to ensure they are defined before lowering any function that may
    // use them.
    lowerModuleVariables(pft);
    // do translation
    for (auto &u : pft.getUnits()) {
      std::visit(
          Fortran::common::visitors{
              [&](Fortran::lower::pft::FunctionLikeUnit &f) { lowerFunc(f); },
              [&](Fortran::lower::pft::ModuleLikeUnit &m) { lowerMod(m); },
              [&](Fortran::lower::pft::BlockDataUnit &b) { lowerBlockData(b); },
              [&](Fortran::lower::pft::CompilerDirectiveUnit &d) {
                setCurrentPosition(
                    d.get<Fortran::parser::CompilerDirective>().source);
                mlir::emitWarning(toLocation(),
                                  "ignoring all compiler directives");
              },
          },
          u);
    }
  }

  /// Declare mlir::FuncOp for all the FunctionLikeUnit defined in the PFT
  /// without any other side-effects.
  void declareFunctions(Fortran::lower::pft::Program &pft) {
    for (auto &u : pft.getUnits()) {
      std::visit(Fortran::common::visitors{
                     [&](Fortran::lower::pft::FunctionLikeUnit &f) {
                       declareFunction(f);
                     },
                     [&](Fortran::lower::pft::ModuleLikeUnit &m) {
                       for (auto &f : m.nestedFunctions)
                         declareFunction(f);
                     },
                     [&](Fortran::lower::pft::BlockDataUnit &) {
                       // No functions defined in block data.
                     },
                     [&](Fortran::lower::pft::CompilerDirectiveUnit &) {
                       // No functions defined.
                     },
                 },
                 u);
    }
  }
  void declareFunction(Fortran::lower::pft::FunctionLikeUnit &funit) {
    for (int entryIndex = 0, last = funit.entryPointList.size();
         entryIndex < last; ++entryIndex) {
      funit.setActiveEntry(entryIndex);
      // Calling CalleeInterface ctor will build the mlir::FuncOp with no other
      // side effects.
      // TODO: when doing some compiler profiling on real apps, it may be worth
      // to check it's better to save the CalleeInterface instead of recomputing
      // it later when lowering the body. CalleeInterface ctor should be linear
      // with the number of arguments, so it is not awful to do it that way for
      // now, but the linear coefficient might be non negligible. Until
      // measured, stick to the solution that impacts the code less.
      Fortran::lower::CalleeInterface{funit, *this};
    }
    funit.setActiveEntry(0);
    for (auto &f : funit.nestedFunctions)
      declareFunction(f); // internal procedure
  }

  /// Loop through modules defined in this file to generate the fir::globalOp
  /// for module variables.
  void lowerModuleVariables(Fortran::lower::pft::Program &pft) {
    for (auto &u : pft.getUnits()) {
      std::visit(Fortran::common::visitors{
                     [&](Fortran::lower::pft::ModuleLikeUnit &m) {
                       lowerModuleVariables(m);
                     },
                     [](auto &) {
                       // Not a module, so no processing needed here.
                     },
                 },
                 u);
    }
  }

  //===--------------------------------------------------------------------===//
  // AbstractConverter overrides
  //===--------------------------------------------------------------------===//

  mlir::Value getSymbolAddress(Fortran::lower::SymbolRef sym) override final {
    return lookupSymbol(sym).getAddr();
  }

  bool lookupLabelSet(Fortran::lower::SymbolRef sym,
                      Fortran::lower::pft::LabelSet &labelSet) override final {
    auto &owningProc = *getEval().getOwningProcedure();
    auto iter = owningProc.assignSymbolLabelMap.find(sym);
    if (iter == owningProc.assignSymbolLabelMap.end())
      return false;
    labelSet = iter->second;
    return true;
  }

  Fortran::lower::pft::Evaluation *
  lookupLabel(Fortran::lower::pft::Label label) override final {
    auto &owningProc = *getEval().getOwningProcedure();
    auto iter = owningProc.labelEvaluationMap.find(label);
    if (iter == owningProc.labelEvaluationMap.end())
      return nullptr;
    return iter->second;
  }

  fir::ExtendedValue genExprAddr(const Fortran::lower::SomeExpr &expr,
                                 Fortran::lower::StatementContext &context,
                                 mlir::Location *loc = nullptr) override final {
    return createSomeExtendedAddress(loc ? *loc : toLocation(), *this, expr,
                                     localSymbols, context);
  }
  fir::ExtendedValue
  genExprValue(const Fortran::lower::SomeExpr &expr,
               Fortran::lower::StatementContext &context,
               mlir::Location *loc = nullptr) override final {
    return createSomeExtendedExpression(loc ? *loc : toLocation(), *this, expr,
                                        localSymbols, context);
  }
  fir::MutableBoxValue
  genExprMutableBox(mlir::Location loc,
                    const Fortran::lower::SomeExpr &expr) override final {
    return createSomeMutableBox(loc, *this, expr, localSymbols);
  }

  Fortran::evaluate::FoldingContext &getFoldingContext() override final {
    return foldingContext;
  }

  mlir::Type genType(const Fortran::lower::SomeExpr &expr) override final {
    return Fortran::lower::translateSomeExprToFIRType(*this, expr);
  }
  mlir::Type genType(const Fortran::lower::pft::Variable &var) override final {
    return Fortran::lower::translateVariableToFIRType(*this, var);
  }
  mlir::Type genType(Fortran::lower::SymbolRef sym) override final {
    return Fortran::lower::translateSymbolToFIRType(*this, sym);
  }
  mlir::Type
  genType(Fortran::common::TypeCategory tc, int kind,
          llvm::ArrayRef<std::int64_t> lenParameters) override final {
    return Fortran::lower::getFIRType(&getMLIRContext(), tc, kind,
                                      lenParameters);
  }
  mlir::Type genType(Fortran::common::TypeCategory tc) override final {
    return Fortran::lower::getFIRType(
        &getMLIRContext(), tc, bridge.getDefaultKinds().GetDefaultKind(tc),
        llvm::None);
  }

  mlir::Location getCurrentLocation() override final { return toLocation(); }

  /// Generate a dummy location.
  mlir::Location genLocation() override final {
    // Note: builder may not be instantiated yet
    return mlir::UnknownLoc::get(&getMLIRContext());
  }

  /// Generate a `Location` from the `CharBlock`.
  mlir::Location
  genLocation(const Fortran::parser::CharBlock &block) override final {
    if (const auto *cooked = bridge.getCookedSource()) {
      auto loc = cooked->GetSourcePositionRange(block);
      if (loc.has_value()) {
        // loc is a pair (begin, end); use the beginning position
        auto &filePos = loc->first;
        return mlir::FileLineColLoc::get(filePos.file.path(), filePos.line,
                                         filePos.column, &getMLIRContext());
      }
    }
    return genLocation();
  }

  virtual mlir::Value locationToFilename(mlir::Location loc) override final {
    if (auto flc = loc.dyn_cast<mlir::FileLineColLoc>()) {
      // must be encoded as asciiz, C string
      auto fn = flc.getFilename().str() + '\0';
      return fir::getBase(createStringLiteral(loc, *this, fn, fn.size()));
    }
    return builder->createNullConstant(loc);
  }
  virtual mlir::Value locationToLineNo(mlir::Location loc,
                                       mlir::Type type) override final {
    if (auto flc = loc.dyn_cast<mlir::FileLineColLoc>())
      return builder->createIntegerConstant(loc, type, flc.getLine());
    return builder->createIntegerConstant(loc, type, 0);
  }

  Fortran::lower::FirOpBuilder &getFirOpBuilder() override final {
    return *builder;
  }

  mlir::ModuleOp &getModuleOp() override final { return bridge.getModule(); }

  mlir::MLIRContext &getMLIRContext() override final {
    return bridge.getMLIRContext();
  }
  std::string
  mangleName(const Fortran::semantics::Symbol &symbol) override final {
    return Fortran::lower::mangle::mangleName(uniquer, symbol);
  }

  std::string uniqueCGIdent(llvm::StringRef prefix,
                            llvm::StringRef name) override final {
    // For "long" identifiers use a hash value
    if (name.size() > nameLengthHashSize) {
      llvm::MD5 hash;
      hash.update(name);
      llvm::MD5::MD5Result result;
      hash.final(result);
      llvm::SmallString<32> str;
      llvm::MD5::stringifyResult(result, str);
      std::string hashName = prefix.str();
      hashName.append(".").append(str.c_str());
      return uniquer.doGenerated(hashName);
    }
    // "Short" identifiers use a reversible hex string
    std::string nm = prefix.str();
    return uniquer.doGenerated(nm.append(".").append(llvm::toHex(name)));
  }

private:
  FirConverter() = delete;
  FirConverter(const FirConverter &) = delete;
  FirConverter &operator=(const FirConverter &) = delete;

  //===--------------------------------------------------------------------===//
  // Helper member functions
  //===--------------------------------------------------------------------===//

  mlir::Value createFIRExpr(mlir::Location loc,
                            const Fortran::semantics::SomeExpr *expr,
                            Fortran::lower::StatementContext &stmtCtx) {
    return fir::getBase(genExprValue(*expr, stmtCtx, &loc));
  }

  /// Find the symbol in the local map or return null.
  Fortran::lower::SymbolBox
  lookupSymbol(const Fortran::semantics::Symbol &sym) {
    if (auto v = localSymbols.lookupSymbol(sym))
      return v;
    return {};
  }

  /// Add the symbol to the local map. If the symbol is already in the map, it
  /// is not updated. Instead the value `false` is returned.
  bool addSymbol(const Fortran::semantics::SymbolRef sym, mlir::Value val,
                 bool forced = false) {
    if (!forced && lookupSymbol(sym))
      return false;
    localSymbols.addSymbol(sym, val, forced);
    return true;
  }

  bool addCharSymbol(const Fortran::semantics::SymbolRef sym, mlir::Value val,
                     mlir::Value len, bool forced = false) {
    if (!forced && lookupSymbol(sym))
      return false;
    // TODO: ensure val type is fir.array<len x fir.char<kind>> like. Insert
    // cast if needed.
    localSymbols.addCharSymbol(sym, val, len, forced);
    return true;
  }

  mlir::Value createTemp(mlir::Location loc,
                         const Fortran::semantics::Symbol &sym,
                         llvm::ArrayRef<mlir::Value> shape = {}) {
    // FIXME: should return fir::ExtendedValue
    if (auto v = lookupSymbol(sym))
      return v.getAddr();
    auto newVal = builder->createTemporary(loc, genType(sym),
                                           toStringRef(sym.name()), shape);
    addSymbol(sym, newVal);
    return newVal;
  }

  bool isNumericScalarCategory(Fortran::common::TypeCategory cat) {
    return cat == Fortran::common::TypeCategory::Integer ||
           cat == Fortran::common::TypeCategory::Real ||
           cat == Fortran::common::TypeCategory::Complex ||
           cat == Fortran::common::TypeCategory::Logical;
  }

  bool isCharacterCategory(Fortran::common::TypeCategory cat) {
    return cat == Fortran::common::TypeCategory::Character;
  }

  mlir::Block *blockOfLabel(Fortran::lower::pft::Evaluation &eval,
                            Fortran::parser::Label label) {
    const auto &labelEvaluationMap =
        eval.getOwningProcedure()->labelEvaluationMap;
    const auto iter = labelEvaluationMap.find(label);
    assert(iter != labelEvaluationMap.end() && "label missing from map");
    auto *block = iter->second->block;
    assert(block && "missing labeled evaluation block");
    return block;
  }

  void genFIRBranch(mlir::Block *targetBlock) {
    assert(targetBlock && "missing unconditional target block");
    builder->create<mlir::BranchOp>(toLocation(), targetBlock);
  }

  void genFIRConditionalBranch(mlir::Value cond, mlir::Block *trueTarget,
                               mlir::Block *falseTarget) {
    assert(trueTarget && "missing conditional branch true block");
    assert(falseTarget && "missing conditional branch false block");
    auto loc = toLocation();
    auto bcc = builder->createConvert(loc, builder->getI1Type(), cond);
    builder->create<mlir::CondBranchOp>(loc, bcc, trueTarget, llvm::None,
                                        falseTarget, llvm::None);
  }
  void genFIRConditionalBranch(mlir::Value cond,
                               Fortran::lower::pft::Evaluation *trueTarget,
                               Fortran::lower::pft::Evaluation *falseTarget) {
    genFIRConditionalBranch(cond, trueTarget->block, falseTarget->block);
  }
  void genFIRConditionalBranch(const Fortran::parser::ScalarLogicalExpr &expr,
                               mlir::Block *trueTarget,
                               mlir::Block *falseTarget) {
    Fortran::lower::StatementContext stmtCtx;
    mlir::Value cond =
        createFIRExpr(toLocation(), Fortran::semantics::GetExpr(expr), stmtCtx);
    stmtCtx.finalize();
    genFIRConditionalBranch(cond, trueTarget, falseTarget);
  }
  void genFIRConditionalBranch(const Fortran::parser::ScalarLogicalExpr &expr,
                               Fortran::lower::pft::Evaluation *trueTarget,
                               Fortran::lower::pft::Evaluation *falseTarget) {
    Fortran::lower::StatementContext stmtCtx;
    auto cond =
        createFIRExpr(toLocation(), Fortran::semantics::GetExpr(expr), stmtCtx);
    stmtCtx.finalize();
    genFIRConditionalBranch(cond, trueTarget->block, falseTarget->block);
  }

  //===--------------------------------------------------------------------===//
  // Termination of symbolically referenced execution units
  //===--------------------------------------------------------------------===//

  /// END of program
  ///
  /// Generate the cleanup block before the program exits
  void genExitRoutine() {
    if (blockIsUnterminated())
      builder->create<mlir::ReturnOp>(toLocation());
  }
  void genFIR(const Fortran::parser::EndProgramStmt &) { genExitRoutine(); }

  /// END of procedure-like constructs
  ///
  /// Generate the cleanup block before the procedure exits
  void genReturnSymbol(const Fortran::semantics::Symbol &functionSymbol) {
    const auto &resultSym =
        functionSymbol.get<Fortran::semantics::SubprogramDetails>().result();
    auto resultSymBox = lookupSymbol(resultSym);
    auto loc = toLocation();
    if (!resultSymBox) {
      mlir::emitError(loc, "failed lowering function return");
      return;
    }
    auto resultVal = resultSymBox.match(
        [&](const fir::CharBoxValue &x) -> mlir::Value {
          return Fortran::lower::CharacterExprHelper{*builder, loc}
              .createEmboxChar(x.getBuffer(), x.getLen());
        },
        [&](const auto &) -> mlir::Value {
          auto resultRef = resultSymBox.getAddr();
          mlir::Type resultRefType = builder->getRefType(genType(resultSym));
          // A function with multiple entry points returning different types
          // tags all result variables with one of the largest types to allow
          // them to share the same storage.  Convert this to the actual type.
          if (resultRef.getType() != resultRefType)
            resultRef = builder->createConvert(loc, resultRefType, resultRef);
          return builder->create<fir::LoadOp>(loc, resultRef);
        });
    builder->create<mlir::ReturnOp>(loc, resultVal);
  }

  /// Get the return value of a call to \p symbol, which is a subroutine entry
  /// point that has alternative return specifiers.
  const mlir::Value
  getAltReturnResult(const Fortran::semantics::Symbol &symbol) {
    assert(Fortran::semantics::HasAlternateReturns(symbol) &&
           "subroutine does not have alternate returns");
    return getSymbolAddress(symbol);
  }

  void genFIRProcedureExit(Fortran::lower::pft::FunctionLikeUnit &funit,
                           const Fortran::semantics::Symbol &symbol) {
    if (auto *finalBlock = funit.finalBlock) {
      // The current block must end with a terminator.
      if (blockIsUnterminated())
        builder->create<mlir::BranchOp>(toLocation(), finalBlock);
      // Set insertion point to final block.
      builder->setInsertionPoint(finalBlock, finalBlock->end());
    }
    if (Fortran::semantics::IsFunction(symbol)) {
      genReturnSymbol(symbol);
    } else if (Fortran::semantics::HasAlternateReturns(symbol)) {
      mlir::Value retval = builder->create<fir::LoadOp>(
          toLocation(), getAltReturnResult(symbol));
      builder->create<mlir::ReturnOp>(toLocation(), retval);
    } else {
      genExitRoutine();
    }
  }

  //
  // Statements that have control-flow semantics
  //

  /// Generate an If[Then]Stmt condition or its negation.
  template <typename A>
  mlir::Value genIfCondition(const A *stmt, bool negate = false) {
    auto loc = toLocation();
    Fortran::lower::StatementContext stmtCtx;
    auto condExpr = createFIRExpr(
        loc,
        Fortran::semantics::GetExpr(
            std::get<Fortran::parser::ScalarLogicalExpr>(stmt->t)),
        stmtCtx);
    stmtCtx.finalize();
    auto cond = builder->createConvert(loc, builder->getI1Type(), condExpr);
    if (negate)
      cond = builder->create<mlir::XOrOp>(
          loc, cond, builder->createIntegerConstant(loc, cond.getType(), 1));
    return cond;
  }

  mlir::FuncOp getFunc(llvm::StringRef name, mlir::FunctionType ty) {
    if (auto func = builder->getNamedFunction(name)) {
      assert(func.getType() == ty);
      return func;
    }
    return builder->createFunction(toLocation(), name, ty);
  }

  /// Lowering of CALL statement
  void genFIR(const Fortran::parser::CallStmt &stmt) {
    Fortran::lower::StatementContext stmtCtx;
    auto &eval = getEval();
    setCurrentPosition(stmt.v.source);
    assert(stmt.typedCall && "Call was not analyzed");
    // Call statement lowering shares code with function call lowering.
    Fortran::semantics::SomeExpr expr{*stmt.typedCall};
    auto res = createFIRExpr(toLocation(), &expr, stmtCtx);
    if (!res)
      return; // "Normal" subroutine call.
    // Call with alternate return specifiers.
    // The call returns an index that selects an alternate return branch target.
    llvm::SmallVector<int64_t, 5> indexList;
    llvm::SmallVector<mlir::Block *, 5> blockList;
    int64_t index = 0;
    for (const auto &arg :
         std::get<std::list<Fortran::parser::ActualArgSpec>>(stmt.v.t)) {
      const auto &actual = std::get<Fortran::parser::ActualArg>(arg.t);
      if (const auto *altReturn =
              std::get_if<Fortran::parser::AltReturnSpec>(&actual.u)) {
        indexList.push_back(++index);
        blockList.push_back(blockOfLabel(eval, altReturn->v));
      }
    }
    blockList.push_back(eval.nonNopSuccessor().block); // default = fallthrough
    stmtCtx.finalize();
    builder->create<fir::SelectOp>(toLocation(), res, indexList, blockList);
  }

  void genFIR(const Fortran::parser::ComputedGotoStmt &stmt) {
    Fortran::lower::StatementContext stmtCtx;
    auto &eval = getEval();
    auto selectExpr =
        createFIRExpr(toLocation(),
                      Fortran::semantics::GetExpr(
                          std::get<Fortran::parser::ScalarIntExpr>(stmt.t)),
                      stmtCtx);
    llvm::SmallVector<int64_t, 8> indexList;
    llvm::SmallVector<mlir::Block *, 8> blockList;
    int64_t index = 0;
    for (auto &label : std::get<std::list<Fortran::parser::Label>>(stmt.t)) {
      indexList.push_back(++index);
      blockList.push_back(blockOfLabel(eval, label));
    }
    blockList.push_back(eval.nonNopSuccessor().block); // default
    stmtCtx.finalize();
    builder->create<fir::SelectOp>(toLocation(), selectExpr, indexList,
                                   blockList);
  }

  void genFIR(const Fortran::parser::ArithmeticIfStmt &stmt) {
    Fortran::lower::StatementContext stmtCtx;
    auto &eval = getEval();
    auto expr = createFIRExpr(
        toLocation(),
        Fortran::semantics::GetExpr(std::get<Fortran::parser::Expr>(stmt.t)),
        stmtCtx);
    auto exprType = expr.getType();
    auto loc = toLocation();
    if (exprType.isSignlessInteger()) {
      // Arithmetic expression has Integer type.  Generate a SelectCaseOp
      // with ranges {(-inf:-1], 0=default, [1:inf)}.
      MLIRContext *context = builder->getContext();
      llvm::SmallVector<mlir::Attribute, 3> attrList;
      llvm::SmallVector<mlir::Value, 3> valueList;
      llvm::SmallVector<mlir::Block *, 3> blockList;
      attrList.push_back(fir::UpperBoundAttr::get(context));
      valueList.push_back(builder->createIntegerConstant(loc, exprType, -1));
      blockList.push_back(blockOfLabel(eval, std::get<1>(stmt.t)));
      attrList.push_back(fir::LowerBoundAttr::get(context));
      valueList.push_back(builder->createIntegerConstant(loc, exprType, 1));
      blockList.push_back(blockOfLabel(eval, std::get<3>(stmt.t)));
      attrList.push_back(mlir::UnitAttr::get(context)); // 0 is the "default"
      blockList.push_back(blockOfLabel(eval, std::get<2>(stmt.t)));
      stmtCtx.finalize();
      builder->create<fir::SelectCaseOp>(loc, expr, attrList, valueList,
                                         blockList);
      return;
    }
    // Arithmetic expression has Real type.  Generate
    //   sum = expr + expr  [ raise an exception if expr is a NaN ]
    //   if (sum < 0.0) goto L1 else if (sum > 0.0) goto L3 else goto L2
    assert(eval.localBlocks.size() == 1 && "missing arithmetic if block");
    stmtCtx.finalize();
    auto sum = builder->create<fir::AddfOp>(loc, expr, expr);
    auto zero = builder->create<mlir::ConstantOp>(
        loc, exprType, builder->getFloatAttr(exprType, 0.0));
    auto cond1 =
        builder->create<mlir::CmpFOp>(loc, mlir::CmpFPredicate::OLT, sum, zero);
    genFIRConditionalBranch(cond1, blockOfLabel(eval, std::get<1>(stmt.t)),
                            eval.localBlocks[0]);
    startBlock(eval.localBlocks[0]);
    auto cond2 =
        builder->create<mlir::CmpFOp>(loc, mlir::CmpFPredicate::OGT, sum, zero);
    genFIRConditionalBranch(cond2, blockOfLabel(eval, std::get<3>(stmt.t)),
                            blockOfLabel(eval, std::get<2>(stmt.t)));
  }

  void genFIR(const Fortran::parser::AssignedGotoStmt &stmt) {
    // Program requirement 1990 8.2.4 -
    //
    //   At the time of execution of an assigned GOTO statement, the integer
    //   variable must be defined with the value of a statement label of a
    //   branch target statement that appears in the same scoping unit.
    //   Note that the variable may be defined with a statement label value
    //   only by an ASSIGN statement in the same scoping unit as the assigned
    //   GOTO statement.

    auto loc = toLocation();
    auto &eval = getEval();
    const auto &symbolLabelMap =
        eval.getOwningProcedure()->assignSymbolLabelMap;
    const auto &symbol = *std::get<Fortran::parser::Name>(stmt.t).symbol;
    auto selectExpr =
        builder->create<fir::LoadOp>(loc, getSymbolAddress(symbol));
    auto iter = symbolLabelMap.find(symbol);
    if (iter == symbolLabelMap.end()) {
      // Fail for a nonconforming program unit that does not have any ASSIGN
      // statements.  The front end should check for this.
      mlir::emitError(loc, "(semantics issue) no assigned goto targets");
      exit(1);
    }
    auto labelSet = iter->second;
    llvm::SmallVector<int64_t, 10> indexList;
    llvm::SmallVector<mlir::Block *, 10> blockList;
    auto addLabel = [&](Fortran::parser::Label label) {
      indexList.push_back(label);
      blockList.push_back(blockOfLabel(eval, label));
    };
    // Add labels from an explicit list.  The list may have duplicates.
    for (auto &label : std::get<std::list<Fortran::parser::Label>>(stmt.t)) {
      if (labelSet.count(label) &&
          std::find(indexList.begin(), indexList.end(), label) ==
              indexList.end()) { // ignore duplicates
        addLabel(label);
      }
    }
    // Absent an explicit list, add all possible label targets.
    if (indexList.empty())
      for (auto &label : labelSet)
        addLabel(label);
    // Add a nop/fallthrough branch to the switch for a nonconforming program
    // unit that violates the program requirement above.
    blockList.push_back(eval.nonNopSuccessor().block); // default
    builder->create<fir::SelectOp>(loc, selectExpr, indexList, blockList);
  }

  /// Collect DO CONCURRENT or FORALL loop control information.
  IncrementLoopNestInfo getConcurrentControl(
      const Fortran::parser::ConcurrentHeader &header,
      const std::list<Fortran::parser::LocalitySpec> &localityList = {}) {
    IncrementLoopNestInfo incrementLoopNestInfo;
    for (const auto &control :
         std::get<std::list<Fortran::parser::ConcurrentControl>>(header.t))
      incrementLoopNestInfo.emplace_back(
          *std::get<0>(control.t).symbol, std::get<1>(control.t),
          std::get<2>(control.t), std::get<3>(control.t), true);
    auto &info = incrementLoopNestInfo.back();
    info.maskExpr = Fortran::semantics::GetExpr(
        std::get<std::optional<Fortran::parser::ScalarLogicalExpr>>(header.t));
    for (const auto &x : localityList) {
      if (const auto *localInitList =
              std::get_if<Fortran::parser::LocalitySpec::LocalInit>(&x.u))
        for (const auto &x : localInitList->v)
          info.localInitSymList.push_back(x.symbol);
      if (std::get_if<Fortran::parser::LocalitySpec::Local>(&x.u))
        TODO("do concurrent locality specs not implemented");
    }
    return incrementLoopNestInfo;
  }

  /// Generate FIR for a DO construct.  There are six variants:
  ///  - unstructured infinite and while loops
  ///  - structured and unstructured increment loops
  ///  - structured and unstructured concurrent loops
  void genFIR(const Fortran::parser::DoConstruct &) {
    // Collect loop information.
    // Generate begin loop code directly for infinite and while loops.
    auto &eval = getEval();
    bool unstructuredContext = eval.lowerAsUnstructured();
    auto &doStmtEval = eval.getFirstNestedEvaluation();
    auto *doStmt = doStmtEval.getIf<Fortran::parser::NonLabelDoStmt>();
    const auto &loopControl =
        std::get<std::optional<Fortran::parser::LoopControl>>(doStmt->t);
    auto *preheaderBlock = doStmtEval.block;
    auto *headerBlock =
        unstructuredContext ? doStmtEval.localBlocks[0] : nullptr;
    auto *bodyBlock = doStmtEval.lexicalSuccessor->block;
    auto *exitBlock = doStmtEval.parentConstruct->constructExit->block;
    IncrementLoopNestInfo incrementLoopNestInfo;
    const Fortran::parser::ScalarLogicalExpr *whileCondition = nullptr;
    bool infiniteLoop = !loopControl.has_value();
    if (infiniteLoop) {
      assert(unstructuredContext && "infinite loop must be unstructured");
      startBlock(headerBlock);
    } else if ((whileCondition =
                    std::get_if<Fortran::parser::ScalarLogicalExpr>(
                        &loopControl->u))) {
      assert(unstructuredContext && "while loop must be unstructured");
      startBlock(headerBlock);
      genFIRConditionalBranch(*whileCondition, bodyBlock, exitBlock);
    } else if (const auto *bounds =
                   std::get_if<Fortran::parser::LoopControl::Bounds>(
                       &loopControl->u)) {
      // Non-concurrent increment loop.
      auto &info = incrementLoopNestInfo.emplace_back(
          *bounds->name.thing.symbol, bounds->lower, bounds->upper,
          bounds->step);
      if (unstructuredContext) {
        maybeStartBlock(preheaderBlock);
        info.hasRealControl = info.loopVariableSym.GetType()->IsNumeric(
            Fortran::common::TypeCategory::Real);
        info.headerBlock = headerBlock;
        info.bodyBlock = bodyBlock;
        info.exitBlock = exitBlock;
      }
    } else {
      const auto *concurrent =
          std::get_if<Fortran::parser::LoopControl::Concurrent>(
              &loopControl->u);
      assert(concurrent && "invalid DO loop variant");
      incrementLoopNestInfo = getConcurrentControl(
          std::get<Fortran::parser::ConcurrentHeader>(concurrent->t),
          std::get<std::list<Fortran::parser::LocalitySpec>>(concurrent->t));
      if (unstructuredContext) {
        maybeStartBlock(preheaderBlock);
        auto &endDoStmtEval = *doStmtEval.controlSuccessor;
        auto beginBlocks = doStmtEval.localBlocks.begin();
        auto endBlocks = endDoStmtEval.localBlocks.end();
        for (auto &info : incrementLoopNestInfo) {
          // The original loop body provides the body and latch blocks of the
          // innermost dimension.  The (first) body block of a non-innermost
          // dimension is the preheader block of the immediately enclosed
          // dimension.  The latch block of a non-innermost dimension is the
          // exit block of the immediately enclosed dimension.  Blocks are
          // generated "in order".
          auto isInnermost = &info == &incrementLoopNestInfo.back();
          auto isOutermost = &info == &incrementLoopNestInfo.front();
          info.headerBlock = *beginBlocks++;
          info.bodyBlock = isInnermost ? bodyBlock : *beginBlocks++;
          info.exitBlock = isOutermost ? exitBlock : *--endBlocks;
          if (info.maskExpr) {
            assert(endDoStmtEval.block &&
                   "missing masked concurrent loop latch block");
            info.maskBlock = *beginBlocks++;
          }
        }
        assert(beginBlocks == doStmtEval.localBlocks.end() &&
               "concurrent header+body+mask block count mismatch");
        assert(endBlocks == endDoStmtEval.localBlocks.begin() &&
               "concurrent latch block count mismatch");
      }
    }

    // Increment loop begin code.  (Infinite/while code was already generated.)
    if (!infiniteLoop && !whileCondition) {
      Fortran::lower::StatementContext stmtCtx;
      genFIRIncrementLoopBegin(incrementLoopNestInfo, stmtCtx);
    }

    // Loop body code - NonLabelDoStmt and EndDoStmt code is generated here.
    // Their genFIR calls are nops except for block management in some cases.
    for (auto &e : eval.getNestedEvaluations())
      genFIR(e, unstructuredContext);

    // Loop end code.
    if (infiniteLoop || whileCondition)
      genFIRBranch(headerBlock);
    else
      genFIRIncrementLoopEnd(incrementLoopNestInfo);
  }

  /// Generate FIR to begin a structured or unstructured increment loop nest.
  void genFIRIncrementLoopBegin(IncrementLoopNestInfo &incrementLoopNestInfo,
                                Fortran::lower::StatementContext &stmtCtx) {
    assert(!incrementLoopNestInfo.empty() && "empty loop nest");
    auto loc = toLocation();
    auto controlType = incrementLoopNestInfo[0].isStructured()
                           ? builder->getIndexType()
                           : genType(incrementLoopNestInfo[0].loopVariableSym);
    auto hasRealControl = incrementLoopNestInfo[0].hasRealControl;
    auto genControlValue = [&](const Fortran::semantics::SomeExpr *expr) {
      if (expr)
        return builder->createConvert(loc, controlType,
                                      createFIRExpr(loc, expr, stmtCtx));
      if (hasRealControl)
        return builder->createRealConstant(loc, controlType, 1u);
      return builder->createIntegerConstant(loc, controlType, 1); // step
    };
    auto genLocalInitAssignments = [](IncrementLoopInfo &info) {
      for (const auto *sym : info.localInitSymList) {
        const auto *hostDetails =
            sym->detailsIf<Fortran::semantics::HostAssocDetails>();
        assert(hostDetails && "missing local_init variable host variable");
        [[maybe_unused]] const Fortran::semantics::Symbol &hostSym =
            hostDetails->symbol();
        TODO("do concurrent locality specs not implemented");
        // assign sym = hostSym
      }
    };
    for (auto &info : incrementLoopNestInfo) {
      info.loopVariable = createTemp(loc, info.loopVariableSym);
      auto lowerValue = genControlValue(info.lowerExpr);
      auto upperValue = genControlValue(info.upperExpr);
      info.stepValue = genControlValue(info.stepExpr);

      // Structured loop - generate fir.do_loop.
      if (info.isStructured()) {
        info.doLoop = builder->create<fir::DoLoopOp>(
            loc, lowerValue, upperValue, info.stepValue, info.isUnordered,
            /*finalCountValue*/ !info.isUnordered);
        builder->setInsertionPointToStart(info.doLoop.getBody());
        // Update the loop variable value, as it may have non-index references.
        auto value = builder->createConvert(loc, genType(info.loopVariableSym),
                                            info.doLoop.getInductionVar());
        builder->create<fir::StoreOp>(loc, value, info.loopVariable);
        if (info.maskExpr) {
          auto ifOp = builder->create<fir::IfOp>(
              loc, createFIRExpr(loc, info.maskExpr, stmtCtx),
              /*withElseRegion=*/false);
          builder->setInsertionPointToStart(&ifOp.thenRegion().front());
        }
        genLocalInitAssignments(info);
        continue;
      }

      // Unstructured loop preheader - initialize tripVariable and loopVariable.
      mlir::Value tripCount;
      if (info.hasRealControl) {
        auto diff1 = builder->create<mlir::SubFOp>(loc, upperValue, lowerValue);
        auto diff2 = builder->create<mlir::AddFOp>(loc, diff1, info.stepValue);
        tripCount = builder->create<mlir::DivFOp>(loc, diff2, info.stepValue);
        controlType = builder->getIndexType();
        tripCount = builder->createConvert(loc, controlType, tripCount);
      } else {
        auto diff1 = builder->create<mlir::SubIOp>(loc, upperValue, lowerValue);
        auto diff2 = builder->create<mlir::AddIOp>(loc, diff1, info.stepValue);
        tripCount =
            builder->create<mlir::SignedDivIOp>(loc, diff2, info.stepValue);
      }
      if (fir::isAlwaysExecuteLoopBody()) { // minimum tripCount is 1
        auto one = builder->createIntegerConstant(loc, controlType, 1);
        auto cond = builder->create<mlir::CmpIOp>(loc, CmpIPredicate::slt,
                                                  tripCount, one);
        tripCount = builder->create<mlir::SelectOp>(loc, cond, one, tripCount);
      }
      info.tripVariable = builder->createTemporary(loc, controlType);
      builder->create<fir::StoreOp>(loc, tripCount, info.tripVariable);
      builder->create<fir::StoreOp>(loc, lowerValue, info.loopVariable);

      // Unstructured loop header - generate loop condition and mask.
      // Note - Currently there is no way to tag a loop as a concurrent loop.
      startBlock(info.headerBlock);
      tripCount = builder->create<fir::LoadOp>(loc, info.tripVariable);
      auto zero = builder->createIntegerConstant(loc, controlType, 0);
      auto cond = builder->create<mlir::CmpIOp>(loc, mlir::CmpIPredicate::sgt,
                                                tripCount, zero);
      if (info.maskExpr) {
        genFIRConditionalBranch(cond, info.maskBlock, info.exitBlock);
        startBlock(info.maskBlock);
        auto latchBlock = getEval().getLastNestedEvaluation().block;
        assert(latchBlock && "missing masked concurrent loop latch block");
        genFIRConditionalBranch(createFIRExpr(loc, info.maskExpr, stmtCtx),
                                info.bodyBlock, latchBlock);
      } else {
        genFIRConditionalBranch(cond, info.bodyBlock, info.exitBlock);
        if (&info != &incrementLoopNestInfo.back()) // not innermost
          startBlock(info.bodyBlock); // preheader block of enclosed dimension
      }
      if (!info.localInitSymList.empty()) {
        auto insertPt = builder->saveInsertionPoint();
        builder->setInsertionPointToStart(info.bodyBlock);
        genLocalInitAssignments(info);
        builder->restoreInsertionPoint(insertPt);
      }
    }
  }

  /// Generate FIR to end a structured or unstructured increment loop nest.
  void genFIRIncrementLoopEnd(IncrementLoopNestInfo &incrementLoopNestInfo) {
    assert(!incrementLoopNestInfo.empty() && "empty loop nest");
    auto loc = toLocation();
    for (auto it = incrementLoopNestInfo.rbegin(),
              rend = incrementLoopNestInfo.rend();
         it != rend; ++it) {
      auto &info = *it;
      if (info.isStructured()) {
        // End fir.do_loop.
        if (!info.isUnordered) {
          builder->setInsertionPointToEnd(info.doLoop.getBody());
          mlir::Value result = builder->create<mlir::AddIOp>(
              loc, info.doLoop.getInductionVar(), info.doLoop.step());
          builder->create<fir::ResultOp>(loc, result);
        }
        builder->setInsertionPointAfter(info.doLoop);
        if (info.isUnordered)
          continue;
        // The loop control variable may be used after loop execution.
        auto lcv = builder->createConvert(loc, genType(info.loopVariableSym),
                                          info.doLoop.getResult(0));
        builder->create<fir::StoreOp>(loc, lcv, info.loopVariable);
        continue;
      }

      // Unstructured loop - decrement tripVariable and step loopVariable.
      mlir::Value tripCount =
          builder->create<fir::LoadOp>(loc, info.tripVariable);
      auto tripVarType = info.hasRealControl ? builder->getIndexType()
                                             : genType(info.loopVariableSym);
      auto one = builder->createIntegerConstant(loc, tripVarType, 1);
      tripCount = builder->create<mlir::SubIOp>(loc, tripCount, one);
      builder->create<fir::StoreOp>(loc, tripCount, info.tripVariable);
      mlir::Value value = builder->create<fir::LoadOp>(loc, info.loopVariable);
      if (info.hasRealControl)
        value = builder->create<mlir::AddFOp>(loc, value, info.stepValue);
      else
        value = builder->create<mlir::AddIOp>(loc, value, info.stepValue);
      builder->create<fir::StoreOp>(loc, value, info.loopVariable);

      genFIRBranch(info.headerBlock);
      if (&info != &incrementLoopNestInfo.front()) // not outermost
        startBlock(info.exitBlock); // latch block of enclosing dimension
    }
  }

  /// Generate structured or unstructured FIR for an IF construct.
  /// The initial statement may be either an IfStmt or an IfThenStmt.
  void genFIR(const Fortran::parser::IfConstruct &) {
    auto loc = toLocation();
    auto &eval = getEval();
    if (eval.lowerAsStructured()) {
      // Structured fir.if nest.
      fir::IfOp topIfOp, currentIfOp;
      for (auto &e : eval.getNestedEvaluations()) {
        auto genIfOp = [&](mlir::Value cond) {
          auto ifOp = builder->create<fir::IfOp>(loc, cond, /*withElse=*/true);
          builder->setInsertionPointToStart(&ifOp.thenRegion().front());
          return ifOp;
        };
        if (auto *s = e.getIf<Fortran::parser::IfThenStmt>()) {
          topIfOp = currentIfOp = genIfOp(genIfCondition(s, e.negateCondition));
        } else if (auto *s = e.getIf<Fortran::parser::IfStmt>()) {
          topIfOp = currentIfOp = genIfOp(genIfCondition(s, e.negateCondition));
        } else if (auto *s = e.getIf<Fortran::parser::ElseIfStmt>()) {
          builder->setInsertionPointToStart(&currentIfOp.elseRegion().front());
          currentIfOp = genIfOp(genIfCondition(s));
        } else if (e.isA<Fortran::parser::ElseStmt>()) {
          builder->setInsertionPointToStart(&currentIfOp.elseRegion().front());
        } else if (e.isA<Fortran::parser::EndIfStmt>()) {
          builder->setInsertionPointAfter(topIfOp);
        } else {
          genFIR(e, /*unstructuredContext=*/false);
        }
      }
      return;
    }

    // Unstructured branch sequence.
    for (auto &e : eval.getNestedEvaluations()) {
      auto genIfBranch = [&](mlir::Value cond) {
        if (e.lexicalSuccessor == e.controlSuccessor) // empty block -> exit
          genFIRConditionalBranch(cond, e.parentConstruct->constructExit,
                                  e.controlSuccessor);
        else // non-empty block
          genFIRConditionalBranch(cond, e.lexicalSuccessor, e.controlSuccessor);
      };
      if (auto *s = e.getIf<Fortran::parser::IfThenStmt>()) {
        maybeStartBlock(e.block);
        genIfBranch(genIfCondition(s, e.negateCondition));
      } else if (auto *s = e.getIf<Fortran::parser::IfStmt>()) {
        maybeStartBlock(e.block);
        genIfBranch(genIfCondition(s, e.negateCondition));
      } else if (auto *s = e.getIf<Fortran::parser::ElseIfStmt>()) {
        startBlock(e.block);
        genIfBranch(genIfCondition(s));
      } else {
        genFIR(e);
      }
    }
  }

  void genFIR(const Fortran::parser::CaseConstruct &) {
    for (auto &e : getEval().getNestedEvaluations())
      genFIR(e);
  }

  /// Generate FIR for a FORALL statement.
  void genFIR(const Fortran::parser::ForallStmt &forallStmt) {
    auto incrementLoopNestInfo = getConcurrentControl(
        std::get<
            Fortran::common::Indirection<Fortran::parser::ConcurrentHeader>>(
            forallStmt.t)
            .value());
    auto &forallAssignment = std::get<Fortran::parser::UnlabeledStatement<
        Fortran::parser::ForallAssignmentStmt>>(forallStmt.t);
    genFIR(incrementLoopNestInfo, forallAssignment.statement);
  }

  /// Generate FIR for a FORALL construct.
  void genFIR(const Fortran::parser::ForallConstruct &forallConstruct) {
    auto &forallConstructStmt = std::get<
        Fortran::parser::Statement<Fortran::parser::ForallConstructStmt>>(
        forallConstruct.t);
    setCurrentPosition(forallConstructStmt.source);
    auto incrementLoopNestInfo = getConcurrentControl(
        std::get<
            Fortran::common::Indirection<Fortran::parser::ConcurrentHeader>>(
            forallConstructStmt.statement.t)
            .value());
    for (auto &s : std::get<std::list<Fortran::parser::ForallBodyConstruct>>(
             forallConstruct.t)) {
      std::visit(
          Fortran::common::visitors{
              [&](const Fortran::parser::Statement<
                  Fortran::parser::ForallAssignmentStmt> &b) {
                setCurrentPosition(b.source);
                genFIR(incrementLoopNestInfo, b.statement);
              },
              [&](const Fortran::parser::Statement<Fortran::parser::WhereStmt>
                      &b) {
                setCurrentPosition(b.source);
                genFIR(b.statement);
              },
              [&](const Fortran::parser::WhereConstruct &b) { genFIR(b); },
              [&](const Fortran::common::Indirection<
                  Fortran::parser::ForallConstruct> &b) { genFIR(b.value()); },
              [&](const Fortran::parser::Statement<Fortran::parser::ForallStmt>
                      &b) {
                setCurrentPosition(b.source);
                genFIR(b.statement);
              },
          },
          s.u);
    }
  }

  /// Generate FIR for a FORALL assignment statement.
  void genFIR(IncrementLoopNestInfo &incrementLoopNestInfo,
              const Fortran::parser::ForallAssignmentStmt &s) {
    Fortran::lower::StatementContext stmtCtx;
    genFIRIncrementLoopBegin(incrementLoopNestInfo, stmtCtx);
    stmtCtx.finalize();
    mlir::emitWarning(toLocation(), "Forall assignments are not temporized, "
                                    "so may be invalid\n");
    std::visit([&](auto &b) { genFIR(b); }, s.u);
    genFIRIncrementLoopEnd(incrementLoopNestInfo);
  }

  void genFIR(const Fortran::parser::CompilerDirective &) {
    mlir::emitWarning(toLocation(), "ignoring all compiler directives");
  }

  void genFIR(const Fortran::parser::OpenACCConstruct &acc) {
    auto insertPt = builder->saveInsertionPoint();
    genOpenACCConstruct(*this, getEval(), acc);
    for (auto &e : getEval().getNestedEvaluations())
      genFIR(e);
    builder->restoreInsertionPoint(insertPt);
  }

  void genFIR(const Fortran::parser::OpenMPConstruct &omp) {
    auto insertPt = builder->saveInsertionPoint();
    genOpenMPConstruct(*this, getEval(), omp);
    for (auto &e : getEval().getNestedEvaluations())
      genFIR(e);
    builder->restoreInsertionPoint(insertPt);
  }

  void genFIR(const Fortran::parser::OmpEndLoopDirective &omp) {
    // `OmpEndLoopDirective` can be captured as part of `OpenMPLoopConstruct`
    // so For now Lower this as a NOP.
  }

  void genFIR(const Fortran::parser::SelectCaseStmt &stmt) {
    auto &eval = getEval();
    using ScalarExpr = Fortran::parser::Scalar<Fortran::parser::Expr>;
    MLIRContext *context = builder->getContext();
    auto loc = toLocation();
    Fortran::lower::StatementContext stmtCtx;
    auto selectExpr = createFIRExpr(
        toLocation(), Fortran::semantics::GetExpr(std::get<ScalarExpr>(stmt.t)),
        stmtCtx);
    auto selectType = selectExpr.getType();
    Fortran::lower::CharacterExprHelper helper{*builder, loc};
    if (helper.isCharacterScalar(selectExpr.getType())) {
      TODO("");
    }
    llvm::SmallVector<mlir::Attribute, 10> attrList;
    llvm::SmallVector<mlir::Value, 10> valueList;
    llvm::SmallVector<mlir::Block *, 10> blockList;
    auto *defaultBlock = eval.parentConstruct->constructExit->block;
    using CaseValue = Fortran::parser::Scalar<Fortran::parser::ConstantExpr>;
    auto addValue = [&](const CaseValue &caseValue) {
      const auto *expr = Fortran::semantics::GetExpr(caseValue.thing);
      const auto v = Fortran::evaluate::ToInt64(*expr);
      valueList.push_back(
          v ? builder->createIntegerConstant(loc, selectType, *v)
            : builder->createConvert(
                  loc, selectType, createFIRExpr(toLocation(), expr, stmtCtx)));
    };
    for (Fortran::lower::pft::Evaluation *e = eval.controlSuccessor; e;
         e = e->controlSuccessor) {
      const auto &caseStmt = e->getIf<Fortran::parser::CaseStmt>();
      assert(e->block && "missing CaseStmt block");
      const auto &caseSelector =
          std::get<Fortran::parser::CaseSelector>(caseStmt->t);
      const auto *caseValueRangeList =
          std::get_if<std::list<Fortran::parser::CaseValueRange>>(
              &caseSelector.u);
      if (!caseValueRangeList) {
        defaultBlock = e->block;
        continue;
      }
      for (auto &caseValueRange : *caseValueRangeList) {
        blockList.push_back(e->block);
        if (const auto *caseValue = std::get_if<CaseValue>(&caseValueRange.u)) {
          attrList.push_back(fir::PointIntervalAttr::get(context));
          addValue(*caseValue);
          continue;
        }
        const auto &caseRange =
            std::get<Fortran::parser::CaseValueRange::Range>(caseValueRange.u);
        if (caseRange.lower && caseRange.upper) {
          attrList.push_back(fir::ClosedIntervalAttr::get(context));
          addValue(*caseRange.lower);
          addValue(*caseRange.upper);
        } else if (caseRange.lower) {
          attrList.push_back(fir::LowerBoundAttr::get(context));
          addValue(*caseRange.lower);
        } else {
          attrList.push_back(fir::UpperBoundAttr::get(context));
          addValue(*caseRange.upper);
        }
      }
    }
    attrList.push_back(mlir::UnitAttr::get(context));
    blockList.push_back(defaultBlock);
    stmtCtx.finalize();
    builder->create<fir::SelectCaseOp>(toLocation(), selectExpr, attrList,
                                       valueList, blockList);
  }

  // Nop statements - No code, or code is generated elsewhere.
  void genFIR(const Fortran::parser::CaseStmt &) {}              // nop
  void genFIR(const Fortran::parser::ContinueStmt &) {}          // nop
  void genFIR(const Fortran::parser::ElseIfStmt &) {}            // nop
  void genFIR(const Fortran::parser::ElseStmt &) {}              // nop
  void genFIR(const Fortran::parser::EndDoStmt &) {}             // nop
  void genFIR(const Fortran::parser::EndForallStmt &) {}         // nop
  void genFIR(const Fortran::parser::EndFunctionStmt &) {}       // nop
  void genFIR(const Fortran::parser::EndIfStmt &) {}             // nop
  void genFIR(const Fortran::parser::EndMpSubprogramStmt &) {}   // nop
  void genFIR(const Fortran::parser::EndSelectStmt &) {}         // nop
  void genFIR(const Fortran::parser::EndSubroutineStmt &) {}     // nop
  void genFIR(const Fortran::parser::EntryStmt &) {}             // nop
  void genFIR(const Fortran::parser::ForallAssignmentStmt &s) {} // nop
  void genFIR(const Fortran::parser::ForallConstructStmt &) {}   // nop
  void genFIR(const Fortran::parser::IfStmt &stmt) {}            // nop
  void genFIR(const Fortran::parser::IfThenStmt &) {}            // nop
  void genFIR(const Fortran::parser::NonLabelDoStmt &) {}        // nop

  void genFIR(const Fortran::parser::AssociateConstruct &) { TODO(""); }
  void genFIR(const Fortran::parser::AssociateStmt &) { TODO(""); }
  void genFIR(const Fortran::parser::EndAssociateStmt &) { TODO(""); }

  void genFIR(const Fortran::parser::BlockConstruct &) { TODO(""); }
  void genFIR(const Fortran::parser::BlockStmt &) { TODO(""); }
  void genFIR(const Fortran::parser::EndBlockStmt &) { TODO(""); }

  void genFIR(const Fortran::parser::ChangeTeamConstruct &construct) {
    genChangeTeamConstruct(*this, getEval(), construct);
  }
  void genFIR(const Fortran::parser::ChangeTeamStmt &stmt) {
    genChangeTeamStmt(*this, getEval(), stmt);
  }
  void genFIR(const Fortran::parser::EndChangeTeamStmt &stmt) {
    genEndChangeTeamStmt(*this, getEval(), stmt);
  }

  void genFIR(const Fortran::parser::CriticalConstruct &) { TODO(""); }
  void genFIR(const Fortran::parser::CriticalStmt &) { TODO(""); }
  void genFIR(const Fortran::parser::EndCriticalStmt &) { TODO(""); }

  void genFIR(const Fortran::parser::SelectRankConstruct &) { TODO(""); }
  void genFIR(const Fortran::parser::SelectRankStmt &) { TODO(""); }
  void genFIR(const Fortran::parser::SelectRankCaseStmt &) { TODO(""); }

  void genFIR(const Fortran::parser::SelectTypeConstruct &) { TODO(""); }
  void genFIR(const Fortran::parser::SelectTypeStmt &) { TODO(""); }
  void genFIR(const Fortran::parser::TypeGuardStmt &) { TODO(""); }

  //===--------------------------------------------------------------------===//
  // IO statements (see io.h)
  //===--------------------------------------------------------------------===//

  void genFIR(const Fortran::parser::BackspaceStmt &stmt) {
    auto iostat = genBackspaceStatement(*this, stmt);
    genIoConditionBranches(getEval(), stmt.v, iostat);
  }
  void genFIR(const Fortran::parser::CloseStmt &stmt) {
    auto iostat = genCloseStatement(*this, stmt);
    genIoConditionBranches(getEval(), stmt.v, iostat);
  }
  void genFIR(const Fortran::parser::EndfileStmt &stmt) {
    auto iostat = genEndfileStatement(*this, stmt);
    genIoConditionBranches(getEval(), stmt.v, iostat);
  }
  void genFIR(const Fortran::parser::FlushStmt &stmt) {
    auto iostat = genFlushStatement(*this, stmt);
    genIoConditionBranches(getEval(), stmt.v, iostat);
  }
  void genFIR(const Fortran::parser::InquireStmt &stmt) {
    auto iostat = genInquireStatement(*this, stmt);
    if (const auto *specs =
            std::get_if<std::list<Fortran::parser::InquireSpec>>(&stmt.u))
      genIoConditionBranches(getEval(), *specs, iostat);
  }
  void genFIR(const Fortran::parser::OpenStmt &stmt) {
    auto iostat = genOpenStatement(*this, stmt);
    genIoConditionBranches(getEval(), stmt.v, iostat);
  }
  void genFIR(const Fortran::parser::PrintStmt &stmt) {
    genPrintStatement(*this, stmt);
  }
  void genFIR(const Fortran::parser::ReadStmt &stmt) {
    auto iostat = genReadStatement(*this, stmt);
    genIoConditionBranches(getEval(), stmt.controls, iostat);
  }
  void genFIR(const Fortran::parser::RewindStmt &stmt) {
    auto iostat = genRewindStatement(*this, stmt);
    genIoConditionBranches(getEval(), stmt.v, iostat);
  }
  void genFIR(const Fortran::parser::WaitStmt &stmt) {
    auto iostat = genWaitStatement(*this, stmt);
    genIoConditionBranches(getEval(), stmt.v, iostat);
  }
  void genFIR(const Fortran::parser::WriteStmt &stmt) {
    auto iostat = genWriteStatement(*this, stmt);
    genIoConditionBranches(getEval(), stmt.controls, iostat);
  }

  template <typename A>
  void genIoConditionBranches(Fortran::lower::pft::Evaluation &eval,
                              const A &specList, mlir::Value iostat) {
    if (!iostat)
      return;

    mlir::Block *endBlock = nullptr;
    mlir::Block *eorBlock = nullptr;
    mlir::Block *errBlock = nullptr;
    for (const auto &spec : specList) {
      std::visit(Fortran::common::visitors{
                     [&](const Fortran::parser::EndLabel &label) {
                       endBlock = blockOfLabel(eval, label.v);
                     },
                     [&](const Fortran::parser::EorLabel &label) {
                       eorBlock = blockOfLabel(eval, label.v);
                     },
                     [&](const Fortran::parser::ErrLabel &label) {
                       errBlock = blockOfLabel(eval, label.v);
                     },
                     [](const auto &) {}},
                 spec.u);
    }
    if (!endBlock && !eorBlock && !errBlock)
      return;

    auto loc = toLocation();
    auto indexType = builder->getIndexType();
    auto selector = builder->createConvert(loc, indexType, iostat);
    llvm::SmallVector<int64_t, 5> indexList;
    llvm::SmallVector<mlir::Block *, 4> blockList;
    if (eorBlock) {
      indexList.push_back(Fortran::runtime::io::IostatEor);
      blockList.push_back(eorBlock);
    }
    if (endBlock) {
      indexList.push_back(Fortran::runtime::io::IostatEnd);
      blockList.push_back(endBlock);
    }
    if (errBlock) {
      indexList.push_back(0);
      blockList.push_back(eval.nonNopSuccessor().block);
      // ERR label statement is the default successor.
      blockList.push_back(errBlock);
    } else {
      // Fallthrough successor statement is the default successor.
      blockList.push_back(eval.nonNopSuccessor().block);
    }
    builder->create<fir::SelectOp>(loc, selector, indexList, blockList);
  }

  //===--------------------------------------------------------------------===//
  // Memory allocation and deallocation
  //===--------------------------------------------------------------------===//

  void genFIR(const Fortran::parser::AllocateStmt &stmt) {
    Fortran::lower::genAllocateStmt(*this, stmt, toLocation());
  }

  void genFIR(const Fortran::parser::DeallocateStmt &stmt) {
    Fortran::lower::genDeallocateStmt(*this, stmt, toLocation());
  }

  /// Nullify pointer object list
  ///
  /// For each pointer object, reset the pointer to a disassociated status.
  /// We do this by setting each pointer to null.
  void genFIR(const Fortran::parser::NullifyStmt &stmt) {
    auto loc = toLocation();
    for (auto &po : stmt.v) {
      std::visit(
          Fortran::common::visitors{
              [&](const Fortran::parser::Name &sym) {
                auto ty = genType(*sym.symbol);
                auto load = builder->create<fir::LoadOp>(
                    loc, getSymbolAddress(*sym.symbol));
                auto idxTy = builder->getIndexType();
                auto zero = builder->create<mlir::ConstantOp>(
                    loc, idxTy, builder->getIntegerAttr(idxTy, 0));
                auto cast = builder->createConvert(loc, ty, zero);
                builder->create<fir::StoreOp>(loc, cast, load);
              },
              [&](const Fortran::parser::StructureComponent &) { TODO(""); },
          },
          po.u);
    }
  }

  //===--------------------------------------------------------------------===//

  void genFIR(const Fortran::parser::EventPostStmt &stmt) {
    genEventPostStatement(*this, stmt);
  }

  void genFIR(const Fortran::parser::EventWaitStmt &stmt) {
    genEventWaitStatement(*this, stmt);
  }

  void genFIR(const Fortran::parser::FormTeamStmt &stmt) {
    genFormTeamStatement(*this, getEval(), stmt);
  }

  void genFIR(const Fortran::parser::LockStmt &stmt) {
    genLockStatement(*this, stmt);
  }

  fir::ExtendedValue
  genInitializerExprValue(const Fortran::lower::SomeExpr &expr,
                          Fortran::lower::StatementContext &stmtCtx) {
    return createSomeInitializerExpression(toLocation(), *this, expr,
                                           localSymbols, stmtCtx);
  }

  /// Array assignment.
  /// 1. Evaluate the lhs to determine the rank and how to form the ArrayLoad
  /// (e.g., if there is a slicing op).
  /// 2. Scan the rhs, creating the ArrayLoads and evaluate the scalar subparts
  /// to be added to the map.
  /// 3. Create the loop nest and evaluate the elemental expression, threading
  /// the results.
  /// 4. Copy the resulting array back with ArrayMergeStore to the lhs as
  /// determined per step 1.
  void genArrayAssignment(const Fortran::evaluate::Assignment &assign,
                          const Fortran::semantics::Symbol *sym,
                          Fortran::lower::StatementContext &stmtCtx) {
    auto loc = toLocation();
    if (Fortran::evaluate::IsConstantExpr(assign.rhs) &&
        (assign.lhs.Rank() == assign.rhs.Rank())) {
      // TODO: Constants get expanded out as inline array values. Probably want
      // to reconsider that based on context. For an initializer, that's fine,
      // but inlining large arrays in a function is not.
      auto rhs = fir::getBase(genInitializerExprValue(assign.rhs, stmtCtx));
      auto lhsCoorExv = genExprAddr(assign.lhs, stmtCtx);
      builder->create<fir::StoreOp>(loc, rhs, fir::getBase(lhsCoorExv));
      return;
    }
    localSymbols.pushScope();
    auto dst =
        createSomeArraySubspace(*this, assign.lhs, localSymbols, stmtCtx);
    auto exv =
        createSomeNewArrayValue(*this, dst, assign.rhs, localSymbols, stmtCtx);
    builder->create<fir::ArrayMergeStoreOp>(loc, dst, fir::getBase(exv),
                                            dst.memref());
    localSymbols.popScope();
  }

  /// Shared for both assignments and pointer assignments.
  void genAssignment(const Fortran::evaluate::Assignment &assign) {
    Fortran::lower::StatementContext stmtCtx;
    std::visit(
        Fortran::common::visitors{
            [&](const Fortran::evaluate::Assignment::Intrinsic &) {
              auto loc = toLocation();
              const auto *sym =
                  Fortran::evaluate::UnwrapWholeSymbolDataRef(assign.lhs);
              // Assignment of allocatable are more complex, the lhs may need to
              // be deallocated/reallocated. See Fortran 2018 10.2.1.3 p3
              const bool isHeap =
                  sym && Fortran::semantics::IsAllocatable(*sym);
              if (isHeap) {
                TODO("assignment to allocatable not implemented");
              }
              // Target of the pointer must be assigned. See Fortran
              // 2018 10.2.1.3 p2
              const bool isPointer = sym && Fortran::semantics::IsPointer(*sym);
              if (assign.lhs.Rank() > 0) {
                // Array assignment
                // See Fortran 2018 10.2.1.3 p5, p6, and p7
                genArrayAssignment(assign, sym, stmtCtx);
                return;
              }

              // Scalar assignment
              auto lhsType = assign.lhs.GetType();
              assert(lhsType && "lhs cannot be typeless");
              if (isNumericScalarCategory(lhsType->category())) {
                // Fortran 2018 10.2.1.3 p8 and p9
                // Conversions should have been inserted by semantic analysis,
                // but they can be incorrect between the rhs and lhs. Correct
                // that here.
                auto addr =
                    fir::getBase(isPointer ? genExprValue(assign.lhs, stmtCtx)
                                           : genExprAddr(assign.lhs, stmtCtx));
                auto val = createFIRExpr(loc, &assign.rhs, stmtCtx);
                // A function with multiple entry points returning different
                // types tags all result variables with one of the largest
                // types to allow them to share the same storage.  Assignment
                // to a result variable of one of the other types requires
                // conversion to the actual type.
                auto toTy = genType(assign.lhs);
                auto cast = builder->convertWithSemantics(loc, toTy, val);
                if (fir::dyn_cast_ptrEleTy(addr.getType()) != toTy) {
                  assert(sym->IsFuncResult() && "type mismatch");
                  addr = builder->createConvert(
                      toLocation(), builder->getRefType(toTy), addr);
                }
                builder->create<fir::StoreOp>(loc, cast, addr);
                return;
              }
              if (isCharacterCategory(lhsType->category())) {
                // Fortran 2018 10.2.1.3 p10 and p11
                auto lhs = genExprAddr(assign.lhs, stmtCtx);
                // Current character assignment only works with in memory
                // characters since !fir.array<> cannot be addressed with
                // fir.coordinate_of without being inside a !fir.ref<> or other
                // memory types. So use genExprAddr for rhs.
                auto rhs = genExprAddr(assign.rhs, stmtCtx);
                Fortran::lower::CharacterExprHelper{*builder, loc}.createAssign(
                    lhs, rhs);
                return;
              }
              if (lhsType->category() ==
                  Fortran::common::TypeCategory::Derived) {
                // Fortran 2018 10.2.1.3 p12 and p13
                TODO("");
              }
              llvm_unreachable("unknown category");
            },
            [&](const Fortran::evaluate::ProcedureRef &) {
              // Defined assignment: call ProcRef
              TODO("");
            },
            [&](const Fortran::evaluate::Assignment::BoundsSpec &) {
              // Pointer assignment with possibly empty bounds-spec
              TODO("");
            },
            [&](const Fortran::evaluate::Assignment::BoundsRemapping &) {
              // Pointer assignment with bounds-remapping
              TODO("");
            },
        },
        assign.u);
  }

  void genFIR(const Fortran::parser::WhereConstruct &) { TODO(""); }
  void genFIR(const Fortran::parser::WhereConstructStmt &) { TODO(""); }
  void genFIR(const Fortran::parser::MaskedElsewhereStmt &) { TODO(""); }
  void genFIR(const Fortran::parser::ElsewhereStmt &) { TODO(""); }
  void genFIR(const Fortran::parser::EndWhereStmt &) { TODO(""); }
  void genFIR(const Fortran::parser::WhereStmt &) { TODO(""); }

  void genFIR(const Fortran::parser::PointerAssignmentStmt &stmt) {
    genAssignment(*stmt.typedAssignment->v);
  }

  void genFIR(const Fortran::parser::AssignmentStmt &stmt) {
    genAssignment(*stmt.typedAssignment->v);
  }

  void genFIR(const Fortran::parser::SyncAllStmt &stmt) {
    genSyncAllStatement(*this, stmt);
  }

  void genFIR(const Fortran::parser::SyncImagesStmt &stmt) {
    genSyncImagesStatement(*this, stmt);
  }

  void genFIR(const Fortran::parser::SyncMemoryStmt &stmt) {
    genSyncMemoryStatement(*this, stmt);
  }

  void genFIR(const Fortran::parser::SyncTeamStmt &stmt) {
    genSyncTeamStatement(*this, stmt);
  }

  void genFIR(const Fortran::parser::UnlockStmt &stmt) {
    genUnlockStatement(*this, stmt);
  }

  void genFIR(const Fortran::parser::AssignStmt &stmt) {
    const auto &symbol = *std::get<Fortran::parser::Name>(stmt.t).symbol;
    auto loc = toLocation();
    const auto labelValue = builder->createIntegerConstant(
        loc, genType(symbol), std::get<Fortran::parser::Label>(stmt.t));
    builder->create<fir::StoreOp>(loc, labelValue, getSymbolAddress(symbol));
  }

  void genFIR(const Fortran::parser::FormatStmt &) {
    // do nothing.

    // FORMAT statements have no semantics. They may be lowered if used by a
    // data transfer statement.
  }

  void genFIR(const Fortran::parser::PauseStmt &stmt) {
    genPauseStatement(*this, stmt);
  }

  void genFIR(const Fortran::parser::NamelistStmt &) { TODO(""); }

  // call FAIL IMAGE in runtime
  void genFIR(const Fortran::parser::FailImageStmt &stmt) {
    genFailImageStatement(*this);
  }

  // call STOP, ERROR STOP in runtime
  void genFIR(const Fortran::parser::StopStmt &stmt) {
    genStopStatement(*this, stmt);
  }

  void genFIR(const Fortran::parser::ReturnStmt &stmt) {
    auto *funit = getEval().getOwningProcedure();
    assert(funit && "not inside main program, function or subroutine");
    if (funit->isMainProgram()) {
      genExitRoutine();
      return;
    }
    auto loc = toLocation();
    if (stmt.v) {
      // Alternate return statement - If this is a subroutine where some
      // alternate entries have alternate returns, but the active entry point
      // does not, ignore the alternate return value.  Otherwise, assign it
      // to the compiler-generated result variable.
      const auto &symbol = funit->getSubprogramSymbol();
      if (Fortran::semantics::HasAlternateReturns(symbol)) {
        Fortran::lower::StatementContext stmtCtx;
        const auto *expr = Fortran::semantics::GetExpr(*stmt.v);
        assert(expr && "missing alternate return expression");
        auto altReturnIndex = builder->createConvert(
            loc, builder->getIndexType(), createFIRExpr(loc, expr, stmtCtx));
        builder->create<fir::StoreOp>(loc, altReturnIndex,
                                      getAltReturnResult(symbol));
      }
    }
    // Branch to the last block of the SUBROUTINE, which has the actual return.
    if (!funit->finalBlock) {
      const auto insPt = builder->saveInsertionPoint();
      funit->finalBlock = builder->createBlock(&builder->getRegion());
      builder->restoreInsertionPoint(insPt);
    }
    builder->create<mlir::BranchOp>(loc, funit->finalBlock);
  }

  void genFIR(const Fortran::parser::CycleStmt &) {
    genFIRBranch(getEval().controlSuccessor->block);
  }
  void genFIR(const Fortran::parser::ExitStmt &) {
    genFIRBranch(getEval().controlSuccessor->block);
  }
  void genFIR(const Fortran::parser::GotoStmt &) {
    genFIRBranch(getEval().controlSuccessor->block);
  }

  /// Generate the FIR for the Evaluation `eval`.
  void genFIR(Fortran::lower::pft::Evaluation &eval,
              bool unstructuredContext = true) {
    if (unstructuredContext) {
      // When transitioning from unstructured to structured code,
      // the structured code could be a target that starts a new block.
      maybeStartBlock(eval.isConstruct() && eval.lowerAsStructured()
                          ? eval.getFirstNestedEvaluation().block
                          : eval.block);
    }

    setCurrentEval(eval);
    setCurrentPosition(eval.position);
    eval.visit([&](const auto &stmt) { genFIR(stmt); });

    if (unstructuredContext && blockIsUnterminated()) {
      // Exit from an unstructured IF or SELECT construct block.
      Fortran::lower::pft::Evaluation *successor{};
      if (eval.isActionStmt())
        successor = eval.controlSuccessor;
      else if (eval.isConstruct() &&
               eval.getLastNestedEvaluation()
                   .lexicalSuccessor->isIntermediateConstructStmt())
        successor = eval.constructExit;
      if (successor && successor->block)
        genFIRBranch(successor->block);
    }
  }
  //===----------------------------------------------------------------===//
  // Variable instantiation
  //===----------------------------------------------------------------===//

  /// AggregateStoreMap is used to keep track of instantiated aggregate stores
  /// when lowering a scope containing equivalences (aliases).
  using AggregateStoreKey =
      std::tuple<const Fortran::semantics::Scope *, std::size_t>;
  using AggregateStoreMap = llvm::DenseMap<AggregateStoreKey, mlir::Value>;

  /// Instantiate variable \p var and add it to the symbol map.
  void instantiateVar(const Fortran::lower::pft::Variable &var,
                      AggregateStoreMap &storeMap) {
    if (var.isAggregateStore()) {
      instantiateAggregateStore(var, storeMap);
    } else if (const auto *common =
                   Fortran::semantics::FindCommonBlockContaining(
                       var.getSymbol().GetUltimate())) {
      instantiateCommon(*common, var);
    } else if (var.isAlias()) {
      instantiateAlias(var, storeMap);
    } else if (var.isGlobal()) {
      instantiateGlobal(var);
    } else {
      instantiateLocal(var);
    }
  }

  //===----------------------------------------------------------------===//
  // Global variables instatiation (not for alias and common)
  //===----------------------------------------------------------------===//

  /// Create the global op declaration without any initializer
  fir::GlobalOp declareGlobal(const Fortran::lower::pft::Variable &var,
                              llvm::StringRef globalName,
                              mlir::StringAttr linkage) {
    const auto &sym = var.getSymbol();
    auto loc = genLocation(sym.name());
    // Resolve potential host and module association before checking that this
    // symbol is an object of a function pointer.
    const auto &ultimate = sym.GetUltimate();
    if (!ultimate.has<Fortran::semantics::ObjectEntityDetails>() &&
        !ultimate.has<Fortran::semantics::ProcEntityDetails>())
      mlir::emitError(loc, "lowering global declaration: symbol '")
          << toStringRef(sym.name()) << "' has unexpected details\n";
    return builder->createGlobal(loc, genType(var), globalName, linkage);
  };
  /// Create the global op and its init if it has one
  fir::GlobalOp defineGlobal(const Fortran::lower::pft::Variable &var,
                             llvm::StringRef globalName,
                             mlir::StringAttr linkage) {
    const auto &sym = var.getSymbol();
    auto loc = genLocation(sym.name());
    bool isConst = sym.attrs().test(Fortran::semantics::Attr::PARAMETER);
    fir::GlobalOp global;
    if (Fortran::semantics::IsAllocatableOrPointer(sym)) {
      auto symTy = genType(var);
      // Pointers may have an initial target
      if (Fortran::semantics::IsPointer(sym))
        mlir::emitError(loc, "TODO: global pointer initialization");
      auto init = [&](Fortran::lower::FirOpBuilder &b) {
        auto box =
            Fortran::lower::createUnallocatedBox(b, loc, symTy, llvm::None);
        b.create<fir::HasValueOp>(loc, box);
      };
      global =
          builder->createGlobal(loc, symTy, globalName, isConst, init, linkage);
    } else if (const auto *details =
                   sym.detailsIf<Fortran::semantics::ObjectEntityDetails>()) {
      if (details->init()) {
        if (!sym.GetType()->AsIntrinsic()) {
          TODO(""); // Derived type / polymorphic
        }
        auto symTy = genType(var);
        if (symTy.isa<fir::CharacterType>()) {
          if (auto chLit = getCharacterLiteralCopy(details->init().value())) {
            auto init = builder->getStringAttr(std::get<std::string>(*chLit));
            global = builder->createGlobal(loc, symTy, globalName, linkage,
                                           init, isConst);
          } else {
            fir::emitFatalError(
                loc, "global CHARACTER has unexpected initial value");
          }
        } else {
          global = builder->createGlobal(
              loc, symTy, globalName, isConst,
              [&](Fortran::lower::FirOpBuilder &builder) {
                Fortran::lower::StatementContext stmtCtx;
                auto initVal =
                    genInitializerExprValue(details->init().value(), stmtCtx);
                auto castTo =
                    builder.createConvert(loc, symTy, fir::getBase(initVal));
                builder.create<fir::HasValueOp>(loc, castTo);
              },
              linkage);
        }
      }
    } else if (sym.has<Fortran::semantics::CommonBlockDetails>()) {
      mlir::emitError(loc, "COMMON symbol processed elsewhere");
    } else {
      TODO("global"); // Procedure pointer or something else
    }
    // Creates undefined initializer for globals without initialziers
    if (!global) {
      auto symTy = genType(var);
      global = builder->createGlobal(
          loc, symTy, globalName, isConst,
          [&](Fortran::lower::FirOpBuilder &builder) {
            builder.create<fir::HasValueOp>(
                loc, builder.create<fir::UndefOp>(loc, symTy));
          },
          linkage);
    }
    // Set public visibility to prevent global definition to be optimized out
    // even if they have no initializer and are unused in this compilation unit.
    global.setVisibility(mlir::SymbolTable::Visibility::Public);
    return global;
  }
  /// Instantiate a global variable. If it hasn't already been processed, add
  /// the global to the ModuleOp as a new uniqued symbol and initialize it with
  /// the correct value. It will be referenced on demand using `fir.addr_of`.
  void instantiateGlobal(const Fortran::lower::pft::Variable &var) {
    const auto &sym = var.getSymbol();
    auto globalName = mangleName(sym);
    auto loc = genLocation(sym.name());
    assert(!var.isAlias() && "must be handled in instantiateAlias");
    fir::GlobalOp global = builder->getNamedGlobal(globalName);
    if (!global) {
      if (var.isDeclaration()) {
        // Using a global from a module not defined in this compilation unit.
        mlir::StringAttr externalLinkage;
        global = declareGlobal(var, globalName, externalLinkage);
      } else {
        // Module and common globals are defined elsewhere.
        // The only globals defined here are the globals owned by procedures
        // and they do not need to be visible in other compilation unit.
        auto internalLinkage = builder->createInternalLinkage();
        global = defineGlobal(var, globalName, internalLinkage);
      }
    }
    auto addrOf = builder->create<fir::AddrOfOp>(loc, global.resultType(),
                                                 global.getSymbol());
    Fortran::lower::StatementContext stmtCtx;
    mapSymbolAttributes(var, stmtCtx, addrOf);
  }

  //===----------------------------------------------------------------===//
  // Local variables instantiation (not for alias)
  //===----------------------------------------------------------------===//

  /// Create a stack slot for a local variable. Precondition: the insertion
  /// point of the builder must be in the entry block, which is currently being
  /// constructed.
  mlir::Value createNewLocal(mlir::Location loc,
                             const Fortran::lower::pft::Variable &var,
                             mlir::Value preAlloc,
                             llvm::ArrayRef<mlir::Value> shape = {},
                             llvm::ArrayRef<mlir::Value> lenParams = {}) {
    if (preAlloc)
      return preAlloc;
    auto nm = mangleName(var.getSymbol());
    auto ty = genType(var);
    if (shape.size())
      if (auto arrTy = ty.dyn_cast<fir::SequenceType>()) {
        // elide the constant dimensions before construction
        assert(shape.size() == arrTy.getDimension());
        llvm::SmallVector<mlir::Value, 8> args;
        auto typeShape = arrTy.getShape();
        for (unsigned i = 0, end = arrTy.getDimension(); i < end; ++i)
          if (typeShape[i] == fir::SequenceType::getUnknownExtent())
            args.push_back(shape[i]);
        return builder->allocateLocal(loc, ty, nm, args, lenParams,
                                      var.isTarget());
      }
    return builder->allocateLocal(loc, ty, nm, shape, lenParams,
                                  var.isTarget());
  }
  /// Instantiate a local variable. Precondition: Each variable will be visited
  /// such that if its properties depend on other variables, the variables upon
  /// which its properties depend will already have been visited.
  void instantiateLocal(const Fortran::lower::pft::Variable &var) {
    assert(!var.isAlias());
    Fortran::lower::StatementContext stmtCtx;
    mapSymbolAttributes(var, stmtCtx);
  }

  //===----------------------------------------------------------------===//
  // Aliased (EQUIVALENCE) variables instantiation
  //===----------------------------------------------------------------===//

  static void insertAggregateStore(AggregateStoreMap &storeMap,
                                   const Fortran::lower::pft::Variable &var,
                                   mlir::Value aggregateStore) {
    auto off = var.getAggregateStore().getOffset();
    AggregateStoreKey key = {var.getOwningScope(), off};
    storeMap[key] = aggregateStore;
  }
  static mlir::Value
  getAggregateStore(AggregateStoreMap &storeMap,
                    const Fortran::lower::pft::Variable &alias) {
    AggregateStoreKey key = {alias.getOwningScope(), alias.getAlias()};
    auto iter = storeMap.find(key);
    assert(iter != storeMap.end());
    return storeMap.find(key)->second;
  }
  /// Build the name for the storage of a global equivalence.
  std::string mangleGlobalAggregateStore(
      const Fortran::lower::pft::Variable::AggregateStore &st) {
    assert(st.isGlobal() && "cannot name local aggregate");
    return mangleName(*st.vars[0]);
  }
  /// Build the type for the storage of an equivalence.
  mlir::TupleType
  getAggregateType(const Fortran::lower::pft::Variable::AggregateStore &st) {
    auto i8Ty = builder->getIntegerType(8);
    llvm::SmallVector<mlir::Type, 8> members;
    std::size_t counter = std::get<0>(st.interval);
    for (const auto *mem : st.vars) {
      if (const auto *memDet =
              mem->detailsIf<Fortran::semantics::ObjectEntityDetails>()) {
        if (mem->offset() > counter) {
          fir::SequenceType::Shape len = {
              static_cast<fir::SequenceType::Extent>(mem->offset() - counter)};
          auto byteTy = builder->getIntegerType(8);
          auto memTy = fir::SequenceType::get(len, byteTy);
          members.push_back(memTy);
          counter = mem->offset();
        }
        if (memDet->init()) {
          auto memTy = genType(*mem);
          members.push_back(memTy);
          counter = mem->offset() + mem->size();
        }
      }
    }
    if (counter < std::get<0>(st.interval) + std::get<1>(st.interval)) {
      fir::SequenceType::Shape len = {static_cast<fir::SequenceType::Extent>(
          std::get<0>(st.interval) + std::get<1>(st.interval) - counter)};
      auto memTy = fir::SequenceType::get(len, i8Ty);
      members.push_back(memTy);
    }
    return mlir::TupleType::get(builder->getContext(), members);
  }
  /// Define a GlobalOp for the storage of a global equivalence described
  /// by \p aggregate. The global is named \p aggName and is created with
  /// the provided \p linkage.
  /// If any of the equivalence members are initialized, an initializer is
  /// created for the equivalence.
  /// This is to be used when lowering the scope that owns the equivalence
  /// (as opposed to simply using it through host or use association).
  /// This is not to be used for equivalence of common block members (they
  /// already have the common block GlobalOp for them, see defineCommonBlock).
  fir::GlobalOp defineGlobalAggregateStore(
      const Fortran::lower::pft::Variable::AggregateStore &aggregate,
      StringRef aggName, mlir::StringAttr linkage) {
    assert(aggregate.isGlobal() && "not a global interval");
    auto loc = toLocation();
    auto idxTy = builder->getIndexType();
    mlir::TupleType aggTy = getAggregateType(aggregate);
    auto initFunc = [&](Fortran::lower::FirOpBuilder &builder) {
      mlir::Value cb = builder.create<fir::UndefOp>(loc, aggTy);
      unsigned tupIdx = 0;
      std::size_t offset = std::get<0>(aggregate.interval);
      LLVM_DEBUG(llvm::dbgs() << "equivalence {\n");
      for (const auto *mem : aggregate.vars) {
        if (const auto *memDet =
                mem->detailsIf<Fortran::semantics::ObjectEntityDetails>()) {
          if (mem->offset() > offset) {
            ++tupIdx;
            offset = mem->offset();
          }
          if (memDet->init()) {
            LLVM_DEBUG(llvm::dbgs() << "offset: " << mem->offset() << " is "
                                    << *mem << '\n');
            Fortran::lower::StatementContext stmtCtx;
            auto initVal =
                genInitializerExprValue(memDet->init().value(), stmtCtx);
            auto offVal = builder.createIntegerConstant(loc, idxTy, tupIdx);
            auto castVal = builder.createConvert(loc, aggTy.getType(tupIdx),
                                                 fir::getBase(initVal));
            cb = builder.create<fir::InsertValueOp>(loc, aggTy, cb, castVal,
                                                    offVal);
            ++tupIdx;
            offset = mem->offset() + mem->size();
          }
        }
      }
      LLVM_DEBUG(llvm::dbgs() << "}\n");
      builder.create<fir::HasValueOp>(loc, cb);
    };
    return builder->createGlobal(loc, aggTy, aggName,
                                 /*isConstant=*/false, initFunc, linkage);
  }
  /// Declare a GlobalOp for the storage of a global equivalence described
  /// by \p aggregate. The global is named \p aggName and is created with
  /// the provided \p linkage.
  /// No initializer is built for the created GlobalOp.
  /// This is to be used when lowering the scope that uses members of an
  /// equivalence it through host or use association.
  /// This is not to be used for equivalence of common block members (they
  /// already have the common block GlobalOp for them, see defineCommonBlock).
  fir::GlobalOp declareGlobalAggregateStore(
      const Fortran::lower::pft::Variable::AggregateStore &aggregate,
      StringRef aggName, mlir::StringAttr linkage) {
    assert(aggregate.isGlobal() && "not a global interval");
    auto loc = toLocation();
    auto aggTy = getAggregateType(aggregate);
    return builder->createGlobal(loc, aggTy, aggName, linkage);
  }
  /// This is an aggregate store for a set of EQUIVALENCED variables. Create the
  /// storage on the stack or global memory and add it to the map.
  void instantiateAggregateStore(const Fortran::lower::pft::Variable &var,
                                 AggregateStoreMap &storeMap) {
    assert(var.isAggregateStore() && "not an interval");
    auto i8Ty = builder->getIntegerType(8);
    auto loc = toLocation();
    if (var.isGlobal()) {
      // The scope of this aggregate is this procedure.
      auto aggName = mangleGlobalAggregateStore(var.getAggregateStore());
      fir::GlobalOp global = builder->getNamedGlobal(aggName);
      if (!global) {
        auto &aggregate = var.getAggregateStore();
        if (var.isDeclaration()) {
          // Using aggregate from a module not defined in the current
          // compilation unit.
          mlir::StringAttr externalLinkage;
          global =
              declareGlobalAggregateStore(aggregate, aggName, externalLinkage);
        } else {
          // The aggregate is owned by a procedure and must not be
          // visible in other compilation units.
          auto internalLinkage = builder->createInternalLinkage();
          global =
              defineGlobalAggregateStore(aggregate, aggName, internalLinkage);
        }
      }
      auto addr = builder->create<fir::AddrOfOp>(loc, global.resultType(),
                                                 global.getSymbol());
      auto size = std::get<1>(var.getInterval());
      fir::SequenceType::Shape shape(1, size);
      auto seqTy = fir::SequenceType::get(shape, i8Ty);
      auto refTy = builder->getRefType(seqTy);
      auto aggregateStore = builder->createConvert(loc, refTy, addr);
      insertAggregateStore(storeMap, var, aggregateStore);
      return;
    }
    // This is a local aggregate, allocate an anonymous block of memory.
    auto size = std::get<1>(var.getInterval());
    fir::SequenceType::Shape shape(1, size);
    auto seqTy = fir::SequenceType::get(shape, i8Ty);
    auto local =
        builder->allocateLocal(toLocation(), seqTy, "", llvm::None, llvm::None,
                               /*target=*/false);
    insertAggregateStore(storeMap, var, local);
  }
  /// Instantiate a member of an equivalence. Compute its address in its
  /// aggregate storage and lower its attributes.
  void instantiateAlias(const Fortran::lower::pft::Variable &var,
                        AggregateStoreMap &storeMap) {
    assert(var.isAlias());
    const auto &sym = var.getSymbol();
    const auto loc = genLocation(sym.name());
    auto idxTy = builder->getIndexType();
    auto aliasOffset = var.getAlias();
    auto store = getAggregateStore(storeMap, var);
    auto i8Ty = builder->getIntegerType(8);
    auto i8Ptr = builder->getRefType(i8Ty);
    auto offset =
        builder->createIntegerConstant(loc, idxTy, sym.offset() - aliasOffset);
    auto ptr = builder->create<fir::CoordinateOp>(loc, i8Ptr, store,
                                                  mlir::ValueRange{offset});
    auto preAlloc =
        builder->createConvert(loc, builder->getRefType(genType(sym)), ptr);
    Fortran::lower::StatementContext stmtCtx;
    mapSymbolAttributes(var, stmtCtx, preAlloc);
  }

  //===--------------------------------------------------------------===//
  // COMMON blocks instantiation
  //===--------------------------------------------------------------===//

  /// Does any member of the common block has an initializer ?
  static bool commonBlockHasInit(
      const Fortran::semantics::MutableSymbolVector &cmnBlkMems) {
    for (const auto &mem : cmnBlkMems) {
      if (const auto *memDet =
              mem->detailsIf<Fortran::semantics::ObjectEntityDetails>())
        if (memDet->init())
          return true;
    }
    return false;
  }
  /// Build a tuple type for a common block based on the common block
  /// members and the common block size.
  /// This type is only needed to build common block initializers where
  /// the initial value is the collection of the member initial values.
  mlir::TupleType getTypeOfCommonWithInit(
      const Fortran::semantics::MutableSymbolVector &cmnBlkMems,
      std::size_t commonSize) {
    llvm::SmallVector<mlir::Type, 8> members;
    std::size_t counter = 0;
    for (const auto &mem : cmnBlkMems) {
      if (const auto *memDet =
              mem->detailsIf<Fortran::semantics::ObjectEntityDetails>()) {
        if (mem->offset() > counter) {
          fir::SequenceType::Shape len = {
              static_cast<fir::SequenceType::Extent>(mem->offset() - counter)};
          auto byteTy = builder->getIntegerType(8);
          auto memTy = fir::SequenceType::get(len, byteTy);
          members.push_back(memTy);
          counter = mem->offset();
        }
        if (memDet->init()) {
          auto memTy = genType(*mem);
          members.push_back(memTy);
          counter = mem->offset() + mem->size();
        }
      }
    }
    if (counter < commonSize) {
      fir::SequenceType::Shape len = {
          static_cast<fir::SequenceType::Extent>(commonSize - counter)};
      auto byteTy = builder->getIntegerType(8);
      auto memTy = fir::SequenceType::get(len, byteTy);
      members.push_back(memTy);
    }
    return mlir::TupleType::get(builder->getContext(), members);
  }

  /// Common block members may have aliases. They are not in the common block
  /// member list from the symbol. We need to know about these aliases if they
  /// have initializer to generate the common initializer.
  /// This function takes care of adding aliases with initializer to the member
  /// list.
  Fortran::semantics::MutableSymbolVector
  getCommonMembersWithInitAliases(const Fortran::semantics::Symbol &common) {
    const auto &commonDetails =
        common.get<Fortran::semantics::CommonBlockDetails>();
    auto members = commonDetails.objects();

    // The number and size of equivalence and common is expected to be small, so
    // no effort is given to optimize this loop of complexity equivalenced
    // common members * common members
    for (const auto &set : common.owner().equivalenceSets())
      for (const auto &obj : set) {
        if (const auto &details =
                obj.symbol
                    .detailsIf<Fortran::semantics::ObjectEntityDetails>()) {
          const auto *com = FindCommonBlockContaining(obj.symbol);
          if (!details->init() || com != &common)
            continue;
          // This is an alias with an init that belongs to the list
          if (std::find(members.begin(), members.end(), obj.symbol) ==
              members.end())
            members.emplace_back(obj.symbol);
        }
      }
    return members;
  }

  /// Define a global for a common block if it does not already exist in the
  /// mlir module.
  /// There is no "declare" version since there is not a
  /// scope that owns common blocks more that the others. All scopes using
  /// a common block attempts to define it with common linkage.
  fir::GlobalOp defineCommonBlock(const Fortran::semantics::Symbol &common) {
    auto commonName = mangleName(common);
    auto global = builder->getNamedGlobal(commonName);
    if (global)
      return global;
    auto cmnBlkMems = getCommonMembersWithInitAliases(common);
    auto loc = genLocation(common.name());
    auto idxTy = builder->getIndexType();
    auto linkage = builder->createCommonLinkage();
    if (!common.name().size() || !commonBlockHasInit(cmnBlkMems)) {
      const auto sz = static_cast<fir::SequenceType::Extent>(common.size());
      // anonymous COMMON must always be initialized to zero
      // a named COMMON sans initializers is also initialized to zero
      fir::SequenceType::Shape shape = {sz};
      auto i8Ty = builder->getIntegerType(8);
      auto commonTy = fir::SequenceType::get(shape, i8Ty);
      auto vecTy = mlir::VectorType::get(sz, i8Ty);
      mlir::Attribute zero = builder->getIntegerAttr(i8Ty, 0);
      auto init = mlir::DenseElementsAttr::get(vecTy, llvm::makeArrayRef(zero));
      return builder->createGlobal(loc, commonTy, commonName, linkage, init);
    }

    // Named common with initializer, sort members by offset before generating
    // the type and initializer.
    std::sort(cmnBlkMems.begin(), cmnBlkMems.end(),
              [](auto &s1, auto &s2) { return s1->offset() < s2->offset(); });
    auto commonTy = getTypeOfCommonWithInit(cmnBlkMems, common.size());
    auto initFunc = [&](Fortran::lower::FirOpBuilder &builder) {
      mlir::Value cb = builder.create<fir::UndefOp>(loc, commonTy);
      unsigned tupIdx = 0;
      std::size_t offset = 0;
      LLVM_DEBUG(llvm::dbgs() << "block {\n");
      for (const auto &mem : cmnBlkMems) {
        if (const auto *memDet =
                mem->detailsIf<Fortran::semantics::ObjectEntityDetails>()) {
          if (mem->offset() > offset) {
            ++tupIdx;
            offset = mem->offset();
          }
          if (memDet->init()) {
            LLVM_DEBUG(llvm::dbgs() << "offset: " << mem->offset() << " is "
                                    << *mem << '\n');
            Fortran::lower::StatementContext stmtCtx;
            auto initVal =
                genInitializerExprValue(memDet->init().value(), stmtCtx);
            auto offVal = builder.createIntegerConstant(loc, idxTy, tupIdx);
            auto castVal = builder.createConvert(loc, commonTy.getType(tupIdx),
                                                 fir::getBase(initVal));
            cb = builder.create<fir::InsertValueOp>(loc, commonTy, cb, castVal,
                                                    offVal);
            ++tupIdx;
            offset = mem->offset() + mem->size();
          }
        }
      }
      LLVM_DEBUG(llvm::dbgs() << "}\n");
      builder.create<fir::HasValueOp>(loc, cb);
    };
    // create the global object
    return builder->createGlobal(loc, commonTy, commonName,
                                 /*isConstant=*/false, initFunc);
  }
  /// The COMMON block is a global structure. `var` will be at some offset
  /// within the COMMON block. Adds the address of `var` (COMMON + offset) to
  /// the symbol map.
  void instantiateCommon(const Fortran::semantics::Symbol &common,
                         const Fortran::lower::pft::Variable &var) {
    const auto &varSym = var.getSymbol();
    auto loc = genLocation(varSym.name());

    mlir::Value commonAddr;
    if (auto symBox = lookupSymbol(common))
      commonAddr = symBox.getAddr();
    if (!commonAddr) {
      // introduce a local AddrOf and add it to the map
      auto global = defineCommonBlock(common);
      commonAddr = builder->create<fir::AddrOfOp>(loc, global.resultType(),
                                                  global.getSymbol());
      addSymbol(common, commonAddr);
    }
    auto byteOffset = varSym.offset();
    auto i8Ty = builder->getIntegerType(8);
    auto i8Ptr = builder->getRefType(i8Ty);
    auto seqTy = builder->getRefType(builder->getVarLenSeqTy(i8Ty));
    auto base = builder->createConvert(loc, seqTy, commonAddr);
    auto offs = builder->createIntegerConstant(loc, builder->getIndexType(),
                                               byteOffset);
    auto varAddr = builder->create<fir::CoordinateOp>(loc, i8Ptr, base,
                                                      mlir::ValueRange{offs});
    auto localTy = builder->getRefType(genType(var.getSymbol()));
    mlir::Value local = builder->createConvert(loc, localTy, varAddr);
    Fortran::lower::StatementContext stmtCtx;
    mapSymbolAttributes(var, stmtCtx, local);
  }

  //===--------------------------------------------------------------===//
  // Lower Variables specification expressions and attributes
  //===--------------------------------------------------------------===//

  /// Lower specification expressions and attributes of variable \p var and
  /// add it to the symbol map.
  /// For global and aliases, the address must be pre-computed and provided
  /// in \p preAlloc.
  /// Dummy arguments must have already been mapped to mlir block arguments
  /// their mapping may be updated here.
  void mapSymbolAttributes(const Fortran::lower::pft::Variable &var,
                           Fortran::lower::StatementContext &stmtCtx,
                           mlir::Value preAlloc = {}) {
    const auto &sym = var.getSymbol();
    const auto loc = genLocation(sym.name());
    auto idxTy = builder->getIndexType();
    const auto isDummy = Fortran::semantics::IsDummy(sym);
    const auto isResult = Fortran::semantics::IsFunctionResult(sym);
    const auto replace = isDummy || isResult;
    const auto isHostAssoc =
        Fortran::semantics::IsHostAssociated(sym, sym.owner());
    Fortran::lower::CharacterExprHelper charHelp{*builder, loc};
    Fortran::lower::BoxAnalyzer sba;
    sba.analyze(sym);

    // First deal with pointers an allocatables, because their handling here
    // is the same regardless of their rank.
    if (Fortran::semantics::IsAllocatableOrPointer(sym)) {
      // Get address of fir.box describing the entity.
      // global
      auto boxAlloc = preAlloc;
      // dummy or passed result
      if (!boxAlloc)
        if (auto symbox = lookupSymbol(sym))
          boxAlloc = symbox.getAddr();
      // local
      if (!boxAlloc)
        boxAlloc = createNewLocal(loc, var, preAlloc);
      // Lower non deferred parameters.
      llvm::SmallVector<mlir::Value, 1> nonDeferredLenParams;
      auto lenTy = builder->getCharacterLengthType();
      if (sba.isChar()) {
        if (auto len = sba.getCharLenConst())
          nonDeferredLenParams.push_back(
              builder->createIntegerConstant(loc, lenTy, *len));
        else if (auto lenExpr = sba.getCharLenExpr())
          nonDeferredLenParams.push_back(
              createFIRExpr(loc, &*lenExpr, stmtCtx));
        // TODO: assumed length allocatable. Need to read the
        // input descriptor.
      }
      // TODO: non deferred derived type length parameters
      auto box = Fortran::lower::createMutableBox(*this, loc, var, boxAlloc,
                                                  nonDeferredLenParams);
      localSymbols.addAllocatableOrPointer(var.getSymbol(), box, replace);
      return;
    }

    // compute extent from lower and upper bound.
    auto computeExtent = [&](mlir::Value lb, mlir::Value ub) -> mlir::Value {
      // let the folder deal with the common `ub - <const> + 1` case
      auto diff = builder->create<mlir::SubIOp>(loc, idxTy, ub, lb);
      auto one = builder->createIntegerConstant(loc, idxTy, 1);
      return builder->create<mlir::AddIOp>(loc, idxTy, diff, one);
    };

    // The origin must be \vec{1}.
    auto populateShape = [&](auto &shapes, const auto &bounds,
                             mlir::Value box) {
      for (auto iter : llvm::enumerate(bounds)) {
        auto *spec = iter.value();
        assert(spec->lbound().GetExplicit() &&
               "lbound must be explicit with constant value 1");
        if (auto high = spec->ubound().GetExplicit()) {
          Fortran::semantics::SomeExpr highEx{*high};
          auto ub = createFIRExpr(loc, &highEx, stmtCtx);
          shapes.emplace_back(builder->createConvert(loc, idxTy, ub));
        } else if (spec->ubound().isDeferred()) {
          assert(box && "deferred bounds require a descriptor");
          auto dim = builder->createIntegerConstant(loc, idxTy, iter.index());
          auto dimInfo = builder->create<fir::BoxDimsOp>(loc, idxTy, idxTy,
                                                         idxTy, box, dim);
          shapes.emplace_back(dimInfo.getResult(1));
        } else if (spec->ubound().isAssumed()) {
          shapes.emplace_back(mlir::Value{});
        } else {
          llvm::report_fatal_error("unknown bound category");
        }
      }
    };

    // The origin is not \vec{1}.
    auto populateLBoundsExtents = [&](auto &lbounds, auto &extents,
                                      const auto &bounds, mlir::Value box) {
      for (auto iter : llvm::enumerate(bounds)) {
        auto *spec = iter.value();
        fir::BoxDimsOp dimInfo;
        mlir::Value ub, lb;
        if (spec->lbound().isDeferred() || spec->ubound().isDeferred()) {
          // This is an assumed shape because allocatables and pointers extents
          // are not constant in the scope and are not read here.
          assert(box && "deferred bounds require a descriptor");
          auto dim = builder->createIntegerConstant(loc, idxTy, iter.index());
          dimInfo = builder->create<fir::BoxDimsOp>(loc, idxTy, idxTy, idxTy,
                                                    box, dim);
          extents.emplace_back(dimInfo.getResult(1));
          if (auto low = spec->lbound().GetExplicit()) {
            auto expr = Fortran::semantics::SomeExpr{*low};
            auto lb = builder->createConvert(
                loc, idxTy, createFIRExpr(loc, &expr, stmtCtx));
            lbounds.emplace_back(lb);
          } else {
            // Implict lower bound is 1 (Fortan 2018 section 8.5.8.3 point 3.)
            lbounds.emplace_back(builder->createIntegerConstant(loc, idxTy, 1));
          }
        } else {
          if (auto low = spec->lbound().GetExplicit()) {
            auto expr = Fortran::semantics::SomeExpr{*low};
            lb = builder->createConvert(loc, idxTy,
                                        createFIRExpr(loc, &expr, stmtCtx));
          } else {
            TODO("assumed rank lowering");
          }

          if (auto high = spec->ubound().GetExplicit()) {
            auto expr = Fortran::semantics::SomeExpr{*high};
            ub = builder->createConvert(loc, idxTy,
                                        createFIRExpr(loc, &expr, stmtCtx));
            lbounds.emplace_back(lb);
            extents.emplace_back(computeExtent(lb, ub));
          } else {
            // An assumed size array. The extent is not computed.
            assert(spec->ubound().isAssumed() && "expected assumed size");
            lbounds.emplace_back(lb);
            extents.emplace_back(mlir::Value{});
          }
        }
      }
    };

    if (isHostAssoc)
      TODO("host associated");

    sba.match(
        //===--------------------------------------------------------------===//
        // Trivial case.
        //===--------------------------------------------------------------===//
        [&](const Fortran::lower::details::ScalarSym &) {
          if (isDummy) {
            // This is an argument.
            if (!lookupSymbol(sym))
              mlir::emitError(loc, "symbol \"")
                  << toStringRef(sym.name()) << "\" must already be in map";
            return;
          }
          // Otherwise, it's a local variable or function result.
          auto local = createNewLocal(loc, var, preAlloc);
          addSymbol(sym, local);
        },

        //===--------------------------------------------------------------===//
        // The non-trivial cases are when we have an argument or local that has
        // a repetition value. Arguments might be passed as simple pointers and
        // need to be cast to a multi-dimensional array with constant bounds
        // (possibly with a missing column), bounds computed in the callee
        // (here), or with bounds from the caller (boxed somewhere else). Locals
        // have the same properties except they are never boxed arguments from
        // the caller and never having a missing column size.
        //===--------------------------------------------------------------===//

        [&](const Fortran::lower::details::ScalarStaticChar &x) {
          // type is a CHARACTER, determine the LEN value
          auto charLen = x.charLen();
          if (replace) {
            auto symBox = lookupSymbol(sym);
            auto unboxchar = charHelp.createUnboxChar(symBox.getAddr());
            auto boxAddr = unboxchar.first;
            // Set/override LEN with a constant
            auto len = builder->createIntegerConstant(loc, idxTy, charLen);
            addCharSymbol(sym, boxAddr, len, true);
            return;
          }
          auto len = builder->createIntegerConstant(loc, idxTy, charLen);
          if (preAlloc) {
            addCharSymbol(sym, preAlloc, len);
            return;
          }
          auto local = createNewLocal(loc, var, preAlloc);
          addCharSymbol(sym, local, len);
        },

        //===--------------------------------------------------------------===//

        [&](const Fortran::lower::details::ScalarDynamicChar &x) {
          // type is a CHARACTER, determine the LEN value
          auto charLen = x.charLen();
          if (replace) {
            auto symBox = lookupSymbol(sym);
            auto boxAddr = symBox.getAddr();
            mlir::Value len;
            auto addrTy = boxAddr.getType();
            if (addrTy.isa<fir::BoxCharType>() || addrTy.isa<fir::BoxType>()) {
              std::tie(boxAddr, len) =
                  charHelp.createUnboxChar(symBox.getAddr());
            } else {
              // dummy from an other entry case: we cannot get a dynamic length
              // for it, it's illegal for the user program to use it. However,
              // since we are lowering all function unit statements regardless
              // of whether the execution will reach them or not, we need to
              // fill a value for the length here.
              len = builder->createIntegerConstant(
                  loc, builder->getCharacterLengthType(), 1);
            }
            // Override LEN with an expression
            if (charLen)
              len = createFIRExpr(loc, &*charLen, stmtCtx);
            addCharSymbol(sym, boxAddr, len, true);
            return;
          }
          // local CHARACTER variable
          mlir::Value len;
          if (charLen)
            len = createFIRExpr(loc, &*charLen, stmtCtx);
          else
            len = builder->createIntegerConstant(loc, idxTy, sym.size());
          if (preAlloc) {
            addCharSymbol(sym, preAlloc, len);
            return;
          }
          llvm::SmallVector<mlir::Value, 1> lengths = {len};
          auto local = createNewLocal(loc, var, preAlloc, llvm::None, lengths);
          addCharSymbol(sym, local, len);
        },

        //===--------------------------------------------------------------===//

        [&](const Fortran::lower::details::StaticArray &x) {
          // object shape is constant, not a character
          auto castTy = builder->getRefType(genType(var));
          mlir::Value addr = lookupSymbol(sym).getAddr();
          if (addr)
            addr = builder->createConvert(loc, castTy, addr);
          if (x.lboundAllOnes()) {
            // if lower bounds are all ones, build simple shaped object
            llvm::SmallVector<mlir::Value, 8> shape;
            for (auto i : x.shapes)
              shape.push_back(builder->createIntegerConstant(loc, idxTy, i));
            mlir::Value local =
                replace ? addr : createNewLocal(loc, var, preAlloc);
            localSymbols.addSymbolWithShape(
                sym, local, shape, /*sourceBox*/ mlir::Value{}, replace);
            return;
          }
          // If object is an array process the lower bound and extent values by
          // constructing constants and populating the lbounds and extents.
          llvm::SmallVector<mlir::Value, 8> extents;
          llvm::SmallVector<mlir::Value, 8> lbounds;
          for (auto [fst, snd] : llvm::zip(x.lbounds, x.shapes)) {
            lbounds.emplace_back(
                builder->createIntegerConstant(loc, idxTy, fst));
            extents.emplace_back(
                builder->createIntegerConstant(loc, idxTy, snd));
          }
          mlir::Value local =
              replace ? addr : createNewLocal(loc, var, preAlloc, extents);
          assert(replace || Fortran::lower::isExplicitShape(sym) ||
                 Fortran::semantics::IsAllocatableOrPointer(sym));
          localSymbols.addSymbolWithBounds(sym, local, extents, lbounds,
                                           /*sourceBox*/ mlir::Value{},
                                           replace);
        },

        //===--------------------------------------------------------------===//

        [&](const Fortran::lower::details::DynamicArray &x) {
          // cast to the known constant parts from the declaration
          auto varType = genType(var);
          mlir::Value addr = lookupSymbol(sym).getAddr();
          mlir::Value argBox;
          auto castTy = builder->getRefType(varType);
          if (addr) {
            if (auto boxTy = addr.getType().dyn_cast<fir::BoxType>()) {
              argBox = addr;
              auto refTy = builder->getRefType(boxTy.getEleTy());
              addr = builder->create<fir::BoxAddrOp>(loc, refTy, argBox);
            }
            addr = builder->createConvert(loc, castTy, addr);
          }
          if (x.lboundAllOnes()) {
            // if lower bounds are all ones, build simple shaped object
            llvm::SmallVector<mlir::Value, 8> shapes;
            populateShape(shapes, x.bounds, argBox);
            if (isDummy || isResult) {
              localSymbols.addSymbolWithShape(sym, addr, shapes,
                                              /*sourceBox*/ argBox, true);
              return;
            }
            // local array with computed bounds
            assert(Fortran::lower::isExplicitShape(sym) ||
                   Fortran::semantics::IsAllocatableOrPointer(sym));
            auto local = createNewLocal(loc, var, preAlloc, shapes);
            localSymbols.addSymbolWithShape(sym, local, shapes);
            return;
          }
          // if object is an array process the lower bound and extent values
          llvm::SmallVector<mlir::Value, 8> extents;
          llvm::SmallVector<mlir::Value, 8> lbounds;
          populateLBoundsExtents(lbounds, extents, x.bounds, argBox);
          if (isDummy || isResult) {
            localSymbols.addSymbolWithBounds(sym, addr, extents, lbounds,
                                             /*sourceBox*/ argBox, true);
            return;
          }
          // local array with computed bounds
          assert(Fortran::lower::isExplicitShape(sym) ||
                 Fortran::semantics::IsAllocatableOrPointer(sym));
          auto local = createNewLocal(loc, var, preAlloc, extents);
          localSymbols.addSymbolWithBounds(sym, local, extents, lbounds);
        },

        //===--------------------------------------------------------------===//

        [&](const Fortran::lower::details::StaticArrayStaticChar &x) {
          // if element type is a CHARACTER, determine the LEN value
          auto charLen = x.charLen();
          mlir::Value addr;
          mlir::Value len;
          if (isDummy || isResult) {
            auto symBox = lookupSymbol(sym);
            auto unboxchar = charHelp.createUnboxChar(symBox.getAddr());
            addr = unboxchar.first;
            // Set/override LEN with a constant
            len = builder->createIntegerConstant(loc, idxTy, charLen);
          } else {
            // local CHARACTER variable
            len = builder->createIntegerConstant(loc, idxTy, charLen);
          }

          // object shape is constant
          auto castTy = builder->getRefType(genType(var));
          if (addr)
            addr = builder->createConvert(loc, castTy, addr);

          if (x.lboundAllOnes()) {
            // if lower bounds are all ones, build simple shaped object
            llvm::SmallVector<mlir::Value, 8> shape;
            for (auto i : x.shapes)
              shape.push_back(builder->createIntegerConstant(loc, idxTy, i));
            mlir::Value local =
                replace ? addr : createNewLocal(loc, var, preAlloc);
            localSymbols.addCharSymbolWithShape(
                sym, local, len, shape, /*sourceBox*/ mlir::Value{}, replace);
            return;
          }

          // if object is an array process the lower bound and extent values
          llvm::SmallVector<mlir::Value, 8> extents;
          llvm::SmallVector<mlir::Value, 8> lbounds;
          // construct constants and populate `bounds`
          for (auto [fst, snd] : llvm::zip(x.lbounds, x.shapes)) {
            lbounds.emplace_back(
                builder->createIntegerConstant(loc, idxTy, fst));
            extents.emplace_back(
                builder->createIntegerConstant(loc, idxTy, snd));
          }

          if (isDummy || isResult) {
            localSymbols.addCharSymbolWithBounds(
                sym, addr, len, extents, lbounds, /*sourceBox*/ mlir::Value{},
                true);
            return;
          }
          // local CHARACTER array with computed bounds
          assert(Fortran::lower::isExplicitShape(sym) ||
                 Fortran::semantics::IsAllocatableOrPointer(sym));
          llvm::SmallVector<mlir::Value, 1> lengths = {len};
          auto local = createNewLocal(loc, var, preAlloc, extents, lengths);
          localSymbols.addCharSymbolWithBounds(sym, local, len, extents,
                                               lbounds);
        },

        //===--------------------------------------------------------------===//

        [&](const Fortran::lower::details::StaticArrayDynamicChar &x) {
          mlir::Value addr;
          mlir::Value len;
          bool mustBeDummy = false;
          auto charLen = x.charLen();
          // if element type is a CHARACTER, determine the LEN value
          if (isDummy || isResult) {
            auto symBox = lookupSymbol(sym);
            auto unboxchar = charHelp.createUnboxChar(symBox.getAddr());
            addr = unboxchar.first;
            if (charLen) {
              // Set/override LEN with an expression
              len = createFIRExpr(loc, &*charLen, stmtCtx);
            } else {
              // LEN is from the boxchar
              len = unboxchar.second;
              mustBeDummy = true;
            }
          } else {
            // local CHARACTER variable
            if (charLen)
              len = createFIRExpr(loc, &*charLen, stmtCtx);
            else
              len = builder->createIntegerConstant(loc, idxTy, sym.size());
          }
          llvm::SmallVector<mlir::Value, 1> lengths = {len};

          // cast to the known constant parts from the declaration
          auto castTy = builder->getRefType(genType(var));
          if (addr)
            addr = builder->createConvert(loc, castTy, addr);

          if (x.lboundAllOnes()) {
            // if lower bounds are all ones, build simple shaped object
            llvm::SmallVector<mlir::Value, 8> shape;
            for (auto i : x.shapes)
              shape.push_back(builder->createIntegerConstant(loc, idxTy, i));
            if (isDummy || isResult) {
              localSymbols.addCharSymbolWithShape(
                  sym, addr, len, shape, /*sourceBox*/ mlir::Value{}, true);
              return;
            }
            // local CHARACTER array with constant size
            auto local =
                createNewLocal(loc, var, preAlloc, llvm::None, lengths);
            localSymbols.addCharSymbolWithShape(sym, local, len, shape);
            return;
          }

          // if object is an array process the lower bound and extent values
          llvm::SmallVector<mlir::Value, 8> extents;
          llvm::SmallVector<mlir::Value, 8> lbounds;

          // construct constants and populate `bounds`
          for (auto [fst, snd] : llvm::zip(x.lbounds, x.shapes)) {
            lbounds.emplace_back(
                builder->createIntegerConstant(loc, idxTy, fst));
            extents.emplace_back(
                builder->createIntegerConstant(loc, idxTy, snd));
          }
          if (isDummy || isResult) {
            localSymbols.addCharSymbolWithBounds(
                sym, addr, len, extents, lbounds, /*sourceBox*/ mlir::Value{},
                true);
            return;
          }
          // local CHARACTER array with computed bounds
          assert((!mustBeDummy) &&
                 (Fortran::lower::isExplicitShape(sym) ||
                  Fortran::semantics::IsAllocatableOrPointer(sym)));
          auto local = createNewLocal(loc, var, preAlloc, llvm::None, lengths);
          localSymbols.addCharSymbolWithBounds(sym, local, len, extents,
                                               lbounds);
        },

        //===--------------------------------------------------------------===//

        [&](const Fortran::lower::details::DynamicArrayStaticChar &x) {
          mlir::Value addr;
          mlir::Value len;
          mlir::Value argBox;
          auto charLen = x.charLen();
          // if element type is a CHARACTER, determine the LEN value
          if (isDummy || isResult) {
            auto actualArg = lookupSymbol(sym).getAddr();
            if (auto boxTy = actualArg.getType().dyn_cast<fir::BoxType>()) {
              argBox = actualArg;
              auto refTy = builder->getRefType(boxTy.getEleTy());
              addr = builder->create<fir::BoxAddrOp>(loc, refTy, argBox);
            } else {
              addr = charHelp.createUnboxChar(actualArg).first;
            }
            // Set/override LEN with a constant
            len = builder->createIntegerConstant(loc, idxTy, charLen);
          } else {
            // local CHARACTER variable
            len = builder->createIntegerConstant(loc, idxTy, charLen);
          }

          // cast to the known constant parts from the declaration
          auto castTy = builder->getRefType(genType(var));
          if (addr)
            addr = builder->createConvert(loc, castTy, addr);
          if (x.lboundAllOnes()) {
            // if lower bounds are all ones, build simple shaped object
            llvm::SmallVector<mlir::Value, 8> shape;
            populateShape(shape, x.bounds, argBox);
            if (isDummy || isResult) {
              localSymbols.addCharSymbolWithShape(
                  sym, addr, len, shape, /*sourceBox*/ mlir::Value{}, true);
              return;
            }
            // local CHARACTER array
            auto local = createNewLocal(loc, var, preAlloc, shape);
            localSymbols.addCharSymbolWithShape(sym, local, len, shape);
            return;
          }
          // if object is an array process the lower bound and extent values
          llvm::SmallVector<mlir::Value, 8> extents;
          llvm::SmallVector<mlir::Value, 8> lbounds;
          populateLBoundsExtents(lbounds, extents, x.bounds, argBox);
          if (isDummy || isResult) {
            localSymbols.addCharSymbolWithBounds(
                sym, addr, len, extents, lbounds, /*sourceBox*/ mlir::Value{},
                true);
            return;
          }
          // local CHARACTER array with computed bounds
          assert(Fortran::lower::isExplicitShape(sym) ||
                 Fortran::semantics::IsAllocatableOrPointer(sym));
          auto local = createNewLocal(loc, var, preAlloc, extents);
          localSymbols.addCharSymbolWithBounds(sym, local, len, extents,
                                               lbounds);
        },

        //===--------------------------------------------------------------===//

        [&](const Fortran::lower::details::DynamicArrayDynamicChar &x) {
          mlir::Value addr;
          mlir::Value len;
          mlir::Value argBox;
          auto charLen = x.charLen();
          // if element type is a CHARACTER, determine the LEN value
          if (isDummy || isResult) {
            auto actualArg = lookupSymbol(sym).getAddr();
            if (auto boxTy = actualArg.getType().dyn_cast<fir::BoxType>()) {
              argBox = actualArg;
              auto refTy = builder->getRefType(boxTy.getEleTy());
              addr = builder->create<fir::BoxAddrOp>(loc, refTy, argBox);
              if (charLen)
                // Set/override LEN with an expression.
                len = createFIRExpr(loc, &*charLen, stmtCtx);
              else
                // Get the length from the actual arguments.
                len = charHelp.readLengthFromBox(argBox);
            } else {
              auto unboxchar = charHelp.createUnboxChar(actualArg);
              addr = unboxchar.first;
              if (charLen) {
                // Set/override LEN with an expression
                len = createFIRExpr(loc, &*charLen, stmtCtx);
              } else {
                // Get the length from the actual arguments.
                len = unboxchar.second;
              }
            }
          } else {
            // local CHARACTER variable
            if (charLen)
              len = createFIRExpr(loc, &*charLen, stmtCtx);
            else
              len = builder->createIntegerConstant(loc, idxTy, sym.size());
          }
          llvm::SmallVector<mlir::Value, 1> lengths = {len};

          // cast to the known constant parts from the declaration
          auto castTy = builder->getRefType(genType(var));
          if (addr)
            addr = builder->createConvert(loc, castTy, addr);
          if (x.lboundAllOnes()) {
            // if lower bounds are all ones, build simple shaped object
            llvm::SmallVector<mlir::Value, 8> shape;
            populateShape(shape, x.bounds, argBox);
            if (isDummy || isResult) {
              localSymbols.addCharSymbolWithShape(
                  sym, addr, len, shape, /*sourceBox*/ mlir::Value{}, true);
              return;
            }
            // local CHARACTER array
            auto local = createNewLocal(loc, var, preAlloc, shape, lengths);
            localSymbols.addCharSymbolWithShape(sym, local, len, shape);
            return;
          }
          // Process the lower bound and extent values.
          llvm::SmallVector<mlir::Value, 8> extents;
          llvm::SmallVector<mlir::Value, 8> lbounds;
          populateLBoundsExtents(lbounds, extents, x.bounds, argBox);
          if (isDummy || isResult) {
            localSymbols.addCharSymbolWithBounds(
                sym, addr, len, extents, lbounds, /*sourceBox*/ mlir::Value{},
                true);
            return;
          }
          // local CHARACTER array with computed bounds
          assert(Fortran::lower::isExplicitShape(sym) ||
                 Fortran::semantics::IsAllocatableOrPointer(sym));
          auto local = createNewLocal(loc, var, preAlloc, extents, lengths);
          localSymbols.addCharSymbolWithBounds(sym, local, len, extents,
                                               lbounds);
        },

        //===--------------------------------------------------------------===//

        [&](const Fortran::lower::BoxAnalyzer::None &) {
          mlir::emitError(loc, "symbol analysis failed on ")
              << toStringRef(sym.name());
        });
  }

  /// Map mlir function block arguments to the corresponding Fortran dummy
  /// variables. When the result is passed as a hidden argument, the Fortran
  /// result is also mapped. The symbol map is used to hold this mapping.
  void mapDummiesAndResults(const Fortran::lower::pft::FunctionLikeUnit &funit,
                            const Fortran::lower::CalleeInterface &callee) {
    assert(builder && "need a builder at this point");
    using PassBy = Fortran::lower::CalleeInterface::PassEntityBy;
    auto mapPassedEntity = [&](const auto arg) -> void {
      if (arg.passBy == PassBy::AddressAndLength) {
        // TODO: now that fir call has some attributes regarding character
        // return, this should PassBy::AddressAndLength should be retired.
        auto loc = toLocation();
        Fortran::lower::CharacterExprHelper charHelp{*builder, loc};
        auto box = charHelp.createEmboxChar(arg.firArgument, arg.firLength);
        addSymbol(arg.entity.get(), box);
      } else {
        addSymbol(arg.entity.get(), arg.firArgument);
      }
    };
    for (const auto &arg : callee.getPassedArguments())
      mapPassedEntity(arg);

    // Allocate local skeleton instances of dummies from other entry points.
    // Most of these locals will not survive into final generated code, but
    // some will.  It is illegal to reference them at run time if they do.
    for (const auto *arg : funit.nonUniversalDummyArguments) {
      if (lookupSymbol(*arg))
        continue;
      auto type = genType(*arg);
      // TODO: Account for VALUE arguments (and possibly other variants).
      type = builder->getRefType(type);
      addSymbol(*arg, builder->create<fir::UndefOp>(toLocation(), type));
    }
    if (auto passedResult = callee.getPassedResult()) {
      mapPassedEntity(*passedResult);
      // FIXME: need to make sure things are OK here. addSymbol is may not be OK
      if (funit.primaryResult &&
          passedResult->entity.get() != *funit.primaryResult)
        addSymbol(*funit.primaryResult, getSymbolAddress(passedResult->entity));
    }
  }

  /// Prepare to translate a new function
  void startNewFunction(Fortran::lower::pft::FunctionLikeUnit &funit) {
    assert(!builder && "expected nullptr");
    Fortran::lower::CalleeInterface callee(funit, *this);
    mlir::FuncOp func = callee.addEntryBlockAndMapArguments();
    builder = new Fortran::lower::FirOpBuilder(func, bridge.getKindMap());
    assert(builder && "FirOpBuilder did not instantiate");
    builder->setInsertionPointToStart(&func.front());
    func.setVisibility(mlir::SymbolTable::Visibility::Public);

    mapDummiesAndResults(funit, callee);

    // Note: not storing Variable references because getOrderedSymbolTable
    // below returns a temporary.
    llvm::SmallVector<Fortran::lower::pft::Variable, 4> deferredFuncResultList;

    // Backup actual argument for entry character results
    // with different lengths. It needs to be added to the non
    // primary results symbol before mapSymbolAttributes is called.
    Fortran::lower::SymbolBox resultArg;
    if (auto passedResult = callee.getPassedResult())
      resultArg = lookupSymbol(passedResult->entity.get());

    AggregateStoreMap storeMap;
    // The front-end is currently not adding module variables referenced
    // in a module procedure as host associated. As a result we need to
    // instantiate all module variables here if this is a module procedure.
    // It is likely that the front-end behaviour should change here.
    if (auto *module =
            funit.parentVariant.getIf<Fortran::lower::pft::ModuleLikeUnit>())
      for (const auto &var : module->getOrderedSymbolTable())
        instantiateVar(var, storeMap);

    mlir::Value primaryFuncResultStorage;
    for (const auto &var : funit.getOrderedSymbolTable()) {
      if (var.isAggregateStore()) {
        instantiateVar(var, storeMap);
        continue;
      }
      const Fortran::semantics::Symbol &sym = var.getSymbol();
      if (!sym.IsFuncResult() || !funit.primaryResult) {
        instantiateVar(var, storeMap);
      } else if (&sym == funit.primaryResult) {
        instantiateVar(var, storeMap);
        primaryFuncResultStorage = getSymbolAddress(sym);
      } else {
        deferredFuncResultList.push_back(var);
      }
    }

    /// TODO: should use same mechanism as equivalence?
    /// One blocking point is character entry returns that need special handling
    /// since they are not locally allocated but come as argument. CHARACTER(*)
    /// is not something that fit wells with equivalence lowering.
    for (const auto &altResult : deferredFuncResultList) {
      if (auto passedResult = callee.getPassedResult())
        addSymbol(altResult.getSymbol(), resultArg.getAddr());
      Fortran::lower::StatementContext stmtCtx;
      mapSymbolAttributes(altResult, stmtCtx, primaryFuncResultStorage);
    }

    // Create most function blocks in advance.
    auto *alternateEntryEval = funit.getEntryEval();
    if (alternateEntryEval) {
      // Move to executable successor.
      alternateEntryEval = alternateEntryEval->lexicalSuccessor;
      auto evalIsNewBlock = alternateEntryEval->isNewBlock;
      alternateEntryEval->isNewBlock = true;
      createEmptyBlocks(funit.evaluationList);
      alternateEntryEval->isNewBlock = evalIsNewBlock;
    } else {
      createEmptyBlocks(funit.evaluationList);
    }

    // Reinstate entry block as the current insertion point.
    builder->setInsertionPointToEnd(&func.front());

    if (callee.hasAlternateReturns()) {
      // Create a local temp to hold the alternate return index.
      // Give it an integer index type and the subroutine name (for dumps).
      // Attach it to the subroutine symbol in the localSymbols map.
      // Initialize it to zero, the "fallthrough" alternate return value.
      const auto &symbol = funit.getSubprogramSymbol();
      auto loc = toLocation();
      auto idxTy = builder->getIndexType();
      const auto altResult =
          builder->createTemporary(loc, idxTy, toStringRef(symbol.name()));
      addSymbol(symbol, altResult);
      const auto zero = builder->createIntegerConstant(loc, idxTy, 0);
      builder->create<fir::StoreOp>(loc, zero, altResult);
    }

    if (alternateEntryEval) {
      genFIRBranch(alternateEntryEval->block);
      builder->setInsertionPointToStart(
          builder->createBlock(&builder->getRegion()));
    }
  }

  /// Create empty blocks for the current function.
  void createEmptyBlocks(
      std::list<Fortran::lower::pft::Evaluation> &evaluationList) {
    auto *region = &builder->getRegion();
    for (auto &eval : evaluationList) {
      if (eval.isNewBlock)
        eval.block = builder->createBlock(region);
      for (auto &block : eval.localBlocks)
        block = builder->createBlock(region);
      if (eval.isConstruct() || eval.isDirective()) {
        if (eval.lowerAsUnstructured()) {
          createEmptyBlocks(eval.getNestedEvaluations());
        } else if (eval.hasNestedEvaluations()) {
          // A structured construct that is a target starts a new block.
          auto &constructStmt = eval.getFirstNestedEvaluation();
          if (constructStmt.isNewBlock)
            constructStmt.block = builder->createBlock(region);
        }
      }
    }
  }

  /// Return the predicate: "current block does not have a terminator branch".
  bool blockIsUnterminated() {
    auto *currentBlock = builder->getBlock();
    return currentBlock->empty() || currentBlock->back().isKnownNonTerminator();
  }

  /// Unconditionally switch code insertion to a new block.
  void startBlock(mlir::Block *newBlock) {
    assert(newBlock && "missing block");
    if (blockIsUnterminated())
      genFIRBranch(newBlock); // default termination is a fallthrough branch
    builder->setInsertionPointToEnd(newBlock); // newBlock might not be empty
  }

  /// Conditionally switch code insertion to a new block.
  void maybeStartBlock(mlir::Block *newBlock) {
    if (newBlock)
      startBlock(newBlock);
  }

  /// Emit return and cleanup after the function has been translated.
  void endNewFunction(Fortran::lower::pft::FunctionLikeUnit &funit) {
    setCurrentPosition(
        Fortran::lower::pft::FunctionLikeUnit::stmtSourceLoc(funit.endStmt));
    if (funit.isMainProgram())
      genExitRoutine();
    else
      genFIRProcedureExit(funit, funit.getSubprogramSymbol());
    funit.finalBlock = nullptr;
    mlir::simplifyRegions({builder->getRegion()}); // remove dead code
    delete builder;
    builder = nullptr;
    localSymbols.clear();
  }

  /// Instantiate the data from a BLOCK DATA unit.
  void lowerBlockData(Fortran::lower::pft::BlockDataUnit &bdunit) {
    // FIXME: get rid of the bogus function context and instantiate the
    // globals directly into the module.
    auto *context = &getMLIRContext();
    auto func = Fortran::lower::FirOpBuilder::createFunction(
        mlir::UnknownLoc::get(context), getModuleOp(),
        uniquer.doGenerated("Sham"),
        mlir::FunctionType::get(context, llvm::None, llvm::None));

    builder = new Fortran::lower::FirOpBuilder(func, bridge.getKindMap());
    AggregateStoreMap fakeMap;
    for (const auto &[_, sym] : bdunit.symTab) {
      Fortran::lower::pft::Variable var(*sym, true);
      instantiateVar(var, fakeMap);
    }

    if (auto *region = func.getCallableRegion())
      region->dropAllReferences();
    func.erase();
    delete builder;
    builder = nullptr;
    localSymbols.clear();
  }

  /// Lower a procedure (nest).
  void lowerFunc(Fortran::lower::pft::FunctionLikeUnit &funit) {
    for (int entryIndex = 0, last = funit.entryPointList.size();
         entryIndex < last; ++entryIndex) {
      funit.setActiveEntry(entryIndex);
      startNewFunction(funit); // this entry point of this procedure
      for (auto &eval : funit.evaluationList)
        genFIR(eval);
      endNewFunction(funit);
    }
    funit.setActiveEntry(0);
    for (auto &f : funit.nestedFunctions)
      lowerFunc(f); // internal procedure
  }

  /// Lower module variable definitions to fir::globalOp
  void lowerModuleVariables(Fortran::lower::pft::ModuleLikeUnit &mod) {
    // FIXME: get rid of the bogus function context and instantiate the
    // globals directly into the module.
    auto *context = &getMLIRContext();
    auto func = Fortran::lower::FirOpBuilder::createFunction(
        mlir::UnknownLoc::get(context), getModuleOp(),
        uniquer.doGenerated("ModuleSham"),
        mlir::FunctionType::get(context, llvm::None, llvm::None));
    builder = new Fortran::lower::FirOpBuilder(func, bridge.getKindMap());
    // linkOnce linkage allows the definition to be kept by LLVM
    // even if it is unused and there is not init (other than undef).
    // Emitting llvm :'@glob = global type undef' would also work, but mlir
    // always insert "external" and removes the undef init when lowering a
    // global without explicit linkage. This ends-up in llvm removing the
    // symbol if unsued potentially creating linking issues. Hence the use
    // of linkOnce.
    auto linkOnce = builder->createLinkOnceLinkage();
    for (const auto &var : mod.getOrderedSymbolTable()) {
      // Only define variable owned by this module
      if (var.isDeclaration())
        continue;
      if (!var.isGlobal()) {
        mlir::emitError(toLocation(),
                        "attempting to lower module variable as local");
        continue;
      }
      // Define aggregate storages for equivalenced objects.
      if (var.isAggregateStore()) {
        auto &aggregate = var.getAggregateStore();
        auto aggName = mangleGlobalAggregateStore(aggregate);
        defineGlobalAggregateStore(aggregate, aggName, linkOnce);
        continue;
      }
      const auto &sym = var.getSymbol();
      if (const auto *common =
              Fortran::semantics::FindCommonBlockContaining(var.getSymbol())) {
        // Define common block containing the variable.
        defineCommonBlock(*common);
      } else if (var.isAlias()) {
        // Do nothing. Mapping will be done on user side.
      } else {
        auto globalName = mangleName(sym);
        defineGlobal(var, globalName, linkOnce);
      }
    }
    if (auto *region = func.getCallableRegion())
      region->dropAllReferences();
    func.erase();
    delete builder;
    builder = nullptr;
  }

  /// Lower functions contained in a module.
  void lowerMod(Fortran::lower::pft::ModuleLikeUnit &mod) {
    for (auto &f : mod.nestedFunctions)
      lowerFunc(f);
  }

  void setCurrentPosition(const Fortran::parser::CharBlock &position) {
    if (position != Fortran::parser::CharBlock{})
      currentPosition = position;
  }

  //
  // Utility methods
  //

  /// Convert a parser CharBlock to a Location
  mlir::Location toLocation(const Fortran::parser::CharBlock &cb) {
    return genLocation(cb);
  }

  mlir::Location toLocation() { return toLocation(currentPosition); }
  void setCurrentEval(Fortran::lower::pft::Evaluation &eval) {
    evalPtr = &eval;
  }
  Fortran::lower::pft::Evaluation &getEval() {
    assert(evalPtr);
    return *evalPtr;
  }

  std::optional<Fortran::evaluate::Shape>
  getShape(const Fortran::lower::SomeExpr &expr) {
    return Fortran::evaluate::GetShape(foldingContext, expr);
  }

  Fortran::lower::LoweringBridge &bridge;
  fir::NameUniquer &uniquer;
  Fortran::evaluate::FoldingContext foldingContext;
  Fortran::lower::FirOpBuilder *builder = nullptr;
  Fortran::lower::pft::Evaluation *evalPtr = nullptr;
  Fortran::lower::SymMap localSymbols;
  Fortran::parser::CharBlock currentPosition;
};

} // namespace

Fortran::evaluate::FoldingContext
Fortran::lower::LoweringBridge::createFoldingContext() const {
  return {getDefaultKinds(), getIntrinsicTable()};
}

void Fortran::lower::LoweringBridge::lower(
    const Fortran::parser::Program &prg,
    const Fortran::semantics::SemanticsContext &semanticsContext) {
  auto pft = Fortran::lower::createPFT(prg, semanticsContext);
  if (dumpBeforeFir)
    Fortran::lower::dumpPFT(llvm::errs(), *pft);
  FirConverter converter{*this, *fir::getNameUniquer(getModule())};
  converter.run(*pft);
}

void Fortran::lower::LoweringBridge::parseSourceFile(llvm::SourceMgr &srcMgr) {
  auto owningRef = mlir::parseSourceFile(srcMgr, &context);
  module.reset(new mlir::ModuleOp(owningRef.get().getOperation()));
  owningRef.release();
}

Fortran::lower::LoweringBridge::LoweringBridge(
    mlir::MLIRContext &context,
    const Fortran::common::IntrinsicTypeDefaultKinds &defaultKinds,
    const Fortran::evaluate::IntrinsicProcTable &intrinsics,
    const Fortran::parser::AllCookedSources &cooked, llvm::Triple &triple,
    fir::NameUniquer &uniquer, fir::KindMapping &kindMap)
    : defaultKinds{defaultKinds}, intrinsics{intrinsics}, cooked{&cooked},
      context{context}, kindMap{kindMap} {
  // Register the diagnostic handler.
  context.getDiagEngine().registerHandler([](mlir::Diagnostic &diag) {
    auto &os = llvm::errs();
    switch (diag.getSeverity()) {
    case mlir::DiagnosticSeverity::Error:
      os << "error: ";
      break;
    case mlir::DiagnosticSeverity::Remark:
      os << "info: ";
      break;
    case mlir::DiagnosticSeverity::Warning:
      os << "warning: ";
      break;
    default:
      break;
    }
    if (!diag.getLocation().isa<UnknownLoc>())
      os << diag.getLocation() << ": ";
    os << diag << '\n';
    os.flush();
    return mlir::success();
  });

  // Create the module and attach the attributes.
  module = std::make_unique<mlir::ModuleOp>(
      mlir::ModuleOp::create(mlir::UnknownLoc::get(&context)));
  assert(module.get() && "module was not created");
  fir::setTargetTriple(*module.get(), triple);
  fir::setNameUniquer(*module.get(), uniquer);
  fir::setKindMapping(*module.get(), kindMap);
}
