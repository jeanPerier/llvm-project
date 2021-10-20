//===-- IterationSpace.cpp ------------------------------------------------===//
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

#include "IterationSpace.h"
#include "flang/Evaluate/expression.h"
#include "flang/Lower/AbstractConverter.h"
#include "flang/Lower/IntrinsicCall.h"
#include "flang/Lower/VectorSubscripts.h"
#include "flang/Lower/Support/Utils.h"
#include "flang/Lower/Todo.h"

namespace {

/// This class can recover the base array in an expression that contains
/// explicit iteration space symbols. Most of the class can be ignored as it is
/// boilerplate Ev::Expr traversal.
class ArrayBaseFinder {
public:
  using RT = bool;

  ArrayBaseFinder(llvm::ArrayRef<Fortran::lower::FrontEndSymbol> syms)
      : controlVars(syms.begin(), syms.end()) {}

  template <typename T>
  void operator()(const T &x) {
    (void)find(x);
  }

  /// Get the list of bases.
  llvm::ArrayRef<Fortran::lower::ExplicitIterSpace::ArrayBases>
  getBases() const {
    return bases;
  }

private:
  // First, the cases that are of interest.
  RT find(const Fortran::semantics::Symbol &symbol) {
    if (symbol.Rank() > 0) {
      bases.push_back(&symbol);
      return true;
    }
    return {};
  }
  RT find(const Fortran::evaluate::Component &x) {
    auto found = find(x.base());
    if (!found && x.base().Rank() == 0 && x.Rank() > 0) {
      bases.push_back(&x);
      return true;
    }
    return found;
  }
  RT find(const Fortran::evaluate::ArrayRef &x) {
    if (x.base().IsSymbol()) {
      if (x.Rank() > 0 || intersection(x.subscript())) {
        bases.push_back(&x);
        return true;
      }
      return {};
    }
    auto found = find(x.base());
    if (!found && ((x.base().Rank() == 0 && x.Rank() > 0) ||
                   intersection(x.subscript()))) {
      bases.push_back(&x);
      return true;
    }
    return found;
  }
  RT find(const Fortran::evaluate::DataRef &x) { return find(x.u); }
  RT find(const Fortran::evaluate::CoarrayRef &x) {
    assert(false && "coarray reference");
    return {};
  }

  template <typename A>
  bool intersection(const A &subscripts) {
    return Fortran::lower::symbolsIntersectSubscripts(controlVars, subscripts);
  }

  // The rest is traversal boilerplate and can be ignored.
  RT find(const Fortran::evaluate::Substring &x) { return find(x.parent()); }
  template <typename A>
  RT find(const Fortran::semantics::SymbolRef x) {
    return find(*x);
  }
  RT find(const Fortran::evaluate::NamedEntity &x) {
    if (x.IsSymbol())
      return find(x.GetFirstSymbol());
    return find(x.GetComponent());
  }

  template <typename A, bool C>
  RT find(const Fortran::common::Indirection<A, C> &x) {
    return find(x.value());
  }
  template <typename A>
  RT find(const std::unique_ptr<A> &x) {
    return find(x.get());
  }
  template <typename A>
  RT find(const std::shared_ptr<A> &x) {
    return find(x.get());
  }
  template <typename A>
  RT find(const A *x) {
    if (x)
      return find(*x);
    return {};
  }
  template <typename A>
  RT find(const std::optional<A> &x) {
    if (x)
      return find(*x);
    return {};
  }
  template <typename... A>
  RT find(const std::variant<A...> &u) {
    return std::visit([&](const auto &v) { return find(v); }, u);
  }
  template <typename A>
  RT find(const std::vector<A> &x) {
    for (auto &v : x)
      (void)find(v);
    return {};
  }
  RT find(const Fortran::evaluate::BOZLiteralConstant &) { return {}; }
  RT find(const Fortran::evaluate::NullPointer &) { return {}; }
  template <typename T>
  RT find(const Fortran::evaluate::Constant<T> &x) {
    return {};
  }
  RT find(const Fortran::evaluate::StaticDataObject &) { return {}; }
  RT find(const Fortran::evaluate::ImpliedDoIndex &) { return {}; }
  RT find(const Fortran::evaluate::BaseObject &x) {
    (void)find(x.u);
    return {};
  }
  RT find(const Fortran::evaluate::TypeParamInquiry &) { return {}; }
  RT find(const Fortran::evaluate::ComplexPart &x) { return {}; }
  template <typename T>
  RT find(const Fortran::evaluate::Designator<T> &x) {
    return find(x.u);
  }
  template <typename T>
  RT find(const Fortran::evaluate::Variable<T> &x) {
    return find(x.u);
  }
  RT find(const Fortran::evaluate::DescriptorInquiry &) { return {}; }
  RT find(const Fortran::evaluate::SpecificIntrinsic &) { return {}; }
  RT find(const Fortran::evaluate::ProcedureDesignator &x) { return {}; }
  RT find(const Fortran::evaluate::ProcedureRef &x) {
    (void)find(x.proc());
    if (x.IsElemental())
      (void)find(x.arguments());
    return {};
  }
  RT find(const Fortran::evaluate::ActualArgument &x) {
    if (const auto *sym = x.GetAssumedTypeDummy())
      (void)find(*sym);
    else
      (void)find(x.UnwrapExpr());
    return {};
  }
  template <typename T>
  RT find(const Fortran::evaluate::FunctionRef<T> &x) {
    (void)find(static_cast<const Fortran::evaluate::ProcedureRef &>(x));
    return {};
  }
  template <typename T>
  RT find(const Fortran::evaluate::ArrayConstructorValue<T> &) {
    return {};
  }
  template <typename T>
  RT find(const Fortran::evaluate::ArrayConstructorValues<T> &) {
    return {};
  }
  template <typename T>
  RT find(const Fortran::evaluate::ImpliedDo<T> &) {
    return {};
  }
  RT find(const Fortran::semantics::ParamValue &) { return {}; }
  RT find(const Fortran::semantics::DerivedTypeSpec &) { return {}; }
  RT find(const Fortran::evaluate::StructureConstructor &) { return {}; }
  template <typename D, typename R, typename O>
  RT find(const Fortran::evaluate::Operation<D, R, O> &op) {
    (void)find(op.left());
    return false;
  }
  template <typename D, typename R, typename LO, typename RO>
  RT find(const Fortran::evaluate::Operation<D, R, LO, RO> &op) {
    (void)find(op.left());
    (void)find(op.right());
    return false;
  }
  RT find(const Fortran::evaluate::Relational<Fortran::evaluate::SomeType> &x) {
    (void)find(x.u);
    return {};
  }
  template <typename T>
  RT find(const Fortran::evaluate::Expr<T> &x) {
    (void)find(x.u);
    return {};
  }

  llvm::SmallVector<Fortran::lower::ExplicitIterSpace::ArrayBases> bases;
  llvm::SmallVector<Fortran::lower::FrontEndSymbol> controlVars;
};

} // namespace

void Fortran::lower::ExplicitIterSpace::leave() {
  ccLoopNest.pop_back();
  --forallContextOpen;
  conditionalCleanup();
}

void Fortran::lower::ExplicitIterSpace::addSymbol(
    Fortran::lower::FrontEndSymbol sym) {
  assert(!symbolStack.empty());
  symbolStack.back().push_back(sym);
}

void Fortran::lower::ExplicitIterSpace::exprBase(Fortran::lower::FrontEndExpr x,
                                                 bool lhs) {
  ArrayBaseFinder finder(collectAllSymbols());
  finder(*x);
  auto bases = finder.getBases();
  if (rhsBases.empty())
    endAssign();
  if (lhs) {
    if (bases.empty()) {
      lhsBases.push_back(llvm::None);
      return;
    }
    assert(bases.size() == 1);
    lhsBases.push_back(bases.front());
    return;
  }
  rhsBases.back().append(bases.begin(), bases.end());
}

void Fortran::lower::ExplicitIterSpace::endAssign() { rhsBases.emplace_back(); }

void Fortran::lower::ExplicitIterSpace::pushLevel() {
  symbolStack.push_back(llvm::SmallVector<Fortran::lower::FrontEndSymbol>{});
}

void Fortran::lower::ExplicitIterSpace::popLevel() { symbolStack.pop_back(); }

void Fortran::lower::ExplicitIterSpace::conditionalCleanup() {
  if (forallContextOpen == 0) {
    // Exiting the outermost FORALL context.
    // Cleanup any residual mask buffers.
    outermostContext().finalize();
    outermostContext().reset();
    // Clear and reset all the cached information.
    symbolStack.clear();
    lhsBases.clear();
    rhsBases.clear();
    loadBindings.clear();
    ccLoopNest.clear();
    stmtCtx.reset();
    innerArgs.clear();
    outerLoop = llvm::None;
    clearLoops();
    counter = 0;
  }
}

void Fortran::lower::ExplicitIterSpace::bindLoad(
    const Fortran::lower::ExplicitIterSpace::ArrayBases &base,
    fir::ArrayLoadOp load) {
  std::visit(
      [&](const auto *p) {
        using T = std::remove_cv_t<std::remove_pointer_t<decltype(p)>>;
        void *vp = static_cast<void *>(const_cast<T *>(p));
        if constexpr (!std::is_same_v<T, Fortran::semantics::Symbol>) {
          // Se::Symbol* are inalienably shared; never assert on them.
          assert(!loadBindings.count(vp) && "duplicate key");
        }
        loadBindings.try_emplace(vp, load);
      },
      base);
}

llvm::Optional<size_t>
Fortran::lower::ExplicitIterSpace::findArgPosition(fir::ArrayLoadOp load) {
  if (lhsBases[counter].hasValue()) {
    [[maybe_unused]] auto optPos = std::visit(
        [&](const auto *x) -> llvm::Optional<size_t> {
          using T = std::remove_cv_t<std::remove_pointer_t<decltype(x)>>;
          void *vp = static_cast<void *>(const_cast<T *>(x));
          auto ld = loadBindings.find(vp);
          if (ld != loadBindings.end() && ld->second == load)
            return {0};
          return llvm::None;
        },
        lhsBases[counter].getValue());
    assert(optPos.hasValue() && "load does not correspond to lhs");
    return {0};
  }
  return llvm::None;
}

llvm::SmallVector<Fortran::lower::FrontEndSymbol>
Fortran::lower::ExplicitIterSpace::collectAllSymbols() {
  llvm::SmallVector<Fortran::lower::FrontEndSymbol> result;
  for (auto vec : symbolStack)
    result.append(vec.begin(), vec.end());
  return result;
}

llvm::raw_ostream &
Fortran::lower::operator<<(llvm::raw_ostream &s,
                           const Fortran::lower::ImplicitIterSpace &e) {
  for (auto &xs : e.getMasks()) {
    s << "{ ";
    for (auto &x : xs)
      x->AsFortran(s << '(') << "), ";
    s << "}\n";
  }
  return s;
}

llvm::raw_ostream &
Fortran::lower::operator<<(llvm::raw_ostream &s,
                           const Fortran::lower::ExplicitIterSpace &e) {
  auto dump = [&](const auto &u) {
    std::visit(Fortran::common::visitors{
                   [&](const Fortran::semantics::Symbol *y) {
                     s << "  " << *y << '\n';
                   },
                   [&](const Fortran::evaluate::ArrayRef *y) {
                     s << "  ";
                     if (y->base().IsSymbol())
                       s << y->base().GetFirstSymbol();
                     else
                       s << y->base().GetComponent().GetLastSymbol();
                     s << '\n';
                   },
                   [&](const Fortran::evaluate::Component *y) {
                     s << "  " << y->GetLastSymbol() << '\n';
                   }},
               u);
  };
  s << "LHS bases:\n";
  for (auto &u : e.lhsBases)
    if (u.hasValue())
      dump(u.getValue());
  s << "RHS bases:\n";
  for (auto &bases : e.rhsBases) {
    for (auto &u : bases)
      dump(u);
    s << '\n';
  }
  return s;
}

void Fortran::lower::ImplicitIterSpace::dump() const {
  llvm::errs() << *this << '\n';
}

void Fortran::lower::ExplicitIterSpace::dump() const {
  llvm::errs() << *this << '\n';
}

static unsigned getForallDepth(const Fortran::lower::ExplicitIterSpace& iterSpace) {
  unsigned depth = 0;
  for (auto nest : iterSpace.getLoopStack())
    depth += nest.size();
  return depth;
}

Fortran::lower::ForallTemp::ForallTemp(Fortran::lower::AbstractConverter& converter, mlir::Location loc, Fortran::lower::ExplicitIterSpace& iterSpace, const Fortran::lower::SomeExpr& rhs, Fortran::lower::StatementContext& stmtCtx) {
  /// Dumb forall temp size = max(i0) * ... * max(in) * max(rhs size).
  /// General case: evaluating the max requires running the loops.
  /// TODO: improve for constant/partial or easy cases by removing the looping.
  auto& builder = converter.getFirOpBuilder();
  auto idxTy = builder.getIndexType();

  rhsType = converter.genType(rhs);
  
  // TODO: use ragged alloc model for dynamically sized rhs.
  // i.e, allocate fir.box<type>
  if (fir::hasDynamicSize(rhsType))
    TODO(loc, "Dynamically sized temp in forall");
  auto one = builder.createIntegerConstant(loc, idxTy, 1);
  auto zero = builder.createIntegerConstant(loc, idxTy, 0);

  /// Get ForallDepth can only be called after loop nest generation.
  iterSpace.genLoopNest();
  auto depth = getForallDepth(iterSpace);
  auto inLoops  = builder.saveInsertionPoint();
  builder.setInsertionPoint(iterSpace.getOuterLoop());

  llvm::SmallVector<mlir::Value> maxIndices;
  for (decltype(depth) i = 0; i < depth; ++i) {
    auto maxIdx = builder.createTemporary(loc, idxTy);
    builder.create<fir::StoreOp>(loc, zero, maxIdx);
    maxIndices.push_back(maxIdx);
  }

  builder.restoreInsertionPoint(inLoops);
  {
    auto indices = Fortran::lower::getZeroBasedInductions(builder, loc, iterSpace);
    for (auto [maxVar, idx] : llvm::zip(maxIndices, indices)) {
      auto oldMax = builder.create<fir::LoadOp>(loc, maxVar);
      auto newMax = Fortran::lower::genMax(builder, loc, {oldMax, idx});
      builder.create<fir::StoreOp>(loc, newMax, maxVar);
    }
  }
  Fortran::lower::createArrayMergeStores(converter, iterSpace);

  for (auto maxVar : maxIndices) {
    auto max = builder.create<fir::LoadOp>(loc, maxVar);
    auto maxExtent = builder.create<mlir::AddIOp>(loc, max, one);
    forallShape.push_back(maxExtent);
  }
  assert(!forallShape.empty() && "at least one forall loop expected");
  mlir::Value size = forallShape[0];
  for (auto extent: llvm::ArrayRef<mlir::Value>(forallShape).drop_front())
    size = builder.create<mlir::MulIOp>(loc, size, extent);

  fir::SequenceType::Shape tempTypeShape;
  if (auto seqTy = rhsType.dyn_cast<fir::SequenceType>()) {
    auto rhsShape = seqTy.getShape();
    tempTypeShape.append(rhsShape.begin(), rhsShape.end());
  }
  tempTypeShape.push_back(fir::SequenceType::getUnknownExtent());
  auto eleTy = fir::unwrapSequenceType(rhsType);
  auto tempType = fir::SequenceType::get(tempTypeShape, eleTy);

  mlir::Value tmp = builder.create<fir::AllocMemOp>(loc, tempType, ".forall.temp", llvm::None, mlir::ValueRange{size});
  temp = tmp;
  auto *bldr = &converter.getFirOpBuilder();
  stmtCtx.attachCleanup(
      [bldr, loc, tmp]() { bldr->create<fir::FreeMemOp>(loc, tmp); });

  if (auto seqTy = rhsType.dyn_cast<fir::SequenceType>())
    for (auto dim : seqTy.getShape())
      extents.push_back(builder.createIntegerConstant(loc, idxTy, dim));

  if (auto charTy = fir::unwrapSequenceType(rhsType).dyn_cast<fir::CharacterType>())
    lengths.push_back(builder.createIntegerConstant(loc, idxTy, charTy.getLen()));
}

Fortran::lower::Variable Fortran::lower::ForallTemp::getOrCreateTempAt(fir::FirOpBuilder& builder, mlir::Location loc, llvm::ArrayRef<mlir::Value> zeroBasedForallInduction, llvm::ArrayRef<mlir::Value> rhsExtents, llvm::ArrayRef<mlir::Value> typeParams) {
  // TODO: allocate here and store metadata in ragged array model.
  return getTempAt(builder, loc, zeroBasedForallInduction);
}

Fortran::lower::Variable Fortran::lower::ForallTemp::getTempAt(fir::FirOpBuilder& builder, mlir::Location loc, llvm::ArrayRef<mlir::Value> zeroBasedForallInduction) {
  // Get temp storage for iteration.
  assert(!zeroBasedForallInduction.empty() && "at least one forall loop expected");
  mlir::ArrayRef<mlir::Value> idxs = zeroBasedForallInduction;
  mlir::ArrayRef<mlir::Value> exts = forallShape;
  auto at = idxs.back();
  auto stride = exts.back();
  auto idxsIt = llvm::reverse(idxs.drop_front());
  auto extsIt = llvm::reverse(exts.drop_front());
  for (auto [idx, ext] : llvm::zip(idxsIt, extsIt)) {
    auto dim = builder.create<mlir::MulIOp>(loc, idx, stride); 
    at = builder.create<mlir::AddIOp>(loc, at, dim);
    stride = builder.create<mlir::MulIOp>(loc, stride, ext); 
  }
  auto zero = builder.createIntegerConstant(loc, builder.getIndexType(), 0);
  auto refTy = builder.getRefType(fir::unwrapSequenceType(rhsType));
  mlir::Value memref = builder.create<fir::CoordinateOp>(loc, refTy, temp, mlir::ValueRange{zero, at});
  memref = builder.createConvert(loc, builder.getRefType(rhsType), memref);

  // Build fir::ExtendedValue from it. 
  auto temp = [&]() -> fir::ExtendedValue {
    if (fir::isa_char(fir::unwrapSequenceType(rhsType))) {
      assert(lengths.size() == 1 && "length must have been lowered");
      if (rhsType.isa<fir::SequenceType>())
        return fir::CharArrayBoxValue{memref, lengths[0], extents};
      return fir::CharBoxValue{memref, lengths[0]};
    }
    if (rhsType.isa<fir::SequenceType>())
      return fir::ArrayBoxValue{memref, extents};
    return memref;
  }();
  return Fortran::lower::Variable(std::move(temp));
}

llvm::SmallVector<mlir::Value> Fortran::lower::getZeroBasedInductions(fir::FirOpBuilder& builder, mlir::Location loc, const Fortran::lower::ExplicitIterSpace& explicitIterSpace) {
  llvm::SmallVector<mlir::Value> indices;
  for (auto loopNest : explicitIterSpace.getLoopStack())
    for (auto loop : loopNest) {
      auto diff = builder.create<mlir::SubIOp>(loc, loop.getInductionVar(), loop.lowerBound());
      auto idx =
        builder.create<mlir::SignedDivIOp>(loc, diff, loop.step()); 
      indices.push_back(idx);
    }
  return indices;   
}
