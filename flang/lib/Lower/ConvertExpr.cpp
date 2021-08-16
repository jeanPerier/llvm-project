//===-- ConvertExpr.cpp ---------------------------------------------------===//
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

#include "flang/Lower/ConvertExpr.h"
#include "BuiltinModules.h"
#include "ConvertVariable.h"
#include "IterationSpace.h"
#include "RTBuilder.h"
#include "StatementContext.h"
#include "flang/Common/default-kinds.h"
#include "flang/Common/unwrap.h"
#include "flang/Evaluate/fold.h"
#include "flang/Evaluate/real.h"
#include "flang/Evaluate/traverse.h"
#include "flang/Lower/Allocatable.h"
#include "flang/Lower/Bridge.h"
#include "flang/Lower/CallInterface.h"
#include "flang/Lower/CharacterRuntime.h"
#include "flang/Lower/Coarray.h"
#include "flang/Lower/ConvertType.h"
#include "flang/Lower/IntrinsicCall.h"
#include "flang/Lower/Mangler.h"
#include "flang/Lower/Runtime.h"
#include "flang/Lower/Support/Utils.h"
#include "flang/Lower/Todo.h"
#include "flang/Optimizer/Builder/Character.h"
#include "flang/Optimizer/Builder/Complex.h"
#include "flang/Optimizer/Dialect/FIRAttr.h"
#include "flang/Optimizer/Dialect/FIRDialect.h"
#include "flang/Optimizer/Dialect/FIROpsSupport.h"
#include "flang/Optimizer/Support/FatalError.h"
#include "flang/Optimizer/Transforms/Factory.h"
#include "flang/Semantics/expression.h"
#include "flang/Semantics/symbol.h"
#include "flang/Semantics/tools.h"
#include "flang/Semantics/type.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/StandardOps/IR/Ops.h"
#include "llvm/ADT/APFloat.h"
#include "llvm/ADT/TypeSwitch.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/raw_ostream.h"
#include <algorithm>

#define DEBUG_TYPE "flang-lower-expr"

//===----------------------------------------------------------------------===//
// The composition and structure of Fortran::evaluate::Expr is defined in
// the various header files in include/flang/Evaluate. You are referred
// there for more information on these data structures. Generally speaking,
// these data structures are a strongly typed family of abstract data types
// that, composed as trees, describe the syntax of Fortran expressions.
//
// This part of the bridge can traverse these tree structures and lower them
// to the correct FIR representation in SSA form.
//===----------------------------------------------------------------------===//

static llvm::cl::opt<bool> generateArrayCoordinate(
    "gen-array-coor",
    llvm::cl::desc("in lowering create ArrayCoorOp instead of CoordinateOp"),
    llvm::cl::init(false));

// The default attempts to balance a modest allocation size with expected user
// input to minimize bounds checks and reallocations during dynamic array
// construction. Some user codes may have very large array constructors for
// which the default can be increased.
static llvm::cl::opt<unsigned> clInitialBufferSize(
    "array-constructor-initial-buffer-size",
    llvm::cl::desc(
        "set the incremental array construction buffer size (default=32)"),
    llvm::cl::init(32u));

/// The various semantics of a program constituent (or a part thereof) as it may
/// appear in an expression.
///
/// Given the following Fortran declarations.
/// ```fortran
///   REAL :: v1, v2, v3
///   REAL, POINTER :: vp1
///   REAL :: a1(c), a2(c)
///   REAL ELEMENTAL FUNCTION f1(arg) ! array -> array
///   FUNCTION f2(arg)                ! array -> array
///   vp1 => v3       ! 1
///   v1 = v2 * vp1   ! 2
///   a1 = a1 + a2    ! 3
///   a1 = f1(a2)     ! 4
///   a1 = f2(a2)     ! 5
/// ```
///
/// In line 1, `vp1` is a BoxAddr to copy a box value into. The box value is
/// constructed from the DataAddr of `v3`.
/// In line 2, `v1` is a DataAddr to copy a value into. The value is constructed
/// from the DataValue of `v2` and `vp1`. DataValue is implicitly a double
/// dereference in the `vp1` case.
/// In line 3, `a1` and `a2` on the rhs are RefTransparent. The `a1` on the lhs
/// is CopyInCopyOut as `a1` is replaced elementally by the additions.
/// In line 4, `a2` can be RefTransparent, ByValueArg, RefOpaque, or BoxAddr if
/// `arg` is declared as C-like pass-by-value, VALUE, INTENT(?), or ALLOCATABLE/
/// POINTER, respectively. `a1` on the lhs is CopyInCopyOut.
///  In line 5, `a2` may be DataAddr or BoxAddr assuming f2 is transformational.
///  `a1` on the lhs is again CopyInCopyOut.
enum class ConstituentSemantics {
  // Scalars : let `v` be the location in memory of a variable with value `x`
  DataValue, // refers to the value `x`
  DataAddr,  // refers to the address `v`
  BoxValue,  // refers to a box value containing `v`
  BoxAddr,   // refers to the address of a box value containing `v`

  // Arrays : let `a` be the location in memory of a sequence of value `[xs]`
  RefTransparent, // refers to the value `[xs]`
  ByValueArg, // refers to an ephemeral address `t` containing a value `x` which
              // is the i-th value in `[xs]` (15.5.2.3.p7 note 2)
  CopyInCopyOut, // refers to the merge of `[xs]` with another value `[ys]`,
                 // which is written into `a`.
  ProjectedCopyInCopyOut, // similar to CopyInCopyOut but `a` may be a transient
                          // projection (rather than a whole array).
  RefOpaque // refers to the address `a+i`, the i-th element of `a`
};

/// Convert parser's INTEGER relational operators to MLIR.  TODO: using
/// unordered, but we may want to cons ordered in certain situation.
static mlir::CmpIPredicate
translateRelational(Fortran::common::RelationalOperator rop) {
  switch (rop) {
  case Fortran::common::RelationalOperator::LT:
    return mlir::CmpIPredicate::slt;
  case Fortran::common::RelationalOperator::LE:
    return mlir::CmpIPredicate::sle;
  case Fortran::common::RelationalOperator::EQ:
    return mlir::CmpIPredicate::eq;
  case Fortran::common::RelationalOperator::NE:
    return mlir::CmpIPredicate::ne;
  case Fortran::common::RelationalOperator::GT:
    return mlir::CmpIPredicate::sgt;
  case Fortran::common::RelationalOperator::GE:
    return mlir::CmpIPredicate::sge;
  }
  llvm_unreachable("unhandled INTEGER relational operator");
}

/// Convert parser's REAL relational operators to MLIR.
/// The choice of order (O prefix) vs unorder (U prefix) follows Fortran 2018
/// requirements in the IEEE context (table 17.1 of F2018). This choice is
/// also applied in other contexts because it is easier and in line with
/// other Fortran compilers.
/// FIXME: The signaling/quiet aspect of the table 17.1 requirement is not
/// fully enforced. FIR and LLVM `fcmp` instructions do not give any guarantee
/// whether the comparison will signal or not in case of quiet NaN argument.
static mlir::CmpFPredicate
translateFloatRelational(Fortran::common::RelationalOperator rop) {
  switch (rop) {
  case Fortran::common::RelationalOperator::LT:
    return mlir::CmpFPredicate::OLT;
  case Fortran::common::RelationalOperator::LE:
    return mlir::CmpFPredicate::OLE;
  case Fortran::common::RelationalOperator::EQ:
    return mlir::CmpFPredicate::OEQ;
  case Fortran::common::RelationalOperator::NE:
    return mlir::CmpFPredicate::UNE;
  case Fortran::common::RelationalOperator::GT:
    return mlir::CmpFPredicate::OGT;
  case Fortran::common::RelationalOperator::GE:
    return mlir::CmpFPredicate::OGE;
  }
  llvm_unreachable("unhandled REAL relational operator");
}

/// Clone subexpression and wrap it as a generic `Fortran::evaluate::Expr`.
template <typename A>
Fortran::evaluate::Expr<Fortran::evaluate::SomeType> toEvExpr(const A &x) {
  return Fortran::evaluate::AsGenericExpr(Fortran::common::Clone(x));
}

/// Lower `opt` (from front-end shape analysis) to MLIR. If `opt` is `nullopt`
/// then issue an error.
static mlir::Value
convertOptExtentExpr(Fortran::lower::AbstractConverter &converter,
                     Fortran::lower::StatementContext &stmtCtx,
                     const Fortran::evaluate::MaybeExtentExpr &opt) {
  auto loc = converter.getCurrentLocation();
  if (!opt.has_value())
    fir::emitFatalError(loc, "shape analysis failed to return an expression");
  auto e = toEvExpr(*opt);
  return fir::getBase(converter.genExprValue(&e, stmtCtx, loc));
}

/// Does this expr designate an allocatable or pointer entity ?
static bool isAllocatableOrPointer(const Fortran::lower::SomeExpr &expr) {
  const auto *sym =
      Fortran::evaluate::UnwrapWholeSymbolOrComponentDataRef(expr);
  return sym && Fortran::semantics::IsAllocatableOrPointer(*sym);
}

/// Given the address of an array element and the ExtendedValue describing the
/// array, returns the ExtendedValue describing the array element. The purpose
/// is to propagate the length parameters of the array to the element.
/// This can be used for elements of `array` or `array(i:j:k)`. If \p element
/// belongs to an array section `array%x` whose base is \p array,
/// arraySectionElementToExtendedValue must be used instead.
static fir::ExtendedValue
arrayElementToExtendedValue(fir::FirOpBuilder &builder, mlir::Location loc,
                            const fir::ExtendedValue &array,
                            mlir::Value element) {
  return array.match(
      [&](const fir::CharBoxValue &cb) -> fir::ExtendedValue {
        return cb.clone(element);
      },
      [&](const fir::CharArrayBoxValue &bv) -> fir::ExtendedValue {
        return bv.cloneElement(element);
      },
      [&](const fir::BoxValue &box) -> fir::ExtendedValue {
        if (box.isCharacter()) {
          auto len = fir::factory::readCharLen(builder, loc, box);
          return fir::CharBoxValue{element, len};
        }
        if (box.isDerivedWithLengthParameters())
          TODO(loc, "get length parameters from derived type BoxValue");
        return element;
      },
      [&](const auto &) -> fir::ExtendedValue { return element; });
}

/// Build the ExtendedValue for \p element that is an element of an array or
/// array section with \p array base (`array` or `array(i:j:k)%x%y`).
/// If it is an array section, \p slice must be provided and be a fir::SliceOp
/// that describes the section.
static fir::ExtendedValue arraySectionElementToExtendedValue(
    fir::FirOpBuilder &builder, mlir::Location loc,
    const fir::ExtendedValue &array, mlir::Value element, mlir::Value slice) {
  if (!slice)
    return arrayElementToExtendedValue(builder, loc, array, element);
  auto sliceOp = mlir::dyn_cast_or_null<fir::SliceOp>(slice.getDefiningOp());
  assert(sliceOp && "slice must be a sliceOp");
  if (sliceOp.fields().empty())
    return arrayElementToExtendedValue(builder, loc, array, element);
  // For F95, using componentToExtendedValue will work, but when PDTs are
  // lowered. It will be required to go down the slice to propagate the length
  // parameters.
  return fir::factory::componentToExtendedValue(builder, loc, array, element);
}

namespace {

/// Lowering of Fortran::evaluate::Expr<T> expressions
class ScalarExprLowering {
public:
  using ExtValue = fir::ExtendedValue;

  explicit ScalarExprLowering(mlir::Location loc,
                              Fortran::lower::AbstractConverter &converter,
                              Fortran::lower::SymMap &symMap,
                              Fortran::lower::StatementContext &stmtCtx,
                              bool initializer = false)
      : location{loc}, converter{converter},
        builder{converter.getFirOpBuilder()}, stmtCtx{stmtCtx}, symMap{symMap},
        inInitializer{initializer} {}

  ExtValue genExtAddr(const Fortran::lower::SomeExpr &expr) {
    return gen(expr);
  }

  /// Lower `expr` to be passed as a fir.box argument. Do not create a temp
  /// for the expr if it is a variable that can be described as a fir.box.
  ExtValue genBoxArg(const Fortran::lower::SomeExpr &expr) {
    bool saveUseBoxArg = useBoxArg;
    useBoxArg = true;
    auto result = gen(expr);
    useBoxArg = saveUseBoxArg;
    return result;
  }

  ExtValue genExtValue(const Fortran::lower::SomeExpr &expr) {
    return genval(expr);
  }

  /// Lower an expression that is a pointer or an allocatable to a
  /// MutableBoxValue.
  fir::MutableBoxValue
  genMutableBoxValue(const Fortran::lower::SomeExpr &expr) {
    // Pointers and allocatables can only be:
    //    - a simple designator "x"
    //    - a component designator "a%b(i,j)%x"
    //    - a function reference "foo()"
    //    - result of NULL() or NULL(MOLD) intrinsic.
    //    NULL() requires some context to be lowered, so it is not handled
    //    here and must be lowered according to the context where it appears.
    auto exv = std::visit(
        [&](const auto &x) { return genMutableBoxValueImpl(x); }, expr.u);
    auto *mutableBox = exv.getBoxOf<fir::MutableBoxValue>();
    if (!mutableBox)
      fir::emitFatalError(getLoc(), "expr was not lowered to MutableBoxValue");
    return *mutableBox;
  }

  template <typename T>
  ExtValue genMutableBoxValueImpl(const T &) {
    // NULL() case should not be handled here.
    fir::emitFatalError(getLoc(), "NULL() must be lowered in its context");
  }

  template <typename T>
  ExtValue
  genMutableBoxValueImpl(const Fortran::evaluate::FunctionRef<T> &funRef) {
    return genRawProcedureRef(funRef, converter.genType(toEvExpr(funRef)));
  }

  template <typename T>
  ExtValue
  genMutableBoxValueImpl(const Fortran::evaluate::Designator<T> &designator) {
    return std::visit(
        Fortran::common::visitors{
            [&](const Fortran::evaluate::SymbolRef &sym) -> ExtValue {
              return symMap.lookupSymbol(*sym).toExtendedValue();
            },
            [&](const Fortran::evaluate::Component &comp) -> ExtValue {
              return genComponent(comp);
            },
            [&](const auto &) -> ExtValue {
              fir::emitFatalError(getLoc(),
                                  "not an allocatable or pointer designator");
            }},
        designator.u);
  }

  template <typename T>
  ExtValue genMutableBoxValueImpl(const Fortran::evaluate::Expr<T> &expr) {
    return std::visit([&](const auto &x) { return genMutableBoxValueImpl(x); },
                      expr.u);
  }

  mlir::Location getLoc() { return location; }

  template <typename A>
  mlir::Value genunbox(const A &expr) {
    auto e = genval(expr);
    if (auto *r = e.getUnboxed())
      return *r;
    fir::emitFatalError(getLoc(), "unboxed expression expected");
  }

  /// Generate an integral constant of `value`
  template <int KIND>
  mlir::Value genIntegerConstant(mlir::MLIRContext *context,
                                 std::int64_t value) {
    auto type = converter.genType(Fortran::common::TypeCategory::Integer, KIND);
    return builder.createIntegerConstant(getLoc(), type, value);
  }

  /// Generate a logical/boolean constant of `value`
  mlir::Value genBoolConstant(bool value) {
    return builder.createBool(getLoc(), value);
  }

  /// Generate a real constant with a value `value`.
  template <int KIND>
  mlir::Value genRealConstant(mlir::MLIRContext *context,
                              const llvm::APFloat &value) {
    auto fltTy = Fortran::lower::convertReal(context, KIND);
    return builder.createRealConstant(getLoc(), fltTy, value);
  }

  mlir::Type getSomeKindInteger() { return builder.getIndexType(); }

  mlir::FuncOp getFunction(llvm::StringRef name, mlir::FunctionType funTy) {
    if (auto func = builder.getNamedFunction(name))
      return func;
    return builder.createFunction(getLoc(), name, funTy);
  }

  template <typename OpTy>
  mlir::Value createCompareOp(mlir::CmpIPredicate pred, const ExtValue &left,
                              const ExtValue &right) {
    if (auto *lhs = left.getUnboxed())
      if (auto *rhs = right.getUnboxed())
        return builder.create<OpTy>(getLoc(), pred, *lhs, *rhs);
    fir::emitFatalError(getLoc(), "array compare should be handled in genarr");
  }
  template <typename OpTy, typename A>
  mlir::Value createCompareOp(const A &ex, mlir::CmpIPredicate pred) {
    return createCompareOp<OpTy>(pred, genval(ex.left()), genval(ex.right()));
  }

  template <typename OpTy>
  mlir::Value createFltCmpOp(mlir::CmpFPredicate pred, const ExtValue &left,
                             const ExtValue &right) {
    if (auto *lhs = left.getUnboxed())
      if (auto *rhs = right.getUnboxed())
        return builder.create<OpTy>(getLoc(), pred, *lhs, *rhs);
    fir::emitFatalError(getLoc(), "array compare should be handled in genarr");
  }
  template <typename OpTy, typename A>
  mlir::Value createFltCmpOp(const A &ex, mlir::CmpFPredicate pred) {
    return createFltCmpOp<OpTy>(pred, genval(ex.left()), genval(ex.right()));
  }

  /// Create a call to the runtime to compare two CHARACTER values.
  /// Precondition: This assumes that the two values have `fir.boxchar` type.
  mlir::Value createCharCompare(mlir::CmpIPredicate pred, const ExtValue &left,
                                const ExtValue &right) {
    return Fortran::lower::genCharCompare(builder, getLoc(), pred, left, right);
  }

  template <typename A>
  mlir::Value createCharCompare(const A &ex, mlir::CmpIPredicate pred) {
    return createCharCompare(pred, genval(ex.left()), genval(ex.right()));
  }

  /// Returns a reference to a symbol or its box/boxChar descriptor if it has
  /// one.
  ExtValue gen(Fortran::semantics::SymbolRef sym) {
    if (auto val = symMap.lookupSymbol(sym))
      return val.match(
          [&](const Fortran::lower::SymbolBox::PointerOrAllocatable &boxAddr) {
            return fir::factory::genMutableBoxRead(builder, getLoc(), boxAddr);
          },
          [&val](auto &) { return val.toExtendedValue(); });
    LLVM_DEBUG(llvm::dbgs()
               << "unknown symbol: " << sym << "\nmap: " << symMap << '\n');
    fir::emitFatalError(getLoc(), "symbol is not mapped to any IR value");
  }

  /// Generate a load of a value from an address.
  ExtValue genLoad(const ExtValue &addr) {
    auto loc = getLoc();
    return addr.match(
        [](const fir::CharBoxValue &box) -> ExtValue { return box; },
        [&](const fir::UnboxedValue &v) -> ExtValue {
          return builder.create<fir::LoadOp>(loc, fir::getBase(v));
        },
        [&](const auto &v) -> ExtValue {
          TODO(getLoc(), "loading array or descriptor");
        });
  }

  ExtValue genval(Fortran::semantics::SymbolRef sym) {
    auto loc = getLoc();
    auto var = gen(sym);
    if (auto *s = var.getUnboxed())
      if (fir::isReferenceLike(s->getType())) {
        // A function with multiple entry points returning different types
        // tags all result variables with one of the largest types to allow
        // them to share the same storage.  A reference to a result variable
        // of one of the other types requires conversion to the actual type.
        auto addr = *s;
        if (Fortran::semantics::IsFunctionResult(sym)) {
          auto resultType = converter.genType(*sym);
          if (addr.getType() != resultType)
            addr = builder.createConvert(loc, builder.getRefType(resultType),
                                         addr);
        }
        return genLoad(addr);
      }
    return var;
  }

  ExtValue genval(const Fortran::evaluate::BOZLiteralConstant &) {
    TODO(getLoc(), "BOZ");
  }

  /// Return indirection to function designated in ProcedureDesignator.
  /// The type of the function indirection is not guaranteed to match the one
  /// of the ProcedureDesignator due to Fortran implicit typing rules.
  ExtValue genval(const Fortran::evaluate::ProcedureDesignator &proc) {
    if (const auto *intrinsic = proc.GetSpecificIntrinsic()) {
      auto signature = Fortran::lower::translateSignature(proc, converter);
      // Intrinsic lowering is based on the generic name, so retrieve it here in
      // case it is different from the specific name. The type of the specific
      // intrinsic is retained in the signature.
      auto genericName =
          converter.getFoldingContext().intrinsics().GetGenericIntrinsicName(
              intrinsic->name);
      auto symbolRefAttr =
          Fortran::lower::getUnrestrictedIntrinsicSymbolRefAttr(
              builder, getLoc(), genericName, signature);
      mlir::Value funcPtr =
          builder.create<fir::AddrOfOp>(getLoc(), signature, symbolRefAttr);
      return funcPtr;
    }
    const auto *symbol = proc.GetSymbol();
    assert(symbol && "expected symbol in ProcedureDesignator");
    if (Fortran::semantics::IsDummy(*symbol)) {
      auto val = symMap.lookupSymbol(*symbol);
      assert(val && "Dummy procedure not in symbol map");
      return val.getAddr();
    }
    auto name = converter.mangleName(*symbol);
    auto func = Fortran::lower::getOrDeclareFunction(name, proc, converter);
    mlir::Value funcPtr = builder.create<fir::AddrOfOp>(
        getLoc(), func.getType(), builder.getSymbolRefAttr(name));
    return funcPtr;
  }
  ExtValue genval(const Fortran::evaluate::NullPointer &) {
    return builder.createNullConstant(getLoc());
  }

  static bool
  isDerivedTypeWithLengthParameters(const Fortran::semantics::Symbol &sym) {
    if (const auto *declTy = sym.GetType())
      if (const auto *derived = declTy->AsDerived())
        return Fortran::semantics::CountLenParameters(*derived) > 0;
    return false;
  }

  static bool isBuiltinCPtr(const Fortran::semantics::Symbol &sym) {
    if (const auto *declType = sym.GetType())
      if (const auto *derived = declType->AsDerived())
        return Fortran::semantics::IsIsoCType(derived);
    return false;
  }

  /// Lower structure constructor without a temporary. This can be used in
  /// fir::GloablOp, and assumes that the structure component is a constant.
  ExtValue genStructComponentInInitializer(
      const Fortran::evaluate::StructureConstructor &ctor) {
    auto loc = getLoc();
    auto ty = translateSomeExprToFIRType(converter, toEvExpr(ctor));
    auto recTy = ty.cast<fir::RecordType>();
    auto fieldTy = fir::FieldType::get(ty.getContext());
    mlir::Value res = builder.create<fir::UndefOp>(loc, recTy);

    for (auto [sym, expr] : ctor.values()) {
      // Parent components need more work because they do not appear in the
      // fir.rec type.
      if (sym->test(Fortran::semantics::Symbol::Flag::ParentComp))
        TODO(loc, "parent component in structure constructor");

      auto name = toStringRef(sym->name());
      auto componentTy = recTy.getType(name);
      // FIXME: type parameters must come from the derived-type-spec
      mlir::Value field = builder.create<fir::FieldIndexOp>(
          loc, fieldTy, name, ty,
          /*typeParams=*/mlir::ValueRange{} /*TODO*/);

      if (Fortran::semantics::IsAllocatable(sym))
        TODO(loc, "allocatable component in structure constructor");

      if (Fortran::semantics::IsPointer(sym)) {
        auto initialTarget = Fortran::lower::genInitialDataTarget(
            converter, loc, componentTy, expr.value());
        res = builder.create<fir::InsertValueOp>(loc, recTy, res, initialTarget,
                                                 field);
        continue;
      }

      if (isDerivedTypeWithLengthParameters(sym))
        TODO(loc, "component with length parameters in structure constructor");

      if (isBuiltinCPtr(sym)) {
        // Builtin c_ptr and c_funptr have special handling because initial
        // value are handled for them as an extension.
        auto addr = Fortran::lower::genExtAddrInInitializer(converter, loc,
                                                            expr.value());
        auto baseAddr = fir::getBase(addr);
        auto undef = builder.create<fir::UndefOp>(loc, componentTy);
        auto cPtrRecTy = componentTy.dyn_cast<fir::RecordType>();
        assert(cPtrRecTy && "c_ptr and c_funptr must be derived types");
        llvm::StringRef addrFieldName = Fortran::lower::builtin::cptrFieldName;
        auto addrFieldTy = cPtrRecTy.getType(addrFieldName);
        mlir::Value addrField = builder.create<fir::FieldIndexOp>(
            loc, fieldTy, addrFieldName, componentTy,
            /*typeParams=*/mlir::ValueRange{});
        auto castAddr = builder.createConvert(loc, addrFieldTy, baseAddr);
        auto val = builder.create<fir::InsertValueOp>(loc, componentTy, undef,
                                                      castAddr, addrField);
        res = builder.create<fir::InsertValueOp>(loc, recTy, res, val, field);
        continue;
      }

      auto val = fir::getBase(genval(expr.value()));
      assert(!fir::isa_ref_type(val.getType()) && "expecting a constant value");
      auto castVal = builder.createConvert(loc, componentTy, val);
      res = builder.create<fir::InsertValueOp>(loc, recTy, res, castVal, field);
    }
    return res;
  }

  /// A structure constructor is lowered two ways. In an initializer context,
  /// the entire structure must be constant, so the aggregate value is
  /// constructed inline. This allows it to be the body of a GlobalOp.
  /// Otherwise, the structure constructor is in an expression. In that case, a
  /// temporary object is constructed in the stack frame of the procedure.
  ExtValue genval(const Fortran::evaluate::StructureConstructor &ctor) {
    if (inInitializer)
      return genStructComponentInInitializer(ctor);
    auto loc = getLoc();
    auto ty = translateSomeExprToFIRType(converter, toEvExpr(ctor));
    auto recTy = ty.cast<fir::RecordType>();
    auto fieldTy = fir::FieldType::get(ty.getContext());
    mlir::Value res = builder.createTemporary(loc, recTy);

    for (auto value : ctor.values()) {
      const auto &sym = value.first;
      auto &expr = value.second;
      // Parent components need more work because they do not appear in the
      // fir.rec type.
      if (sym->test(Fortran::semantics::Symbol::Flag::ParentComp))
        TODO(loc, "parent component in structure constructor");

      if (isDerivedTypeWithLengthParameters(sym))
        TODO(loc, "component with length parameters in structure constructor");

      auto name = toStringRef(sym->name());
      // FIXME: type parameters must come from the derived-type-spec
      mlir::Value field = builder.create<fir::FieldIndexOp>(
          loc, fieldTy, name, ty,
          /*typeParams=*/mlir::ValueRange{} /*TODO*/);
      auto coorTy = builder.getRefType(recTy.getType(name));
      auto coor = builder.create<fir::CoordinateOp>(loc, coorTy,
                                                    fir::getBase(res), field);
      auto to = fir::factory::componentToExtendedValue(builder, loc, res, coor);
      to.match(
          [&](const fir::UnboxedValue &toPtr) {
            // FIXME: if toPtr is a derived type, it is incorrect after F95 to
            // simply load/store derived type since they may have allocatable
            // components that require deep-copy or may have defined assignment
            // procedures.
            auto val = fir::getBase(genval(expr.value()));
            auto cast = builder.createConvert(
                loc, fir::dyn_cast_ptrEleTy(toPtr.getType()), val);
            builder.create<fir::StoreOp>(loc, cast, toPtr);
          },
          [&](const fir::CharBoxValue &) {
            fir::factory::CharacterExprHelper{builder, loc}.createAssign(
                to, genval(expr.value()));
          },
          [&](const fir::ArrayBoxValue &) {
            Fortran::lower::createSomeArrayAssignment(
                converter, to, expr.value(), symMap, stmtCtx);
          },
          [&](const fir::CharArrayBoxValue &) {
            Fortran::lower::createSomeArrayAssignment(
                converter, to, expr.value(), symMap, stmtCtx);
          },
          [&](const fir::BoxValue &toBox) {
            fir::emitFatalError(loc, "derived type components must not be "
                                     "represented by fir::BoxValue");
          },
          [&](const fir::MutableBoxValue &toBox) {
            if (toBox.isPointer()) {
              Fortran::lower::associateMutableBox(
                  converter, loc, toBox, expr.value(), /*lbounds=*/llvm::None,
                  stmtCtx);
              return;
            }
            // For allocatable components, a deep copy is needed.
            TODO(loc, "allocatable components in derived type assignment");
          },
          [&](const fir::ProcBoxValue &toBox) {
            TODO(loc, "procedure pointer component in derived type assignment");
          });
    }
    return builder.create<fir::LoadOp>(loc, res);
  }

  /// Lowering of an <i>ac-do-variable</i>, which is not a Symbol.
  ExtValue genval(const Fortran::evaluate::ImpliedDoIndex &var) {
    return converter.impliedDoBinding(toStringRef(var.name));
  }

  ExtValue genval(const Fortran::evaluate::DescriptorInquiry &desc) {
    auto exv = desc.base().IsSymbol() ? gen(desc.base().GetLastSymbol())
                                      : gen(desc.base().GetComponent());
    auto idxTy = builder.getIndexType();
    auto loc = getLoc();
    auto castResult = [&](mlir::Value v) {
      using ResTy = Fortran::evaluate::DescriptorInquiry::Result;
      return builder.createConvert(
          loc, converter.genType(ResTy::category, ResTy::kind), v);
    };
    switch (desc.field()) {
    case Fortran::evaluate::DescriptorInquiry::Field::Len:
      return castResult(fir::factory::readCharLen(builder, loc, exv));
    case Fortran::evaluate::DescriptorInquiry::Field::LowerBound:
      return castResult(fir::factory::readLowerBound(
          builder, loc, exv, desc.dimension(),
          builder.createIntegerConstant(loc, idxTy, 1)));
    case Fortran::evaluate::DescriptorInquiry::Field::Extent:
      return castResult(
          fir::factory::readExtent(builder, loc, exv, desc.dimension()));
    case Fortran::evaluate::DescriptorInquiry::Field::Rank:
      TODO(loc, "rank inquiry on assumed rank");
    case Fortran::evaluate::DescriptorInquiry::Field::Stride:
      // So far the front end does not generate this inquiry.
      TODO(loc, "Stride inquiry");
    }
    llvm_unreachable("unknown descriptor inquiry");
  }

  ExtValue genval(const Fortran::evaluate::TypeParamInquiry &) {
    TODO(getLoc(), "type parameter inquiry");
  }

  mlir::Value extractComplexPart(mlir::Value cplx, bool isImagPart) {
    return fir::factory::ComplexExprHelper{builder, getLoc()}
        .extractComplexPart(cplx, isImagPart);
  }

  template <int KIND>
  ExtValue genval(const Fortran::evaluate::ComplexComponent<KIND> &part) {
    return extractComplexPart(genunbox(part.left()), part.isImaginaryPart);
  }

  template <int KIND>
  ExtValue genval(const Fortran::evaluate::Negate<Fortran::evaluate::Type<
                      Fortran::common::TypeCategory::Integer, KIND>> &op) {
    auto input = genunbox(op.left());
    // Like LLVM, integer negation is the binary op "0 - value"
    auto zero = genIntegerConstant<KIND>(builder.getContext(), 0);
    return builder.create<mlir::SubIOp>(getLoc(), zero, input);
  }
  template <int KIND>
  ExtValue genval(const Fortran::evaluate::Negate<Fortran::evaluate::Type<
                      Fortran::common::TypeCategory::Real, KIND>> &op) {
    return builder.create<mlir::NegFOp>(getLoc(), genunbox(op.left()));
  }
  template <int KIND>
  ExtValue genval(const Fortran::evaluate::Negate<Fortran::evaluate::Type<
                      Fortran::common::TypeCategory::Complex, KIND>> &op) {
    return builder.create<fir::NegcOp>(getLoc(), genunbox(op.left()));
  }

  template <typename OpTy>
  mlir::Value createBinaryOp(const ExtValue &left, const ExtValue &right) {
    assert(fir::isUnboxedValue(left) && fir::isUnboxedValue(right));
    auto lhs = fir::getBase(left);
    auto rhs = fir::getBase(right);
    assert(lhs.getType() == rhs.getType() && "types must be the same");
    return builder.create<OpTy>(getLoc(), lhs, rhs);
  }

  template <typename OpTy, typename A>
  mlir::Value createBinaryOp(const A &ex) {
    return createBinaryOp<OpTy>(genval(ex.left()), genval(ex.right()));
  }

#undef GENBIN
#define GENBIN(GenBinEvOp, GenBinTyCat, GenBinFirOp)                           \
  template <int KIND>                                                          \
  ExtValue genval(const Fortran::evaluate::GenBinEvOp<Fortran::evaluate::Type< \
                      Fortran::common::TypeCategory::GenBinTyCat, KIND>> &x) { \
    return createBinaryOp<GenBinFirOp>(x);                                     \
  }

  GENBIN(Add, Integer, mlir::AddIOp)
  GENBIN(Add, Real, mlir::AddFOp)
  GENBIN(Add, Complex, fir::AddcOp)
  GENBIN(Subtract, Integer, mlir::SubIOp)
  GENBIN(Subtract, Real, mlir::SubFOp)
  GENBIN(Subtract, Complex, fir::SubcOp)
  GENBIN(Multiply, Integer, mlir::MulIOp)
  GENBIN(Multiply, Real, mlir::MulFOp)
  GENBIN(Multiply, Complex, fir::MulcOp)
  GENBIN(Divide, Integer, mlir::SignedDivIOp)
  GENBIN(Divide, Real, mlir::DivFOp)
  GENBIN(Divide, Complex, fir::DivcOp)

  template <Fortran::common::TypeCategory TC, int KIND>
  ExtValue genval(
      const Fortran::evaluate::Power<Fortran::evaluate::Type<TC, KIND>> &op) {
    auto ty = converter.genType(TC, KIND);
    auto lhs = genunbox(op.left());
    auto rhs = genunbox(op.right());
    return Fortran::lower::genPow(builder, getLoc(), ty, lhs, rhs);
  }

  template <Fortran::common::TypeCategory TC, int KIND>
  ExtValue genval(
      const Fortran::evaluate::RealToIntPower<Fortran::evaluate::Type<TC, KIND>>
          &op) {
    auto ty = converter.genType(TC, KIND);
    auto lhs = genunbox(op.left());
    auto rhs = genunbox(op.right());
    return Fortran::lower::genPow(builder, getLoc(), ty, lhs, rhs);
  }

  template <int KIND>
  ExtValue genval(const Fortran::evaluate::ComplexConstructor<KIND> &op) {
    return fir::factory::ComplexExprHelper{builder, getLoc()}.createComplex(
        KIND, genunbox(op.left()), genunbox(op.right()));
  }

  template <int KIND>
  ExtValue genval(const Fortran::evaluate::Concat<KIND> &op) {
    auto lhs = genval(op.left());
    auto rhs = genval(op.right());
    auto *lhsChar = lhs.getCharBox();
    auto *rhsChar = rhs.getCharBox();
    if (lhsChar && rhsChar)
      return fir::factory::CharacterExprHelper{builder, getLoc()}
          .createConcatenate(*lhsChar, *rhsChar);
    TODO(getLoc(), "character array concatenate");
  }

  /// MIN and MAX operations
  template <Fortran::common::TypeCategory TC, int KIND>
  ExtValue
  genval(const Fortran::evaluate::Extremum<Fortran::evaluate::Type<TC, KIND>>
             &op) {
    auto lhs = genunbox(op.left());
    auto rhs = genunbox(op.right());
    switch (op.ordering) {
    case Fortran::evaluate::Ordering::Greater:
      return Fortran::lower::genMax(builder, getLoc(),
                                    llvm::ArrayRef<mlir::Value>{lhs, rhs});
    case Fortran::evaluate::Ordering::Less:
      return Fortran::lower::genMin(builder, getLoc(),
                                    llvm::ArrayRef<mlir::Value>{lhs, rhs});
    case Fortran::evaluate::Ordering::Equal:
      llvm_unreachable("Equal is not a valid ordering in this context");
    }
    llvm_unreachable("unknown ordering");
  }

  template <int KIND>
  ExtValue genval(const Fortran::evaluate::SetLength<KIND> &) {
    TODO(getLoc(), "evaluate::SetLength lowering");
  }

  template <int KIND>
  ExtValue genval(const Fortran::evaluate::Relational<Fortran::evaluate::Type<
                      Fortran::common::TypeCategory::Integer, KIND>> &op) {
    return createCompareOp<mlir::CmpIOp>(op, translateRelational(op.opr));
  }
  template <int KIND>
  ExtValue genval(const Fortran::evaluate::Relational<Fortran::evaluate::Type<
                      Fortran::common::TypeCategory::Real, KIND>> &op) {
    return createFltCmpOp<mlir::CmpFOp>(op, translateFloatRelational(op.opr));
  }
  template <int KIND>
  ExtValue genval(const Fortran::evaluate::Relational<Fortran::evaluate::Type<
                      Fortran::common::TypeCategory::Complex, KIND>> &op) {
    return createFltCmpOp<fir::CmpcOp>(op, translateFloatRelational(op.opr));
  }
  template <int KIND>
  ExtValue genval(const Fortran::evaluate::Relational<Fortran::evaluate::Type<
                      Fortran::common::TypeCategory::Character, KIND>> &op) {
    return createCharCompare(op, translateRelational(op.opr));
  }

  ExtValue
  genval(const Fortran::evaluate::Relational<Fortran::evaluate::SomeType> &op) {
    return std::visit([&](const auto &x) { return genval(x); }, op.u);
  }

  template <Fortran::common::TypeCategory TC1, int KIND,
            Fortran::common::TypeCategory TC2>
  ExtValue
  genval(const Fortran::evaluate::Convert<Fortran::evaluate::Type<TC1, KIND>,
                                          TC2> &convert) {
    auto ty = converter.genType(TC1, KIND);
    auto operand = genunbox(convert.left());
    return builder.convertWithSemantics(getLoc(), ty, operand);
  }

  template <typename A>
  ExtValue genval(const Fortran::evaluate::Parentheses<A> &op) {
    auto input = genval(op.left());
    auto base = fir::getBase(input);
    mlir::Value newBase =
        builder.create<fir::NoReassocOp>(getLoc(), base.getType(), base);
    return fir::substBase(input, newBase);
  }

  template <int KIND>
  ExtValue genval(const Fortran::evaluate::Not<KIND> &op) {
    auto logical = genunbox(op.left());
    auto one = genBoolConstant(true);
    auto val = builder.createConvert(getLoc(), builder.getI1Type(), logical);
    return builder.create<mlir::XOrOp>(getLoc(), val, one);
  }

  template <int KIND>
  ExtValue genval(const Fortran::evaluate::LogicalOperation<KIND> &op) {
    auto i1Type = builder.getI1Type();
    auto slhs = genunbox(op.left());
    auto srhs = genunbox(op.right());
    auto lhs = builder.createConvert(getLoc(), i1Type, slhs);
    auto rhs = builder.createConvert(getLoc(), i1Type, srhs);
    switch (op.logicalOperator) {
    case Fortran::evaluate::LogicalOperator::And:
      return createBinaryOp<mlir::AndOp>(lhs, rhs);
    case Fortran::evaluate::LogicalOperator::Or:
      return createBinaryOp<mlir::OrOp>(lhs, rhs);
    case Fortran::evaluate::LogicalOperator::Eqv:
      return createCompareOp<mlir::CmpIOp>(mlir::CmpIPredicate::eq, lhs, rhs);
    case Fortran::evaluate::LogicalOperator::Neqv:
      return createCompareOp<mlir::CmpIOp>(mlir::CmpIPredicate::ne, lhs, rhs);
    case Fortran::evaluate::LogicalOperator::Not:
      // lib/evaluate expression for .NOT. is Fortran::evaluate::Not<KIND>.
      llvm_unreachable(".NOT. is not a binary operator");
    }
    llvm_unreachable("unhandled logical operation");
  }

  /// Convert a scalar literal constant to IR.
  template <Fortran::common::TypeCategory TC, int KIND>
  ExtValue genScalarLit(
      const Fortran::evaluate::Scalar<Fortran::evaluate::Type<TC, KIND>>
          &value) {
    if constexpr (TC == Fortran::common::TypeCategory::Integer) {
      return genIntegerConstant<KIND>(builder.getContext(), value.ToInt64());
    } else if constexpr (TC == Fortran::common::TypeCategory::Logical) {
      return genBoolConstant(value.IsTrue());
    } else if constexpr (TC == Fortran::common::TypeCategory::Real) {
      std::string str = value.DumpHexadecimal();
      if constexpr (KIND == 2) {
        llvm::APFloat floatVal{llvm::APFloatBase::IEEEhalf(), str};
        return genRealConstant<KIND>(builder.getContext(), floatVal);
      } else if constexpr (KIND == 3) {
        llvm::APFloat floatVal{llvm::APFloatBase::BFloat(), str};
        return genRealConstant<KIND>(builder.getContext(), floatVal);
      } else if constexpr (KIND == 4) {
        llvm::APFloat floatVal{llvm::APFloatBase::IEEEsingle(), str};
        return genRealConstant<KIND>(builder.getContext(), floatVal);
      } else if constexpr (KIND == 10) {
        llvm::APFloat floatVal{llvm::APFloatBase::x87DoubleExtended(), str};
        return genRealConstant<KIND>(builder.getContext(), floatVal);
      } else if constexpr (KIND == 16) {
        llvm::APFloat floatVal{llvm::APFloatBase::IEEEquad(), str};
        return genRealConstant<KIND>(builder.getContext(), floatVal);
      } else {
        // convert everything else to double
        llvm::APFloat floatVal{llvm::APFloatBase::IEEEdouble(), str};
        return genRealConstant<KIND>(builder.getContext(), floatVal);
      }
    } else if constexpr (TC == Fortran::common::TypeCategory::Complex) {
      using TR =
          Fortran::evaluate::Type<Fortran::common::TypeCategory::Real, KIND>;
      Fortran::evaluate::ComplexConstructor<KIND> ctor(
          Fortran::evaluate::Expr<TR>{
              Fortran::evaluate::Constant<TR>{value.REAL()}},
          Fortran::evaluate::Expr<TR>{
              Fortran::evaluate::Constant<TR>{value.AIMAG()}});
      return genunbox(ctor);
    } else /*constexpr*/ {
      llvm_unreachable("unhandled constant");
    }
  }
  /// Convert a ascii scalar literal CHARACTER to IR. (specialization)
  ExtValue
  genAsciiScalarLit(const Fortran::evaluate::Scalar<Fortran::evaluate::Type<
                        Fortran::common::TypeCategory::Character, 1>> &value,
                    int64_t len) {
    assert(value.size() == static_cast<std::uint64_t>(len));
    // Outline character constant in ro data if it is not in an initializer.
    if (!inInitializer)
      return fir::factory::createStringLiteral(builder, getLoc(), value);
    // When in an initializer context, construct the literal op itself and do
    // not construct another constant object in rodata.
    auto stringLit = builder.createStringLitOp(getLoc(), value);
    auto lenp = builder.createIntegerConstant(
        getLoc(), builder.getCharacterLengthType(), len);
    return fir::CharBoxValue{stringLit.getResult(), lenp};
  }
  /// Convert a non ascii scalar literal CHARACTER to IR. (specialization)
  template <int KIND>
  ExtValue
  genScalarLit(const Fortran::evaluate::Scalar<Fortran::evaluate::Type<
                   Fortran::common::TypeCategory::Character, KIND>> &value,
               int64_t len) {
    using ET = typename std::decay_t<decltype(value)>::value_type;
    if constexpr (KIND == 1) {
      return genAsciiScalarLit(value, len);
    }
    auto type = fir::CharacterType::get(builder.getContext(), KIND, len);
    auto consLit = [&]() -> fir::StringLitOp {
      auto context = builder.getContext();
      std::int64_t size = static_cast<std::int64_t>(value.size());
      auto shape = mlir::VectorType::get(
          llvm::ArrayRef<std::int64_t>{size},
          mlir::IntegerType::get(builder.getContext(), sizeof(ET) * 8));
      auto strAttr = mlir::DenseElementsAttr::get(
          shape, llvm::ArrayRef<ET>{value.data(), value.size()});
      auto valTag = mlir::Identifier::get(fir::StringLitOp::value(), context);
      mlir::NamedAttribute dataAttr(valTag, strAttr);
      auto sizeTag = mlir::Identifier::get(fir::StringLitOp::size(), context);
      mlir::NamedAttribute sizeAttr(sizeTag, builder.getI64IntegerAttr(len));
      llvm::SmallVector<mlir::NamedAttribute> attrs{dataAttr, sizeAttr};
      return builder.create<fir::StringLitOp>(
          getLoc(), llvm::ArrayRef<mlir::Type>{type}, llvm::None, attrs);
    };

    auto lenp = builder.createIntegerConstant(
        getLoc(), builder.getCharacterLengthType(), len);
    // When in an initializer context, construct the literal op itself and do
    // not construct another constant object in rodata.
    if (inInitializer)
      return fir::CharBoxValue{consLit().getResult(), lenp};

    // Otherwise, the string is in a plain old expression so "outline" the value
    // by hashconsing it to a constant literal object.

    // FIXME: For wider char types, lowering ought to use an array of i16 or
    // i32. But for now, lowering just fakes that the string value is a range of
    // i8 to get it past the C++ compiler.
    std::string globalName =
        fir::factory::uniqueCGIdent("cl", (const char *)value.c_str());
    auto global = builder.getNamedGlobal(globalName);
    if (!global)
      global = builder.createGlobalConstant(
          getLoc(), type, globalName,
          [&](fir::FirOpBuilder &builder) {
            auto str = consLit();
            builder.create<fir::HasValueOp>(getLoc(), str);
          },
          builder.createLinkOnceLinkage());
    auto addr = builder.create<fir::AddrOfOp>(getLoc(), global.resultType(),
                                              global.getSymbol());
    return fir::CharBoxValue{addr, lenp};
  }

  template <Fortran::common::TypeCategory TC, int KIND>
  ExtValue genArrayLit(
      const Fortran::evaluate::Constant<Fortran::evaluate::Type<TC, KIND>>
          &con) {
    llvm::SmallVector<mlir::Value> lbounds;
    llvm::SmallVector<mlir::Value> extents;
    auto idxTy = builder.getIndexType();
    for (auto [lb, extent] : llvm::zip(con.lbounds(), con.shape())) {
      lbounds.push_back(builder.createIntegerConstant(getLoc(), idxTy, lb - 1));
      extents.push_back(builder.createIntegerConstant(getLoc(), idxTy, extent));
    }
    if constexpr (TC == Fortran::common::TypeCategory::Character) {
      fir::SequenceType::Shape shape;
      shape.append(con.shape().begin(), con.shape().end());
      auto chTy = converter.genType(TC, KIND, {con.LEN()});
      auto arrayTy = fir::SequenceType::get(shape, chTy);
      mlir::Value array = builder.create<fir::UndefOp>(getLoc(), arrayTy);
      Fortran::evaluate::ConstantSubscripts subscripts = con.lbounds();
      do {
        auto charVal =
            fir::getBase(genScalarLit<KIND>(con.At(subscripts), con.LEN()));
        llvm::SmallVector<mlir::Value> idx;
        for (auto [dim, lb] : llvm::zip(subscripts, con.lbounds()))
          idx.push_back(
              builder.createIntegerConstant(getLoc(), idxTy, dim - lb));
        array = builder.create<fir::InsertValueOp>(getLoc(), arrayTy, array,
                                                   charVal, idx);
      } while (con.IncrementSubscripts(subscripts));
      auto len = builder.createIntegerConstant(getLoc(), idxTy, con.LEN());
      return fir::CharArrayBoxValue{array, len, extents, lbounds};
    } else {
      // Convert Ev::ConstantSubs to SequenceType::Shape
      fir::SequenceType::Shape shape(con.shape().begin(), con.shape().end());
      auto eleTy = converter.genType(TC, KIND);
      auto arrayTy = fir::SequenceType::get(shape, eleTy);
      mlir::Value array = builder.create<fir::UndefOp>(getLoc(), arrayTy);
      Fortran::evaluate::ConstantSubscripts subscripts = con.lbounds();
      bool foundRange = false;
      mlir::Value rangeValue;
      llvm::SmallVector<mlir::Value> rangeStartIdx;
      Fortran::evaluate::ConstantSubscripts rangeStartSubscripts;
      uint64_t elemsInRange = 0;
      const uint64_t minRangeSize = 2;

      do {
        auto constant =
            fir::getBase(genScalarLit<TC, KIND>(con.At(subscripts)));
        auto createIndexes = [&](Fortran::evaluate::ConstantSubscripts subs) {
          llvm::SmallVector<mlir::Value> idx;
          for (auto [dim, lb] : llvm::zip(subs, con.lbounds()))
            // Add normalized upper bound index to idx.
            idx.push_back(
                builder.createIntegerConstant(getLoc(), idxTy, dim - lb));

          return idx;
        };

        auto idx = createIndexes(subscripts);
        auto insVal = builder.createConvert(getLoc(), eleTy, constant);
        auto nextSubs = subscripts;

        // Check to see if the next value is the same as the current value
        bool nextIsSame = con.IncrementSubscripts(nextSubs) &&
                          con.At(subscripts) == con.At(nextSubs);
        bool newRange = (nextIsSame != foundRange) && !foundRange;
        bool endOfRange = (nextIsSame != foundRange) && foundRange;
        bool continueRange = nextIsSame && foundRange;

        if (newRange) {
          // Mark the start of the range
          rangeStartIdx = idx;
          rangeStartSubscripts = subscripts;
          rangeValue = insVal;
          foundRange = true;
          elemsInRange = 1;
        } else if (endOfRange) {
          ++elemsInRange;
          if (elemsInRange >= minRangeSize) {
            // Zip together the upper and lower bounds of the range for each
            // index in the form [lb0, up0, lb1, up1, ... , lbn, upn] to pass
            // to the InserOnEangeOp.
            llvm::SmallVector<mlir::Value> zippedRange;
            for (size_t i = 0; i < idx.size(); ++i) {
              zippedRange.push_back(rangeStartIdx[i]);
              zippedRange.push_back(idx[i]);
            }
            array = builder.create<fir::InsertOnRangeOp>(
                getLoc(), arrayTy, array, rangeValue, zippedRange);
          } else {
            while (true) {
              idx = createIndexes(rangeStartSubscripts);
              array = builder.create<fir::InsertValueOp>(
                  getLoc(), arrayTy, array, rangeValue, idx);
              if (rangeStartSubscripts == subscripts)
                break;
              con.IncrementSubscripts(rangeStartSubscripts);
            }
          }
          foundRange = false;
        } else if (continueRange) {
          // Loop until the end of the range is found.
          ++elemsInRange;
          continue;
        } else /* no range */ {
          // If a range has not been found then insert the current value.
          array = builder.create<fir::InsertValueOp>(getLoc(), arrayTy, array,
                                                     insVal, idx);
        }
      } while (con.IncrementSubscripts(subscripts));
      return fir::ArrayBoxValue{array, extents, lbounds};
    }
  }

  fir::ExtendedValue genArrayLit(
      const Fortran::evaluate::Constant<Fortran::evaluate::SomeDerived> &con) {
    llvm::SmallVector<mlir::Value> lbounds;
    llvm::SmallVector<mlir::Value> extents;
    auto idxTy = builder.getIndexType();
    for (auto [lb, extent] : llvm::zip(con.lbounds(), con.shape())) {
      lbounds.push_back(builder.createIntegerConstant(getLoc(), idxTy, lb - 1));
      extents.push_back(builder.createIntegerConstant(getLoc(), idxTy, extent));
    }
    fir::SequenceType::Shape shape;
    shape.append(con.shape().begin(), con.shape().end());
    auto recTy = converter.genType(con.GetType().GetDerivedTypeSpec());
    auto arrayTy = fir::SequenceType::get(shape, recTy);
    mlir::Value array = builder.create<fir::UndefOp>(getLoc(), arrayTy);
    Fortran::evaluate::ConstantSubscripts subscripts = con.lbounds();
    do {
      auto derivedVal = fir::getBase(genval(con.At(subscripts)));
      llvm::SmallVector<mlir::Value> idx;
      for (auto [dim, lb] : llvm::zip(subscripts, con.lbounds()))
        idx.push_back(builder.createIntegerConstant(getLoc(), idxTy, dim - lb));
      array = builder.create<fir::InsertValueOp>(getLoc(), arrayTy, array,
                                                 derivedVal, idx);
    } while (con.IncrementSubscripts(subscripts));
    return fir::ArrayBoxValue{array, extents, lbounds};
  }

  template <Fortran::common::TypeCategory TC, int KIND>
  ExtValue
  genval(const Fortran::evaluate::Constant<Fortran::evaluate::Type<TC, KIND>>
             &con) {
    if (con.Rank() > 0)
      return genArrayLit(con);
    auto opt = con.GetScalarValue();
    assert(opt.has_value() && "constant has no value");
    if constexpr (TC == Fortran::common::TypeCategory::Character) {
      return genScalarLit<KIND>(opt.value(), con.LEN());
    } else {
      return genScalarLit<TC, KIND>(opt.value());
    }
  }
  fir::ExtendedValue genval(
      const Fortran::evaluate::Constant<Fortran::evaluate::SomeDerived> &con) {
    if (con.Rank() > 0)
      return genArrayLit(con);
    if (auto ctor = con.GetScalarValue())
      return genval(ctor.value());
    fir::emitFatalError(getLoc(),
                        "constant of derived type has no constructor");
  }

  template <typename A>
  ExtValue genval(const Fortran::evaluate::ArrayConstructor<A> &) {
    fir::emitFatalError(getLoc(),
                        "array constructor: lowering should not reach here");
  }

  ExtValue gen(const Fortran::evaluate::ComplexPart &x) {
    auto loc = getLoc();
    auto idxTy = builder.getIndexType();
    auto exv = gen(x.complex());
    auto base = fir::getBase(exv);
    fir::factory::ComplexExprHelper helper{builder, loc};
    auto eleTy =
        helper.getComplexPartType(fir::dyn_cast_ptrEleTy(base.getType()));
    auto offset = builder.createIntegerConstant(
        loc, idxTy,
        x.part() == Fortran::evaluate::ComplexPart::Part::RE ? 0 : 1);
    mlir::Value result = builder.create<fir::CoordinateOp>(
        loc, builder.getRefType(eleTy), base, mlir::ValueRange{offset});
    return {result};
  }
  ExtValue genval(const Fortran::evaluate::ComplexPart &x) {
    return genLoad(gen(x));
  }

  /// Reference to a substring.
  ExtValue gen(const Fortran::evaluate::Substring &s) {
    // Get base string
    auto baseString = std::visit(
        Fortran::common::visitors{
            [&](const Fortran::evaluate::DataRef &x) { return gen(x); },
            [&](const Fortran::evaluate::StaticDataObject::Pointer &p)
                -> ExtValue {
              if (auto str = p->AsString())
                return fir::factory::createStringLiteral(builder, getLoc(),
                                                         *str);
              // TODO: convert StaticDataObject to Constant<T> and use normal
              // constant path. Beware that StaticDataObject data() takes into
              // account build machine endianness.
              TODO(getLoc(),
                   "StaticDataObject::Pointer substring with kind > 1");
            },
        },
        s.parent());
    llvm::SmallVector<mlir::Value> bounds;
    auto lower = genunbox(s.lower());
    bounds.push_back(lower);
    if (auto upperBound = s.upper()) {
      auto upper = genunbox(*upperBound);
      bounds.push_back(upper);
    }
    fir::factory::CharacterExprHelper charHelper{builder, getLoc()};
    return baseString.match(
        [&](const fir::CharBoxValue &x) -> ExtValue {
          return charHelper.createSubstring(x, bounds);
        },
        [&](const fir::CharArrayBoxValue &) -> ExtValue {
          fir::emitFatalError(
              getLoc(),
              "array substring should be handled in array expression");
        },
        [&](const auto &) -> ExtValue {
          fir::emitFatalError(getLoc(), "substring base is not a CharBox");
        });
  }

  /// The value of a substring.
  ExtValue genval(const Fortran::evaluate::Substring &ss) {
    // FIXME: why is the value of a substring being lowered the same as the
    // address of a substring?
    return gen(ss);
  }

  ExtValue genval(const Fortran::evaluate::Subscript &subs) {
    if (auto *s = std::get_if<Fortran::evaluate::IndirectSubscriptIntegerExpr>(
            &subs.u)) {
      if (s->value().Rank() > 0)
        fir::emitFatalError(getLoc(), "vector subscript is not scalar");
      return {genval(s->value())};
    }
    fir::emitFatalError(getLoc(), "subscript triple notation is not scalar");
  }
  ExtValue genSubscript(const Fortran::evaluate::Subscript &subs) {
    return genval(subs);
  }

  ExtValue gen(const Fortran::evaluate::DataRef &dref) {
    return std::visit([&](const auto &x) { return gen(x); }, dref.u);
  }
  ExtValue genval(const Fortran::evaluate::DataRef &dref) {
    return std::visit([&](const auto &x) { return genval(x); }, dref.u);
  }

  // Helper function to turn the left-recursive Component structure into a list
  // that does not contain allocatable or pointer components other than the last
  // one.
  // Returns the object used as the base coordinate for the component chain.
  static Fortran::evaluate::DataRef const *
  reverseComponents(const Fortran::evaluate::Component &cmpt,
                    std::list<const Fortran::evaluate::Component *> &list) {
    list.push_front(&cmpt);
    return std::visit(
        Fortran::common::visitors{
            [&](const Fortran::evaluate::Component &x) {
              // Stop the list when a component is an allocatable or pointer
              // because the component cannot be lowered into a single
              // fir.coordinate_of.
              if (Fortran::semantics::IsAllocatableOrPointer(x.GetLastSymbol()))
                return &cmpt.base();
              return reverseComponents(x, list);
            },
            [&](auto &) { return &cmpt.base(); },
        },
        cmpt.base().u);
  }

  // Return the coordinate of the component reference
  ExtValue genComponent(const Fortran::evaluate::Component &cmpt) {
    std::list<const Fortran::evaluate::Component *> list;
    auto *base = reverseComponents(cmpt, list);
    llvm::SmallVector<mlir::Value> coorArgs;
    auto obj = gen(*base);
    auto ty = fir::dyn_cast_ptrOrBoxEleTy(fir::getBase(obj).getType());
    auto loc = getLoc();
    auto fldTy = fir::FieldType::get(&converter.getMLIRContext());
    // FIXME: need to thread the LEN type parameters here.
    for (auto *field : list) {
      auto recTy = ty.cast<fir::RecordType>();
      const auto *sym = &field->GetLastSymbol();
      auto name = toStringRef(sym->name());
      coorArgs.push_back(builder.create<fir::FieldIndexOp>(
          loc, fldTy, name, recTy, fir::getTypeParams(obj)));
      ty = recTy.getType(name);
    }
    ty = builder.getRefType(ty);
    return fir::factory::componentToExtendedValue(
        builder, loc, obj,
        builder.create<fir::CoordinateOp>(loc, ty, fir::getBase(obj),
                                          coorArgs));
  }

  ExtValue gen(const Fortran::evaluate::Component &cmpt) {
    // Components may be pointer or allocatable. In the gen() path, the mutable
    // aspect is lost to simplify handling on the client side. To retain the
    // mutable aspect, genMutableBoxValue should be used.
    return genComponent(cmpt).match(
        [&](const fir::MutableBoxValue &mutableBox) {
          return fir::factory::genMutableBoxRead(builder, getLoc(), mutableBox);
        },
        [](auto &box) -> ExtValue { return box; });
  }

  ExtValue genval(const Fortran::evaluate::Component &cmpt) {
    return genLoad(gen(cmpt));
  }

  // Determine the result type after removing `dims` dimensions from the array
  // type `arrTy`
  mlir::Type genSubType(mlir::Type arrTy, unsigned dims) {
    auto unwrapTy = fir::dyn_cast_ptrOrBoxEleTy(arrTy);
    assert(unwrapTy && "must be a pointer or box type");
    auto seqTy = unwrapTy.cast<fir::SequenceType>();
    auto shape = seqTy.getShape();
    assert(shape.size() > 0 && "removing columns for sequence sans shape");
    assert(dims <= shape.size() && "removing more columns than exist");
    fir::SequenceType::Shape newBnds;
    // follow Fortran semantics and remove columns (from right)
    auto e = shape.size() - dims;
    for (decltype(e) i{0}; i < e; ++i)
      newBnds.push_back(shape[i]);
    if (!newBnds.empty())
      return fir::SequenceType::get(newBnds, seqTy.getEleTy());
    return seqTy.getEleTy();
  }

  // Generate the code for a Bound value.
  ExtValue genval(const Fortran::semantics::Bound &bound) {
    if (bound.isExplicit()) {
      auto sub = bound.GetExplicit();
      if (sub.has_value())
        return genval(*sub);
      return genIntegerConstant<8>(builder.getContext(), 1);
    }
    TODO(getLoc(), "non explicit semantics::Bound lowering");
  }

  static bool isSlice(const Fortran::evaluate::ArrayRef &aref) {
    for (auto &sub : aref.subscript())
      if (std::holds_alternative<Fortran::evaluate::Triplet>(sub.u))
        return true;
    return false;
  }

  /// Lower an ArrayRef to a fir.coordinate_of given its lowered base.
  ExtValue genCoordinateOp(const ExtValue &array,
                           const Fortran::evaluate::ArrayRef &aref) {
    auto loc = getLoc();
    // References to array of rank > 1 with non constant shape that are not
    // fir.box must be collapsed into an offset computation in lowering already.
    // The same is needed with dynamic length character arrays of all ranks.
    auto baseType = fir::dyn_cast_ptrOrBoxEleTy(fir::getBase(array).getType());
    if ((array.rank() > 1 && fir::hasDynamicSize(baseType)) ||
        fir::characterWithDynamicLen(fir::unwrapSequenceType(baseType)))
      if (!array.getBoxOf<fir::BoxValue>())
        return genOffsetAndCoordinateOp(array, aref);
    // Generate a fir.coordinate_of with zero based array indexes.
    llvm::SmallVector<mlir::Value> args;
    for (auto &subsc : llvm::enumerate(aref.subscript())) {
      auto subVal = genSubscript(subsc.value());
      assert(fir::isUnboxedValue(subVal) && "subscript must be simple scalar");
      auto val = fir::getBase(subVal);
      auto ty = val.getType();
      auto lb = getLBound(array, subsc.index(), ty);
      args.push_back(builder.create<mlir::SubIOp>(loc, ty, val, lb));
    }

    auto base = fir::getBase(array);
    auto seqTy =
        fir::dyn_cast_ptrOrBoxEleTy(base.getType()).cast<fir::SequenceType>();
    assert(args.size() == seqTy.getDimension());
    auto ty = builder.getRefType(seqTy.getEleTy());
    auto addr = builder.create<fir::CoordinateOp>(loc, ty, base, args);
    return arrayElementToExtendedValue(builder, loc, array, addr);
  }

  /// Lower an ArrayRef to a fir.coordinate_of using an element offset instead
  /// of array indexes.
  /// This generates offset computation from the indexes and length parameters,
  /// and use the offset to access the element with a fir.coordinate_of. This
  /// must only be used if it is not possible to generate a normal
  /// fir.coordinate_of using array indexes (i.e. when the shape information is
  /// unavailable in the IR).
  ExtValue genOffsetAndCoordinateOp(const ExtValue &array,
                                    const Fortran::evaluate::ArrayRef &aref) {
    auto loc = getLoc();
    auto addr = fir::getBase(array);
    auto arrTy = fir::dyn_cast_ptrEleTy(addr.getType());
    auto eleTy = arrTy.cast<fir::SequenceType>().getEleTy();
    auto seqTy = builder.getRefType(builder.getVarLenSeqTy(eleTy));
    auto refTy = builder.getRefType(eleTy);
    auto base = builder.createConvert(loc, seqTy, addr);
    auto idxTy = builder.getIndexType();
    auto one = builder.createIntegerConstant(loc, idxTy, 1);
    auto zero = builder.createIntegerConstant(loc, idxTy, 0);
    auto getLB = [&](const auto &arr, unsigned dim) -> mlir::Value {
      return arr.getLBounds().empty() ? one : arr.getLBounds()[dim];
    };
    auto genFullDim = [&](const auto &arr, mlir::Value delta) -> mlir::Value {
      mlir::Value total = zero;
      assert(arr.getExtents().size() == aref.subscript().size());
      delta = builder.createConvert(loc, idxTy, delta);
      unsigned dim = 0;
      for (auto [ext, sub] : llvm::zip(arr.getExtents(), aref.subscript())) {
        auto subVal = genSubscript(sub);
        assert(fir::isUnboxedValue(subVal));
        auto val = builder.createConvert(loc, idxTy, fir::getBase(subVal));
        auto lb = builder.createConvert(loc, idxTy, getLB(arr, dim));
        auto diff = builder.create<mlir::SubIOp>(loc, val, lb);
        auto prod = builder.create<mlir::MulIOp>(loc, delta, diff);
        total = builder.create<mlir::AddIOp>(loc, prod, total);
        if (ext)
          delta = builder.create<mlir::MulIOp>(loc, delta, ext);
        ++dim;
      }
      return builder.create<fir::CoordinateOp>(
          loc, refTy, base, llvm::ArrayRef<mlir::Value>{total});
    };
    return array.match(
        [&](const fir::ArrayBoxValue &arr) -> ExtValue {
          // FIXME: this check can be removed when slicing is implemented
          if (isSlice(aref))
            fir::emitFatalError(
                getLoc(),
                "slice should be handled in array expression context");
          return genFullDim(arr, one);
        },
        [&](const fir::CharArrayBoxValue &arr) -> ExtValue {
          auto delta = arr.getLen();
          // If the length is known in the type, fir.coordinate_of will
          // already take the length into account.
          if (fir::factory::CharacterExprHelper::hasConstantLengthInType(arr))
            delta = one;
          return fir::CharBoxValue(genFullDim(arr, delta), arr.getLen());
        },
        [&](const fir::BoxValue &arr) -> ExtValue {
          // CoordinateOp for BoxValue is not generated here. The dimensions
          // must be kept in the fir.coordinate_op so that potential fir.box
          // strides can be applied by codegen.
          fir::emitFatalError(
              loc, "internal: BoxValue in dim-collapsed fir.coordinate_of");
        },
        [&](const auto &) -> ExtValue {
          fir::emitFatalError(loc, "internal: array lowering failed");
        });
  }

  /// Lower an ArrayRef to a fir.array_coor.
  ExtValue genArrayCoorOp(const ExtValue &exv,
                          const Fortran::evaluate::ArrayRef &aref) {
    auto loc = getLoc();
    auto addr = fir::getBase(exv);
    auto arrTy = fir::dyn_cast_ptrOrBoxEleTy(addr.getType());
    auto eleTy = arrTy.cast<fir::SequenceType>().getEleTy();
    auto refTy = builder.getRefType(eleTy);
    auto idxTy = builder.getIndexType();
    llvm::SmallVector<mlir::Value> arrayCoorArgs;
    // The ArrayRef is expected to be scalar here, arrays are handled in array
    // expression lowering. So no vector subscript or triplet is expected here.
    for (const auto &sub : aref.subscript()) {
      auto subVal = genSubscript(sub);
      assert(fir::isUnboxedValue(subVal));
      arrayCoorArgs.push_back(
          builder.createConvert(loc, idxTy, fir::getBase(subVal)));
    }
    auto shape = builder.createShape(loc, exv);
    auto elementAddr = builder.create<fir::ArrayCoorOp>(
        loc, refTy, addr, shape, /*slice=*/mlir::Value{}, arrayCoorArgs,
        fir::getTypeParams(exv));
    return arrayElementToExtendedValue(builder, loc, exv, elementAddr);
  }

  /// Return the coordinate of the array reference.
  ExtValue gen(const Fortran::evaluate::ArrayRef &aref) {
    auto base = aref.base().IsSymbol() ? gen(aref.base().GetFirstSymbol())
                                       : gen(aref.base().GetComponent());
    // Check for command-line override to use array_coor op.
    if (generateArrayCoordinate)
      return genArrayCoorOp(base, aref);
    // Otherwise, use coordinate_of op.
    return genCoordinateOp(base, aref);
  }

  /// Return lower bounds of \p box in dimension \p dim. The returned value
  /// has type \ty.
  mlir::Value getLBound(const ExtValue &box, unsigned dim, mlir::Type ty) {
    assert(box.rank() > 0 && "must be an array");
    auto loc = getLoc();
    auto one = builder.createIntegerConstant(loc, ty, 1);
    auto lb = fir::factory::readLowerBound(builder, loc, box, dim, one);
    return builder.createConvert(loc, ty, lb);
  }

  ExtValue genval(const Fortran::evaluate::ArrayRef &aref) {
    return genLoad(gen(aref));
  }

  ExtValue gen(const Fortran::evaluate::CoarrayRef &coref) {
    return Fortran::lower::CoarrayExprHelper{converter, getLoc(), symMap}
        .genAddr(coref);
  }

  ExtValue genval(const Fortran::evaluate::CoarrayRef &coref) {
    return Fortran::lower::CoarrayExprHelper{converter, getLoc(), symMap}
        .genValue(coref);
  }

  template <typename A>
  ExtValue gen(const Fortran::evaluate::Designator<A> &des) {
    return std::visit([&](const auto &x) { return gen(x); }, des.u);
  }
  template <typename A>
  ExtValue genval(const Fortran::evaluate::Designator<A> &des) {
    return std::visit([&](const auto &x) { return genval(x); }, des.u);
  }

  mlir::Type genType(const Fortran::evaluate::DynamicType &dt) {
    if (dt.category() != Fortran::common::TypeCategory::Derived)
      return converter.genType(dt.category(), dt.kind());
    return converter.genType(dt.GetDerivedTypeSpec());
  }

  /// Apply the function `func` and return a reference to the resultant value.
  /// This is required for lowering expressions such as `f1(f2(v))`.
  template <typename A>
  ExtValue gen(const Fortran::evaluate::FunctionRef<A> &func) {
    if (!func.GetType().has_value())
      mlir::emitError(getLoc(), "internal: a function must have a type");
    auto resTy = genType(*func.GetType());
    auto retVal = genProcedureRef(func, {resTy});
    auto retValBase = fir::getBase(retVal);
    if (fir::conformsWithPassByRef(retValBase.getType()))
      return retVal;
    auto mem = builder.create<fir::AllocaOp>(getLoc(), retValBase.getType());
    builder.create<fir::StoreOp>(getLoc(), retValBase, mem);
    return fir::substBase(retVal, mem.getResult());
  }

  /// Helper to lower intrinsic arguments for inquiry intrinsic.
  ExtValue
  lowerIntrinsicArgumentAsInquired(const Fortran::lower::SomeExpr &expr) {
    if (isAllocatableOrPointer(expr))
      return genMutableBoxValue(expr);
    return gen(expr);
  }

  /// Generate a call to an intrinsic function.
  ExtValue
  genIntrinsicRef(const Fortran::evaluate::ProcedureRef &procRef,
                  const Fortran::evaluate::SpecificIntrinsic &intrinsic,
                  llvm::Optional<mlir::Type> resultType) {
    llvm::SmallVector<ExtValue> operands;

    llvm::StringRef name = intrinsic.name;
    const auto *argLowering =
        Fortran::lower::getIntrinsicArgumentLowering(name);
    for (const auto &[arg, dummy] :
         llvm::zip(procRef.arguments(),
                   intrinsic.characteristics.value().dummyArguments)) {
      auto *expr = Fortran::evaluate::UnwrapExpr<
          Fortran::evaluate::Expr<Fortran::evaluate::SomeType>>(arg);
      if (!expr) {
        // Absent optional.
        operands.emplace_back(Fortran::lower::getAbsentIntrinsicArgument());
        continue;
      }
      if (!argLowering) {
        // No argument lowering instruction, lower by value.
        operands.emplace_back(genval(*expr));
        continue;
      }
      // Ad-hoc argument lowering handling.
      auto lowerAs = Fortran::lower::lowerIntrinsicArgumentAs(
          getLoc(), *argLowering, dummy.name);
      switch (lowerAs) {
      case Fortran::lower::LowerIntrinsicArgAs::Value:
        operands.emplace_back(genval(*expr));
        continue;
      case Fortran::lower::LowerIntrinsicArgAs::Addr:
        operands.emplace_back(gen(*expr));
        continue;
      case Fortran::lower::LowerIntrinsicArgAs::Box:
        operands.emplace_back(builder.createBox(getLoc(), genBoxArg(*expr)));
        continue;
      case Fortran::lower::LowerIntrinsicArgAs::Inquired:
        operands.emplace_back(lowerIntrinsicArgumentAsInquired(*expr));
        continue;
      }
      llvm_unreachable("bad switch");
    }
    // Let the intrinsic library lower the intrinsic procedure call
    return Fortran::lower::genIntrinsicCall(builder, getLoc(), name, resultType,
                                            operands, stmtCtx);
  }

  template <typename A>
  bool isCharacterType(const A &exp) {
    if (auto type = exp.GetType())
      return type->category() == Fortran::common::TypeCategory::Character;
    return false;
  }

  /// helper to detect statement functions
  static bool
  isStatementFunctionCall(const Fortran::evaluate::ProcedureRef &procRef) {
    if (const auto *symbol = procRef.proc().GetSymbol())
      if (const auto *details =
              symbol->detailsIf<Fortran::semantics::SubprogramDetails>())
        return details->stmtFunction().has_value();
    return false;
  }
  /// Generate Statement function calls
  ExtValue genStmtFunctionRef(const Fortran::evaluate::ProcedureRef &procRef) {
    const auto *symbol = procRef.proc().GetSymbol();
    assert(symbol && "expected symbol in ProcedureRef of statement functions");
    const auto &details = symbol->get<Fortran::semantics::SubprogramDetails>();

    // Statement functions have their own scope, we just need to associate
    // the dummy symbols to argument expressions. They are no
    // optional/alternate return arguments. Statement functions cannot be
    // recursive (directly or indirectly) so it is safe to add dummy symbols to
    // the local map here.
    symMap.pushScope();
    for (auto [arg, bind] :
         llvm::zip(details.dummyArgs(), procRef.arguments())) {
      assert(arg && "alternate return in statement function");
      assert(bind && "optional argument in statement function");
      const auto *expr = bind->UnwrapExpr();
      // TODO: assumed type in statement function, that surprisingly seems
      // allowed, probably because nobody thought of restricting this usage.
      // gfortran/ifort compiles this.
      assert(expr && "assumed type used as statement function argument");
      // As per Fortran 2018 C1580, statement function arguments can only be
      // scalars, so just pass the box with the address.
      symMap.addSymbol(*arg, gen(*expr));
    }

    // Explicitly map statement function host associated symbols to their
    // parent scope lowered symbol box.
    for (const Fortran::semantics::SymbolRef &sym :
         Fortran::evaluate::CollectSymbols(*details.stmtFunction())) {
      if (const auto *details =
              sym->detailsIf<Fortran::semantics::HostAssocDetails>()) {
        if (!symMap.lookupSymbol(*sym)) {
          symMap.addSymbol(
              *sym, symMap.lookupSymbol(details->symbol()).toExtendedValue());
        }
      }
    }

    auto result = genval(details.stmtFunction().value());
    LLVM_DEBUG(llvm::dbgs() << "stmt-function: " << result << '\n');
    symMap.popScope();
    return result;
  }

  /// Helper to package a Value and its properties into an ExtendedValue.
  ExtValue toExtendedValue(mlir::Value base,
                           llvm::ArrayRef<mlir::Value> extents,
                           llvm::ArrayRef<mlir::Value> lengths) {
    auto type = base.getType();
    if (type.isa<fir::BoxType>())
      return fir::BoxValue(base, /*lbounds=*/{}, lengths, extents);
    if (auto pointedType = fir::dyn_cast_ptrEleTy(type))
      type = pointedType;
    if (type.isa<fir::BoxType>())
      return fir::MutableBoxValue(base, lengths, /*mutableProperties*/ {});
    if (auto seqTy = type.dyn_cast<fir::SequenceType>()) {
      if (seqTy.getDimension() != extents.size())
        fir::emitFatalError(getLoc(), "incorrect number of extents for array");
      if (seqTy.getEleTy().isa<fir::CharacterType>()) {
        if (lengths.empty())
          fir::emitFatalError(getLoc(), "missing length for character");
        assert(lengths.size() == 1);
        return fir::CharArrayBoxValue(base, lengths[0], extents);
      }
      return fir::ArrayBoxValue(base, extents);
    }
    if (type.isa<fir::CharacterType>()) {
      if (lengths.empty())
        fir::emitFatalError(getLoc(), "missing length for character");
      assert(lengths.size() == 1);
      return fir::CharBoxValue(base, lengths[0]);
    }
    return base;
  }

  // Find the argument that corresponds to the host associations.
  // Verify some assumptions about how the signature was built here.
  [[maybe_unused]] static unsigned findHostAssocTuplePos(mlir::FuncOp fn) {
    // Scan the argument list from last to first as the host associations are
    // appended for now.
    for (unsigned i = fn.getNumArguments(); i > 0; --i)
      if (fn.getArgAttr(i - 1, fir::getHostAssocAttrName())) {
        // Host assoc tuple must be last argument (for now).
        assert(i == fn.getNumArguments() && "tuple must be last");
        return i - 1;
      }
    llvm_unreachable("anyFuncArgsHaveAttr failed");
  }

  /// Create a contiguous temporary array with the same shape,
  /// length parameters and type as mold
  ExtValue genTempFromMold(const ExtValue &mold, llvm::StringRef tempName) {
    auto type = fir::dyn_cast_ptrOrBoxEleTy(fir::getBase(mold).getType());
    assert(type && "expected descriptor or memory type");
    auto loc = getLoc();
    auto extents = fir::factory::getExtents(builder, loc, mold);
    auto typeParams = fir::getTypeParams(mold);
    mlir::Value temp = builder.create<fir::AllocMemOp>(loc, type, tempName,
                                                       typeParams, extents);
    auto *bldr = &converter.getFirOpBuilder();
    // TODO: call finalizer if needed.
    stmtCtx.attachCleanup([=]() { bldr->create<fir::FreeMemOp>(loc, temp); });
    if (fir::unwrapSequenceType(type).isa<fir::CharacterType>()) {
      auto len = typeParams.empty()
                     ? fir::factory::readCharLen(builder, loc, mold)
                     : typeParams[0];
      return fir::CharArrayBoxValue{temp, len, extents};
    }
    return fir::ArrayBoxValue{temp, extents};
  }

  /// Copy \p source array into \p dest array. Both arrays must be
  /// conforming, but neither array must be contiguous.
  void genArrayCopy(ExtValue dest, ExtValue source) {
    return createSomeArrayAssignment(converter, dest, source, symMap, stmtCtx);
  }

  /// Lower a non-elemental procedure reference and read allocatable and pointer
  /// results into normal values.
  ExtValue genProcedureRef(const Fortran::evaluate::ProcedureRef &procRef,
                           llvm::Optional<mlir::Type> resultType) {
    auto res = genRawProcedureRef(procRef, resultType);
    // In most contexts, pointers and allocatable do not appear as allocatable
    // or pointer variable on the caller side (see 8.5.3 note 1 for
    // allocatables). The few context where this can happen must call
    // genRawProcedureRef directly.
    if (const auto *box = res.getBoxOf<fir::MutableBoxValue>())
      return fir::factory::genMutableBoxRead(builder, getLoc(), *box);
    return res;
  }

  /// Lower a non-elemental procedure reference.
  ExtValue genRawProcedureRef(const Fortran::evaluate::ProcedureRef &procRef,
                              llvm::Optional<mlir::Type> resultType) {
    if (const auto *intrinsic = procRef.proc().GetSpecificIntrinsic())
      return genIntrinsicRef(procRef, *intrinsic, resultType);

    if (isStatementFunctionCall(procRef))
      return genStmtFunctionRef(procRef);

    auto loc = getLoc();
    Fortran::lower::CallerInterface caller(procRef, converter);
    using PassBy = Fortran::lower::CallerInterface::PassEntityBy;

    llvm::SmallVector<fir::MutableBoxValue> mutableModifiedByCall;
    // List of <var, temp> where temp must be copied into var after the call.
    llvm::SmallVector<std::pair<ExtValue, ExtValue>, 4> copyOutPairs;

    auto callSiteType = caller.genFunctionType();
    for (const auto &arg : caller.getPassedArguments()) {
      const auto *actual = arg.entity;
      auto argTy = callSiteType.getInput(arg.firArgument);
      if (!actual) {
        // Optional dummy argument for which there is no actual argument.
        caller.placeInput(arg, builder.create<fir::AbsentOp>(loc, argTy));
        continue;
      }
      const auto *expr = actual->UnwrapExpr();
      if (!expr)
        TODO(loc, "assumed type actual argument lowering");

      if (arg.passBy == PassBy::Value) {
        auto argVal = genval(*expr);
        if (!fir::isUnboxedValue(argVal))
          fir::emitFatalError(
              loc, "internal error: passing non trivial value by value");
        caller.placeInput(arg, fir::getBase(argVal));
        continue;
      }

      if (arg.passBy == PassBy::MutableBox) {
        if (Fortran::evaluate::UnwrapExpr<Fortran::evaluate::NullPointer>(
                *expr)) {
          // If expr is NULL(), the mutableBox created must be a deallocated
          // pointer with the dummy argument characteristics (see table 16.5
          // in Fortran 2018 standard).
          // No length parameters are set for the created box because any non
          // deferred type parameters of the dummy will be evaluated on the
          // callee side, and it is illegal to use NULL without a MOLD if any
          // dummy length parameters are assumed.
          auto boxTy = fir::dyn_cast_ptrEleTy(argTy);
          assert(boxTy && boxTy.isa<fir::BoxType>() &&
                 "must be a fir.box type");
          auto boxStorage = builder.createTemporary(loc, boxTy);
          auto nullBox = fir::factory::createUnallocatedBox(
              builder, loc, boxTy, /*nonDeferredParams=*/{});
          builder.create<fir::StoreOp>(loc, nullBox, boxStorage);
          caller.placeInput(arg, boxStorage);
          continue;
        }
        auto mutableBox = genMutableBoxValue(*expr);
        auto irBox = fir::factory::getMutableIRBox(builder, loc, mutableBox);
        caller.placeInput(arg, irBox);
        if (arg.mayBeModifiedByCall())
          mutableModifiedByCall.emplace_back(std::move(mutableBox));
        continue;
      }

      if (arg.passBy == PassBy::BaseAddress || arg.passBy == PassBy::BoxChar) {
        auto argAddr = [&]() -> ExtValue {
          // Non contiguous variable need to be copied into a contiguous temp,
          // and the temp need to be copied back after the call in case it was
          // modified.
          if (Fortran::evaluate::IsVariable(*expr) && expr->Rank() > 0 &&
              !Fortran::evaluate::IsSimplyContiguous(
                  *expr, converter.getFoldingContext())) {
            auto box = genBoxArg(*expr);
            auto temp = genTempFromMold(box, ".copyinout");
            if (arg.mayBeReadByCall())
              genArrayCopy(temp, box);
            if (arg.mayBeModifiedByCall())
              copyOutPairs.emplace_back(box, temp);
            return temp;
          }
          auto baseAddr = genExtAddr(*expr);
          // Scalar and contiguous expressions may be lowered to a fir.box,
          // either to account for potential polymorphism, or because lowering
          // did not account for some contiguity hints.
          // Here, polymorphism does not matter (an entity of the declared type
          // is passed, not one of the dynamic type), and the expr is known to
          // be simply contiguous, so it is safe to unbox it and pass the
          // address without making a copy.
          if (const auto *box = baseAddr.getBoxOf<fir::BoxValue>())
            return fir::factory::readBoxValue(builder, loc, *box);
          return baseAddr;
        }();
        if (arg.passBy == PassBy::BaseAddress) {
          caller.placeInput(arg, fir::getBase(argAddr));
        } else {
          assert(arg.passBy == PassBy::BoxChar);
          auto helper = fir::factory::CharacterExprHelper{builder, loc};
          auto boxChar = argAddr.match(
              [&](const fir::CharBoxValue &x) { return helper.createEmbox(x); },
              [&](const fir::CharArrayBoxValue &x) {
                return helper.createEmbox(x);
              },
              [&](const auto &) -> mlir::Value {
                fir::emitFatalError(
                    loc, "internal error: actual argument is not a character");
              });
          caller.placeInput(arg, boxChar);
        }
      } else if (arg.passBy == PassBy::Box) {
        // Before lowering to an address, handle the allocatable/pointer actual
        // argument to optional fir.box dummy. It is legal to pass
        // unallocated/disassociated entity to an optional. In this case, an
        // absent fir.box must be created instead of a fir.box with a null value
        // (Fortran 2018 15.5.2.12 point 1).
        if (arg.isOptional() && isAllocatableOrPointer(*expr)) {
          // Note that passing an absent allocatable to a non-allocatable
          // optional dummy argument is illegal (15.5.2.12 point 3 (8)). So
          // nothing has to be done to generate an absent argument in this case,
          // and it is OK to unconditionally read the mutable box here.
          auto mutableBox = genMutableBoxValue(*expr);
          auto isAllocated = fir::factory::genIsAllocatedOrAssociatedTest(
              builder, loc, mutableBox);
          auto absent = builder.create<fir::AbsentOp>(loc, argTy);
          /// For now, assume it is not OK to pass the allocatable/pointer
          /// descriptor to a non pointer/allocatable dummy. That is a strict
          /// interpretation of 18.3.6 point 4 that stipulates the descriptor
          /// has the dummy attributes in BIND(C) contexts.
          auto box = builder.createBox(
              loc, fir::factory::genMutableBoxRead(builder, loc, mutableBox));
          // Need the box types to be exactly similar for the selectOp.
          auto convertedBox = builder.createConvert(loc, argTy, box);
          caller.placeInput(arg, builder.create<mlir::SelectOp>(
                                     loc, isAllocated, convertedBox, absent));
        } else {
          auto box = builder.createBox(loc, genBoxArg(*expr));
          caller.placeInput(arg, box);
        }
      } else if (arg.passBy == PassBy::AddressAndLength) {
        auto argRef = genExtAddr(*expr);
        caller.placeAddressAndLengthInput(arg, fir::getBase(argRef),
                                          fir::getLen(argRef));
      } else {
        TODO(loc, "pass by value in non elemental function call");
      }
    }

    // Handle cases where caller must allocate the result or a fir.box for it.
    bool mustPopSymMap = false;
    if (caller.mustMapInterfaceSymbols()) {
      symMap.pushScope();
      mustPopSymMap = true;
      Fortran::lower::mapCallInterfaceSymbols(converter, caller, symMap);
    }

    auto idxTy = builder.getIndexType();
    auto lowerSpecExpr = [&](const auto &expr) -> mlir::Value {
      return builder.createConvert(
          loc, idxTy, fir::getBase(converter.genExprValue(expr, stmtCtx)));
    };
    llvm::SmallVector<mlir::Value> resultLengths;
    auto allocatedResult = [&]() -> llvm::Optional<ExtValue> {
      llvm::SmallVector<mlir::Value> extents;
      llvm::SmallVector<mlir::Value> lengths;
      if (!caller.callerAllocateResult())
        return {};
      auto type = caller.getResultStorageType();
      if (type.isa<fir::SequenceType>())
        caller.walkResultExtents([&](const Fortran::lower::SomeExpr &e) {
          extents.emplace_back(lowerSpecExpr(e));
        });
      caller.walkResultLengths([&](const Fortran::lower::SomeExpr &e) {
        lengths.emplace_back(lowerSpecExpr(e));
      });
      /// Result lengths parameters should not be provided to box storage
      /// allocation and save_results, but they are still useful information to
      /// keep in the ExtentdedValue if non-deferred.
      if (!type.isa<fir::BoxType>())
        resultLengths = lengths;
      auto temp =
          builder.createTemporary(loc, type, ".result", extents, resultLengths);
      return toExtendedValue(temp, extents, lengths);
    }();

    if (mustPopSymMap)
      symMap.popScope();

    // Place allocated result or prepare the fir.save_result arguments.
    mlir::Value arrayResultShape;
    if (allocatedResult) {
      if (auto resultArg = caller.getPassedResult()) {
        if (resultArg->passBy == PassBy::AddressAndLength)
          caller.placeAddressAndLengthInput(*resultArg,
                                            fir::getBase(*allocatedResult),
                                            fir::getLen(*allocatedResult));
        else if (resultArg->passBy == PassBy::BaseAddress)
          caller.placeInput(*resultArg, fir::getBase(*allocatedResult));
        else
          fir::emitFatalError(
              loc, "only expect character scalar result to be passed by ref");
      } else {
        assert(caller.mustSaveResult());
        arrayResultShape = allocatedResult->match(
            [&](const fir::CharArrayBoxValue &) {
              return builder.createShape(loc, *allocatedResult);
            },
            [&](const fir::ArrayBoxValue &) {
              return builder.createShape(loc, *allocatedResult);
            },
            [&](const auto &) { return mlir::Value{}; });
      }
    }

    // In older Fortran, procedure argument types are inferred. This may lead
    // different view of what the function signature is in different locations.
    // Casts are inserted as needed below to acomodate this.

    // The mlir::FuncOp type prevails, unless it has a different number of
    // arguments which can happen in legal program if it was passed as a dummy
    // procedure argument earlier with no further type information.
    mlir::Value funcPointer;
    mlir::SymbolRefAttr funcSymbolAttr;
    bool addHostAssociations = false;
    if (const auto *sym = caller.getIfIndirectCallSymbol()) {
      funcPointer = symMap.lookupSymbol(*sym).getAddr();
      assert(funcPointer &&
             "dummy procedure or procedure pointer not in symbol map");
    } else {
      auto funcOpType = caller.getFuncOp().getType();
      auto symbolAttr = builder.getSymbolRefAttr(caller.getMangledName());
      if (callSiteType.getNumResults() == funcOpType.getNumResults() &&
          callSiteType.getNumInputs() + 1 == funcOpType.getNumInputs() &&
          fir::anyFuncArgsHaveAttr(caller.getFuncOp(),
                                   fir::getHostAssocAttrName())) {
        // The number of arguments is off by one, and we're lowering a function
        // with host associations. Modify call to include host associations
        // argument by appending the value at the end of the operands.
        assert(funcOpType.getInput(findHostAssocTuplePos(caller.getFuncOp())) ==
               converter.hostAssocTupleValue().getType());
        addHostAssociations = true;
      }
      if (!addHostAssociations &&
          (callSiteType.getNumResults() != funcOpType.getNumResults() ||
           callSiteType.getNumInputs() != funcOpType.getNumInputs())) {
        // Deal with argument number mismatch by making a function pointer so
        // that function type cast can be inserted. Do not emit a warning here
        // because this can happen in legal program if the function is not
        // defined here and it was first passed as an argument without any more
        // information.
        funcPointer =
            builder.create<fir::AddrOfOp>(loc, funcOpType, symbolAttr);
      } else if (callSiteType.getResults() != funcOpType.getResults()) {
        // Implicit interface result type mismatch are not standard Fortran, but
        // some compilers are not complaining about it.  The front end is not
        // protecting lowering from this currently. Support this with a
        // discouraging warning.
        LLVM_DEBUG(mlir::emitWarning(
            loc, "a return type mismatch is not standard compliant and may "
                 "lead to undefined behavior."));
        // Cast the actual function to the current caller implicit type because
        // that is the behavior we would get if we could not see the definition.
        funcPointer =
            builder.create<fir::AddrOfOp>(loc, funcOpType, symbolAttr);
      } else {
        funcSymbolAttr = symbolAttr;
      }
    }

    auto funcType = funcPointer ? callSiteType : caller.getFuncOp().getType();
    llvm::SmallVector<mlir::Value> operands;
    // First operand of indirect call is the function pointer. Cast it to
    // required function type for the call to handle procedures that have a
    // compatible interface in Fortran, but that have different signatures in
    // FIR.
    if (funcPointer)
      operands.push_back(builder.createConvert(loc, funcType, funcPointer));

    // Deal with potential mismatches in arguments types. Passing an array to a
    // scalar argument should for instance be tolerated here.
    for (auto [fst, snd] :
         llvm::zip(caller.getInputs(), funcType.getInputs())) {
      auto cast = builder.convertWithSemantics(getLoc(), snd, fst);
      operands.push_back(cast);
    }

    // Add host associations as necessary.
    if (addHostAssociations)
      operands.push_back(converter.hostAssocTupleValue());

    auto call = builder.create<fir::CallOp>(loc, funcType.getResults(),
                                            funcSymbolAttr, operands);

    if (caller.mustSaveResult())
      builder.create<fir::SaveResultOp>(
          loc, call.getResult(0), fir::getBase(allocatedResult.getValue()),
          arrayResultShape, resultLengths);

    // Sync pointers and allocatables that may have been modified during the
    // call.
    for (const auto &mutableBox : mutableModifiedByCall)
      fir::factory::syncMutableBoxFromIRBox(builder, loc, mutableBox);
    // Handle case where result was passed as argument

    // Copy-out temps that were created for non contiguous variable arguments if
    // needed.
    for (auto [var, temp] : copyOutPairs)
      genArrayCopy(var, temp);

    if (allocatedResult) {
      allocatedResult->match(
          [&](const fir::MutableBoxValue &box) {
            if (box.isAllocatable()) {
              // 9.7.3.2 point 4. Finalize allocatables.
              auto *bldr = &converter.getFirOpBuilder();
              stmtCtx.attachCleanup(
                  [=]() { fir::factory::genFinalization(*bldr, loc, box); });
            }
          },
          [](const auto &) {});
      return *allocatedResult;
    }

    if (!resultType.hasValue())
      return mlir::Value{}; // subroutine call
    // For now, Fortran return values are implemented with a single MLIR
    // function return value.
    assert(call.getNumResults() == 1 &&
           "Expected exactly one result in FUNCTION call");
    return call.getResult(0);
  }

  template <Fortran::common::TypeCategory TC, int KIND>
  ExtValue
  genval(const Fortran::evaluate::FunctionRef<Fortran::evaluate::Type<TC, KIND>>
             &funRef) {
    auto retTy = converter.genType(TC, KIND);
    return genProcedureRef(funRef, {retTy});
  }

  ExtValue genval(const Fortran::evaluate::ProcedureRef &procRef) {
    llvm::Optional<mlir::Type> resTy;
    if (procRef.hasAlternateReturns())
      resTy = builder.getIndexType();
    return genProcedureRef(procRef, resTy);
  }

  template <typename A>
  bool isScalar(const A &x) {
    return x.Rank() == 0;
  }

  /// Helper to detect Transformational function reference.
  template <typename T>
  bool isTransformationalRef(const T &) {
    return false;
  }
  template <typename T>
  bool isTransformationalRef(const Fortran::evaluate::FunctionRef<T> &funcRef) {
    return !funcRef.IsElemental() && funcRef.Rank();
  }
  template <typename T>
  bool isTransformationalRef(Fortran::evaluate::Expr<T> expr) {
    return std::visit([&](const auto &e) { return isTransformationalRef(e); },
                      expr.u);
  }

  template <typename A>
  ExtValue asArray(const A &x) {
    auto expr = toEvExpr(x);
    auto optShape =
        Fortran::evaluate::GetShape(converter.getFoldingContext(), expr);
    return Fortran::lower::createSomeArrayTempValue(converter, optShape, expr,
                                                    symMap, stmtCtx);
  }

  /// Lower an array value as an argument. This argument can be passed as a box
  /// value, so it may be possible to avoid making a temporary.
  template <typename A>
  ExtValue asArrayArg(const Fortran::evaluate::Expr<A> &x) {
    return std::visit([&](const auto &e) { return asArrayArg(e, x); }, x.u);
  }
  template <typename A, typename B>
  ExtValue asArrayArg(const Fortran::evaluate::Expr<A> &x, const B &y) {
    return std::visit([&](const auto &e) { return asArrayArg(e, y); }, x.u);
  }
  template <typename A, typename B>
  ExtValue asArrayArg(const Fortran::evaluate::Designator<A> &, const B &x) {
    // Designator is being passed as an argument to a procedure. Lower the
    // expression to a boxed value.
    return Fortran::lower::createSomeArrayBox(converter, toEvExpr(x), symMap,
                                              stmtCtx);
  }
  template <typename A, typename B>
  ExtValue asArrayArg(const A &, const B &x) {
    // If the expression to pass as an argument is not a designator, then create
    // an array temp.
    return asArray(x);
  }

  template <typename A>
  ExtValue gen(const Fortran::evaluate::Expr<A> &x) {
    // Whole array symbols or components, and results of transformational
    // functions already have a storage and the scalar expression lowering path
    // is used to not create a new temporary storage.
    if (isScalar(x) ||
        Fortran::evaluate::UnwrapWholeSymbolOrComponentDataRef(x) ||
        isTransformationalRef(x))
      return std::visit([&](const auto &e) { return genref(e); }, x.u);
    if (useBoxArg)
      return asArrayArg(x);
    return asArray(x);
  }
  template <typename A>
  ExtValue genval(const Fortran::evaluate::Expr<A> &x) {
    if (isScalar(x) || Fortran::evaluate::UnwrapWholeSymbolDataRef(x) ||
        inInitializer)
      return std::visit([&](const auto &e) { return genval(e); }, x.u);
    return asArray(x);
  }

  template <int KIND>
  ExtValue genval(const Fortran::evaluate::Expr<Fortran::evaluate::Type<
                      Fortran::common::TypeCategory::Logical, KIND>> &exp) {
    return std::visit([&](const auto &e) { return genval(e); }, exp.u);
  }

  using RefSet =
      std::tuple<Fortran::evaluate::ComplexPart, Fortran::evaluate::Substring,
                 Fortran::evaluate::DataRef, Fortran::evaluate::Component,
                 Fortran::evaluate::ArrayRef, Fortran::evaluate::CoarrayRef,
                 Fortran::semantics::SymbolRef>;
  template <typename A>
  static constexpr bool inRefSet = Fortran::common::HasMember<A, RefSet>;

  template <typename A, typename = std::enable_if_t<inRefSet<A>>>
  ExtValue genref(const A &a) {
    return gen(a);
  }
  template <typename A>
  ExtValue genref(const A &a) {
    auto exv = genval(a);
    auto valBase = fir::getBase(exv);
    // Functions are always referent.
    if (valBase.getType().template isa<mlir::FunctionType>() ||
        fir::conformsWithPassByRef(valBase.getType()))
      return exv;

    // Since `a` is not itself a valid referent, determine its value and
    // create a temporary location at the begining of the function for
    // referencing.
    auto val = valBase;
    if constexpr (!Fortran::common::HasMember<
                      A, Fortran::evaluate::TypelessExpression>) {
      if constexpr (A::Result::category ==
                    Fortran::common::TypeCategory::Logical) {
        // Ensure logicals that may have been lowered to `i1` are cast to the
        // Fortran logical type before being placed in memory.
        auto type = converter.genType(A::Result::category, A::Result::kind);
        val = builder.createConvert(getLoc(), type, valBase);
      }
    }
    auto func = builder.getFunction();
    auto initPos = builder.saveInsertionPoint();
    builder.setInsertionPointToStart(&func.front());
    auto mem = builder.create<fir::AllocaOp>(getLoc(), val.getType());
    builder.restoreInsertionPoint(initPos);
    builder.create<fir::StoreOp>(getLoc(), val, mem);
    return fir::substBase(exv, mem.getResult());
  }

  template <typename A, template <typename> typename T,
            typename B = std::decay_t<T<A>>,
            std::enable_if_t<
                std::is_same_v<B, Fortran::evaluate::Expr<A>> ||
                    std::is_same_v<B, Fortran::evaluate::Designator<A>> ||
                    std::is_same_v<B, Fortran::evaluate::FunctionRef<A>>,
                bool> = true>
  ExtValue genref(const T<A> &x) {
    return gen(x);
  }

private:
  mlir::Location location;
  Fortran::lower::AbstractConverter &converter;
  fir::FirOpBuilder &builder;
  Fortran::lower::StatementContext &stmtCtx;
  Fortran::lower::SymMap &symMap;
  bool inInitializer;
  bool useBoxArg{false}; // expression lowered as argument
};
} // namespace

//===----------------------------------------------------------------------===//
//
// Lowering of array expressions.
//
//===----------------------------------------------------------------------===//

// Helper for changing the semantics in a given context. Preserves the current
// semantics which is resumed when the "push" goes out of scope.
#define PushSemantics(PushVal)                                                 \
  [[maybe_unused]] auto pushSemanticsLocalVariable97201 =                      \
      Fortran::common::ScopedSet(semant, PushVal);

namespace {
class ArrayExprLowering {
  /// An array is accessed with a vector of subscript values. In an aray
  /// expression with a user-defined explicit iteration space, the vector may be
  /// composed of both subscripts using the explicit iteration space and
  /// implicit subscripts. The explicit subscripts may come before or after the
  /// implicit subscripts in the vector as applied to a component path.
  ///
  /// For example, given a designator of rank 2, `a1(i,j)%a2%a3(k), `i` and `j`
  /// could be pre-explicit, then two (rank of `a2`) impplicit values, and
  /// finally `k` which could be a post-explicit access value.
  enum class AccessKind { Implicit, PreExplicit, PostExplicit };

  /// AccessValue is a (value, tag) pair used by an IterationSpace to keep track
  /// of the access kind of values in the iteration space so they may be applied
  /// in a continuation properly.
  struct AccessValue {
    explicit AccessValue(AccessKind k, mlir::Value v) : kind{k}, val{v} {}
    AccessValue(mlir::Value v) : AccessValue(AccessKind::Implicit, v) {}

    mlir::Value value() const { return val; }
    AccessKind access() const { return kind; }

  private:
    AccessKind kind;
    mlir::Value val;
  };

  struct IterationSpace {
    IterationSpace() = default;

    template <typename A>
    explicit IterationSpace(mlir::Value inArg, mlir::Value outRes,
                            llvm::iterator_range<A> range)
        : inArg{inArg}, outRes{outRes}, indices{range.begin(), range.end()} {}

    explicit IterationSpace(const IterationSpace &from,
                            llvm::ArrayRef<mlir::Value> idxs)
        : inArg(from.inArg), outRes(from.outRes), element(from.element),
          indices(idxs.begin(), idxs.end()) {}

    mlir::Value innerArgument() const { return inArg; }
    mlir::Value outerResult() const { return outRes; }
    llvm::SmallVector<mlir::Value> iterVec() const {
      llvm::SmallVector<mlir::Value> result;
      for (auto i : indices)
        result.push_back(i.value());
      return result;
    }
    mlir::Value iterValue(std::size_t i) const {
      assert(i < indices.size());
      return indices[i].value();
    }

    /// Set (rewrite) the Value at a given index.
    void setIndexValue(std::size_t i, mlir::Value v) {
      assert(i < indices.size());
      indices[i] = v;
    }

    void insertIndexValue(std::size_t i, AccessValue av) {
      assert(i <= indices.size());
      indices.insert(indices.begin() + i, av);
    }

    void insertIndexValue(std::size_t i, mlir::Value v) {
      insertIndexValue(i, AccessValue(AccessKind::Implicit, v));
    }

    /// Append an argument to the iteration space. Like prepended arguments, it
    /// is assumed the appended arguments will be added in a reverse order.
    void appendIndexValue(mlir::Value v) {
      auto appendPoint = endImplicitIndex();
      insertIndexValue(appendPoint, AccessValue(AccessKind::PostExplicit, v));
    }

    /// Prepend an argument to the iteration space. Arguments are assumed to be
    /// prepended in reverse order.
    void prependIndexValue(mlir::Value v) {
      insertIndexValue(0, AccessValue(AccessKind::PreExplicit, v));
    }

    /// Set the `element` value. This is the SSA value that corresponds to an
    /// element of the resultant array value.
    void setElement(mlir::Value ele) {
      assert(!element && "result element already set");
      element = ele;
    }

    mlir::Value getElement() const {
      assert(element);
      return element;
    }

    /// Index of the first implicit access in indices. Returns the index
    /// immediately after the last pre explicit index or 0. Returns 0 if there
    /// are no indices in this iteration space.
    size_t beginImplicitIndex() const {
      const auto size = indices.size();
      std::remove_const_t<decltype(size)> result = 0;
      while (result < size &&
             indices[result].access() == AccessKind::PreExplicit)
        ++result;
      return result;
    }

    /// Index of element immediately after the last implicit access index.
    /// Returns the index immediately before the first post explicit index or 0.
    /// Returns 0 if there are no indices in this iteration space.
    size_t endImplicitIndex() const {
      auto result = indices.size();
      while (result > 0 &&
             indices[result - 1].access() == AccessKind::PostExplicit)
        --result;
      return result;
    }

  private:
    mlir::Value inArg;
    mlir::Value outRes;
    mlir::Value element;
    llvm::SmallVector<AccessValue> indices;
  };

  using ExtValue = fir::ExtendedValue;
  using IterSpace = const IterationSpace &;      // active iteration space
  using CC = std::function<ExtValue(IterSpace)>; // current continuation
  using PC =
      std::function<IterationSpace(IterSpace)>; // projection continuation
  using ArrayBaseTy =
      std::variant<std::monostate, const Fortran::evaluate::ArrayRef *,
                   const Fortran::evaluate::DataRef *>;

  struct ComponentCollection {
    ComponentCollection() : pc{[=](IterSpace s) { return s; }} {}
    ComponentCollection(const ComponentCollection &) = delete;
    ComponentCollection &operator=(const ComponentCollection &) = delete;

    llvm::SmallVector<mlir::Value> trips;
    llvm::SmallVector<mlir::Value> components;
    PC pc;
  };

public:
  //===--------------------------------------------------------------------===//
  // Regular array assignment
  //===--------------------------------------------------------------------===//

  /// Entry point for array assignments. Both the left-hand and right-hand sides
  /// can either be ExtendedValue or evaluate::Expr.
  template <typename TL, typename TR>
  static void lowerArrayAssignment(Fortran::lower::AbstractConverter &converter,
                                   Fortran::lower::SymMap &symMap,
                                   Fortran::lower::StatementContext &stmtCtx,
                                   const TL &lhs, const TR &rhs) {
    ArrayExprLowering ael{converter, stmtCtx, symMap,
                          ConstituentSemantics::CopyInCopyOut};
    ael.lowerArrayAssignment(lhs, rhs);
  }

  template <typename TL, typename TR>
  void lowerArrayAssignment(const TL &lhs, const TR &rhs) {
    auto loc = getLoc();
    /// Here the target subspace is not necessarily contiguous. The ArrayUpdate
    /// continuation is implicitly returned in `ccDest` and the ArrayLoad in
    /// `destination`.
    PushSemantics(ConstituentSemantics::ProjectedCopyInCopyOut);
    ccDest = genarr(lhs);
    determineShapeOfDest(lhs);
    semant = ConstituentSemantics::RefTransparent;
    auto exv = lowerArrayExpression(rhs);
    builder.create<fir::ArrayMergeStoreOp>(
        loc, destination, fir::getBase(exv), destination.memref(),
        destination.slice(), destination.typeparams());
  }

  void lowerArrayAssignment(
      fir::ArrayLoadOp lhs,
      const Fortran::evaluate::Expr<Fortran::evaluate::SomeType> &rhs,
      llvm::ArrayRef<mlir::Value> indices,
      llvm::ArrayRef<mlir::Value> extents) {
    auto loc = getLoc();
    destination = lhs;
    auto lambda = [=](IterSpace iters) -> ExtValue {
      auto innerArg = iters.innerArgument();
      auto resTy = adjustedArrayElementType(innerArg.getType());
      auto arrUpdate = builder.create<fir::ArrayUpdateOp>(
          loc, resTy, innerArg, iters.getElement(), iters.iterVec(),
          destination.typeparams());
      return abstractArrayExtValue(arrUpdate);
    };
    auto pc = [=](IterSpace iters) {
      IterationSpace newIters = iters;
      for (auto i : llvm::reverse(indices))
        newIters.prependIndexValue(i);
      return newIters;
    };
    ccDest = [=](IterSpace iters) { return lambda(pc(iters)); };
    destShape.assign(extents.begin(), extents.end());
    semant = ConstituentSemantics::RefTransparent;
    auto exv = lowerArrayExpression(rhs);
    // The array_merge_store will be created by the caller. Finish the implicit
    // loop nest here.
    builder.create<fir::ResultOp>(getLoc(), fir::getBase(exv));
  }

  //===--------------------------------------------------------------------===//
  // WHERE array assignment, FORALL assignment, and FORALL+WHERE array
  // assignment
  //===--------------------------------------------------------------------===//

  /// Entry point for array assignment when the iteration space is explicitly
  /// defined (Fortran's FORALL) with or without masks, and/or the implied
  /// iteration space involves masks (Fortran's WHERE). Both contexts (explicit
  /// space and implicit space with masks) may be present.
  static void lowerAnyMaskedArrayAssignment(
      Fortran::lower::AbstractConverter &converter,
      Fortran::lower::SymMap &symMap, Fortran::lower::StatementContext &stmtCtx,
      const Fortran::evaluate::Expr<Fortran::evaluate::SomeType> &lhs,
      const Fortran::evaluate::Expr<Fortran::evaluate::SomeType> &rhs,
      Fortran::lower::ExplicitIterSpace &explicitSpace,
      Fortran::lower::ImplicitIterSpace &implicitSpace) {
    ArrayExprLowering ael(converter, stmtCtx, symMap,
                          ConstituentSemantics::CopyInCopyOut, &explicitSpace,
                          &implicitSpace);
    ael.lowerArrayAssignment(lhs, rhs);
  }

  //===--------------------------------------------------------------------===//
  // Array assignment to allocatable array
  //===--------------------------------------------------------------------===//

  /// Entry point for assignment to allocatable array.
  static void lowerAllocatableArrayAssignment(
      Fortran::lower::AbstractConverter &converter,
      Fortran::lower::SymMap &symMap, Fortran::lower::StatementContext &stmtCtx,
      const fir::MutableBoxValue &lhs,
      const Fortran::evaluate::Expr<Fortran::evaluate::SomeType> &rhs,
      Fortran::lower::ExplicitIterSpace &explicitSpace,
      Fortran::lower::ImplicitIterSpace &implicitSpace) {
    ArrayExprLowering ael(converter, stmtCtx, symMap,
                          ConstituentSemantics::CopyInCopyOut, &explicitSpace,
                          &implicitSpace);
    ael.lowerAllocatableArrayAssignment(lhs, rhs);
  }

  /// Assignment to allocatable array.
  ///
  /// The semantics are reverse that of a "regular" array assignment. The rhs
  /// defines the iteration space of the computation and the lhs is
  /// resized/reallocated to fit if necessary.
  void lowerAllocatableArrayAssignment(
      const fir::MutableBoxValue &mutableBox,
      const Fortran::evaluate::Expr<Fortran::evaluate::SomeType> &rhs) {
    // With assignment to allocatable, we want to lower the rhs first and use
    // its shape to determine if we need to reallocate, etc.
    auto loc = getLoc();
    auto resultTy = converter.genType(rhs);
    auto rhsCC = [&]() {
      PushSemantics(ConstituentSemantics::RefTransparent);
      return genarr(rhs);
    }();
    if (!arrayOperandLoads.empty())
      destShape = getShape(arrayOperandLoads[0]);

    llvm::SmallVector<mlir::Value> lengthParams;
    // Currently no safe way to gather length from rhs (at least for
    // character, it cannot be taken from array_loads since it may be
    // changed by concatenations).
    if ((mutableBox.isCharacter() && !mutableBox.hasNonDeferredLenParams()) ||
        mutableBox.isDerivedWithLengthParameters())
      TODO(loc, "gather rhs length parameters in assignment to allocatable");

    // The allocatable must take lower bounds from the expr if reallocated.
    // An expr has lbounds only if it is an array symbol or component.
    llvm::SmallVector<mlir::Value> lbounds;
    // takeLboundsIfRealloc is only true iff the rhs is a single dataref.
    const bool takeLboundsIfRealloc =
        Fortran::evaluate::UnwrapWholeSymbolOrComponentDataRef(rhs);
    if (takeLboundsIfRealloc && !arrayOperandLoads.empty()) {
      assert(arrayOperandLoads.size() == 1 &&
             "lbounds can only come from one array");
      auto lbs = fir::factory::getOrigins(arrayOperandLoads[0].shape());
      lbounds.append(lbs.begin(), lbs.end());
    }
    fir::factory::genReallocIfNeeded(builder, loc, mutableBox, lbounds,
                                     destShape, lengthParams);
    // Create ArrayLoad for the mutable box and save it into `destination`.
    PushSemantics(ConstituentSemantics::ProjectedCopyInCopyOut);
    ccDest = genarr(fir::factory::genMutableBoxRead(builder, loc, mutableBox));
    // If the rhs is scalar, get shape from the allocatable ArrayLoad.
    if (destShape.empty())
      destShape = getShape(destination);
    // Finish lowering the loop nest.
    assert(destination && "destination must have been set");
    auto exv = lowerArrayExpression(rhsCC, resultTy);
    builder.create<fir::ArrayMergeStoreOp>(
        loc, destination, fir::getBase(exv), destination.memref(),
        destination.slice(), destination.typeparams());
  }

  /// Entry point for when an array expression appears on the lhs of an
  /// assignment. In the default case, the rhs is fully evaluated prior to any
  /// of the results being written back to the lhs. (CopyInCopyOut semantics.)
  static fir::ArrayLoadOp lowerArraySubspace(
      Fortran::lower::AbstractConverter &converter,
      Fortran::lower::SymMap &symMap, Fortran::lower::StatementContext &stmtCtx,
      const Fortran::evaluate::Expr<Fortran::evaluate::SomeType> &expr) {
    ArrayExprLowering ael{converter, stmtCtx, symMap,
                          ConstituentSemantics::CopyInCopyOut};
    return ael.lowerArraySubspace(expr);
  }

  fir::ArrayLoadOp lowerArraySubspace(
      const Fortran::evaluate::Expr<Fortran::evaluate::SomeType> &exp) {
    return std::visit(
        [&](const auto &e) {
          auto f = genarr(e);
          auto exv = f(IterationSpace{});
          if (auto *defOp = fir::getBase(exv).getDefiningOp())
            if (auto arrLd = mlir::dyn_cast<fir::ArrayLoadOp>(defOp))
              return arrLd;
          fir::emitFatalError(getLoc(), "array must be loaded");
        },
        exp.u);
  }

  /// Entry point for when an array expression appears in a context where the
  /// result must be boxed. (BoxValue semantics.)
  static ExtValue lowerBoxedArrayExpression(
      Fortran::lower::AbstractConverter &converter,
      Fortran::lower::SymMap &symMap, Fortran::lower::StatementContext &stmtCtx,
      const Fortran::evaluate::Expr<Fortran::evaluate::SomeType> &expr) {
    ArrayExprLowering ael{converter, stmtCtx, symMap,
                          ConstituentSemantics::BoxValue};
    return ael.lowerBoxedArrayExpr(expr);
  }

  ExtValue lowerBoxedArrayExpr(
      const Fortran::evaluate::Expr<Fortran::evaluate::SomeType> &exp) {
    return std::visit(
        [&](const auto &e) {
          auto f = genarr(e);
          auto exv = f(IterationSpace{});
          if (fir::getBase(exv).getType().template isa<fir::BoxType>())
            return exv;
          fir::emitFatalError(getLoc(), "array must be emboxed");
        },
        exp.u);
  }

  /// Entry point into lowering an expression with rank. This entry point is for
  /// lowering a rhs expression, for example. (RefTransparent semantics.)
  static ExtValue lowerSomeNewArrayExpression(
      Fortran::lower::AbstractConverter &converter,
      Fortran::lower::SymMap &symMap, Fortran::lower::StatementContext &stmtCtx,
      const std::optional<Fortran::evaluate::Shape> &shape,
      const Fortran::evaluate::Expr<Fortran::evaluate::SomeType> &expr) {
    ArrayExprLowering ael{converter, stmtCtx, symMap};
    if (shape.has_value())
      ael.determineShapeOfDest(*shape);
    auto loopRes = ael.lowerArrayExpression(expr);
    auto dest = ael.destination;
    auto tempRes = dest.memref();
    auto &builder = converter.getFirOpBuilder();
    auto loc = converter.getCurrentLocation();
    builder.create<fir::ArrayMergeStoreOp>(loc, dest, fir::getBase(loopRes),
                                           tempRes, dest.slice(),
                                           dest.typeparams());

    auto arrTy =
        fir::dyn_cast_ptrEleTy(tempRes.getType()).cast<fir::SequenceType>();
    if (auto charTy =
            arrTy.getEleTy().template dyn_cast<fir::CharacterType>()) {
      if (charTy.getLen() <= 0)
        TODO(loc, "CHARACTER does not have constant LEN");
      auto len = builder.createIntegerConstant(
          loc, builder.getCharacterLengthType(), charTy.getLen());
      return fir::CharArrayBoxValue(tempRes, len, dest.getExtents());
    }
    return fir::ArrayBoxValue(tempRes, dest.getExtents());
  }

  void determineShapeOfDest(const fir::ExtendedValue &lhs) {
    destShape = fir::factory::getExtents(builder, getLoc(), lhs);
  }

  void determineShapeOfDest(
      const Fortran::evaluate::Expr<Fortran::evaluate::SomeType> &lhs) {
    if (explicitSpace) {
      assert(destination);
      destShape = getShape(destination);
    } else if (auto shape = Fortran::evaluate::GetShape(
                   converter.getFoldingContext(), lhs)) {
      determineShapeOfDest(*shape);
    }
  }

  static bool evalShapeIsConstant(const Fortran::evaluate::Shape &shape) {
    for (const auto &s : shape)
      if (!s || !Fortran::evaluate::IsConstantExpr(*s))
        return false;
    return true;
  }

  void convertFEShape(const Fortran::evaluate::Shape &shape,
                      llvm::SmallVectorImpl<mlir::Value> &result) {
    if (evalShapeIsConstant(shape)) {
      auto idxTy = builder.getIndexType();
      auto loc = getLoc();
      for (const auto &s : shape)
        result.emplace_back(builder.createConvert(
            loc, idxTy, convertOptExtentExpr(converter, stmtCtx, s)));
    }
  }

  /// Convert the shape computed by the front end if it is constant. Modifies
  /// `destShape` when successful.
  void determineShapeOfDest(const Fortran::evaluate::Shape &shape) {
    assert(destShape.empty());
    convertFEShape(shape, destShape);
  }

  /// CHARACTER and derived type elements are treated as memory references. The
  /// numeric types are treated as values.
  static bool isAdjustedArrayElementType(mlir::Type t) {
    return fir::isa_char(t) || t.isa<fir::RecordType>();
  }
  static bool elementTypeWasAdjusted(mlir::Type t) {
    if (auto ty = t.dyn_cast<fir::ReferenceType>())
      return isAdjustedArrayElementType(ty.getEleTy());
    return false;
  }
  static mlir::Type adjustedArrayElementType(mlir::Type t) {
    return isAdjustedArrayElementType(t) ? fir::ReferenceType::get(t) : t;
  }

  ExtValue lowerArrayExpression(
      const Fortran::evaluate::Expr<Fortran::evaluate::SomeType> &exp) {
    auto resTy = explicitSpace ? destination->getResult(0).getType()
                               : converter.genType(exp);
    return std::visit(
        [&](const auto &e) { return lowerArrayExpression(genarr(e), resTy); },
        exp.u);
  }
  ExtValue lowerArrayExpression(const ExtValue &exv) {
    assert(!explicitSpace);
    auto resTy = fir::unwrapPassByRefType(fir::getBase(exv).getType());
    return lowerArrayExpression(genarr(exv), resTy);
  }

  // Extract the subelement type and perform any trivial conversions, such as
  // `index -> i32` or `i1 -> !fir.logical`.
  mlir::Value adjustedArrayInsert(const ExtValue &exv, mlir::Type arrTy) {
    auto eleTy = fir::unwrapSequenceType(arrTy);
    return builder.createConvert(getLoc(), adjustedArrayElementType(eleTy),
                                 fir::getBase(exv));
  }

  /// For an elemental array expression.
  ///   1. Lower the scalars and array loads.
  ///   2. Create the iteration space.
  ///   3. Create the element-by-element computation in the loop.
  ///   4. Return the resulting array value.
  /// If no destination was set in the array context, a temporary of
  /// \p resultTy will be created to hold the evaluated expression.
  /// Otherwise, \p resultTy is ignored and the expression is evaluated
  /// in the destination. \p f is a continuation built from an
  /// evaluate::Expr or an ExtendedValue.
  ExtValue lowerArrayExpression(CC f, mlir::Type resultTy) {
    auto loc = getLoc();
    auto [iterSpace, insPt] = genIterSpace(resultTy);
    auto innerArg = iterSpace.innerArgument();
    auto exv = f(iterSpace);
    mlir::Value upd;
    if (ccDest.hasValue()) {
      auto element = fir::getBase(exv);
      iterSpace.setElement(element);
      upd = fir::getBase(ccDest.getValue()(iterSpace));
    } else {
      auto resTy = adjustedArrayElementType(innerArg.getType());
      auto element = adjustedArrayInsert(exv, innerArg.getType());
      upd = builder.create<fir::ArrayUpdateOp>(loc, resTy, innerArg, element,
                                               iterSpace.iterVec(),
                                               destination.typeparams());
    }
    if (explicitSpace)
      explicitSpace->freeInnerContext();
    builder.create<fir::ResultOp>(loc, upd);
    builder.restoreInsertionPoint(insPt);
    return abstractArrayExtValue(iterSpace.outerResult());
  }

  /// Compute the shape of a slice.
  llvm::SmallVector<mlir::Value> computeSliceShape(mlir::Value slice) {
    llvm::SmallVector<mlir::Value> slicedShape;
    auto slOp = mlir::cast<fir::SliceOp>(slice.getDefiningOp());
    auto triples = slOp.triples();
    auto idxTy = builder.getIndexType();
    auto loc = getLoc();
    auto zero = builder.createIntegerConstant(loc, idxTy, 0);
    for (unsigned i = 0, end = triples.size(); i < end; i += 3) {
      if (!mlir::isa_and_nonnull<fir::UndefOp>(
              triples[i + 1].getDefiningOp())) {
        // (..., lb:ub:step, ...) case:  extent = max((ub-lb+step)/step, 0)
        // See Fortran 2018 9.5.3.3.2 section for more details.
        auto lb = builder.createConvert(loc, idxTy, triples[i]);
        auto ub = builder.createConvert(loc, idxTy, triples[i + 1]);
        auto step = builder.createConvert(loc, idxTy, triples[i + 2]);
        auto diff = builder.create<mlir::SubIOp>(loc, ub, lb);
        auto add = builder.create<mlir::AddIOp>(loc, diff, step);
        auto div = builder.create<mlir::SignedDivIOp>(loc, add, step);
        auto cmp = builder.create<mlir::CmpIOp>(loc, mlir::CmpIPredicate::sgt,
                                                div, zero);
        slicedShape.emplace_back(
            builder.create<mlir::SelectOp>(loc, cmp, div, zero));
      } else {
        // do nothing. `..., i, ...` case, so dimension is dropped.
      }
    }
    return slicedShape;
  }

  /// Get the shape from an ArrayLoad. The shape of the array is adjusted if the
  /// array was sliced.
  llvm::SmallVector<mlir::Value> getShape(fir::ArrayLoadOp arrayLoad) {
    if (arrayLoad.slice())
      return computeSliceShape(arrayLoad.slice());
    auto memref = arrayLoad.memref();
    if (memref.getType().isa<fir::BoxType>())
      return fir::factory::readExtents(builder, getLoc(),
                                       fir::BoxValue{memref});
    auto extents = arrayLoad.getExtents();
    return {extents.begin(), extents.end()};
  }

  /// Generate the shape of the iteration space over the array expression. The
  /// iteration space may be implicit, explicit, or both. If it is implied it is
  /// based on the destination and operand array loads, or an optional
  /// Fortran::evaluate::Shape from the front end. If the shape is explicit,
  /// this returns any implicit shape component, if it exists.
  llvm::SmallVector<mlir::Value> genIterationShape() {
    if (explicitSpace)
      return explicitImpliedShape;
    // Use the precomputed destination shape.
    if (!destShape.empty())
      return destShape;
    // Otherwise, use the destination's shape.
    if (destination)
      return getShape(destination);
    // Otherwise, use the first ArrayLoad operand shape.
    if (!arrayOperandLoads.empty())
      return getShape(arrayOperandLoads[0]);
    fir::emitFatalError(getLoc(),
                        "failed to compute the array expression shape");
  }

  /// Force the binding of an explicit symbol. This is used to bind and re-bind
  /// a concurrent control symbol to its value.
  void forceControlVariableBinding(mlir::Location loc,
                                   const Fortran::semantics::Symbol *sym,
                                   mlir::Value inducVar) {
    assert(sym && "There must be a symbol to bind");
    auto toTy = converter.genType(*sym);
    // FIXME: this should be a "per iteration" temporary.
    auto tmp = builder.createTemporary(
        loc, toTy, toStringRef(sym->name()),
        llvm::ArrayRef<mlir::NamedAttribute>{getAdaptToByRefAttr()});
    auto cast = builder.createConvert(loc, toTy, inducVar);
    builder.create<fir::StoreOp>(loc, cast, tmp);
    symMap.addSymbol(*sym, tmp, /*force=*/true);
  }

  /// Lower the mask expression `fex` associated with the concurrent control
  /// header. A concurrent control header has 0 or 1 mask expressions.
  ///
  /// The algorithm is:
  /// 1. Lower the explicit loop nest and conditionalize along the way.
  /// 2. Compute the mask values and store them into a compressed temporary.
  /// 3. Return the temporary array and the shape op that describes it.
  Fortran::lower::MaskAddrAndShape
  genExplicitMaskExpr(Fortran::lower::FrontEndExpr fex) {
    using FESymbol = Fortran::lower::ExplicitIterSpace::FrontEndSymbol;
    assert(explicitSpace);
    auto loc = getLoc();

    // Compute the shape of the temporary. The shape has 1 dimension for each
    // concurrent control. Each concurrent control instantiates a dense map of
    // logicals on the range [0 .. n), where n the extent.
    llvm::SmallVector<mlir::Value> shapeTmp;
    llvm::SmallVector<mlir::Value> lows;
    llvm::SmallVector<mlir::Value> highs;
    llvm::SmallVector<mlir::Value> steps;
    llvm::SmallVector<FESymbol> ivars;
    llvm::SmallVector<size_t> nestDepth;
    llvm::SmallVector<Fortran::lower::FrontEndExpr> maskExprs;
    auto idxTy = builder.getIndexType();
    for (const auto &spec : explicitSpace->getSpecs()) {
      for (const auto &dim : spec.first) {
        auto m1 = fir::getBase(asScalar(*std::get<1>(dim)));
        lows.push_back(builder.createConvert(loc, idxTy, m1));
        auto m2 = fir::getBase(asScalar(*std::get<2>(dim)));
        highs.push_back(builder.createConvert(loc, idxTy, m2));
        auto m3 = std::get<3>(dim)
                      ? fir::getBase(asScalar(*std::get<3>(dim)))
                      : builder.createIntegerConstant(loc, m1.getType(), 1);
        steps.push_back(builder.createConvert(loc, idxTy, m3));
        ivars.push_back(std::get<0>(dim));
        // Compute the extent in this dimension. See 11.1.7.4.2.
        auto e1 = builder.create<mlir::SubIOp>(loc, m2, m1);
        auto e2 = builder.create<mlir::AddIOp>(loc, e1, m3);
        auto e3 = builder.create<mlir::SignedDivIOp>(loc, e2, m3);
        auto zero = builder.createIntegerConstant(loc, m1.getType(), 0);
        auto cond = builder.create<mlir::CmpIOp>(loc, mlir::CmpIPredicate::sgt,
                                                 e3, zero);
        mlir::Value e4 = builder.create<mlir::SelectOp>(loc, cond, e3, zero);
        shapeTmp.push_back(builder.createConvert(loc, idxTy, e4));
      }
      maskExprs.push_back(spec.second);
      nestDepth.push_back(spec.first.size());
      if (spec.second == fex)
        break;
    }
    auto maskEleTy = converter.genType(*fex);
    auto maskTy = fir::SequenceType::get(maskEleTy, shapeTmp.size());
    auto maskRefEleTy = builder.getRefType(maskEleTy);
    auto i1Ty = builder.getI1Type();
    auto shapeOp = builder.consShape(loc, shapeTmp);
    // TODO: The temporary arrays used to cache the results of evaluating the
    // mask expressions are defined in terms of !fir.logical<n> elements. This
    // is a waste of memory resources, since all that is really needed are i1
    // elements.
    auto loadTmp = createAndLoadSomeArrayTemp(maskTy, shapeTmp);
    llvm::SmallVector<fir::DoLoopOp> loops;
    auto iterMask = maskExprs.begin();
    auto resArg = loadTmp.getResult();
    auto iterDepth = nestDepth.begin();
    auto depth = *iterDepth;
    auto iterIvars = ivars.begin();
    for (auto iter1 = lows.begin(), iter2 = highs.begin(),
              iter3 = steps.begin();
         iter2 != highs.end(); ++iter1, ++iter2, ++iter3) {
      auto lp = builder.create<fir::DoLoopOp>(loc, *iter1, *iter2, *iter3,
                                              /*unordered=*/true,
                                              /*finalCount=*/false,
                                              mlir::ValueRange{resArg});
      if (!loops.empty())
        builder.create<fir::ResultOp>(loc, lp.getResult(0));
      resArg = lp.getRegionIterArgs().front();
      builder.setInsertionPointToStart(lp.getBody());
      const Fortran::semantics::Symbol *sym = *iterIvars;
      forceControlVariableBinding(loc, sym, lp.getInductionVar());
      iterIvars++;
      // Created the appropriate number of nested loops for this header?
      if (--depth == 0) {
        ++iterDepth;
        depth = iterDepth == nestDepth.end() ? 0 : *iterDepth;
        if (Fortran::lower::FrontEndExpr mask = *iterMask++) {
          llvm::SmallVector<mlir::Value> indices;
          auto ilo = lows.begin();
          auto ist = steps.begin();
          for (auto iiv = ivars.begin(); iiv != iterIvars;
               ++iiv, ++ilo, ++ist) {
            auto ind =
                fir::getBase(symMap.lookupSymbol(*iiv).toExtendedValue());
            auto ld = builder.create<fir::LoadOp>(loc, ind);
            auto diff = builder.create<mlir::SubIOp>(
                loc, builder.createConvert(loc, idxTy, ld), *ilo);
            mlir::Value adj =
                builder.create<mlir::SignedDivIOp>(loc, diff, *ist);
            indices.push_back(adj);
          }
          if (mask == fex) {
            auto maskVal = fir::getBase(asScalar(*mask));
            auto castVal = builder.createConvert(loc, maskEleTy, maskVal);
            auto upd = builder.create<fir::ArrayUpdateOp>(
                loc, maskTy, resArg, castVal, indices,
                /*typeparams=*/llvm::None);
            builder.create<fir::ResultOp>(loc, upd.getResult());
          } else {
            auto [maskAddr, maskShape] =
                explicitSpace->getBindingWithShape(mask);
            auto ad = builder.create<fir::ArrayCoorOp>(
                loc, maskRefEleTy, maskAddr, maskShape, /*slice=*/mlir::Value{},
                indices, /*typeparams=*/llvm::None);
            auto ld = builder.create<fir::LoadOp>(loc, ad);
            auto cond = builder.createConvert(loc, i1Ty, ld);
            auto ifOp = builder.create<fir::IfOp>(
                loc, mlir::TypeRange{resArg.getType()}, cond,
                /*withElseRegion=*/true);
            builder.create<fir::ResultOp>(loc, ifOp.getResult(0));
            builder.setInsertionPointToStart(&ifOp.elseRegion().front());
            builder.create<fir::ResultOp>(loc, resArg);
            builder.setInsertionPointToStart(&ifOp.thenRegion().front());
          }
        }
      }
      loops.push_back(lp);
    }
    builder.setInsertionPointAfter(loops[0]);
    builder.create<fir::ArrayMergeStoreOp>(
        loc, loadTmp, loops[0].getResult(0), loadTmp.memref(),
        /*slice=*/mlir::Value{}, /*typeparams=*/llvm::None);
    return {loadTmp.memref(), shapeOp};
  }

  bool hasImpliedMasks() { return masks && !masks->empty(); }

  /// The explicit loop nest is built separately. It is built in user-defined
  /// order and with the concurrent header mask expressions introduced once per
  /// construct level.
  void genExplicitIterSpace(llvm::SmallVectorImpl<fir::DoLoopOp> &loops,
                            llvm::SmallVectorImpl<mlir::Value> &maskOffs,
                            mlir::Value &innerArg) {
    assert(loops.empty() && maskOffs.empty() && "must be empty");
    auto loc = getLoc();
    auto idxTy = builder.getIndexType();
    llvm::SmallVector<Fortran::lower::FrontEndExpr> lpLowers;
    llvm::SmallVector<Fortran::lower::FrontEndExpr> lpUppers;
    llvm::SmallVector<Fortran::lower::FrontEndExpr> lpSteps;
    llvm::SmallVector<const Fortran::semantics::Symbol *> ctrlBindings;
    llvm::SmallVector<size_t> nestDepth;
    llvm::SmallVector<Fortran::lower::FrontEndExpr> masks;
    for (auto &spec : explicitSpace->getSpecs()) {
      for (auto &isd : spec.first) {
        // An explicit loop has an iteration variable that has a front-end
        // type. All of the loop control values are converted to that
        // front-end type.
        lpLowers.push_back(std::get<1>(isd));
        lpUppers.push_back(std::get<2>(isd));
        lpSteps.push_back(std::get<3>(isd));
        ctrlBindings.push_back(std::get<0>(isd));
      }
      nestDepth.push_back(spec.first.size());
      masks.push_back(spec.second);
    }
    assert(lpLowers.empty() || lpLowers.size() == lpUppers.size());
    auto iterIvars = ctrlBindings.begin();
    auto iterDepth = nestDepth.begin();
    auto depth = *iterDepth;
    auto iterMask = masks.begin();
    for (auto iter1 = lpLowers.begin(), iter2 = lpUppers.begin(),
              iter3 = lpSteps.begin();
         iter1 != lpLowers.end(); ++iter1, ++iter2, ++iter3) {
      auto lo =
          builder.createConvert(loc, idxTy, fir::getBase(asScalar(**iter1)));
      auto hi =
          builder.createConvert(loc, idxTy, fir::getBase(asScalar(**iter2)));
      auto step = *iter3 ? builder.createConvert(
                               loc, idxTy, fir::getBase(asScalar(**iter3)))
                         : builder.createIntegerConstant(loc, idxTy, 1);
      auto lp = builder.create<fir::DoLoopOp>(
          loc, lo, hi, step, /*unordered=*/true,
          /*finalCount=*/false, mlir::ValueRange{innerArg});
      if (!loops.empty())
        builder.create<fir::ResultOp>(loc, lp.getResult(0));
      innerArg = lp.getRegionIterArgs().front();
      builder.setInsertionPointToStart(lp.getBody());
      forceControlVariableBinding(loc, *iterIvars, lp.getInductionVar());
      auto ind =
          fir::getBase(symMap.lookupSymbol(*iterIvars++).toExtendedValue());
      auto ld = builder.create<fir::LoadOp>(loc, ind);
      auto diff = builder.create<mlir::SubIOp>(
          loc, builder.createConvert(loc, idxTy, ld), lo);
      mlir::Value adj = builder.create<mlir::SignedDivIOp>(loc, diff, step);
      maskOffs.push_back(adj);
      // Created the appropriate number of nested loops for this header?
      if (--depth == 0) {
        // Yes, then generate the optional mask as required.
        ++iterDepth;
        depth = iterDepth == nestDepth.end() ? 0 : *iterDepth;
        if (Fortran::lower::FrontEndExpr mask = *iterMask++) {
          auto [maskAddr, maskShape] = explicitSpace->getBindingWithShape(mask);
          auto maskTy = maskAddr.getType();
          auto maskEleTy = fir::dyn_cast_ptrEleTy(maskTy);
          auto maskRefEleTy =
              builder.getRefType(fir::unwrapSequenceType(maskEleTy));
          auto ad = builder.create<fir::ArrayCoorOp>(
              loc, maskRefEleTy, maskAddr, maskShape, /*slice=*/mlir::Value{},
              maskOffs, /*typeparams=*/llvm::None);
          auto ld = builder.create<fir::LoadOp>(loc, ad);
          auto i1Ty = builder.getI1Type();
          auto cond = builder.createConvert(loc, i1Ty, ld);
          auto ifOp = builder.create<fir::IfOp>(
              loc, mlir::TypeRange{innerArg.getType()}, cond,
              /*withElseRegion=*/true);
          builder.create<fir::ResultOp>(loc, ifOp.getResult(0));
          builder.setInsertionPointToStart(&ifOp.elseRegion().front());
          builder.create<fir::ResultOp>(loc, innerArg);
          builder.setInsertionPointToStart(&ifOp.thenRegion().front());
        }
      }
      loops.push_back(lp);
    }
  }

  /// Compute the extents of the explicit iteration space. See 11.1.7.4.2.
  llvm::SmallVector<mlir::Value> genExplicitExtents() {
    llvm::SmallVector<mlir::Value> extents;
    assert(explicitSpace && "must be a FORALL context");
    auto loc = getLoc();
    auto idxTy = builder.getIndexType();
    for (const auto &spec : explicitSpace->getSpecs()) {
      for (const auto &dim : spec.first) {
        auto m1 = fir::getBase(asScalar(*std::get<1>(dim)));
        auto m2 = fir::getBase(asScalar(*std::get<2>(dim)));
        auto m3 = std::get<3>(dim)
                      ? fir::getBase(asScalar(*std::get<3>(dim)))
                      : builder.createIntegerConstant(loc, m1.getType(), 1);
        auto e1 = builder.create<mlir::SubIOp>(loc, m2, m1);
        auto e2 = builder.create<mlir::AddIOp>(loc, e1, m3);
        auto e3 = builder.create<mlir::SignedDivIOp>(loc, e2, m3);
        auto zero = builder.createIntegerConstant(loc, m1.getType(), 0);
        auto cond = builder.create<mlir::CmpIOp>(loc, mlir::CmpIPredicate::sgt,
                                                 e3, zero);
        auto e4 = builder.create<mlir::SelectOp>(loc, cond, e3, zero);
        extents.push_back(builder.createConvert(loc, idxTy, e4));
      }
    }
    return extents;
  }

  std::pair<llvm::SmallVector<mlir::Value>, fir::DoLoopOp>
  genExplicitLoopsForWhereMasks(fir::ArrayLoadOp resLd) {
    using FESymbol = Fortran::lower::ExplicitIterSpace::FrontEndSymbol;
    assert(explicitSpace && "must be a FORALL context");
    auto loc = getLoc();
    auto idxTy = builder.getIndexType();
    auto i1Ty = builder.getI1Type();
    llvm::SmallVector<mlir::Value> indices;
    llvm::SmallVector<mlir::Value> lows;
    llvm::SmallVector<mlir::Value> highs;
    llvm::SmallVector<mlir::Value> steps;
    llvm::SmallVector<FESymbol> ivars;
    llvm::SmallVector<size_t> nestDepth;
    llvm::SmallVector<Fortran::lower::FrontEndExpr> maskExprs;
    for (const auto &spec : explicitSpace->getSpecs()) {
      for (const auto &dim : spec.first) {
        auto m1 = fir::getBase(asScalar(*std::get<1>(dim)));
        lows.push_back(builder.createConvert(loc, idxTy, m1));
        auto m2 = fir::getBase(asScalar(*std::get<2>(dim)));
        highs.push_back(builder.createConvert(loc, idxTy, m2));
        auto m3 = std::get<3>(dim)
                      ? fir::getBase(asScalar(*std::get<3>(dim)))
                      : builder.createIntegerConstant(loc, m1.getType(), 1);
        steps.push_back(builder.createConvert(loc, idxTy, m3));
        ivars.push_back(std::get<0>(dim));
      }
      maskExprs.push_back(spec.second);
      nestDepth.push_back(spec.first.size());
    }

    llvm::SmallVector<fir::DoLoopOp> loops;
    auto iterMask = maskExprs.begin();
    auto resArg = resLd.getResult();
    auto iterDepth = nestDepth.begin();
    auto depth = *iterDepth;
    auto iterIvars = ivars.begin();
    for (auto iter1 = lows.begin(), iter2 = highs.begin(),
              iter3 = steps.begin();
         iter2 != highs.end(); ++iter1, ++iter2, ++iter3) {
      auto lp = builder.create<fir::DoLoopOp>(loc, *iter1, *iter2, *iter3,
                                              /*unordered=*/true,
                                              /*finalCount=*/false,
                                              mlir::ValueRange{resArg});
      if (!loops.empty())
        builder.create<fir::ResultOp>(loc, lp.getResult(0));
      resArg = lp.getRegionIterArgs().front();
      builder.setInsertionPointToStart(lp.getBody());
      const Fortran::semantics::Symbol *sym = *iterIvars;
      forceControlVariableBinding(loc, sym, lp.getInductionVar());
      iterIvars++;
      // Created the appropriate number of nested loops for this header?
      if (--depth == 0) {
        ++iterDepth;
        depth = iterDepth == nestDepth.end() ? 0 : *iterDepth;
        // Build the normalized indices for the explict part of the mask.
        auto ilo = lows.begin();
        auto ist = steps.begin();
        for (auto iiv = ivars.begin(); iiv != iterIvars; ++iiv, ++ilo, ++ist) {
          auto ind = fir::getBase(symMap.lookupSymbol(*iiv).toExtendedValue());
          auto ld = builder.create<fir::LoadOp>(loc, ind);
          auto diff = builder.create<mlir::SubIOp>(
              loc, builder.createConvert(loc, idxTy, ld), *ilo);
          mlir::Value adj = builder.create<mlir::SignedDivIOp>(loc, diff, *ist);
          indices.push_back(adj);
        }
        // If the FORALL has a mask, it was already computed. Generate the
        // interleaved conditionals to evaluate the WHERE masks correctly.
        if (Fortran::lower::FrontEndExpr mask = *iterMask++) {
          auto [maskAddr, maskShape] = explicitSpace->getBindingWithShape(mask);
          auto ad = builder.create<fir::ArrayCoorOp>(
              loc, i1Ty, maskAddr, maskShape, /*slice=*/mlir::Value{}, indices,
              /*typeparams=*/llvm::None);
          auto ld = builder.create<fir::LoadOp>(loc, ad);
          auto cond = builder.createConvert(loc, i1Ty, ld);
          auto ifOp = builder.create<fir::IfOp>(
              loc, mlir::TypeRange{resArg.getType()}, cond,
              /*withElseRegion=*/true);
          builder.create<fir::ResultOp>(loc, ifOp.getResult(0));
          builder.setInsertionPointToStart(&ifOp.elseRegion().front());
          builder.create<fir::ResultOp>(loc, resArg);
          // The context continues inside an explicit mask that's true.
          builder.setInsertionPointToStart(&ifOp.thenRegion().front());
        }
      }
      loops.push_back(lp);
    }
    return {indices, loops[0]};
  }

  void genMasks() {
    auto loc = getLoc();
    // Lower explicit mask expressions, if any.
    if (explicitSpace) {
      // Explicit masks are expressions which must be lowered in a copy of the
      // explicit space.
      for (const auto *e : explicitSpace->getMasks())
        if (e && !explicitSpace->isLowered(e)) {
          auto le = genExplicitMaskExpr(e);
          explicitSpace->bind(e, le);
        }

      if (hasImpliedMasks()) {
        for (const auto *e : masks->getExprs())
          if (e && !masks->isLowered(e)) {
            auto extents = genExplicitExtents();
            extents.append(explicitImpliedShape.rbegin(),
                           explicitImpliedShape.rend());
            // Allocate a temporary to cache the mask results.
            auto tmpShape = builder.consShape(loc, extents);
            auto tmp = createAndLoadSomeArrayTemp(
                fir::SequenceType::get(
                    llvm::SmallVector<fir::SequenceType::Extent>(
                        extents.size(), fir::SequenceType::getUnknownExtent()),
                    builder.getI1Type()),
                extents);
            // Build the explicit loops. They are outermost.
            auto [indices, loop0] = genExplicitLoopsForWhereMasks(tmp);
            // Evaluate like any other nested array expression.
            ArrayExprLowering ael{converter, masks->stmtContext(), symMap,
                                  ConstituentSemantics::ProjectedCopyInCopyOut};
            ael.lowerArrayAssignment(tmp, *e, indices, explicitImpliedShape);
            masks->bind(e, tmp.memref(), tmpShape);
            builder.setInsertionPointAfter(loop0);
            builder.create<fir::ArrayMergeStoreOp>(loc, tmp, loop0.getResult(0),
                                                   tmp.memref(), tmp.slice(),
                                                   tmp.typeparams());
          }
      }
      return;
    }

    // Lower the mask expressions, if any.
    if (hasImpliedMasks()) {
      // Mask expressions are array expressions too.
      for (const auto *e : masks->getExprs())
        if (e && !masks->isLowered(e)) {
          auto optShape =
              Fortran::evaluate::GetShape(converter.getFoldingContext(), *e);
          auto tmp = Fortran::lower::createSomeArrayTempValue(
              converter, optShape, *e, symMap, stmtCtx);
          auto shape = builder.createShape(loc, tmp);
          masks->bind(e, fir::getBase(tmp), shape);
        }
    }
  }

  /// Build the iteration space into which the array expression will be lowered.
  /// The resultType is used to create a temporary, if needed.
  std::pair<IterationSpace, mlir::OpBuilder::InsertPoint>
  genIterSpace(mlir::Type resultType) {
    auto loc = getLoc();

    // Generate any mask expressions, as necessary. This is the compute step
    // that creates the effective masks. See 10.2.3.2 in particular.
    genMasks();

    auto shape = genIterationShape();
    if (!destination) {
      // Allocate storage for the result if it is not already provided.
      destination = createAndLoadSomeArrayTemp(resultType, shape);
    }

    auto idxTy = builder.getIndexType();
    auto one = builder.createIntegerConstant(loc, idxTy, 1);
    auto zero = builder.createIntegerConstant(loc, idxTy, 0);
    llvm::SmallVector<mlir::Value> loopUppers;

    // Convert any implied shape to closed interval form. The fir.do_loop will
    // run from 0 to `extent - 1` inclusive.
    for (auto extent : shape) {
      auto up = builder.create<mlir::SubIOp>(loc, extent, one);
      loopUppers.push_back(up);
    }

    // Iteration space is created with outermost columns, innermost rows
    mlir::Value innerArg = destination.getResult();
    llvm::SmallVector<fir::DoLoopOp> loops;
    llvm::SmallVector<mlir::Value> explicitOffsets;
    // FORALL loops are outermost.
    if (explicitSpace)
      genExplicitIterSpace(loops, explicitOffsets, innerArg);

    // Now handle the implicit loops.
    const auto loopFirst = loops.size();
    const auto loopDepth = loopUppers.size();
    llvm::SmallVector<mlir::Value> ivars;
    if (loopDepth > 0) {
      for (auto i : llvm::enumerate(llvm::reverse(loopUppers))) {
        if (i.index() > 0) {
          assert(!loops.empty());
          builder.setInsertionPointToStart(loops.back().getBody());
        }
        auto loop = builder.create<fir::DoLoopOp>(
            loc, zero, i.value(), one, /*unordered=*/true,
            /*finalCount=*/false, mlir::ValueRange{innerArg});
        innerArg = loop.getRegionIterArgs().front();
        ivars.push_back(loop.getInductionVar());
        loops.push_back(loop);
      }
      // Add the fir.result for all loops except the innermost one. We must also
      // terminate the innermost explicit bounds loop here as well.
      for (std::remove_const_t<decltype(loopFirst)> i =
               loopFirst ? loopFirst - 1 : 0;
           i + 1 < loopFirst + loopDepth; ++i) {
        builder.setInsertionPointToEnd(loops[i].getBody());
        builder.create<fir::ResultOp>(loc, loops[i + 1].getResult(0));
      }

      // Move insertion point to the start of the innermost loop in the nest.
      builder.setInsertionPointToStart(loops.back().getBody());
    }
    // Set `afterLoopNest` to just after the entire loop nest.
    auto currPt = builder.saveInsertionPoint();
    builder.setInsertionPointAfter(loops[0]);
    auto afterLoopNest = builder.saveInsertionPoint();
    builder.restoreInsertionPoint(currPt);

    // Put the implicit loop variables in row to column order to match FIR's
    // Ops. (The loops were constructed from outermost column to innermost row.)
    mlir::Value outerRes = loops[0].getResult(0);
    IterationSpace iters{innerArg, outerRes, llvm::reverse(ivars)};

    // Generate the mask conditional structure, if there are masks. Unlike the
    // explicit masks, which are interleaved, these mask expression appear in
    // the innermost loop.
    if (hasImpliedMasks()) {
      auto prependAsNeeded = [&](auto &&indices) {
        llvm::SmallVector<mlir::Value> result;
        if (explicitSpace)
          result.append(explicitOffsets);
        result.append(indices.begin(), indices.end());
        return result;
      };
      auto genCond = [&](Fortran::lower::MaskAddrAndShape &&mask,
                         IterSpace iters) {
        auto tmp = mask.first;
        auto shape = mask.second;
        auto arrTy = fir::dyn_cast_ptrOrBoxEleTy(tmp.getType());
        auto eleTy = arrTy.cast<fir::SequenceType>().getEleTy();
        auto eleRefTy = builder.getRefType(eleTy);
        auto i1Ty = builder.getI1Type();
        // Adjust indices for any shift of the origin of the array.
        auto indexes = prependAsNeeded(fir::factory::originateIndices(
            loc, builder, tmp.getType(), shape, iters.iterVec()));
        auto addr = builder.create<fir::ArrayCoorOp>(
            loc, eleRefTy, tmp, shape, /*slice=*/mlir::Value{}, indexes,
            /*typeParams=*/llvm::None);
        auto load = builder.create<fir::LoadOp>(loc, addr);
        return builder.createConvert(loc, i1Ty, load);
      };

      // Handle the negated conditions. See 10.2.3.2p4 as to why this control
      // structure is produced.
      auto maskExprs = masks->getExprs();
      const auto size = maskExprs.size() - 1;
      for (std::remove_const_t<decltype(size)> i = 0; i < size; ++i) {
        auto ifOp = builder.create<fir::IfOp>(
            loc, mlir::TypeRange{innerArg.getType()},
            fir::getBase(
                genCond(masks->getBindingWithShape(maskExprs[i]), iters)),
            /*withElseRegion=*/true);
        builder.create<fir::ResultOp>(loc, ifOp.getResult(0));
        builder.setInsertionPointToStart(&ifOp.thenRegion().front());
        builder.create<fir::ResultOp>(loc, innerArg);
        builder.setInsertionPointToStart(&ifOp.elseRegion().front());
      }

      // The last condition is either non-negated or unconditionally negated.
      if (maskExprs[size]) {
        auto ifOp = builder.create<fir::IfOp>(
            loc, mlir::TypeRange{innerArg.getType()},
            fir::getBase(
                genCond(masks->getBindingWithShape(maskExprs[size]), iters)),
            /*withElseRegion=*/true);
        builder.create<fir::ResultOp>(loc, ifOp.getResult(0));
        builder.setInsertionPointToStart(&ifOp.elseRegion().front());
        builder.create<fir::ResultOp>(loc, innerArg);
        builder.setInsertionPointToStart(&ifOp.thenRegion().front());
      } else {
        // do nothing
      }
    }

    // We're ready to lower the body (an assignment statement) for this context
    // of loop nests at this point.
    return {iters, afterLoopNest};
  }

  fir::ArrayLoadOp
  createAndLoadSomeArrayTemp(mlir::Type type,
                             llvm::ArrayRef<mlir::Value> shape) {
    auto seqTy = type.dyn_cast<fir::SequenceType>();
    assert(seqTy && "must be an array");
    auto loc = getLoc();
    // TODO: Need to thread the length parameters here. For character, they may
    // differ from the operands length (e.g concatenation). So the array loads
    // type parameters are not enough.
    if (auto charTy = seqTy.getEleTy().dyn_cast<fir::CharacterType>())
      if (charTy.hasDynamicLen())
        TODO(loc, "character array expression temp with dynamic length");
    if (auto recTy = seqTy.getEleTy().dyn_cast<fir::RecordType>())
      if (recTy.getNumLenParams() > 0)
        TODO(loc, "derived type array expression temp with length parameters");
    mlir::Value temp = seqTy.hasConstantShape()
                           ? builder.create<fir::AllocMemOp>(loc, type)
                           : builder.create<fir::AllocMemOp>(
                                 loc, type, ".array.expr", llvm::None, shape);
    auto *bldr = &converter.getFirOpBuilder();
    stmtCtx.attachCleanup([=]() { bldr->create<fir::FreeMemOp>(loc, temp); });
    auto shapeOp = genShapeOp(shape);
    return builder.create<fir::ArrayLoadOp>(
        loc, seqTy, temp, shapeOp, /*slice=*/mlir::Value{}, llvm::None);
  }

  fir::ShapeOp genShapeOp(llvm::ArrayRef<mlir::Value> shape) {
    auto loc = getLoc();
    auto idxTy = builder.getIndexType();
    llvm::SmallVector<mlir::Value> idxShape;
    for (auto s : shape)
      idxShape.push_back(builder.createConvert(loc, idxTy, s));
    auto shapeTy = fir::ShapeType::get(builder.getContext(), idxShape.size());
    return builder.create<fir::ShapeOp>(loc, shapeTy, idxShape);
  }

  //===--------------------------------------------------------------------===//
  // Expression traversal and lowering.
  //===--------------------------------------------------------------------===//

  // Lower the expression in a scalar context.
  template <typename A>
  ExtValue asScalar(const A &x) {
    return ScalarExprLowering{getLoc(), converter, symMap, stmtCtx}.genval(x);
  }

  // Lower the expression in a scalar context to a (boxed) reference.
  template <typename A>
  ExtValue asScalarRef(const A &x) {
    return ScalarExprLowering{getLoc(), converter, symMap, stmtCtx}.gen(x);
  }

  // An expression with non-zero rank is an array expression.
  template <typename A>
  bool isArray(const A &x) const {
    return x.Rank() != 0 || explicitSpace;
  }

  // Attribute for an alloca that is a trivial adaptor for converting a value to
  // pass-by-ref semantics for a VALUE parameter. The optimizer may be able to
  // eliminate these.
  mlir::NamedAttribute getAdaptToByRefAttr() {
    return {mlir::Identifier::get("adapt.valuebyref", builder.getContext()),
            builder.getUnitAttr()};
  }

  // A procedure reference to a Fortran elemental intrinsic procedure.
  CC genElementalIntrinsicProcRef(
      const Fortran::evaluate::ProcedureRef &procRef,
      llvm::Optional<mlir::Type> retTy,
      const Fortran::evaluate::SpecificIntrinsic &intrinsic) {
    llvm::SmallVector<CC> operands;
    llvm::StringRef name = intrinsic.name;
    const auto *argLowering =
        Fortran::lower::getIntrinsicArgumentLowering(name);
    auto loc = getLoc();
    for (const auto &[arg, dummy] :
         llvm::zip(procRef.arguments(),
                   intrinsic.characteristics.value().dummyArguments)) {
      auto *expr = Fortran::evaluate::UnwrapExpr<
          Fortran::evaluate::Expr<Fortran::evaluate::SomeType>>(arg);
      if (!expr) {
        // Absent optional.
        operands.emplace_back([=](IterSpace) { return mlir::Value{}; });
      } else if (!argLowering) {
        // No argument lowering instruction, lower by value.
        PushSemantics(ConstituentSemantics::RefTransparent);
        auto lambda = genarr(*expr);
        operands.emplace_back([=](IterSpace iters) { return lambda(iters); });
      } else {
        // Ad-hoc argument lowering handling.
        switch (Fortran::lower::lowerIntrinsicArgumentAs(getLoc(), *argLowering,
                                                         dummy.name)) {
        case Fortran::lower::LowerIntrinsicArgAs::Value: {
          PushSemantics(ConstituentSemantics::RefTransparent);
          auto lambda = genarr(*expr);
          operands.emplace_back([=](IterSpace iters) { return lambda(iters); });
        } break;
        case Fortran::lower::LowerIntrinsicArgAs::Addr: {
          // Note: assume does not have Fortran VALUE attribute semantics.
          PushSemantics(ConstituentSemantics::RefOpaque);
          auto lambda = genarr(*expr);
          operands.emplace_back([=](IterSpace iters) { return lambda(iters); });
        } break;
        case Fortran::lower::LowerIntrinsicArgAs::Box: {
          PushSemantics(ConstituentSemantics::RefOpaque);
          auto lambda = genarr(*expr);
          operands.emplace_back([=](IterSpace iters) {
            return builder.createBox(loc, lambda(iters));
          });
        } break;
        case Fortran::lower::LowerIntrinsicArgAs::Inquired:
          TODO(loc, "intrinsic function with inquired argument");
          break;
        }
      }
    }

    // Let the intrinsic library lower the intrinsic procedure call
    return [=](IterSpace iters) {
      llvm::SmallVector<ExtValue> args;
      for (const auto &cc : operands)
        args.push_back(cc(iters));
      return Fortran::lower::genIntrinsicCall(builder, loc, name, retTy, args,
                                              stmtCtx);
    };
  }

  // A procedure reference to a user-defined elemental procedure.
  CC genElementalUserDefinedProcRef(
      const Fortran::evaluate::ProcedureRef &procRef,
      llvm::Optional<mlir::Type> retTy) {
    using PassBy = Fortran::lower::CallerInterface::PassEntityBy;

    Fortran::lower::CallerInterface caller(procRef, converter);
    llvm::SmallVector<CC> operands(caller.getNumFIRArguments());
    auto loc = getLoc();
    auto callSiteType = caller.genFunctionType();
    for (const auto &arg : caller.getPassedArguments()) {
      const auto *actual = arg.entity;
      auto argTy = callSiteType.getInput(arg.firArgument);
      if (!actual) {
        // Optional dummy argument for which there is no actual argument.
        auto absent = builder.create<fir::AbsentOp>(loc, argTy);
        operands[arg.firArgument] = [=](IterSpace) { return absent; };
        continue;
      }
      const auto *expr = actual->UnwrapExpr();
      if (!expr)
        TODO(loc, "assumed type actual argument lowering");

      LLVM_DEBUG(expr->AsFortran(llvm::dbgs()
                                 << "argument: " << arg.firArgument << " = [")
                 << "]\n");
      switch (arg.passBy) {
      case PassBy::Value: {
        // True pass-by-value semantics.
        PushSemantics(ConstituentSemantics::RefTransparent);
        auto lambda = genarr(*expr);
        operands[arg.firArgument] = [=](IterSpace iters) {
          return lambda(iters);
        };
      } break;
      case PassBy::BaseAddressValueAttribute: {
        // VALUE attribute or pass-by-reference to a copy semantics. (byval*)
        if (isArray(*expr)) {
          PushSemantics(ConstituentSemantics::ByValueArg);
          auto lambda = genarr(*expr);
          operands[arg.firArgument] = [=](IterSpace iters) {
            return lambda(iters);
          };
        } else {
          // Store scalar value in a temp to fulfill VALUE attribute.
          auto val = fir::getBase(asScalar(*expr));
          auto temp = builder.createTemporary(
              loc, val.getType(),
              llvm::ArrayRef<mlir::NamedAttribute>{getAdaptToByRefAttr()});
          builder.create<fir::StoreOp>(loc, val, temp);
          operands[arg.firArgument] = [=](IterSpace iters) -> ExtValue {
            return temp;
          };
        }
      } break;
      case PassBy::BaseAddress: {
        if (isArray(*expr)) {
          PushSemantics(ConstituentSemantics::RefOpaque);
          auto lambda = genarr(*expr);
          operands[arg.firArgument] = [=](IterSpace iters) {
            return lambda(iters);
          };
        } else {
          auto exv = asScalarRef(*expr);
          operands[arg.firArgument] = [=](IterSpace iters) { return exv; };
        }
      } break;
      case PassBy::CharBoxValueAttribute:
        TODO(loc, "CHARACTER, VALUE");
        break;
      case PassBy::BoxChar:
        TODO(loc, "CHARACTER");
        break;
      case PassBy::AddressAndLength:
        TODO(loc, "address and length argument");
        break;
      case PassBy::Box:
      case PassBy::MutableBox:
        // See C15100 and C15101
        fir::emitFatalError(loc, "cannot be POINTER, ALLOCATABLE");
      }
    }

    if (caller.getIfIndirectCallSymbol())
      fir::emitFatalError(loc, "cannot be indirect call");
    auto funcSym = builder.getSymbolRefAttr(caller.getMangledName());
    auto resTys = caller.getFuncOp().getType().getResults();
    if (caller.getFuncOp().getType().getResults() !=
        caller.genFunctionType().getResults())
      fir::emitFatalError(loc, "type mismatch on declared function");
    return [=](IterSpace iters) -> ExtValue {
      llvm::SmallVector<mlir::Value> args;
      for (const auto &cc : operands)
        args.push_back(fir::getBase(cc(iters)));
      auto call = builder.create<fir::CallOp>(loc, resTys, funcSym, args);
      return call.getResult(0);
    };
  }

  /// Generate a procedure reference.
  CC genProcRef(const Fortran::evaluate::ProcedureRef &procRef,
                llvm::Optional<mlir::Type> retTy) {
    auto loc = getLoc();
    if (procRef.IsElemental()) {
      if (const auto *intrin = procRef.proc().GetSpecificIntrinsic()) {
        // Elemental intrinsic call.
        // The intrinsic procedure is called once per element of the array.
        return genElementalIntrinsicProcRef(procRef, retTy, *intrin);
      }
      if (ScalarExprLowering::isStatementFunctionCall(procRef))
        fir::emitFatalError(loc, "statement function cannot be elemental");

      // Elemental call.
      // The procedure is called once per element of the array argument(s).
      return genElementalUserDefinedProcRef(procRef, retTy);
    }

    // Transformational call.
    // The procedure is called once and produces a value of rank > 0.
    if (const auto *intrinsic = procRef.proc().GetSpecificIntrinsic()) {
      if (explicitSpace)
        return [=](IterSpace iters) {
          return ScalarExprLowering{loc, converter, symMap,
                                    explicitSpace->markInnerContext()}
              .genIntrinsicRef(procRef, *intrinsic, retTy);
        };
      auto resultBox =
          ScalarExprLowering{loc, converter, symMap, stmtCtx}.genIntrinsicRef(
              procRef, *intrinsic, retTy);
      return genarr(resultBox);
    }

    if (explicitSpace) {
      // Call from within an explicit space. Arguments to the call may contain
      // references to the control variables, so the call is done inside the
      // loop nest.
      return [=](IterSpace iters) {
        return ScalarExprLowering{loc, converter, symMap,
                                  explicitSpace->markInnerContext()}
            .genProcedureRef(procRef, retTy);
      };
    }

    // In the default case, the call can be hoisted out of the loop nest. Apply
    // the iterations to the result, which may be an array value.
    auto resultBox =
        ScalarExprLowering{loc, converter, symMap, stmtCtx}.genProcedureRef(
            procRef, retTy);
    return genarr(resultBox);
  }

  CC genarr(const Fortran::evaluate::ProcedureDesignator &) {
    TODO(getLoc(), "procedure designator");
  }
  CC genarr(const Fortran::evaluate::ProcedureRef &x) {
    if (x.hasAlternateReturns())
      fir::emitFatalError(getLoc(),
                          "array procedure reference with alt-return");
    return genProcRef(x, llvm::None);
  }
  template <typename A>
  CC genscl(const A &x) {
    auto result = asScalar(x);
    return [=](IterSpace) { return result; };
  }
  template <typename A, typename = std::enable_if_t<Fortran::common::HasMember<
                            A, Fortran::evaluate::TypelessExpression>>>
  CC genarr(const A &x) {
    return genscl(x);
  }

  template <typename A>
  CC genarr(const Fortran::evaluate::Expr<A> &x) {
    if (isArray(x))
      return std::visit([&](const auto &e) { return genarr(e); }, x.u);
    return genscl(x);
  }

  // Converting a value of memory bound type requires creating a temp and
  // copying the value.
  static ExtValue convertAdjustedType(fir::FirOpBuilder &builder,
                                      mlir::Location loc, mlir::Type toType,
                                      const ExtValue &exv) {
    auto lenFromBufferType = [&](mlir::Type ty) {
      return builder.create<mlir::ConstantIndexOp>(
          loc, fir::dyn_cast_ptrEleTy(ty).cast<fir::CharacterType>().getLen());
    };
    return exv.match(
        [&](const fir::CharBoxValue &cb) -> ExtValue {
          auto typeParams = fir::getTypeParams(exv);
          auto len = typeParams.size() > 0
                         ? typeParams[0]
                         : lenFromBufferType(cb.getBuffer().getType());
          auto mem =
              builder.create<fir::AllocaOp>(loc, toType, mlir::ValueRange{len});
          fir::CharBoxValue result(mem, len);
          fir::factory::CharacterExprHelper{builder, loc}.createAssign(
              ExtValue{result}, exv);
          return result;
        },
        [&](const auto &) -> ExtValue {
          fir::emitFatalError(loc, "convert on adjusted extended value");
        });
  }
  template <Fortran::common::TypeCategory TC1, int KIND,
            Fortran::common::TypeCategory TC2>
  CC genarr(const Fortran::evaluate::Convert<Fortran::evaluate::Type<TC1, KIND>,
                                             TC2> &x) {
    assert(isArray(x));
    auto loc = getLoc();
    auto lambda = genarr(x.left());
    auto ty = converter.genType(TC1, KIND);
    return [=](IterSpace iters) -> ExtValue {
      auto exv = lambda(iters);
      auto val = fir::getBase(exv);
      if (elementTypeWasAdjusted(val.getType()))
        return convertAdjustedType(builder, loc, ty, exv);
      return builder.createConvert(loc, ty, val);
    };
  }

  template <int KIND>
  CC genarr(const Fortran::evaluate::ComplexComponent<KIND> &x) {
    auto loc = getLoc();
    auto lambda = genarr(x.left());
    auto isImagPart = x.isImaginaryPart;
    return [=](IterSpace iters) -> ExtValue {
      auto lhs = fir::getBase(lambda(iters));
      return fir::factory::ComplexExprHelper{builder, loc}.extractComplexPart(
          lhs, isImagPart);
    };
  }
  template <Fortran::common::TypeCategory TC, int KIND>
  CC genarr(
      const Fortran::evaluate::Parentheses<Fortran::evaluate::Type<TC, KIND>>
          &x) {
    auto loc = getLoc();
    auto f = genarr(x.left());
    return [=](IterSpace iters) -> ExtValue {
      auto val = f(iters);
      auto base = fir::getBase(val);
      auto newBase =
          builder.create<fir::NoReassocOp>(loc, base.getType(), base);
      return fir::substBase(val, newBase);
    };
  }
  template <int KIND>
  CC genarr(const Fortran::evaluate::Negate<Fortran::evaluate::Type<
                Fortran::common::TypeCategory::Integer, KIND>> &x) {
    auto loc = getLoc();
    auto f = genarr(x.left());
    return [=](IterSpace iters) -> ExtValue {
      auto val = fir::getBase(f(iters));
      auto ty = converter.genType(Fortran::common::TypeCategory::Integer, KIND);
      auto zero = builder.createIntegerConstant(loc, ty, 0);
      return builder.create<mlir::SubIOp>(loc, zero, val);
    };
  }
  template <int KIND>
  CC genarr(const Fortran::evaluate::Negate<Fortran::evaluate::Type<
                Fortran::common::TypeCategory::Real, KIND>> &x) {
    auto loc = getLoc();
    auto f = genarr(x.left());
    return [=](IterSpace iters) -> ExtValue {
      return builder.create<mlir::NegFOp>(loc, fir::getBase(f(iters)));
    };
  }
  template <int KIND>
  CC genarr(const Fortran::evaluate::Negate<Fortran::evaluate::Type<
                Fortran::common::TypeCategory::Complex, KIND>> &x) {
    auto loc = getLoc();
    auto f = genarr(x.left());
    return [=](IterSpace iters) -> ExtValue {
      return builder.create<fir::NegcOp>(loc, fir::getBase(f(iters)));
    };
  }

  //===--------------------------------------------------------------------===//
  // Binary elemental ops
  //===--------------------------------------------------------------------===//

  template <typename OP, typename A>
  CC createBinaryOp(const A &evEx) {
    auto loc = getLoc();
    auto lambda = genarr(evEx.left());
    auto rf = genarr(evEx.right());
    return [=](IterSpace iters) -> ExtValue {
      auto left = fir::getBase(lambda(iters));
      auto right = fir::getBase(rf(iters));
      return builder.create<OP>(loc, left, right);
    };
  }

#undef GENBIN
#define GENBIN(GenBinEvOp, GenBinTyCat, GenBinFirOp)                           \
  template <int KIND>                                                          \
  CC genarr(const Fortran::evaluate::GenBinEvOp<Fortran::evaluate::Type<       \
                Fortran::common::TypeCategory::GenBinTyCat, KIND>> &x) {       \
    return createBinaryOp<GenBinFirOp>(x);                                     \
  }

  GENBIN(Add, Integer, mlir::AddIOp)
  GENBIN(Add, Real, mlir::AddFOp)
  GENBIN(Add, Complex, fir::AddcOp)
  GENBIN(Subtract, Integer, mlir::SubIOp)
  GENBIN(Subtract, Real, mlir::SubFOp)
  GENBIN(Subtract, Complex, fir::SubcOp)
  GENBIN(Multiply, Integer, mlir::MulIOp)
  GENBIN(Multiply, Real, mlir::MulFOp)
  GENBIN(Multiply, Complex, fir::MulcOp)
  GENBIN(Divide, Integer, mlir::SignedDivIOp)
  GENBIN(Divide, Real, mlir::DivFOp)
  GENBIN(Divide, Complex, fir::DivcOp)

  template <Fortran::common::TypeCategory TC, int KIND>
  CC genarr(
      const Fortran::evaluate::Power<Fortran::evaluate::Type<TC, KIND>> &x) {
    auto loc = getLoc();
    auto ty = converter.genType(TC, KIND);
    auto lf = genarr(x.left());
    auto rf = genarr(x.right());
    return [=](IterSpace iters) -> ExtValue {
      auto lhs = fir::getBase(lf(iters));
      auto rhs = fir::getBase(rf(iters));
      return Fortran::lower::genPow(builder, loc, ty, lhs, rhs);
    };
  }
  template <Fortran::common::TypeCategory TC, int KIND>
  CC genarr(
      const Fortran::evaluate::Extremum<Fortran::evaluate::Type<TC, KIND>> &x) {
    auto loc = getLoc();
    auto lf = genarr(x.left());
    auto rf = genarr(x.right());
    switch (x.ordering) {
    case Fortran::evaluate::Ordering::Greater:
      return [=](IterSpace iters) -> ExtValue {
        auto lhs = fir::getBase(lf(iters));
        auto rhs = fir::getBase(rf(iters));
        return Fortran::lower::genMax(builder, loc,
                                      llvm::ArrayRef<mlir::Value>{lhs, rhs});
      };
    case Fortran::evaluate::Ordering::Less:
      return [=](IterSpace iters) -> ExtValue {
        auto lhs = fir::getBase(lf(iters));
        auto rhs = fir::getBase(rf(iters));
        return Fortran::lower::genMin(builder, loc,
                                      llvm::ArrayRef<mlir::Value>{lhs, rhs});
      };
    case Fortran::evaluate::Ordering::Equal:
      llvm_unreachable("Equal is not a valid ordering in this context");
    }
    llvm_unreachable("unknown ordering");
  }
  template <Fortran::common::TypeCategory TC, int KIND>
  CC genarr(
      const Fortran::evaluate::RealToIntPower<Fortran::evaluate::Type<TC, KIND>>
          &x) {
    auto loc = getLoc();
    auto ty = converter.genType(TC, KIND);
    auto lf = genarr(x.left());
    auto rf = genarr(x.right());
    return [=](IterSpace iters) {
      auto lhs = fir::getBase(lf(iters));
      auto rhs = fir::getBase(rf(iters));
      return Fortran::lower::genPow(builder, loc, ty, lhs, rhs);
    };
  }
  template <int KIND>
  CC genarr(const Fortran::evaluate::ComplexConstructor<KIND> &x) {
    auto loc = getLoc();
    auto lf = genarr(x.left());
    auto rf = genarr(x.right());
    return [=](IterSpace iters) -> ExtValue {
      auto lhs = fir::getBase(lf(iters));
      auto rhs = fir::getBase(rf(iters));
      return fir::factory::ComplexExprHelper{builder, loc}.createComplex(
          KIND, lhs, rhs);
    };
  }

  /// Fortran's concatenation operator `//`.
  template <int KIND>
  CC genarr(const Fortran::evaluate::Concat<KIND> &x) {
    auto loc = getLoc();
    auto lf = genarr(x.left());
    auto rf = genarr(x.right());
    return [=](IterSpace iters) -> ExtValue {
      auto lhs = lf(iters);
      auto rhs = rf(iters);
      auto *lchr = lhs.getCharBox();
      auto *rchr = rhs.getCharBox();
      if (lchr && rchr) {
        return fir::factory::CharacterExprHelper{builder, loc}
            .createConcatenate(*lchr, *rchr);
      }
      TODO(loc, "concat on unexpected extended values");
      return mlir::Value{};
    };
  }

  template <int KIND>
  CC genarr(const Fortran::evaluate::SetLength<KIND> &x) {
    auto lf = genarr(x.left());
    auto rhs = fir::getBase(asScalar(x.right()));
    return [=](IterSpace iters) -> ExtValue {
      auto lhs = fir::getBase(lf(iters));
      return fir::CharBoxValue{lhs, rhs};
    };
  }

  template <typename A>
  CC genarr(const Fortran::evaluate::Constant<A> &x) {
    if (explicitSpace && x.Rank() == 0)
      return genscl(x);
    auto loc = getLoc();
    auto idxTy = builder.getIndexType();
    auto arrTy = converter.genType(toEvExpr(x));
    std::string globalName = Fortran::lower::mangle::mangleArrayLiteral(x);
    auto global = builder.getNamedGlobal(globalName);
    if (!global) {
      global = builder.createGlobalConstant(
          loc, arrTy, globalName,
          [&](fir::FirOpBuilder &builder) {
            Fortran::lower::StatementContext stmtCtx;
            auto result = Fortran::lower::createSomeInitializerExpression(
                loc, converter, toEvExpr(x), symMap, stmtCtx);
            auto castTo =
                builder.createConvert(loc, arrTy, fir::getBase(result));
            builder.create<fir::HasValueOp>(loc, castTo);
          },
          builder.createInternalLinkage());
    }
    auto addr = builder.create<fir::AddrOfOp>(getLoc(), global.resultType(),
                                              global.getSymbol());
    auto seqTy = global.getType().cast<fir::SequenceType>();
    llvm::SmallVector<mlir::Value> extents;
    for (auto extent : seqTy.getShape())
      extents.push_back(builder.createIntegerConstant(loc, idxTy, extent));
    if (auto charTy = seqTy.getEleTy().dyn_cast<fir::CharacterType>()) {
      auto len = builder.createIntegerConstant(loc, builder.getI64Type(),
                                               charTy.getLen());
      return genarr(fir::CharArrayBoxValue{addr, len, extents});
    }
    return genarr(fir::ArrayBoxValue{addr, extents});
  }

  //===--------------------------------------------------------------------===//
  // A vector subscript expression may be wrapped with a cast to INTEGER*8.
  // Get rid of it here so the vector can be loaded. Add it back when
  // generating the elemental evaluation (inside the loop nest).

  static Fortran::evaluate::Expr<Fortran::evaluate::SomeType>
  ignoreEvConvert(const Fortran::evaluate::Expr<Fortran::evaluate::Type<
                      Fortran::common::TypeCategory::Integer, 8>> &x) {
    return std::visit([&](const auto &v) { return ignoreEvConvert(v); }, x.u);
  }
  template <Fortran::common::TypeCategory FROM>
  static Fortran::evaluate::Expr<Fortran::evaluate::SomeType> ignoreEvConvert(
      const Fortran::evaluate::Convert<
          Fortran::evaluate::Type<Fortran::common::TypeCategory::Integer, 8>,
          FROM> &x) {
    return toEvExpr(x.left());
  }
  template <typename A>
  static Fortran::evaluate::Expr<Fortran::evaluate::SomeType>
  ignoreEvConvert(const A &x) {
    return toEvExpr(x);
  }

  //===--------------------------------------------------------------------===//
  // Get the `Se::Symbol*` for the subscript expression, `x`. This symbol can
  // be used to determine the lbound, ubound of the vector.

  template <typename A>
  static const Fortran::semantics::Symbol *
  extractSubscriptSymbol(const Fortran::evaluate::Expr<A> &x) {
    return std::visit([&](const auto &v) { return extractSubscriptSymbol(v); },
                      x.u);
  }
  template <typename A>
  static const Fortran::semantics::Symbol *
  extractSubscriptSymbol(const Fortran::evaluate::Designator<A> &x) {
    return Fortran::evaluate::UnwrapWholeSymbolDataRef(x);
  }
  template <typename A>
  static const Fortran::semantics::Symbol *extractSubscriptSymbol(const A &x) {
    return nullptr;
  }

  //===--------------------------------------------------------------------===//

  /// Get the declared lower bound value of the array `x` in dimension `dim`.
  /// The argument `one` must be an ssa-value for the constant 1.
  mlir::Value getLBound(const ExtValue &x, unsigned dim, mlir::Value one) {
    return fir::factory::readLowerBound(builder, getLoc(), x, dim, one);
  }

  /// Get the declared upper bound value of the array `x` in dimension `dim`.
  /// The argument `one` must be an ssa-value for the constant 1.
  mlir::Value getUBound(const ExtValue &x, unsigned dim, mlir::Value one) {
    auto loc = getLoc();
    auto lb = getLBound(x, dim, one);
    auto extent = fir::factory::readExtent(builder, loc, x, dim);
    auto add = builder.create<mlir::AddIOp>(loc, lb, extent);
    return builder.create<mlir::SubIOp>(loc, add, one);
  }

  /// Return the extent of the boxed array `x` in dimesion `dim`.
  mlir::Value getExtent(const ExtValue &x, unsigned dim) {
    return fir::factory::readExtent(builder, getLoc(), x, dim);
  }

  // Build a components path for a component that is type Ev::ArrayRef. The base
  // of `x` must be an Ev::Component, and that base must be a trailing array
  // expression. The left-most ranked expression will not be part of a sliced
  // path expression.
  std::tuple<ExtValue, mlir::Type>
  buildComponentsPathArrayRef(ComponentCollection &cmptData,
                              const Fortran::evaluate::ArrayRef &x) {
    auto loc = getLoc();
    const auto &arrBase = x.base();
    assert(!arrBase.IsSymbol());
    const auto &cmpt = arrBase.GetComponent();
    assert(cmpt.base().Rank() > 0);
    llvm::SmallVector<mlir::Value> subs;
    // All subscripts must be present, complete, and cannot be vectors nor
    // slice operations.
    for (const auto &ss : x.subscript())
      std::visit(
          Fortran::common::visitors{
              [&](const Fortran::evaluate::IndirectSubscriptIntegerExpr &ie) {
                const auto &e = ie.value(); // get rid of bonus dereference
                if (isArray(e))
                  fir::emitFatalError(loc,
                                      "multiple components along single path "
                                      "generating array subexpressions");
                // Lower scalar index expression, append it to subs.
                subs.push_back(fir::getBase(asScalar(e)));
              },
              [&](const auto &) {
                fir::emitFatalError(loc,
                                    "multiple components along single path "
                                    "generating array subexpressions");
              }},
          ss.u);
    auto tup = buildComponentsPath(cmptData, cmpt);
    cmptData.components.append(subs.begin(), subs.end());
    return tup;
  }

  template <typename A>
  ExtValue genArrayBase(const A &base) {
    ScalarExprLowering sel{getLoc(), converter, symMap, stmtCtx};
    return base.IsSymbol() ? sel.gen(base.GetFirstSymbol())
                           : sel.gen(base.GetComponent());
  }

  /// When we have an array reference, the expressions specified in each
  /// dimension may be slice operations (e.g. `i:j:k`), vectors, or simple
  /// (loop-invarianet) scalar expressions. This returns the base entity, the
  /// resulting type, and a continuation to adjust the default iteration space.
  std::tuple<ExtValue, mlir::Type>
  genSliceIndices(ComponentCollection &cmptData,
                  const Fortran::evaluate::ArrayRef &x) {
    auto loc = getLoc();
    auto idxTy = builder.getIndexType();
    auto one = builder.createIntegerConstant(loc, idxTy, 1);
    auto &trips = cmptData.trips;
    auto base = x.base();
    auto arrExt = genArrayBase(base);
    LLVM_DEBUG(llvm::dbgs() << "array: " << arrExt << '\n');
    auto &pc = cmptData.pc;
    for (auto sub : llvm::enumerate(x.subscript())) {
      std::visit(
          Fortran::common::visitors{
              [&](const Fortran::evaluate::Triplet &t) {
                // Generate a slice operation for the triplet. The first and
                // second position of the triplet may be omitted, and the
                // declared lbound and/or ubound expression values,
                // respectively, should be used instead.
                if (auto optLo = t.lower())
                  trips.push_back(fir::getBase(asScalar(*optLo)));
                else
                  trips.push_back(getLBound(arrExt, sub.index(), one));
                if (auto optUp = t.upper())
                  trips.push_back(fir::getBase(asScalar(*optUp)));
                else
                  trips.push_back(getUBound(arrExt, sub.index(), one));
                trips.push_back(fir::getBase(asScalar(t.stride())));
              },
              [&](const Fortran::evaluate::IndirectSubscriptIntegerExpr &ie) {
                const auto &e = ie.value(); // get rid of bonus dereference
                if (isArray(e)) {
                  // vector-subscript: Use the index values as read from a
                  // vector to determine the temporary array value.
                  // Note: 9.5.3.3.3(3) specifies undefined behavior for
                  // multiple updates to any specific array element through a
                  // vector subscript with replicated values.
                  assert(!isBoxValue() &&
                         "fir.box cannot be created with vector subscripts");
                  auto base = x.base();
                  auto exv = genArrayBase(base);
                  auto arrExpr = ignoreEvConvert(e);
                  auto arrLoad =
                      lowerArraySubspace(converter, symMap, stmtCtx, arrExpr);
                  auto arrLd = arrLoad.getResult();
                  auto eleTy =
                      arrLd.getType().cast<fir::SequenceType>().getEleTy();
                  auto currentPC = pc;
                  auto dim = sub.index();
                  auto lb =
                      fir::factory::readLowerBound(builder, loc, exv, dim, one);
                  auto arrLdTypeParams = arrLoad.typeparams();
                  pc = [=](IterSpace iters) {
                    IterationSpace newIters = currentPC(iters);
                    auto iter = newIters.iterVec()[dim];
                    auto resTy = adjustedArrayElementType(eleTy);
                    auto fetch = builder.create<fir::ArrayFetchOp>(
                        loc, resTy, arrLd, mlir::ValueRange{iter},
                        arrLdTypeParams);
                    auto cast = builder.createConvert(loc, idxTy, fetch);
                    auto val =
                        builder.create<mlir::SubIOp>(loc, idxTy, cast, lb);
                    newIters.setIndexValue(dim, val);
                    return newIters;
                  };
                  auto useInexactRange = [&]() {
                    // Get the range of the array in this dimension, [1:n:1].
                    trips.push_back(one);
                    trips.push_back(getExtent(arrExt, sub.index()));
                    trips.push_back(one);
                  };
                  if (const auto *sym = extractSubscriptSymbol(arrExpr)) {
                    auto symVal = symMap.lookupSymbol(*sym);
                    symVal.match(
                        [&](const fir::ArrayBoxValue &v) {
                          trips.push_back(getLBound(v, 0, one));
                          trips.push_back(getUBound(v, 0, one));
                          trips.push_back(one);
                        },
                        [&](auto) { useInexactRange(); });
                  } else {
                    useInexactRange();
                  }
                } else {
                  // A regular scalar index, which does not yield an array
                  // section. Use a degenerate slice operation `(e:undef:undef)`
                  // in this dimension as a placeholder. This does not
                  // necessarily change the rank of the original array, so the
                  // iteration space must also be extended to include this
                  // expression in this dimension to adjust to the array's
                  // declared rank.
                  auto base = x.base();
                  auto exv = genArrayBase(base);
                  auto v = fir::getBase(asScalar(e));
                  trips.push_back(v);
                  auto undef = builder.create<fir::UndefOp>(loc, idxTy);
                  trips.push_back(undef);
                  trips.push_back(undef);
                  auto currentPC = pc;
                  // Cast `e` to index type.
                  auto iv = builder.createConvert(loc, idxTy, v);
                  auto dim = sub.index();
                  auto lb =
                      fir::factory::readLowerBound(builder, loc, exv, dim, one);
                  // Normalize `e` by subtracting the declared lbound.
                  mlir::Value ivAdj =
                      builder.create<mlir::SubIOp>(loc, idxTy, iv, lb);
                  // Add lbound adjusted value of `e` to the iteration vector
                  // (except when creating a box because the iteration vector is
                  // empty).
                  if (!isBoxValue())
                    pc = [=](IterSpace iters) {
                      IterationSpace newIters = currentPC(iters);
                      newIters.insertIndexValue(dim, ivAdj);
                      return newIters;
                    };
                }
              }},
          sub.value().u);
    }
    auto ty = fir::dyn_cast_ptrOrBoxEleTy(fir::getBase(arrExt).getType());
    return {arrExt, ty};
  }

  /// Is there a Symbol in common between the concurrent header set and the set
  /// of symbols in the expression?
  template <typename A>
  bool symbolSetsIntersect(
      llvm::ArrayRef<const Fortran::semantics::Symbol *> ctrlSet,
      const A &exprSyms) {
    for (const auto &sym : exprSyms)
      if (std::find(ctrlSet.begin(), ctrlSet.end(), &sym.get()) !=
          ctrlSet.end())
        return true;
    return false;
  }

  mlir::Type raiseBaseType(const Fortran::evaluate::Component &x) {
    auto baseTy = raiseBaseType(x.base());
    auto recTy = baseTy.cast<fir::RecordType>();
    auto rv = recTy.getType(toStringRef(x.GetLastSymbol().name()));
    LLVM_DEBUG(llvm::dbgs() << "base type c " << rv << '\n');
    return rv;
  }
  mlir::Type raiseBaseType(const Fortran::semantics::SymbolRef &sym) {
    auto rv = converter.genType(sym);
    LLVM_DEBUG(llvm::dbgs() << "base type s " << rv << '\n');
    return rv;
  }
  mlir::Type raiseBaseType(const Fortran::evaluate::ArrayRef &x) {
    auto &base = x.base();
    mlir::Type baseTy = base.IsSymbol() ? raiseBaseType(base.GetLastSymbol())
                                        : raiseBaseType(base.GetComponent());
    auto seqTy = baseTy.cast<fir::SequenceType>();
    auto rv = seqTy.getEleTy();
    LLVM_DEBUG(llvm::dbgs() << "base type a " << rv << '\n');
    return rv;
  }
  mlir::Type raiseBaseType(const Fortran::evaluate::DataRef &x) {
    return std::visit(
        Fortran::common::visitors{
            [&](const Fortran::evaluate::CoarrayRef &) -> mlir::Type {
              TODO(getLoc(), "coarray reference");
            },
            [&](const auto &c) { return raiseBaseType(c); }},
        x.u);
  }

  std::pair<CC, mlir::Type> raiseBase(const Fortran::semantics::SymbolRef &x) {
    return {genarr(x), raiseBaseType(x)};
  }
  std::pair<CC, mlir::Type> raiseBase(const Fortran::evaluate::Component &x) {
    return {genComponent(x), raiseBaseType(x)};
  }
  template <typename A>
  std::pair<CC, mlir::Type> raiseRankedBase(const A &x) {
    auto result = raiseBase(x);
    if (isProjectedCopyInCopyOut()) {
      auto optShape = Fortran::evaluate::GetShape(x);
      assert(optShape.has_value());
      convertFEShape(*optShape, explicitImpliedShape);
    }
    return result;
  }
  template <typename A>
  std::pair<CC, mlir::Type> raiseComponent(llvm::Optional<CC> fopt, const A &x,
                                           mlir::Type inTy, bool onRight) {
    if (fopt.hasValue()) {
      CC lambda = fopt.getValue();
      auto loc = getLoc();
      auto fieldTy = fir::FieldType::get(inTy.getContext());
      auto name = toStringRef(x.GetLastSymbol().name());
      auto recTy = inTy.cast<fir::RecordType>();
      auto fldTy = recTy.getType(name);
      auto pc = [=](IterSpace iters) {
        IterationSpace newIters = iters;
        auto fld = builder.create<fir::FieldIndexOp>(
            loc, fieldTy, name, recTy, /*typeparams=*/mlir::ValueRange{});
        if (onRight)
          newIters.appendIndexValue(fld);
        else
          newIters.prependIndexValue(fld);
        return newIters;
      };
      return {[=](IterSpace iters) { return lambda(pc(iters)); }, fldTy};
    }
    fir::emitFatalError(getLoc(), "expected a continuation");
  }
  template <typename A>
  std::pair<CC, mlir::Type> raiseRankedComponent(llvm::Optional<CC> cc,
                                                 const A &x, mlir::Type inTy) {
    auto result = raiseComponent(cc, x, inTy, false);
    if (isProjectedCopyInCopyOut()) {
      auto optShape = Fortran::evaluate::GetShape(x);
      assert(optShape.has_value());
      convertFEShape(*optShape, explicitImpliedShape);
    }
    return result;
  }

  using RaiseRT = std::tuple<llvm::Optional<CC>, mlir::Type, bool, bool>;

  RaiseRT
  raiseToArray(const Fortran::evaluate::DataRef &x,
               llvm::ArrayRef<const Fortran::semantics::Symbol *> ctrlSet) {
    return std::visit(
        Fortran::common::visitors{
            [&](const Fortran::semantics::SymbolRef &s) -> RaiseRT {
              if (x.Rank() > 0) {
                auto [fopt2, ty2] = raiseRankedBase(s);
                return {fopt2, ty2, false, true};
              }
              return {llvm::None, mlir::Type{}, false, false};
            },
            [&](const Fortran::evaluate::CoarrayRef &) -> RaiseRT {
              TODO(getLoc(), "coarray reference");
              return {llvm::None, mlir::Type{}, false, false};
            },
            [&](const auto &y) -> RaiseRT { return raiseToArray(y, ctrlSet); }},
        x.u);
  }
  RaiseRT
  raiseToArray(const Fortran::evaluate::Component &x,
               llvm::ArrayRef<const Fortran::semantics::Symbol *> ctrlSet) {
    auto [fopt, ty, inrank, ranked] = raiseToArray(x.base(), ctrlSet);
    if (fopt.hasValue()) {
      if (!ranked && x.Rank() > 0) {
        auto [fopt2, ty2] = raiseRankedComponent(fopt, x, ty);
        return {fopt2, ty2, ranked, true};
      }
      auto [fopt2, ty2] = raiseComponent(fopt, x, ty, ranked);
      return {fopt2, ty2, ranked, ranked};
    }
    if (x.Rank() > 0) {
      auto [fopt2, ty2] = raiseRankedBase(x);
      return {fopt2, ty2, ranked, true};
    }
    return {fopt, ty, inrank, ranked};
  }
  RaiseRT
  raiseToArray(const Fortran::evaluate::ArrayRef &x,
               llvm::ArrayRef<const Fortran::semantics::Symbol *> ctrlSet) {
    const auto &base = x.base();
    auto accessUsesControlVariable = [&]() {
      for (const auto &subs : x.subscript())
        if (auto *intExpr =
                std::get_if<Fortran::evaluate::IndirectSubscriptIntegerExpr>(
                    &subs.u)) {
          if (symbolSetsIntersect(
                  ctrlSet, Fortran::evaluate::CollectSymbols(intExpr->value())))
            return true;
        } else {
          // Subscript is a triplet, so x.Rank() should have been
          // greater than zero.
          llvm_unreachable("front end reported the rank of this "
                           "component 0 but it has triple notation");
        }
      return false;
    };
    return raiseSubscript(
        [&]() -> RaiseRT {
          if (base.IsSymbol()) {
            auto &sym = base.GetFirstSymbol();
            if (x.Rank() > 0 || accessUsesControlVariable()) {
              auto [fopt2, ty2] = raiseBase(sym);
              return RaiseRT{fopt2, fir::unwrapSequenceType(ty2), false,
                             x.Rank() > 0};
            }
            return RaiseRT{llvm::None, mlir::Type{}, false, false};
          }
          // Otherwise, it's a component.
          auto [fopt, ty, inrank, ranked] =
              raiseToArray(base.GetComponent(), ctrlSet);
          if (fopt.hasValue())
            return RaiseRT{fopt, ty, inrank, ranked};
          if (x.Rank() > 0 || accessUsesControlVariable()) {
            auto [fopt2, ty2] = raiseBase(base.GetComponent());
            return RaiseRT{fopt2, ty2, inrank, x.Rank() > 0};
          }
          return RaiseRT{fopt, ty, inrank, ranked};
        }(),
        x);
  }
  RaiseRT raiseSubscript(const RaiseRT &tup,
                         const Fortran::evaluate::ArrayRef &x) {
    auto fopt = std::get<llvm::Optional<CC>>(tup);
    if (fopt.hasValue()) {
      auto ty = std::get<mlir::Type>(tup);
      auto prevRanked = std::get<2>(tup);
      auto ranked = std::get<3>(tup);
      auto lambda = fopt.getValue();
      auto loc = getLoc();
      auto idxTy = builder.getIndexType();
      auto &subs = x.subscript();
      llvm::SmallVector<llvm::Optional<CC>> vectorCoor;
      const auto implicitArguments = !prevRanked && ranked;
      if (implicitArguments && isProjectedCopyInCopyOut()) {
        // If the array reference has subscripts that use control variables
        // from the explicit space, then those dimensions should not be
        // considered as contributing to the implied part of the iteration
        // space.
        if (explicitImpliedShape.empty()) {
          assert(destination && "destination must be set");
          auto feShape = getShape(destination);
          if (subs.empty()) {
            explicitImpliedShape.assign(feShape);
          } else {
            unsigned ii = 0;
            unsigned vi = 0;
            vectorCoor.resize(feShape.size());
            // Filter out subscripts that are scalar expressions. If it is a
            // scalar expression it is either loop-invariant or a function of
            // the explicit loop control variables.
            for (const auto &ss : subs)
              if (auto *intExpr = std::get_if<
                      Fortran::evaluate::IndirectSubscriptIntegerExpr>(&ss.u)) {
                if (intExpr->value().Rank() > 0) {
                  explicitImpliedShape.push_back(feShape[ii++]);
                  vectorCoor[vi++] = genarr(intExpr->value());
                }
              } else {
                // This is a triple which may be using an explicit control
                // variable.
                explicitImpliedShape.push_back(feShape[ii++]);
              }
          }
        }
      }
      auto one = builder.createIntegerConstant(loc, idxTy, 1);
      auto pc = [=](IterSpace iters) {
        IterationSpace newIters = iters;
        const auto firstImplicitIndex = iters.beginImplicitIndex();
        auto implicitIndex = iters.endImplicitIndex();
        assert(firstImplicitIndex <= implicitIndex);
        implicitIndex = implicitIndex ? implicitIndex - 1 : 0;
        unsigned vi = vectorCoor.size() - 1;
        for (auto &ss : llvm::reverse(subs)) {
          std::visit(
              Fortran::common::visitors{
                  [&](const Fortran::evaluate::Triplet &t) {
                    if (implicitArguments) {
                      auto tripSt = fir::getBase(asScalar(t.stride()));
                      auto castSt = builder.createConvert(loc, idxTy, tripSt);
                      mlir::Value off = builder.create<mlir::MulIOp>(
                          loc, castSt, iters.iterValue(implicitIndex));
                      if (auto optLo = t.lower()) {
                        auto tripLo = fir::getBase(asScalar(*optLo));
                        auto base = genArrayBase(x.base());
                        auto lb = getLBound(
                            base, implicitIndex - firstImplicitIndex, one);
                        auto clo = builder.createConvert(loc, idxTy, tripLo);
                        auto add = builder.create<mlir::AddIOp>(loc, off, clo);
                        off = builder.create<mlir::SubIOp>(loc, add, lb);
                      }
                      newIters.setIndexValue(implicitIndex, off);
                      implicitIndex = implicitIndex ? implicitIndex - 1 : 0;
                    } else {
                      // triple found in unexpected part of the path
                      fir::emitFatalError(loc, "unexpected triple");
                    }
                  },
                  [&](const Fortran::evaluate::IndirectSubscriptIntegerExpr
                          &ie) {
                    const auto &e = ie.value();
                    if (e.Rank() == 0) {
                      // subscript is a vanilla scalar expression and may refer
                      // to the explicit control variables.
                      auto sc = fir::getBase(asScalar(e));
                      auto v = builder.createConvert(loc, idxTy, sc);
                      if (implicitArguments)
                        newIters.insertIndexValue(
                            implicitIndex > 0 ? implicitIndex : 0, v);
                      else if (ranked)
                        newIters.appendIndexValue(v);
                      else
                        newIters.prependIndexValue(v);
                    } else if (implicitArguments) {
                      // subscript is a vector
                      auto lambda = vectorCoor[vi--].getValue()(iters);
                      auto vs = fir::getBase(lambda);
                      auto vecIdx = builder.createConvert(loc, idxTy, vs);
                      newIters.setIndexValue(implicitIndex--, vecIdx);
                    } else {
                      // vector found in unexpected part of the path
                      fir::emitFatalError(loc, "unexpected vactor");
                    }
                  }},
              ss.u);
        }
        return newIters;
      };
      return RaiseRT{[=](IterSpace iters) { return lambda(pc(iters)); }, ty,
                     prevRanked, ranked};
    }
    return tup;
  }

  /// Get all the explicit (user-defined) control symbols.
  llvm::SmallVector<const Fortran::semantics::Symbol *>
  collectControlSymbols() {
    // While this is technically O(n), we expect n to typically be very small.
    assert(explicitSpace && "must be FORALL context");
    llvm::SmallVector<const Fortran::semantics::Symbol *> result;
    for (const auto &d : explicitSpace->getDims())
      result.push_back(std::get<0>(d));
    return result;
  }

  /// Given a data reference `x`, which may be scalar, raise it to an array
  /// value expression. In the case of an explicit iteration space, multiple
  /// components in a path may participate in array operations concurrently.
  /// Because of this, it is not generally possible to create a box that
  /// describes the array's projection, as every box is currently limited to a
  /// linear Diophantine equation and access expressions using explicit control
  /// variables, i.e. `array(func(i))`, are not.
  template <typename A>
  CC raiseToArray(const A &x) {
    auto tup = raiseToArray(x, collectControlSymbols());
    auto fopt = std::get<llvm::Optional<CC>>(tup);
    assert(fopt.hasValue() && "continuation must be returned");
    return fopt.getValue();
  }

  /// Array reference with subscripts. If this has rank > 0, this is a form
  /// of an array section (slice).
  ///
  /// There are two "slicing" primitives that may be applied on a dimension by
  /// dimension basis: (1) triple notation and (2) vector addressing. Since
  /// dimensions can be selectively sliced, some dimensions may contain
  /// regular scalar expressions and those dimensions do not participate in
  /// the array expression evaluation.
  CC genarr(const Fortran::evaluate::ArrayRef &x) {
    if (explicitSpace)
      return raiseToArray(x);
    const auto &arrBase = x.base();
    if (!arrBase.IsSymbol()) {
      // `x` is a component with rank.
      const auto &cmpt = arrBase.GetComponent();
      if (cmpt.base().Rank() > 0) {
        // `x` is right of the base/component giving rise to the ranked expr.
        // In this case, the array in question is to the left of this
        // component. This component is an intraobject slice.
        ComponentCollection cmptData;
        auto tup = buildComponentsPathArrayRef(cmptData, x);
        auto lambda = genSlicePath(std::get<ExtValue>(tup), cmptData.trips,
                                   cmptData.components);
        auto pc = cmptData.pc;
        return [=](IterSpace iters) { return lambda(pc(iters)); };
      }
    }
    ComponentCollection cmptData;
    auto tup = genSliceIndices(cmptData, x);
    auto lambda = genSlicePath(std::get<ExtValue>(tup), cmptData.trips,
                               cmptData.components);
    auto pc = cmptData.pc;
    return [=](IterSpace iters) { return lambda(pc(iters)); };
  }

  CC genarr(const Fortran::evaluate::NamedEntity &entity) {
    if (entity.IsSymbol())
      return genarr(Fortran::semantics::SymbolRef{entity.GetFirstSymbol()});
    return genarr(entity.GetComponent());
  }

  CC genarr(const Fortran::semantics::SymbolRef &sym) {
    if (explicitSpace &&
        symbolSetsIntersect(collectControlSymbols(),
                            llvm::ArrayRef<Fortran::semantics::SymbolRef>{sym}))
      return [=](IterSpace) { return asScalar(sym); };
    return genarr(asScalarRef(sym));
  }
  CC genarr(const Fortran::semantics::Symbol &sym) {
    return genarr(Fortran::semantics::SymbolRef{sym});
  }

  static mlir::Type adjustedArraySubtype(mlir::Type ty,
                                         mlir::ValueRange indices) {
    return adjustedArrayElementType(fir::applyPathToType(ty, indices));
  }

  /// Build an ExtendedValue from a fir.array<?x...?xT> without actually
  /// setting the actual extents and lengths. This is only to allow their
  /// propagation as ExtendedValue without triggering verifier failure when
  /// propagating character/arrays as unboxed values. Only the base of the
  /// resulting ExtendedValue should be used, it is undefined to use its the
  /// length or extents,
  ExtValue abstractArrayExtValue(mlir::Value val) {
    auto ty = val.getType();
    if (auto ety = fir::dyn_cast_ptrEleTy(ty))
      ty = ety;
    auto idxTy = builder.getIndexType();
    auto loc = getLoc();
    auto seqTy = ty.cast<fir::SequenceType>();
    auto undef = builder.create<fir::UndefOp>(loc, idxTy);
    llvm::SmallVector<mlir::Value> extents(seqTy.getDimension(), undef);
    if (fir::isa_char(seqTy.getEleTy()))
      return fir::CharArrayBoxValue(val, undef, extents);
    return fir::ArrayBoxValue(val, extents);
  }

  mlir::Value maybeGenStringCopy(mlir::Type toTy, mlir::Value fromVal) {
    // FIXME: For now, just blindly insert a fir.convert, but that doesn't
    // really capture the semantics properly. The string needs to be memmoved
    // with truncation, padding, or neither.
    if (fir::isa_char(toTy)) {
      // assert(elementTypeWasAdjusted(fromVal.getType()));
      return builder.createConvert(getLoc(), builder.getRefType(toTy), fromVal);
    }
    return fromVal;
  }
  mlir::Value tryUpdateConversions(mlir::Type toTy, mlir::Value fromVal) {
    if (fir::isa_char(toTy))
      return maybeGenStringCopy(toTy, fromVal);
    if (!fir::isa_ref_type(toTy) && !fir::isa_ref_type(fromVal.getType()))
      return builder.createConvert(getLoc(), toTy, fromVal);
    return fromVal;
  }

  /// Base case of generating an array reference,
  CC genarr(const ExtValue &extMemref) {
    auto loc = getLoc();
    auto memref = fir::getBase(extMemref);
    auto arrTy = fir::dyn_cast_ptrOrBoxEleTy(memref.getType());
    assert(arrTy.isa<fir::SequenceType>());
    auto shape = builder.createShape(loc, extMemref);
    mlir::Value slice;
    if (inSlice) {
      slice = builder.createSlice(loc, extMemref, sliceTriple, slicePath);
      if (!slicePath.empty()) {
        auto seqTy = arrTy.cast<fir::SequenceType>();
        auto eleTy = fir::applyPathToType(seqTy.getEleTy(), slicePath);
        if (!eleTy)
          fir::emitFatalError(loc, "slicing path is ill-formed");
        if (auto realTy = eleTy.dyn_cast<fir::RealType>())
          eleTy = Fortran::lower::convertReal(realTy.getContext(),
                                              realTy.getFKind());

        // create the type of the projected array.
        arrTy = fir::SequenceType::get(seqTy.getShape(), eleTy);
        LLVM_DEBUG(llvm::dbgs()
                   << "type of array projection from component slicing: "
                   << eleTy << ", " << arrTy << '\n');
      }
    }
    if (isBoxValue()) {
      // Semantics are a reference to a boxed array.
      // This case just requires that an embox operation be created to box the
      // value. The value of the box is forwarded in the continuation.
      auto boxTy = fir::BoxType::get(reduceRank(arrTy, slice));
      mlir::Value embox =
          memref.getType().isa<fir::BoxType>()
              ? builder.create<fir::ReboxOp>(loc, boxTy, memref, shape, slice)
                    .getResult()
              : builder
                    .create<fir::EmboxOp>(loc, boxTy, memref, shape, slice,
                                          fir::getTypeParams(extMemref))
                    .getResult();
      return [=](IterSpace) -> ExtValue { return fir::BoxValue(embox); };
    }
    auto eleTy = arrTy.cast<fir::SequenceType>().getEleTy();
    if (isReferentiallyOpaque()) {
      // Semantics are an opaque reference to an array.
      // This case forwards a continuation that will generate the address
      // arithmetic to the array element. No attempt to preserve the value at
      // the address during the interpretation of Fortran statement is made.
      auto refEleTy = builder.getRefType(eleTy);
      return [=](IterSpace iters) -> ExtValue {
        // ArrayCoorOp does not expect zero based indices.
        auto indices = fir::factory::originateIndices(
            loc, builder, memref.getType(), shape, iters.iterVec());
        mlir::Value coor = builder.create<fir::ArrayCoorOp>(
            loc, refEleTy, memref, shape, slice, indices,
            fir::getTypeParams(extMemref));
        return arraySectionElementToExtendedValue(builder, loc, extMemref, coor,
                                                  slice);
      };
    }
    auto arrLoad = builder.create<fir::ArrayLoadOp>(
        loc, arrTy, memref, shape, slice, fir::getTypeParams(extMemref));
    auto arrLd = arrLoad.getResult();
    if (isProjectedCopyInCopyOut()) {
      // Semantics are projected copy-in copy-out.
      // The backing store of the destination of an array expression may be
      // partially modified. These updates are recorded in FIR by forwarding a
      // continuation that generates an `array_update` Op. The destination is
      // always loaded at the beginning of the statement and merged at the
      // end.
      destination = arrLoad;
      return [=](IterSpace iters) -> ExtValue {
        auto innerArg = iters.innerArgument();
        auto resTy = adjustedArrayElementType(innerArg.getType());
        assert(resTy == innerArg.getType());
        auto eleTy = fir::applyPathToType(resTy, iters.iterVec());
        auto element = tryUpdateConversions(eleTy, iters.getElement());
        auto arrUpdate = builder.create<fir::ArrayUpdateOp>(
            loc, resTy, innerArg, element, iters.iterVec(),
            destination.typeparams());
        return abstractArrayExtValue(arrUpdate);
      };
    }
    arrayOperandLoads.emplace_back(arrLoad);
    if (isCopyInCopyOut()) {
      // Semantics are copy-in copy-out.
      // The continuation simply forwards the result of the `array_load` Op,
      // which is the value of the array as it was when loaded. All data
      // references with rank > 0 in an array expression typically have
      // copy-in copy-out semantics.
      return [=](IterSpace) -> ExtValue { return arrLd; };
    }
    auto arrLdTypeParams = arrLoad.typeparams();
    if (isValueAttribute()) {
      // Semantics are value attribute.
      // Here the continuation will `array_fetch` a value from an array and
      // then store that value in a temporary. One can thus imitate pass by
      // value even when the call is pass by reference.
      return [=](IterSpace iters) -> ExtValue {
        auto arrFetch = builder.create<fir::ArrayFetchOp>(
            loc, adjustedArraySubtype(arrTy, iters.iterVec()), arrLd,
            iters.iterVec(), arrLdTypeParams);
        auto base = arrFetch.getResult();
        auto temp = builder.createTemporary(
            loc, base.getType(),
            llvm::ArrayRef<mlir::NamedAttribute>{getAdaptToByRefAttr()});
        builder.create<fir::StoreOp>(loc, base, temp);
        return arraySectionElementToExtendedValue(builder, loc, extMemref, temp,
                                                  slice);
      };
    }
    // In the default case, the array reference forwards an `array_fetch` Op
    // in the continuation.
    return [=](IterSpace iters) -> ExtValue {
      auto arrFetch = builder.create<fir::ArrayFetchOp>(
          loc, adjustedArraySubtype(arrTy, iters.iterVec()), arrLd,
          iters.iterVec(), arrLdTypeParams);
      return arraySectionElementToExtendedValue(builder, loc, extMemref,
                                                arrFetch, slice);
    };
  }

  /// Reduce the rank of a array to be boxed based on the slice's operands.
  static mlir::Type reduceRank(mlir::Type arrTy, mlir::Value slice) {
    if (slice) {
      auto slOp = mlir::dyn_cast<fir::SliceOp>(slice.getDefiningOp());
      assert(slOp);
      auto seqTy = arrTy.dyn_cast<fir::SequenceType>();
      assert(seqTy);
      auto triples = slOp.triples();
      fir::SequenceType::Shape shape;
      // reduce the rank for each invariant dimension
      for (unsigned i = 1, end = triples.size(); i < end; i += 3)
        if (!mlir::isa_and_nonnull<fir::UndefOp>(triples[i].getDefiningOp()))
          shape.push_back(fir::SequenceType::getUnknownExtent());
      return fir::SequenceType::get(shape, seqTy.getEleTy());
    }
    // not sliced, so no change in rank
    return arrTy;
  }

  /// Lower a component path with rank or raise it from a scalar expression to
  /// have array semantics.
  /// Example: <code>array%baz%qux%waldo</code>
  CC genarr(const Fortran::evaluate::Component &x) {
    return explicitSpace ? raiseToArray(x) : genComponent(x);
  }
  CC genComponent(const Fortran::evaluate::Component &x) {
    ComponentCollection cmptData;
    auto tup = buildComponentsPath(cmptData, x);
    auto lambda = genSlicePath(std::get<ExtValue>(tup), cmptData.trips,
                               cmptData.components);
    auto pc = cmptData.pc;
    return [=](IterSpace iters) { return lambda(pc(iters)); };
  }

  /// The `Ev::Component` structure is tailmost down to head, so the
  /// expression <code>a%b%c</code> will be presented as <code>(component
  /// (dataref (component (dataref (symbol 'a)) (symbol 'b))) (symbol
  /// 'c))</code>.
  std::tuple<ExtValue, mlir::Type>
  buildComponentsPath(ComponentCollection &cmptData,
                      const Fortran::evaluate::Component &x) {
    using RT = std::tuple<ExtValue, mlir::Type>;
    auto loc = getLoc();
    auto dr = x.base();
    if (dr.Rank() == 0) {
      auto exv = asScalarRef(x);
      return RT{exv, fir::getBase(exv).getType()};
    }
    auto addComponent = [&](const ExtValue &exv, mlir::Type ty) {
      assert(ty.isa<fir::SequenceType>());
      auto arrTy = ty.cast<fir::SequenceType>();
      auto name = toStringRef(x.GetLastSymbol().name());
      auto recTy = arrTy.getEleTy();
      auto eleTy = recTy.cast<fir::RecordType>().getType(name);
      auto fldTy = fir::FieldType::get(eleTy.getContext());
      cmptData.components.push_back(builder.create<fir::FieldIndexOp>(
          getLoc(), fldTy, name, recTy, fir::getTypeParams(exv)));
      auto refOfTy = eleTy.isa<fir::SequenceType>()
                         ? eleTy
                         : fir::SequenceType::get(arrTy.getShape(), eleTy);
      return RT{exv, builder.getRefType(refOfTy)};
    };
    return std::visit(
        Fortran::common::visitors{
            [&](const Fortran::evaluate::Component &c) {
              auto [exv, refTy] = buildComponentsPath(cmptData, c);
              auto ty = fir::dyn_cast_ptrOrBoxEleTy(refTy);
              return addComponent(exv, ty);
            },
            [&](const Fortran::semantics::SymbolRef &y) {
              auto exv = asScalarRef(y);
              auto ty =
                  fir::dyn_cast_ptrOrBoxEleTy(fir::getBase(exv).getType());
              return addComponent(exv, ty);
            },
            [&](const Fortran::evaluate::ArrayRef &r) -> RT {
              auto arrBase = r.base();
              if (arrBase.Rank() > 0 && !arrBase.IsSymbol())
                if (const auto &cmpt = arrBase.GetComponent();
                    cmpt.base().Rank() > 0) {
                  auto [exv, refTy] = buildComponentsPathArrayRef(cmptData, r);
                  auto ty = fir::dyn_cast_ptrOrBoxEleTy(refTy);
                  return addComponent(exv, ty);
                }
              auto [exv, ty] = genSliceIndices(cmptData, r);
              return addComponent(exv, ty);
            },
            [&](const Fortran::evaluate::CoarrayRef &r) -> RT {
              TODO(loc, "");
            }},
        dr.u);
  }

  /// Example: <code>array%RE</code>
  CC genarr(const Fortran::evaluate::ComplexPart &x) {
    auto loc = getLoc();
    auto i32Ty = builder.getI32Type(); // llvm's GEP requires i32
    auto offset = builder.createIntegerConstant(
        loc, i32Ty,
        x.part() == Fortran::evaluate::ComplexPart::Part::RE ? 0 : 1);
    auto lambda = genSlicePath(x.complex(), {}, {offset});
    return [=](IterSpace iters) { return lambda(iters); };
  }

  template <typename A>
  CC genSlicePath(const A &x, mlir::ValueRange trips, mlir::ValueRange path) {
    if (!sliceTriple.empty())
      fir::emitFatalError(getLoc(), "multiple slices");
    auto saveInSlice = inSlice;
    inSlice = true;
    auto sz = slicePath.size();
    sliceTriple.append(trips.begin(), trips.end());
    slicePath.append(path.begin(), path.end());
    auto result = genarr(x);
    sliceTriple.clear();
    slicePath.resize(sz);
    inSlice = saveInSlice;
    return result;
  }

  CC genarr(const Fortran::evaluate::CoarrayRef &) {
    TODO(getLoc(), "coarray ref");
  }

  /// 9.4.1 Substrings
  CC genarr(const Fortran::evaluate::Substring &x) {
    auto loc = getLoc();
    auto pf = std::visit(
        Fortran::common::visitors{
            [&](const Fortran::evaluate::DataRef &p) { return genarr(p); },
            [&](const Fortran::evaluate::StaticDataObject::Pointer &) -> CC {
              fir::emitFatalError(loc, "substring of static array object");
            }},
        x.parent());
    // lower and upper *must* be scalars
    llvm::SmallVector<mlir::Value> bounds = {fir::getBase(asScalar(x.lower()))};
    if (auto upper = x.upper())
      bounds.push_back(fir::getBase(asScalar(*upper)));
    return [=](IterSpace iters) -> ExtValue {
      auto base = pf(iters);
      if (auto *chr = base.getCharBox())
        return fir::factory::CharacterExprHelper{builder, loc}.createSubstring(
            *chr, bounds);
      TODO(loc, "unhandled substring base type");
      return mlir::Value{};
    };
  }

  template <Fortran::common::TypeCategory TC, int KIND>
  CC genarr(
      const Fortran::evaluate::FunctionRef<Fortran::evaluate::Type<TC, KIND>>
          &x) {
    return genProcRef(x, {converter.genType(TC, KIND)});
  }

  //===--------------------------------------------------------------------===//
  // Array construction
  //===--------------------------------------------------------------------===//

  // Lower the expr cases in an ac-value-list.
  template <typename A>
  std::pair<ExtValue, bool>
  genArrayCtorInitializer(const Fortran::evaluate::Expr<A> &x, mlir::Type,
                          mlir::Value, mlir::Value, mlir::Value,
                          Fortran::lower::StatementContext &stmtCtx) {
    if (isArray(x)) {
      auto e = toEvExpr(x);
      auto sh = Fortran::evaluate::GetShape(converter.getFoldingContext(), e);
      return {lowerSomeNewArrayExpression(converter, symMap, stmtCtx, sh, e),
              /*needCopy=*/true};
    }
    return {asScalar(x), /*needCopy=*/true};
  }

  // Target agnostic computation of the size of an element in the array.
  // Returns the size in bytes with type `index` or a null Value if the
  // element size is not constant.
  mlir::Value computeElementSize(mlir::Type eleTy, mlir::Type eleRefTy,
                                 mlir::Type resRefTy) {
    if (fir::hasDynamicSize(eleTy))
      return {};
    auto loc = getLoc();
    auto idxTy = builder.getIndexType();
    auto nullPtr = builder.createNullConstant(loc, resRefTy);
    auto one = builder.createIntegerConstant(loc, idxTy, 1);
    auto offset = builder.create<fir::CoordinateOp>(loc, eleRefTy, nullPtr,
                                                    mlir::ValueRange{one});
    return builder.createConvert(loc, idxTy, offset);
  }

  /// Get the function signature of the LLVM memcpy intrinsic.
  mlir::FunctionType memcpyType() {
    return fir::factory::getLlvmMemcpy(builder).getType();
  }

  /// Create a call to the LLVM memcpy intrinsic.
  void createCallMemcpy(llvm::ArrayRef<mlir::Value> args) {
    auto loc = getLoc();
    auto memcpyFunc = fir::factory::getLlvmMemcpy(builder);
    auto funcSymAttr = builder.getSymbolRefAttr(memcpyFunc.getName());
    auto funcTy = memcpyFunc.getType();
    builder.create<fir::CallOp>(loc, funcTy.getResults(), funcSymAttr, args);
  }

  // Construct code to check for a buffer overrun and realloc the buffer when
  // space is depleted. This is done between each item in the ac-value-list.
  mlir::Value growBuffer(mlir::Value mem, mlir::Value needed,
                         mlir::Value bufferSize, mlir::Value buffSize,
                         mlir::Value eleSz) {
    auto loc = getLoc();
    auto reallocFunc = fir::factory::getRealloc(builder);
    auto cond = builder.create<mlir::CmpIOp>(loc, mlir::CmpIPredicate::sle,
                                             bufferSize, needed);
    auto ifOp = builder.create<fir::IfOp>(loc, mem.getType(), cond,
                                          /*withElseRegion=*/true);
    auto insPt = builder.saveInsertionPoint();
    builder.setInsertionPointToStart(&ifOp.thenRegion().front());
    // Not enough space, resize the buffer.
    auto idxTy = builder.getIndexType();
    auto two = builder.createIntegerConstant(loc, idxTy, 2);
    auto newSz = builder.create<mlir::MulIOp>(loc, needed, two);
    builder.create<fir::StoreOp>(loc, newSz, buffSize);
    mlir::Value byteSz = builder.create<mlir::MulIOp>(loc, newSz, eleSz);
    auto funcSymAttr = builder.getSymbolRefAttr(reallocFunc.getName());
    auto funcTy = reallocFunc.getType();
    auto newMem = builder.create<fir::CallOp>(
        loc, funcTy.getResults(), funcSymAttr,
        llvm::ArrayRef<mlir::Value>{
            builder.createConvert(loc, funcTy.getInputs()[0], mem),
            builder.createConvert(loc, funcTy.getInputs()[1], byteSz)});
    auto castNewMem =
        builder.createConvert(loc, mem.getType(), newMem.getResult(0));
    builder.create<fir::ResultOp>(loc, castNewMem);
    builder.setInsertionPointToStart(&ifOp.elseRegion().front());
    // Otherwise, just forward the buffer.
    builder.create<fir::ResultOp>(loc, mem);
    builder.restoreInsertionPoint(insPt);
    return ifOp.getResult(0);
  }

  // Copy the next value (or vector of values) into the array being
  // constructed.
  mlir::Value copyNextArrayCtorSection(const ExtValue &exv, mlir::Value buffPos,
                                       mlir::Value buffSize, mlir::Value mem,
                                       mlir::Value eleSz, mlir::Type eleTy,
                                       mlir::Type eleRefTy, mlir::Type resTy) {
    auto loc = getLoc();
    auto off = builder.create<fir::LoadOp>(loc, buffPos);
    auto limit = builder.create<fir::LoadOp>(loc, buffSize);
    auto idxTy = builder.getIndexType();
    auto one = builder.createIntegerConstant(loc, idxTy, 1);

    if (fir::isRecordWithAllocatableMember(eleTy))
      TODO(loc, "deep copy on allocatable members");

    if (!eleSz) {
      // Compute the element size at runtime.
      assert(fir::hasDynamicSize(eleTy));
      if (auto charTy = eleTy.dyn_cast<fir::CharacterType>()) {
        auto charBytes =
            builder.getKindMap().getCharacterBitsize(charTy.getFKind()) / 8;
        auto bytes = builder.createIntegerConstant(loc, idxTy, charBytes);
        auto length = fir::getLen(exv);
        if (!length)
          fir::emitFatalError(loc, "result is not boxed character");
        eleSz = builder.create<mlir::MulIOp>(loc, bytes, length);
      } else {
        TODO(loc, "PDT size");
        // Will call the PDT's size function with the type parameters.
      }
    }

    // Compute the coordinate using `fir.coordinate_of`, or, if the type has
    // dynamic size, generating the pointer arithmetic.
    auto computeCoordinate = [&](mlir::Value buff, mlir::Value off) {
      auto refTy = eleRefTy;
      if (fir::hasDynamicSize(eleTy)) {
        if (auto charTy = eleTy.dyn_cast<fir::CharacterType>()) {
          // Scale a simple pointer using dynamic length and offset values.
          auto chTy = fir::CharacterType::getSingleton(charTy.getContext(),
                                                       charTy.getFKind());
          refTy = builder.getRefType(chTy);
          auto toTy = builder.getRefType(builder.getVarLenSeqTy(chTy));
          buff = builder.createConvert(loc, toTy, buff);
          off = builder.create<mlir::MulIOp>(loc, off, eleSz);
        } else {
          TODO(loc, "PDT offset");
        }
      }
      auto coor = builder.create<fir::CoordinateOp>(loc, refTy, buff,
                                                    mlir::ValueRange{off});
      return builder.createConvert(loc, eleRefTy, coor);
    };

    // Lambda to lower an abstract array box value.
    auto doAbstractArray = [&](const auto &v) {
      // Compute the array size.
      auto arrSz = one;
      for (auto ext : v.getExtents())
        arrSz = builder.create<mlir::MulIOp>(loc, arrSz, ext);

      // Grow the buffer as needed.
      auto endOff = builder.create<mlir::AddIOp>(loc, off, arrSz);
      mem = growBuffer(mem, endOff, limit, buffSize, eleSz);

      // Copy the elements to the buffer.
      mlir::Value byteSz = builder.create<mlir::MulIOp>(loc, arrSz, eleSz);
      auto buff = builder.createConvert(loc, fir::HeapType::get(resTy), mem);
      auto buffi = computeCoordinate(buff, off);
      auto args = Fortran::lower::createArguments(
          builder, loc, memcpyType(), buffi, v.getAddr(), byteSz,
          /*volatile=*/builder.createBool(loc, false));
      createCallMemcpy(args);

      // Save the incremented buffer position.
      builder.create<fir::StoreOp>(loc, endOff, buffPos);
    };

    // Copy the value.
    exv.match(
        [&](const mlir::Value &v) {
          // Increment the buffer position.
          auto plusOne = builder.create<mlir::AddIOp>(loc, off, one);

          // Grow the buffer as needed.
          mem = growBuffer(mem, plusOne, limit, buffSize, eleSz);

          // Store the element in the buffer.
          auto buff =
              builder.createConvert(loc, fir::HeapType::get(resTy), mem);
          auto buffi = builder.create<fir::CoordinateOp>(loc, eleRefTy, buff,
                                                         mlir::ValueRange{off});
          auto val = builder.createConvert(loc, eleTy, v);
          builder.create<fir::StoreOp>(loc, val, buffi);

          builder.create<fir::StoreOp>(loc, plusOne, buffPos);
        },
        [&](const fir::CharBoxValue &v) {
          // Increment the buffer position.
          auto plusOne = builder.create<mlir::AddIOp>(loc, off, one);

          // Grow the buffer as needed.
          mem = growBuffer(mem, plusOne, limit, buffSize, eleSz);

          // Store the element in the buffer.
          auto buff =
              builder.createConvert(loc, fir::HeapType::get(resTy), mem);
          auto buffi = computeCoordinate(buff, off);
          auto args = Fortran::lower::createArguments(
              builder, loc, memcpyType(), buffi, v.getAddr(), eleSz,
              /*volatile=*/builder.createBool(loc, false));
          createCallMemcpy(args);

          builder.create<fir::StoreOp>(loc, plusOne, buffPos);
        },
        [&](const fir::ArrayBoxValue &v) { doAbstractArray(v); },
        [&](const fir::CharArrayBoxValue &v) { doAbstractArray(v); },
        [&](const auto &) {
          TODO(loc, "unhandled array constructor expression");
        });
    return mem;
  }

  // Lower an ac-implied-do in an ac-value-list.
  template <typename A>
  std::pair<ExtValue, bool>
  genArrayCtorInitializer(const Fortran::evaluate::ImpliedDo<A> &x,
                          mlir::Type resTy, mlir::Value mem,
                          mlir::Value buffPos, mlir::Value buffSize,
                          Fortran::lower::StatementContext &) {
    auto loc = getLoc();
    auto idxTy = builder.getIndexType();
    auto lo =
        builder.createConvert(loc, idxTy, fir::getBase(asScalar(x.lower())));
    auto up =
        builder.createConvert(loc, idxTy, fir::getBase(asScalar(x.upper())));
    auto step =
        builder.createConvert(loc, idxTy, fir::getBase(asScalar(x.stride())));
    auto seqTy = resTy.template cast<fir::SequenceType>();
    auto eleTy = fir::unwrapSequenceType(seqTy);
    auto loop =
        builder.create<fir::DoLoopOp>(loc, lo, up, step, /*unordered=*/false,
                                      /*finalCount=*/false, mem);
    // create a new binding for x.name(), to ac-do-variable, to the iteration
    // value.
    symMap.pushImpliedDoBinding(toStringRef(x.name()), loop.getInductionVar());
    auto insPt = builder.saveInsertionPoint();
    builder.setInsertionPointToStart(loop.getBody());
    // Thread mem inside the loop via loop argument.
    mem = loop.getRegionIterArgs()[0];

    auto eleRefTy = builder.getRefType(eleTy);
    auto eleSz = computeElementSize(eleTy, eleRefTy, builder.getRefType(resTy));

    // Cleanups for temps in loop body. Any temps created in the loop body
    // need to be freed before the end of the loop.
    Fortran::lower::StatementContext loopCtx;
    for (const Fortran::evaluate::ArrayConstructorValue<A> &acv : x.values()) {
      auto [exv, copyNeeded] = std::visit(
          [&](const auto &v) {
            return genArrayCtorInitializer(v, resTy, mem, buffPos, buffSize,
                                           loopCtx);
          },
          acv.u);
      mem = copyNeeded ? copyNextArrayCtorSection(exv, buffPos, buffSize, mem,
                                                  eleSz, eleTy, eleRefTy, resTy)
                       : fir::getBase(exv);
    }
    loopCtx.finalize();

    builder.create<fir::ResultOp>(loc, mem);
    builder.restoreInsertionPoint(insPt);
    mem = loop.getResult(0);
    symMap.popImpliedDoBinding();
    llvm::SmallVector<mlir::Value> extents = {
        builder.create<fir::LoadOp>(loc, buffPos).getResult()};

    // Convert to extended value.
    if (auto charTy =
            seqTy.getEleTy().template dyn_cast<fir::CharacterType>()) {
      auto len = builder.createIntegerConstant(loc, builder.getI64Type(),
                                               charTy.getLen());
      return {fir::CharArrayBoxValue{mem, len, extents}, /*needCopy=*/false};
    }
    return {fir::ArrayBoxValue{mem, extents}, /*needCopy=*/false};
  }

  // To simplify the handling and interaction between the various cases, array
  // constructors are always lowered to the incremental construction code
  // pattern, even if the extent of the array value is constant. After the
  // MemToReg pass and constant folding, the optimizer should be able to
  // determine that all the buffer overrun tests are false when the
  // incremental construction wasn't actually required.
  template <typename A>
  CC genarr(const Fortran::evaluate::ArrayConstructor<A> &x) {
    auto loc = getLoc();
    auto evExpr = toEvExpr(x);
    auto resTy = translateSomeExprToFIRType(converter, evExpr);
    auto idxTy = builder.getIndexType();
    auto seqTy = resTy.template cast<fir::SequenceType>();
    auto eleTy = fir::unwrapSequenceType(resTy);
    auto buffSize = builder.createTemporary(loc, idxTy, ".buff.size");
    auto zero = builder.createIntegerConstant(loc, idxTy, 0);
    auto buffPos = builder.createTemporary(loc, idxTy, ".buff.pos");
    builder.create<fir::StoreOp>(loc, zero, buffPos);
    // Allocate space for the array to be constructed.
    mlir::Value mem;
    if (fir::hasDynamicSize(resTy)) {
      if (fir::hasDynamicSize(eleTy)) {
        // The size of each element may depend on a general expression. Defer
        // creating the buffer until after the expression is evaluated.
        mem = builder.createNullConstant(loc, builder.getRefType(eleTy));
        builder.create<fir::StoreOp>(loc, zero, buffSize);
      } else {
        auto initBuffSz =
            builder.createIntegerConstant(loc, idxTy, clInitialBufferSize);
        mem = builder.create<fir::AllocMemOp>(
            loc, eleTy, /*typeparams=*/llvm::None, initBuffSz);
        builder.create<fir::StoreOp>(loc, initBuffSz, buffSize);
      }
    } else {
      mem = builder.create<fir::AllocMemOp>(loc, resTy);
      int64_t buffSz = 1;
      for (auto extent : seqTy.getShape())
        buffSz *= extent;
      auto initBuffSz = builder.createIntegerConstant(loc, idxTy, buffSz);
      builder.create<fir::StoreOp>(loc, initBuffSz, buffSize);
    }
    // Compute size of element
    auto eleRefTy = builder.getRefType(eleTy);
    auto eleSz = computeElementSize(eleTy, eleRefTy, builder.getRefType(resTy));

    // Populate the buffer with the elements, growing as necessary.
    for (const auto &expr : x) {
      auto [exv, copyNeeded] = std::visit(
          [&](const auto &e) {
            return genArrayCtorInitializer(e, resTy, mem, buffPos, buffSize,
                                           stmtCtx);
          },
          expr.u);
      mem = copyNeeded ? copyNextArrayCtorSection(exv, buffPos, buffSize, mem,
                                                  eleSz, eleTy, eleRefTy, resTy)
                       : fir::getBase(exv);
    }
    mem = builder.createConvert(loc, fir::HeapType::get(resTy), mem);
    llvm::SmallVector<mlir::Value> extents = {
        builder.create<fir::LoadOp>(loc, buffPos)};

    // Cleanup the temporary.
    auto *bldr = &converter.getFirOpBuilder();
    stmtCtx.attachCleanup(
        [bldr, loc, mem]() { bldr->create<fir::FreeMemOp>(loc, mem); });

    // Return the continuation.
    if (auto charTy =
            seqTy.getEleTy().template dyn_cast<fir::CharacterType>()) {
      auto len = builder.createIntegerConstant(loc, builder.getI64Type(),
                                               charTy.getLen());
      return genarr(fir::CharArrayBoxValue{mem, len, extents});
    }
    return genarr(fir::ArrayBoxValue{mem, extents});
  }

  CC genarr(const Fortran::evaluate::ImpliedDoIndex &) {
    fir::emitFatalError(getLoc(), "implied do index cannot have rank > 0");
  }
  CC genarr(const Fortran::evaluate::TypeParamInquiry &x) {
    TODO(getLoc(), "array expr type parameter inquiry");
    return [](IterSpace iters) -> ExtValue { return mlir::Value{}; };
  }
  CC genarr(const Fortran::evaluate::DescriptorInquiry &x) {
    TODO(getLoc(), "array expr descriptor inquiry");
    return [](IterSpace iters) -> ExtValue { return mlir::Value{}; };
  }
  CC genarr(const Fortran::evaluate::StructureConstructor &x) {
    TODO(getLoc(), "structure constructor");
    return [](IterSpace iters) -> ExtValue { return mlir::Value{}; };
  }

  //===--------------------------------------------------------------------===//
  // LOCICAL operators (.NOT., .AND., .EQV., etc.)
  //===--------------------------------------------------------------------===//

  template <int KIND>
  CC genarr(const Fortran::evaluate::Not<KIND> &x) {
    auto loc = getLoc();
    auto i1Ty = builder.getI1Type();
    auto lambda = genarr(x.left());
    auto truth = builder.createBool(loc, true);
    return [=](IterSpace iters) -> ExtValue {
      auto logical = fir::getBase(lambda(iters));
      auto val = builder.createConvert(loc, i1Ty, logical);
      return builder.create<mlir::XOrOp>(loc, val, truth);
    };
  }
  template <typename OP, typename A>
  CC createBinaryBoolOp(const A &x) {
    auto loc = getLoc();
    auto i1Ty = builder.getI1Type();
    auto lf = genarr(x.left());
    auto rf = genarr(x.right());
    return [=](IterSpace iters) -> ExtValue {
      auto left = fir::getBase(lf(iters));
      auto right = fir::getBase(rf(iters));
      auto lhs = builder.createConvert(loc, i1Ty, left);
      auto rhs = builder.createConvert(loc, i1Ty, right);
      return builder.create<OP>(loc, lhs, rhs);
    };
  }
  template <typename OP, typename A>
  CC createCompareBoolOp(mlir::CmpIPredicate pred, const A &x) {
    auto loc = getLoc();
    auto i1Ty = builder.getI1Type();
    auto lf = genarr(x.left());
    auto rf = genarr(x.right());
    return [=](IterSpace iters) -> ExtValue {
      auto left = fir::getBase(lf(iters));
      auto right = fir::getBase(rf(iters));
      auto lhs = builder.createConvert(loc, i1Ty, left);
      auto rhs = builder.createConvert(loc, i1Ty, right);
      return builder.create<OP>(loc, pred, lhs, rhs);
    };
  }
  template <int KIND>
  CC genarr(const Fortran::evaluate::LogicalOperation<KIND> &x) {
    switch (x.logicalOperator) {
    case Fortran::evaluate::LogicalOperator::And:
      return createBinaryBoolOp<mlir::AndOp>(x);
    case Fortran::evaluate::LogicalOperator::Or:
      return createBinaryBoolOp<mlir::OrOp>(x);
    case Fortran::evaluate::LogicalOperator::Eqv:
      return createCompareBoolOp<mlir::CmpIOp>(mlir::CmpIPredicate::eq, x);
    case Fortran::evaluate::LogicalOperator::Neqv:
      return createCompareBoolOp<mlir::CmpIOp>(mlir::CmpIPredicate::ne, x);
    case Fortran::evaluate::LogicalOperator::Not:
      llvm_unreachable(".NOT. handled elsewhere");
    }
    llvm_unreachable("unhandled case");
  }

  //===--------------------------------------------------------------------===//
  // Relational operators (<, <=, ==, etc.)
  //===--------------------------------------------------------------------===//

  template <typename OP, typename PRED, typename A>
  CC createCompareOp(PRED pred, const A &x) {
    auto loc = getLoc();
    auto lf = genarr(x.left());
    auto rf = genarr(x.right());
    return [=](IterSpace iters) -> ExtValue {
      auto lhs = fir::getBase(lf(iters));
      auto rhs = fir::getBase(rf(iters));
      return builder.create<OP>(loc, pred, lhs, rhs);
    };
  }
  template <typename A>
  CC createCompareCharOp(mlir::CmpIPredicate pred, const A &x) {
    auto loc = getLoc();
    auto lf = genarr(x.left());
    auto rf = genarr(x.right());
    return [=](IterSpace iters) -> ExtValue {
      auto lhs = lf(iters);
      auto rhs = rf(iters);
      return Fortran::lower::genCharCompare(builder, loc, pred, lhs, rhs);
    };
  }
  template <int KIND>
  CC genarr(const Fortran::evaluate::Relational<Fortran::evaluate::Type<
                Fortran::common::TypeCategory::Integer, KIND>> &x) {
    return createCompareOp<mlir::CmpIOp>(translateRelational(x.opr), x);
  }
  template <int KIND>
  CC genarr(const Fortran::evaluate::Relational<Fortran::evaluate::Type<
                Fortran::common::TypeCategory::Character, KIND>> &x) {
    return createCompareCharOp(translateRelational(x.opr), x);
  }
  template <int KIND>
  CC genarr(const Fortran::evaluate::Relational<Fortran::evaluate::Type<
                Fortran::common::TypeCategory::Real, KIND>> &x) {
    return createCompareOp<mlir::CmpFOp>(translateFloatRelational(x.opr), x);
  }
  template <int KIND>
  CC genarr(const Fortran::evaluate::Relational<Fortran::evaluate::Type<
                Fortran::common::TypeCategory::Complex, KIND>> &x) {
    return createCompareOp<fir::CmpcOp>(translateFloatRelational(x.opr), x);
  }
  CC genarr(
      const Fortran::evaluate::Relational<Fortran::evaluate::SomeType> &r) {
    return std::visit([&](const auto &x) { return genarr(x); }, r.u);
  }

  //===--------------------------------------------------------------------===//
  // Boilerplate variants
  //===--------------------------------------------------------------------===//

  template <typename A>
  CC genarr(const Fortran::evaluate::Designator<A> &des) {
    return std::visit([&](const auto &x) { return genarr(x); }, des.u);
  }
  CC genarr(const Fortran::evaluate::DataRef &d) {
    return std::visit([&](const auto &x) { return genarr(x); }, d.u);
  }

private:
  explicit ArrayExprLowering(Fortran::lower::AbstractConverter &converter,
                             Fortran::lower::StatementContext &stmtCtx,
                             Fortran::lower::SymMap &symMap)
      : converter{converter}, builder{converter.getFirOpBuilder()},
        stmtCtx{stmtCtx}, symMap{symMap} {}

  explicit ArrayExprLowering(Fortran::lower::AbstractConverter &converter,
                             Fortran::lower::StatementContext &stmtCtx,
                             Fortran::lower::SymMap &symMap,
                             ConstituentSemantics sem)
      : converter{converter}, builder{converter.getFirOpBuilder()},
        stmtCtx{stmtCtx}, symMap{symMap}, semant{sem} {}

  explicit ArrayExprLowering(Fortran::lower::AbstractConverter &converter,
                             Fortran::lower::StatementContext &stmtCtx,
                             Fortran::lower::SymMap &symMap,
                             ConstituentSemantics sem,
                             Fortran::lower::ExplicitIterSpace *expSpace,
                             Fortran::lower::ImplicitIterSpace *impSpace)
      : converter{converter}, builder{converter.getFirOpBuilder()},
        stmtCtx{stmtCtx}, symMap{symMap},
        explicitSpace(expSpace->empty() ? nullptr : expSpace),
        masks(expSpace->empty() && impSpace->empty() ? nullptr : impSpace),
        semant{sem} {}

  mlir::Location getLoc() { return converter.getCurrentLocation(); }

  /// Array appears in a lhs context such that it is assigned after the rhs is
  /// fully evaluated.
  bool isCopyInCopyOut() {
    return semant == ConstituentSemantics::CopyInCopyOut;
  }

  /// Array appears in a lhs (or temp) context such that a projected,
  /// discontiguous subspace of the array is assigned after the rhs is fully
  /// evaluated. That is, the rhs array value is merged into a section of the
  /// lhs array.
  bool isProjectedCopyInCopyOut() {
    return semant == ConstituentSemantics::ProjectedCopyInCopyOut;
  }

  /// Array appears in a context where it must be boxed.
  bool isBoxValue() { return semant == ConstituentSemantics::BoxValue; }

  /// Array appears in a context where differences in the memory reference can
  /// be observable in the computational results. For example, an array
  /// element is passed to an impure procedure.
  bool isReferentiallyOpaque() {
    return semant == ConstituentSemantics::RefOpaque;
  }

  /// Array appears in a context where it is passed as a VALUE argument.
  bool isValueAttribute() { return semant == ConstituentSemantics::ByValueArg; }

  Fortran::lower::AbstractConverter &converter;
  fir::FirOpBuilder &builder;
  Fortran::lower::StatementContext &stmtCtx;
  Fortran::lower::SymMap &symMap;
  /// The continuation to generate code to update the destination.
  llvm::Optional<CC> ccDest;
  /// The destination is the loaded array into which the results will be
  /// merged.
  fir::ArrayLoadOp destination;
  /// The shape of the destination.
  llvm::SmallVector<mlir::Value> destShape;
  /// List of arrays in the expression that have been loaded.
  llvm::SmallVector<fir::ArrayLoadOp> arrayOperandLoads;
  llvm::SmallVector<mlir::Value> sliceTriple;
  llvm::SmallVector<mlir::Value> slicePath;
  /// If there is a user-defined iteration space, explicitShape will hold the
  /// information from the front end.
  Fortran::lower::ExplicitIterSpace *explicitSpace = nullptr;
  /// Even in an explicitly defined iteration space, one can have an
  /// assignment with rank > 0 and thus an implied shape on a component in the
  /// path.
  llvm::SmallVector<mlir::Value> explicitImpliedShape;
  Fortran::lower::ImplicitIterSpace *masks = nullptr;
  ConstituentSemantics semant = ConstituentSemantics::RefTransparent;
  bool inSlice = false;
  // Does the lhs, if any, must take lbounds from rhs if lhs is reallocated ?
};
} // namespace

fir::ExtendedValue Fortran::lower::createSomeExtendedExpression(
    mlir::Location loc, Fortran::lower::AbstractConverter &converter,
    const Fortran::evaluate::Expr<Fortran::evaluate::SomeType> &expr,
    Fortran::lower::SymMap &symMap, Fortran::lower::StatementContext &stmtCtx) {
  LLVM_DEBUG(expr.AsFortran(llvm::dbgs() << "expr: ") << '\n');
  return ScalarExprLowering{loc, converter, symMap, stmtCtx}.genval(expr);
}

fir::ExtendedValue Fortran::lower::createSomeInitializerExpression(
    mlir::Location loc, Fortran::lower::AbstractConverter &converter,
    const Fortran::evaluate::Expr<Fortran::evaluate::SomeType> &expr,
    Fortran::lower::SymMap &symMap, Fortran::lower::StatementContext &stmtCtx) {
  LLVM_DEBUG(expr.AsFortran(llvm::dbgs() << "expr: ") << '\n');
  return ScalarExprLowering{loc, converter, symMap, stmtCtx,
                            /*initializer=*/true}
      .genval(expr);
}

fir::ExtendedValue Fortran::lower::createSomeExtendedAddress(
    mlir::Location loc, Fortran::lower::AbstractConverter &converter,
    const Fortran::evaluate::Expr<Fortran::evaluate::SomeType> &expr,
    Fortran::lower::SymMap &symMap, Fortran::lower::StatementContext &stmtCtx) {
  LLVM_DEBUG(expr.AsFortran(llvm::dbgs() << "address: ") << '\n');
  return ScalarExprLowering{loc, converter, symMap, stmtCtx}.gen(expr);
}

void Fortran::lower::createSomeArrayAssignment(
    Fortran::lower::AbstractConverter &converter,
    const Fortran::evaluate::Expr<Fortran::evaluate::SomeType> &lhs,
    const Fortran::evaluate::Expr<Fortran::evaluate::SomeType> &rhs,
    Fortran::lower::SymMap &symMap, Fortran::lower::StatementContext &stmtCtx) {
  LLVM_DEBUG(lhs.AsFortran(llvm::dbgs() << "onto array: ") << '\n';
             rhs.AsFortran(llvm::dbgs() << "assign expression: ") << '\n';);
  ArrayExprLowering::lowerArrayAssignment(converter, symMap, stmtCtx, lhs, rhs);
}

void Fortran::lower::createSomeArrayAssignment(
    Fortran::lower::AbstractConverter &converter, const fir::ExtendedValue &lhs,
    const Fortran::evaluate::Expr<Fortran::evaluate::SomeType> &rhs,
    Fortran::lower::SymMap &symMap, Fortran::lower::StatementContext &stmtCtx) {
  LLVM_DEBUG(llvm::dbgs() << "onto array: " << lhs << '\n';
             rhs.AsFortran(llvm::dbgs() << "assign expression: ") << '\n';);
  ArrayExprLowering::lowerArrayAssignment(converter, symMap, stmtCtx, lhs, rhs);
}
void Fortran::lower::createSomeArrayAssignment(
    Fortran::lower::AbstractConverter &converter, const fir::ExtendedValue &lhs,
    const fir::ExtendedValue &rhs, Fortran::lower::SymMap &symMap,
    Fortran::lower::StatementContext &stmtCtx) {
  LLVM_DEBUG(llvm::dbgs() << "onto array: " << lhs << '\n';
             llvm::dbgs() << "assign expression: " << rhs << '\n';);
  ArrayExprLowering::lowerArrayAssignment(converter, symMap, stmtCtx, lhs, rhs);
}

void Fortran::lower::createAnyMaskedArrayAssignment(
    Fortran::lower::AbstractConverter &converter,
    const Fortran::evaluate::Expr<Fortran::evaluate::SomeType> &lhs,
    const Fortran::evaluate::Expr<Fortran::evaluate::SomeType> &rhs,
    Fortran::lower::ExplicitIterSpace &explicitSpace,
    Fortran::lower::ImplicitIterSpace &implicitSpace,
    Fortran::lower::SymMap &symMap, Fortran::lower::StatementContext &stmtCtx) {
  LLVM_DEBUG(lhs.AsFortran(llvm::dbgs() << "onto array: ") << '\n';
             rhs.AsFortran(llvm::dbgs() << "assign expression: ")
             << " given the explicit iteration space:\n"
             << explicitSpace << "\n and implied mask conditions:\n"
             << implicitSpace << '\n';);
  ArrayExprLowering::lowerAnyMaskedArrayAssignment(
      converter, symMap, stmtCtx, lhs, rhs, explicitSpace, implicitSpace);
}

void Fortran::lower::createAllocatableArrayAssignment(
    Fortran::lower::AbstractConverter &converter,
    const fir::MutableBoxValue &lhs,
    const Fortran::evaluate::Expr<Fortran::evaluate::SomeType> &rhs,
    Fortran::lower::ExplicitIterSpace &explicitSpace,
    Fortran::lower::ImplicitIterSpace &implicitSpace,
    Fortran::lower::SymMap &symMap, Fortran::lower::StatementContext &stmtCtx) {
  LLVM_DEBUG(llvm::dbgs() << "defining array: " << lhs << '\n';
             rhs.AsFortran(llvm::dbgs() << "assign expression: ")
             << " given the explicit iteration space:\n"
             << explicitSpace << "\n and implied mask conditions:\n"
             << implicitSpace << '\n';);
  ArrayExprLowering::lowerAllocatableArrayAssignment(
      converter, symMap, stmtCtx, lhs, rhs, explicitSpace, implicitSpace);
}

fir::ExtendedValue Fortran::lower::createSomeArrayTempValue(
    Fortran::lower::AbstractConverter &converter,
    const std::optional<Fortran::evaluate::Shape> &shape,
    const Fortran::evaluate::Expr<Fortran::evaluate::SomeType> &expr,
    Fortran::lower::SymMap &symMap, Fortran::lower::StatementContext &stmtCtx) {
  LLVM_DEBUG(expr.AsFortran(llvm::dbgs() << "array value: ") << '\n');
  return ArrayExprLowering::lowerSomeNewArrayExpression(converter, symMap,
                                                        stmtCtx, shape, expr);
}

fir::ExtendedValue Fortran::lower::createSomeArrayBox(
    Fortran::lower::AbstractConverter &converter,
    const Fortran::evaluate::Expr<Fortran::evaluate::SomeType> &expr,
    Fortran::lower::SymMap &symMap, Fortran::lower::StatementContext &stmtCtx) {
  LLVM_DEBUG(expr.AsFortran(llvm::dbgs() << "box designator: ") << '\n');
  return ArrayExprLowering::lowerBoxedArrayExpression(converter, symMap,
                                                      stmtCtx, expr);
}

fir::MutableBoxValue Fortran::lower::createSomeMutableBox(
    mlir::Location loc, Fortran::lower::AbstractConverter &converter,
    const Fortran::evaluate::Expr<Fortran::evaluate::SomeType> &expr,
    Fortran::lower::SymMap &symMap) {
  // MutableBox lowering StatementContext does not need to be propagated
  // to the caller because the result value is a variable, not a temporary
  // expression. The StatementContext clean-up can occur before using the
  // resulting MutableBoxValue. Variables of all other types are handled in the
  // bridge.
  Fortran::lower::StatementContext dummyStmtCtx;
  return ScalarExprLowering{loc, converter, symMap, dummyStmtCtx}
      .genMutableBoxValue(expr);
}
