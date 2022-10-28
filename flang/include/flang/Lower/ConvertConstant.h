
#include "flang/Evaluate/constant.h"
#include "flang/Lower/Support/Utils.h"
#include "flang/Optimizer/Builder/BoxValue.h"
#include "flang/Optimizer/Builder/FIRBuilder.h"

namespace Fortran::lower {
template <typename T>
class ConstantBuilder {};

template <common::TypeCategory TC, int KIND>
class ConstantBuilder<evaluate::Type<TC, KIND>> {
public:
  static fir::ExtendedValue
  gen(fir::FirOpBuilder &builder, mlir::Location loc,
      const evaluate::Constant<evaluate::Type<TC, KIND>> &constant,
      bool outlineCharacterScalarInReadOnlyMemory);
};

template <common::TypeCategory TC, int KIND>
using IntrinsicConstantBuilder = ConstantBuilder<evaluate::Type<TC, KIND>>;

using namespace evaluate;
FOR_EACH_INTRINSIC_KIND(extern template class ConstantBuilder, )

/// Create a global array symbol with the Dense attribute
fir::GlobalOp tryCreatingDenseGlobal(fir::FirOpBuilder &builder,
                                     mlir::Location loc, mlir::Type symTy,
                                     llvm::StringRef globalName,
                                     mlir::StringAttr linkage, bool isConst,
                                     const Fortran::lower::SomeExpr &initExpr);

} // namespace Fortran::lower
