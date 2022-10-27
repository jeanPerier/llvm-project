
#include "flang/Evaluate/constant.h"
#include "flang/Optimizer/Builder/BoxValue.h"
#include "flang/Optimizer/Builder/FIRBuilder.h"

namespace Fortran::lower {
template <typename T>
class ConstantBuilderImpl {};

template <common::TypeCategory TC, int KIND>
class ConstantBuilderImpl<evaluate::Type<TC, KIND>> {
public:
  static fir::ExtendedValue
  genScalarLit(fir::FirOpBuilder &builder, mlir::Location loc,
               const evaluate::Scalar<evaluate::Type<TC, KIND>> &value);

  static mlir::Type
  convertToAttribute(fir::FirOpBuilder &builder,
                     const evaluate::Scalar<evaluate::Type<TC, KIND>> &value,
                     llvm::SmallVectorImpl<mlir::Attribute> &outputAttributes);
};

template <common::TypeCategory TC, int KIND>
using ConstantBuilder = ConstantBuilderImpl<evaluate::Type<TC, KIND>>;

template <int KIND>
class ConstantBuilderImpl<
    evaluate::Type<common::TypeCategory::Character, KIND>> {
public:
  static fir::ExtendedValue genScalarLit(
      fir::FirOpBuilder &builder, mlir::Location loc,
      const evaluate::Scalar<
          evaluate::Type<common::TypeCategory::Character, KIND>> &value,
      int64_t len, bool outlineInReadOnlyMemory);
};

template <int KIND>
using ConstantCharBuilder =
    ConstantBuilderImpl<evaluate::Type<common::TypeCategory::Character, KIND>>;

using namespace evaluate;
FOR_EACH_INTRINSIC_KIND(extern template class ConstantBuilderImpl, )

} // namespace Fortran::lower
