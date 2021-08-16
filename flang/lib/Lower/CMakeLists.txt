
get_property(dialect_libs GLOBAL PROPERTY MLIR_DIALECT_LIBS)

add_flang_library(FortranLower
  Allocatable.cpp
  Bridge.cpp
  CallInterface.cpp
  CharacterRuntime.cpp
  Coarray.cpp
  ConvertExpr.cpp
  ConvertType.cpp
  ConvertVariable.cpp
  DerivedRuntime.cpp
  HostAssociations.cpp
  IntrinsicCall.cpp
  IO.cpp
  Mangler.cpp
  NumericRuntime.cpp
  OpenACC.cpp
  OpenMP.cpp
  PFTBuilder.cpp
  ReductionRuntime.cpp
  Runtime.cpp
  SymbolMap.cpp
  TransformationalRuntime.cpp

  DEPENDS
  FIRBuilder
  FIRDialect
  FIRSupport
  FIRTransforms
  ${dialect_libs}
  omp_gen
  acc_gen

  LINK_LIBS
  FIRBuilder
  FIRDialect
  FIRSupport
  FIRTransforms
  ${dialect_libs}
  FortranCommon
  FortranParser
  FortranEvaluate
  FortranSemantics
  MLIRAffineToStandard
  MLIRLLVMIR
  MLIRSCFToStandard
  MLIRStandard

  LINK_COMPONENTS
  Support
)
