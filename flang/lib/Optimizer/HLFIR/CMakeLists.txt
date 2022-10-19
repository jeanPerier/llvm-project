add_flang_library(HLFIRDialect
  HLFIRDialect.cpp

  DEPENDS
  FIRDialect
  FIRSupport
  HLFIROpsIncGen

  LINK_LIBS
  FIRDialect
  FIRSupport

  LINK_COMPONENTS
  AsmParser
  AsmPrinter
  Remarks
)
