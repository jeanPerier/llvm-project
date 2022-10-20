add_flang_library(HLFIRDialect
  HLFIRDialect.cpp

  DEPENDS
  FIRDialect
  HLFIROpsIncGen

  LINK_LIBS
  FIRDialect

  LINK_COMPONENTS
  AsmParser
  AsmPrinter
  Remarks
)
