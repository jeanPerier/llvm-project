
add_flang_library(FIRBuilder
  BoxValue.cpp
  Character.cpp
  Complex.cpp
  DoLoopHelper.cpp
  FIRBuilder.cpp
  MutableBox.cpp

  DEPENDS
  FIRDialect
  FIRSupport

  LINK_LIBS
  FIRDialect
  FIRSupport
)
