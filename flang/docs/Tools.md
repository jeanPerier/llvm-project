# Flang Development Tools

Flang is a project to build a highly optimizing compiler for the latest Fortran
standard (2018). While flang is a distinct compiler, it is part of the LLVM
project tree and makes use of components from clang, mlir, llvm, and other
projects.

Flang is structured into distinct phases. Scanning, preprocessing, parsing,
semantics checking, lowering to a high-level IR (FIR), high-level optimization,
generating low-level IR code (LLVM), low-level optimization, etc.

Flang developers have created tools to aid in the development and testing of
these various phases of the compiler and rely heavily on these tools. The Flang
developer community reserves the right to build, maintain, and use tools they
develop for their productivity and convenience. These tools may or may not use
other LLVM project components in the form of libraries in ways that are deemded
appropriate and practical by the flang developer community.

Having alternative tools allows for smaller and more feature focused tools to be
built and provide easier, more accessible debugging. Separate tools that do not
use precisely the same options and transformations also aid in exposing
engineering design problems which may otherwise be hidden and avoid bugs that
cancel out other upstream bugs.

## Tools

### bbc

This tool is for testing the passes that lower the data structures produced by
the parser and semantics checking to FIR. There is currently no way to marshal
the results of parsing + semantics out to a text file, so bbc uses Fortran
source as an input instead. bbc emits FIR as an output. bbc also includes some
predefined high-level optimization passes for writing tests against some of the
successive rewrite passes.

Ideally, the input to this tool would be the output of semantics checking.
For now, though, its input is Fortran source input instead.

### tco

This tool is for testing the passes that transform FIR to LLVM. Converting FIR
to LLVM requires a series of passes to be run in a specific order. It includes
the same high-level optimization passes as bbc. It takes FIR as an input and
produces LLVM IR as output.

This tool tests from high-level optimizations through the generation of LLVM
IR as designed and intended.

### fir-opt

This tool is used for testing high-level optimization passes in isolation. As
such it takes FIR as an input and emits FIR as an output.
