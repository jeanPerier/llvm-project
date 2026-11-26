// RUN: mlir-translate -mlir-to-llvmir %s | FileCheck %s

// Test that the map names use the file position of a fused, named or call site
// location.

#file = #llvm.di_file<"t.f90" in "">
#cu = #llvm.di_compile_unit<id = distinct[0]<>, sourceLanguage = DW_LANG_Fortran95,
 file = #file, isOptimized = false, emissionKind = Full>
#st = #llvm.di_subroutine_type<>
#sp = #llvm.di_subprogram<id = distinct[1]<>, compileUnit = #cu, scope = #file,
 name = "f", file = #file, line = 1, scopeLine = 1,
 subprogramFlags = Definition, type = #st>

module attributes {omp.target_triples = ["amdgcn-amd-amdhsa"]} {
  llvm.func @f() {
    %0 = llvm.mlir.constant(1 : i64) : i64
    %1 = llvm.alloca %0 x i32 : (i64) -> !llvm.ptr
    %2 = omp.map.info var_ptr(%1 : !llvm.ptr, i32) map_clauses(to) capture(ByRef) -> !llvm.ptr loc(fused<"scope">["fused.f90":3:4])
    %3 = omp.map.info var_ptr(%1 : !llvm.ptr, i32) map_clauses(to) capture(ByRef) -> !llvm.ptr loc("x"(fused<"scope">["named.f90":5:6]))
    %4 = omp.map.info var_ptr(%1 : !llvm.ptr, i32) map_clauses(to) capture(ByRef) -> !llvm.ptr loc(callsite("callee.f90":7:8 at "caller.f90":9:10))
    omp.target_enter_data map_entries(%2, %3, %4 : !llvm.ptr, !llvm.ptr, !llvm.ptr)
    llvm.return
  } loc(fused<#sp>["t.f90":1:1])
}

// CHECK-DAG: @[[FUSED:[0-9]+]] = private unnamed_addr constant {{.*}} c";unknown;fused.f90;3;4;;\00"
// CHECK-DAG: @[[NAMED:[0-9]+]] = private unnamed_addr constant {{.*}} c";x;named.f90;5;6;;\00"
// CHECK-DAG: @[[CALLSITE:[0-9]+]] = private unnamed_addr constant {{.*}} c";unknown;callee.f90;7;8;;\00"
// CHECK: @.offload_mapnames = private constant [3 x ptr] [ptr @[[FUSED]], ptr @[[NAMED]], ptr @[[CALLSITE]]]
