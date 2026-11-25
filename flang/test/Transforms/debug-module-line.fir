// RUN: fir-opt --add-debug-info --mlir-print-debuginfo %s | FileCheck %s

// The line of a DIModuleAttr comes from the location of the module's scope in
// the location of the builtin.module, which is that of the MODULE statement. A
// module not listed there with a known location is not defined in this
// compilation unit, so it is described as a declaration and carries no line at
// all.

// A module whose MODULE statement is on line 4, well before its first member on
// line 20. It has no USE.
#no_use = #fir.scope<module "no_use", loc("mod.f90":4:1)>
// A module with no members of its own, only used by `with_use` below.
#other_mod = #fir.scope<module "other_mod", loc("mod.f90":6:1)>
// Same as `no_use`, but with a USE statement.
#with_use = #fir.scope<module "with_use", uses = [#fir.use<"other_mod", loc("mod.f90":9:3)>], loc("mod.f90":8:1)>

module {
  fir.global @_QMno_useEx : i32 {
    %0 = fir.zero_bits i32
    fir.has_value %0 : i32
  } loc(#loc_x)

  fir.global @_QMwith_useEy : i32 {
    %0 = fir.zero_bits i32
    fir.has_value %0 : i32
  } loc(#loc_y)

  // Not listed in the module location, so this module is only used here.
  fir.global @_QMlegacyEz : i32 {
    %0 = fir.zero_bits i32
    fir.has_value %0 : i32
  } loc(#loc_z)
} loc(fused<[#no_use, #other_mod, #with_use]>["mod.f90":0:0])
#loc_x = loc(fused<#fir.global_name<"_QMno_useEx">>["mod.f90":20:14])
#loc_y = loc(fused<#fir.global_name<"_QMwith_useEy">>["mod.f90":24:14])
#loc_z = loc("mod.f90":30:14)

// The MODULE statement line is used, not 19 as guessed from the member.
// CHECK-DAG: #llvm.di_module<{{.*}}name = "no_use", line = 4>

// CHECK-DAG: #llvm.di_module<{{.*}}name = "with_use", line = 8>

// CHECK-DAG: #llvm.di_module<name = "legacy", isDecl = true>
