// RUN: fir-opt --add-debug-info --mlir-print-debuginfo %s | FileCheck %s

// Test the imports of USE statements in BLOCK constructs. The module variables
// are found from the names in the USE descriptions through the fir.global
// names, without a declaration in the procedure.

// subroutine s(n)
//   use m1
//   integer :: n
//   block
//     use m2, only: a, c => b
//     integer :: n
//     n = a + c
//     block
//       use m1
//       n = n + v
//     end block
//   end block
// end subroutine

module {
  fir.global @_QMm1Ev : i32 loc(fused<#fir.global_name<"_QMm1Ev">>["m.f90":2:14])
  fir.global @_QMm2Ea : i32 loc(fused<#fir.global_name<"_QMm2Ea">>["m.f90":2:14])
  fir.global @_QMm2Eb : i32 loc(fused<#fir.global_name<"_QMm2Eb">>["m.f90":2:14])
  func.func @_QPs(%arg0: !fir.ref<i32>) {
    %0 = fircg.ext_declare %arg0 uniq_name("_QFsEn") : (!fir.ref<i32>) -> !fir.ref<i32> loc(#loc_n)
    %1 = fir.alloca i32 loc(#loc_b1_n)
    %2 = fircg.ext_declare %1 uniq_name("_QFsB1En") : (!fir.ref<i32>) -> !fir.ref<i32> loc(#loc_b1_n)
    %3 = fir.address_of(@_QMm2Ea) : !fir.ref<i32> loc(#loc_b1_stmt)
    %4 = fir.load %3 : !fir.ref<i32> loc(#loc_b1_stmt)
    fir.store %4 to %2 : !fir.ref<i32> loc(#loc_b1_stmt)
    %5 = fir.load %2 : !fir.ref<i32> loc(#loc_b2_stmt)
    fir.store %5 to %2 : !fir.ref<i32> loc(#loc_b2_stmt)
    return loc(#loc_end)
  } loc(#loc_func)
} loc(#loc_module)

#m1_mod = #fir.scope<module "m1", exports = [<"v">], loc(unknown)>
#m2_mod = #fir.scope<module "m2", loc(unknown)>
#use_m1 = #fir.use<"m1", loc("t.f90":2:3)>
#s = #fir.scope<procedure "s", uses = [#use_m1], loc("t.f90":1:1)>
#b1 = #fir.scope<construct, parent = #s, uses = [#fir.use<"m2", only [<"a">, <"c" -> "b">], loc("t.f90":5:5)>], loc("t.f90":4:3)>
#b2 = #fir.scope<construct, parent = #b1, uses = [#fir.use<"m1", loc("t.f90":9:7)>], loc("t.f90":8:5)>
#loc_module = loc(fused<[#m1_mod, #m2_mod]>["t.f90":0:0])
#loc_func = loc(fused<#s>["t.f90":1:1])
#loc_n = loc("t.f90":3:14)
#loc_b1_n = loc(fused<#b1>["t.f90":6:16])
#loc_b1_stmt = loc(fused<#b1>["t.f90":7:5])
#loc_b2_stmt = loc(fused<#b2>["t.f90":10:7])
#loc_end = loc("t.f90":13:1)

// The lexical blocks are built on the recursive self reference of the
// subprogram so that the imports, the variables and the code use the same
// blocks.
// CHECK-DAG: #[[M1:.*]] = #llvm.di_module<{{.*}}name = "m1"{{.*}}>
// CHECK-DAG: #[[SELF:.*]] = #llvm.di_subprogram<recId = distinct{{\[}}[[REC:[0-9]+]]{{\]}}<>, isRecSelf = true{{.*}}>
// CHECK-DAG: #[[LB1:.*]] = #llvm.di_lexical_block<scope = #[[SELF]], file = #{{.*}}, line = 4, column = 3>
// CHECK-DAG: #[[LB2:.*]] = #llvm.di_lexical_block<scope = #[[LB1]], file = #{{.*}}, line = 8, column = 5>
// CHECK-DAG: #[[A:.*]] = #llvm.di_global_variable<scope = #{{.*}}, name = "a", linkageName = "_QMm2Ea"{{.*}}>
// CHECK-DAG: #[[B:.*]] = #llvm.di_global_variable<scope = #{{.*}}, name = "b", linkageName = "_QMm2Eb"{{.*}}>
// CHECK-DAG: #[[V:.*]] = #llvm.di_global_variable<scope = #{{.*}}, name = "v", linkageName = "_QMm1Ev"{{.*}}>
// CHECK-DAG: #[[IMP_S:.*]] = #llvm.di_imported_entity<tag = DW_TAG_imported_module, scope = #[[SELF]], entity = #[[M1]], file = #{{.*}}, line = 2>
// CHECK-DAG: #[[IMP_A:.*]] = #llvm.di_imported_entity<tag = DW_TAG_imported_declaration, scope = #[[LB1]], entity = #[[A]], file = #{{.*}}, line = 5>
// CHECK-DAG: #[[IMP_C:.*]] = #llvm.di_imported_entity<tag = DW_TAG_imported_declaration, scope = #[[LB1]], entity = #[[B]], file = #{{.*}}, line = 5, name = "c">
// The nested whole module import also imports the variables of m1 one by one,
// so that they hide what the outer scopes import.
// CHECK-DAG: #[[IMP_M1:.*]] = #llvm.di_imported_entity<tag = DW_TAG_imported_module, scope = #[[LB2]], entity = #[[M1]], file = #{{.*}}, line = 9>
// CHECK-DAG: #[[IMP_V:.*]] = #llvm.di_imported_entity<tag = DW_TAG_imported_declaration, scope = #[[LB2]], entity = #[[V]], file = #{{.*}}, line = 9>
// CHECK-DAG: #[[SP:.*]] = #llvm.di_subprogram<recId = distinct{{\[}}[[REC]]{{\]}}<>, {{.*}}name = "s"{{.*}}retainedNodes = [#[[IMP_S]], #[[IMP_A]], #[[IMP_C]], #[[IMP_M1]], #[[IMP_V]]]>
// CHECK-DAG: #llvm.di_local_variable<scope = #[[SP]], name = "n", {{.*}}line = 3{{.*}}>
// CHECK-DAG: #llvm.di_local_variable<scope = #[[LB1]], name = "n", {{.*}}line = 6{{.*}}>
// CHECK-DAG: #[[L10:.*]] = loc("t.f90":10:7)
// CHECK-DAG: #[[L10_B2:.*]] = loc(fused<#{{.*}}>[#[[L10]]])
// CHECK-DAG: loc(fused<#[[LB2]]>[#[[L10_B2]]])
