// RUN: fir-opt --add-debug-info --mlir-print-debuginfo %s | FileCheck %s
// RUN: fir-opt --add-debug-info --mlir-print-debuginfo %s | FileCheck %s --check-prefix=STORAGE

// Test the module variables that their globals do not describe completely,
// which the scopes of their modules list. Only module d is defined here, and
// the USE statements lead to the scope of module m.
//
// Module m is compiled elsewhere:
//
// module m
//   integer, bind(c, name="c_x") :: x
//   integer :: z(2)
//   integer :: a, b(2:3)
//   common /blk/ a, b
//   integer :: e1(0:3), e2(4)
//   equivalence (e1(1), e2(3))
// end module
//
// subroutine s()
//   use m, only: x, a, e2
//   block
//     use m
//   end block
// end subroutine
//
// Module d is defined here:
//
// module d
//   real :: r(0:1)
//   integer :: c
//   common /dblk/ c
//   integer :: f(2), g
//   equivalence (f(2), g)
// end module

module {
  fir.global @c_x : i32 loc(#loc_m)
  fir.global @_QMmEz : !fir.array<2xi32> loc(#loc_m)
  fir.global common @blk_(dense<0> : vector<12xi8>) : !fir.array<12xi8> loc(#loc_m)
  fir.global @_QMmEe1 : !fir.array<20xi8> loc(#loc_m)
  fir.global @_QMdEr : !fir.array<2xf32> {
    %0 = fir.zero_bits !fir.array<2xf32>
    fir.has_value %0 : !fir.array<2xf32>
  } loc(#loc_r)
  fir.global common @dblk_(dense<0> : vector<4xi8>) : !fir.array<4xi8> loc(#loc_dblk)
  fir.global @_QMdEf : !fir.array<8xi8> {
    %0 = fir.zero_bits !fir.array<8xi8>
    fir.has_value %0 : !fir.array<8xi8>
  } loc(#loc_f)
  func.func @_QPs() {
    %0 = fir.address_of(@c_x) : !fir.ref<i32> loc(#loc_block)
    return loc(#loc_end)
  } loc(#loc_func)
} loc(#loc_module)

#m = #fir.scope<module "m", variables = [
  #fir.module_var<"a" : i32, common "blk", loc(unknown)>,
  #fir.module_var<"b" : !fir.array<2xi32>, common "blk", offset 4, lbounds [2], loc(unknown)>,
  #fir.module_var<"e1" : !fir.array<4xi32>, equivalence "e1", offset 4, lbounds [0], loc(unknown)>,
  #fir.module_var<"e2" : !fir.array<4xi32>, equivalence "e1", loc(unknown)>,
  #fir.module_var<"x", bind_c "c_x", loc(unknown)>], loc(unknown)>
#d = #fir.scope<module "d", variables = [
  #fir.module_var<"c" : i32, common "dblk", loc("d.f90":3:14)>,
  #fir.module_var<"f" : !fir.array<2xi32>, equivalence "f", loc("d.f90":5:14)>,
  #fir.module_var<"g" : i32, equivalence "f", offset 4, loc("d.f90":5:20)>,
  #fir.module_var<"r", lbounds [0], loc("d.f90":2:11)>], loc("d.f90":1:1)>
#s = #fir.scope<procedure "s", uses = [#fir.use<#m, only [<"x">, <"a">, <"e2">], loc("t.f90":2:3)>], loc("t.f90":1:1)>
#b = #fir.scope<construct, parent = #s, uses = [#fir.use<#m, loc("t.f90":4:5)>], loc("t.f90":3:3)>
#loc_m = loc("t.f90":2:3)
#loc_r = loc("d.f90":2:11)
#loc_dblk = loc("d.f90":4:3)
#loc_f = loc("d.f90":5:14)
#loc_module = loc(fused<[#d]>["t.f90":0:0])
#loc_func = loc(fused<#s>["t.f90":1:1])
#loc_block = loc(fused<#b>["t.f90":4:5])
#loc_end = loc("t.f90":6:1)

// CHECK: fir.global @c_x : i32 loc(#[[X_LOC:loc[0-9]+]])
// CHECK: fir.global @_QMmEz : !fir.array<2xi32> loc(#[[Z_LOC:loc[0-9]+]])
// CHECK: fir.global common @blk_({{.*}}) : !fir.array<12xi8> loc(#[[BLK_LOC:loc[0-9]+]])
// CHECK: fir.global @_QMmEe1 : !fir.array<20xi8> loc(#[[E1_LOC:loc[0-9]+]])
// CHECK: fir.global @_QMdEr : !fir.array<2xf32> {
// CHECK: } loc(#[[R_LOC:loc[0-9]+]])
// CHECK: fir.global common @dblk_({{.*}}) : !fir.array<4xi8> loc(#[[DBLK_LOC:loc[0-9]+]])
// CHECK: fir.global @_QMdEf : !fir.array<8xi8> {
// CHECK: } loc(#[[F_LOC:loc[0-9]+]])

// The global of a variable with a binding label is named after it.
// CHECK-DAG: #[[M:.*]] = #llvm.di_module<name = "m", isDecl = true>
// CHECK-DAG: #[[X:.*]] = #llvm.di_global_variable<scope = #[[M]], name = "x", linkageName = "c_x"{{.*}}>
// CHECK-DAG: #[[Z:.*]] = #llvm.di_global_variable<scope = #[[M]], name = "z", linkageName = "_QMmEz"{{.*}}>
// CHECK-DAG: #[[X_LOC]] = loc(fused<[#[[X_EXPR:.*]]]>
// CHECK-DAG: #[[X_EXPR]] = #llvm.di_global_variable_expression<var = #[[X]]>
// CHECK-DAG: #[[Z_LOC]] = loc(fused<[#[[Z_EXPR:.*]]]>
// CHECK-DAG: #[[Z_EXPR]] = #llvm.di_global_variable_expression<var = #[[Z]]>

// The variables of a COMMON block of a module are in it, in the module, at
// their offsets into the global of the COMMON block. The module is defined
// elsewhere, so they are declarations.
// CHECK-DAG: #[[BLK:.*]] = #llvm.di_common_block<scope = #[[M]], name = "blk"{{.*}}>
// CHECK-DAG: #[[B_TYPE:.*]] = #llvm.di_composite_type<tag = DW_TAG_array_type, {{.*}}elements = #llvm.di_subrange<count = 2 : i64, lowerBound = 2 : i64>>
// CHECK-DAG: #[[A:.*]] = #llvm.di_global_variable<scope = #[[BLK]], name = "a", file = #{{.*}}, line = 2, type = #{{.*}}>
// CHECK-DAG: #[[B:.*]] = #llvm.di_global_variable<scope = #[[BLK]], name = "b", file = #{{.*}}, line = 2, type = #[[B_TYPE]]>
// CHECK-DAG: #[[BLK_LOC]] = loc(fused<[#[[A_EXPR:.*]], #[[B_EXPR:.*]]]>
// CHECK-DAG: #[[A_EXPR]] = #llvm.di_global_variable_expression<var = #[[A]]>
// CHECK-DAG: #[[B_EXPR]] = #llvm.di_global_variable_expression<var = #[[B]], expr = <[DW_OP_plus_uconst(4)]>>

// The storage of an EQUIVALENCE is not a variable, although it is named after
// one: its variables are at their offsets into it.
// STORAGE-NOT: linkageName = "_QMmEe1"
// CHECK-DAG: #[[E1_TYPE:.*]] = #llvm.di_composite_type<tag = DW_TAG_array_type, {{.*}}elements = #llvm.di_subrange<count = 4 : i64, lowerBound = 0 : i64>>
// CHECK-DAG: #[[E1:.*]] = #llvm.di_global_variable<scope = #[[M]], name = "e1", file = #{{.*}}, line = 2, type = #[[E1_TYPE]]>
// CHECK-DAG: #[[E2:.*]] = #llvm.di_global_variable<scope = #[[M]], name = "e2", file = #{{.*}}, line = 2, type = #{{.*}}>
// CHECK-DAG: #[[E1_LOC]] = loc(fused<[#[[E2_EXPR:.*]], #[[E1_EXPR:.*]]]>
// CHECK-DAG: #[[E1_EXPR]] = #llvm.di_global_variable_expression<var = #[[E1]], expr = <[DW_OP_plus_uconst(4)]>>
// CHECK-DAG: #[[E2_EXPR]] = #llvm.di_global_variable_expression<var = #[[E2]]>

// A USE statement imports them, and a USE of the whole module in a nested
// scope imports them one by one, like the other variables.
// CHECK-DAG: #di_imported_entity{{.*}} = #llvm.di_imported_entity<tag = DW_TAG_imported_declaration, scope = #[[SP:di_subprogram[0-9]*]], entity = #[[X]], file = #{{.*}}, line = 2>
// CHECK-DAG: #di_imported_entity{{.*}} = #llvm.di_imported_entity<tag = DW_TAG_imported_declaration, scope = #[[SP]], entity = #[[A]], file = #{{.*}}, line = 2>
// CHECK-DAG: #di_imported_entity{{.*}} = #llvm.di_imported_entity<tag = DW_TAG_imported_declaration, scope = #[[SP]], entity = #[[E2]], file = #{{.*}}, line = 2>
// CHECK-DAG: #di_imported_entity{{.*}} = #llvm.di_imported_entity<tag = DW_TAG_imported_module, scope = #[[LB:di_lexical_block[0-9]*]], entity = #[[M]], file = #{{.*}}, line = 4>
// CHECK-DAG: #di_imported_entity{{.*}} = #llvm.di_imported_entity<tag = DW_TAG_imported_declaration, scope = #[[LB]], entity = #[[X]], file = #{{.*}}, line = 4>
// CHECK-DAG: #di_imported_entity{{.*}} = #llvm.di_imported_entity<tag = DW_TAG_imported_declaration, scope = #[[LB]], entity = #[[Z]], file = #{{.*}}, line = 4>
// CHECK-DAG: #di_imported_entity{{.*}} = #llvm.di_imported_entity<tag = DW_TAG_imported_declaration, scope = #[[LB]], entity = #[[A]], file = #{{.*}}, line = 4>
// CHECK-DAG: #di_imported_entity{{.*}} = #llvm.di_imported_entity<tag = DW_TAG_imported_declaration, scope = #[[LB]], entity = #[[B]], file = #{{.*}}, line = 4>
// CHECK-DAG: #di_imported_entity{{.*}} = #llvm.di_imported_entity<tag = DW_TAG_imported_declaration, scope = #[[LB]], entity = #[[E1]], file = #{{.*}}, line = 4>
// CHECK-DAG: #di_imported_entity{{.*}} = #llvm.di_imported_entity<tag = DW_TAG_imported_declaration, scope = #[[LB]], entity = #[[E2]], file = #{{.*}}, line = 4>

// The variables of a module defined here are described whether they are used
// or not, with the lower bounds that their FIR types do not have.
// CHECK-DAG: #[[D:.*]] = #llvm.di_module<file = #{{.*}}, scope = #{{.*}}, name = "d", line = 1>
// CHECK-DAG: #[[R_TYPE:.*]] = #llvm.di_composite_type<tag = DW_TAG_array_type, {{.*}}elements = #llvm.di_subrange<count = 2 : i64, lowerBound = 0 : i64>>
// CHECK-DAG: #[[R:.*]] = #llvm.di_global_variable<scope = #[[D]], name = "r", linkageName = "_QMdEr", file = #{{.*}}, line = 2, type = #[[R_TYPE]], isDefined = true>
// CHECK-DAG: #[[R_LOC]] = loc(fused<[#[[R_EXPR:.*]]]>
// CHECK-DAG: #[[R_EXPR]] = #llvm.di_global_variable_expression<var = #[[R]]>
// CHECK-DAG: #[[DBLK:.*]] = #llvm.di_common_block<scope = #[[D]], name = "dblk", file = #{{.*}}, line = 4>
// CHECK-DAG: #[[C:.*]] = #llvm.di_global_variable<scope = #[[DBLK]], name = "c", file = #{{.*}}, line = 3, type = #{{.*}}, isDefined = true>
// CHECK-DAG: #[[DBLK_LOC]] = loc(fused<[#[[C_EXPR:.*]]]>
// CHECK-DAG: #[[C_EXPR]] = #llvm.di_global_variable_expression<var = #[[C]]>
// CHECK-DAG: #[[F:.*]] = #llvm.di_global_variable<scope = #[[D]], name = "f", file = #{{.*}}, line = 5, type = #{{.*}}, isDefined = true>
// CHECK-DAG: #[[G:.*]] = #llvm.di_global_variable<scope = #[[D]], name = "g", file = #{{.*}}, line = 5, type = #{{.*}}, isDefined = true>
// CHECK-DAG: #[[F_LOC]] = loc(fused<[#[[F_EXPR:.*]], #[[G_EXPR:.*]]]>
// CHECK-DAG: #[[F_EXPR]] = #llvm.di_global_variable_expression<var = #[[F]]>
// CHECK-DAG: #[[G_EXPR]] = #llvm.di_global_variable_expression<var = #[[G]], expr = <[DW_OP_plus_uconst(4)]>>
