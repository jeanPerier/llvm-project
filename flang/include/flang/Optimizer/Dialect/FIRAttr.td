//===- FIRAttr.td - FIR Attributes -------------------------*- tablegen -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file declares the FIR dialect attributes.
//
//===----------------------------------------------------------------------===//

#ifndef FIR_DIALECT_FIR_ATTRS
#define FIR_DIALECT_FIR_ATTRS

include "flang/Optimizer/Dialect/FIRDialect.td"
include "mlir/Dialect/OpenACC/OpenACCAttributes.td"
include "mlir/IR/CommonAttrConstraints.td"
include "mlir/IR/EnumAttr.td"
include "mlir/IR/OpAsmInterface.td"

class fir_Attr<string name, list<Trait> traits = []>
    : AttrDef<FIROpsDialect, name, traits> {
  let cppIncludes = ["flang/Optimizer/Dialect/FIRAttr.h"];
}

//===----------------------------------------------------------------------===//
// Linkage
//===----------------------------------------------------------------------===//

// Linkage attribute used on fir.global. This is a subset of LLVM linkage
// types currently supported by FIR. The names match the keywords in the
// custom assembly form of fir.global. External is the default when omitted.
def FIR_LinkageExternal
    : I64EnumAttrCase<"External", 0, "external">;
def FIR_LinkageInternal
    : I64EnumAttrCase<"Internal", 1, "internal">;
def FIR_LinkageLinkonce
    : I64EnumAttrCase<"Linkonce", 2, "linkonce">;
def FIR_LinkageLinkonceODR
    : I64EnumAttrCase<"LinkonceODR", 3, "linkonce_odr">;
def FIR_LinkageWeak
    : I64EnumAttrCase<"Weak", 4, "weak">;
def FIR_LinkageCommon
    : I64EnumAttrCase<"Common", 5, "common">;

def fir_LinkageEnum : I64EnumAttr<
    "LinkageEnum",
    "FIR linkage types",
    [FIR_LinkageExternal, FIR_LinkageInternal, FIR_LinkageLinkonce,
     FIR_LinkageLinkonceODR, FIR_LinkageWeak, FIR_LinkageCommon]> {
  let cppNamespace = "::fir";
  let genSpecializedAttr = 0;
}

def fir_LinkageAttr : fir_Attr<"Linkage"> {
  let mnemonic = "linkage";
  let parameters = (ins "LinkageEnum":$linkage);
  let assemblyFormat = "`<` $linkage `>`";
}

def fir_Linkage : DialectAttr<
    FIROpsDialect,
    CPred<"::llvm::isa<::fir::LinkageAttr>($_self)">,
    "FIR linkage specification"> {
  let storageType = "::fir::LinkageAttr";
  let returnType = "::fir::LinkageEnum";
  let convertFromStorage = "$_self.getLinkage()";
  let constBuilderCall =
          "::fir::LinkageAttr::get($_builder.getContext(), $0)";
}

def FIRnoAttributes  : I32BitEnumAttrCaseNone<"None">;
def FIRallocatable  : I32BitEnumAttrCaseBit<"allocatable", 0>;
def FIRasynchronous : I32BitEnumAttrCaseBit<"asynchronous", 1>;
def FIRbind_c       : I32BitEnumAttrCaseBit<"bind_c", 2>;
def FIRcontiguous   : I32BitEnumAttrCaseBit<"contiguous", 3>;
def FIRintent_in    : I32BitEnumAttrCaseBit<"intent_in", 4>;
def FIRintent_inout : I32BitEnumAttrCaseBit<"intent_inout", 5>;
def FIRintent_out   : I32BitEnumAttrCaseBit<"intent_out", 6>;
def FIRoptional     : I32BitEnumAttrCaseBit<"optional", 7>;
def FIRparameter    : I32BitEnumAttrCaseBit<"parameter", 8>;
def FIRpointer      : I32BitEnumAttrCaseBit<"pointer", 9>;
def FIRtarget       : I32BitEnumAttrCaseBit<"target", 10>;
def FIRvalue        : I32BitEnumAttrCaseBit<"value", 11>;
def FIRvolatile     : I32BitEnumAttrCaseBit<"fortran_volatile", 12, "volatile">;
// Used inside internal procedure to flag variables host associated from parent procedure.
def FIRHostAssoc    : I32BitEnumAttrCaseBit<"host_assoc", 13>;
// Marks storage accessible to internal procedures through host association.
def FIRInternalAssoc    : I32BitEnumAttrCaseBit<"internal_assoc", 14>;
// Used by alias analysis
def FIRcray_pointer : I32BitEnumAttrCaseBit<"cray_pointer", 15>;
def FIRcray_pointee : I32BitEnumAttrCaseBit<"cray_pointee", 16>;

def fir_FortranVariableFlagsEnum : I32BitEnumAttr<
    "FortranVariableFlagsEnum",
    "Fortran variable attributes",
    [FIRnoAttributes, FIRallocatable, FIRasynchronous, FIRbind_c, FIRcontiguous,
     FIRintent_in, FIRintent_inout, FIRintent_out, FIRoptional, FIRparameter,
     FIRpointer, FIRtarget, FIRvalue, FIRvolatile, FIRHostAssoc, FIRInternalAssoc,
     FIRcray_pointer, FIRcray_pointee]> {
  let separator = ", ";
  let cppNamespace = "::fir";
  let printBitEnumPrimaryGroups = 1;
}

def fir_FortranVariableFlagsAttr : fir_Attr<"FortranVariableFlags"> {
  let mnemonic = "var_attrs";

  let parameters = (ins
    "FortranVariableFlagsEnum":$flags
  );
  let hasCustomAssemblyFormat = 1;
  let returnType = "::fir::FortranVariableFlagsEnum";
  let convertFromStorage = "$_self.getFlags()";
  let constBuilderCall =
        "::fir::FortranVariableFlagsAttr::get($_builder.getContext(), $0)";
}


/// Fortran procedure attributes (F2023 15.6.2.1). BIND attribute (18.3.7)
/// is also tracked in the same enum. Recursive (resp. Impure) attribute
/// is implied by the absence of opposite NonRecursive (resp. Pure) attribute.
/// Beware that "elemental" does not implicitly imply "pure" as it does in
/// Fortran, "pure" must be made explicit when generating the FIR attribute.
def FIRfuncNoAttributes  : I32BitEnumAttrCaseNone<"none">;
def FIRfuncElemental     : I32BitEnumAttrCaseBit<"elemental", 0>;
def FIRfuncPure          : I32BitEnumAttrCaseBit<"pure", 1>;
def FIRfuncNonRecursive  : I32BitEnumAttrCaseBit<"non_recursive", 2>;
def FIRfuncSimple        : I32BitEnumAttrCaseBit<"simple", 3>;
def FIRfuncBind_c        : I32BitEnumAttrCaseBit<"bind_c", 4>;
def FIRfuncRecursive     : I32BitEnumAttrCaseBit<"recursive", 5>;
/// We also track whether a procedure call is to an "intrinsic", meaning that the
/// callee was found in an instrinsic module. This allows us to make sure calls are
/// to known intrinsic functions and not user-defined functions with the same
/// name when optimizing.
def FIRfuncIntrinsic     : I32BitEnumAttrCaseBit<"intrinsic", 6>;

def fir_FortranProcedureFlagsEnum
    : I32BitEnumAttr<"FortranProcedureFlagsEnum",
                     "Fortran procedure attributes",
                     [FIRfuncNoAttributes, FIRfuncElemental, FIRfuncPure,
                      FIRfuncNonRecursive, FIRfuncSimple, FIRfuncBind_c,
                      FIRfuncRecursive, FIRfuncIntrinsic]> {
  let separator = ", ";
  let cppNamespace = "::fir";
  let genSpecializedAttr = 0;
  let printBitEnumPrimaryGroups = 1;
}

def fir_FortranProcedureFlagsAttr :
    EnumAttr<FIROpsDialect, fir_FortranProcedureFlagsEnum, "proc_attrs">;

def fir_BoxFieldAttr : I32EnumAttr<
    "BoxFieldAttr", "",
    [
      I32EnumAttrCase<"base_addr", 0>,
      I32EnumAttrCase<"derived_type", 1>
    ]> {
  let cppNamespace = "fir";
}

def fir_ReduceOperationEnum : I32BitEnumAttr<"ReduceOperationEnum",
    "intrinsic operations and functions supported by DO CONCURRENT REDUCE",
    [
      I32BitEnumAttrCaseBit<"Add", 0, "add">,
      I32BitEnumAttrCaseBit<"Multiply", 1, "multiply">,
      I32BitEnumAttrCaseBit<"AND", 2, "and">,
      I32BitEnumAttrCaseBit<"OR", 3, "or">,
      I32BitEnumAttrCaseBit<"EQV", 4, "eqv">,
      I32BitEnumAttrCaseBit<"NEQV", 5, "neqv">,
      I32BitEnumAttrCaseBit<"MAX", 6, "max">,
      I32BitEnumAttrCaseBit<"MIN", 7, "min">,
      I32BitEnumAttrCaseBit<"IAND", 8, "iand">,
      I32BitEnumAttrCaseBit<"IOR", 9, "ior">,
      I32BitEnumAttrCaseBit<"IEOR", 10, "ieor">
    ]> {
  let separator = ", ";
  let cppNamespace = "::fir";
  let printBitEnumPrimaryGroups = 1;
}

def fir_ReduceAttr : fir_Attr<"Reduce"> {
  let mnemonic = "reduce_attr";

  let parameters = (ins
    "ReduceOperationEnum":$reduce_operation
  );

  let assemblyFormat = "`<` $reduce_operation `>`";
}

// mlir::SideEffects::Resource for modelling operations which add debugging information
def DebuggingResource : Resource<"::fir::DebuggingResource">;

def FortranRuntimeResource : Resource<"::fir::FortranRuntimeResource">;

def fir_RuntimeCheckKind : I32EnumAttr<"RuntimeCheckKind",
    "Fortran runtime check category", [
      I32EnumAttrCase<"status", 0>,
      I32EnumAttrCase<"conformance", 1>,
      I32EnumAttrCase<"argument", 2>,
      I32EnumAttrCase<"bounds", 3>,
      I32EnumAttrCase<"pointer", 4>,
      I32EnumAttrCase<"doLoop", 5, "do">,
      I32EnumAttrCase<"bits", 6>,
      I32EnumAttrCase<"divide", 7>
    ]> {
  let genSpecializedAttr = 0;
  let cppNamespace = "::fir";
}
def fir_RuntimeCheckKindAttr
    : EnumAttr<FIROpsDialect, fir_RuntimeCheckKind, "runtime_check_kind">;

def fir_LowerBoundModifierAttribute : I32EnumAttr<
    "LowerBoundModifierAttribute",
    "Describes how to modify lower bounds",
    [
      I32EnumAttrCase<"Preserve", 0, "preserve">,
      I32EnumAttrCase<"SetToOnes", 1, "ones">,
      I32EnumAttrCase<"SetToZeroes", 2, "zeroes">,
    ]> {
  let cppNamespace = "::fir";
}

def fir_LocationKind : I32EnumAttr<"LocationKind", "Flang location kind",
  [
    I32EnumAttrCase<"Base", 0, "base">,
    I32EnumAttrCase<"Inclusion", 1, "inclusion">,
  ]> {
  let genSpecializedAttr = 0;
  let cppNamespace = "::fir";
}
def fir_LocationKindAttr : EnumAttr<FIROpsDialect, fir_LocationKind, "loc_kind">;

def LocationKindArrayAttr : ArrayOfAttr<FIROpsDialect, "LocationKindArray",
    "loc_kind_array", "LocationKindAttr">;

def fir_ScopeKind : I32EnumAttr<"ScopeKind", "Kind of a Fortran scope", [
    I32EnumAttrCase<"Module", 0, "module">,
    I32EnumAttrCase<"Procedure", 1, "procedure">,
    I32EnumAttrCase<"Construct", 2, "construct">,
  ]> {
  let genSpecializedAttr = 0;
  let cppNamespace = "::fir";
}

// A string parameter that is not parsed when absent, so that it can lead an
// optional group.
class fir_OptionalStringParameter<string desc = "">
    : OptionalParameter<"mlir::StringAttr", desc> {
  let parser = [{
    [&]() -> mlir::FailureOr<mlir::StringAttr> {
      auto str = mlir::FieldParser<std::optional<mlir::StringAttr>>::parse(
          $_parser);
      if (mlir::failed(str))
        return mlir::failure();
      return str->value_or(mlir::StringAttr{});
    }()
  }];
}

def fir_ScopeAttr : fir_Attr<"Scope",
    [DeclareAttrInterfaceMethods<OpAsmAttrInterface, ["getAlias"]>]> {
  let mnemonic = "scope";
  let summary = "A Fortran scope, for debug information";
  let description = [{
    Describes a module or submodule, a procedure, or a construct scope in a
    procedure: a BLOCK construct, or a branch of a construct that associates
    names such as ASSOCIATE or SELECT TYPE. It lives in locations, so that it
    costs no operation, survives the transformations that keep locations, and
    goes away with the rest of the debug information:

    The procedure and construct scopes of executable code are fused into
    its locations. The compilation unit lists module scopes. A module read
    from a module file has an unknown opening location. A module's `exports`
    list describes its public data objects under their accessible names,
    including objects made accessible by use association. Each element can
    name the ultimate declaring module. USE statements refer to modules by
    Fortran name, so a chain of USE statements does not nest these lists.

    `parent` is the lexical host, or the parent of a submodule. `uses`
    records the USE statements written in the scope.

    ```
    #scope_mod_m = #fir.scope<module "m", loc("a.f90":1:1)>
    #scope_mod_other = #fir.scope<module "other", loc(unknown)>
    #scope = #fir.scope<procedure "s", parent = #scope_mod_m,
        uses = [#fir.use<"other", loc("a.f90":6:3)>],
        loc("a.f90":5:1)>
    #scope1 = #fir.scope<construct, parent = #scope, loc("a.f90":9:3)>
    ```
  }];
  let parameters = (ins EnumParameter<fir_ScopeKind>:$kind,
      fir_OptionalStringParameter<>:$name,
      "mlir::LocationAttr":$loc,
      OptionalParameter<"::fir::ScopeAttr">:$parent,
      OptionalArrayRefParameter<"::fir::UseAttr">:$uses,
      OptionalArrayRefParameter<"::fir::UseEntityAttr">:$exports);
  let assemblyFormat = [{
    `<` $kind ($name^)? `,` (`parent` `=` qualified($parent)^ `,`)?
    (`uses` `=` `[` qualified($uses)^ `]` `,`)?
    (`exports` `=` `[` $exports^ `]` `,`)? $loc `>`
  }];
  let genVerifyDecl = 1;
  let extraClassDeclaration = [{
    bool isModule() const { return getKind() == ScopeKind::Module; }
    bool isProcedure() const { return getKind() == ScopeKind::Procedure; }
    bool isConstruct() const { return getKind() == ScopeKind::Construct; }
  }];
}

def fir_UseEntityAttr : fir_Attr<"UseEntity"> {
  let mnemonic = "use_entity";
  let summary = "A data object that a Fortran USE statement makes visible";
  let description = [{
    In a USE statement, `local_name` and `name` are the local-name and
    use-name. In a module's export list they are the public identifier
    and the name of the ultimate entity in its declaring module.
    Equal names are spelled once. A re-export names its declaring module
    with `from`.

    ```
    #fir.use_entity<"x">                   // x
    #fir.use_entity<"y" -> "z">            // y => z
    #fir.use_entity<"y" -> "z" from "m1">   // y => z, z is declared in m1
    ```
  }];
  let parameters = (ins "mlir::StringAttr":$local_name,
                        OptionalParameter<"mlir::StringAttr">:$name,
                        OptionalParameter<"mlir::StringAttr">:$module);
  let assemblyFormat = "`<` $local_name (`->` $name^)? (`from` $module^)? `>`";
}

// The ONLY flag of a USE statement, printed as the `only` keyword when set.
class fir_OnlyParameter<string desc = "">
    : DefaultValuedParameter<"bool", "false", desc> {
  let parser = [{
    ::mlir::FailureOr<bool>(
        ::mlir::succeeded($_parser.parseOptionalKeyword("only")))
  }];
  let printer = [{ $_printer << "only" }];
}

def fir_UseAttr : fir_Attr<"Use"> {
  let mnemonic = "use";
  let summary = "A Fortran USE statement, for debug information";
  let description = [{
    `module` is the name of the module that the statement uses. With
    `only`, `entities` are all the data objects that the statement makes
    visible. Without it, the whole module is visible, and `entities` are
    those that the statement renames. A USE with an empty ONLY list makes
    nothing visible and is not described. `loc` is the location of the
    statement.

    ```
    #fir.use<"m", loc("a.f90":2:3)>                              // use m
    #fir.use<"m", only [<"x">, <"y" -> "z">], loc("a.f90":3:3)>  // use m, only: x, y => z
    #fir.use<"m", [<"y" -> "z">], loc("a.f90":4:3)>              // use m, y => z
    ```
  }];
  let parameters = (ins "mlir::StringAttr":$module, fir_OnlyParameter<>:$only,
      OptionalArrayRefParameter<"::fir::UseEntityAttr">:$entities,
      "mlir::LocationAttr":$loc);
  let assemblyFormat = [{
    `<` $module `,` ($only^)? (` ` `[` $entities^ `]` `,`)? $loc `>`
  }];
  let genVerifyDecl = 1;
  let extraClassDeclaration = [{
    /// The Fortran name of \p entity in its defining module.
    mlir::StringAttr getEntityName(UseEntityAttr entity) const {
      return entity.getName() ? entity.getName() : entity.getLocalName();
    }
    /// The scope of the defining module of \p entity.
    mlir::StringAttr getEntityModule(UseEntityAttr entity) const {
      return entity.getModule() ? entity.getModule() : getModule();
    }
  }];
}

/// Optimization heuristics for fir.pack_array operation.
def fir_PackArrayHeuristics
    : I32BitEnumAttr<"PackArrayHeuristics", "",
                     [
                         /// fir.pack_array cannot be optimized based on the
                         /// array usage pattern.
                         I32BitEnumAttrCaseNone<"None", "none">,
                         /// fir.pack_array can be optimized away, if the array
                         /// is not used in a loop.
                         I32BitEnumAttrCaseBit<"LoopOnly", 0, "loop_only">,
]> {
  let separator = ", ";
  let cppNamespace = "::fir";
  let genSpecializedAttr = 0;
}

def fir_PackArrayHeuristicsAttr
    : EnumAttr<FIROpsDialect, fir_PackArrayHeuristics,
               "pack_array_heuristics">;

def fir_OpenACCSafeTempArrayCopyAttr : fir_Attr<"OpenACCSafeTempArrayCopy"> {
  let mnemonic = "acc_safe_temp_array_copy";
  let description = [{
    An attribute implementing SafeTempArrayCopyAttrInterface.
    It specifies whether it is possible to dynamically check
    if creating a temporary copy of a Fortran array is safe
    in the context of OpenACC.
    It also provides the methods to generate those dynamic checks.
  }];
}

def fir_OpenMPSafeTempArrayCopyAttr : fir_Attr<"OpenMPSafeTempArrayCopy"> {
  let mnemonic = "omp_safe_temp_array_copy";
  let description = [{
    An attribute implementing SafeTempArrayCopyAttrInterface.
    It specifies whether it is possible to dynamically check
    if creating a temporary copy of a Fortran array is safe
    in the context of OpenMP.
    It also provides the methods to generate those dynamic checks.
  }];
}

def LocalitySpecTypeLocal : I32EnumAttrCase<"Local", 0, "local">;
def LocalitySpecTypeLocalInit
    : I32EnumAttrCase<"LocalInit", 1, "local_init">;
def LocalitySpecTypeShared : I32EnumAttrCase<"Shared", 2, "shared">;

def LocalitySpecifierType : I32EnumAttr<
    "LocalitySpecifierType",
    "Type of a locality specifier", [
      LocalitySpecTypeLocal,
      LocalitySpecTypeLocalInit,
      LocalitySpecTypeShared
    ]> {
  let genSpecializedAttr = 0;
  let cppNamespace = "::fir";
}

def LocalitySpecifierTypeAttr : EnumAttr<FIROpsDialect, LocalitySpecifierType,
                                                "locality_specifier_type">;

/// Fortran inline attribute
def FIRinlineNone : I32BitEnumAttrCaseNone<"none">;
def FIRinlineNo : I32BitEnumAttrCaseBit<"no_inline", 0>;
def FIRinlineAlways : I32BitEnumAttrCaseBit<"always_inline", 1>;
def FIRinlineHint : I32BitEnumAttrCaseBit<"inline_hint", 2>;

def fir_FortranInlineEnum
    : I32BitEnumAttr<"FortranInlineEnum", "Fortran inline attributes",
                     [FIRinlineNone, FIRinlineNo, FIRinlineAlways,
                      FIRinlineHint]> {
  let separator = ", ";
  let cppNamespace = "::fir";
  let genSpecializedAttr = 0;
  let printBitEnumPrimaryGroups = 1;
}

def fir_FortranInlineAttr
    : EnumAttr<FIROpsDialect, fir_FortranInlineEnum, "inline_attrs">;

//===----------------------------------------------------------------------===//
// Allocation policy
//===----------------------------------------------------------------------===//

// Module attribute recording where array allocations should be placed. It is
// set by lowering and read back with fir::getAllocationPolicy, so that
// policy-aware passes make consistent placement decisions and dumped IR
// replays with the policy it was compiled with.
//
// fir::AllocationPolicy.
def fir_AllocationPolicyAttr : fir_Attr<"AllocationPolicy"> {
  let mnemonic = "allocation_policy";
  let summary = "Records the module-level array allocation policy";
  let description = [{
    Records the policy used by policy-aware passes to choose between stack
    (`fir.alloca`) and heap (`fir.allocmem`) storage for array allocations.
    Lowering sets this attribute from the compiler options. Passes that create
    allocations can consult it when selecting the allocation operation, and
    allocation-placement uses it when reconsidering existing allocations.

    Fields:
      - stack_arrays: place all arrays on the stack when possible
        (`-fstack-arrays`).
      - small_array_threshold: maximum size in bytes of a "small" constant-size
        array temporary.
      - total_stack_limit: per-function budget in bytes for small arrays placed
        on the stack.
  }];
  let parameters = (ins
    "bool":$stack_arrays,
    "uint64_t":$small_array_threshold,
    "uint64_t":$total_stack_limit
  );
  let assemblyFormat = "`<` struct(params) `>`";
}

//===----------------------------------------------------------------------===//
// OpenACC
//===----------------------------------------------------------------------===//

// Fortran-specific variable information for OpenACC.
// Carries metadata that cannot be recovered from the FIR type system alone
// and is required in the FIR implementation of OpenACC type interfaces.
def fir_OpenACCFortranVariableInfoAttr
    : AttrDef<FIROpsDialect, "OpenACCFortranVariableInfo",
              [OpenACC_IsVariableInfoAttr],
              "::mlir::acc::VariableInfoAttr"> {
  let mnemonic = "acc_var_info";
  let parameters = (ins "bool":$mayBeOptional);
  let assemblyFormat = "`<` struct(params) `>`";
}

#endif // FIR_DIALECT_FIR_ATTRS
