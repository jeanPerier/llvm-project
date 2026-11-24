// Test the verification of the attributes that describe the scopes for the
// debug information.

// RUN: fir-opt --split-input-file --verify-diagnostics %s

#f = #fir.scope<procedure "f", loc(unknown)>
// expected-error@+1 {{a USE statement must use a module scope}}
func.func private @g() attributes {use = #fir.use<#f, loc(unknown)>}

// -----

#m = #fir.scope<module "m", loc(unknown)>
// expected-error@+1 {{the ONLY list of a USE statement cannot be empty}}
func.func private @g() attributes {use = #fir.use<#m, only loc(unknown)>}

// -----

// expected-error@+1 {{a variable cannot be in a COMMON block and in an EQUIVALENCE}}
func.func private @g() attributes {var = #fir.module_var<"x" : i32, common "blk", equivalence "x", loc(unknown)>}

// -----

// expected-error@+1 {{a member of a COMMON block or of an EQUIVALENCE must have a type}}
func.func private @g() attributes {var = #fir.module_var<"x", common "blk", loc(unknown)>}
