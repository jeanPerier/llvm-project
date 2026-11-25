// Test the verification of the attributes that describe the scopes for the
// debug information.

// RUN: fir-opt --split-input-file --verify-diagnostics %s

// expected-error@+1 {{a module scope must have a name}}
func.func private @g() attributes {scope = #fir.scope<module, loc(unknown)>}

// -----

#f = #fir.scope<procedure "f", loc(unknown)>
// expected-error@+1 {{a submodule must have a module or submodule parent}}
func.func private @g() attributes {scope = #fir.scope<module "m", parent = #f, loc(unknown)>}

// -----

// expected-error@+1 {{a USE entity names an identifier in the used module}}
func.func private @g() attributes {use = #fir.use<"m", only [<"x" from "other">], loc(unknown)>}

// -----

// expected-error@+1 {{a USE statement must name a module}}
func.func private @g() attributes {use = #fir.use<"", loc(unknown)>}

// -----

// expected-error@+1 {{a re-export must name its declaring module}}
func.func private @g() attributes {scope = #fir.scope<module "m", exports = [<"x" from "">], loc(unknown)>}

// -----

// expected-error@+1 {{only a module scope can have public exports}}
func.func private @g() attributes {scope = #fir.scope<procedure "f", exports = [<"x">], loc(unknown)>}

// -----

#m = #fir.scope<module "m", loc(unknown)>
// expected-error@+1 {{invalid lexical parent for this scope}}
func.func private @g() attributes {scope = #fir.scope<construct, parent = #m, loc(unknown)>}

// -----

// expected-error@+1 {{the ONLY list of a USE statement cannot be empty}}
func.func private @g() attributes {use = #fir.use<"m", only loc(unknown)>}

// -----

// expected-error@+1 {{a variable cannot be in a COMMON block and in an EQUIVALENCE}}
func.func private @g() attributes {var = #fir.module_var<"x" : i32, common "blk" ("blk_"), equivalence "x", loc(unknown)>}

// -----

// expected-error@+1 {{a member of a COMMON block or of an EQUIVALENCE must have a type}}
func.func private @g() attributes {var = #fir.module_var<"x", common "blk" ("blk_"), loc(unknown)>}

// -----

// expected-error@+1 {{only a module scope can describe module variables}}
func.func private @f() attributes {scope = #fir.scope<procedure "f", variables = [#fir.module_var<"v", bind_c "v", loc(unknown)>], loc(unknown)>}
