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

// expected-error@+1 {{a USE entity must name its declaring module}}
func.func private @g() attributes {use = #fir.use<"m", only [<"x" from "">], loc(unknown)>}

// -----

// expected-error@+1 {{a USE statement must name a module}}
func.func private @g() attributes {use = #fir.use<"", loc(unknown)>}

// -----

#m = #fir.scope<module "m", loc(unknown)>
// expected-error@+1 {{invalid lexical parent for this scope}}
func.func private @g() attributes {scope = #fir.scope<construct, parent = #m, loc(unknown)>}
