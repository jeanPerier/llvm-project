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
func.func private @g() attributes {entity = #fir.use_entity<"x" from "">}

// -----

// expected-error@+1 {{only an entity declared in another module can be renamed there}}
func.func private @g() attributes {entity = #fir.use_entity<"x" named "y">}

// -----

// expected-error@+1 {{a USE statement must name a module}}
func.func private @g() attributes {use = #fir.use<"", loc(unknown)>}

// -----

#m = #fir.scope<module "m", loc(unknown)>
// expected-error@+1 {{invalid lexical parent for this scope}}
func.func private @g() attributes {scope = #fir.scope<construct, parent = #m, loc(unknown)>}

// -----

// expected-error@+1 {{the ONLY list of a USE statement cannot be empty}}
func.func private @g() attributes {use = #fir.use<"m", only loc(unknown)>}

// -----

// expected-error@+1 {{an entity in a USE statement without ONLY must be renamed}}
func.func private @g() attributes {use = #fir.use<"m", [<"x">], loc(unknown)>}

// -----

// expected-error@+1 {{a USE entity must have a local name}}
func.func private @g() attributes {entity = #fir.use_entity<"">}

// -----

// expected-error@+1 {{a renamed USE entity must have a use name}}
func.func private @g() attributes {entity = #fir.use_entity<"x" -> "">}

// -----

// expected-error@+1 {{a USE entity must have a declared name}}
func.func private @g() attributes {entity = #fir.use_entity<"x" from "m" named "">}
