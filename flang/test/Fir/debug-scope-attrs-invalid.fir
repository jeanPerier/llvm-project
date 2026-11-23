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

