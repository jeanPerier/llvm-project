!RUN: %f18 -E %s 2>&1 | FileCheck %s
!CHECK: if(.not.(.true.)) error stop "assert(" // ".TRUE." // ") failed at " //
!CHECK: assert.F90"
!CHECK: // ":" // "7"
#define STR(x) #x
#define assert(x) if(.not.(x)) error stop "assert(" // #x // ") failed at " // __FILE__ // ":" // STR(__LINE__)
assert(.TRUE.)
end
