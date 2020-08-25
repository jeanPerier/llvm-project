`DO CONCURRENT` isn't necessarily concurrent
============================================

Introduction
============
In the Fortran 2008 language standard, a variant form was
added to Fortran's primary looping construct with the goal
of more effective automatic parallel execution of code
written in the standard language without the use of
non-standard directives.
Spelled `DO CONCURRENT`, it takes a rectilinear iteration
space specification like `FORALL`, and allows us to write
a multidimensional loop nest construct with a single `DO`
statement and a single terminating `END DO` statement.

Within the body of a `DO CONCURRENT` loop, we have to respect
a long list of restrictions on our use of the language.
Actions that obviously can't be executed in parallel or that
don't allow all iterations to execute are prohibited.
These include:
* Control flow statements that would prevent the loop nest from
  executing all its iterations: `RETURN`, `EXIT`, and any
  `GOTO` or `CYCLE` that leaves the construct.
* Image control statements: `STOP`, `SYNC`, `LOCK`/`UNLOCK`, `EVENT`,
  and `ALLOCATE`/`DEALLOCATE` of a coarray.
* Calling a procedure that is not `PURE`.
* Deallocation of any polymorphic entity, because an impure final
  subroutine might be called.
* Messing with the IEEE floating-point control and status flags.
* Accepting some restrictions on data flow between iterations
  (i.e., none), liveness of modified objects after the loop, &c.
  (The details are spelled out later.)

But the deal is a good one -- when you live within these rules, your compiler
should generate code that exploits the parallel features of the target
machine to run the iterations of the `DO CONCURRENT` construct
as efficiently as they can.
You don't need to add OpenACC or OpenMP directives.
It's a great idea, really -- we know that our loop is parallel,
and now the compiler does too.

But it turns out that these rules, though *necessary* for safe parallel
execution, are not *sufficient*.

Localization
============
The J3 Fortran committee didn't actually define `DO CONCURRENT` as a
concurrent construct, or even as a construct that imposes sufficient
requirements on the programmer to allow for parallel execution.
Instead, `DO CONCURRENT` is defined as executing the iterations
of the loop in some arbitrary order (see subclause 11.1.7.4.3 paragraph 3).

I mentioned earlier that there are restrictions on data flow
in a `DO CONCURRENT` construct.
You can't modify an object in one iteration and expect to be
able to read it in another, or read it in one before it gets
modified by another -- there's no way to synchronize inter-iteration
communication with critical sections or atomics.

But you *can* modify an object in multiple iterations of the
loop in a conforming program, so long as you only read from that
object *after* you have modified it in the *same* iteration.
(See 11.1.7.5 paragraph 4 for the details.)

For example:

```
  DO CONCURRENT (J=1:N)
    TMP = A(J) + B(J)
    C(J) = TMP
  END DO
  ! And TMP is undefined afterwards
```

The scalar variable `TMP` is used in this loop in a way that obeys
the rules because every use follows a definition of `TMP` earlier
in the same iteration.

The idea, of course, is that a parallelizing compiler isn't required to
use the same word of memory to hold the value of `TMP`;
for parallel execution, `TMP` can be _localized_.
This means that the loop can be internally rewritten as if it had been
```
  DO CONCURRENT (J=1:N)
    BLOCK
      REAL :: TMP
      TMP = A(J) + B(J)
      C(J) = TMP
    END BLOCK
  END DO
```
and thus any risk of data flow between the iterations is removed.

The problem
===========
As mentioned above, the J3 Fortran committee did not define
the `DO CONCURRENT` construct as a concurrent loop construct;
it's a serial construct whose restrictions were intended to
enable automatic parallelization by compilers.
Defining things in this way is easier, since the language standard
does not have to define concepts like the memory and threading models
of C++.

If the automatic localization rules that allow usage like `TMP`
(above) had been limited to simple local scalars, everything would
have been fine.
Or if no automatic localization had been in the language, we would
use `BLOCK` and `ASSOCIATE` within the loop to define local objects
in each iteration where it made sense to do so, and we'd grumble
a little but still be fine.

*But*: the automatic localization rules weren't restricted to
simple local scalars.
They apply to arbitrary variables, and apply in cases that a
compiler can't always figure out, due to indexing, indirection,
and interprocedural data flow.

Let's see why this turns out to be a problem.

Examples:
```
  DO CONCURRENT (J=1:N)
    T(IX(J)) = A(J) + B(J)
    C(J) = T(IY(J))
  END DO
```
This loop conforms to the standard language if,
whenever `IX(J)` equals `IY(J')` for any distinct pair of iterations
`J` and `J'`,
then the load must be reading a value stored earlier in the
same iteration -- so `IX(J')==IY(J')`, and hence `IX(J)==IX(J')` too,
in this example.
Otherwise, a load in one iteration might depend on a store
in another.

When all values of `IX(J)` are distinct, and the program conforms
to the rules, a compiler can parallelize the loop trivially without
localization of `T(...)`.
When some values of `IX(J)` are duplicates, a compiler can parallelize
the loop by forwarding the stored value to the load in those
iterations.
But at compilation time, there's _no way to distinguish_ these
cases in general, and a conservative implementation has to assume
the worst and run the loop's iterations serially.
(Or compare `IX(J)` with `IY(J)` at runtime and forward the
stored value conditionally, which adds overhead and becomes
quickly impractical in loops with multiple loads and stores.)

In
```
  TYPE :: T
    REAL, POINTER :: P
  END TYPE
  TYPE(T) :: T1(N), T2(N)
  DO CONCURRENT (J=1:N)
    T1(J)%P = A(J) + B(J)
    C(J) = T2(J)%P
  END DO
```
we have the same kind of ambiguity from the compiler's perspective.
Are the targets of the pointers used for the stores all distinct
from the targets of the pointers used for the loads?
Maybe, but the compiler can't know for sure.
Again, to be conservative and safe, straightforward parallel execution
isn't an option.

Here's another case:
```
  MODULE M
    REAL :: T
  END MODULE
  ...
  USE M
  INTERFACE
    PURE REAL FUNCTION F(X)
      REAL, INTENT(IN) :: X
    END FUNCTION
  END INTERFACE
  DO CONCURRENT (J=1:N)
    T = A(J) + B(J)
    D(J) = F(A(J)) + T
  END DO
```
The variable `T` is obviously local, but the compiler can't be sure
that the pure function `F` doesn't read from `T`, and if it did,
there wouldn't be a practical way to localize it.

In summary, J3 defined `DO CONCURRENT` as a serial
construct with constraints for parallelization without
all of the complexity of threading or memory models, and
added the automatic localization rule in an attempt to provide
convenient temporaries without using `BLOCK` or `ASSOCIATE`.
But ambiguous cases are allowed in which a compiler can neither
1. prove that automatic localization *is* required for a given
   object in every iteration, nor
1. prove that automatic localization *isn't* required in any iteration.

The attempted fix
=================
The Fortran 2018 standard added "locality specifiers" to the
`DO CONCURRENT` statement.
These allow us to define some variable names as being `LOCAL` or
`SHARED`, relegating the automatic localization rules so that they
apply only in cases of "unspecified" locality.

`LOCAL` variables are those that can be defined by more than one
iteration, but referenced in any iteration only after having been defined
earlier.
`SHARED` variables are those that we stipulate will, if defined in
any iteration, not be defined or referenced in any other iteration.

(There is also a `LOCAL_INIT` specifier that is not relevant to the
problem at hand, and a `DEFAULT(NONE)` specifier that requires a
locality specifier be present for every variable mentioned in the
`DO CONCURRENT` construct.)

These locality specifiers can help resolve some otherwise ambiguous
cases of localization, but they're not a complete solution to the problem.
Several problems remain.

First, the specifiers allow explicit localization of objects
(like the scalar `T` in `MODULE M` above) that are not local variables
of the subprogram.
`DO CONCURRENT` still allows a pure procedure called from the loop
to reference `T`, and so explicit localization just confirms the
worst-case assumptions about interprocedural data flow
within an iteration that a compiler must make anyway.

Second, the specifiers allow arbitary variables to be localized,
not just scalars.
Want a local copy of a million-element array of derived type
with allocatable components to be created in each iteration?
Probably not, but the language allows it.
(Are they finalized at the end of each iteration?  Good question;
they aren't mentioned in the standard's list of finalizable items,
so probably not.)

Third, as Fortran uses context to distinguish references to
pointers from (de)references to their targets, it's not clear
whether `LOCAL(PTR)` localizes a pointer, its target, or both.

Fourth, the specifiers can be applied only to variable _names_,
not to anything with subscripts or component references.
You may have defined a derived type to hold your representation
of a sparse matrix, using `ALLOCATABLE` components to store its
packed data and indexing structures, but you can't localize some
parts of it and share the rest.
(Well, maybe you can, if you wrap `ASSOCIATE` constructs around the
`DO CONCURRENT` construct;
the interaction between locality specifiers and construct entities is
not clearly defined in the language.)
In the example above that defines `T(IX(J))` and reads from `T(IY(J))`,
the locality specifiers can't be used to share those elements of `T()`
that are modified at most once and localize the cases where
`IX(J)` is a duplicate and `IY(J)==IX(J)`.

Last, when a loop both defines and references many shared objects,
including potential references to globally accessible object
in called procedures, one may need to name all of them in a `SHARED`
specifier.  Did you miss one?
You'll need to check the compiler's report of optimizing transformations
to make sure.

What to do now
==============
These problems have been presented to J3, but their responses in
recent [e-mail discussions](https://mailman.j3-fortran.org/pipermail/j3/2020-July/thread.html)
did not include an intent to address them in future standards or corrigenda.
The most effective-looking response -- essentially, "just use
`DEFAULT(SHARED)` to disable all automatic localization" -- is not an
viable option, since the language does not include such a specifier.

So programmers writing `DO CONCURRENT` loops that are safely parallelizable
need an effective means to convey to their compilers that the compilers
do not have to assume only the weaker stipulations required by
today's `DO CONCURRENT` without having to write verbose and
error-prone locality specifiers.
Specifically, we need an easy means to state that localization
should apply at most only to the obvious cases of local non-pointer
non-allocatable scalars.

In the LLVM Fortran compiler project (a/k/a "flang", "f18") we considered
several solutions to this problem.
1. Add syntax (e.g., `DO PARALLEL` or `DO CONCURRENT() DEFAULT(PARALLEL)`)
   by which we can inform the compiler that it should localize only
   the obvious cases of simple local scalars.
   Such syntax seems unlikely to ever be standardized, so its usage
   would be nonportable.
1. Add a command-line option &/or a source directive to stipulate
   the stronger guarantees.  Obvious non-parallelizable usage in the construct
   would elicit a stern warning.  The `DO CONCURRENT` loops in the source
   would continue to be portable to other compilers.
1. Assume that these stronger conditions hold by default, and add a command-line
   option &/or a source directive to "opt out" back to the standard
   behavior in the event that the program contains one of those
   non-parallelizable `DO CONCURRENT` loops that should never have
   been possible to write in a conforming program in the first place.
   Programmers writing parallel `DO CONCURRENT` loops would get what
   they wanted without ever knowing that there's a problem with the
   standard.
   But this could lead to non-standard behavior for codes that depend,
   accidentally or not, on non-parallelizable implicit localization.
1. Accept the standard as it exists, do the best job of automatic
   parallelization that can be done, and refer dissatisfied users to J3.
   This would be avoiding the problem.

None of these options is without a fairly obvious disadvantage.
The best option seems to be the one that assumes that users who write
`DO CONCURRENT` constructs will do so with the intent to write parallel code.

As of August 2020, we observe that the GNU Fortran compiler (10.1) does not
yet implement the Fortran 2018 locality clauses, but will parallelize some
`DO CONCURRENT` constructs without ambiguous data dependences when the automatic
parallelization option is enabled.

The Intel Fortran compiler supports the new locality clauses and will parallelize
some `DO CONCURRENT` constructs when automatic parallelization option is enabled.
When OpenMP is enabled, ifort reports that all `DO CONCURRENT` constructs are
parallelized, but they seem to execute in a serial fashion when data flow
hazards are present.
