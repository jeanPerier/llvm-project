<!--===- docs/DesignGuideline.md

   Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
   See https://llvm.org/LICENSE.txt for license information.
   SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

-->
# Design Guideline

```eval_rst
.. contents::
   :local:
```
## Documenting the design

### Designing support for a new feature

When working on a new feature in flang, some design document should
be produced before submitting patches to the code.

The preferred organization of such documents is:
1) Problem description
2) Proposed solution
3) Implementation details overview
4) Testing plan

If several solutions can be considered, it is best to briefly describe the
alternate solutions in 2) and why they were not retained.
  
The design document should be added to the `docs` folder as a markdown document, ideally using
the name of the feature as the document name. Its approval on Phabricator is the
pre-requisite to submitting patches implementing new features. 
  
An RFC on flang https://discourse.llvm.org can first be made as one sees fit, but this document
should still be produced to summarize, organize, and formalize the discussions. If a related discourse
RFC was made it is a good idea to give a link to it in the document for future reference.

### Updating the implementation solution of a feature                                                                         

When doing a significant change to the implementation solution for a feature, the related design document
should be updated so that it will justify the new solution.

## Design tips

### Design document style

The document does not have to be long. It is highly encouraged to:
- Stick to Fortran well defined terms when talking about Fortran
  (definitions of these terms can be found in Section 3. of Fortran 2018 standard)
- Be precise (e.g., pointing to the standard reference or constraint numbers).
  References should currently be given against the Fortran 2018 standard version.
- Illustrate with a few small Fortran code snippets if applicable
- When dealing with lowering, illustrate lowering output with a few FIR, and or LLVM IR code snippets
- Illustrations do not have to be fully functional programs, it is better to keep
  them small and focused on the feature. More detailed expectations
  can be added in a second time or in parallel as LIT tests for example.

### Thinking through the design of a Fortran feature

Below is a set of suggested steps that one can take to fully apprehend a
Fortran feature before writing a design for its implementation in flang.

- Identify the relevant sections and constraints in the standard.
- Write Fortran program using the feature and, if possible,
  verify your expectations with existing compilers.
- Check if the related constraints (Cxxx numbers in the standard) are enforced by semantics.
  If not it is good idea to start by adding the related checks (this does not require writing a design document).
- Identify if the feature affects compatibility with programs compiled by other Fortran compilers,
  or if a given solution for flang could not be changed in the future without breaking compatibility
  with programs previously compiled with flang. It is not a goal to be 100% binary compatible with
  other compilers outside of Fortran 77, but sources of incompatibility should be known and justified.
- Identify related features, or contexts that matter for the feature (e.g, does being in an internal procedure,
  a module, a block… changes anything to what should happen?).
- Consider if a solution using some runtime is possible.
- For inlined code, consider what should happen when generating the FIR,
  and what should happen when lowering the FIR to LLVM IR.
- For inlined ops, look at how the existing dialects can be used,
  if new FIR operations are required, justify their purpose.
- Look at the related representation in Semantics (e.g., Is some information from the parse tree,
  the Symbol or evaluate::Expr required ? Are they tools to query this information easily?).
