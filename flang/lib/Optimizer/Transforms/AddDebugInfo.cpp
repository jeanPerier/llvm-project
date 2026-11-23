//===-------------- AddDebugInfo.cpp -- add debug info --------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

//===----------------------------------------------------------------------===//
/// \file
/// This pass populates some debug information for the module and functions.
//===----------------------------------------------------------------------===//

#include "flang/Optimizer/Builder/FIRBuilder.h"
#include "flang/Optimizer/Dialect/CUF/Attributes/CUFAttr.h"
#include "flang/Optimizer/Dialect/FIRAttr.h"
#include "flang/Optimizer/Dialect/FIRCG/CGOps.h"
#include "flang/Optimizer/Dialect/FIRDialect.h"
#include "flang/Optimizer/Dialect/FIROps.h"
#include "flang/Optimizer/Dialect/FIROpsSupport.h"
#include "flang/Optimizer/Dialect/FIRType.h"
#include "flang/Optimizer/Dialect/Support/FIRContext.h"
#include "flang/Optimizer/Support/InternalNames.h"
#include "flang/Optimizer/Support/Utils.h"
#include "flang/Optimizer/Transforms/DebugTypeGenerator.h"
#include "flang/Optimizer/Transforms/Passes.h"
#include "flang/Support/Version.h"
#include "mlir/Dialect/DLTI/DLTI.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/LLVMIR/LLVMModuleOps.h"
#include "mlir/Dialect/OpenMP/OpenMPDialect.h"
#include "mlir/IR/Matchers.h"
#include "mlir/IR/TypeUtilities.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/DialectConversion.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "llvm/ADT/MapVector.h"
#include "llvm/ADT/SetVector.h"
#include "llvm/BinaryFormat/Dwarf.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/FormatVariadic.h"
#include "llvm/Support/Path.h"
#include "llvm/Support/raw_ostream.h"

namespace fir {
#define GEN_PASS_DEF_ADDDEBUGINFO
#include "flang/Optimizer/Transforms/Passes.h.inc"
} // namespace fir

#define DEBUG_TYPE "flang-add-debug-info"

namespace {

class AddDebugInfoPass : public fir::impl::AddDebugInfoBase<AddDebugInfoPass> {
  void handleDeclareOp(fir::cg::XDeclareOp declOp,
                       mlir::LLVM::DIFileAttr fileAttr,
                       mlir::LLVM::DIScopeAttr scopeAttr,
                       fir::DebugTypeGenerator &typeGen,
                       mlir::SymbolTable *symbolTable, mlir::Value dummyScope);
  void handleDeclareValueOp(fir::DeclareValueOp declOp,
                            mlir::LLVM::DIFileAttr fileAttr,
                            mlir::LLVM::DIScopeAttr scopeAttr,
                            fir::DebugTypeGenerator &typeGen,
                            mlir::SymbolTable *symbolTable,
                            mlir::Value dummyScope);

public:
  AddDebugInfoPass(fir::AddDebugInfoOptions options) : Base(options) {}
  void runOnOperation() override;

private:
  llvm::StringMap<mlir::LLVM::DIModuleAttr> moduleMap;
  llvm::StringMap<mlir::LLVM::DICommonBlockAttr> commonBlockMap;
  /// Whether this module is being compiled for an OpenMP target device.
  bool isTargetDevice = false;
  // List of GlobalVariableExpressionAttr that are attached to a given global
  // that represents the storage for common block.
  llvm::DenseMap<fir::GlobalOp, llvm::SmallVector<mlir::Attribute>>
      globalToGlobalExprsMap;

  /// The modules that lowering describes, by debug name. Those that this
  /// compilation unit does not define have no location.
  llvm::StringMap<fir::ScopeAttr> moduleScopes;
  /// The described module variables, by defining module and source name.
  llvm::DenseMap<std::pair<mlir::StringAttr, mlir::StringAttr>,
                 mlir::LLVM::DIGlobalVariableAttr>
      moduleVariables;

  mlir::LLVM::DIModuleAttr createModuleAttr(const std::string &name,
                                            mlir::LLVM::DIFileAttr fileAttr,
                                            mlir::LLVM::DIScopeAttr scope);
  mlir::LLVM::DIModuleAttr
  getOrCreateModuleAttr(const std::string &name,
                        mlir::LLVM::DIFileAttr fileAttr,
                        mlir::LLVM::DIScopeAttr scope);
  llvm::SmallVector<mlir::LLVM::DINodeAttr>
  buildSubmoduleImports(llvm::ArrayRef<fir::ScopeAttr> moduleScopes,
                        mlir::LLVM::DIFileAttr fileAttr,
                        mlir::LLVM::DIScopeAttr cuAttr);
  mlir::LLVM::DICommonBlockAttr
  getOrCreateCommonBlockAttr(llvm::StringRef name,
                             mlir::LLVM::DIFileAttr fileAttr,
                             mlir::LLVM::DIScopeAttr scope, unsigned line);

  void handleGlobalOp(fir::GlobalOp glocalOp, mlir::LLVM::DIFileAttr fileAttr,
                      mlir::LLVM::DIScopeAttr scope,
                      fir::DebugTypeGenerator &typeGen,
                      mlir::SymbolTable *symbolTable,
                      fir::cg::XDeclareOp declOp);
  void handleFuncOp(mlir::func::FuncOp funcOp, mlir::LLVM::DIFileAttr fileAttr,
                    mlir::LLVM::DICompileUnitAttr cuAttr,
                    fir::DebugTypeGenerator &typeGen,
                    mlir::SymbolTable *symbolTable);
  void expandUseStmtForDebug(
      fir::UseAttr use, mlir::LLVM::DISubprogramAttr spAttr,
      mlir::LLVM::DIFileAttr fileAttr, mlir::LLVM::DICompileUnitAttr cuAttr,
      llvm::SetVector<mlir::LLVM::DIImportedEntityAttr> &importedEntities,
      llvm::DenseSet<mlir::StringAttr> &seenModuleNames);
  bool createCommonBlockGlobal(fir::cg::XDeclareOp declOp,
                               const std::string &name,
                               mlir::LLVM::DIFileAttr fileAttr,
                               mlir::LLVM::DIScopeAttr scopeAttr,
                               fir::DebugTypeGenerator &typeGen,
                               mlir::SymbolTable *symbolTable);
  std::optional<mlir::LLVM::DIModuleAttr>
  getModuleAttrFromGlobalOp(fir::GlobalOp globalOp,
                            mlir::LLVM::DIFileAttr fileAttr,
                            mlir::LLVM::DIScopeAttr scope);

  template <typename Op>
  void handleLocalVariable(Op declOp, llvm::StringRef name,
                           mlir::LLVM::DIFileAttr fileAttr,
                           mlir::LLVM::DIScopeAttr scopeAttr,
                           fir::DebugTypeGenerator &typeGen,
                           mlir::Value dummyScope, mlir::Type typeToConvert,
                           fir::cg::XDeclareOp typeGenDeclOp);
};

// A submodule has no name of its own in the debug info. Like gfortran, name it
// after its ancestor module and itself, which the standard guarantees to be
// enough to tell the submodules of a program apart. Join the two with a '.', as
// gfortran does. A '.' cannot appear in a Fortran identifier, so the result can
// never collide with a module that is really called that, which matters because
// a module is looked up by this name alone.
std::string getDebugModuleName(llvm::StringRef ancestor, llvm::StringRef name) {
  if (ancestor.empty())
    return name.str();
  return (ancestor + "." + name).str();
}

// The (sub)module that owns an entity, as named in the debug info. A uniqued
// name records the whole ancestry, from the module down to the innermost
// submodule, so the first and the last of those are what we need.
std::string getDebugModuleName(llvm::ArrayRef<std::string> modules) {
  assert(!modules.empty() && "not a module entity");
  if (modules.size() == 1)
    return modules.front();
  return getDebugModuleName(modules.front(), modules.back());
}

// The debug name of a module or submodule described by lowering.
std::string getDebugModuleName(fir::ScopeAttr moduleScope) {
  fir::ScopeAttr root = moduleScope;
  while (root.getParent())
    root = root.getParent();
  if (root == moduleScope)
    return moduleScope.getName().str();
  return getDebugModuleName(root.getName().getValue(),
                            moduleScope.getName().getValue());
}

/// Whether \p loc already carries debug information of type \c AttrT, fused
/// onto it by this pass. A location is fused for unrelated reasons too, most
/// notably one that came from an INCLUDE'd file, so what the fusion holds has
/// to be examined rather than the fusion merely detected. Each caller names
/// the attribute it would attach itself, so nothing here needs updating when
/// another kind of debug information is generated elsewhere. Only the
/// outermost fusion is examined, because that is the one this pass adds.
template <typename AttrT>
bool debugInfoIsAlreadySet(mlir::Location loc) {
  auto fusedLoc = mlir::dyn_cast<mlir::FusedLoc>(loc);
  return fusedLoc && mlir::isa_and_present<AttrT>(fusedLoc.getMetadata());
}

// Generates the name for the artificial DISubprogram that we are going to
// generate for omp::TargetOp. Its logic is borrowed from
// getTargetEntryUniqueInfo and
// TargetRegionEntryInfo::getTargetRegionEntryFnName to generate the same name.
// But even if there was a slight mismatch, it is not a problem because this
// name is artificial and not important to debug experience.
mlir::StringAttr getTargetFunctionName(mlir::MLIRContext *context,
                                       mlir::Location Loc,
                                       llvm::StringRef parentName) {
  auto fileLoc = Loc->findInstanceOf<mlir::FileLineColLoc>();

  assert(fileLoc && "No file found from location");
  llvm::StringRef fileName = fileLoc.getFilename().getValue();

  llvm::sys::fs::UniqueID id;
  uint64_t line = fileLoc.getLine();
  size_t fileId;
  size_t deviceId;
  if (auto ec = llvm::sys::fs::getUniqueID(fileName, id)) {
    fileId = llvm::hash_value(fileName.str());
    deviceId = 0xdeadf17e;
  } else {
    fileId = id.getFile();
    deviceId = id.getDevice();
  }
  return mlir::StringAttr::get(
      context,
      std::string(llvm::formatv("__omp_offloading_{0:x-}_{1:x-}_{2}_l{3}",
                                deviceId, fileId, parentName, line)));
}

} // namespace

// Check if a name is that of a data object, which in Fortran is a variable or
// a named constant.
static bool isDataObjectName(fir::NameUniquer::NameKind kind) {
  return kind == fir::NameUniquer::NameKind::VARIABLE ||
         kind == fir::NameUniquer::NameKind::CONSTANT;
}

// Check if a name belongs to a module rather than to a procedure.
static bool isModuleLevelName(const fir::NameUniquer::DeconstructedName &name) {
  return name.procs.empty() && !name.modules.empty();
}

// Check if the storage of a variable with the given CUDA Fortran data
// attribute can be addressed by the host.
static bool isHostAddressable(cuf::DataAttributeAttr dataAttr) {
  if (!dataAttr)
    return true;
  switch (dataAttr.getValue()) {
  case cuf::DataAttribute::Constant:
  case cuf::DataAttribute::Device:
  case cuf::DataAttribute::Shared:
    return false;
  case cuf::DataAttribute::Managed:
  case cuf::DataAttribute::Pinned:
  case cuf::DataAttribute::Unified:
    return true;
  }
  llvm_unreachable("unknown CUDA Fortran data attribute");
}

// Check if the operation belongs to a procedure that is compiled for the
// device only, whose debug info is generated separately.
static bool isInDeviceOnlyProcedure(mlir::Operation *op) {
  auto funcOp = op->getParentOfType<mlir::func::FuncOp>();
  if (!funcOp)
    return false;
  auto procAttr =
      funcOp->getAttrOfType<cuf::ProcAttributeAttr>(cuf::getProcAttrName());
  if (!procAttr)
    return false;
  switch (procAttr.getValue()) {
  case cuf::ProcAttribute::Host:
  case cuf::ProcAttribute::HostDevice:
    return false;
  case cuf::ProcAttribute::Device:
  case cuf::ProcAttribute::Global:
  case cuf::ProcAttribute::GridGlobal:
    return true;
  }
  llvm_unreachable("unknown CUDA Fortran procedure attribute");
}

// Check if a global represents a data object declared in a module.
static bool isModuleDataObject(fir::GlobalOp globalOp) {
  std::pair result = fir::NameUniquer::deconstruct(globalOp.getSymName());
  return isDataObjectName(result.first) && isModuleLevelName(result.second);
}

// Return the DIGlobalVariable that describes a global, if any.
static std::optional<mlir::LLVM::DIGlobalVariableAttr>
lookupDIGlobalVariable(fir::GlobalOp globalOp) {
  auto fusedLoc =
      globalOp.getLoc()->findInstanceOf<mlir::FusedLocWith<mlir::ArrayAttr>>();
  if (!fusedLoc)
    return std::nullopt;
  for (mlir::Attribute elem : fusedLoc.getMetadata())
    if (auto gvExpr =
            mlir::dyn_cast<mlir::LLVM::DIGlobalVariableExpressionAttr>(elem))
      return gvExpr.getVar();
  return std::nullopt;
}

bool AddDebugInfoPass::createCommonBlockGlobal(
    fir::cg::XDeclareOp declOp, const std::string &name,
    mlir::LLVM::DIFileAttr fileAttr, mlir::LLVM::DIScopeAttr scopeAttr,
    fir::DebugTypeGenerator &typeGen, mlir::SymbolTable *symbolTable) {
  mlir::MLIRContext *context = &getContext();
  mlir::OpBuilder builder(context);

  std::optional<std::int64_t> offset;
  mlir::Value storage = declOp.getStorage();
  if (!storage)
    return false;

  // Extract offset from storage_offset attribute
  uint64_t storageOffset = declOp.getStorageOffset();
  if (storageOffset != 0)
    offset = static_cast<std::int64_t>(storageOffset);

  // Get the GlobalOp from the storage value.
  // The storage may be wrapped in ConvertOp, so unwrap it first.
  mlir::Operation *storageOp = storage.getDefiningOp();
  if (auto convertOp = mlir::dyn_cast_if_present<fir::ConvertOp>(storageOp))
    storageOp = convertOp.getValue().getDefiningOp();

  auto addrOfOp = mlir::dyn_cast_if_present<fir::AddrOfOp>(storageOp);
  if (!addrOfOp)
    return false;

  mlir::SymbolRefAttr sym = addrOfOp.getSymbol();
  fir::GlobalOp global =
      symbolTable->lookup<fir::GlobalOp>(sym.getRootReference());
  if (!global)
    return false;

  // Check if the global is actually a common block by demangling its name.
  // Module EQUIVALENCE variables also use storage operands but are mangled
  // as VARIABLE type, so we reject them to avoid treating them as common
  // blocks.
  llvm::StringRef globalSymbol = sym.getRootReference();
  auto globalResult = fir::NameUniquer::deconstruct(globalSymbol);
  if (globalResult.first == fir::NameUniquer::NameKind::VARIABLE)
    return false;

  // FIXME: We are trying to extract the name of the common block from the
  // name of the global. As part of mangling, GetCommonBlockObjectName can
  // add a trailing _ in the name of that global. The demangle function
  // does not seem to handle such cases. So the following hack is used to
  // remove the trailing '_'.
  llvm::StringRef commonName = globalSymbol;
  if (commonName != Fortran::common::blankCommonObjectName &&
      !commonName.empty() && commonName.back() == '_')
    commonName = commonName.drop_back();

  // Create the debug attributes.
  unsigned line = fir::getLineFromLoc(global.getLoc());
  mlir::LLVM::DICommonBlockAttr commonBlock = getOrCreateCommonBlockAttr(
      commonName, fir::getFileAttrFromLoc(global.getLoc(), fileAttr), scopeAttr,
      line);

  mlir::LLVM::DITypeAttr diType = typeGen.convertType(
      fir::unwrapRefType(declOp.getType()), fileAttr, scopeAttr, declOp);

  line = fir::getLineFromLoc(declOp.getLoc());
  auto gvAttr = mlir::LLVM::DIGlobalVariableAttr::get(
      context, commonBlock, mlir::StringAttr::get(context, name),
      declOp.getUniqName(), fir::getFileAttrFromLoc(declOp.getLoc(), fileAttr),
      line, diType,
      /*isLocalToUnit*/ false, /*isDefinition*/ true, /* alignInBits*/ 0);

  // Create DIExpression for offset if needed
  mlir::LLVM::DIExpressionAttr expr;
  if (offset && *offset != 0) {
    llvm::SmallVector<mlir::LLVM::DIExpressionElemAttr> ops;
    ops.push_back(mlir::LLVM::DIExpressionElemAttr::get(
        context, llvm::dwarf::DW_OP_plus_uconst, *offset));
    expr = mlir::LLVM::DIExpressionAttr::get(context, ops);
  }

  auto dbgExpr = mlir::LLVM::DIGlobalVariableExpressionAttr::get(
      global.getContext(), gvAttr, expr);
  globalToGlobalExprsMap[global].push_back(dbgExpr);

  return true;
}

// Create fake uses for compiler-generated internal variables that represent
// values needed by a debugger.  This prevents values from being optimized out
// such as the count and lower bound of dynamic arrays.
template <typename Op>
static void InsertFakeUseForDebugVar(mlir::OpBuilder &builder, Op declOp,
                                     mlir::Value var) {
  if (auto funcOp = declOp->template getParentOfType<mlir::func::FuncOp>()) {
    if (declOp->getBlock() == &funcOp.getBody().front()) {
      for (mlir::Block &block : funcOp.getBody()) {
        if (auto returnOp =
                mlir::dyn_cast<mlir::func::ReturnOp>(block.getTerminator())) {
          mlir::OpBuilder::InsertionGuard guard(builder);
          builder.setInsertionPoint(returnOp);
          if (!fir::getIntIfConstant(var))
            fir::FakeUseOp::create(builder, declOp.getLoc(), var);
        }
      }
    }
  }
}

template <typename Op>
void AddDebugInfoPass::handleLocalVariable(Op declOp, llvm::StringRef name,
                                           mlir::LLVM::DIFileAttr fileAttr,
                                           mlir::LLVM::DIScopeAttr scopeAttr,
                                           fir::DebugTypeGenerator &typeGen,
                                           mlir::Value dummyScope,
                                           mlir::Type typeToConvert,
                                           fir::cg::XDeclareOp typeGenDeclOp) {
  // Exclude variables that the host cannot address.
  if (!isInDeviceOnlyProcedure(declOp) &&
      !isHostAddressable(declOp.getDataAttrAttr()))
    return;

  mlir::MLIRContext *context = &getContext();
  mlir::OpBuilder builder(context);

  // Get the dummy argument position from the explicit attribute.
  unsigned argNo = 0;
  if (dummyScope && declOp.getDummyScope() == dummyScope) {
    if (auto argNoOpt = declOp.getDummyArgNo()) {
      argNo = *argNoOpt;
      if (emitFakeUseForDebugVars) {
        if constexpr (std::is_same_v<Op, fir::cg::XDeclareOp>) {
          InsertFakeUseForDebugVar(builder, declOp, declOp.getMemref());
        }
      }
    }
  }

  auto tyAttr =
      typeGen.convertType(typeToConvert, fileAttr, scopeAttr, typeGenDeclOp);

  if (emitFakeUseForDebugVars) {
    // Create fake uses for internal variables that represent count and lower
    // bound of dynamic arrays to ensure they are not optimized out. A value
    // that convertType has put in a stack slot needs none, as the debug info
    // describes the slot and nothing has to keep the value itself alive.
    if (auto arrayTy =
            mlir::dyn_cast<mlir::LLVM::DICompositeTypeAttr>(tyAttr)) {
      if (arrayTy.getTag() == llvm::dwarf::DW_TAG_array_type) {
        if constexpr (std::is_same_v<Op, fir::cg::XDeclareOp>) {
          // Count is represented as a value in the shape attribute
          for (auto val : declOp.getShape())
            if (!typeGen.hasBoundSlot(val, declOp))
              InsertFakeUseForDebugVar(builder, declOp, val);
          // Lower bound is represented as a value in the shift attribute
          for (auto val : declOp.getShift())
            if (!typeGen.hasBoundSlot(val, declOp))
              InsertFakeUseForDebugVar(builder, declOp, val);
        }
      }
    }

    // Create fake uses for the length of character arrays to ensure they
    // are not optimized out.
    if constexpr (std::is_same_v<Op, fir::cg::XDeclareOp>) {
      for (auto val : declOp.getTypeparams())
        if (!typeGen.hasBoundSlot(val, declOp))
          InsertFakeUseForDebugVar(builder, declOp, val);
    }
  }

  auto localVarAttr = mlir::LLVM::DILocalVariableAttr::get(
      context, scopeAttr, mlir::StringAttr::get(context, name),
      fir::getFileAttrFromLoc(declOp.getLoc(), fileAttr),
      fir::getLineFromLoc(declOp.getLoc()), argNo, /* alignInBits*/ 0, tyAttr,
      mlir::LLVM::DIFlags::Zero);
  declOp->setLoc(builder.getFusedLoc({declOp->getLoc()}, localVarAttr));
}

void AddDebugInfoPass::handleDeclareOp(fir::cg::XDeclareOp declOp,
                                       mlir::LLVM::DIFileAttr fileAttr,
                                       mlir::LLVM::DIScopeAttr scopeAttr,
                                       fir::DebugTypeGenerator &typeGen,
                                       mlir::SymbolTable *symbolTable,
                                       mlir::Value dummyScope) {
  auto result = fir::NameUniquer::deconstruct(declOp.getUniqName());

  if (!isDataObjectName(result.first))
    return;

  if (createCommonBlockGlobal(declOp, result.second.name, fileAttr, scopeAttr,
                              typeGen, symbolTable))
    return;

  // If this DeclareOp actually represents a global then treat it as such.
  mlir::Operation *defOp = declOp.getMemref().getDefiningOp();
  if (defOp && llvm::isa<fir::AddrOfOp>(defOp)) {
    if (auto global =
            symbolTable->lookup<fir::GlobalOp>(declOp.getUniqName())) {
      handleGlobalOp(global, fileAttr, scopeAttr, typeGen, symbolTable, declOp);
      return;
    }
  }

  handleLocalVariable(declOp, result.second.name, fileAttr, scopeAttr, typeGen,
                      dummyScope, fir::unwrapRefType(declOp.getType()), declOp);
}

void AddDebugInfoPass::handleDeclareValueOp(fir::DeclareValueOp declOp,
                                            mlir::LLVM::DIFileAttr fileAttr,
                                            mlir::LLVM::DIScopeAttr scopeAttr,
                                            fir::DebugTypeGenerator &typeGen,
                                            mlir::SymbolTable *symbolTable,
                                            mlir::Value dummyScope) {
  auto result = fir::NameUniquer::deconstruct(declOp.getUniqName());

  if (result.first != fir::NameUniquer::NameKind::VARIABLE)
    return;

  handleLocalVariable(declOp, result.second.name, fileAttr, scopeAttr, typeGen,
                      dummyScope, declOp.getValue().getType(), nullptr);
}

mlir::LLVM::DICommonBlockAttr AddDebugInfoPass::getOrCreateCommonBlockAttr(
    llvm::StringRef name, mlir::LLVM::DIFileAttr fileAttr,
    mlir::LLVM::DIScopeAttr scope, unsigned line) {
  mlir::MLIRContext *context = &getContext();
  mlir::LLVM::DICommonBlockAttr cbAttr;
  if (auto iter{commonBlockMap.find(name)}; iter != commonBlockMap.end()) {
    cbAttr = iter->getValue();
  } else {
    cbAttr = mlir::LLVM::DICommonBlockAttr::get(
        context, scope, nullptr, mlir::StringAttr::get(context, name), fileAttr,
        line);
    commonBlockMap[name] = cbAttr;
  }
  return cbAttr;
}

// The `module` does not have a first class representation in the `FIR`. We
// extract information about it from the name of the identifiers and keep a
// map to avoid duplication.
mlir::LLVM::DIModuleAttr
AddDebugInfoPass::createModuleAttr(const std::string &name,
                                   mlir::LLVM::DIFileAttr fileAttr,
                                   mlir::LLVM::DIScopeAttr scope) {
  mlir::MLIRContext *context = &getContext();
  unsigned line = 0;
  auto iter = moduleScopes.find(name);
  bool decl = iter == moduleScopes.end() ||
              mlir::isa<mlir::UnknownLoc>(iter->second.getLoc());
  // The location of a module defined here is that of its MODULE statement.
  if (!decl) {
    mlir::Location loc = iter->second.getLoc();
    line = fir::getLineFromLoc(loc);
    fileAttr = fir::getFileAttrFromLoc(loc, fileAttr);
  }

  // When decl is true, it means that module is only being used in this
  // compilation unit and it is defined elsewhere. But if the file/line/scope
  // fields are valid, the module is not merged with its definition and is
  // considered different. So we only set those fields when decl is false.
  return mlir::LLVM::DIModuleAttr::get(
      context, decl ? nullptr : fileAttr, decl ? nullptr : scope,
      mlir::StringAttr::get(context, name),
      /* configMacros */ mlir::StringAttr(),
      /* includePath */ mlir::StringAttr(),
      /* apinotes */ mlir::StringAttr(), decl ? 0 : line, decl);
}

mlir::LLVM::DIModuleAttr
AddDebugInfoPass::getOrCreateModuleAttr(const std::string &name,
                                        mlir::LLVM::DIFileAttr fileAttr,
                                        mlir::LLVM::DIScopeAttr scope) {
  if (auto iter{moduleMap.find(name)}; iter != moduleMap.end())
    return iter->getValue();
  mlir::LLVM::DIModuleAttr modAttr = createModuleAttr(name, fileAttr, scope);
  moduleMap[name] = modAttr;
  return modAttr;
}

/// If globalOp represents a module variable, return a ModuleAttr that
/// represents that module.
std::optional<mlir::LLVM::DIModuleAttr>
AddDebugInfoPass::getModuleAttrFromGlobalOp(fir::GlobalOp globalOp,
                                            mlir::LLVM::DIFileAttr fileAttr,
                                            mlir::LLVM::DIScopeAttr scope) {
  mlir::MLIRContext *context = &getContext();
  mlir::OpBuilder builder(context);

  std::pair result = fir::NameUniquer::deconstruct(globalOp.getSymName());
  // Only look for module if this variable is not part of a function.
  if (!result.second.procs.empty() || result.second.modules.empty())
    return std::nullopt;

  // DWARF5 says following about the fortran modules:
  // A Fortran 90 module may also be represented by a module entry
  // (but no declaration attribute is warranted because Fortran has no concept
  // of a corresponding module body).
  // But in practice, compilers use declaration attribute with a module in cases
  // where module was defined in another source file (only being used in this
  // one). Whether that is the case here is settled by getOrCreateModuleAttr.
  mlir::LLVM::DISubprogramAttr sp =
      mlir::dyn_cast_if_present<mlir::LLVM::DISubprogramAttr>(scope);
  // Modules are generated at compile unit scope
  if (sp)
    scope = sp.getCompileUnit();

  return getOrCreateModuleAttr(getDebugModuleName(result.second.modules),
                               fileAttr, scope);
}

void AddDebugInfoPass::handleGlobalOp(fir::GlobalOp globalOp,
                                      mlir::LLVM::DIFileAttr fileAttr,
                                      mlir::LLVM::DIScopeAttr scope,
                                      fir::DebugTypeGenerator &typeGen,
                                      mlir::SymbolTable *symbolTable,
                                      fir::cg::XDeclareOp declOp) {
  // A global is described by an array of DIGlobalVariableExpressionAttr.
  if (debugInfoIsAlreadySet<mlir::ArrayAttr>(globalOp.getLoc()))
    return;
  mlir::MLIRContext *context = &getContext();
  mlir::OpBuilder builder(context);

  std::pair result = fir::NameUniquer::deconstruct(globalOp.getSymName());
  switch (result.first) {
  case fir::NameUniquer::NameKind::VARIABLE:
    break;
  case fir::NameUniquer::NameKind::CONSTANT:
    // A constant local to a procedure is described while walking that
    // procedure, where `scope` is its DISubprogramAttr. Reaching here with any
    // other scope means the procedure is not in the IR, typically because it
    // was never called and got removed while its constant survived. There is
    // no procedure to attach the constant to, and describing it at compile
    // unit scope would wrongly make it visible everywhere.
    if (!isModuleLevelName(result.second) &&
        !mlir::isa<mlir::LLVM::DISubprogramAttr>(scope))
      return;
    // Don't describe a constant for which we only have a declaration. It could
    // leave an unresolved symbol in the debug information.
    if (!globalOp.isInitialized())
      return;
    break;
  default:
    return;
  }

  if (fir::NameUniquer::isSpecialSymbol(result.second.name))
    return;

  unsigned line = fir::getLineFromLoc(globalOp.getLoc());
  std::optional<mlir::LLVM::DIModuleAttr> modOpt =
      getModuleAttrFromGlobalOp(globalOp, fileAttr, scope);
  if (modOpt)
    scope = *modOpt;

  // An entity with internal linkage, such as a constant or a SAVE variable
  // declared inside a procedure, is not visible outside this compilation unit.
  // It also needs no linkage name because there is no external symbol for a
  // debugger to match it against.
  const bool isLocalToUnit =
      globalOp.getLinkage() == fir::LinkageEnum::Internal;
  mlir::StringAttr linkageName =
      isLocalToUnit ? mlir::StringAttr()
                    : mlir::StringAttr::get(context, globalOp.getName());

  mlir::LLVM::DITypeAttr diType =
      typeGen.convertType(globalOp.getType(), fileAttr, scope, declOp);
  auto gvAttr = mlir::LLVM::DIGlobalVariableAttr::get(
      context, scope, mlir::StringAttr::get(context, result.second.name),
      linkageName, fir::getFileAttrFromLoc(globalOp.getLoc(), fileAttr), line,
      diType, isLocalToUnit,
      /*isDefinition*/ globalOp.isInitialized(), /* alignInBits*/ 0);
  auto dbgExpr = mlir::LLVM::DIGlobalVariableExpressionAttr::get(
      globalOp.getContext(), gvAttr, nullptr);
  auto arrayAttr = mlir::ArrayAttr::get(context, {dbgExpr});
  globalOp->setLoc(builder.getFusedLoc({globalOp.getLoc()}, arrayAttr));
}

static mlir::LLVM::DISubprogramAttr
getScope(mlir::Operation *op, mlir::LLVM::DISubprogramAttr defaultScope) {
  if (auto tOp = op->getParentOfType<mlir::omp::TargetOp>()) {
    if (auto fusedLoc = llvm::dyn_cast<mlir::FusedLoc>(tOp.getLoc())) {
      if (auto sp = llvm::dyn_cast<mlir::LLVM::DISubprogramAttr>(
              fusedLoc.getMetadata()))
        return sp;
    }
  }
  return defaultScope;
}

static bool hasDebugScope(mlir::Location loc) {
  auto fusedLoc = llvm::dyn_cast<mlir::FusedLoc>(loc);
  return fusedLoc && llvm::isa_and_nonnull<mlir::LLVM::DILocalScopeAttr>(
                         fusedLoc.getMetadata());
}

/// Return the lexical scope that lowering fused into \p loc, if any. Inlined
/// code is in the scope of its call.
static fir::ScopeAttr getLexicalScope(mlir::Location loc) {
  fir::ScopeAttr lexicalScope;
  loc->walk([&](mlir::Location subLoc) {
    if (auto callSiteLoc = mlir::dyn_cast<mlir::CallSiteLoc>(subLoc)) {
      lexicalScope = getLexicalScope(callSiteLoc.getCaller());
      return mlir::WalkResult::interrupt();
    }
    if (auto fusedLoc = mlir::dyn_cast<mlir::FusedLoc>(subLoc))
      if (auto scope =
              mlir::dyn_cast_if_present<fir::ScopeAttr>(fusedLoc.getMetadata());
          scope && scope.isConstruct())
        lexicalScope = scope;
    return lexicalScope ? mlir::WalkResult::interrupt()
                        : mlir::WalkResult::advance();
  });
  return lexicalScope;
}

/// Return the scope of the procedure that lowering fused into the location of
/// \p funcOp, if any. A declaration may have the location of a call instead.
static fir::ScopeAttr getProcedureScope(mlir::func::FuncOp funcOp) {
  if (funcOp.isExternal())
    return {};
  if (auto fusedLoc =
          funcOp.getLoc()->findInstanceOf<mlir::FusedLocWith<fir::ScopeAttr>>())
    if (fusedLoc.getMetadata().isProcedure())
      return fusedLoc.getMetadata();
  return {};
}

namespace {
/// Puts the operations of each lexical scope in a set, and of each region in a
/// set, in a lexical block, nested in the lexical block of any enclosing one.
class LexicalBlockBuilder {
public:
  LexicalBlockBuilder(mlir::MLIRContext *context,
                      mlir::LLVM::DIFileAttr fileAttr)
      : fileAttr(fileAttr), builder(context) {}

  bool empty() const { return lexicalScopes.empty() && scopedRegions.empty(); }

  void run(mlir::func::FuncOp funcOp, mlir::LLVM::DISubprogramAttr spAttr) {
    visitRegion(funcOp.getBody(), spAttr, {});
  }

  llvm::DenseSet<fir::ScopeAttr> lexicalScopes;
  llvm::SmallPtrSet<mlir::Region *, 4> scopedRegions;

private:
  /// The operations of \p region in \p lexicalScope, that of the operation that
  /// holds the region, are in \p scope.
  void visitRegion(mlir::Region &region, mlir::LLVM::DIScopeAttr scope,
                   fir::ScopeAttr lexicalScope) {
    if (region.empty())
      return;
    if (scopedRegions.contains(&region)) {
      auto [line, column] = getPosition(region.front());
      scope = mlir::LLVM::DILexicalBlockAttr::get(builder.getContext(), scope,
                                                  fileAttr, line, column);
    }
    for (mlir::Operation &op : region.getOps()) {
      fir::ScopeAttr opLexicalScope = getLexicalScope(op.getLoc());
      if (!opLexicalScope)
        opLexicalScope = lexicalScope;
      mlir::LLVM::DIScopeAttr opScope =
          getLexicalBlock(opLexicalScope, lexicalScope, scope);
      // A target region carries the subprogram that describes it.
      if (llvm::isa<mlir::LLVM::DILexicalBlockAttr>(opScope) &&
          !hasDebugScope(op.getLoc()))
        op.setLoc(builder.getFusedLoc({op.getLoc()}, opScope));
      // The body of an outlined region becomes a separate function.
      if (llvm::isa<mlir::omp::OutlineableOpenMPOpInterface>(op))
        continue;
      for (mlir::Region &subRegion : op.getRegions())
        visitRegion(subRegion, opScope, opLexicalScope);
    }
  }

  /// Return the scope of the operations in \p lexicalScope, given that those
  /// in \p outer, an enclosing lexical scope, are in \p scope.
  mlir::LLVM::DIScopeAttr getLexicalBlock(fir::ScopeAttr lexicalScope,
                                          fir::ScopeAttr outer,
                                          mlir::LLVM::DIScopeAttr scope) {
    if (!lexicalScope || !lexicalScope.isConstruct() || lexicalScope == outer)
      return scope;
    mlir::LLVM::DIScopeAttr parent =
        getLexicalBlock(lexicalScope.getParent(), outer, scope);
    if (!lexicalScopes.contains(lexicalScope))
      return parent;
    mlir::Location loc = lexicalScope.getLoc();
    auto fileLoc = loc->findInstanceOf<mlir::FileLineColLoc>();
    return mlir::LLVM::DILexicalBlockAttr::get(
        builder.getContext(), parent, fir::getFileAttrFromLoc(loc, fileAttr),
        fileLoc ? fileLoc.getLine() : 0, fileLoc ? fileLoc.getColumn() : 0);
  }

  /// Lexical blocks at the same position are the same attribute, and all the
  /// declarations of a name share a position, so skip the declarations.
  std::pair<unsigned, unsigned> getPosition(mlir::Block &block) {
    mlir::Operation *posOp = &block.front();
    for (mlir::Operation &op : block) {
      if (!llvm::isa<fir::cg::XDeclareOp>(op)) {
        posOp = &op;
        break;
      }
    }
    if (auto fileLoc = posOp->getLoc()->findInstanceOf<mlir::FileLineColLoc>())
      return {fileLoc.getLine(), fileLoc.getColumn()};
    return {0, 0};
  }

  mlir::LLVM::DIFileAttr fileAttr;
  mlir::OpBuilder builder;
};
} // namespace

void AddDebugInfoPass::handleFuncOp(mlir::func::FuncOp funcOp,
                                    mlir::LLVM::DIFileAttr fileAttr,
                                    mlir::LLVM::DICompileUnitAttr cuAttr,
                                    fir::DebugTypeGenerator &typeGen,
                                    mlir::SymbolTable *symbolTable) {
  mlir::Location l = funcOp->getLoc();
  // If fused location has already been created then nothing to do
  // Otherwise, create a fused location. A function is described by a
  // DISubprogramAttr.
  if (debugInfoIsAlreadySet<mlir::LLVM::DISubprogramAttr>(l))
    return;

  mlir::MLIRContext *context = &getContext();
  mlir::OpBuilder builder(context);
  unsigned int CC = (funcOp.getName() == fir::NameUniquer::doProgramEntry())
                        ? llvm::dwarf::getCallingConvention("DW_CC_program")
                        : llvm::dwarf::getCallingConvention("DW_CC_normal");

  mlir::StringAttr fullName = mlir::StringAttr::get(context, funcOp.getName());
  mlir::Attribute attr = funcOp->getAttr(fir::getInternalFuncNameAttrName());
  mlir::StringAttr funcName =
      (attr) ? mlir::cast<mlir::StringAttr>(attr)
             : mlir::StringAttr::get(context, funcOp.getName());

  auto result = fir::NameUniquer::deconstruct(funcName);
  funcName = mlir::StringAttr::get(context, result.second.name);

  // try to use a better function name than _QQmain for the program statement
  bool isMain = false;
  if (funcName == fir::NameUniquer::doProgramEntry()) {
    isMain = true;
    // The main program symbol name is uppercased in the cooked character stream
    // so that it cannot clash with any other symbol. Go through the presentable
    // name so that the PROGRAM name is spelled the same way here as every other
    // name in the debug information.
    funcName =
        mlir::StringAttr::get(context, fir::getPresentableFunctionName(funcOp));
  }

  bool lineTableOrDirectivesOnly =
      debugLevel == mlir::LLVM::DIEmissionKind::LineTablesOnly ||
      debugLevel == mlir::LLVM::DIEmissionKind::DebugDirectivesOnly;

  // Dont generate any type at LineTablesOnly and DebugDirectivesOnly level.
  llvm::SmallVector<mlir::LLVM::DITypeAttr> types;
  if (!lineTableOrDirectivesOnly) {
    for (auto resTy : funcOp.getResultTypes()) {
      auto tyAttr =
          typeGen.convertType(resTy, fileAttr, cuAttr, /*declOp=*/nullptr);
      types.push_back(tyAttr);
    }
    // If no return type then add a null type as a place holder for that.
    if (types.empty())
      types.push_back(mlir::LLVM::DINullTypeAttr::get(context));
    for (auto inTy : funcOp.getArgumentTypes()) {
      auto tyAttr = typeGen.convertType(fir::unwrapRefType(inTy), fileAttr,
                                        cuAttr, /*declOp=*/nullptr);
      types.push_back(tyAttr);
    }
  }

  mlir::LLVM::DISubroutineTypeAttr subTypeAttr =
      mlir::LLVM::DISubroutineTypeAttr::get(context, CC, types);
  mlir::LLVM::DIFileAttr funcFileAttr = fir::getFileAttrFromLoc(l, fileAttr);

  // Only definitions need a distinct identifier and a compilation unit.
  mlir::DistinctAttr id, id2;
  mlir::LLVM::DIScopeAttr Scope = fileAttr;
  mlir::LLVM::DICompileUnitAttr compilationUnit;
  mlir::LLVM::DISubprogramFlags subprogramFlags =
      mlir::LLVM::DISubprogramFlags{};
  if (isOptimized)
    subprogramFlags = mlir::LLVM::DISubprogramFlags::Optimized;
  if (isMain)
    subprogramFlags =
        subprogramFlags | mlir::LLVM::DISubprogramFlags::MainSubprogram;
  if (!funcOp.isExternal()) {
    // Place holder and final function have to have different IDs, otherwise
    // translation code will reject one of them.
    id = mlir::DistinctAttr::create(mlir::UnitAttr::get(context));
    id2 = mlir::DistinctAttr::create(mlir::UnitAttr::get(context));
    compilationUnit = cuAttr;
    subprogramFlags =
        subprogramFlags | mlir::LLVM::DISubprogramFlags::Definition;
  }

  // Check if the function has the pure, elemental, or recursive procedure
  // attribute
  if (fir::hasProcedureAttr<fir::FortranProcedureFlagsEnum::pure>(funcOp))
    subprogramFlags = subprogramFlags | mlir::LLVM::DISubprogramFlags::Pure;

  if (fir::hasProcedureAttr<fir::FortranProcedureFlagsEnum::elemental>(funcOp))
    subprogramFlags =
        subprogramFlags | mlir::LLVM::DISubprogramFlags::Elemental;

  if (fir::hasProcedureAttr<fir::FortranProcedureFlagsEnum::recursive>(funcOp))
    subprogramFlags =
        subprogramFlags | mlir::LLVM::DISubprogramFlags::Recursive;

  unsigned line = fir::getLineFromLoc(l);
  fir::ScopeAttr procedureScope = getProcedureScope(funcOp);
  // A target device module holds the internal procedures and the outlined
  // target regions, but not the host procedure they are contained in, which
  // only the host runs. Scoping them at it would leave a DW_TAG_subprogram
  // with no code as their parent, and a debugger does not look inside such a
  // subtree, so scope them at the file instead. Nothing on the device reads a
  // variable through that scope, so only the name qualification is lost.
  if (fir::isInternalProcedure(funcOp) && !isTargetDevice) {
    // For contained functions, the scope is the parent subroutine.
    mlir::SymbolRefAttr sym = mlir::cast<mlir::SymbolRefAttr>(
        funcOp->getAttr(fir::getHostSymbolAttrName()));
    if (sym) {
      if (auto func =
              symbolTable->lookup<mlir::func::FuncOp>(sym.getLeafReference())) {
        // Make sure that parent is processed.
        handleFuncOp(func, fileAttr, cuAttr, typeGen, symbolTable);
        if (auto fusedLoc =
                mlir::dyn_cast_if_present<mlir::FusedLoc>(func.getLoc())) {
          if (auto spAttr =
                  mlir::dyn_cast_if_present<mlir::LLVM::DISubprogramAttr>(
                      fusedLoc.getMetadata()))
            Scope = spAttr;
        }
      }
    }
  } else if (procedureScope && procedureScope.getParent() &&
             procedureScope.getParent().isModule()) {
    // A separate module procedure is mangled with the module that declares its
    // interface, so only its scope names the submodule that defines it.
    Scope = getOrCreateModuleAttr(
        getDebugModuleName(procedureScope.getParent()), fileAttr, cuAttr);
  } else if (!result.second.modules.empty()) {
    Scope = getOrCreateModuleAttr(getDebugModuleName(result.second.modules),
                                  fileAttr, cuAttr);
  }

  auto addTargetOpDISP = [&](llvm::ArrayRef<mlir::Attribute> entities) {
    // When we process the DeclareOp inside the OpenMP target region, all the
    // variables get the DISubprogram of the parent function of the target op as
    // the scope. In the codegen (to llvm ir), OpenMP target op results in the
    // creation of a separate function. As the variables in the debug info have
    // the DISubprogram of the parent function as the scope, the variables
    // need to be updated at codegen time to avoid verification failures.

    // This updating after the fact becomes more and more difficult when types
    // are dependent on local variables like in the case of variable size arrays
    // or string. We not only have to generate new variables but also new types.
    // We can avoid this problem by generating a DISubprogramAttr here for the
    // target op and make sure that all the variables inside the target region
    // get the correct scope in the first place.
    funcOp.walk([&](mlir::omp::TargetOp targetOp) {
      unsigned line = fir::getLineFromLoc(targetOp.getLoc());
      mlir::StringAttr name =
          getTargetFunctionName(context, targetOp.getLoc(), funcOp.getName());
      mlir::LLVM::DISubprogramFlags flags =
          mlir::LLVM::DISubprogramFlags::Definition |
          mlir::LLVM::DISubprogramFlags::LocalToUnit;
      if (isOptimized)
        flags = flags | mlir::LLVM::DISubprogramFlags::Optimized;

      mlir::DistinctAttr id =
          mlir::DistinctAttr::create(mlir::UnitAttr::get(context));
      llvm::SmallVector<mlir::LLVM::DITypeAttr> types;
      if (!lineTableOrDirectivesOnly) {
        types.push_back(mlir::LLVM::DINullTypeAttr::get(context));
        for (auto arg : targetOp.getRegion().getArguments()) {
          auto tyAttr =
              typeGen.convertType(fir::unwrapRefType(arg.getType()), fileAttr,
                                  cuAttr, /*declOp=*/nullptr);
          types.push_back(tyAttr);
        }
      }
      CC = llvm::dwarf::getCallingConvention("DW_CC_normal");
      mlir::LLVM::DISubroutineTypeAttr spTy =
          mlir::LLVM::DISubroutineTypeAttr::get(context, CC, types);
      if (lineTableOrDirectivesOnly || entities.empty()) {
        auto spAttr = mlir::LLVM::DISubprogramAttr::get(
            context, id, compilationUnit, Scope, name, name, funcFileAttr, line,
            line, flags, spTy, /*retainedNodes=*/{}, /*annotations=*/{});
        targetOp->setLoc(builder.getFusedLoc({targetOp.getLoc()}, spAttr));
        return;
      }
      mlir::DistinctAttr recId =
          mlir::DistinctAttr::create(mlir::UnitAttr::get(context));
      auto spAttr = mlir::LLVM::DISubprogramAttr::get(
          context, recId, /*isRecSelf=*/true, id, compilationUnit, Scope, name,
          name, funcFileAttr, line, line, flags, spTy, /*retainedNodes=*/{},
          /*annotations=*/{});

      // Make sure that information about the imported modules is copied in the
      // new function.
      llvm::SmallVector<mlir::Attribute> opEntities;
      for (mlir::Attribute N : entities) {
        if (auto entity = mlir::dyn_cast<mlir::LLVM::DIImportedEntityAttr>(N)) {
          auto importedEntity = mlir::LLVM::DIImportedEntityAttr::get(
              context, entity.getTag(), spAttr, entity.getEntity(),
              entity.getFile(), entity.getLine(), entity.getName(),
              entity.getElements());
          opEntities.push_back(importedEntity);
        }
      }

      id = mlir::DistinctAttr::create(mlir::UnitAttr::get(context));
      spAttr = mlir::LLVM::DISubprogramAttr::get(
          context, recId, /*isRecSelf=*/false, id, compilationUnit, Scope, name,
          name, funcFileAttr, line, line, flags, spTy, opEntities,
          /*annotations=*/{});
      targetOp->setLoc(builder.getFusedLoc({targetOp.getLoc()}, spAttr));
    });
  };

  // Don't process variables if user asked for line tables or debug directives
  // only.
  if (lineTableOrDirectivesOnly) {
    auto spAttr = mlir::LLVM::DISubprogramAttr::get(
        context, id, compilationUnit, Scope, funcName, fullName, funcFileAttr,
        line, line, subprogramFlags, subTypeAttr, /*retainedNodes=*/{},
        /*annotations=*/{});
    funcOp->setLoc(builder.getFusedLoc({l}, spAttr));
    addTargetOpDISP(/*entities=*/{});
    return;
  }

  mlir::LLVM::DISubprogramAttr spAttr;
  llvm::SmallVector<mlir::Attribute> retainedNodes;

  if (procedureScope && !procedureScope.getUses().empty()) {
    mlir::DistinctAttr recId =
        mlir::DistinctAttr::create(mlir::UnitAttr::get(context));
    // The debug attribute in MLIR are readonly once created. But in case of
    // imported entities, we have a circular dependency. The
    // DIImportedEntityAttr requires scope information (DISubprogramAttr in this
    // case) and DISubprogramAttr requires the list of imported entities. The
    // MLIR provides a way where a DISubprogramAttr an be created with a certain
    // recID and be used in places like DIImportedEntityAttr. After that another
    // DISubprogramAttr can be created with same recID but with list of entities
    // now available. The MLIR translation code takes care of updating the
    // references. Note that references will be updated only in the things that
    // are part of DISubprogramAttr (like DIImportedEntityAttr) so we have to
    // create the final DISubprogramAttr before we process local variables.
    // Look at DIRecursiveTypeAttrInterface for more details.
    spAttr = mlir::LLVM::DISubprogramAttr::get(
        context, recId, /*isRecSelf=*/true, id, compilationUnit, Scope,
        funcName, fullName, funcFileAttr, line, line, subprogramFlags,
        subTypeAttr, /*retainedNodes=*/{}, /*annotations=*/{});

    // Process USE statements (module globals are already processed)
    llvm::SetVector<mlir::LLVM::DIImportedEntityAttr> importedEntities;
    llvm::DenseSet<mlir::StringAttr> seenModuleNames;
    for (fir::UseAttr use : procedureScope.getUses())
      expandUseStmtForDebug(use, spAttr, fileAttr, cuAttr, importedEntities,
                            seenModuleNames);

    retainedNodes.append(importedEntities.begin(), importedEntities.end());

    // Create final DISubprogramAttr with imported entities and same recId
    spAttr = mlir::LLVM::DISubprogramAttr::get(
        context, recId, /*isRecSelf=*/false, id2, compilationUnit, Scope,
        funcName, fullName, funcFileAttr, line, line, subprogramFlags,
        subTypeAttr, retainedNodes, /*annotations=*/{});
  } else
    // No USE statements - create final DISubprogramAttr directly
    spAttr = mlir::LLVM::DISubprogramAttr::get(
        context, id, compilationUnit, Scope, funcName, fullName, funcFileAttr,
        line, line, subprogramFlags, subTypeAttr, /*retainedNodes=*/{},
        /*annotations=*/{});

  funcOp->setLoc(builder.getFusedLoc({l}, spAttr));
  addTargetOpDISP(retainedNodes);

  // Find the first dummy_scope definition. This is the one of the current
  // function. The other ones may come from inlined calls. The variables inside
  // those inlined calls should not be identified as arguments of the current
  // function.
  mlir::Value dummyScope;
  funcOp.walk([&](fir::UndefOp undef) -> mlir::WalkResult {
    // TODO: delay fir.dummy_scope translation to undefined until
    // codegeneration. This is nicer and safer to match.
    if (llvm::isa<fir::DummyScopeType>(undef.getType())) {
      dummyScope = undef;
      return mlir::WalkResult::interrupt();
    }
    return mlir::WalkResult::advance();
  });

  // Constructs such as SELECT TYPE declare a name once per branch, and lowering
  // tags the code of each branch with a lexical scope. Give each scope that
  // declares such a name a lexical block, so that only the declaration of the
  // current branch is visible.
  llvm::MapVector<llvm::StringRef, llvm::SmallVector<fir::cg::XDeclareOp>>
      declaresByName;
  funcOp.walk([&](fir::cg::XDeclareOp declOp) {
    // TODO: Scope the names in outlined OpenMP regions too. A name there may be
    // declared on a pointer loaded from the outlined function's argument, and
    // at -O0 LLVM can lose such a variable in a lexical block.
    if (declOp->getParentOfType<mlir::omp::OutlineableOpenMPOpInterface>())
      return;
    declaresByName[declOp.getUniqName()].push_back(declOp);
  });
  LexicalBlockBuilder lexicalBlocks(context, funcFileAttr);
  for (auto &declOps : llvm::make_second_range(declaresByName)) {
    if (declOps.size() < 2)
      continue;
    // A name declared in a region of an operation in the same lexical scope,
    // such as the private copy in an OpenMP loop, is visible in the region.
    for (fir::cg::XDeclareOp declOp : declOps) {
      fir::ScopeAttr lexicalScope = getLexicalScope(declOp.getLoc());
      mlir::Operation *parentOp = declOp->getParentOp();
      if (parentOp != funcOp &&
          getLexicalScope(parentOp->getLoc()) == lexicalScope)
        lexicalBlocks.scopedRegions.insert(declOp->getParentRegion());
      else if (lexicalScope)
        lexicalBlocks.lexicalScopes.insert(lexicalScope);
    }
  }
  if (!lexicalBlocks.empty())
    lexicalBlocks.run(funcOp, spAttr);

  funcOp.walk([&](fir::cg::XDeclareOp declOp) {
    mlir::LLVM::DIScopeAttr scope = getScope(declOp, spAttr);
    if (auto fusedLoc = mlir::dyn_cast<mlir::FusedLoc>(declOp.getLoc()))
      if (auto blockAttr =
              mlir::dyn_cast_if_present<mlir::LLVM::DILexicalBlockAttr>(
                  fusedLoc.getMetadata()))
        scope = blockAttr;
    handleDeclareOp(declOp, fileAttr, scope, typeGen, symbolTable, dummyScope);
  });
  funcOp.walk([&](fir::DeclareValueOp declOp) {
    mlir::LLVM::DISubprogramAttr spTy = getScope(declOp, spAttr);
    handleDeclareValueOp(declOp, fileAttr, spTy, typeGen, symbolTable,
                         dummyScope);
  });
  // commonBlockMap ensures that we don't create multiple DICommonBlockAttr of
  // the same name in one function. But it is ok (rather required) to create
  // them in different functions if common block of the same name has been used
  // there.
  commonBlockMap.clear();
}

// Expand the USE statements of modules described in this compilation unit.
void AddDebugInfoPass::expandUseStmtForDebug(
    fir::UseAttr use, mlir::LLVM::DISubprogramAttr spAttr,
    mlir::LLVM::DIFileAttr fileAttr, mlir::LLVM::DICompileUnitAttr cuAttr,
    llvm::SetVector<mlir::LLVM::DIImportedEntityAttr> &importedEntities,
    llvm::DenseSet<mlir::StringAttr> &seenModuleNames) {
  mlir::StringAttr module = use.getModule();
  if (!seenModuleNames.insert(module).second)
    return;

  auto file = fir::getFileAttrFromLoc(use.getLoc(), fileAttr);
  unsigned line = fir::getLineFromLoc(use.getLoc());
  llvm::SmallVector<mlir::LLVM::DINodeAttr> renames;
  for (fir::UseEntityAttr entity : use.getEntities()) {
    auto gv = moduleVariables.lookup(
        {use.getEntityModule(entity), use.getEntityName(entity)});
    if (!gv)
      continue;
    mlir::StringAttr local = entity.getLocalName();
    if (local == gv.getName())
      local = {};
    auto decl = mlir::LLVM::DIImportedEntityAttr::get(
        &getContext(), llvm::dwarf::DW_TAG_imported_declaration, spAttr, gv,
        file, line, local, /*elements=*/{});
    if (use.getOnly())
      importedEntities.insert(decl);
    else
      renames.push_back(decl);
  }
  // Preserve whole-module debug lookup when lowering could not resolve any
  // ONLY items in the program unit, as for USE statements inside a BLOCK.
  if (use.getOnly() && !use.getEntities().empty())
    return;
  importedEntities.insert(mlir::LLVM::DIImportedEntityAttr::get(
      &getContext(), llvm::dwarf::DW_TAG_imported_module, spAttr,
      getOrCreateModuleAttr(module.str(), fileAttr, cuAttr), file, line,
      /*name=*/nullptr, renames));
  if (auto scope = moduleScopes.lookup(module.getValue()))
    for (fir::UseAttr child : scope.getUses())
      expandUseStmtForDebug(child, spAttr, fileAttr, cuAttr, importedEntities,
                            seenModuleNames);
}

// A submodule has access to everything in the submodule directly above it, and
// through that one to the whole of its ancestry, but the name of its
// DW_TAG_module says nothing about it. Import the parent into each submodule,
// as gfortran does, so that a debugger stopped in a procedure of a submodule
// can still find the entities declared above it. The imports of a nested
// submodule form a chain that it walks up to the module at the root.
//
// These belong to the compile unit rather than to any one procedure, so that a
// submodule needs a single one however many procedures it holds. That makes the
// compile unit refer to a module that refers back to it, which is what the
// recursive attribute the caller builds is for.
//
// The modules named here are built directly rather than taken from the cache,
// because they name the placeholder unit while every other use of them names
// the real one, and only the latter should be handed out. Everything else about
// them is settled before either is built, so the two describe the same module
// and become one once the unit they name is resolved.
llvm::SmallVector<mlir::LLVM::DINodeAttr>
AddDebugInfoPass::buildSubmoduleImports(
    llvm::ArrayRef<fir::ScopeAttr> moduleScopes,
    mlir::LLVM::DIFileAttr fileAttr, mlir::LLVM::DIScopeAttr cuAttr) {
  mlir::MLIRContext *context = &getContext();
  llvm::SmallVector<mlir::LLVM::DINodeAttr> imports;
  for (fir::ScopeAttr moduleScope : moduleScopes) {
    if (!moduleScope.getParent() ||
        mlir::isa<mlir::UnknownLoc>(moduleScope.getLoc()))
      continue;
    mlir::Location loc = moduleScope.getLoc();
    imports.push_back(mlir::LLVM::DIImportedEntityAttr::get(
        context, llvm::dwarf::DW_TAG_imported_module,
        createModuleAttr(getDebugModuleName(moduleScope), fileAttr, cuAttr),
        createModuleAttr(getDebugModuleName(moduleScope.getParent()), fileAttr,
                         cuAttr),
        fir::getFileAttrFromLoc(loc, fileAttr), fir::getLineFromLoc(loc),
        /*name=*/mlir::StringAttr(), /*elements=*/{}));
  }
  return imports;
}

void AddDebugInfoPass::runOnOperation() {
  mlir::ModuleOp module = getOperation();
  mlir::MLIRContext *context = &getContext();
  mlir::SymbolTable symbolTable(module);
  if (auto offloadMod =
          mlir::dyn_cast<mlir::omp::OffloadModuleInterface>(*module))
    isTargetDevice = offloadMod.getIsTargetDevice();
  // Work out which modules this compilation unit defines. It has to be settled
  // before any DIModuleAttr is built, because the attribute is immutable and
  // the first member to mention a module fixes it for all the others. Lowering
  // lists them in the location of the module: no individual member can say it.
  // An array named constant from an intrinsic module, for one, is materialized
  // locally in every unit that uses it, although the module is defined in no
  // object file at all.
  llvm::SmallVector<fir::ScopeAttr> listedModuleScopes;
  moduleScopes.clear();
  moduleVariables.clear();
  if (auto fusedLoc =
          module.getLoc()
              ->findInstanceOf<mlir::FusedLocWith<mlir::ArrayAttr>>())
    for (mlir::Attribute attr : fusedLoc.getMetadata())
      if (auto moduleScope = mlir::dyn_cast<fir::ScopeAttr>(attr);
          moduleScope && moduleScope.isModule()) {
        listedModuleScopes.push_back(moduleScope);
        moduleScopes[getDebugModuleName(moduleScope)] = moduleScope;
      }
  llvm::StringRef fileName;
  std::string filePath;
  std::optional<mlir::DataLayout> dl =
      fir::support::getOrSetMLIRDataLayout(module, /*allowDefaultLayout=*/true);
  if (!dl) {
    mlir::emitError(module.getLoc(), "Missing data layout attribute in module");
    signalPassFailure();
    return;
  }
  mlir::OpBuilder builder(context);
  if (dwarfVersion > 0) {
    mlir::OpBuilder::InsertionGuard guard(builder);
    builder.setInsertionPointToEnd(module.getBody());
    llvm::SmallVector<mlir::Attribute> moduleFlags;
    mlir::IntegerType int32Ty = mlir::IntegerType::get(context, 32);
    moduleFlags.push_back(builder.getAttr<mlir::LLVM::ModuleFlagAttr>(
        mlir::LLVM::ModFlagBehavior::Max,
        mlir::StringAttr::get(context, "Dwarf Version"),
        mlir::IntegerAttr::get(int32Ty, dwarfVersion)));
    mlir::LLVM::ModuleFlagsOp::create(builder, module.getLoc(),
                                      builder.getArrayAttr(moduleFlags));
  }
  // We need 2 type of file paths here.
  // 1. Name of the file as was presented to compiler. This can be absolute
  // or relative to 2.
  // 2. Current working directory
  //
  // We are also dealing with 2 different situations below. One is normal
  // compilation where we will have a value in 'inputFilename' and we can
  // obtain the current directory using 'current_path'.
  // The 2nd case is when this pass is invoked directly from 'fir-opt' tool.
  // In that case, 'inputFilename' may be empty. Location embedded in the
  // module will be used to get file name and its directory.
  if (inputFilename.empty()) {
    if (auto fileLoc =
            module.getLoc()->findInstanceOf<mlir::FileLineColLoc>()) {
      fileName = llvm::sys::path::filename(fileLoc.getFilename().getValue());
      filePath = llvm::sys::path::parent_path(fileLoc.getFilename().getValue());
    } else
      fileName = "-";
  } else {
    fileName = inputFilename;
    llvm::SmallString<256> cwd;
    if (!llvm::sys::fs::current_path(cwd))
      filePath = cwd.str();
  }

  mlir::LLVM::DIFileAttr fileAttr =
      mlir::LLVM::DIFileAttr::get(context, fileName, filePath);
  // Match Clang style by starting with the full compiler version and
  // appending -dwarf-debug-flags content when provided.
  std::string producerString = Fortran::common::getFlangFullVersion();
  if (!dwarfDebugFlags.empty())
    producerString += " " + dwarfDebugFlags;
  mlir::StringAttr producer = mlir::StringAttr::get(context, producerString);
  // A unit that only emits line directives has no .debug_info, so the header
  // of its compile unit, and with it the label the DWARF 5 accelerator table
  // points at, is never written. Asking for a name table would have that table
  // reference a label that does not exist. Clang leaves the name table off for
  // the same reason.
  mlir::LLVM::DINameTableKind cuNameTableKind =
      debugLevel == mlir::LLVM::DIEmissionKind::DebugDirectivesOnly
          ? mlir::LLVM::DINameTableKind::None
          : nameTableKind;
  // Each submodule compiled here imports the one above it, and those entries
  // hang off the compile unit. They are scoped at modules that name the unit
  // they are part of, so build them against a placeholder of it first, and give
  // the unit the matching recursive id once it holds them.
  mlir::DistinctAttr cuRecId =
      mlir::DistinctAttr::create(mlir::UnitAttr::get(context));
  llvm::SmallVector<mlir::LLVM::DINodeAttr> cuImports = buildSubmoduleImports(
      listedModuleScopes, fileAttr,
      mlir::cast<mlir::LLVM::DICompileUnitAttr>(
          mlir::LLVM::DICompileUnitAttr::getRecSelf(cuRecId)));

  mlir::LLVM::DICompileUnitAttr cuAttr = mlir::LLVM::DICompileUnitAttr::get(
      mlir::DistinctAttr::create(mlir::UnitAttr::get(context)),
      llvm::dwarf::getLanguage("DW_LANG_Fortran95"), fileAttr, producer,
      isOptimized, debugLevel, debugInfoForProfiling, cuNameTableKind,
      splitDwarfFile.empty() ? mlir::StringAttr()
                             : mlir::StringAttr::get(context, splitDwarfFile),
      cuImports);
  // A unit that compiles no submodule holds no import, and stays as it was.
  if (!cuImports.empty())
    cuAttr =
        mlir::cast<mlir::LLVM::DICompileUnitAttr>(cuAttr.withRecId(cuRecId));

  fir::DebugTypeGenerator typeGen(
      module, &symbolTable, *dl, boundsInStackSlots,
      [&](llvm::ArrayRef<std::string> modules) -> mlir::LLVM::DIScopeAttr {
        return getOrCreateModuleAttr(getDebugModuleName(modules), fileAttr,
                                     cuAttr);
      });

  // Process module globals early.
  // Walk through all DeclareOps in functions and process globals that are
  // module data objects. This ensures that when we process USE statements,
  // the DIGlobalVariable lookups will succeed.
  if (debugLevel == mlir::LLVM::DIEmissionKind::Full) {
    module.walk([&](fir::cg::XDeclareOp declOp) {
      mlir::Operation *defOp = declOp.getMemref().getDefiningOp();
      if (defOp && llvm::isa<fir::AddrOfOp>(defOp)) {
        if (auto globalOp =
                symbolTable.lookup<fir::GlobalOp>(declOp.getUniqName())) {
          // Only process module data objects here, not SAVE variables
          if (isModuleDataObject(globalOp)) {
            handleGlobalOp(globalOp, fileAttr, cuAttr, typeGen, &symbolTable,
                           declOp);
          }
        }
      }
    });
    // A USE identifies a variable by its source module and name. Prefer the
    // type supplied by a declaration above when one is available.
    for (fir::GlobalOp global : module.getOps<fir::GlobalOp>()) {
      if (!isModuleDataObject(global))
        continue;
      handleGlobalOp(global, fileAttr, cuAttr, typeGen, &symbolTable, {});
      if (auto gv = lookupDIGlobalVariable(global))
        if (auto scope = mlir::dyn_cast_if_present<mlir::LLVM::DIModuleAttr>(
                gv->getScope()))
          moduleVariables[{scope.getName(), gv->getName()}] = *gv;
    }
  }

  module.walk([&](mlir::func::FuncOp funcOp) {
    handleFuncOp(funcOp, fileAttr, cuAttr, typeGen, &symbolTable);
  });
  // We have processed all function. Attach common block variables to the
  // global that represent the storage.
  for (auto [global, exprs] : globalToGlobalExprsMap) {
    auto arrayAttr = mlir::ArrayAttr::get(context, exprs);
    global->setLoc(builder.getFusedLoc({global.getLoc()}, arrayAttr));
  }
  // Process any global which was not processed through DeclareOp.
  if (debugLevel == mlir::LLVM::DIEmissionKind::Full) {
    // Process 'GlobalOp' only if full debug info is requested.
    for (auto globalOp : module.getOps<fir::GlobalOp>())
      handleGlobalOp(globalOp, fileAttr, cuAttr, typeGen, &symbolTable,
                     /*declOp=*/nullptr);
  }
}

std::unique_ptr<mlir::Pass>
fir::createAddDebugInfoPass(fir::AddDebugInfoOptions options) {
  return std::make_unique<AddDebugInfoPass>(options);
}
