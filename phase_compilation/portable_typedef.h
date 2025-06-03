#ifndef EXPERIMENTAL_USERS_SDASGUP_PJRT_PHASE_COMPILATION_PORTABLE_TYPEDEF_H_
#define EXPERIMENTAL_USERS_SDASGUP_PJRT_PHASE_COMPILATION_PORTABLE_TYPEDEF_H_

#include <memory>
#include <string>
#include <utility>

#include "third_party/absl/status/status.h"
#include "third_party/absl/status/statusor.h"
#include "third_party/absl/strings/str_cat.h"
#include "third_party/llvm/llvm-project/llvm/include/llvm/Support/LogicalResult.h"
#include "third_party/llvm/llvm-project/llvm/include/llvm/Support/raw_ostream.h"
#include "third_party/llvm/llvm-project/mlir/include/mlir/IR/BuiltinOps.h"
#include "third_party/llvm/llvm-project/mlir/include/mlir/IR/MLIRContext.h"
#include "third_party/llvm/llvm-project/mlir/include/mlir/IR/OwningOpRef.h"
#include "third_party/stablehlo/stablehlo/api/PortableApi.h"
#include "third_party/stablehlo/stablehlo/dialect/Serialization.h"
#include "third_party/tensorflow/compiler/xla/debug_options_flags.h"
#include "third_party/tensorflow/compiler/xla/hlo/ir/hlo_module.h"

// SHLOType is a wrapper around the StableHLO portable API.
class SHLOType {
 public:
  static absl::StatusOr<mlir::OwningOpRef<mlir::ModuleOp>> deserialize(
      const std::string& program, mlir::MLIRContext& context) {
    mlir::OwningOpRef<mlir::ModuleOp> module_op =
        mlir::stablehlo::deserializePortableArtifact(program, &context);

    if (!module_op) {
      return absl::InvalidArgumentError(
          "SHLOFormat deserialization failed: Invalid StableHLO artifact");
    }
    return module_op;
  }

  static absl::StatusOr<std::string> serialize(
      const mlir::OwningOpRef<mlir::ModuleOp>& module_op) {
    auto version = mlir::stablehlo::getCurrentVersion();
    std::string bytecode;
    llvm::raw_string_ostream os(bytecode);
    if (failed(mlir::stablehlo::serializePortableArtifact(*module_op, version,
                                                          os, true))) {
      return absl::InvalidArgumentError(
          "SHLOFormat serialization failed: Could not serialize MLIR module");
    }
    return bytecode;
  }

 private:
  SHLOType() = delete;
};

// HLOType is a wrapper around the HLO portable API.
class HLOType {
 public:
  static absl::StatusOr<std::unique_ptr<xla::HloModule>> deserialize(
      const std::string& program) {
    xla::HloModuleProto proto;
    if (!proto.ParseFromString(program)) {
      return absl::InvalidArgumentError(
          "HLOFormat deserialization failed: Invalid HLO protobuf");
    }
    auto module_config = xla::HloModule::CreateModuleConfigFromProto(
        proto, xla::GetDebugOptionsFromFlags());
    auto hlo_module_status =
        xla::HloModule::CreateFromProto(proto, *module_config);

    if (!hlo_module_status.ok()) {
      return absl::InvalidArgumentError(
          absl::StrCat("HLOFormat deserialization failed: Could not create "
                       "HloModule from proto: ",
                       hlo_module_status.status().message()));
    }

    return std::move(hlo_module_status.value());
  }

  static absl::StatusOr<std::string> serialize(
      const std::unique_ptr<xla::HloModule>& hlo_module) {
    return hlo_module->ToProto().SerializeAsString();
  }

 private:
  HLOType() = delete;
};

#endif  // EXPERIMENTAL_USERS_SDASGUP_PJRT_PHASE_COMPILATION_PORTABLE_TYPEDEF_H_
