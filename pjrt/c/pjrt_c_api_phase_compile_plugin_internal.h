/* Copyright 2023 The OpenXLA Authors.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

#ifndef TENSORFLOW_COMPILER_XLA_PJRT_C_PJRT_C_API_PHASE_COMPILE_PLUGIN_INTERNAL_H_
#define TENSORFLOW_COMPILER_XLA_PJRT_C_PJRT_C_API_PHASE_COMPILE_PLUGIN_INTERNAL_H_

#include <string>

#include "third_party/absl/status/status.h"
#include "third_party/absl/status/statusor.h"
#include "third_party/llvm/llvm-project/llvm/include/llvm/Support/LogicalResult.h"
#include "third_party/llvm/llvm-project/llvm/include/llvm/Support/raw_ostream.h"
#include "third_party/llvm/llvm-project/mlir/include/mlir/IR/BuiltinOps.h"
#include "third_party/llvm/llvm-project/mlir/include/mlir/IR/MLIRContext.h"
#include "third_party/llvm/llvm-project/mlir/include/mlir/IR/OwningOpRef.h"
#include "third_party/stablehlo/stablehlo/api/PortableApi.h"
#include "third_party/stablehlo/stablehlo/dialect/Serialization.h"
#include "third_party/tensorflow/compiler/xla/pjrt/c/pjrt_c_api.h"

namespace pjrt {
namespace phase_compile_cpu_plugin {

class StableHLOTypeSerialization {
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
  StableHLOTypeSerialization() = delete;
};

const PJRT_Api* GetPhaseCompileForCpuPjrtApi();

}  // namespace phase_compile_cpu_plugin
}  // namespace pjrt

#endif  // TENSORFLOW_COMPILER_XLA_PJRT_C_PJRT_C_API_PHASE_COMPILE_PLUGIN_INTERNAL_H_
