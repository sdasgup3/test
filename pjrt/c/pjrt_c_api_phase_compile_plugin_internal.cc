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

#include "third_party/tensorflow/compiler/xla/pjrt/c/pjrt_c_api_phase_compile_plugin_internal.h"

#include <cstdio>
#include <cstdlib>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "third_party/absl/status/status.h"
#include "third_party/absl/status/statusor.h"
#include "third_party/absl/strings/str_cat.h"
#include "third_party/absl/strings/str_format.h"
#include "third_party/absl/strings/string_view.h"
#include "third_party/llvm/llvm-project/llvm/include/llvm/Support/LogicalResult.h"
#include "third_party/llvm/llvm-project/mlir/include/mlir/Dialect/Func/IR/FuncOps.h"
#include "third_party/llvm/llvm-project/mlir/include/mlir/IR/BuiltinOps.h"
#include "third_party/llvm/llvm-project/mlir/include/mlir/IR/MLIRContext.h"
#include "third_party/llvm/llvm-project/mlir/include/mlir/IR/OwningOpRef.h"
#include "third_party/llvm/llvm-project/mlir/include/mlir/Pass/PassManager.h"
#include "third_party/llvm/llvm-project/mlir/include/mlir/Transforms/Passes.h"
#include "third_party/stablehlo/stablehlo/transforms/optimization/Passes.h"
#include "third_party/tensorflow/compiler/xla/pjrt/c/pjrt_c_api.h"
#include "third_party/tensorflow/compiler/xla/pjrt/c/pjrt_c_api_layouts_extension.h"
#include "third_party/tensorflow/compiler/xla/pjrt/c/pjrt_c_api_phase_compile_extension.h"
#include "third_party/tensorflow/compiler/xla/pjrt/c/pjrt_c_api_phase_compile_internal.h"
#include "third_party/tensorflow/compiler/xla/pjrt/c/pjrt_c_api_wrapper_impl.h"
#include "third_party/tensorflow/compiler/xla/pjrt/pjrt_client.h"
#include "third_party/tensorflow/compiler/xla/pjrt/pjrt_compiler.h"
#include "third_party/tensorflow/compiler/xla/pjrt/proto/pjrt_partial_program.proto.h"

namespace pjrt {
namespace phase_compile_cpu_plugin {

namespace {

enum class PjRtPartialProgramFormat { kStablehloBytecode = 0, kUnknown = -1 };

constexpr absl::string_view kPhaseName = "stablehlo_to_optimized_stablehlo";

absl::Status PhaseValidator(
    const std::vector<xla::PjRtPartialProgramProto>& input_programs) {
  for (const auto& input_program : input_programs) {
    if (input_program.program_format() !=
        static_cast<size_t>(PjRtPartialProgramFormat::kStablehloBytecode)) {
      return absl::InvalidArgumentError(
          "Input programs are not in SHLO format.");
    }
  }
  return absl::OkStatus();
}

absl::StatusOr<std::vector<xla::PjRtPartialProgramProto>> PhaseCompiler(
    xla::CompileOptions compile_options,
    const std::vector<xla::PjRtPartialProgramProto>& input_programs) {
  std::vector<xla::PjRtPartialProgramProto> serialized_output_objects;
  mlir::MLIRContext context;

  for (const auto& input_program : input_programs) {
    // Deserialize from PjRtPartialProgramProto to StableHLO module
    absl::StatusOr<mlir::OwningOpRef<mlir::ModuleOp>>
        deserialized_input_status = StableHLOTypeSerialization::deserialize(
            input_program.program(), context);
    if (!deserialized_input_status.ok()) {
      return absl::InvalidArgumentError(
          absl::StrCat("Input deserialization failed: ",
                       deserialized_input_status.status().message()));
    }

    mlir::OwningOpRef<mlir::ModuleOp> current_module =
        std::move(deserialized_input_status.value());

    // Convert stablehlo to optimized stablehlo
    mlir::PassManager pm(current_module->getContext());
    mlir::GreedyRewriteConfig config;
    pm.addNestedPass<mlir::func::FuncOp>(
        mlir::stablehlo::createStablehloAggressiveSimplificationPass({},
                                                                     config));
    if (failed(pm.run(current_module.get()))) {
      return absl::InvalidArgumentError("Failed to simplify StableHLO module");
    }

    // Serialize to PjRtPartialProgramProto
    absl::StatusOr<std::string> serialized_output_status =
        StableHLOTypeSerialization::serialize(
            current_module);  // Pass OwningOpRef directly
    if (!serialized_output_status.ok()) {
      return absl::InternalError(
          absl::StrCat("Output serialization failed: ",
                       serialized_output_status.status().message()));
    }

    xla::PjRtPartialProgramProto serialized_output_object;
    serialized_output_object.set_program(serialized_output_status.value());
    serialized_output_object.set_program_format(
        static_cast<size_t>(PjRtPartialProgramFormat::kStablehloBytecode));
    serialized_output_object.set_generating_phase(std::string(kPhaseName));
    serialized_output_object.add_next_phases({"stablehlo_to_hlo"});
    serialized_output_object.set_version("1.0");

    serialized_output_objects.push_back(std::move(serialized_output_object));
  }

  return serialized_output_objects;
}

}  // namespace

PJRT_Error* PJRT_ExecuteContext_Create(PJRT_ExecuteContext_Create_Args* args) {
  return new PJRT_Error{absl::UnimplementedError(
      "ExecuteContext not implemented for phase compile CPU.")};
}

PJRT_Error* PJRT_DeviceTopology_Create(
    PJRT_TopologyDescription_Create_Args* args) {
  return new PJRT_Error{absl::UnimplementedError(
      "Topology not implemented for phase compile CPU.")};
}

const PJRT_Api* GetPhaseCompileForCpuPjrtApi() {
  static PJRT_Layouts_Extension layouts_extension =
      pjrt::CreateLayoutsExtension(nullptr);

  // Create phases
  auto compiler = std::make_unique<xla::PjRtPhaseCompiler>();
  auto status = compiler->RegisterPhase(std::string(kPhaseName), PhaseCompiler,
                                        PhaseValidator);
  if (!status.ok()) {
    absl::FPrintF(stderr, "Failed to register partial compiler: %s\n",
                  status.message());
    return nullptr;
  }
  xla::PjRtRegisterCompiler("partial_compile", std::move(compiler));

  // Create partial compile extension
  static PJRT_PhaseCompile_Extension phase_compile_extension =
      pjrt::CreatePhaseCompileExtension(&layouts_extension.base);

  static const PJRT_Api pjrt_api = pjrt::CreatePjrtApi(
      nullptr, PJRT_ExecuteContext_Create, PJRT_DeviceTopology_Create,
      pjrt::PJRT_Plugin_Initialize_NoOp, &phase_compile_extension.base,
      pjrt::PJRT_Plugin_Attributes_Xla);

  return &pjrt_api;
}

}  // namespace phase_compile_cpu_plugin
}  // namespace pjrt
