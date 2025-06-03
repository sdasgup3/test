#include "experimental/users/sdasgup/pjrt/phase_compilation/plugin.h"

#include <cstdio>
#include <cstdlib>
#include <functional>
#include <iostream>
#include <map>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "experimental/users/sdasgup/pjrt/phase_compilation/portable_typedef.h"
#include "third_party/absl/container/flat_hash_map.h"
#include "third_party/absl/status/status.h"
#include "third_party/absl/status/statusor.h"
#include "third_party/absl/strings/str_cat.h"
#include "third_party/llvm/llvm-project/mlir/include/mlir/IR/BuiltinOps.h"
#include "third_party/llvm/llvm-project/mlir/include/mlir/IR/MLIRContext.h"
#include "third_party/llvm/llvm-project/mlir/include/mlir/IR/OwningOpRef.h"
#include "third_party/tensorflow/compiler/xla/hlo/ir/hlo_module.h"
#include "third_party/tensorflow/compiler/xla/hlo/translate/stablehlo.h"
#include "third_party/tensorflow/compiler/xla/pjrt/c/pjrt_c_api.h"
#include "third_party/tensorflow/compiler/xla/pjrt/c/pjrt_c_api_layouts_extension.h"
#include "third_party/tensorflow/compiler/xla/pjrt/c/pjrt_c_api_partial_compile_extension.h"
#include "third_party/tensorflow/compiler/xla/pjrt/c/pjrt_c_api_partial_compile_internal.h"
#include "third_party/tensorflow/compiler/xla/pjrt/c/pjrt_c_api_wrapper_impl.h"
#include "third_party/tensorflow/compiler/xla/pjrt/pjrt_client.h"
#include "third_party/tensorflow/compiler/xla/pjrt/pjrt_compiler.h"
#include "third_party/tensorflow/compiler/xla/pjrt/pjrt_partial_program.h"

namespace xla {

PJRT_Error* PJRT_ShloToHloExecuteContext_Create(
    PJRT_ExecuteContext_Create_Args* args) {
  return new PJRT_Error{absl::UnimplementedError(
      "ExecuteContext not supported for HloInterpreter execution.")};
}

PJRT_Error* PJRT_ShloToHloDeviceTopology_Create(
    PJRT_TopologyDescription_Create_Args* args) {
  return new PJRT_Error{absl::UnimplementedError(
      "Topology not supported for HloInterpreter compilation.")};
}

enum class PjRtPartialProgramFormat {
  kStablehloBytecode = 0,
  kHloProto = 1,
  kIsaProgramProto = 2,
  kUnknown = -1
};

absl::Status Phase1ValidateInput(
    const std::vector<PjRtPartialProgram>& input_programs) {
  for (const auto& input_program : input_programs) {
    if (input_program.GetFormat() !=
        static_cast<size_t>(
            xla::PjRtPartialProgramFormat::kStablehloBytecode)) {
      return absl::InvalidArgumentError(
          "Input programs are not in SHLO format.");
    }
  }
  return absl::OkStatus();
}

absl::StatusOr<std::vector<PjRtPartialProgram>> Phase1PartialCompile(
    xla::CompileOptions compile_options,
    const std::vector<PjRtPartialProgram>& input_programs) {
  auto status = Phase1ValidateInput(input_programs);
  if (!status.ok()) {
    return status;
  }

  // Deserialize to mlir module
  mlir::MLIRContext context;
  std::vector<mlir::OwningOpRef<mlir::ModuleOp>> deserialized_input_object;
  for (const auto& input_program : input_programs) {
    absl::StatusOr<mlir::OwningOpRef<mlir::ModuleOp>>
        deserialized_input_status =
            SHLOType::deserialize(input_program.GetProgram(), context);
    if (!deserialized_input_status.ok()) {
      return absl::InvalidArgumentError(
          absl::StrCat("Input deserialization failed: ",
                       deserialized_input_status.status().message()));
    }
    deserialized_input_object.push_back(
        std::move(deserialized_input_status.value()));
  }

  // Convert to HLO module
  std::vector<std::unique_ptr<xla::HloModule>> deserialized_output_objects;
  for (const auto& module_op : deserialized_input_object) {
    absl::StatusOr<std::unique_ptr<xla::HloModule>> deserialized_output_status =
        xla::ConvertStablehloToHlo(module_op.get());

    if (!deserialized_output_status.ok()) {
      return absl::InvalidArgumentError(
          absl::StrCat("Conversion failed: Could not convert input object to "
                       "output object: ",
                       deserialized_output_status.status().message()));
    }
    auto deserialized_output_object =
        std::move(deserialized_output_status).value();
    deserialized_output_objects.push_back(
        std::move(deserialized_output_object));
  }

  // Serialize HLO module
  std::vector<PjRtPartialProgram> serialized_output_objects;
  for (const auto& hlo_module : deserialized_output_objects) {
    absl::StatusOr<std::string> serialized_output_status =
        HLOType::serialize(hlo_module);
    if (!serialized_output_status.ok()) {
      return absl::InternalError(
          absl::StrCat("Output serialization failed: ",
                       serialized_output_status.status().message()));
    }

    PjRtPartialProgram serialized_output_object;
    serialized_output_object.SetProgram(serialized_output_status.value());
    serialized_output_object.SetFormat(
        static_cast<size_t>(xla::PjRtPartialProgramFormat::kHloProto));
    serialized_output_object.SetGeneratingPhase("shlo_to_hlo");
    serialized_output_object.SetNextPhases({"hlo_to_hlo"});
    serialized_output_object.SetVersion("1.0");

    serialized_output_objects.push_back(std::move(serialized_output_object));
  }

  return serialized_output_objects;
}

struct MyExperimentalPartialPjrtCompiler : public xla::PjRtPhaseCompiler {
  absl::flat_hash_map<std::string, PhaseCompiler> phases_map;
  std::vector<std::string> phase_names;

  absl::Status RegisterPhase(const std::string& phase_name,
                             PhaseCompiler phase_compiler) final {
    if (phase_name.empty()) {
      return absl::InvalidArgumentError("Phase name cannot be empty");
    }
    if (phase_compiler == nullptr) {
      return absl::InvalidArgumentError("Phase compiler cannot be null");
    }
    auto it = phases_map.insert({phase_name, phase_compiler});
    if (!it.second) {
      return absl::AlreadyExistsError(
          absl::StrCat("Phase name ", phase_name, " already exists"));
    }
    phase_names.push_back(phase_name);
    return absl::OkStatus();
  }

  absl::StatusOr<PhaseCompiler> GetPhaseCompiler(
      const std::string& phase_name) final {
    auto it = phases_map.find(phase_name);
    if (it == phases_map.end()) {
      return absl::NotFoundError(absl::StrCat(
          "No partial compile phase found for phase name ", phase_name));
    }
    return it->second;
  }

  absl::StatusOr<std::vector<std::string>> GetPhaseNames() final {
    return phase_names;
  }
};

}  // namespace xla

const PJRT_Api* GetPjrtApi() {
  printf("C++ Calling GetPjrtApi");
  static PJRT_Layouts_Extension layouts_extension =
      pjrt::CreateLayoutsExtension(nullptr);

  // Create phases
  std::string phase_name = "shlo_to_hlo";
  auto compiler = std::make_unique<xla::MyExperimentalPartialPjrtCompiler>();
  auto status = compiler->RegisterPhase(phase_name, xla::Phase1PartialCompile);
  if (!status.ok()) {
    std::cerr << "Failed to register partial compiler: " << status.message()
              << "\n";
    exit(1);
  }
  xla::PjRtRegisterCompiler("partial_compile", std::move(compiler));

  // Create partial compile extension
  static PJRT_PartialCompile_Extension partial_compile_extension =
      pjrt::CreatePartialCompileExtension(&layouts_extension.base);

  static const PJRT_Api pjrt_api = pjrt::CreatePjrtApi(
      nullptr, xla::PJRT_ShloToHloExecuteContext_Create,
      xla::PJRT_ShloToHloDeviceTopology_Create,
      pjrt::PJRT_Plugin_Initialize_NoOp, &partial_compile_extension.base,
      pjrt::PJRT_Plugin_Attributes_Xla);

  return &pjrt_api;
}
