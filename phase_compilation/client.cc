#include <cstdlib>
#include <iostream>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "experimental/users/sdasgup/pjrt/phase_compilation/plugin.h"
#include "experimental/users/sdasgup/pjrt/phase_compilation/portable_typedef.h"
#include "learning/brain/research/pjrt/pjrt_topology_utils.h"
#include "testing/base/public/gmock.h"
#include "testing/base/public/gunit.h"
#include "third_party/absl/status/statusor.h"
#include "third_party/absl/strings/string_view.h"
#include "third_party/llvm/llvm-project/mlir/include/mlir/IR/BuiltinOps.h"
#include "third_party/llvm/llvm-project/mlir/include/mlir/IR/MLIRContext.h"
#include "third_party/llvm/llvm-project/mlir/include/mlir/IR/OwningOpRef.h"
#include "third_party/stablehlo/stablehlo/reference/Api.h"
#include "third_party/tensorflow/compiler/xla/backends/cpu/codegen/cpu_features.h"
#include "third_party/tensorflow/compiler/xla/pjrt/c/pjrt_c_api.h"
#include "third_party/tensorflow/compiler/xla/pjrt/c/pjrt_c_api_helpers.h"
#include "third_party/tensorflow/compiler/xla/pjrt/c/pjrt_c_api_partial_compile_extension.h"
#include "third_party/tensorflow/compiler/xla/pjrt/c/pjrt_c_api_partial_compile_internal.h"
#include "third_party/tensorflow/compiler/xla/pjrt/c/pjrt_c_api_partial_compile_utils.h"
#include "third_party/tensorflow/compiler/xla/pjrt/pjrt_compiler.h"
#include "third_party/tensorflow/compiler/xla/pjrt/pjrt_executable.h"
#include "third_party/tensorflow/compiler/xla/pjrt/pjrt_partial_program.h"
#include "third_party/tensorflow/compiler/xla/pjrt/plugin/xla_cpu/cpu_topology_description.h"
#include "third_party/tensorflow/compiler/xla/xla.proto.h"

namespace {

constexpr absl::string_view kShloModuleStr1 = R"(
  module {
    func.func @main(%arg0: tensor<4xi32>) -> tensor<4xi32> {
      %0 = stablehlo.constant dense<1> : tensor<4xi32>
      %1 = stablehlo.add %arg0, %0 : tensor<4xi32>
      func.return %1 : tensor<4xi32>
    }
  }
  )";

constexpr absl::string_view kShloModuleStr2 = R"(
  module {
    func.func @main(%arg0: tensor<4xi32>) -> tensor<4xi32> {
      %0 = stablehlo.constant dense<1> : tensor<4xi32>
      %1 = stablehlo.subtract %arg0, %0 : tensor<4xi32>
      func.return %1 : tensor<4xi32>
    }
  }
  )";

std::vector<xla::PjRtPartialProgram> PreparePartialPrograms() {
  std::string program_code1{kShloModuleStr1};
  std::string program_code2{kShloModuleStr2};

  mlir::MLIRContext context;
  std::vector<mlir::OwningOpRef<mlir::ModuleOp>> shlo_modules;
  shlo_modules.push_back(
      mlir::stablehlo::parseStablehloModule(program_code1, context).value());
  shlo_modules.push_back(
      mlir::stablehlo::parseStablehloModule(program_code2, context).value());

  std::vector<std::string> bytecodes;
  for (const auto& shlo_module : shlo_modules) {
    auto bytecode_status = SHLOType::serialize(shlo_module);
    if (!bytecode_status.ok()) {
      exit(1);
    }
    bytecodes.push_back(std::move(bytecode_status.value()));
  }

  std::vector<xla::PjRtPartialProgram> partial_programs;
  for (const auto& bytecode : bytecodes) {
    xla::PjRtPartialProgram partial_program;
    partial_program.SetProgram(bytecode);
    partial_program.SetFormat(
        0);  // 0 expresses StableHLO bytecode per plugin.cc
    partial_program.SetGeneratingPhase("n/a");
    partial_program.SetNextPhases({"shlo_to_hlo"});
    partial_program.SetVersion("1.0");
    partial_programs.push_back(partial_program);
  }
  return partial_programs;
}

}  // namespace

TEST(ShloToHloCApiTest, PartialCompileViaExtension) {
  const PJRT_Api* api = GetPjrtApi();
  const PJRT_PartialCompile_Extension* partial_compile_extension =
      pjrt::FindExtension<PJRT_PartialCompile_Extension>(
          api, PJRT_Extension_Type::PJRT_Extension_Type_PartialCompile);
  ASSERT_NE(partial_compile_extension, nullptr);

  // Prepare the input programs.
  std::string phase_name = "shlo_to_hlo";
  std::vector<xla::PjRtPartialProgram> partial_programs_in =
      PreparePartialPrograms();
  std::vector<std::string> machine_attributes =
      xla::cpu::DetectMachineAttributes();
  const xla::CpuTopologyDescription topology_description(
      xla::CpuId(), xla::CpuName(), "<unknown>", /*cpu_devices=*/{},
      /*machine_attributes=*/machine_attributes);
  ASSERT_OK_AND_ASSIGN(std::unique_ptr<xla::PjRtTopologyDescription> topology,
                       xla::CreateTopology(&topology_description));

  // Run the partial compile phase.
  std::vector<std::string> phases_to_run = {"shlo_to_hlo"};
  auto partial_programs_out =
      pjrt::RunPhaseCompile(partial_compile_extension, xla::CompileOptions(),
                            *topology, partial_programs_in, phases_to_run);
  EXPECT_OK(partial_programs_out);

  // Destroy the partial programs.
  for (auto& partial_program : partial_programs_in) {
    partial_program.Destroy();
  }
  for (auto& partial_program : partial_programs_out.value()) {
    partial_program.Destroy();
  }

  // Get all the phases.
  std::vector<std::string> phase_names =
      pjrt::GetPhaseNames(partial_compile_extension).value();
  EXPECT_EQ(phase_names.size(), 1);
  EXPECT_EQ(phase_names[0], "shlo_to_hlo");
}
