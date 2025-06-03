/* Copyright 2022 The OpenXLA Authors.

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
#include "third_party/tensorflow/compiler/xla/pjrt/c/pjrt_c_api_phase_compile_plugin.h"

#include <cstdlib>
#include <string>
#include <vector>

#include "testing/base/public/gmock.h"
#include "testing/base/public/gunit.h"
#include "third_party/absl/status/statusor.h"
#include "third_party/absl/strings/string_view.h"
#include "third_party/llvm/llvm-project/mlir/include/mlir/IR/BuiltinOps.h"
#include "third_party/llvm/llvm-project/mlir/include/mlir/IR/MLIRContext.h"
#include "third_party/llvm/llvm-project/mlir/include/mlir/IR/OwningOpRef.h"
#include "third_party/stablehlo/stablehlo/reference/Api.h"
#include "third_party/tensorflow/compiler/xla/pjrt/c/pjrt_c_api.h"
#include "third_party/tensorflow/compiler/xla/pjrt/c/pjrt_c_api_helpers.h"
#include "third_party/tensorflow/compiler/xla/pjrt/c/pjrt_c_api_phase_compile_extension.h"
#include "third_party/tensorflow/compiler/xla/pjrt/c/pjrt_c_api_phase_compile_internal.h"
#include "third_party/tensorflow/compiler/xla/pjrt/c/pjrt_c_api_phase_compile_plugin_internal.h"
#include "third_party/tensorflow/compiler/xla/pjrt/pjrt_compiler.h"
#include "third_party/tensorflow/compiler/xla/pjrt/plugin/xla_cpu/cpu_topology_description.h"
#include "third_party/tensorflow/compiler/xla/pjrt/proto/pjrt_partial_program.proto.h"

namespace pjrt {
namespace {

constexpr absl::string_view kStablehloModuleStr = R"(
  module {
    func.func @main(%arg0: tensor<4xi32>) -> tensor<4xi32> {
      %0 = stablehlo.constant dense<0> : tensor<4xi32>
      %1 = stablehlo.add %arg0, %0 : tensor<4xi32>
      func.return %1 : tensor<4xi32>
    }
  }
  )";

constexpr absl::string_view kPhaseName = "stablehlo_to_optimized_stablehlo";

std::vector<xla::PjRtPartialProgramProto> PrepareInputPartialPrograms() {
  std::string program_code{kStablehloModuleStr};

  mlir::MLIRContext context;
  mlir::OwningOpRef<mlir::ModuleOp> stablehlo_module =
      mlir::stablehlo::parseStablehloModule(program_code, context).value();

  auto bytecode_status =
      pjrt::phase_compile_cpu_plugin::StableHLOTypeSerialization::serialize(
          stablehlo_module);
  if (!bytecode_status.ok()) {
    exit(1);
  }

  xla::PjRtPartialProgramProto partial_program;
  partial_program.set_program(bytecode_status.value());
  partial_program.set_program_format(
      0);  // 0 expresses StableHLO bytecode per plugin
  partial_program.set_generating_phase("n/a");
  partial_program.add_next_phases({"stablehlo_to_optimized_stablehlo"});
  partial_program.set_version("1.0");
  return {partial_program};
}

const PJRT_PhaseCompile_Extension* GetPhaseCompileExtension(
    const PJRT_Api* api) {
  return pjrt::FindExtension<PJRT_PhaseCompile_Extension>(
      api, PJRT_Extension_Type::PJRT_Extension_Type_PhaseCompile);
}

}  // namespace

class PhaseCompileTest : public ::testing::Test {
 protected:
  static const PJRT_PhaseCompile_Extension* phase_compile_extension_;
  static xla::PjRtTopologyDescription* topology_description_;

  static void SetUpTestSuite() {
    phase_compile_extension_ = GetPhaseCompileExtension(GetPjrtApi());

    std::vector<std::string> machine_attributes;
    machine_attributes.push_back("abc");
    xla::CpuTopologyDescription* cpu_topology_description_impl =
        new xla::CpuTopologyDescription(xla::CpuId(), xla::CpuName(),
                                        "<unknown>",
                                        /*cpu_devices=*/{}, machine_attributes);
    topology_description_ = cpu_topology_description_impl;
  }

  static void TearDownTestSuite() { delete topology_description_; }
};

const PJRT_PhaseCompile_Extension* PhaseCompileTest::phase_compile_extension_ =
    nullptr;
xla::PjRtTopologyDescription* PhaseCompileTest::topology_description_ = nullptr;

TEST_F(PhaseCompileTest, LookupExtension) {
  ASSERT_NE(phase_compile_extension_, nullptr);
}

TEST_F(PhaseCompileTest, RunPhases) {
  // Prepare the input programs.
  std::vector<xla::PjRtPartialProgramProto> partial_programs_in =
      PrepareInputPartialPrograms();

  // Run the partial compile phase.
  std::vector<std::string> phases_to_run = {std::string(kPhaseName)};
  auto partial_programs_out = pjrt::RunPhases(
      phase_compile_extension_, xla::CompileOptions(), *topology_description_,
      partial_programs_in, phases_to_run);
  EXPECT_OK(partial_programs_out);

  // Print the output programs.
  for (auto& partial_program : partial_programs_out.value()) {
    mlir::MLIRContext context;
    auto deserialized_module =
        phase_compile_cpu_plugin::StableHLOTypeSerialization::deserialize(
            partial_program.program(), context);
    EXPECT_OK(deserialized_module);
    deserialized_module.value()->dump();
  }
}

// Running a phase with empty input programs should fail.
TEST_F(PhaseCompileTest, ConsumeEmptyPrograms) {
  // Prepare the input programs.
  std::vector<xla::PjRtPartialProgramProto> partial_programs_in = {};

  // Run the partial compile phase.
  std::vector<std::string> phases_to_run = {std::string(kPhaseName)};
  auto partial_programs_out = pjrt::RunPhases(
      phase_compile_extension_, xla::CompileOptions(), *topology_description_,
      partial_programs_in, phases_to_run);
  EXPECT_FALSE(partial_programs_out.ok());
  EXPECT_EQ(partial_programs_out.status().message(),
            "Input partial programs cannot be empty");
}

// Running a phase with empty "phases to run" should fail.
TEST_F(PhaseCompileTest, ConsumeEmptyPhases) {
  // Prepare the input programs.
  std::vector<xla::PjRtPartialProgramProto> partial_programs_in =
      PrepareInputPartialPrograms();

  // Run the partial compile phase.
  std::vector<std::string> phases_to_run = {};
  auto partial_programs_out = pjrt::RunPhases(
      phase_compile_extension_, xla::CompileOptions(), *topology_description_,
      partial_programs_in, phases_to_run);
  EXPECT_FALSE(partial_programs_out.ok());
  EXPECT_EQ(partial_programs_out.status().message(),
            "Phases to run cannot be empty");
}

// Running a phase with empty phase name should fail.
TEST_F(PhaseCompileTest, ConsumeEmptyPhaseName) {
  // Prepare the input programs.
  std::vector<xla::PjRtPartialProgramProto> partial_programs_in =
      PrepareInputPartialPrograms();

  // Run the partial compile phase.
  std::vector<std::string> phases_to_run = {""};
  auto partial_programs_out = pjrt::RunPhases(
      phase_compile_extension_, xla::CompileOptions(), *topology_description_,
      partial_programs_in, phases_to_run);
  EXPECT_FALSE(partial_programs_out.ok());
  EXPECT_EQ(partial_programs_out.status().message(),
            "Phase name cannot be empty");
}

// Running a phase which is not expected to be run on the input programs should
// fail.
TEST_F(PhaseCompileTest, ConsumeProgramWithIncompatiblePhase) {
  // Prepare the input programs.
  std::vector<xla::PjRtPartialProgramProto> partial_programs_in =
      PrepareInputPartialPrograms();

  // Run the partial compile phase.
  std::vector<std::string> phases_to_run = {"IllegalPhaseName"};
  auto partial_programs_out = pjrt::RunPhases(
      phase_compile_extension_, xla::CompileOptions(), *topology_description_,
      partial_programs_in, phases_to_run);
  EXPECT_FALSE(partial_programs_out.ok());
  EXPECT_EQ(partial_programs_out.status().message(),
            "Input partial program cannot be consumed by a phase with name "
            "IllegalPhaseName");
}

TEST_F(PhaseCompileTest, GetPhaseNames) {
  std::vector<std::string> phase_names =
      pjrt::GetPhaseNames(phase_compile_extension_).value();
  EXPECT_EQ(phase_names.size(), 1) << "Failure: Incorrect number of phases.";
  EXPECT_EQ(phase_names[0], kPhaseName);
}

}  // namespace pjrt
