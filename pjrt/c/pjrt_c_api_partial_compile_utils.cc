/* Copyright 2025 The OpenXLA Authors.

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

#include "third_party/tensorflow/compiler/xla/pjrt/c/pjrt_c_api_partial_compile_utils.h"

#include <cstddef>
#include <cstring>
#include <iostream>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "third_party/absl/status/status.h"
#include "third_party/absl/status/statusor.h"
#include "third_party/absl/strings/str_cat.h"
#include "third_party/absl/strings/string_view.h"
#include "third_party/llvm/llvm-project/llvm/include/llvm/Support/LogicalResult.h"
#include "third_party/llvm/llvm-project/llvm/include/llvm/Support/raw_ostream.h"
#include "third_party/llvm/llvm-project/mlir/include/mlir/IR/BuiltinOps.h"
#include "third_party/llvm/llvm-project/mlir/include/mlir/IR/MLIRContext.h"
#include "third_party/llvm/llvm-project/mlir/include/mlir/IR/OwningOpRef.h"
#include "third_party/stablehlo/stablehlo/api/PortableApi.h"
#include "third_party/stablehlo/stablehlo/dialect/Serialization.h"
#include "third_party/tensorflow/compiler/xla/debug_options_flags.h"
#include "third_party/tensorflow/compiler/xla/hlo/ir/hlo_module.h"
#include "third_party/tensorflow/compiler/xla/pjrt/c/pjrt_c_api_partial_compile_extension.h"
#include "third_party/tensorflow/compiler/xla/pjrt/pjrt_partial_program.h"

namespace {
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
}  // namespace

void printSerializedModule(const xla::PjRtPartialProgram& program,
                           const std::string& message) {
  std::cout << message << "\n";
  std::cout << "Raw program: " << size_t(program.GetProgramBuffer()) << "\n";
  std::cout << "Raw program size: " << program.GetProgramBufferSize() << "\n";
  mlir::MLIRContext context;
  auto module_op = SHLOType::deserialize(program.GetProgram(), context);
  module_op.value()->dump();
}

std::vector<std::string> ConvertCharBufferToCppStrings(const char** char_buffer,
                                                       size_t num_strings) {
  std::vector<std::string> cpp_strings;
  cpp_strings.reserve(num_strings);
  for (size_t i = 0; i < num_strings; ++i) {
    cpp_strings.push_back(std::string(char_buffer[i]));
  }

  // Destroy the char buffers.
  for (size_t i = 0; i < num_strings; ++i) {
    delete[] char_buffer[i];
  }
  delete[] char_buffer;

  return cpp_strings;
}

void ConvertCppStringsToCharBuffer(const std::vector<std::string>& strings,
                                   const char*** char_buffer,
                                   size_t* num_strings) {
  *num_strings = strings.size();
  const char** buffer_of_pointers = new const char*[*num_strings];
  for (size_t i = 0; i < *num_strings; ++i) {
    char* single_string_buffer = new char[strings[i].size() + 1];
    memcpy(single_string_buffer, strings[i].c_str(), strings[i].size() + 1);
    buffer_of_pointers[i] = single_string_buffer;
  }
  *char_buffer = buffer_of_pointers;
}

absl::StatusOr<std::vector<xla::PjRtPartialProgram>>
ConvertCPartialProgramsToCppPartialPrograms(const PJRT_PartialProgram* programs,
                                            size_t num_programs) {
  // If there are no programs, return an empty vector.
  if (num_programs == 0) {
    return std::vector<xla::PjRtPartialProgram>();
  }

  // Validate the inputs.
  if (programs == nullptr) {
    return absl::InvalidArgumentError("Input 'programs_in is null.");
  }

  std::vector<xla::PjRtPartialProgram> programs_out;
  programs_out.reserve(num_programs);
  for (size_t i = 0; i < num_programs; ++i) {
    xla::PjRtPartialProgram cpp_program;
    cpp_program.SetProgramBuffer(programs[i].program);  // Zero copy
    cpp_program.SetProgramBufferSize(programs[i].program_size);
    cpp_program.SetGeneratingPhase(std::string(programs[i].generating_phase));
    cpp_program.SetNextPhases(ConvertCharBufferToCppStrings(
        programs[i].next_phases, programs[i].num_next_phases));
    cpp_program.SetVersion(std::string(programs[i].version));
    cpp_program.SetFormat(programs[i].format);
    programs_out.push_back(std::move(cpp_program));

    delete[] programs[i].generating_phase;
    delete[] programs[i].version;
  }

  delete[] programs;
  return programs_out;
}

absl::Status ConvertCppPartialProgramsToCPartialPrograms(
    const std::vector<xla::PjRtPartialProgram>& programs,
    PJRT_PartialProgram** programs_out, size_t* num_programs_out) {
  *programs_out = nullptr;
  *num_programs_out = 0;

  size_t num_programs = programs.size();
  *num_programs_out = num_programs;
  *programs_out = new PJRT_PartialProgram[num_programs];

  if (num_programs == 0) {
    return absl::OkStatus();
  }

  for (size_t i = 0; i < num_programs; ++i) {
    PJRT_PartialProgram& program_out = (*programs_out)[i];

    // copy 'program' pointer
    program_out.program = programs[i].GetProgramBuffer();  // Zero copy
    program_out.program_size = programs[i].GetProgramBufferSize();

    // Deep copy 'generating_phase'
    const std::string& generating_phase_str = programs[i].GetGeneratingPhase();
    program_out.generating_phase = new char[generating_phase_str.size() + 1];
    memcpy(program_out.generating_phase, generating_phase_str.c_str(),
           generating_phase_str.size() + 1);

    // Deep copy 'next_phases'
    ConvertCppStringsToCharBuffer(programs[i].GetNextPhases(),
                                  &(program_out.next_phases),
                                  &(program_out.num_next_phases));

    // Deep copy 'version'
    const std::string& version_str = programs[i].GetVersion();
    program_out.version = new char[version_str.size() + 1];
    memcpy(program_out.version, version_str.c_str(), version_str.size() + 1);

    // Assign 'format' directly as it's a value type
    program_out.format = programs[i].GetFormat();
  }

  return absl::OkStatus();
}
