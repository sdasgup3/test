/* Copyright 2024 The OpenXLA Authors.

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

#ifndef TENSORFLOW_COMPILER_XLA_PJRT_C_PJRT_C_API_PHASE_COMPILE_INTERNAL_H_
#define TENSORFLOW_COMPILER_XLA_PJRT_C_PJRT_C_API_PHASE_COMPILE_INTERNAL_H_

#include <string>
#include <vector>

#include "third_party/absl/status/statusor.h"
#include "third_party/tensorflow/compiler/xla/pjrt/c/pjrt_c_api.h"
#include "third_party/tensorflow/compiler/xla/pjrt/c/pjrt_c_api_phase_compile_extension.h"
#include "third_party/tensorflow/compiler/xla/pjrt/pjrt_compiler.h"
#include "third_party/tensorflow/compiler/xla/pjrt/pjrt_executable.h"
#include "third_party/tensorflow/compiler/xla/pjrt/proto/pjrt_partial_program.proto.h"

namespace pjrt {

// Returns the names of all the phases following the order of the
// their registration.
absl::StatusOr<std::vector<std::string>> GetPhaseNames(
    const PJRT_PhaseCompile_Extension* phase_compile_extension);

// Runs the compilation phase with the given phases 'phases_to_run' on the
// input programs 'partial_programs_in' and returns the output programs.
absl::StatusOr<std::vector<xla::PjRtPartialProgramProto>> RunPhases(
    const PJRT_PhaseCompile_Extension* phase_compile_extension,
    xla::CompileOptions options, const xla::PjRtTopologyDescription& topology,
    const std::vector<xla::PjRtPartialProgramProto>& partial_programs_in,
    const std::vector<std::string>& phases_to_run);

// Creates and initializes a PJRT_PhaseCompile_Extension struct.
PJRT_PhaseCompile_Extension CreatePhaseCompileExtension(
    PJRT_Extension_Base* next);

}  // namespace pjrt

#endif  // TENSORFLOW_COMPILER_XLA_PJRT_C_PJRT_C_API_PHASE_COMPILE_INTERNAL_H_
