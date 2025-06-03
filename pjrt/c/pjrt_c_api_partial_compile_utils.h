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

#ifndef TENSORFLOW_COMPILER_XLA_PJRT_C_PJRT_C_API_PARTIAL_COMPILE_UTILS_H_
#define TENSORFLOW_COMPILER_XLA_PJRT_C_PJRT_C_API_PARTIAL_COMPILE_UTILS_H_

#include <cstddef>
#include <memory>
#include <string>
#include <vector>

#include "third_party/absl/status/status.h"
#include "third_party/absl/status/statusor.h"
#include "third_party/tensorflow/compiler/xla/pjrt/c/pjrt_c_api_wrapper_impl.h"
#include "third_party/tensorflow/compiler/xla/pjrt/pjrt_partial_program.h"

void printSerializedModule(const xla::PjRtPartialProgram& program,
                           const std::string& message);

std::vector<std::string> ConvertCharBufferToCppStrings(const char** char_buffer,
                                                       size_t num_strings);
void ConvertCppStringsToCharBuffer(const std::vector<std::string>& strings,
                                   const char*** char_buffer,
                                   size_t* num_strings);

absl::StatusOr<std::vector<xla::PjRtPartialProgram>>
ConvertCPartialProgramsToCppPartialPrograms(const PJRT_PartialProgram* programs,
                                            size_t num_programs);

absl::Status ConvertCppPartialProgramsToCPartialPrograms(
    const std::vector<xla::PjRtPartialProgram>& programs,
    PJRT_PartialProgram** programs_out, size_t* num_programs_out);

#endif  // TENSORFLOW_COMPILER_XLA_PJRT_C_PJRT_C_API_PARTIAL_COMPILE_UTILS_H_
