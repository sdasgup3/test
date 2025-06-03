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

#ifndef TENSORFLOW_COMPILER_XLA_PJRT_C_PJRT_C_API_PHASE_COMPILE_UTILS_H_
#define TENSORFLOW_COMPILER_XLA_PJRT_C_PJRT_C_API_PHASE_COMPILE_UTILS_H_

#include <cstddef>
#include <string>
#include <vector>

std::vector<std::string> ConvertCharBufferToCppStrings(
    const char** char_buffer, const size_t* char_buffer_sizes,
    size_t num_strings);
void ConvertCppStringsToCharBuffer(const std::vector<std::string>& strings,
                                   const char*** char_buffer,
                                   const size_t** char_buffer_sizes,
                                   size_t* num_strings,
                                   bool is_null_terminated = true);


#endif  // TENSORFLOW_COMPILER_XLA_PJRT_C_PJRT_C_API_PHASE_COMPILE_UTILS_H_
