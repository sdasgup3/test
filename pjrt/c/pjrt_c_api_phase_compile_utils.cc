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

#include "third_party/tensorflow/compiler/xla/pjrt/c/pjrt_c_api_phase_compile_utils.h"

#include <cassert>
#include <cstddef>
#include <cstring>
#include <string>
#include <vector>

std::vector<std::string> ConvertCharBufferToCppStrings(
    const char** char_buffers, const size_t* char_buffer_sizes,
    size_t num_strings) {
  assert(char_buffers != nullptr);

  std::vector<std::string> cpp_strings;
  cpp_strings.reserve(num_strings);
  for (size_t i = 0; i < num_strings; ++i) {
    size_t char_buffer_size = char_buffer_sizes != nullptr
                                  ? char_buffer_sizes[i]
                                  : strlen(char_buffers[i]);
    cpp_strings.push_back(std::string(char_buffers[i], char_buffer_size));
  }

  // Destroy the char buffers.
  for (size_t i = 0; i < num_strings; ++i) {
    delete[] char_buffers[i];
  }
  delete[] char_buffer_sizes;
  delete[] char_buffers;

  return cpp_strings;
}

void ConvertCppStringsToCharBuffer(const std::vector<std::string>& strings,
                                   const char*** char_buffers,
                                   const size_t** char_buffer_sizes,
                                   size_t* num_strings,
                                   bool is_null_terminated) {
  *num_strings = strings.size();
  const char** buffer_pointers = new const char*[*num_strings];
  size_t* buffer_sizes = nullptr;
  if (!is_null_terminated) {
    buffer_sizes = new size_t[*num_strings];
  }

  for (size_t i = 0; i < *num_strings; ++i) {
    size_t string_size =
        is_null_terminated ? strings[i].size() + 1 : strings[i].size();
    auto data_pointer =
        is_null_terminated ? strings[i].c_str() : strings[i].data();

    char* string_buffer = new char[string_size];
    memcpy(string_buffer, data_pointer, string_size);
    buffer_pointers[i] = string_buffer;
    if (!is_null_terminated) {
      buffer_sizes[i] = string_size;
    }
  }
  *char_buffers = buffer_pointers;
  if (!is_null_terminated) {
    *char_buffer_sizes = buffer_sizes;
  }
}
