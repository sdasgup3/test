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

#include "third_party/tensorflow/compiler/xla/pjrt/pjrt_compiler.h"

#include <algorithm>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "third_party/absl/container/flat_hash_map.h"
#include "third_party/absl/log/log.h"
#include "third_party/absl/status/status.h"
#include "third_party/absl/strings/str_cat.h"
#include "third_party/absl/strings/string_view.h"
#include "third_party/absl/synchronization/mutex.h"
#include "third_party/llvm/llvm-project/llvm/include/llvm/Support/Casting.h"
#include "third_party/tensorflow/compiler/xla/pjrt/metrics.h"
#include "third_party/tensorflow/compiler/xla/pjrt/pjrt_partial_program.h"

namespace xla {

ABSL_CONST_INIT absl::Mutex registry_mutex(absl::kConstInit);
absl::flat_hash_map<std::string, std::unique_ptr<PjRtCompiler>>*
CompilerRegistry() {
  static auto* compiler_registry =
      new absl::flat_hash_map<std::string, std::unique_ptr<PjRtCompiler>>();
  return compiler_registry;
}

// An RAII helper for streamz metrics.
class ScopedMetricHelper {
 public:
  explicit ScopedMetricHelper(absl::string_view metric_name)
      : metric_name_(metric_name) {
    if (metric_name == metrics::kPjrtCompilerCompileComputationMetricName) {
      metrics::RecordPjrtCompilerCompileComputationStatus(true);
    } else if (metric_name == metrics::kPjrtCompilerCompileModuleMetricName) {
      metrics::RecordPjrtCompilerCompileModuleStatus(true);
    } else {
      LOG(ERROR) << "No corresponding handler function for metric: "
                 << metric_name;
    }
  }

  ~ScopedMetricHelper() {
    if (metric_name_ == metrics::kPjrtCompilerCompileComputationMetricName) {
      metrics::RecordPjrtCompilerCompileComputationStatus(false);
    } else if (metric_name_ == metrics::kPjrtCompilerCompileModuleMetricName) {
      metrics::RecordPjrtCompilerCompileModuleStatus(false);
    }
  }

 private:
  absl::string_view metric_name_;
};

void PjRtRegisterCompiler(absl::string_view platform_name,
                          std::unique_ptr<PjRtCompiler> compiler) {
  CHECK(compiler != nullptr);
  absl::MutexLock l(&registry_mutex);
  auto* compiler_registry = CompilerRegistry();
  CHECK(!compiler_registry->contains(platform_name));
  (*compiler_registry)[platform_name] = std::move(compiler);
}

absl::StatusOr<std::unique_ptr<PjRtExecutable>> PjRtCompile(
    CompileOptions options, const XlaComputation& computation,
    const PjRtTopologyDescription& topology, PjRtClient* client) {
  auto topology_compiler = topology.compiler();
  ScopedMetricHelper helper(metrics::kPjrtCompilerCompileComputationMetricName);
  if (topology_compiler.has_value()) {
    return (*topology_compiler)
        ->Compile(std::move(options), computation, topology, client);
  }
  absl::ReaderMutexLock l(&registry_mutex);
  const auto* compiler_registry = CompilerRegistry();
  auto it = compiler_registry->find(topology.platform_name());
  if (it == compiler_registry->end()) {
    return tsl::errors::NotFound(absl::StrCat(
        "No compiler registered for platform ", topology.platform_name()));
  }
  return it->second->Compile(std::move(options), computation, topology, client);
}

absl::StatusOr<std::unique_ptr<PjRtExecutable>> PjRtCompile(
    CompileOptions options, mlir::ModuleOp module,
    const PjRtTopologyDescription& topology, PjRtClient* client) {
  auto topology_compiler = topology.compiler();
  ScopedMetricHelper helper(metrics::kPjrtCompilerCompileModuleMetricName);
  if (topology_compiler.has_value()) {
    return (*topology_compiler)
        ->Compile(std::move(options), module, topology, client);
  }
  absl::ReaderMutexLock l(&registry_mutex);
  const auto* compiler_registry = CompilerRegistry();
  auto it = compiler_registry->find(topology.platform_name());
  if (it == compiler_registry->end()) {
    return absl::NotFoundError(absl::StrCat(
        "No compiler registered for platform ", topology.platform_name()));
  }
  return it->second->Compile(std::move(options), module, topology, client);
}

absl::StatusOr<std::vector<PjRtPartialProgram>> PjRtPartialCompile(
    CompileOptions options,
    const std::vector<PjRtPartialProgram>& input_programs,
    const PjRtTopologyDescription& topology,
    const std::vector<std::string>& phases_to_run) {
  absl::ReaderMutexLock l(&registry_mutex);
  const auto* compiler_registry = CompilerRegistry();
  auto it = compiler_registry->find("partial_compile");
  if (it == compiler_registry->end()) {
    return absl::NotFoundError("No compiler registered for phase compilation ");
  }
  PjRtPhaseCompiler* compiler =
      static_cast<PjRtPhaseCompiler*>(it->second.get());
  if (!compiler) {
    return absl::InvalidArgumentError(
        "Compiler is not a PjRtCompiler. This is a bug.");
  }
  return compiler->CompilePhase(std::move(options), input_programs,
                                phases_to_run);
}

absl::StatusOr<std::vector<std::string>> PjRtGetPhaseNames() {
  absl::ReaderMutexLock l(&registry_mutex);
  const auto* compiler_registry = CompilerRegistry();
  auto it = compiler_registry->find("partial_compile");
  if (it == compiler_registry->end()) {
    return tsl::errors::NotFound(
        absl::StrCat("No compiler registered for phase compilation "));
  }
  PjRtPhaseCompiler* compiler =
      static_cast<PjRtPhaseCompiler*>(it->second.get());
  if (!compiler) {
    return absl::InvalidArgumentError(
        "Compiler is not a PjRtCompiler. This is a bug.");
  }
  return compiler->GetPhaseNames();
}

}  // namespace xla
