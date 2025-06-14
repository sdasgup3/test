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

#ifndef TENSORFLOW_COMPILER_XLA_PJRT_PJRT_COMPILER_H_
#define TENSORFLOW_COMPILER_XLA_PJRT_PJRT_COMPILER_H_

#include <algorithm>
#include <cstdint>
#include <functional>
#include <memory>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include "third_party/absl/container/flat_hash_map.h"
#include "third_party/absl/status/status.h"
#include "third_party/absl/status/statusor.h"
#include "third_party/absl/strings/str_cat.h"
#include "third_party/absl/strings/string_view.h"
#include "third_party/absl/types/span.h"
#include "third_party/llvm/llvm-project/mlir/include/mlir/IR/BuiltinOps.h"
#include "third_party/tensorflow/compiler/xla/hlo/builder/xla_computation.h"
#include "third_party/tensorflow/compiler/xla/layout.h"
#include "third_party/tensorflow/compiler/xla/pjrt/pjrt_device_description.h"
#include "third_party/tensorflow/compiler/xla/pjrt/pjrt_executable.h"
#include "third_party/tensorflow/compiler/xla/pjrt/proto/pjrt_partial_program.proto.h"
#include "third_party/tensorflow/tsl/platform/fingerprint.h"

namespace xla {

using PjRtPlatformId = uint64_t;

inline const char* CpuName() {
  static constexpr char kCpuName[] = "cpu";
  return kCpuName;
}
inline const char* CudaName() {
  static constexpr char kCudaName[] = "cuda";
  return kCudaName;
}
inline const char* RocmName() {
  static constexpr char kRocmName[] = "rocm";
  return kRocmName;
}
inline const char* SyclName() {
  static constexpr char kSyclName[] = "sycl";
  return kSyclName;
}
inline const char* TpuName() {
  static constexpr char kTpuName[] = "tpu";
  return kTpuName;
}
inline PjRtPlatformId CpuId() {
  static const PjRtPlatformId kCpuId = tsl::Fingerprint64(CpuName());
  return kCpuId;
}
inline PjRtPlatformId CudaId() {
  static const PjRtPlatformId kCudaId = tsl::Fingerprint64(CudaName());
  return kCudaId;
}
inline PjRtPlatformId RocmId() {
  static const PjRtPlatformId kRocmId = tsl::Fingerprint64(RocmName());
  return kRocmId;
}
inline PjRtPlatformId SyclId() {
  static const PjRtPlatformId kSyclId = tsl::Fingerprint64(SyclName());
  return kSyclId;
}
inline PjRtPlatformId TpuId() {
  static const PjRtPlatformId kTpuId = tsl::Fingerprint64(TpuName());
  return kTpuId;
}

class PjRtCompiler;
class PjRtClient;

// Abstract interface to represent device topology that is used by the compiler.
class PjRtTopologyDescription {
 public:
  virtual ~PjRtTopologyDescription() = default;

  // Return an ID that identifies the platform (CPU/GPU/TPU).
  virtual PjRtPlatformId platform_id() const = 0;

  // Returns a string that identifies the platform (CPU/GPU/TPU).
  virtual absl::string_view platform_name() const = 0;

  // Returns a string containing human-readable, platform-specific version info
  // (e.g. the CUDA version on GPU or libtpu version on Cloud TPU).
  virtual absl::string_view platform_version() const = 0;

  // If non-null, overrides the compiler for this topology.
  virtual std::optional<PjRtCompiler*> compiler() const { return std::nullopt; }

  // Returns an unordered list of descriptions for all devices in this topology.
  virtual std::vector<std::unique_ptr<const PjRtDeviceDescription>>
  DeviceDescriptions() const = 0;

  // Returns true if the topology represents subslice.
  virtual bool is_subslice_topology() const { return false; }

  // Returns the number of processes (usually the number of hosts, except in
  // topologies with multiple processes per host).
  virtual absl::StatusOr<int> ProcessCount() const {
    return absl::UnimplementedError("ProcessCount is unsupported.");
  }

  // Returns the total number of cores of the default type.
  virtual absl::StatusOr<int> CoreCountOfDefaultType() const {
    return absl::UnimplementedError("CoreCountOfDefaultType is unsupported.");
  }

  // Returns the total number of logical devices of the default type.
  virtual absl::StatusOr<int> LogicalDeviceCountOfDefaultType() const {
    return absl::UnimplementedError(
        "LogicalDeviceCountOfDefaultType is unsupported.");
  }

  // Returns the number of cores of the default type per process.
  virtual absl::StatusOr<int> CoreCountOfDefaultTypePerProcess() const {
    return absl::UnimplementedError(
        "CoreCountOfDefaultTypePerProcess is unsupported.");
  }

  // Returns the number of cores per chip for the default type.
  virtual absl::StatusOr<int> CoreCountOfDefaultTypePerChip() const {
    return absl::UnimplementedError(
        "CoreCountOfDefaultTypePerChip is unsupported.");
  }

  // Serializes the topology for use in cache keys. (No guarantees on
  // stability).
  virtual absl::StatusOr<std::string> Serialize() const = 0;

  // Returns vendor specific attributes about the topology.
  virtual const absl::flat_hash_map<std::string, PjRtDeviceAttribute>&
  Attributes() const = 0;

  // Returns the default device layout for a buffer with `element_type` and
  // `dims`. The default layout is a platform-specific layout used when no other
  // layout is specified, e.g. for host-to-device transfers. When compiling, the
  // default layout is used for program arguments and outputs unless
  // user-specified or compiler-chosen layouts are requested via the
  // "mhlo.layout_mode" attribute.
  virtual absl::StatusOr<Layout> GetDefaultLayout(
      PrimitiveType element_type, absl::Span<const int64_t> dims) const = 0;
};

// Abstract interface that all registered compilers must implement.
class PjRtCompiler {
 public:
  virtual ~PjRtCompiler() = default;

  // Compiles the 'computation' and returns a 'PjRtExecutable'. The returned
  // PjRtExecutable must be loaded by a compatible client before execution.
  virtual absl::StatusOr<std::unique_ptr<PjRtExecutable>> Compile(
      CompileOptions options, const XlaComputation& computation,
      const PjRtTopologyDescription& topology, PjRtClient* client) {
    return absl::UnimplementedError("Compile HloComputation is unsupported.");
  }

  // Variant of `Compile` that accepts an MLIR module.
  virtual absl::StatusOr<std::unique_ptr<PjRtExecutable>> Compile(
      CompileOptions options, mlir::ModuleOp module,
      const PjRtTopologyDescription& topology, PjRtClient* client) {
    return absl::UnimplementedError("Compile mlir::ModuleOp is unsupported.");
  }
};

typedef std::function<absl::StatusOr<std::vector<PjRtPartialProgramProto>>(
    CompileOptions, const std::vector<PjRtPartialProgramProto>&)>
    PhaseCompiler;
typedef std::function<absl::Status(const std::vector<PjRtPartialProgramProto>&)>
    PhaseValidator;

class PjRtPhaseCompiler : public PjRtCompiler {
 public:
  ~PjRtPhaseCompiler() override = default;
  virtual absl::Status RegisterPhase(const std::string& phase_name,
                                     PhaseCompiler phase_compiler,
                                     PhaseValidator phase_validator);
  virtual absl::StatusOr<xla::PhaseCompiler> GetPhaseCompiler(
      const std::string& phase_name);
  virtual absl::StatusOr<xla::PhaseValidator> GetPhaseValidator(
      const std::string& phase_name);
  virtual absl::StatusOr<std::vector<std::string>> GetPhaseNames();

  absl::StatusOr<std::vector<PjRtPartialProgramProto>> CompilePhase(
      CompileOptions options,
      const std::vector<PjRtPartialProgramProto>& input_programs,
      const std::vector<std::string>& phases_to_run);

 protected:
  absl::Status ValidatePhase(
      const std::vector<PjRtPartialProgramProto>& input_programs,
      const std::string& phase_to_run);

 private:
  absl::flat_hash_map<std::string, PhaseCompiler> phase_compiler_map_;
  absl::flat_hash_map<std::string, PhaseValidator> phase_validator_map_;
  std::vector<std::string> phase_names_;
};

// Registers a compiler to compile programs for 'platform_name'.
// Takes ownership of 'compiler'.
//
// REQUIRES: No compiler has been registered for the platform yet.
void PjRtRegisterCompiler(absl::string_view platform_name,
                          std::unique_ptr<PjRtCompiler> compiler);

// Compiles a 'computation' and generates a 'PjRtExecutable' using the compiler
// registered for the platform using PjRtRegisterCompiler. The returned
// PjRtExecutable must be loaded by a compatible client before execution.
//
// The actual compiler used may be overridden by Topology::compiler().
//
// Returns error::NotFound if a compiler has not been registered for the
// platform. Forwards errors returned from the registered compiler in case of a
// compilation failure.
absl::StatusOr<std::unique_ptr<PjRtExecutable>> PjRtCompile(
    CompileOptions options, const XlaComputation& computation,
    const PjRtTopologyDescription& topology, PjRtClient* client = nullptr);

// Variant of `PjRtCompile` that accepts an MLIR module.
absl::StatusOr<std::unique_ptr<PjRtExecutable>> PjRtCompile(
    CompileOptions options, mlir::ModuleOp module,
    const PjRtTopologyDescription& topology, PjRtClient* client = nullptr);

// Variant of `PjRtCompile` that accepts a vector of `PjRtPartialProgramProto`
// and compiles them using the compiler registered for the platform using
// PjRtRegisterCompiler for a given phase.
absl::StatusOr<std::vector<PjRtPartialProgramProto>> PjRtPhaseCompile(
    CompileOptions options,
    const std::vector<PjRtPartialProgramProto>& input_programs,
    const PjRtTopologyDescription& topology,
    const std::vector<std::string>& phases_to_run);

absl::StatusOr<std::vector<std::string>> PjRtGetPhaseNames();

}  // namespace xla

#endif  // TENSORFLOW_COMPILER_XLA_PJRT_PJRT_COMPILER_H_
