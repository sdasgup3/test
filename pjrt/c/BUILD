load("//platforms/xla/tests:build_defs.bzl", "xla_test")
load(
    "//third_party/tensorflow/compiler/xla:xla.google.bzl",
    "xla_cc_binary",
    "xla_cc_test",
)
load(
    "//third_party/tensorflow/compiler/xla/tsl:tsl.bzl",
    "if_google",
    "if_macos",
    "internal_visibility",
)
load("//third_party/tensorflow/compiler/xla/tsl/platform:rules_cc.bzl", "cc_library")
load(
    "//third_party/tensorflow/compiler/xla/tsl/platform/google:cuda_build_defs.bzl",
    "if_cuda_is_configured",
)
load(
    "//third_party/tensorflow/third_party/rocm/google:build_defs.bzl",
    "if_rocm_is_configured",
)

package(
    default_applicable_licenses = ["//third_party/tensorflow:license"],  # copybara:comment
    default_visibility = internal_visibility(["//third_party/tensorflow/compiler/xla:internal"]),
)

cc_library(
    name = "pjrt_c_api_hdrs",
    hdrs = ["pjrt_c_api.h"],
    visibility = ["//visibility:public"],
)

cc_library(
    name = "pjrt_c_api_macros",
    hdrs = [
        "pjrt_c_api_macros.h",
    ],
    visibility = ["//visibility:public"],
)

cc_library(
    name = "pjrt_c_api_tpu_hdrs",
    hdrs = ["pjrt_c_api_tpu.h"],
    visibility = ["//visibility:public"],
    deps = [
        ":pjrt_c_api_hdrs",
    ],
)

cc_library(
    name = "pjrt_c_api_ffi_extension_hdrs",
    hdrs = ["pjrt_c_api_ffi_extension.h"],
    visibility = ["//visibility:public"],
    deps = [
        ":pjrt_c_api_hdrs",
    ],
)

cc_library(
    name = "pjrt_c_api_ffi_internal",
    srcs = ["pjrt_c_api_ffi_internal.cc"],
    hdrs = ["pjrt_c_api_ffi_internal.h"],
    visibility = ["//visibility:public"],
    deps = [
        ":pjrt_c_api_ffi_extension_hdrs",
        ":pjrt_c_api_hdrs",
        ":pjrt_c_api_helpers",
        ":pjrt_c_api_wrapper_impl",
        "//third_party/absl/status",
        "//third_party/absl/strings:string_view",
        "//third_party/tensorflow/compiler/xla/ffi:execution_context",
        "//third_party/tensorflow/compiler/xla/ffi:type_id_registry",
    ],
)

cc_library(
    name = "pjrt_c_api_phase_compile_utils",
    srcs = ["pjrt_c_api_phase_compile_utils.cc"],
    hdrs = ["pjrt_c_api_phase_compile_utils.h"],
    visibility = ["//visibility:public"],
    deps = [
        ":pjrt_c_api_phase_compile_extension_hdrs",
        ":pjrt_c_api_wrapper_impl",
        "//third_party/tensorflow/compiler/xla/pjrt/proto:pjrt_partial_program_proto_cc",
    ],
)

cc_library(
    name = "pjrt_c_api_phase_compile_extension_hdrs",
    hdrs = ["pjrt_c_api_phase_compile_extension.h"],
    visibility = ["//visibility:public"],
    deps = [":pjrt_c_api_hdrs"],
)

cc_library(
    name = "pjrt_c_api_phase_compile_internal",
    srcs = ["pjrt_c_api_phase_compile_internal.cc"],
    hdrs = ["pjrt_c_api_phase_compile_internal.h"],
    visibility = ["//visibility:public"],
    deps = [
        ":pjrt_c_api_hdrs",
        ":pjrt_c_api_helpers",
        ":pjrt_c_api_phase_compile_extension_hdrs",
        ":pjrt_c_api_phase_compile_utils",
        ":pjrt_c_api_wrapper_impl",
        "//third_party/absl/status",
        "//third_party/absl/status:statusor",
        "//third_party/absl/strings",
        "//third_party/absl/strings:string_view",
        "//third_party/tensorflow/compiler/xla/pjrt:pjrt_compiler",
        "//third_party/tensorflow/compiler/xla/pjrt:pjrt_executable",
        "//third_party/tensorflow/compiler/xla/pjrt/proto:pjrt_partial_program_proto_cc",
        "//third_party/tensorflow/compiler/xla/tsl/platform:statusor",
    ],
    alwayslink = True,
)

cc_library(
    name = "pjrt_c_api_triton_extension_hdrs",
    hdrs = ["pjrt_c_api_triton_extension.h"],
    visibility = ["//visibility:public"],
    deps = [
        ":pjrt_c_api_hdrs",
    ],
)

cc_library(
    name = "pjrt_c_api_triton_internal",
    srcs = ["pjrt_c_api_triton_internal.cc"],
    hdrs = ["pjrt_c_api_triton_internal.h"],
    visibility = ["//visibility:public"],
    deps = [
        ":pjrt_c_api_hdrs",
        ":pjrt_c_api_helpers",
        ":pjrt_c_api_triton_extension_hdrs",
        ":pjrt_c_api_wrapper_impl",
        "//third_party/absl/strings:string_view",
        "//third_party/tensorflow/compiler/xla/pjrt:triton",
    ],
)

cc_library(
    name = "pjrt_c_api_gpu_extension_hdrs",
    hdrs = ["pjrt_c_api_gpu_extension.h"],
    visibility = ["//visibility:public"],
    deps = [
        ":pjrt_c_api_hdrs",
    ],
)

cc_library(
    name = "pjrt_c_api_stream_extension_hdrs",
    hdrs = ["pjrt_c_api_stream_extension.h"],
    visibility = ["//visibility:public"],
    deps = [
        ":pjrt_c_api_hdrs",
    ],
)

cc_library(
    name = "pjrt_c_api_profiler_extension_hdrs",
    hdrs = ["pjrt_c_api_profiler_extension.h"],
    visibility = ["//visibility:public"],
    deps = [
        ":pjrt_c_api_hdrs",
        "//third_party/tensorflow/compiler/xla/backends/profiler/plugin:profiler_c_api_hdrs",
    ],
)

cc_library(
    name = "pjrt_c_api_raw_buffer_extension_hdrs",
    hdrs = ["pjrt_c_api_raw_buffer_extension.h"],
    visibility = ["//visibility:public"],
    deps = [
        ":pjrt_c_api_hdrs",
    ],
)

cc_library(
    name = "pjrt_c_api_raw_buffer_internal",
    srcs = ["pjrt_c_api_raw_buffer_internal.cc"],
    hdrs = ["pjrt_c_api_raw_buffer_internal.h"],
    visibility = ["//visibility:public"],
    deps = [
        ":pjrt_c_api_hdrs",
        ":pjrt_c_api_helpers",
        ":pjrt_c_api_raw_buffer_extension_hdrs",
        ":pjrt_c_api_wrapper_impl",
        "//third_party/absl/status",
        "//third_party/tensorflow/compiler/xla/pjrt:raw_buffer",
        "//third_party/tensorflow/compiler/xla/tsl/concurrency:ref_count",
    ],
)

cc_library(
    name = "pjrt_c_api_raw_buffer_external",
    srcs = ["pjrt_c_api_raw_buffer_external.cc"],
    hdrs = ["pjrt_c_api_raw_buffer_external.h"],
    visibility = internal_visibility(["//third_party/tensorflow/compiler/xla/pjrt:friends"]),
    deps = [
        ":pjrt_c_api_hdrs",
        ":pjrt_c_api_helpers",
        ":pjrt_c_api_raw_buffer_extension_hdrs",
        "//third_party/absl/status",
        "//third_party/absl/status:statusor",
        "//third_party/tensorflow/compiler/xla/pjrt:pjrt_c_api_client",
        "//third_party/tensorflow/compiler/xla/pjrt:pjrt_future",
        "//third_party/tensorflow/compiler/xla/pjrt:raw_buffer",
        "//third_party/tensorflow/compiler/xla/tsl/concurrency:ref_count",
        "//third_party/tensorflow/compiler/xla/tsl/platform:statusor",
        "//third_party/tensorflow/tsl/platform:casts",
    ],
)

cc_library(
    name = "pjrt_c_api_custom_partitioner_extension_hdrs",
    hdrs = ["pjrt_c_api_custom_partitioner_extension.h"],
    visibility = ["//visibility:public"],
    deps = [
        ":pjrt_c_api_hdrs",
    ],
)

cc_library(
    name = "pjrt_c_api_layouts_extension_hdrs",
    hdrs = ["pjrt_c_api_layouts_extension.h"],
    visibility = ["//visibility:public"],
    deps = [
        ":pjrt_c_api_hdrs",
    ],
)

cc_library(
    name = "pjrt_c_api_memory_descriptions_extension_hdrs",
    hdrs = ["pjrt_c_api_memory_descriptions_extension.h"],
    visibility = ["//visibility:public"],
    deps = [
        ":pjrt_c_api_hdrs",
    ],
)

cc_library(
    name = "pjrt_c_api_wrapper_impl",
    srcs = ["pjrt_c_api_wrapper_impl.cc"],
    hdrs = ["pjrt_c_api_wrapper_impl.h"],
    visibility = ["//visibility:public"],
    deps = [
        ":pjrt_c_api_hdrs",
        ":pjrt_c_api_helpers",
        ":pjrt_c_api_layouts_extension_hdrs",
        ":pjrt_c_api_memory_descriptions_extension_hdrs",
        "//third_party/absl/base:core_headers",
        "//third_party/absl/container:flat_hash_map",
        "//third_party/absl/container:flat_hash_set",
        "//third_party/absl/container:inlined_vector",
        "//third_party/absl/functional:any_invocable",
        "//third_party/absl/log",
        "//third_party/absl/log:check",
        "//third_party/absl/status",
        "//third_party/absl/status:statusor",
        "//third_party/absl/strings",
        "//third_party/absl/synchronization",
        "//third_party/absl/time",
        "//third_party/absl/types:span",
        "//third_party/llvm/llvm-project/mlir:IR",
        "//third_party/tensorflow/compiler/xla:literal",
        "//third_party/tensorflow/compiler/xla:shape_util",
        "//third_party/tensorflow/compiler/xla:util",
        "//third_party/tensorflow/compiler/xla:xla_data_proto_cc",
        "//third_party/tensorflow/compiler/xla:xla_proto_cc",
        "//third_party/tensorflow/compiler/xla/hlo/builder:xla_computation",
        "//third_party/tensorflow/compiler/xla/hlo/ir:hlo",
        "//third_party/tensorflow/compiler/xla/pjrt:mlir_to_hlo",
        "//third_party/tensorflow/compiler/xla/pjrt:pjrt_client",
        "//third_party/tensorflow/compiler/xla/pjrt:pjrt_common",
        "//third_party/tensorflow/compiler/xla/pjrt:pjrt_compiler",
        "//third_party/tensorflow/compiler/xla/pjrt:pjrt_device_description",
        "//third_party/tensorflow/compiler/xla/pjrt:pjrt_executable",
        "//third_party/tensorflow/compiler/xla/pjrt:pjrt_future",
        "//third_party/tensorflow/compiler/xla/pjrt:pjrt_layout",
        "//third_party/tensorflow/compiler/xla/pjrt/distributed:key_value_store_interface",
        "//third_party/tensorflow/compiler/xla/pjrt/proto:compile_options_proto_cc",
        "//third_party/tensorflow/compiler/xla/service:computation_placer_hdr",
        "//third_party/tensorflow/compiler/xla/service:hlo_proto_cc",
        "//third_party/tensorflow/compiler/xla/tsl/framework:allocator",
        "//third_party/tensorflow/compiler/xla/tsl/platform:errors",
        "//third_party/tensorflow/compiler/xla/tsl/platform:statusor",
        "//third_party/tensorflow/tsl/platform:casts",
        "//third_party/tensorflow/tsl/profiler/lib:connected_traceme",
        "//third_party/tensorflow/tsl/profiler/lib:context_types_hdrs",
    ],
)

cc_library(
    name = "pjrt_c_api_helpers",
    srcs = ["pjrt_c_api_helpers.cc"],
    hdrs = ["pjrt_c_api_helpers.h"],
    visibility = ["//visibility:public"],
    deps = [
        ":pjrt_c_api_hdrs",
        ":pjrt_c_api_layouts_extension_hdrs",
        ":pjrt_c_api_memory_descriptions_extension_hdrs",
        ":pjrt_c_api_profiler_extension_hdrs",
        "//third_party/absl/base:core_headers",
        "//third_party/absl/container:flat_hash_map",
        "//third_party/absl/container:inlined_vector",
        "//third_party/absl/functional:any_invocable",
        "//third_party/absl/log",
        "//third_party/absl/status",
        "//third_party/absl/status:statusor",
        "//third_party/absl/strings",
        "//third_party/absl/time",
        "//third_party/absl/types:span",
        "//third_party/stablehlo:version",
        "//third_party/tensorflow/compiler/xla:shape_util",
        "//third_party/tensorflow/compiler/xla:util",
        "//third_party/tensorflow/compiler/xla:xla_data_proto_cc",
        "//third_party/tensorflow/compiler/xla/pjrt:pjrt_client",
        "//third_party/tensorflow/compiler/xla/pjrt:pjrt_common",
        "//third_party/tensorflow/compiler/xla/pjrt:pjrt_device_description",
        "//third_party/tensorflow/compiler/xla/pjrt:pjrt_executable",
        "//third_party/tensorflow/compiler/xla/pjrt:pjrt_future",
        "//third_party/tensorflow/compiler/xla/pjrt/distributed:key_value_store_interface",
        "//third_party/tensorflow/compiler/xla/tsl/platform:errors",
        "//third_party/tensorflow/compiler/xla/tsl/platform:logging",
        "//third_party/tensorflow/compiler/xla/tsl/platform:status",
        "//third_party/tensorflow/compiler/xla/tsl/platform:statusor",
        "//third_party/tensorflow/compiler/xla/tsl/protobuf:error_codes_proto_impl_cc",
        "//third_party/tensorflow/tsl/profiler/lib:connected_traceme",
        "//third_party/tensorflow/tsl/profiler/lib:context_types_hdrs",
    ],
)

cc_library(
    name = "pjrt_c_api_cpu_internal",
    srcs = ["pjrt_c_api_cpu_internal.cc"],
    hdrs = ["pjrt_c_api_cpu_internal.h"],
    visibility = ["//visibility:public"],
    deps = [
        ":pjrt_c_api_ffi_extension_hdrs",
        ":pjrt_c_api_ffi_internal",
        ":pjrt_c_api_hdrs",
        ":pjrt_c_api_helpers",
        ":pjrt_c_api_layouts_extension_hdrs",
        ":pjrt_c_api_memory_descriptions_extension_hdrs",
        ":pjrt_c_api_wrapper_impl",
        "//third_party/absl/container:flat_hash_map",
        "//third_party/absl/log",
        "//third_party/absl/status",
        "//third_party/tensorflow/compiler/xla/pjrt:pjrt_client",
        "//third_party/tensorflow/compiler/xla/pjrt:pjrt_common",
        "//third_party/tensorflow/compiler/xla/pjrt:pjrt_executable",
        "//third_party/tensorflow/compiler/xla/pjrt/plugin/xla_cpu:cpu_client_options",
        "//third_party/tensorflow/compiler/xla/pjrt/plugin/xla_cpu:xla_cpu_pjrt_client",
    ],
)

cc_library(
    name = "pjrt_c_api_cpu",
    srcs = ["pjrt_c_api_cpu.cc"],
    hdrs = ["pjrt_c_api_cpu.h"],
    visibility = ["//visibility:public"],
    deps = [
        ":pjrt_c_api_cpu_internal",
        ":pjrt_c_api_hdrs",
    ],
)

# PJRT CPU plugin.
xla_cc_binary(
    name = "pjrt_c_api_cpu_plugin.so",
    additional_linker_inputs = [
        ":pjrt_c_api_cpu_version_script.lds",
    ],
    linkopts = if_macos(
        [
            "-Wl,-exported_symbol,_GetPjrtApi",
            "-Wl,-undefined,error",
        ],
        [
            "-Wl,--version-script,$(location :pjrt_c_api_cpu_version_script.lds)",
            "-Wl,--no-undefined",
        ],
    ),
    linkshared = True,
    tags = [
        "noasan",
        "nomsan",
        "notsan",
    ],
    visibility = ["//visibility:public"],
    deps = [
        ":pjrt_c_api_cpu",
    ],
)

cc_library(
    name = "pjrt_c_api_gpu_internal",
    srcs = ["pjrt_c_api_gpu_internal.cc"],
    hdrs = ["pjrt_c_api_gpu_internal.h"],
    local_defines = if_rocm_is_configured(["TENSORFLOW_USE_ROCM=1"]),
    visibility = ["//visibility:public"],
    deps = [
        ":pjrt_c_api_custom_partitioner_extension_hdrs",
        ":pjrt_c_api_ffi_extension_hdrs",
        ":pjrt_c_api_ffi_internal",
        ":pjrt_c_api_gpu_extension_hdrs",
        ":pjrt_c_api_hdrs",
        ":pjrt_c_api_helpers",
        ":pjrt_c_api_layouts_extension_hdrs",
        ":pjrt_c_api_memory_descriptions_extension_hdrs",
        ":pjrt_c_api_profiler_extension_hdrs",
        ":pjrt_c_api_stream_extension_hdrs",
        ":pjrt_c_api_triton_extension_hdrs",
        ":pjrt_c_api_triton_internal",
        ":pjrt_c_api_wrapper_impl",
        "//third_party/absl/container:flat_hash_map",
        "//third_party/absl/status",
        "//third_party/absl/strings:str_format",
        "//third_party/tensorflow/compiler/xla/backends/profiler:profiler_backends",  # To register the Host Tracers for GPU Plugin.
        "//third_party/tensorflow/compiler/xla/backends/profiler/gpu:device_tracer",  # To register the GPU Tracers with the GPU Plugin.
        "//third_party/tensorflow/compiler/xla/backends/profiler/plugin:plugin_tracer_impl",
        "//third_party/tensorflow/compiler/xla/backends/profiler/plugin:profiler_c_api_hdrs",
        "//third_party/tensorflow/compiler/xla/backends/profiler/plugin:profiler_error",
        "//third_party/tensorflow/compiler/xla/client:local_client",
        "//third_party/tensorflow/compiler/xla/ffi",
        "//third_party/tensorflow/compiler/xla/ffi:ffi_api",
        "//third_party/tensorflow/compiler/xla/ffi/api:c_api",
        "//third_party/tensorflow/compiler/xla/pjrt:pjrt_client",
        "//third_party/tensorflow/compiler/xla/pjrt:pjrt_common",
        "//third_party/tensorflow/compiler/xla/pjrt:pjrt_compiler",
        "//third_party/tensorflow/compiler/xla/pjrt:pjrt_device_description",
        "//third_party/tensorflow/compiler/xla/pjrt:pjrt_executable",
        "//third_party/tensorflow/compiler/xla/pjrt/gpu:gpu_helpers",
        "//third_party/tensorflow/compiler/xla/pjrt/gpu:gpu_topology",
        "//third_party/tensorflow/compiler/xla/pjrt/gpu:se_gpu_pjrt_client",
        "//third_party/tensorflow/compiler/xla/pjrt/gpu:se_gpu_pjrt_compiler",  # buildcleaner: keep to register GPU AOT compiler
        "//third_party/tensorflow/compiler/xla/python:custom_call_batch_partitioner",
        "//third_party/tensorflow/compiler/xla/python:custom_partition_callback",
        "//third_party/tensorflow/compiler/xla/python:debug_callback_partitioner",  # To register "DebugCallbackCustomCallPartitioner" custom partitioning handler.
        "//third_party/tensorflow/compiler/xla/python:inspect_sharding",  # To register "InspectSharding" custom partitioning handler.
        "//third_party/tensorflow/compiler/xla/service:compiler",
        "//third_party/tensorflow/compiler/xla/service:custom_call_target_registry",
    ],
)

cc_library(
    name = "pjrt_c_api_gpu",
    srcs = ["pjrt_c_api_gpu.cc"],
    hdrs = ["pjrt_c_api_gpu.h"],
    deps = [
        ":pjrt_c_api_gpu_internal",
        ":pjrt_c_api_hdrs",
        ":pjrt_c_api_macros",
        "//third_party/absl/base",
        "//third_party/absl/log:initialize",
        "//third_party/tensorflow/tsl/platform",
    ],
    alwayslink = 1,
)

# PJRT GPU plugin. Can be configured to be built for CUDA or ROCM.
xla_cc_binary(
    name = "pjrt_c_api_gpu_plugin.so",
    additional_linker_inputs = [
        ":pjrt_c_api_gpu_version_script.lds",
    ],
    linkopts = [
        "-Wl,--version-script,$(location :pjrt_c_api_gpu_version_script.lds)",
        "-Wl,--no-undefined",
    ],
    linkshared = True,
    tags = [
        "no_mac",  # No GPU on mac.
        "noasan",
        "nomsan",
        "notsan",
    ],
    deps = [
        ":pjrt_c_api_gpu",
        "//third_party/tensorflow/compiler/xla/service:gpu_plugin",
    ] + if_cuda_is_configured([
        "//third_party/tensorflow/compiler/xla/stream_executor:cuda_platform",
    ]) + if_rocm_is_configured([
        "//third_party/tensorflow/compiler/xla/stream_executor:rocm_platform",
    ]),
)

cc_library(
    name = "pjrt_c_api_phase_compile_plugin_internal",
    srcs = ["pjrt_c_api_phase_compile_plugin_internal.cc"],
    hdrs = ["pjrt_c_api_phase_compile_plugin_internal.h"],
    visibility = ["//visibility:public"],
    deps = [
        ":pjrt_c_api_hdrs",
        ":pjrt_c_api_layouts_extension_hdrs",
        ":pjrt_c_api_phase_compile_extension_hdrs",
        ":pjrt_c_api_phase_compile_internal",
        ":pjrt_c_api_wrapper_impl",
        "//third_party/absl/status",
        "//third_party/absl/status:statusor",
        "//third_party/absl/strings",
        "//third_party/absl/strings:str_format",
        "//third_party/llvm/llvm-project/llvm:Support",
        "//third_party/llvm/llvm-project/mlir:FuncDialect",
        "//third_party/llvm/llvm-project/mlir:IR",
        "//third_party/llvm/llvm-project/mlir:Pass",
        "//third_party/llvm/llvm-project/mlir:Transforms",
        "//third_party/stablehlo:stablehlo_passes_optimization",
        "//third_party/stablehlo:stablehlo_portable_api",
        "//third_party/stablehlo:stablehlo_serialization",
        "//third_party/tensorflow/compiler/xla/pjrt:pjrt_client",
        "//third_party/tensorflow/compiler/xla/pjrt:pjrt_compiler",
        "//third_party/tensorflow/compiler/xla/pjrt/proto:pjrt_partial_program_proto_cc",
    ],
)

cc_library(
    name = "pjrt_c_api_phase_compile_plugin",
    srcs = ["pjrt_c_api_phase_compile_plugin.cc"],
    hdrs = ["pjrt_c_api_phase_compile_plugin.h"],
    visibility = ["//visibility:public"],
    deps = [
        ":pjrt_c_api_hdrs",
        ":pjrt_c_api_phase_compile_plugin_internal",
    ],
)

xla_cc_test(
    name = "pjrt_c_api_phase_compile_plugin_test",
    srcs = ["pjrt_c_api_phase_compile_plugin_test.cc"],
    deps = [
        ":pjrt_c_api_hdrs",
        ":pjrt_c_api_helpers",
        ":pjrt_c_api_phase_compile_extension_hdrs",
        ":pjrt_c_api_phase_compile_internal",
        ":pjrt_c_api_phase_compile_plugin",
        ":pjrt_c_api_phase_compile_plugin_internal",
        "//testing/base/public:gunit_main",
        "//third_party/absl/status:statusor",
        "//third_party/absl/strings:string_view",
        "//third_party/llvm/llvm-project/mlir:IR",
        "//third_party/stablehlo:reference_api",
        "//third_party/tensorflow/compiler/xla/pjrt:pjrt_compiler",
        "//third_party/tensorflow/compiler/xla/pjrt/plugin/xla_cpu:cpu_topology_description",
        "//third_party/tensorflow/compiler/xla/pjrt/proto:pjrt_partial_program_proto_cc",
    ],
)

cc_library(
    name = "pjrt_c_api_test_base",
    testonly = 1,
    srcs = ["pjrt_c_api_test_base.cc"],
    hdrs = ["pjrt_c_api_test_base.h"],
    visibility = ["//visibility:public"],
    deps = [
        ":pjrt_c_api_hdrs",
        ":pjrt_c_api_helpers",
        ":pjrt_c_api_wrapper_impl",
        "//testing/base/public:gunit_for_library_testonly",
        "//third_party/absl/log",
        "//third_party/absl/log:check",
        "//third_party/absl/status",
        "//third_party/absl/strings:string_view",
        "//third_party/absl/types:span",
        "//third_party/tensorflow/compiler/xla:shape_util",
        "//third_party/tensorflow/compiler/xla:xla_data_proto_cc",
        "//third_party/tensorflow/compiler/xla/client:executable_build_options",
        "//third_party/tensorflow/compiler/xla/hlo/builder:xla_builder",
        "//third_party/tensorflow/compiler/xla/hlo/builder:xla_computation",
        "//third_party/tensorflow/compiler/xla/pjrt:pjrt_client",
        "//third_party/tensorflow/compiler/xla/pjrt:pjrt_executable",
        "//third_party/tensorflow/compiler/xla/pjrt:pjrt_future",
        "//third_party/tensorflow/compiler/xla/pjrt/proto:compile_options_proto_cc",
        "//third_party/tensorflow/compiler/xla/service:computation_placer_hdr",
        "//third_party/tensorflow/compiler/xla/tsl/platform:status",
    ],
)

xla_test(
    name = "pjrt_c_api_gpu_test",
    srcs = ["pjrt_c_api_gpu_test.cc"],
    backends = ["gpu"],
    tags = if_google([
        "config-cuda-only",
    ]),
    deps = [
        ":pjrt_c_api_ffi_extension_hdrs",
        ":pjrt_c_api_gpu",
        ":pjrt_c_api_gpu_extension_hdrs",
        ":pjrt_c_api_gpu_internal",
        ":pjrt_c_api_hdrs",
        ":pjrt_c_api_helpers",
        ":pjrt_c_api_test_base",
        ":pjrt_c_api_test_common",
        ":pjrt_c_api_triton_extension_hdrs",
        ":pjrt_c_api_wrapper_impl",
        "//testing/base/public:gunit_main",
        "//third_party/absl/container:flat_hash_map",
        "//third_party/absl/log:check",
        "//third_party/absl/status",
        "//third_party/absl/status:statusor",
        "//third_party/absl/strings:str_format",
        "//third_party/absl/strings:string_view",
        "//third_party/absl/types:span",
        "//third_party/tensorflow/compiler/xla:literal",
        "//third_party/tensorflow/compiler/xla:literal_util",
        "//third_party/tensorflow/compiler/xla:shape_util",
        "//third_party/tensorflow/compiler/xla:util",
        "//third_party/tensorflow/compiler/xla/client:client_library",
        "//third_party/tensorflow/compiler/xla/ffi:execution_context",
        "//third_party/tensorflow/compiler/xla/ffi:ffi_api",
        "//third_party/tensorflow/compiler/xla/ffi:type_id_registry",
        "//third_party/tensorflow/compiler/xla/ffi/api:ffi",
        "//third_party/tensorflow/compiler/xla/pjrt:pjrt_common",
        "//third_party/tensorflow/compiler/xla/pjrt:pjrt_compiler",
        "//third_party/tensorflow/compiler/xla/pjrt:pjrt_future",
        "//third_party/tensorflow/compiler/xla/pjrt/distributed:in_memory_key_value_store",
        "//third_party/tensorflow/compiler/xla/pjrt/gpu:se_gpu_pjrt_client",
        "//third_party/tensorflow/compiler/xla/service:custom_call_target_registry",
        "//third_party/tensorflow/compiler/xla/stream_executor/gpu:gpu_init",
        "//third_party/tensorflow/compiler/xla/tests:literal_test_util",
        "//third_party/tensorflow/compiler/xla/tsl/lib/core:status_test_util",
        "//third_party/tensorflow/compiler/xla/tsl/platform:status",
        "//third_party/tensorflow/compiler/xla/tsl/platform:status_matchers",
        "//third_party/tensorflow/compiler/xla/tsl/platform:statusor",
        "//third_party/tensorflow/tsl/platform:platform_port",
    ],
)

xla_cc_test(
    name = "pjrt_c_api_helpers_test",
    srcs = ["pjrt_c_api_helpers_test.cc"],
    deps = [
        ":pjrt_c_api_hdrs",
        ":pjrt_c_api_helpers",
        ":pjrt_c_api_wrapper_impl",
        "//testing/base/public:gunit_main",
        "//third_party/absl/container:flat_hash_map",
        "//third_party/absl/status",
        "//third_party/absl/status:statusor",
        "//third_party/absl/time",
        "//third_party/stablehlo:version",
        "//third_party/tensorflow/compiler/xla:shape_util",
        "//third_party/tensorflow/compiler/xla/pjrt:pjrt_common",
        "//third_party/tensorflow/compiler/xla/pjrt/distributed:in_memory_key_value_store",
        "//third_party/tensorflow/compiler/xla/tsl/lib/core:status_test_util",
        "//third_party/tensorflow/compiler/xla/tsl/platform:statusor",
    ],
)

xla_cc_test(
    name = "pjrt_c_api_cpu_test",
    srcs = ["pjrt_c_api_cpu_test.cc"],
    deps = [
        ":pjrt_c_api_cpu",
        ":pjrt_c_api_hdrs",
        ":pjrt_c_api_test_common",
        "//testing/base/public:gunit_main",
    ],
)

cc_library(
    name = "pjrt_c_api_test_common",
    testonly = 1,
    srcs = ["pjrt_c_api_test.cc"],
    hdrs = ["pjrt_c_api_test.h"],
    visibility = ["//visibility:public"],
    deps = [
        ":pjrt_c_api_hdrs",
        ":pjrt_c_api_helpers",
        ":pjrt_c_api_memory_descriptions_extension_hdrs",
        ":pjrt_c_api_test_base",
        "//testing/base/public:gunit_for_library_testonly",
        "//third_party/absl/base:core_headers",
        "//third_party/absl/container:flat_hash_map",
        "//third_party/absl/log",
        "//third_party/absl/log:check",
        "//third_party/absl/status",
        "//third_party/absl/status:statusor",
        "//third_party/absl/strings",
        "//third_party/absl/synchronization",
        "//third_party/absl/types:span",
        "//third_party/tensorflow/compiler/xla:literal",
        "//third_party/tensorflow/compiler/xla:literal_util",
        "//third_party/tensorflow/compiler/xla:shape_util",
        "//third_party/tensorflow/compiler/xla:xla_data_proto_cc",
        "//third_party/tensorflow/compiler/xla:xla_proto_cc",
        "//third_party/tensorflow/compiler/xla/client:executable_build_options",
        "//third_party/tensorflow/compiler/xla/hlo/ir:hlo",
        "//third_party/tensorflow/compiler/xla/hlo/parser:hlo_parser",
        "//third_party/tensorflow/compiler/xla/pjrt:pjrt_client",
        "//third_party/tensorflow/compiler/xla/pjrt:pjrt_device_description",
        "//third_party/tensorflow/compiler/xla/pjrt:pjrt_future",
        "//third_party/tensorflow/compiler/xla/pjrt/proto:compile_options_proto_cc",
        "//third_party/tensorflow/compiler/xla/service:computation_placer_hdr",
        "//third_party/tensorflow/compiler/xla/service:hlo_proto_cc",
        "//third_party/tensorflow/compiler/xla/tests:literal_test_util",
        "//third_party/tensorflow/compiler/xla/tsl/platform:status",
    ],
)

cc_library(
    name = "pjrt_c_api_tpu_internal",
    hdrs = ["pjrt_c_api_tpu_internal.h"],
    deps = [":pjrt_c_api_hdrs"],
)
