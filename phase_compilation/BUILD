load("//third_party/bazel_rules/rules_cc/cc:cc_library.bzl", "cc_library")
load("//third_party/tensorflow/compiler/xla:xla.google.bzl", "xla_cc_test")

package(
    default_applicable_licenses = ["//third_party/tensorflow:license"],  # copybara:comment
    default_visibility = ["//visibility:public"],
    licenses = ["notice"],
)

cc_library(
    name = "portable_typedef",
    srcs = [
        "portable_typedef.h",
    ],
    hdrs = ["portable_typedef.h"],
    deps = [
        "//third_party/absl/status",
        "//third_party/absl/status:statusor",
        "//third_party/absl/strings",
        "//third_party/llvm/llvm-project/llvm:Support",
        "//third_party/llvm/llvm-project/mlir:IR",
        "//third_party/stablehlo:stablehlo_portable_api",
        "//third_party/stablehlo:stablehlo_serialization",
        "//third_party/tensorflow/compiler/xla:debug_options_flags",
        "//third_party/tensorflow/compiler/xla/hlo/ir:hlo",
    ],
    alwayslink = 1,
)

cc_library(
    name = "plugin",
    srcs = [
        "plugin.cc",
    ],
    hdrs = ["plugin.h"],
    deps = [
        ":portable_typedef",
        "//third_party/absl/container:flat_hash_map",
        "//third_party/absl/status",
        "//third_party/absl/status:statusor",
        "//third_party/absl/strings",
        "//third_party/llvm/llvm-project/mlir:IR",
        "//third_party/tensorflow/compiler/xla/hlo/ir:hlo",
        "//third_party/tensorflow/compiler/xla/hlo/translate:stablehlo",
        "//third_party/tensorflow/compiler/xla/pjrt:pjrt_client",
        "//third_party/tensorflow/compiler/xla/pjrt:pjrt_compiler",
        "//third_party/tensorflow/compiler/xla/pjrt:pjrt_partial_program",
        "//third_party/tensorflow/compiler/xla/pjrt/c:pjrt_c_api_hdrs",
        "//third_party/tensorflow/compiler/xla/pjrt/c:pjrt_c_api_layouts_extension_hdrs",
        "//third_party/tensorflow/compiler/xla/pjrt/c:pjrt_c_api_partial_compile_extension_hdrs",
        "//third_party/tensorflow/compiler/xla/pjrt/c:pjrt_c_api_partial_compile_internal",
        "//third_party/tensorflow/compiler/xla/pjrt/c:pjrt_c_api_wrapper_impl",
    ],
    alwayslink = 1,
)

xla_cc_test(
    name = "client",
    srcs = ["client.cc"],
    deps = [
        ":plugin",
        ":portable_typedef",
        "//learning/brain/research/pjrt:pjrt_topology_utils",
        "//testing/base/public:gunit",
        "//third_party/absl/status:statusor",
        "//third_party/absl/strings:string_view",
        "//third_party/llvm/llvm-project/mlir:IR",
        "//third_party/stablehlo:reference_api",
        "//third_party/tensorflow/compiler/xla:xla_proto_cc",
        "//third_party/tensorflow/compiler/xla/backends/cpu/codegen:cpu_features",
        "//third_party/tensorflow/compiler/xla/pjrt:pjrt_compiler",
        "//third_party/tensorflow/compiler/xla/pjrt:pjrt_executable",
        "//third_party/tensorflow/compiler/xla/pjrt:pjrt_partial_program",
        "//third_party/tensorflow/compiler/xla/pjrt/c:pjrt_c_api_hdrs",
        "//third_party/tensorflow/compiler/xla/pjrt/c:pjrt_c_api_helpers",
        "//third_party/tensorflow/compiler/xla/pjrt/c:pjrt_c_api_partial_compile_extension_hdrs",
        "//third_party/tensorflow/compiler/xla/pjrt/c:pjrt_c_api_partial_compile_internal",
        "//third_party/tensorflow/compiler/xla/pjrt/c:pjrt_c_api_partial_compile_utils",
        "//third_party/tensorflow/compiler/xla/pjrt/plugin/xla_cpu:cpu_topology_description",
        "//third_party/tensorflow/compiler/xla/tests:xla_internal_test_main",
    ],
)
