load("//third_party/bazel_rules/rules_python/python:proto.bzl", "py_proto_library")  # copybara:comment
load(
    "//third_party/tensorflow/compiler/xla/tsl/platform:build_config.bzl",
    "tf_proto_library",
)

package(
    default_applicable_licenses = ["//third_party/tensorflow:license"],  # copybara:comment
    licenses = ["notice"],
)

tf_proto_library(
    name = "compile_options_proto",
    srcs = ["compile_options.proto"],
    protodeps = [
        "//third_party/tensorflow/compiler/xla:xla_data_proto",
        "//third_party/tensorflow/compiler/xla:xla_proto",
        "//third_party/tensorflow/compiler/xla/stream_executor:device_description_proto",
    ],
    visibility = ["//visibility:public"],
)

# copybara:comment_begin(google-only)
py_proto_library(
    name = "compile_options_py_pb2",
    visibility = ["//visibility:public"],
    deps = [":compile_options_proto"],
)
# copybara:comment_end

tf_proto_library(
    name = "execute_options_proto",
    srcs = ["execute_options.proto"],
    visibility = ["//visibility:public"],
)

tf_proto_library(
    name = "executable_metadata_proto",
    srcs = ["executable_metadata.proto"],
    visibility = ["//visibility:public"],
)

tf_proto_library(
    name = "pjrt_value_type_proto",
    srcs = ["pjrt_value_type.proto"],
    visibility = ["//visibility:public"],
)

tf_proto_library(
    name = "pjrt_partial_program_proto",
    srcs = ["pjrt_partial_program.proto"],
    visibility = ["//visibility:public"],
)
