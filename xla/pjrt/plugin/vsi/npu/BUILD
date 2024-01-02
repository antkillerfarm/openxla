load("//xla:xla.bzl", "xla_cc_test")

cc_library(
    name = "pjrt_c_api_npu_internal",
    srcs = [
        "pjrt_c_api_npu_internal.cc",
        "se_npu_pjrt_client.cc",
        "vsi_platform_id.cc",
        "vsi_platform.cc",
        "vsi_executor.cc",
        "vsi_compiler.cc",
        "vsi_executable.cc",
        "vsi_transfer_manager.cc",
        "visitors/visitor_base.cc",
        ],
    hdrs = [
        "pjrt_c_api_npu_internal.h",
        "se_npu_pjrt_client.h",
        "vsi_platform_id.h",
        "vsi_platform.h",
        "vsi_executor.h",
        "vsi_compiler.h",
        "vsi_executable.h",
        "vsi_transfer_manager.h",
        "vsi_utils.h",
        "visitors/visitor_base.h",
        ],
    visibility = ["//visibility:public"],
    deps = [
        "//xla/client:client_library",
        "//xla/pjrt:pjrt_stream_executor_client",
        "//xla/pjrt/c:pjrt_c_api_wrapper_impl",
        "//xla/service:compiler",
        "//xla/service:hlo_execution_profile",
        "//xla/stream_executor/host:host_stream",
        "@TIM_VX//:tim-vx_interface",
    ],
    linkstatic = True,
    alwayslink = True,
)

cc_library(
    name = "npu_lib",
    srcs = ["pjrt_c_api_npu.cc"],
    hdrs = ["pjrt_c_api_npu.h"],
    linkstatic = True,
    deps = [
        "//xla/pjrt/c:pjrt_c_api_hdrs",
        ":pjrt_c_api_npu_internal",
    ],
    alwayslink = True,
)

cc_binary(
    name = "libnpu.so",
    linkshared = True,
    visibility = ["//visibility:public"],
    deps = [
        ":npu_lib",
    ],
)

xla_cc_test(
    name = "pjrt_c_api_npu_test",
    srcs = ["pjrt_c_api_npu_test.cc"],
    deps = [
        ":npu_lib",
        "//xla/pjrt/c:pjrt_c_api_test_common",
        "//xla/pjrt/c:pjrt_c_api_wrapper_impl",
        "@com_google_googletest//:gtest_main",
    ],
)
