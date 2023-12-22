/* Copyright (c) 2023 Intel Corporation

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

// #include <cstdint>
// #include <memory>
// #include <optional>
// #include <set>
// #include <string>
// #include <utility>
// #include <vector>

// #include "absl/container/flat_hash_map.h"
// #include "absl/status/status.h"
// #include "absl/strings/str_format.h"
// #include "xla/backends/profiler/plugin/plugin_tracer_impl.h"
// #include "xla/backends/profiler/plugin/profiler_c_api.h"
// #include "xla/backends/profiler/plugin/profiler_error.h"
// #include "tsl/platform/errors.h"
// #include "xla/pjrt/c/pjrt_c_api.h"
// #include "xla/pjrt/c/pjrt_c_api_gpu_extension.h"
// #include "xla/pjrt/c/pjrt_c_api_helpers.h"
// #include "xla/pjrt/c/pjrt_c_api_profiler_extension.h"
// #include "xla/pjrt/gpu/gpu_helpers.h"
// #include "xla/pjrt/pjrt_c_api_wrapper_impl.h"
// #include "xla/pjrt/pjrt_client.h"
// #include "xla/pjrt/pjrt_common.h"
// #include "xla/pjrt/se_xpu_pjrt_client.h"
// #include "xla/service/custom_call_target_registry.h"

#include "xla/pjrt/c/pjrt_c_api_wrapper_impl.h"
#include "pjrt_c_api_npu_internal.h"
#include "se_npu_pjrt_client.h"


namespace pjrt {
namespace npu_plugin {

PJRT_Error* PJRT_Client_Create(PJRT_Client_Create_Args* args) {
  PJRT_ASSIGN_OR_RETURN(std::unique_ptr<xla::PjRtClient> client,
                        xla::GetStreamExecutorNpuClient());
  args->client = pjrt::CreateWrapperClient(std::move(client));
  return nullptr;
}

PJRT_Error* PJRT_NpuDeviceTopology_Create(
    PJRT_TopologyDescription_Create_Args* args) {
  return new PJRT_Error{tsl::errors::Unimplemented(
      "Topology not supported for NPU compilation.")};
}

constexpr PJRT_Api pjrt_api = pjrt::CreatePjrtApi(
    pjrt::npu_plugin::PJRT_Client_Create,
    pjrt::npu_plugin::PJRT_NpuDeviceTopology_Create,
    pjrt::PJRT_Plugin_Initialize_NoOp);

const PJRT_Api* GetNpuPjrtApi() { return &pjrt_api; }

}  // namespace npu_plugin
}  // namespace pjrt
