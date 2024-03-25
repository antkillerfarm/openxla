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
