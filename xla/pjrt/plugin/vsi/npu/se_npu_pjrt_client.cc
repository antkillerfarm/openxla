/* Copyright 2020 The TensorFlow Authors. All Rights Reserved.

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

#include "se_npu_pjrt_client.h"

#include <memory>
#include <utility>
#include <vector>

#include "absl/strings/str_cat.h"
#include "xla/client/client_library.h"
#include "xla/pjrt/pjrt_stream_executor_client.h"
#include "xla/service/platform_util.h"

namespace xla {

static const char kNpuPlatformName[] = "npu";

StreamExecutorNpuDevice::StreamExecutorNpuDevice(
    int id, std::unique_ptr<LocalDeviceState> local_device_state)
    : PjRtStreamExecutorDevice(id, std::move(local_device_state),
                               /*device_kind=*/kNpuPlatformName) {}

StatusOr<std::unique_ptr<PjRtClient>> GetStreamExecutorNpuClient() {
  TF_ASSIGN_OR_RETURN(se::Platform * platform,
                      PlatformUtil::GetPlatform("npu"));
  if (platform->VisibleDeviceCount() != 1) {
    return FailedPrecondition(
        "vsi-npu platform should have exactly one device.");
  }
  LocalClientOptions options;
  options.set_platform(platform);
  TF_ASSIGN_OR_RETURN(LocalClient * client,
                      ClientLibrary::GetOrCreateLocalClient(options));

  std::vector<std::unique_ptr<PjRtStreamExecutorDevice>> devices;
  se::StreamExecutor* executor = client->backend().stream_executor(0).value();
  auto device_state = std::make_unique<LocalDeviceState>(
      executor, client, LocalDeviceState::kSynchronous,
      /*max_inflight_computations=*/1,
      /*allow_event_reuse=*/false, /*use_callback_stream=*/false);
  auto device = std::make_unique<StreamExecutorNpuDevice>(0, std::move(device_state));
  devices.push_back(std::move(device));

  return std::unique_ptr<PjRtClient>(std::make_unique<PjRtStreamExecutorClient>(
      "npu", client, std::move(devices), /*process_index=*/0,
      /*allocator=*/nullptr, /*host_memory_allocator=*/nullptr,
      /*should_stage_host_to_device_transfers=*/false,
      /*gpu_run_options=*/nullptr));
}

}  // namespace xla
