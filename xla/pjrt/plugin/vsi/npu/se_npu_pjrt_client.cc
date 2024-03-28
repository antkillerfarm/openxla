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
#include "tsl/platform/stacktrace.h"
#include "xla/client/client_library.h"
#include "xla/pjrt/pjrt_stream_executor_client.h"
#include "xla/service/platform_util.h"

namespace xla {

static const char kNpuPlatformName[] = "npu";

StreamExecutorNpuDevice::StreamExecutorNpuDevice(
    int id, std::unique_ptr<LocalDeviceState> local_device_state)
    : PjRtStreamExecutorDevice(id, std::move(local_device_state),
                               /*device_kind=*/kNpuPlatformName) {}

// Builds a LocalDeviceState for each NPU present.
StatusOr<std::map<int, std::unique_ptr<LocalDeviceState>>>
BuildLocalDeviceStates(LocalClient* xla_client) {
  std::map<int, std::unique_ptr<LocalDeviceState>> addressable_devices;
  for (se::StreamExecutor* executor :
       xla_client->backend().stream_executors()) {
    addressable_devices.emplace(
        executor->device_ordinal(),
        std::make_unique<LocalDeviceState>(
            executor, xla_client, LocalDeviceState::kSynchronous,
            /*max_inflight_computations=*/1,
            /*allow_event_reuse=*/false, /*use_callback_stream=*/false));
  }
  return std::move(addressable_devices);
}

std::vector<std::unique_ptr<PjRtStreamExecutorDevice>> BuildLocalDevices(
    std::map<int, std::unique_ptr<LocalDeviceState>> local_device_states) {
  std::vector<std::unique_ptr<PjRtStreamExecutorDevice>> devices;
  for (auto& ordinal_and_device : local_device_states) {
    auto device = std::make_unique<StreamExecutorNpuDevice>(
        ordinal_and_device.first, std::move(ordinal_and_device.second));
    devices.push_back(std::move(device));
  }
  return devices;
}

StatusOr<std::unique_ptr<PjRtClient>> GetStreamExecutorNpuClient() {
  TF_ASSIGN_OR_RETURN(se::Platform * platform,
                      PlatformUtil::GetPlatform("npu"));
  LOG(INFO) << "FTT GetStreamExecutorNpuClient 0: "
            << platform->VisibleDeviceCount();
  //   LOG(INFO) << tsl::CurrentStackTrace();
  //   if (platform->VisibleDeviceCount() != 1) {
  //     return FailedPrecondition(
  //         "vsi-npu platform should have exactly one device.");
  //   }
  LocalClientOptions options;
  options.set_platform(platform);
  TF_ASSIGN_OR_RETURN(LocalClient * client,
                      ClientLibrary::GetOrCreateLocalClient(options));
  LOG(INFO) << "FTT GetStreamExecutorNpuClient 1: "
            << client->backend().stream_executors().size();

  std::map<int, std::unique_ptr<LocalDeviceState>> local_device_states;
  TF_ASSIGN_OR_RETURN(local_device_states, BuildLocalDeviceStates(client));
  std::vector<std::unique_ptr<PjRtStreamExecutorDevice>> devices;
  devices = BuildLocalDevices(std::move(local_device_states));

  return std::unique_ptr<PjRtClient>(std::make_unique<PjRtStreamExecutorClient>(
      "npu", client, std::move(devices), /*process_index=*/0,
      /*allocator=*/nullptr, /*host_memory_allocator=*/nullptr,
      /*should_stage_host_to_device_transfers=*/false,
      /*gpu_run_options=*/nullptr));
}

}  // namespace xla
