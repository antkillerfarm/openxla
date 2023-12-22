/* Copyright (c) 2023 Intel Corporation

Copyright 2015 The TensorFlow Authors. All Rights Reserved.

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

#include "vsi_platform.h"

// #include "absl/base/call_once.h"
// #include "absl/base/const_init.h"
// #include "absl/memory/memory.h"
// #include "absl/strings/str_cat.h"
#include "absl/strings/str_format.h"
// #include "xla/stream_executor/sycl/sycl_driver.h"
// #include "xla/stream_executor/gpu/gpu_executor.h"
// #include "xla/stream_executor/platform/initialize.h"
// #include "tsl/platform/errors.h"
// #include "tsl/platform/status.h"

#include "vsi_executor.h"

namespace xla{
namespace vsiplugin{

VsiPlatform::VsiPlatform() {
  kVsiContext = tim::vx::Context::Create();
}

VsiPlatform::~VsiPlatform() {}

se::Platform::Id VsiPlatform::id() const { return id_; }

int VsiPlatform::VisibleDeviceCount() const { return 1; }

const std::string& VsiPlatform::Name() const { return name_; }

tsl::StatusOr<std::unique_ptr<se::DeviceDescription>>
VsiPlatform::DescriptionForDevice(int ordinal) const {
  
  // todo: open it when executor finish.
  //return VsiExecutor::CreateDeviceDescription(ordinal);
}

tsl::StatusOr<se::StreamExecutor*> VsiPlatform::ExecutorForDevice(
    int ordinal) {
  se::StreamExecutorConfig config;
  config.ordinal = ordinal;
  config.plugin_config = se::PluginConfig();
  config.device_options = se::DeviceOptions::Default();
  return GetExecutor(config);
}

tsl::StatusOr<se::StreamExecutor*>
VsiPlatform::ExecutorForDeviceWithPluginConfig(
    int ordinal, const se::PluginConfig& conplugin_configfig){
  se::StreamExecutorConfig config;
  config.ordinal = ordinal;
  config.plugin_config = se::PluginConfig();
  config.device_options = se::DeviceOptions::Default();
  return GetExecutor(config);
}

tsl::StatusOr<se::StreamExecutor*> VsiPlatform::GetExecutor(
    const se::StreamExecutorConfig& config) {
  return executor_cache_.GetOrCreate(
      config, [&]() { return GetUncachedExecutor(config); });
}

tsl::StatusOr<std::unique_ptr<se::StreamExecutor>>
VsiPlatform::GetUncachedExecutor(
    const se::StreamExecutorConfig& config) {

// LOG(FATAL) << "not yet implemented: register executor trace listener";

// TODO: open it when to finish implement of VsiExecutor
  auto executor = absl::make_unique<se::StreamExecutor>(
      this, absl::make_unique<VsiExecutor>(kVsiContext, config.ordinal, config.plugin_config),
      config.ordinal);
  auto init_status = executor->Init(config.device_options);
  if (!init_status.ok()) {
    return tsl::Status{
        absl::StatusCode::kInternal,
        absl::StrFormat(
            "failed initializing StreamExecutor for device ordinal %d: %s",
            config.ordinal, init_status.ToString())};
  }

  return std::move(executor);
}

void VsiPlatform::RegisterTraceListener(
    std::unique_ptr<se::TraceListener> listener) {
  LOG(FATAL) << "not yet implemented: register executor trace listener";
}

void VsiPlatform::UnregisterTraceListener(se::TraceListener* listener) {
  LOG(FATAL) << "not yet implemented: unregister executor trace listener";
}

static void InitializeVsiPlatform() {
 auto status = se::MultiPlatformManager::PlatformWithName("vsi-npu");
 if (!status.ok()) {

    std::unique_ptr<se::Platform> platform(new VsiPlatform);
    TF_CHECK_OK(se::MultiPlatformManager::RegisterPlatform(std::move(platform)));
 }
}
} // namespace vsiplugin
} // namespace xla

REGISTER_MODULE_INITIALIZER(
    vsi_platform,
    xla::vsiplugin::InitializeVsiPlatform());

// Note that module initialization sequencing is not supported in the
// open-source project, so this will be a no-op there.
REGISTER_MODULE_INITIALIZER_SEQUENCE(vsi_platform,
                                     multi_platform_manager);
REGISTER_MODULE_INITIALIZER_SEQUENCE(multi_platform_manager_listener,
                                      vsi_platform);
