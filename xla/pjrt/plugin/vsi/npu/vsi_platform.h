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

#ifndef XLA_STREAM_EXECUTOR_VSI_PLATFORM_H_
#define XLA_STREAM_EXECUTOR_VSI_PLATFORM_H_

#include <memory>
#include <vector>

#include "absl/base/thread_annotations.h"
#include "xla/stream_executor/executor_cache.h"
#include "xla/stream_executor/multi_platform_manager.h"
#include "xla/stream_executor/platform.h"
#include "xla/stream_executor/platform/port.h"
#include "xla/stream_executor/stream_executor.h"
#include "xla/stream_executor/stream_executor_internal.h"
#include "xla/stream_executor/trace_listener.h"
#include "tsl/platform/statusor.h"

#include "tim/vx/context.h"
#include "vsi_platform_id.h"

namespace xla{
namespace vsiplugin{

const char* const DEVICE_XLA_VSI_NPU = "NPU";
const char* const DEVICE_VSI_NPU_XLA_JIT = "XLA_NPU_JIT";
const char* const PLATFORM_NAME = "npu";

namespace se = stream_executor;

class VsiPlatform : public se::Platform{
 public:
  VsiPlatform();
  ~VsiPlatform() override;

  Platform::Id id() const override;

  int VisibleDeviceCount() const override;

  const std::string& Name() const override;

  std::shared_ptr<tim::vx::Context> getContext() { return vsi_context_;}
  
  tsl::StatusOr<std::unique_ptr<se::DeviceDescription>> DescriptionForDevice(
      int ordinal) const override;

  tsl::StatusOr<se::StreamExecutor*> ExecutorForDevice(int ordinal) override;

  tsl::StatusOr<se::StreamExecutor*> GetExecutor(
      const se::StreamExecutorConfig& config) override;

  tsl::StatusOr<std::unique_ptr<se::StreamExecutor>> GetUncachedExecutor(
      const se::StreamExecutorConfig& config) override;

 private:
  // This platform's name.
  std::string name_ = "npu";
  // This platform's id.
  Platform::Id id_ = vsi_platform_id;

  // Cache of created StreamExecutors.
  se::ExecutorCache executor_cache_;

  std::shared_ptr<tim::vx::Context> vsi_context_;

  SE_DISALLOW_COPY_AND_ASSIGN(VsiPlatform);
};

} // namespace vsiplugin
} // namespace xla


#endif  // XLA_STREAM_EXECUTOR_VSI_PLATFORM_H_
