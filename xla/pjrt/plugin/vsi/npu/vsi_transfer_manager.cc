/* Copyright 2017 The TensorFlow Authors. All Rights Reserved.

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

#include "vsi_transfer_manager.h"
#include "vsi_platform_id.h"

namespace xla {
namespace vsiplugin {

VsiTransferManager::VsiTransferManager(se::Platform::Id id,
                                       unsigned pointer_size)
    : GenericTransferManager(id, pointer_size) {}

Status VsiTransferManager::TransferLiteralToInfeed(
    se::StreamExecutor* executor, const LiteralSlice& literal) {
  LOG(FATAL) << "Not Implemented";
  return ::tsl::OkStatus();
}
Status VsiTransferManager::TransferLiteralFromOutfeed(
    se::StreamExecutor* executor, MutableBorrowingLiteral literal) {
  LOG(FATAL) << "Not Implemented";
  return ::tsl::OkStatus();
}
Status VsiTransferManager::ResetDevices(
    absl::Span<se::StreamExecutor* const> executors) {
  LOG(FATAL) << "Not Implemented";
  return ::tsl::OkStatus();
}

}  // namespace vsiplugin
}  // namespace xla

static std::unique_ptr<xla::TransferManager>
CreateInterpreterTransferManager() {
  return absl::make_unique<xla::vsiplugin::VsiTransferManager>(
      xla::vsiplugin::vsi_platform_id, 8);
}

static bool InitModule() {
  xla::TransferManager::RegisterTransferManager(
      xla::vsiplugin::vsi_platform_id, &CreateInterpreterTransferManager);
  return true;
}

static bool module_initialized = InitModule();