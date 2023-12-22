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

#ifndef XLA_STREAM_EXECUTOR_VSI_EXECUTOR_H_
#define XLA_STREAM_EXECUTOR_VSI_EXECUTOR_H_

#include <unordered_map>
#include <mutex>

// #include "tensorflow/stream_executor/stream_executor.h" 
// #include "tensorflow/stream_executor/stream_executor_internal.h"
// #include "tensorflow/stream_executor/host/host_stream.h"
// #include "tensorflow/stream_executor/event.h"

// #include "tensorflow/compiler/plugin/vsi/driver/vsi_utils.h"
#include "tsl/platform/statusor.h"
#include "xla/stream_executor/stream_executor.h"
#include "xla/stream_executor/host/host_stream.h"
#include "tim/vx/context.h"
#include "tim/vx/graph.h"

namespace se = stream_executor;

namespace xla{
namespace vsiplugin{

extern const int invalid_index;

class VsiExecutor : public se::internal::StreamExecutorInterface {
public:
    explicit VsiExecutor(std::shared_ptr<tim::vx::Context>vsiCtx, const int device_ordinal, se::PluginConfig pluginConfig);
    ~VsiExecutor();

    // std::shared_ptr<tim::vx::Graph> getGraph(int ordinal = 0) {
    //     // if(kVsiGraphContainer.find(ordinal) != kVsiGraphContainer.end())
    //     //     return kVsiGraphContainer[ordinal];
    //     return nullptr;
    // }
    std::shared_ptr<tim::vx::Context> getContext() { return kVsiContext; }

    // std::shared_ptr<tim::vx::Tensor> getTensor(int index) {
    //     // if(index > kVsiTensorContainer.size()){
    //     //     return nullptr;
    //     // }
    //     // return kVsiTensorContainer[index];
    //     return nullptr;
    // }
    // int getTensorIndex(tim::vx::Tensor *t){
    //     // auto it = std::find_if(kVsiTensorContainer.begin(), kVsiTensorContainer.end(),
    //     // [&](std::shared_ptr<tim::vx::Tensor> p){
    //     //     return p.get() == t;
    //     // });
    //     // if(it != kVsiTensorContainer.end()){
    //     //     return it - kVsiTensorContainer.begin();
    //     // }
    //     return invalid_index;
    // }

    // int setTensor(std::shared_ptr<tim::vx::Tensor> t){
    //     // std::unique_lock<std::mutex> lock(mutex_);
    //     // kVsiTensorContainer.push_back(t);
    //     // return kVsiTensorContainer.size() - 1;
    //     return 0;
    // }

    se::internal::StreamExecutorInterface *GetUnderlyingExecutor() override { return this; }
    tsl::Status Init(int device_ordinal,
                            se::DeviceOptions device_options) override {
        // if(kVsiGraphContainer.find(device_ordinal) == kVsiGraphContainer.end()){
        //     kVsiGraphContainer[device_ordinal] = kVsiContext->CreateGraph();
        // }
        return ::tsl::OkStatus();
    }

    tsl::Status GetKernel(const se::MultiKernelLoaderSpec &spec,
                                 se::KernelBase *kernel) {
        return tsl::errors::Unimplemented("Not Implemented");
    }

    tsl::Status LoadModule(const se::MultiModuleLoaderSpec &spec,
                            se::ModuleHandle *module_handle) {
        return tsl::errors::Unimplemented("Not Implemented");
    }

    tsl::Status Launch(se::Stream *stream, const se::ThreadDim &thread_dims,
                                const se::BlockDim &block_dims, const se::KernelBase &k,
                                const se::KernelArgsArrayBase &args) {
        return tsl::errors::Unimplemented("Not Implemented");
    }

    se::DeviceMemoryBase Allocate(uint64_t size, int64_t memory_space) override;
    void *GetSubBuffer(se::DeviceMemoryBase *parent, uint64_t offset, uint64_t size) override;
    void Deallocate(se::DeviceMemoryBase *mem) override;

    // Allocates unified memory space of the given size, if supported.
    // graphcore didnot support this interface
    void *UnifiedMemoryAllocate(uint64_t size) { return nullptr; }
 
    // Deallocates unified memory space previously allocated with
    // UnifiedMemoryAllocate.
    void UnifiedMemoryDeallocate(void *mem) {}

    void *HostMemoryAllocate(uint64_t size) override;

    void HostMemoryDeallocate(void *mem) override;

    bool HostMemoryRegister(void *mem, uint64_t size) override;

    bool HostMemoryUnregister(void *mem) override;
    bool SynchronizeAllActivity() override;

    tsl::Status SynchronousMemZero(se::DeviceMemoryBase *location,
                                    uint64_t size) override;
    tsl::Status SynchronousMemSet(se::DeviceMemoryBase *location, int value,
                                   uint64_t size) override;
    tsl::Status SynchronousMemcpy(se::DeviceMemoryBase *gpu_dst,
                                   const void *host_src, uint64_t size) override;
    tsl::Status SynchronousMemcpy(void *host_dst,
                                   const se::DeviceMemoryBase &gpu_src,
                                   uint64_t size) override;
    tsl::Status SynchronousMemcpyDeviceToDevice(
        se::DeviceMemoryBase *gpu_dst, const se::DeviceMemoryBase &gpu_src,
        uint64_t size) override;
    tsl::Status MemZero(se::Stream *stream, se::DeviceMemoryBase *location,
                         uint64_t size) override;
    tsl::Status Memset(se::Stream *stream, se::DeviceMemoryBase *location,
                        uint8_t pattern, uint64_t size) {
      return tsl::errors::Unimplemented("Not implemented");
    }

    tsl::Status Memset32(se::Stream *stream, se::DeviceMemoryBase *location,
                          uint32_t pattern, uint64_t size) override;

    bool Memcpy(se::Stream *stream, void *host_dst,
                const se::DeviceMemoryBase &gpu_src, uint64_t size) override;
    bool Memcpy(se::Stream *stream, se::DeviceMemoryBase *gpu_dst,
                const void *host_src, uint64_t size) override;
    bool MemcpyDeviceToDevice(se::Stream *stream, se::DeviceMemoryBase *gpu_dst,
                              const se::DeviceMemoryBase &gpu_src,
                              uint64_t size) override;
    se::host::HostStream* AsVsiStream(se::Stream* stream);
    bool HostCallback(se::Stream *stream,
                    absl::AnyInvocable<tsl::Status() &&> callback) override;
    tsl::Status AllocateEvent(se::Event *event) override;
    tsl::Status DeallocateEvent(se::Event *event) override;
    tsl::Status RecordEvent(se::Stream *stream, se::Event *event) override;
    tsl::Status WaitForEvent(se::Stream *stream, se::Event *event) override;

    se::Event::Status PollForEventStatus(se::Event *event) override;

    bool AllocateStream(se::Stream *stream) override;
    void DeallocateStream(se::Stream *stream) override;
    bool CreateStreamDependency(se::Stream *dependent, se::Stream *other) override;

    tsl::Status BlockHostUntilDone(se::Stream *stream) override;
    tsl::Status GetStatus(se::Stream *stream) {
        return tsl::Status(absl::StatusCode::kUnimplemented,
                        "GetStatus is not supported on this executor.");
    }
    int PlatformDeviceCount() override;
    tsl::Status EnablePeerAccessTo(StreamExecutorInterface *other) override;
    bool CanEnablePeerAccessTo(StreamExecutorInterface *other) override;

    // Creates a new DeviceDescription object. Ownership is transferred to the
    // caller.
    tsl::StatusOr<std::unique_ptr<se::DeviceDescription>>
    CreateDeviceDescription() const override;

    // Each call creates a new instance of the platform-specific implementation of
    // the corresponding interface type.
    std::unique_ptr<se::internal::EventInterface> CreateEventImplementation() override;
    std::unique_ptr<se::internal::KernelInterface> CreateKernelImplementation() override;
    std::unique_ptr<se::internal::StreamInterface> GetStreamImplementation() override;

    // Return allocator statistics.
    absl::optional<se::AllocatorStats> GetAllocatorStats() {
        return absl::nullopt;
    }

private:
    std::mutex mutex_;
    int ordinal_;
    se::PluginConfig plugConfig_;
    std::shared_ptr<tim::vx::Context> kVsiContext;
    SE_DISALLOW_COPY_AND_ASSIGN(VsiExecutor);
};

} // namespace vsiplugin
} // namespace xla
#endif