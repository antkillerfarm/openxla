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

#include "vsi_executable.h"

#include <algorithm>
#include <cstring>
#include <string>
#include <utility>
#include <vector>

#include "tsl/platform/env.h"
#include "tsl/platform/errors.h"
#include "tsl/profiler/lib/scoped_annotation.h"
#include "tsl/profiler/lib/traceme.h"
#include "vsi_transfer_manager.h"
#include "xla/service/transfer_manager.h"

namespace xla {
namespace vsiplugin {

// Check that instruction output is of type - constant,
// and no compute node exist in graph.
bool IsWithoutComputeGraph(const HloInstruction* root_inst) {
  if (root_inst->opcode() == HloOpcode::kConstant ||
      root_inst->opcode() == HloOpcode::kParameter) {
    return true;
  }
  if (root_inst->opcode() != HloOpcode::kTuple &&
      root_inst->opcode() != HloOpcode::kGetTupleElement) {
    return false;
  }
  for (auto* inst : root_inst->operands()) {
    if (!IsWithoutComputeGraph(inst)) {
      return false;
    }
  }
  return true;
}

Status TransformDirectlyOutput(const HloInstruction* root_inst, 
                               std::vector<Literal>& result) {
  if (root_inst->opcode() == HloOpcode::kConstant){
    result.emplace_back(root_inst->literal().Clone());
    return ::tsl::OkStatus();
  }
  for (auto* inst : root_inst->operands()) {
    if (inst->opcode() == HloOpcode::kTuple) {
      TF_RETURN_IF_ERROR(TransformDirectlyOutput(inst, result));
    }
    else if (inst->opcode() == HloOpcode::kConstant) {
      result.emplace_back(inst->literal().Clone());
    }
    else {
      return tsl::errors::FailedPrecondition(
          "The graph contained compute node: ", inst->ToString());
    }
  }
  return ::tsl::OkStatus();
}

StatusOr<ExecutionOutput> VsiExecutable::HandleWithoutComputeGraph(
    HloInstruction* root_instr, ScopedShapedBuffer& result_buffers) {
  const Shape& result_shape = root_instr->shape();
  std::vector<Literal> directly_outputs;
  TF_RETURN_IF_ERROR(TransformDirectlyOutput(root_instr, directly_outputs));
  if (!result_shape.IsTuple()) {
    const auto& pair = result_buffers.buffers().begin();
    const ShapeIndex& index = pair->first;
    se::DeviceMemoryBase& memory_base = pair->second;
    const Shape& subshape =
        ShapeUtil::GetSubshape(result_buffers.on_device_shape(), index);
    LOG(INFO) << "no-tuple  result buffer info " << subshape.ToString();
    const auto& constant = directly_outputs.at(0);
    std::memcpy(memory_base.opaque(), constant.untyped_data(),
                constant.size_bytes());
  } else {
    LOG(WARNING) << "Process without compute graph,"
                 << "multi outputs code not validate yet!";
    se::DeviceMemoryBase& top_memory_base =
        result_buffers.buffers().begin()->second;
    LOG(INFO) << "top_memory_base location is " << top_memory_base.opaque();
    int32_t count = 0;
    for (auto& pair : result_buffers.buffers()) {
      if (count == 0) {
        // skip first buffer
        count++;
        continue;
      }
      const ShapeIndex& index = pair.first;
      se::DeviceMemoryBase& memory_base = pair.second;
      const Shape& subshape =
          ShapeUtil::GetSubshape(result_buffers.on_device_shape(), index);
      LOG(INFO) << "tuple result buffer info " << subshape.ToString();

      const auto& constant = root_instr->operand(count - 1)->literal();
      std::memcpy(memory_base.opaque(), constant.untyped_data(),
                  constant.size_bytes());

      *(size_t*)(top_memory_base.opaque() + sizeof(void*) * (count - 1)) =
          (size_t)memory_base.opaque();
      LOG(INFO) << "sub tensor " << count
                << " mem is: " << memory_base.opaque();
      count++;
    }
  }
  ExecutionOutput result(std::move(result_buffers));
  LOG(INFO) << "Leave WCG:" << module().name() << " :: " << (void*)this
            << " :: " << tsl::Env::Default()->GetCurrentThreadId();
  return result;
}

VsiExecutable::VsiExecutable(std::shared_ptr<HloModule> hlo_module,
                             VsiExecutor* executor)
    : Executable(hlo_module,
                 /*hlo_profile_printer_data=*/nullptr,
                 /*hlo_profile_index_map=*/nullptr),
      visitor_(std::move(std::make_unique<BaseVisitor>(executor))),
      executor_(executor) {
  LOG(INFO) << "XXTT VsiExecutable::VsiExecutable";
  visitor_->ResetVisitStates();
}

VsiExecutable::~VsiExecutable() {}

tsl::mutex vsi_executable_mtx;
StatusOr<ExecutionOutput> VsiExecutable::ExecuteAsyncOnStream(
    const ServiceExecutableRunOptions* run_options,
    std::vector<ExecutionInput> arguments,
    HloExecutionProfile* hlo_execution_profile) {
  tsl::mutex_lock l(vsi_executable_mtx);
  LOG(INFO) << "ExecuteAsyncOnStream " << module().name()
            << " :: " << (void*)this
            << " :: " << tsl::Env::Default()->GetCurrentThreadId();
  LOG(INFO) << module().ToString();

  se::Stream* stream = run_options->stream();
  se::StreamExecutor* executor = stream->parent();
  const se::Platform* platform = executor->platform();

  // Convert the ShapeTree to a ShapedBuffer. We do this so we can call
  // TransferManager methods below.
  std::vector<ShapedBuffer> argument_buffers;
  argument_buffers.reserve(arguments.size());
  int device_ordinal = run_options->device_ordinal();
  if (device_ordinal < 0) {
    device_ordinal = 0;
  }
  for (auto& argument : arguments) {
    const ShapeTree<MaybeOwningDeviceMemory>& buffers = argument.Buffers();
    argument_buffers.push_back(
        ShapedBuffer(buffers.shape(),
                     /*device_ordinal=*/executor->device_ordinal()));
    auto in_it = buffers.begin();
    auto out_it = argument_buffers.back().buffers().begin();
    for (; in_it != buffers.end(); ++in_it, ++out_it) {
      out_it->second = in_it->second.AsDeviceMemoryBase();
    }
  }

  VLOG(1) << "Execute " << module().name();
  if (VLOG_IS_ON(2)) {
    for (const auto& a : argument_buffers) {
      VLOG(2) << "-- argument " << a;
    }
  }

  const HloComputation* computation = module().entry_computation();
  if (computation->num_parameters() != arguments.size()) {
    return tsl::errors::Internal(
        "Mismatch between argument count and graph parameter count.");
  }

  TF_ASSIGN_OR_RETURN(TransferManager * transfer_manager,
                      TransferManager::GetForPlatform(platform));
  // Transform the ShapedBuffer arguments into literals which the evaluator
  // consumes.
  std::vector<Literal> arg_literals;
  for (int64_t p = 0; p < computation->num_parameters(); ++p) {
    TF_ASSIGN_OR_RETURN(Literal arg_literal,
                        transfer_manager->TransferLiteralFromDevice(
                            run_options->stream(), argument_buffers[p]));
    arg_literals.push_back(std::move(arg_literal));
  }

  // Transform the result literal back into a ShapedBuffer.
  auto root_instr = computation->root_instruction();
  const Shape& result_shape = root_instr->shape();
  LOG(INFO) << "computation->num_parameters: " << computation->num_parameters();

  LOG(INFO) << "ExecuteAsyncOnStream NNN 0: ";
  TF_ASSIGN_OR_RETURN(
      ScopedShapedBuffer result_buffers,
      transfer_manager->AllocateScopedShapedBuffer(
          result_shape, run_options->allocator(), executor->device_ordinal()));

  bool is_without_compute_graph = IsWithoutComputeGraph(root_instr);
  LOG(INFO) << "is_without_compute_graph: " << is_without_compute_graph;

  if (is_without_compute_graph) {
    return HandleWithoutComputeGraph(root_instr, result_buffers);
  }

  auto tensor = visitor_->Evaluate(*computation, arg_literals).value();

  if (!result_shape.IsTuple()) {
    for (auto& pair : result_buffers.buffers()) {
      const ShapeIndex& index = pair.first;
      se::DeviceMemoryBase& memory_base = pair.second;
      const Shape& subshape =
          ShapeUtil::GetSubshape(result_buffers.on_device_shape(), index);
      LOG(INFO) << "no-tuple  result buffer info " << subshape.ToString();
      tensor[0]->CopyDataFromTensor(memory_base.opaque());
      float* val = (float*)(memory_base.opaque());
      LOG(INFO) << "memory_base.opaque: " << *val;
    }
  } else {
    int32_t count = 0;
    auto top_shape_memory = result_buffers.buffers();

    se::DeviceMemoryBase top_memory_base;
    for (auto& pair : result_buffers.buffers()) {
      if (count == 0) {
        top_memory_base = pair.second;
        LOG(INFO) << "top_memory_base location is " << top_memory_base.opaque();
        // count++;
        break;
      }
    }

    count = -1;
    for (auto& pair : result_buffers.buffers()) {
      count++;
      if (count == 0) continue;
      const ShapeIndex& index = pair.first;
      se::DeviceMemoryBase& memory_base = pair.second;
      const Shape& subshape =
          ShapeUtil::GetSubshape(result_buffers.on_device_shape(), index);
      LOG(INFO) << "tuple result buffer info " << subshape.ToString();

      tensor[count - 1]->CopyDataFromTensor(memory_base.opaque());
      *(size_t*)(top_memory_base.opaque() + sizeof(void*) * (count - 1)) =
          (size_t)memory_base.opaque();
      LOG(INFO) << "sub tensor mem is " << memory_base.opaque();
    }
  }

  ExecutionOutput result(std::move(result_buffers));
  LOG(INFO) << "Leave " << module().name() << " :: " << (void*)this
            << " :: " << tsl::Env::Default()->GetCurrentThreadId();
  return result;
}

StatusOr<std::vector<ScopedShapedBuffer>> VsiExecutable::ExecuteOnStreams(
    absl::Span<const ServiceExecutableRunOptions> run_options,
    absl::Span<const absl::Span<const ShapedBuffer* const>> arguments) {
  LOG(FATAL) << "not implement";
}

}  // namespace vsiplugin
}  // namespace xla