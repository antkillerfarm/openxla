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

#ifndef TENSORFLOW_COMPILER_PLUGIN_VSI_DRIVER_VISITORS_VISITOR_BASE_H_
#define TENSORFLOW_COMPILER_PLUGIN_VSI_DRIVER_VISITORS_VISITOR_BASE_H_

#include <mutex>
#include <string>
#include <unordered_map>

#include "xla/hlo/ir/dfs_hlo_visitor_with_default.h"
#include "xla/hlo/evaluator/hlo_evaluator.h"
#include "../vsi_executor.h"
#include "tim/vx/context.h"
#include "tim/vx/graph.h"
#include "tim/vx/tensor.h"
#include "tim/vx/types.h"
namespace xla {
namespace vsiplugin {

/*
 * The base visitor handles all operations that are element-wise.
 * This includes all explicitly element-wise ops, for temprarily, they
 * are implemented by hlo_evaluator, and repalce it with AIM implment
 * step by step. All of these have no element to element dependencies.
 */
class BaseVisitor : public DfsHloVisitorWithDefault {
 public:
  BaseVisitor(VsiExecutor* executor)
      : executor_(executor), graph_(executor->getContext()->CreateGraph()){
    // custom_call_map_.emplace(
    //     std::make_pair("CustomCallLogSoftmax", CustomCallLogSoftmax));
    // custom_call_map_.emplace(std::make_pair("CustomCallTopK", CustomCallTopK));
    // custom_call_map_.emplace(
    //     std::make_pair("CustomCallSqueeze", CustomCallSqueeze));

  };

  std::shared_ptr<tim::vx::Tensor> CreateTensorFromTupleShape(
      const Shape& shape, int64_t index,
      tim::vx::TensorAttribute attr = tim::vx::TensorAttribute::INPUT) {
    tim::vx::Quantization timQuant;
    LOG(INFO) << "shape info 0: ";

    auto output_shape = shape.tuple_shapes(index);
    tim::vx::ShapeType timShape;

#if 0
    if (output_shape.is_static() && output_shape.has_layout()) {
      for (auto d : output_shape.layout().minor_to_major())
        timShape.push_back(output_shape.dimensions(d));
    }
#else
    if (output_shape.is_static()) {
      timShape.resize(shape.rank());
      for (uint32_t i = 0; i < output_shape.rank(); i++) {
        timShape[output_shape.rank() - 1 - i] = output_shape.dimensions(i);
      }
    }
#endif

    if (timShape.size() == 0) {
      timShape.push_back(1);
    }

    {
      std::ostringstream ss;
      for (uint32_t i = 0; i < timShape.size(); i++) {
        ss << timShape[i] << " ";
      }
      LOG(INFO) << __FUNCTION__ << " : " << ss.str();
    }

    auto type = convertTfPrimitiveTypeToTim(output_shape.element_type());
    std::unique_lock<std::mutex> lock(mutex_);
    tim::vx::TensorSpec timSpec(type, timShape, attr, timQuant);
    return graph_->CreateTensor(timSpec);
  }

  std::shared_ptr<tim::vx::Tensor> CreateTensorFromShape(
      const Shape& shape,
      tim::vx::TensorAttribute attr = tim::vx::TensorAttribute::INPUT) {
    tim::vx::ShapeType timShape;
    tim::vx::Quantization timQuant;
    LOG(INFO) << "shape info 1: ";
#if 0
    if (shape.is_static() && shape.has_layout()) {
      for (auto d : shape.layout().minor_to_major())
        timShape.push_back(shape.dimensions(d));
    }
#else
    if (shape.is_static()) {
      timShape.resize(shape.rank());
      for (uint32_t i = 0; i < shape.rank(); i++) {
        timShape[shape.rank() - 1 - i] = shape.dimensions(i);
      }
    }
#endif
    if (timShape.size() == 0) {
      timShape.push_back(1);
    }

    {
      std::ostringstream ss;
      for (uint32_t i = 0; i < timShape.size(); i++) {
        ss << timShape[i] << " ";
      }
      LOG(INFO) << __FUNCTION__ << " : " << ss.str();
    }

    auto type = convertTfPrimitiveTypeToTim(shape.element_type());
    std::unique_lock<std::mutex> lock(mutex_);
    tim::vx::TensorSpec timSpec(type, timShape, attr, timQuant);
    return graph_->CreateTensor(timSpec);
  }

  std::shared_ptr<tim::vx::Tensor> CreateTensorFromShape(
      tim::vx::DataType dataType, std::vector<uint32_t> shape,
      tim::vx::TensorAttribute attr = tim::vx::TensorAttribute::INPUT) {
    tim::vx::ShapeType timShape;
    tim::vx::Quantization timQuant;
    for (auto d : shape) timShape.push_back(d);
    LOG(INFO) << "shape info 2: ";
    if (timShape.size() == 0) {
      timShape.push_back(1);
    }

    {
      std::ostringstream ss;
      for (uint32_t i = 0; i < timShape.size(); i++) {
        ss << timShape[i] << " ";
      }
      LOG(INFO) << __FUNCTION__ << " : " << ss.str();
    }

    std::unique_lock<std::mutex> lock(mutex_);
    tim::vx::TensorSpec timSpec(dataType, timShape, attr, timQuant);
    return graph_->CreateTensor(timSpec);
  }

  static tim::vx::DataType convertTfPrimitiveTypeToTim(
      xla::PrimitiveType xlaType) {
    LOG(INFO) << "convertTfPrimitiveTypeToTim: xlaType: " << xlaType;
    switch (xlaType) {
      case PRED: {
        return tim::vx::DataType::BOOL8;
      }
      case S64: {
        return tim::vx::DataType::INT32;
      }
      case S8: {
        return tim::vx::DataType::INT8;
      }
      case U8: {
        return tim::vx::DataType::UINT8;
      }
      case S16: {
        return tim::vx::DataType::INT16;
      }
      case U16: {
        return tim::vx::DataType::UINT16;
      }
      case S32: {
        return tim::vx::DataType::INT32;
      }
      case U32: {
        return tim::vx::DataType::UINT32;
      }
      case F32: {
        return tim::vx::DataType::FLOAT32;
      }
      case BF16: {
        return tim::vx::DataType::FLOAT16;
      }
      case F16: {
        return tim::vx::DataType::FLOAT16;
      }
      case F64: {
        return tim::vx::DataType::FLOAT32;
      }
      default:
        LOG(FATAL) << "not supported datat type";
    }
  }

  // XLA Layout -> Device Layout
  std::shared_ptr<tim::vx::Tensor> InsertTransposeToDeviceLayout(
      const HloInstruction* hlo, std::vector<uint32_t>& dim_index);

  // Device Layout -> XLA Layout
  std::shared_ptr<tim::vx::Tensor> InsertTransposeFromDeviceLayout(
      const HloInstruction* hlo, std::vector<uint32_t>& dim_index);

  virtual const Shape& GetOutputShape(HloInstruction*) const;

  StatusOr<std::vector<std::shared_ptr<tim::vx::Tensor>>> Evaluate(
      const HloComputation& computation,
      std::vector<Literal>& argument_literals);

  Status HandleHloOp(HloInstruction* hlo);

  Status FinishVisit(HloInstruction* root) final;

  // Returns the already-evaluated literal result for the instruction.
  //
  // A Constant instruction is considered evaluated and its literal will be
  // returned directly without looking up the cache.
  //
  // Similarly, a Parameter instruction is considered evaluated and its literal
  // is looked up in arg_literals.
  //
  // Crash with log if the given instruction has not been evaluated previously.
  const Literal& GetEvaluatedLiteralFor(const HloInstruction* hlo) {
    if (hlo->IsConstant()) {
      return hlo->literal();
    }
    // if (hlo->opcode() == HloOpcode::kParameter) {
    //     return *arg_literals_.at(hlo->parameter_number());
    // }
    auto it = evaluated_.find(hlo);
    CHECK(it != evaluated_.end())
        << "could not find evaluated value for: " << hlo->ToString();
    return it->second;
  }

  const std::vector<std::shared_ptr<tim::vx::Tensor>> GetEvaluatedTensorFor(
      const HloInstruction* hlo) {
    // return CreateTensorFromShape(hlo->shape());
    auto it = vsi_run_tensor_container_.find(hlo);
    CHECK(it != vsi_run_tensor_container_.end())
        << "could not find evaluated value for: " << hlo->ToString();
    return vsi_run_tensor_container_[hlo];
  }

  // Called by HandleElementwiseBinarythe FinishVisit.
  virtual Status FinishScopedVisit(HloInstruction* root) {
    return ::tsl::OkStatus();
  }

  template <typename T>
  Status HandleSimpleElementwiseBinary(HloInstruction* hlo);

  template <typename T>
  Status HandleSimpleElementwiseUnary(HloInstruction* hlo);

  template <typename T>
  Status CreateCompareOp(std::shared_ptr<tim::vx::Tensor>& lhs_tensor,
                         std::shared_ptr<tim::vx::Tensor>& rhs_tensor,
                         std::shared_ptr<tim::vx::Tensor>& out_tensor);

  template <typename T>
  Status CreateReduceOp(std::shared_ptr<tim::vx::Tensor>& input,
                        std::shared_ptr<tim::vx::Tensor>& output,
                        std::vector<int32_t>& axis);

  Status HandleReduceOpMap(HloOpcode opcode,
                           std::shared_ptr<tim::vx::Tensor>& input,
                           std::shared_ptr<tim::vx::Tensor>& output,
                           std::vector<int32_t>& axis);

  Status HandleElementwiseBinary(HloInstruction* hlo) override;

  Status HandleElementwiseUnary(HloInstruction* hlo) override;

  Status HandleConstant(HloInstruction* hlo) override;

  Status HandleParameter(HloInstruction* hlo) override;

  Status HandleReshape(HloInstruction* hlo) override;

  Status HandleTranspose(HloInstruction* hlo) override;

  Status HandleTuple(HloInstruction* hlo) override;

  Status HandleGetTupleElement(HloInstruction* hlo) override;

  Status HandleConvolution(HloInstruction* hlo) override;

  Status HandleReverse(HloInstruction* hlo) override;

  Status HandleConvert(HloInstruction* hlo) override;

  Status HandlePad(HloInstruction* hlo) override;

  // Status HandleSlice(HloInstruction* hlo) override;

  Status HandleBroadcast(HloInstruction* hlo) override;

  Status HandleConcatenate(HloInstruction* hlo) override;

  Status HandleCompare(HloInstruction* hlo) override;

  Status HandleSelect(HloInstruction* hlo) override;

  Status HandleReduce(HloInstruction* hlo) override;

  Status HandleReduceWindow(HloInstruction* hlo) override;

  Status HandleDot(HloInstruction* hlo) override;

  Status HandleIota(HloInstruction* hlo) override;

  Status HandleCopy(HloInstruction* hlo) override;

  Status HandleScatter(HloInstruction* hlo) override;

  Status HandleSelectAndScatter(HloInstruction* hlo) override;

  Status HandleGather(HloInstruction* hlo) override;

  Status HandleSlice(HloInstruction* hlo) override;

  // Status HandleCustomCall(HloInstruction* inst) override;

  // static Status CustomCallLogSoftmax(HloInstruction* inst, BaseVisitor* bv);

  // static Status CustomCallTopK(HloInstruction* inst, BaseVisitor* bv);

  // static Status CustomCallSqueeze(HloInstruction* inst, BaseVisitor* bv);

  // static Status CustomCallSoftmax(HloInstruction* inst, BaseVisitor* bv);

  // static Status CustomCallArgMax(HloInstruction* inst, BaseVisitor* bv);

  Status DefaultAction(HloInstruction* hlo) override {
    return Unimplemented("unhandled HLO ops for VsiBaseVisitor: %s.",
                         HloOpcodeString(hlo->opcode()));
  }

 protected:
  const std::string name_ = "vsi base visitor";

  std::unique_ptr<HloEvaluator> cpu_evaluator_;

 private:
  VsiExecutor* executor_;

  // Tracks the HLO instruction and its evaluated literal result.
  // Parameters and constants aren't stored here,
  // TODO: it is better the Literal value was repalced with device memory
  //       handle.
  std::mutex mutex_;
  std::unordered_map<const HloInstruction*, Literal> evaluated_
      TF_GUARDED_BY(mutex_);
  std::unordered_map<const HloInstruction*,
                     std::vector<std::shared_ptr<tim::vx::Tensor>>>
      vsi_run_tensor_container_ TF_GUARDED_BY(mutex_);
  std::vector<Literal> arg_literals_;
  std::unordered_map<int64_t, uint32_t> kVsiInputId_ TF_GUARDED_BY(mutex_);
  std::shared_ptr<tim::vx::Graph> graph_;
  typedef Status (*cc_ptr)(xla::HloInstruction*, BaseVisitor* bv);
  std::unordered_map<std::string, cc_ptr> custom_call_map_;
  bool is_build_ = false;
};

}  // namespace vsiplugin
}  // namespace xla

#endif
