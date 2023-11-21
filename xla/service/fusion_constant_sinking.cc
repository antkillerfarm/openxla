/* Copyright 2023 The TensorFlow Authors. All Rights Reserved.

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

#include "xla/service/fusion_constant_sinking.h"

#include <algorithm>
#include <cstdint>
#include <optional>
#include <vector>

#include "absl/container/flat_hash_map.h"
#include "absl/container/flat_hash_set.h"
#include "absl/log/check.h"
#include "absl/log/log.h"
#include "absl/strings/string_view.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_module.h"
#include "xla/hlo/ir/hlo_opcode.h"
#include "xla/service/hlo_dce.h"
#include "xla/shape_util.h"
#include "xla/statusor.h"
#include "xla/util.h"
#include "tsl/platform/statusor.h"

namespace xla {

// Given the fusion instruction and the operand to the fusion, checks:
//   1. the operand is scalar and constant
//   2. the parameter instruction representing the operand is not used in any
//   fusion instructions with a single operand.
// if the checks hold, it returns the parameter instruction representing the
// operand in the fusion computation, otherwise nullopt.
bool CanSink(HloInstruction* fusion, const HloInstruction* operand) {
  if (!fusion->IsLoopFusion() && !fusion->IsOutputFusion()) {
    return false;
  }

  if (fusion->operand_count() == 1) {
    return false;
  }

  if (!ShapeUtil::IsScalar(operand->shape()) || !operand->IsConstant()) {
    return false;
  }

  int64_t operand_idx = fusion->operand_index(operand);
  HloInstruction* fused_param = fusion->fused_parameter(operand_idx);
  for (HloInstruction* user : fused_param->users()) {
    if (user->opcode() == HloOpcode::kFusion && user->operand_count() == 1) {
      return false;
    }
  }
  return true;
}

bool ProcessScalar(HloInstruction* scalar) {
  if (!ShapeUtil::IsScalar(scalar->shape()) || !scalar->IsConstant()) {
    return false;
  }
  bool processed = false;
  for (auto* use : scalar->users()) {
    if (CanSink(use, scalar)) {
      auto fused_scalar = use->FuseInstruction(scalar);
      processed = true;
      ProcessScalar(fused_scalar);
    }
  }
  return processed;
}

StatusOr<bool> FusionConstantSinking::Run(
    HloModule* module,
    const absl::flat_hash_set<absl::string_view>& execution_threads) {
  VLOG(3) << "HLO module before FusionConstantSinking:";
  XLA_VLOG_LINES(3, module->ToString());

  bool changed = false;
  for (auto* c : module->MakeNonfusionComputations()) {
    for (auto* i : c->MakeInstructionPostOrder()) {
      changed |= ProcessScalar(i);
    }
  }

  if (changed) {
    TF_ASSIGN_OR_RETURN(bool dce, HloDCE{}.Run(module, execution_threads));
    changed |= dce;
  }

  VLOG(3) << "HLO module after FusionConstantSinking:";
  XLA_VLOG_LINES(3, module->ToString());
  return changed;
}

}  // namespace xla
