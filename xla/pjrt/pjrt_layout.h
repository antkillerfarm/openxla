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

#ifndef XLA_PJRT_PJRT_LAYOUT_H_
#define XLA_PJRT_PJRT_LAYOUT_H_

#include "xla/layout.h"

namespace xla {

class PjRtLayout {
 public:
  virtual ~PjRtLayout() = default;

  // Returns the serialized layout as a string. Can be deserialized with
  // PjRtClient::DeserializeLayout.
  virtual StatusOr<std::string> Serialize() const = 0;

  // Human-readable string for error messages, user introspection, etc.
  virtual std::string ToString() const = 0;
};

// PjRtLayout backed by an xla::Layout. We intentionally don't publicly expose
// the xla::Layout itself. This is to limit the API surface area, since
// xla::Layout may contain experimental or other fields used internally by XLA
// that aren't necessarily supported in higher levels of the stack.
class PjRtXlaLayout : public PjRtLayout {
 public:
  explicit PjRtXlaLayout(const Layout& layout) : xla_layout_(layout) {}

  const absl::Span<const int64_t> minor_to_major() const {
    return xla_layout_.minor_to_major();
  }

  StatusOr<std::string> Serialize() const override {
    return xla_layout_.ToString();
  }
  std::string ToString() const { return xla_layout_.ToString(); }

 private:
  Layout xla_layout_;
};

}  // namespace xla

#endif  // XLA_PJRT_PJRT_LAYOUT_H_
