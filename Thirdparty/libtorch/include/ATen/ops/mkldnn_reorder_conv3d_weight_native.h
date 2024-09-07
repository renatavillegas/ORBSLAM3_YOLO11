#pragma once

// @generated by torchgen/gen.py from NativeFunction.h

#include <c10/core/Scalar.h>
#include <c10/core/Storage.h>
#include <c10/core/TensorOptions.h>
#include <c10/util/Deprecated.h>
#include <c10/util/Optional.h>
#include <c10/core/QScheme.h>
#include <ATen/core/Reduction.h>
#include <ATen/core/Tensor.h>
#include <tuple>
#include <vector>


namespace at {
namespace native {
TORCH_API at::Tensor & mkldnn_reorder_conv3d_weight_out_symint(const at::Tensor & self, c10::SymIntArrayRef padding, c10::SymIntArrayRef stride, c10::SymIntArrayRef dilation, c10::SymInt groups, at::OptionalSymIntArrayRef input_size, at::Tensor & out);
TORCH_API at::Tensor mkldnn_reorder_conv3d_weight(const at::Tensor & self, at::IntArrayRef padding=0, at::IntArrayRef stride=1, at::IntArrayRef dilation=1, int64_t groups=1, at::OptionalIntArrayRef input_size=::std::nullopt);
} // namespace native
} // namespace at
