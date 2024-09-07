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
TORCH_API at::Tensor & copy_out(const at::Tensor & self, const at::Tensor & src, bool non_blocking, at::Tensor & out);
TORCH_API at::Tensor & copy_(at::Tensor & self, const at::Tensor & src, bool non_blocking=false);
TORCH_API at::Tensor & copy_nested_(at::Tensor & self, const at::Tensor & src, bool non_blocking=false);
TORCH_API at::Tensor & copy_sparse_wrapper_(at::Tensor & self, const at::Tensor & src, bool non_blocking=false);
TORCH_API at::Tensor & copy_sparse_compressed_(at::Tensor & self, const at::Tensor & src, bool non_blocking=false);
TORCH_API at::Tensor copy_meta(const at::Tensor & self, const at::Tensor & src, bool non_blocking=false);
TORCH_API at::Tensor & copy_mkldnn_(at::Tensor & self, const at::Tensor & src, bool non_blocking=false);
TORCH_API at::Tensor copy(const at::Tensor & self, const at::Tensor & src, bool non_blocking=false);
} // namespace native
} // namespace at
