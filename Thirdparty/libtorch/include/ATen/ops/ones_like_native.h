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
TORCH_API at::Tensor ones_like(const at::Tensor & self, ::std::optional<at::ScalarType> dtype={}, ::std::optional<at::Layout> layout={}, ::std::optional<at::Device> device={}, ::std::optional<bool> pin_memory={}, ::std::optional<at::MemoryFormat> memory_format=::std::nullopt);
TORCH_API at::Tensor & ones_like_out(const at::Tensor & self, ::std::optional<at::MemoryFormat> memory_format, at::Tensor & out);
} // namespace native
} // namespace at
