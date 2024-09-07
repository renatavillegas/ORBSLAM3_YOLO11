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
TORCH_API at::Tensor linspace(const at::Scalar & start, const at::Scalar & end, int64_t steps, ::std::optional<at::ScalarType> dtype={}, ::std::optional<at::Layout> layout={}, ::std::optional<at::Device> device={}, ::std::optional<bool> pin_memory={});
TORCH_API at::Tensor & linspace_out(const at::Scalar & start, const at::Scalar & end, int64_t steps, at::Tensor & out);
TORCH_API at::Tensor & linspace_cuda_out(const at::Scalar & start, const at::Scalar & end, int64_t steps, at::Tensor & out);
TORCH_API at::Tensor linspace(const at::Tensor & start, const at::Tensor & end, int64_t steps, ::std::optional<at::ScalarType> dtype={}, ::std::optional<at::Layout> layout={}, ::std::optional<at::Device> device={}, ::std::optional<bool> pin_memory={});
TORCH_API at::Tensor & linspace_out(const at::Tensor & start, const at::Tensor & end, int64_t steps, at::Tensor & out);
TORCH_API at::Tensor linspace(const at::Tensor & start, const at::Scalar & end, int64_t steps, ::std::optional<at::ScalarType> dtype={}, ::std::optional<at::Layout> layout={}, ::std::optional<at::Device> device={}, ::std::optional<bool> pin_memory={});
TORCH_API at::Tensor & linspace_out(const at::Tensor & start, const at::Scalar & end, int64_t steps, at::Tensor & out);
TORCH_API at::Tensor linspace(const at::Scalar & start, const at::Tensor & end, int64_t steps, ::std::optional<at::ScalarType> dtype={}, ::std::optional<at::Layout> layout={}, ::std::optional<at::Device> device={}, ::std::optional<bool> pin_memory={});
TORCH_API at::Tensor & linspace_out(const at::Scalar & start, const at::Tensor & end, int64_t steps, at::Tensor & out);
} // namespace native
} // namespace at
