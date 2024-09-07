#pragma once

// @generated by torchgen/gen.py from Operator.h

#include <tuple>
#include <vector>

// Forward declarations of any types needed in the operator signatures.
// We can't directly include these classes because it will cause circular include dependencies.
// This file is included by TensorBody.h, which defines the Tensor class.
#include <ATen/core/ATen_fwd.h>

namespace at {
namespace _ops {


struct TORCH_API repeat_interleave_Tensor {
  using schema = at::Tensor (const at::Tensor &, ::std::optional<c10::SymInt>);
  using ptr_schema = schema*;
  // See Note [static constexpr char* members for windows NVCC]
  STATIC_CONSTEXPR_STR_INL_EXCEPT_WIN_CUDA(name, "aten::repeat_interleave")
  STATIC_CONSTEXPR_STR_INL_EXCEPT_WIN_CUDA(overload_name, "Tensor")
  STATIC_CONSTEXPR_STR_INL_EXCEPT_WIN_CUDA(schema_str, "repeat_interleave.Tensor(Tensor repeats, *, SymInt? output_size=None) -> Tensor")
  static at::Tensor call(const at::Tensor & repeats, ::std::optional<c10::SymInt> output_size);
  static at::Tensor redispatch(c10::DispatchKeySet dispatchKeySet, const at::Tensor & repeats, ::std::optional<c10::SymInt> output_size);
};

struct TORCH_API repeat_interleave_self_Tensor {
  using schema = at::Tensor (const at::Tensor &, const at::Tensor &, ::std::optional<int64_t>, ::std::optional<c10::SymInt>);
  using ptr_schema = schema*;
  // See Note [static constexpr char* members for windows NVCC]
  STATIC_CONSTEXPR_STR_INL_EXCEPT_WIN_CUDA(name, "aten::repeat_interleave")
  STATIC_CONSTEXPR_STR_INL_EXCEPT_WIN_CUDA(overload_name, "self_Tensor")
  STATIC_CONSTEXPR_STR_INL_EXCEPT_WIN_CUDA(schema_str, "repeat_interleave.self_Tensor(Tensor self, Tensor repeats, int? dim=None, *, SymInt? output_size=None) -> Tensor")
  static at::Tensor call(const at::Tensor & self, const at::Tensor & repeats, ::std::optional<int64_t> dim, ::std::optional<c10::SymInt> output_size);
  static at::Tensor redispatch(c10::DispatchKeySet dispatchKeySet, const at::Tensor & self, const at::Tensor & repeats, ::std::optional<int64_t> dim, ::std::optional<c10::SymInt> output_size);
};

struct TORCH_API repeat_interleave_self_int {
  using schema = at::Tensor (const at::Tensor &, c10::SymInt, ::std::optional<int64_t>, ::std::optional<c10::SymInt>);
  using ptr_schema = schema*;
  // See Note [static constexpr char* members for windows NVCC]
  STATIC_CONSTEXPR_STR_INL_EXCEPT_WIN_CUDA(name, "aten::repeat_interleave")
  STATIC_CONSTEXPR_STR_INL_EXCEPT_WIN_CUDA(overload_name, "self_int")
  STATIC_CONSTEXPR_STR_INL_EXCEPT_WIN_CUDA(schema_str, "repeat_interleave.self_int(Tensor self, SymInt repeats, int? dim=None, *, SymInt? output_size=None) -> Tensor")
  static at::Tensor call(const at::Tensor & self, c10::SymInt repeats, ::std::optional<int64_t> dim, ::std::optional<c10::SymInt> output_size);
  static at::Tensor redispatch(c10::DispatchKeySet dispatchKeySet, const at::Tensor & self, c10::SymInt repeats, ::std::optional<int64_t> dim, ::std::optional<c10::SymInt> output_size);
};

struct TORCH_API repeat_interleave_Tensor_out {
  using schema = at::Tensor & (const at::Tensor &, ::std::optional<c10::SymInt>, at::Tensor &);
  using ptr_schema = schema*;
  // See Note [static constexpr char* members for windows NVCC]
  STATIC_CONSTEXPR_STR_INL_EXCEPT_WIN_CUDA(name, "aten::repeat_interleave")
  STATIC_CONSTEXPR_STR_INL_EXCEPT_WIN_CUDA(overload_name, "Tensor_out")
  STATIC_CONSTEXPR_STR_INL_EXCEPT_WIN_CUDA(schema_str, "repeat_interleave.Tensor_out(Tensor repeats, *, SymInt? output_size=None, Tensor(a!) out) -> Tensor(a!)")
  static at::Tensor & call(const at::Tensor & repeats, ::std::optional<c10::SymInt> output_size, at::Tensor & out);
  static at::Tensor & redispatch(c10::DispatchKeySet dispatchKeySet, const at::Tensor & repeats, ::std::optional<c10::SymInt> output_size, at::Tensor & out);
};

}} // namespace at::_ops
