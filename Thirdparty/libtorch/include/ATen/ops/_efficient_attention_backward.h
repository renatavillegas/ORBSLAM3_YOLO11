#pragma once

// @generated by torchgen/gen.py from Function.h

#include <ATen/Context.h>
#include <ATen/DeviceGuard.h>
#include <ATen/TensorUtils.h>
#include <ATen/TracerMode.h>
#include <ATen/core/Generator.h>
#include <ATen/core/Reduction.h>
#include <ATen/core/Tensor.h>
#include <c10/core/Scalar.h>
#include <c10/core/Storage.h>
#include <c10/core/TensorOptions.h>
#include <c10/util/Deprecated.h>
#include <c10/util/Optional.h>



#include <ATen/ops/_efficient_attention_backward_ops.h>

namespace at {


// aten::_efficient_attention_backward(Tensor grad_out_, Tensor query, Tensor key, Tensor value, Tensor? bias, Tensor out, Tensor? cu_seqlens_q, Tensor? cu_seqlens_k, SymInt max_seqlen_q, SymInt max_seqlen_k, Tensor logsumexp, float dropout_p, Tensor philox_seed, Tensor philox_offset, int custom_mask_type, bool bias_requires_grad, *, float? scale=None, int? num_splits_key=None, int? window_size=None, bool shared_storage_dqdkdv=False) -> (Tensor, Tensor, Tensor, Tensor)
inline ::std::tuple<at::Tensor,at::Tensor,at::Tensor,at::Tensor> _efficient_attention_backward(const at::Tensor & grad_out_, const at::Tensor & query, const at::Tensor & key, const at::Tensor & value, const ::std::optional<at::Tensor> & bias, const at::Tensor & out, const ::std::optional<at::Tensor> & cu_seqlens_q, const ::std::optional<at::Tensor> & cu_seqlens_k, int64_t max_seqlen_q, int64_t max_seqlen_k, const at::Tensor & logsumexp, double dropout_p, const at::Tensor & philox_seed, const at::Tensor & philox_offset, int64_t custom_mask_type, bool bias_requires_grad, ::std::optional<double> scale=::std::nullopt, ::std::optional<int64_t> num_splits_key=::std::nullopt, ::std::optional<int64_t> window_size=::std::nullopt, bool shared_storage_dqdkdv=false) {
    return at::_ops::_efficient_attention_backward::call(grad_out_, query, key, value, bias, out, cu_seqlens_q, cu_seqlens_k, max_seqlen_q, max_seqlen_k, logsumexp, dropout_p, philox_seed, philox_offset, custom_mask_type, bias_requires_grad, scale, num_splits_key, window_size, shared_storage_dqdkdv);
}
namespace symint {
  template <typename T, typename = std::enable_if_t<std::is_same<T, int64_t>::value>>
  ::std::tuple<at::Tensor,at::Tensor,at::Tensor,at::Tensor> _efficient_attention_backward(const at::Tensor & grad_out_, const at::Tensor & query, const at::Tensor & key, const at::Tensor & value, const ::std::optional<at::Tensor> & bias, const at::Tensor & out, const ::std::optional<at::Tensor> & cu_seqlens_q, const ::std::optional<at::Tensor> & cu_seqlens_k, int64_t max_seqlen_q, int64_t max_seqlen_k, const at::Tensor & logsumexp, double dropout_p, const at::Tensor & philox_seed, const at::Tensor & philox_offset, int64_t custom_mask_type, bool bias_requires_grad, ::std::optional<double> scale=::std::nullopt, ::std::optional<int64_t> num_splits_key=::std::nullopt, ::std::optional<int64_t> window_size=::std::nullopt, bool shared_storage_dqdkdv=false) {
    return at::_ops::_efficient_attention_backward::call(grad_out_, query, key, value, bias, out, cu_seqlens_q, cu_seqlens_k, max_seqlen_q, max_seqlen_k, logsumexp, dropout_p, philox_seed, philox_offset, custom_mask_type, bias_requires_grad, scale, num_splits_key, window_size, shared_storage_dqdkdv);
  }
}

// aten::_efficient_attention_backward(Tensor grad_out_, Tensor query, Tensor key, Tensor value, Tensor? bias, Tensor out, Tensor? cu_seqlens_q, Tensor? cu_seqlens_k, SymInt max_seqlen_q, SymInt max_seqlen_k, Tensor logsumexp, float dropout_p, Tensor philox_seed, Tensor philox_offset, int custom_mask_type, bool bias_requires_grad, *, float? scale=None, int? num_splits_key=None, int? window_size=None, bool shared_storage_dqdkdv=False) -> (Tensor, Tensor, Tensor, Tensor)
inline ::std::tuple<at::Tensor,at::Tensor,at::Tensor,at::Tensor> _efficient_attention_backward_symint(const at::Tensor & grad_out_, const at::Tensor & query, const at::Tensor & key, const at::Tensor & value, const ::std::optional<at::Tensor> & bias, const at::Tensor & out, const ::std::optional<at::Tensor> & cu_seqlens_q, const ::std::optional<at::Tensor> & cu_seqlens_k, c10::SymInt max_seqlen_q, c10::SymInt max_seqlen_k, const at::Tensor & logsumexp, double dropout_p, const at::Tensor & philox_seed, const at::Tensor & philox_offset, int64_t custom_mask_type, bool bias_requires_grad, ::std::optional<double> scale=::std::nullopt, ::std::optional<int64_t> num_splits_key=::std::nullopt, ::std::optional<int64_t> window_size=::std::nullopt, bool shared_storage_dqdkdv=false) {
    return at::_ops::_efficient_attention_backward::call(grad_out_, query, key, value, bias, out, cu_seqlens_q, cu_seqlens_k, max_seqlen_q, max_seqlen_k, logsumexp, dropout_p, philox_seed, philox_offset, custom_mask_type, bias_requires_grad, scale, num_splits_key, window_size, shared_storage_dqdkdv);
}
namespace symint {
  template <typename T, typename = std::enable_if_t<std::is_same<T, c10::SymInt>::value>>
  ::std::tuple<at::Tensor,at::Tensor,at::Tensor,at::Tensor> _efficient_attention_backward(const at::Tensor & grad_out_, const at::Tensor & query, const at::Tensor & key, const at::Tensor & value, const ::std::optional<at::Tensor> & bias, const at::Tensor & out, const ::std::optional<at::Tensor> & cu_seqlens_q, const ::std::optional<at::Tensor> & cu_seqlens_k, c10::SymInt max_seqlen_q, c10::SymInt max_seqlen_k, const at::Tensor & logsumexp, double dropout_p, const at::Tensor & philox_seed, const at::Tensor & philox_offset, int64_t custom_mask_type, bool bias_requires_grad, ::std::optional<double> scale=::std::nullopt, ::std::optional<int64_t> num_splits_key=::std::nullopt, ::std::optional<int64_t> window_size=::std::nullopt, bool shared_storage_dqdkdv=false) {
    return at::_ops::_efficient_attention_backward::call(grad_out_, query, key, value, bias, out, cu_seqlens_q, cu_seqlens_k, max_seqlen_q, max_seqlen_k, logsumexp, dropout_p, philox_seed, philox_offset, custom_mask_type, bias_requires_grad, scale, num_splits_key, window_size, shared_storage_dqdkdv);
  }
}

}
