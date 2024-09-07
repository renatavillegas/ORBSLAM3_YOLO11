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



#include <ATen/ops/_scaled_dot_product_flash_attention_for_cpu_backward_ops.h>

namespace at {


// aten::_scaled_dot_product_flash_attention_for_cpu_backward(Tensor grad_out, Tensor query, Tensor key, Tensor value, Tensor out, Tensor logsumexp, float dropout_p, bool is_causal, *, Tensor? attn_mask=None, float? scale=None) -> (Tensor grad_query, Tensor grad_key, Tensor grad_value)
inline ::std::tuple<at::Tensor,at::Tensor,at::Tensor> _scaled_dot_product_flash_attention_for_cpu_backward(const at::Tensor & grad_out, const at::Tensor & query, const at::Tensor & key, const at::Tensor & value, const at::Tensor & out, const at::Tensor & logsumexp, double dropout_p, bool is_causal, const ::std::optional<at::Tensor> & attn_mask={}, ::std::optional<double> scale=::std::nullopt) {
    return at::_ops::_scaled_dot_product_flash_attention_for_cpu_backward::call(grad_out, query, key, value, out, logsumexp, dropout_p, is_causal, attn_mask, scale);
}

}
