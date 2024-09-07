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



#include <ATen/ops/linalg_ldl_factor_ops.h>

namespace at {


// aten::linalg_ldl_factor(Tensor self, *, bool hermitian=False) -> (Tensor LD, Tensor pivots)
inline ::std::tuple<at::Tensor,at::Tensor> linalg_ldl_factor(const at::Tensor & self, bool hermitian=false) {
    return at::_ops::linalg_ldl_factor::call(self, hermitian);
}

// aten::linalg_ldl_factor.out(Tensor self, *, bool hermitian=False, Tensor(a!) LD, Tensor(b!) pivots) -> (Tensor(a!) LD, Tensor(b!) pivots)
inline ::std::tuple<at::Tensor &,at::Tensor &> linalg_ldl_factor_out(at::Tensor & LD, at::Tensor & pivots, const at::Tensor & self, bool hermitian=false) {
    return at::_ops::linalg_ldl_factor_out::call(self, hermitian, LD, pivots);
}
// aten::linalg_ldl_factor.out(Tensor self, *, bool hermitian=False, Tensor(a!) LD, Tensor(b!) pivots) -> (Tensor(a!) LD, Tensor(b!) pivots)
inline ::std::tuple<at::Tensor &,at::Tensor &> linalg_ldl_factor_outf(const at::Tensor & self, bool hermitian, at::Tensor & LD, at::Tensor & pivots) {
    return at::_ops::linalg_ldl_factor_out::call(self, hermitian, LD, pivots);
}

}
