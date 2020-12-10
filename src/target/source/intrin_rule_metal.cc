/*
 * Licensed to the Apache Software Foundation (ASF) under one
 * or more contributor license agreements.  See the NOTICE file
 * distributed with this work for additional information
 * regarding copyright ownership.  The ASF licenses this file
 * to you under the Apache License, Version 2.0 (the
 * "License"); you may not use this file except in compliance
 * with the License.  You may obtain a copy of the License at
 *
 *   http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
 * KIND, either express or implied.  See the License for the
 * specific language governing permissions and limitations
 * under the License.
 */

/*!
 * \file intrin_rule_metal.cc
 * \brief Metal intrinsic rules.
 */
#include "../intrin_rule.h"
#include <tvm/tir/op.h>

namespace tvm {
namespace codegen {
namespace intrin {

TVM_REGISTER_GLOBAL("tvm.intrin.rule.metal.floor").set_body(DispatchPureExtern<Direct>);

TVM_REGISTER_GLOBAL("tvm.intrin.rule.metal.ceil").set_body(DispatchPureExtern<Direct>);

TVM_REGISTER_GLOBAL("tvm.intrin.rule.metal.trunc").set_body(DispatchPureExtern<Direct>);

TVM_REGISTER_GLOBAL("tvm.intrin.rule.metal.fabs").set_body(DispatchPureExtern<Direct>);

TVM_REGISTER_GLOBAL("tvm.intrin.rule.metal.round").set_body(DispatchPureExtern<Direct>);

TVM_REGISTER_GLOBAL("tvm.intrin.rule.metal.exp").set_body(DispatchPureExtern<Direct>);

TVM_REGISTER_GLOBAL("tvm.intrin.rule.metal.exp2").set_body(DispatchPureExtern<Direct>);

TVM_REGISTER_GLOBAL("tvm.intrin.rule.metal.exp10").set_body(DispatchPureExtern<Direct>);

TVM_REGISTER_GLOBAL("tvm.intrin.rule.metal.log").set_body(DispatchPureExtern<Direct>);

TVM_REGISTER_GLOBAL("tvm.intrin.rule.metal.log2").set_body(DispatchPureExtern<Direct>);

TVM_REGISTER_GLOBAL("tvm.intrin.rule.metal.log10").set_body(DispatchPureExtern<Direct>);

TVM_REGISTER_GLOBAL("tvm.intrin.rule.metal.tanh").set_body(DispatchPureExtern<Direct>);

TVM_REGISTER_GLOBAL("tvm.intrin.rule.metal.sqrt").set_body(DispatchPureExtern<Direct>);

TVM_REGISTER_GLOBAL("tvm.intrin.rule.metal.pow").set_body(DispatchPureExtern<Direct>);

TVM_REGISTER_GLOBAL("tvm.intrin.rule.metal.popcount").set_body(DispatchPureExtern<Direct>);

TVM_REGISTER_GLOBAL("tvm.intrin.rule.metal.fmod").set_body(DispatchPureExtern<Direct>);

TVM_REGISTER_GLOBAL("tvm.intrin.rule.metal.sin").set_body(DispatchPureExtern<Direct>);

TVM_REGISTER_GLOBAL("tvm.intrin.rule.metal.sinh").set_body(DispatchPureExtern<Direct>);

TVM_REGISTER_GLOBAL("tvm.intrin.rule.metal.cos").set_body(DispatchPureExtern<Direct>);

TVM_REGISTER_GLOBAL("tvm.intrin.rule.metal.cosh").set_body(DispatchPureExtern<Direct>);

TVM_REGISTER_GLOBAL("tvm.intrin.rule.metal.erf").set_body([](const TVMArgs& args, TVMRetValue* rv) {
      PrimExpr e = args[0];
      const CallNode* call = e.as<CallNode>();
      ICHECK(call != nullptr);
      // A&S formula 7.1.26
      auto dtype = e.dtype();
      auto a1 = tir::make_const(dtype, 0.254829592);
      auto a2 = tir::make_const(dtype, -0.284496736);
      auto a3 = tir::make_const(dtype, 1.421413741);
      auto a4 = tir::make_const(dtype, -1.453152027);
      auto a5 = tir::make_const(dtype, 1.061405429);
      auto p = tir::make_const(dtype, 0.3275911);
      auto one = tir::make_const(dtype, 1);
      auto x = abs(e);
      auto t = one / (one + p * x);
      auto y = one - ((((a5 * t + a4) * t + a3) * t + a2) * t + a1) *
                   t * tvm::exp(-x * x);

      *rv = Select(e >=0, y, -y);
    });

}  // namespace intrin
}  // namespace codegen
}  // namespace tvm
