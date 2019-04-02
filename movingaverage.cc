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
 * Copyright (c) 2015 by Contributors
 * \file Quantization_int8.cc
 * \brief
 * \author Jingqiu Zhou
*/

#include "./movingaverage-inl.h"

#include <nnvm/op_attr_types.h>

namespace mshadow{
template<typename DType>
void moving_average_fwd(Tensor<cpu, 3, DType> data,Tensor<cpu, 3, DType> &out,
                        DType *data_his,DType decay_rate,bool init_not,Stream<cpu> *s){ 
    int num = data.size(0)*data.size(1)*data.size(2);
    for(int idx=0;idx<num;idx++){
        *(out.dptr_+idx)=init_not?*(data.dptr_+idx):*(data.dptr_+idx)*(1-decay_rate)+(*(data_his+idx))*decay_rate;
        *(data_his+idx)=*(out.dptr_+idx);
    }
}

template<typename DType>
void moving_average_BP(Tensor<cpu, 3, DType> grad,Tensor<cpu, 3, DType> &gdata,DType decay_rate,Stream<cpu> *s){ 
    //the quantization function
    int num = gdata.size(0)*gdata.size(1)*gdata.size(2);
    for(int idx=0;idx<num;idx++){
        *(gdata.dptr_+idx)=(1-decay_rate)*(*(grad.dptr_+idx));
    }
}
}

namespace mxnet {
namespace op {

template<>
Operator *CreateOp<cpu>(Moving_averagePara param, int dtype) {
  Operator* op = nullptr;
  MSHADOW_REAL_TYPE_SWITCH(dtype, DType, {
    op = new Moving_averageOp<cpu, DType>(param);
  });
  return op;
}

Operator *Moving_averageProp::CreateOperatorEx(Context ctx, std::vector<TShape> *in_shape,
                                          std::vector<int> *in_type) const {
  DO_BIND_DISPATCH(CreateOp, param_, in_type->at(0));
}

DMLC_REGISTER_PARAMETER(Moving_averagePara);

MXNET_REGISTER_OP_PROPERTY(Moving_average, Moving_averageProp)
.describe(R"code(perform moving average on data)code" ADD_FILELINE)
.add_argument("data", "NDArray-or-Symbol", "Input data to activation function.")
.add_arguments(Moving_averagePara::__FIELDS__());

NNVM_REGISTER_OP(Moving_average)
.set_attr<nnvm::FSetInputVarAttrOnCompose>("FSetInputVarAttrOnCompose",
    [](const nnvm::NodeAttrs& attrs, nnvm::NodePtr var, const int index) {
      if (index == 1 && var->attrs.dict.find("__init__") == var->attrs.dict.end()) {
        var->attrs.dict["__init__"] = "[\"Constant\", {\"value\": 0.}]";
      }
    });

}  // namespace op
}  // namespace mxnet
