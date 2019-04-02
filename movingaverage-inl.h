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
 * \file movingaverage-inl.h
 * \brief
 * \author Jingqiu Zhou
*/
#ifndef MXNET_OPERATOR_MOVINGAVERAGE_INL_H_
#define MXNET_OPERATOR_MOVINGAVERAGE_INL_H_


#include <dmlc/logging.h>
#include <dmlc/parameter.h>
#include <mxnet/operator.h>
#include <cstring>
#include <map>
#include <string>
#include <vector>
#include <utility>
#include "./operator_common.h"
#include "./mshadow_op.h"
#include "./tensor/elemwise_binary_broadcast_op.h"
#include "../common/random_generator.h"
#include "./random/sampler.h"
#include "./random/sample_op.h"
#include<cuda.h>

namespace mxnet {
namespace op {

namespace Moving_average {
enum Moving_averageOpInputs {kData};
enum Moving_averageOpOutputs {kOut};
enum Moving_averageOpResource {kRandom};
}  // namespace leakyrelu

struct Moving_averagePara : public dmlc::Parameter<Moving_averagePara> {
  // use int for enumeration
  float decay;
  DMLC_DECLARE_PARAMETER(Moving_averagePara) {
    DMLC_DECLARE_FIELD(decay).set_default(0.9)
    .describe("decay rate of the ema");
  }
};

template<typename xpu, typename DType>
class Moving_averageOp : public Operator {
 public:
  explicit Moving_averageOp(Moving_averagePara param) {
    param_ = param;
    decay_rate=DType(param_.decay);
    init_not=true;
  }

  virtual void Forward(const OpContext &ctx,
                       const std::vector<TBlob> &in_data,
                       const std::vector<OpReqType> &req,
                       const std::vector<TBlob> &out_data,
                       const std::vector<TBlob> &aux_args) {
    using namespace mshadow;
    using namespace mshadow::expr;

    CHECK_EQ(in_data.size(), 1U);
    Stream<xpu> *s = ctx.get_stream<xpu>();
    Tensor<xpu, 3, DType> data;
    Tensor<xpu, 3, DType> out;

    int n = in_data[Moving_average::kData].shape_[0];
    int k = (in_data[Moving_average::kData].ndim() > 1) ? in_data[Moving_average::kData].shape_[1] : 1;
    Shape<3> dshape = Shape3(n, k, in_data[Moving_average::kData].Size()/n/k);
    data = in_data[Moving_average::kData].get_with_shape<xpu, 3, DType>(dshape, s);
    out = out_data[Moving_average::kOut].get_with_shape<xpu, 3, DType>(dshape, s);

    if(init_not){
        average_his =(DType *)malloc(sizeof(DType)*data.size(0)*data.size(1)*data.size(2));
    }

    moving_average_fwd(data,out,average_his,decay_rate,init_not,s);
    if(init_not){
        init_not=false;
    }

  }
  

  virtual void Backward(const OpContext & ctx,
                        const std::vector<TBlob> &out_grad,
                        const std::vector<TBlob> &in_data,
                        const std::vector<TBlob> &out_data,
                        const std::vector<OpReqType> &req,
                        const std::vector<TBlob> &in_grad,
                        const std::vector<TBlob> &aux_args) {
    using namespace mshadow;
    using namespace mshadow::expr;

    CHECK_EQ(out_grad.size(), 1U);

    Stream<xpu> *s = ctx.get_stream<xpu>();
 

    Tensor<xpu, 3, DType> gdata;
    Tensor<xpu, 3, DType> grad;
 
    int n = out_grad[Moving_average::kOut].shape_[0];
    int k = (out_grad[Moving_average::kOut].ndim() > 1) ? out_grad[Moving_average::kOut].shape_[1] : 1;
    Shape<3> dshape = Shape3(n, k, out_grad[Moving_average::kOut].Size()/n/k);
    grad = out_grad[Moving_average::kOut].get_with_shape<xpu, 3, DType>(dshape, s);
    gdata = in_grad[Moving_average::kData].get_with_shape<xpu, 3, DType>(dshape, s);
    
    moving_average_BP(grad,gdata,decay_rate,s);
  }

 private:
  Moving_averagePara param_;
  bool init_not;
  DType decay_rate;
  DType *average_his;

};  // class LeakyReLUOp

template<typename xpu>
Operator* CreateOp(Moving_averagePara type, int dtype);

#if DMLC_USE_CXX11
class Moving_averageProp : public OperatorProperty {
 public:
  void Init(const std::vector<std::pair<std::string, std::string> >& kwargs) override {
    param_.Init(kwargs);
  }

  std::map<std::string, std::string> GetParams() const override {
    return param_.__DICT__();
  }

  bool InferShape(std::vector<TShape> *in_shape,
                  std::vector<TShape> *out_shape,
                  std::vector<TShape> *aux_shape) const override {
    using namespace mshadow;

    CHECK_EQ(in_shape->size(), 1U) << "Input:[data]";
    const TShape &dshape = in_shape->at(Moving_average::kData);

    if (dshape.ndim() == 0) return false;

    out_shape->clear();
    out_shape->push_back(dshape);

    return true;
  }

  bool InferType(std::vector<int> *in_type,
                 std::vector<int> *out_type,
                 std::vector<int> *aux_type) const override {
    int dtype = -1;
    for (const int& type : *in_type) {
      type_assign(&dtype, type);
    }
    for (const int& type : *out_type) {
      type_assign(&dtype, type);
    }

    for (size_t i = 0; i < in_type->size(); ++i) {
      TYPE_ASSIGN_CHECK(*in_type, i, dtype);
    }
    for (size_t i = 0; i < out_type->size(); ++i) {
      TYPE_ASSIGN_CHECK(*out_type, i, dtype);
    }
    return dtype != -1;
  }

  OperatorProperty* Copy() const override {
    auto ptr = new Moving_averageProp();
    ptr->param_ = param_;
    return ptr;
  }

  std::string TypeString() const override {
    return "Moving_average";
  }

  // decalre dependency and inplace optimization options
  std::vector<int> DeclareBackwardDependency(
    const std::vector<int> &out_grad,
    const std::vector<int> &in_data,
    const std::vector<int> &out_data) const override {
    return {out_grad[Moving_average::kOut], out_data[Moving_average::kData]};
  }

  std::vector<std::pair<int, void*> > BackwardInplaceOption(
    const std::vector<int> &out_grad,
    const std::vector<int> &in_data,
    const std::vector<int> &out_data,
    const std::vector<void*> &in_grad) const override {
    return {{out_grad[Moving_average::kOut], in_grad[Moving_average::kData]}};
  }

  std::vector<std::pair<int, void*> > ForwardInplaceOption(
    const std::vector<int> &in_data,
    const std::vector<void*> &out_data) const override {

    return {{in_data[Moving_average::kData], out_data[Moving_average::kOut]}};
  }

  std::vector<std::string> ListArguments() const override {
    return {"data"};
  }

  std::vector<std::string> ListOutputs() const override {
    return {"output"};
  }

  int NumOutputs() const override {
    return 1;
  }

  int NumVisibleOutputs() const override {
    return 1;
  }

  std::vector<ResourceRequest> ForwardResource(
      const std::vector<TShape> &in_shape) const override {
    return std::vector<ResourceRequest>();
  }

  std::vector<ResourceRequest> BackwardResource(
      const std::vector<TShape> &in_shape) const override {
    return {ResourceRequest::kTempSpace};
  }

  Operator* CreateOperator(Context ctx) const override {
    LOG(FATAL) << "Not Implemented.";
    return NULL;
  }

  Operator* CreateOperatorEx(Context ctx, std::vector<TShape> *in_shape,
                           std::vector<int> *in_type) const override;

 private:
  Moving_averagePara param_;
};
#endif  // DMLC_USE_CXX11
}  // namespace op
}  // namespace mxnet

#endif  // MXNET_OPERATOR_Moving_average_INL_H_

