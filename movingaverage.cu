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
 * \file Quantization_int8.cu
 * \brief
 * \author Jingqiu Zhou
*/

#include "./movingaverage-inl.h"
#include<cuda.h>
#include "../common/cuda_utils.h"


namespace mxnet {
  namespace op {
    template<typename DType>
    struct MVAVE_FWD_GPU{
      __device__ static void Map(int i,DType *data,DType *out,DType *data_his,
                                 DType decay_rate,bool init_not){
        *(out+i) = init_not?*(data+i):*(data+i)*(1-decay_rate)+decay_rate*(*(data_his+i));
        *(data_his+i)=*(out+i);
      }
    };

    template<typename DType>
    struct MVAVE_BP_GPU{
      __device__ static void Map(int i,DType *grad,DType *gdata,DType decay_rate){
        *(gdata+i)=*(grad+i)*(1-decay_rate);
      }
    };
  }
}
namespace mshadow{
  template<typename DType>
  void moving_average_fwd(Tensor<gpu, 3, DType> data,Tensor<gpu, 3, DType> &out,
                        DType *data_his,DType decay_rate,bool init_not,Stream<gpu> *s){
    int num = out.size(0)*out.size(1)*out.size(2);
    DType *Temp;
    cudaMalloc((void **)&Temp,sizeof(DType)*num);
    cudaMemcpy(Temp,data_his,sizeof(DType)*num,cudaMemcpyHostToDevice);
    mxnet::op::mxnet_op::Kernel<mxnet::op::MVAVE_FWD_GPU<DType>,gpu>::Launch(s,num,
                                                                    data.dptr_,out.dptr_,
                                                                    Temp,decay_rate,init_not);
    cudaMemcpy(data_his,Temp,sizeof(DType)*num,cudaMemcpyDeviceToHost);
    cudaFree(Temp);
  }
  template<typename DType>
  void moving_average_BP(Tensor<gpu, 3, DType> grad,Tensor<gpu, 3, DType> &gdata,DType decay_rate,Stream<gpu> *s){
    int num = gdata.size(0)*gdata.size(1)*gdata.size(2);
   
    mxnet::op::mxnet_op::Kernel<mxnet::op::MVAVE_BP_GPU<DType>,gpu>::Launch(s,num,
                                                                    grad.dptr_,gdata.dptr_,
                                                                    decay_rate);
  }
}

namespace mxnet{
  namespace op{
template<>
Operator *CreateOp<gpu>(Moving_averagePara param, int dtype) {
  Operator* op = nullptr;
  MSHADOW_REAL_TYPE_SWITCH(dtype, DType, {
    op = new Moving_averageOp<gpu, DType>(param);
  });
  return op;
}

}  // namespace op
}  // namespace mxnet

