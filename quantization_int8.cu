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

#include "./quantization_int8-inl.h"
#include<cuda.h>
#include "../common/cuda_utils.h"

#define QUANT_LEVEL 255
#define THEAD_PER_BLOCK 256
namespace mxnet {
  namespace op {
    template<typename DType>
    struct QUANT_WEIGHT_GPU{
      __device__ static void Map(int i,DType *data,DType *out,
                                 DType *src_max,DType *src_min){

        __shared__ DType quant_unit;
        __shared__ DType S_min_f;
        __shared__ DType S_max_f;
        int tidx=threadIdx.x;
        //compute quantization inside each block
        if(tidx<1){
    
          S_min_f=*src_min;
          S_max_f=*src_max;

          //insure 0 in the range
          if(S_min_f>DType(-1e-8)){
            S_min_f=DType(-1e-2);
          }
          if(S_max_f<DType(1e-8)){
            S_max_f=DType(1e-2);
          }
          //calculate a possible quant_unit
          quant_unit = (S_max_f-S_min_f)/DType(QUANT_LEVEL);
          DType delta = quant_unit + S_min_f/ceil(-S_min_f/quant_unit);
          //adjust range 
          quant_unit = quant_unit-delta;
          S_max_f=S_max_f-delta*DType(QUANT_LEVEL)/DType(2.);
          S_min_f=S_min_f+delta*DType(QUANT_LEVEL)/DType(2.);
        }

        __syncthreads();
        DType temp = *(data+i)>S_max_f?S_max_f:*(data+i);
        temp = temp<S_min_f?S_min_f:temp;
        *(out+i)=floor((temp-S_min_f)/quant_unit+0.5)*quant_unit+S_min_f;
      }
    };

    template<typename DType>
    struct QUANT_ACT_GPU{
      __device__ static void Map(int i,DType *data,DType *out,DType *S_act,DType *max_S,DType *min_S,
                                 DType decay,int quant_countdown,bool init){
        DType S_max_f;
        DType S_min_f;
        DType quant_unit;
        if(init){
          S_max_f = *max_S;
          S_min_f = *min_S;
        } else {
          S_max_f = *S_act*decay+(1-decay)*(*max_S);
          S_min_f = *(S_act+1)*decay+(1-decay)*(*min_S);
        }
        if(S_max_f<1e-7){
          S_max_f=1e-2;
        }
        if(S_min_f>-1e-7){
          S_min_f=-1e-2;
        }

        if(i==0){
          *S_act = S_max_f;
          *(S_act+1) = S_min_f;
        }
        if(quant_countdown==0){
          quant_unit = (S_max_f-S_min_f)/DType(QUANT_LEVEL);
          //use i= 0 to update the recorded max/min
          DType temp = *(data+i)>S_max_f?S_max_f:*(data+i);
          temp = temp<S_min_f?S_min_f:temp;
          *(out+i)=floor((temp-S_min_f)/quant_unit+0.5)*quant_unit+S_min_f;      
        } else {
          *(out+i)=*(data+i);
        }
        
      }
    };


    template<typename DType>
    struct Launch_warper{ 
      __device__ static void warpReduce_max(volatile DType * max_arr,int tid){
        max_arr[tid] = max_arr[tid]>max_arr[tid+32]?max_arr[tid]:max_arr[tid+32];
        max_arr[tid] = max_arr[tid]>max_arr[tid+16]?max_arr[tid]:max_arr[tid+16];
        max_arr[tid] = max_arr[tid]>max_arr[tid+8]?max_arr[tid]:max_arr[tid+8];
        max_arr[tid] = max_arr[tid]>max_arr[tid+4]?max_arr[tid]:max_arr[tid+4];
        max_arr[tid] = max_arr[tid]>max_arr[tid+2]?max_arr[tid]:max_arr[tid+2];
        max_arr[tid] = max_arr[tid]>max_arr[tid+1]?max_arr[tid]:max_arr[tid+1];
      }

      __device__ static void warpReduce_min(volatile DType * min_arr,int tid){
        min_arr[tid] = min_arr[tid]<min_arr[tid+32]?min_arr[tid]:min_arr[tid+32];
        min_arr[tid] = min_arr[tid]<min_arr[tid+16]?min_arr[tid]:min_arr[tid+16];
        min_arr[tid] = min_arr[tid]<min_arr[tid+8]?min_arr[tid]:min_arr[tid+8];
        min_arr[tid] = min_arr[tid]<min_arr[tid+4]?min_arr[tid]:min_arr[tid+4];
        min_arr[tid] = min_arr[tid]<min_arr[tid+2]?min_arr[tid]:min_arr[tid+2];
        min_arr[tid] = min_arr[tid]<min_arr[tid+1]?min_arr[tid]:min_arr[tid+1];
      }
      __device__ static void Map(int i,DType *src_max,DType *dst_max,
                                DType *src_min,DType *dst_min,int pre_num){
        //moving pinters
        int tid = threadIdx.x;

        __shared__ DType max_arr[THEAD_PER_BLOCK];
        __shared__ DType min_arr[THEAD_PER_BLOCK];

        //load data into shared memory
        if(2*i+1<pre_num+1){
          max_arr[tid]=*(src_max+2*i)>*(src_max+2*i+1)?*(src_max+2*i):*(src_max+2*i+1);
          min_arr[tid]=*(src_min+2*i)<*(src_min+2*i+1)?*(src_min+2*i):*(src_min+2*i+1);
        } else if(2*i+1==pre_num){
          max_arr[tid]=*(src_max+2*i);
          min_arr[tid]=*(src_min+2*i);
        } else {
          max_arr[tid] = DType(0.);
          min_arr[tid] = DType(0.);
        }
        __syncthreads();
        //call the function
        //compute max/min
        for(int s=blockDim.x/2;s>32;s>>=1){
          if(tid<s){
            max_arr[tid] = max_arr[tid]>max_arr[tid+s]?max_arr[tid]:max_arr[tid+s];
            min_arr[tid] = min_arr[tid]<min_arr[tid+s]?min_arr[tid]:min_arr[tid+s];        
          }
          __syncthreads();
        }
        if(tid<32){
          warpReduce_max(max_arr,tid);
        }
        __syncthreads();
        if(tid<32){
          warpReduce_min(min_arr,tid);
        }
        __syncthreads();
        if(tid==0){
          dst_max[blockIdx.x]=max_arr[0];
          dst_min[blockIdx.x]=min_arr[0];          
        }
        __syncthreads();
      }
    };
  }
}
namespace mshadow{
  template<typename DType>
  void quantization_int8_weight(Tensor<gpu, 3, DType> data,Tensor<gpu, 3, DType> &out,Stream<gpu> *s){
    //find min and max
    int num = out.size(0)*out.size(1)*out.size(2);
    int offset = (num+2*THEAD_PER_BLOCK)/(2*THEAD_PER_BLOCK);
    DType *Temp;
    cudaMalloc((void **)&Temp,sizeof(DType)*offset*4);
    
    //find the max and min first
 
    int current_num = num;
    int pre_num;
    int current_i;
    DType *src_max=data.dptr_;
    DType *src_min=data.dptr_;
    DType *dst_max=Temp;
    DType *dst_min=Temp+offset;
    DType *inter_media;
    bool first_iter = true;

    while(current_num>1){
      //after this iteration num of ele
      pre_num = current_num;
      current_i = (current_num+1)/2;
      
      mxnet::op::mxnet_op::Kernel<mxnet::op::Launch_warper<DType>,gpu>::Launch(s,current_i,
                                                                              src_max,dst_max,
                                                                              src_min,dst_min,
                                                                              pre_num);

      current_num = (current_num+2*THEAD_PER_BLOCK-1)/(THEAD_PER_BLOCK*2);

      if(first_iter){
        src_max = dst_max;
        src_min = dst_min;
        dst_max = Temp + 2*offset;
        dst_min = Temp + 3*offset;
        first_iter=false;
      } else {
        inter_media = src_max;
        src_max = dst_max;
        dst_max = inter_media;
        inter_media = src_min;
        src_min = dst_min;
        dst_min = inter_media;
      }
    }
    //perform quantization
    mxnet::op::mxnet_op::Kernel<mxnet::op::QUANT_WEIGHT_GPU<DType>,gpu>::Launch(s,num,
                                                                    data.dptr_,out.dptr_,
                                                                    src_max,src_min);
    cudaFree(Temp);
  }
  template<typename DType>
  void quantization_int8_act(Tensor<gpu, 3, DType> data,Tensor<gpu, 3, DType> &out,
                             DType *S_act,DType *Temp,
                             DType decay,Stream<gpu> *s,int quant_countdown,bool init){
    int num = out.size(0)*out.size(1)*out.size(2);
    DType *S_act_gpu;
    int offset =  (num+2*THEAD_PER_BLOCK)/(2*THEAD_PER_BLOCK);

    cudaMalloc((void**)&S_act_gpu,sizeof(DType)*2);
    cudaMalloc((void **)&Temp,sizeof(DType)*offset*4);
    
    //find the max and min first
 
    int current_num = num;
    int pre_num;
    int current_i;

    DType *src_max=data.dptr_;
    DType *src_min=data.dptr_;
    DType *dst_max=Temp;
    DType *dst_min=Temp+offset;
    DType *inter_media;
    bool first_iter = true;

    while(current_num>1){
      //after this iteration num of ele
      pre_num = current_num;
      current_i = (current_num+1)/2;
      
      mxnet::op::mxnet_op::Kernel<mxnet::op::Launch_warper<DType>,gpu>::Launch(s,current_i,
                                                                              src_max,dst_max,
                                                                              src_min,dst_min,
                                                                              pre_num);

      current_num = (current_num+2*THEAD_PER_BLOCK-1)/(THEAD_PER_BLOCK*2);

      if(first_iter){
        src_max = dst_max;
        src_min = dst_min;
        dst_max = Temp + 2*offset;
        dst_min = Temp + 3*offset;
        first_iter=false;
      } else {
        inter_media = src_max;
        src_max = dst_max;
        dst_max = inter_media;
        inter_media = src_min;
        src_min = dst_min;
        dst_min = inter_media;
      }
    }

    cudaMemcpy(S_act_gpu,S_act,sizeof(DType)*2,cudaMemcpyHostToDevice);
   
    mxnet::op::mxnet_op::Kernel<mxnet::op::QUANT_ACT_GPU<DType>,gpu>::Launch(s,num,
                                                                    data.dptr_,out.dptr_,
                                                                    S_act_gpu,src_max,src_min,
                                                                    decay,quant_countdown,init);
    cudaMemcpy(S_act,S_act_gpu,sizeof(DType)*2,cudaMemcpyDeviceToHost);
    cudaFree(Temp);
    cudaFree(S_act_gpu);
  }
}

namespace mxnet{
  namespace op{
template<>
Operator *CreateOp<gpu>(Quantization_int8Para param, int dtype) {
  Operator* op = nullptr;
  MSHADOW_REAL_TYPE_SWITCH(dtype, DType, {
    op = new Quantization_int8Op<gpu, DType>(param);
  });
  return op;
}

}  // namespace op
}  // namespace mxnet

