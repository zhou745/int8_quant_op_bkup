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
    struct QUANT_WEIGHT_GPU_MINMAX{
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
    struct QUANT_WEIGHT_GPU_POWER2{
      __device__ static void Map(int i,DType *data,DType *out,
                                 DType log2t){

        __shared__ DType quant_unit;

        int tidx=threadIdx.x;
        //compute quantization inside each block
        if(tidx<1){
          quant_unit=(::pow(2.0,::ceil(log2t))*DType(2.0))/DType(QUANT_LEVEL);
        }

        __syncthreads();
        DType int8_val=DType(floor(*(data+i)/quant_unit+0.5));
        int8_val=int8_val>DType(QUANT_LEVEL/2-1)?DType(QUANT_LEVEL/2-1):int8_val;
        int8_val=int8_val<-DType(QUANT_LEVEL/2)?-DType(QUANT_LEVEL/2):int8_val;
        *(out+i)=int8_val*quant_unit;
      }
    };

    template<typename DType>
    struct QUANT_ACT_GPU_MINMAX{
      __device__ static void Map(int i,DType *data,DType *out,DType *S_act,
                                 int quant_countdown,bool is_train){
        DType S_max_f=*S_act;
        DType S_min_f=*(S_act+1);
        DType quant_unit;

        if(quant_countdown==0||~is_train){
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
    struct UPDATE_MINMAX{
      __device__ static void Map(int i,DType *S_act,DType *max_S,DType *min_S,
                                 DType decay,bool init,bool is_train){
        DType S_max_f=*S_act;
        DType S_min_f=*(S_act+1);

        if(is_train){
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
          *S_act = S_max_f;
          *(S_act+1) = S_min_f;
        }
        
      }
    };

    template<typename DType>
    struct REDUCE_MINMAX{ 
      __device__ static void warpReduce_max(volatile DType * max_arr,int tid,int i,int pre_num){
        if(2*(i+32)<pre_num){max_arr[tid] = max_arr[tid]>max_arr[tid+32]?max_arr[tid]:max_arr[tid+32];}
        if(2*(i+16)<pre_num){max_arr[tid] = max_arr[tid]>max_arr[tid+16]?max_arr[tid]:max_arr[tid+16];}
        if(2*(i+8)<pre_num){max_arr[tid] = max_arr[tid]>max_arr[tid+8]?max_arr[tid]:max_arr[tid+8];}
        if(2*(i+4)<pre_num){max_arr[tid] = max_arr[tid]>max_arr[tid+4]?max_arr[tid]:max_arr[tid+4];}
        if(2*(i+2)<pre_num){max_arr[tid] = max_arr[tid]>max_arr[tid+2]?max_arr[tid]:max_arr[tid+2];}
        if(2*(i+1)<pre_num){max_arr[tid] = max_arr[tid]>max_arr[tid+1]?max_arr[tid]:max_arr[tid+1];}
      }

      __device__ static void warpReduce_min(volatile DType * min_arr,int tid,int i,int pre_num){
        if(2*(i+32)<pre_num){min_arr[tid] = min_arr[tid]<min_arr[tid+32]?min_arr[tid]:min_arr[tid+32];}
        if(2*(i+16)<pre_num){min_arr[tid] = min_arr[tid]<min_arr[tid+16]?min_arr[tid]:min_arr[tid+16];}
        if(2*(i+8)<pre_num){min_arr[tid] = min_arr[tid]<min_arr[tid+8]?min_arr[tid]:min_arr[tid+8];}
        if(2*(i+4)<pre_num){min_arr[tid] = min_arr[tid]<min_arr[tid+4]?min_arr[tid]:min_arr[tid+4];}
        if(2*(i+2)<pre_num){min_arr[tid] = min_arr[tid]<min_arr[tid+2]?min_arr[tid]:min_arr[tid+2];}
        if(2*(i+1)<pre_num){min_arr[tid] = min_arr[tid]<min_arr[tid+1]?min_arr[tid]:min_arr[tid+1];}
      }
      __device__ static void Map(int i,DType *src_max,DType *dst_max,
                                DType *src_min,DType *dst_min,int pre_num){
        //moving pinters
        int tid = threadIdx.x;

        __shared__ DType max_arr[THEAD_PER_BLOCK];
        __shared__ DType min_arr[THEAD_PER_BLOCK];

        //load data into shared memory
        if(2*i+1<pre_num){
          max_arr[tid]=*(src_max+2*i)>*(src_max+2*i+1)?*(src_max+2*i):*(src_max+2*i+1);
          min_arr[tid]=*(src_min+2*i)<*(src_min+2*i+1)?*(src_min+2*i):*(src_min+2*i+1);
        } else {
          max_arr[tid]=*(src_max+2*i);
          min_arr[tid]=*(src_min+2*i);
        }

        //dst_max[blockIdx.x*THEAD_PER_BLOCK+tid]=max_arr[tid];
        //dst_min[blockIdx.x*THEAD_PER_BLOCK+tid]=min_arr[tid]; 
        
        __syncthreads();
        //call the function
        //compute max/min
        for(int s=blockDim.x/2;s>0;s>>=1){
          if(tid<s&&2*(i+s)<pre_num){
            max_arr[tid] = max_arr[tid]>max_arr[tid+s]?max_arr[tid]:max_arr[tid+s];
            min_arr[tid] = min_arr[tid]<min_arr[tid+s]?min_arr[tid]:min_arr[tid+s];        
          }
          __syncthreads();
        }

        /*
        if(tid<32){
          warpReduce_max(max_arr,tid,i,pre_num);
        }
       
        if(tid<32){
          warpReduce_min(min_arr,tid,i,pre_num);
        }*/
        
        if(tid==0){
          dst_max[blockIdx.x]=max_arr[0];
          dst_min[blockIdx.x]=min_arr[0];          
        }
        
        
      }
    };

    template<typename DType>
    struct GRAD_POWER2{
      __device__ static void Map(int i,DType *data,DType *gdata,DType *out,DType log2t){
        __shared__ DType quant_unit;

        int tidx=threadIdx.x;
        //compute quantization inside each block
        if(tidx<1){
          quant_unit=(::pow(2.0,::ceil(log2t))*DType(2.0))/DType(QUANT_LEVEL);
        }
        __syncthreads();

        DType int8_val=DType(floor(*(data+i)/quant_unit+0.5));
        int8_val=int8_val>DType(QUANT_LEVEL/2-1)?DType(QUANT_LEVEL/2-1):int8_val;
        int8_val=int8_val<-DType(QUANT_LEVEL/2)?-DType(QUANT_LEVEL/2):int8_val;

        DType local_grad=logf(2.0)*quant_unit*int8_val;
        
        *(out+i)=*(gdata+i)*local_grad;     
      }
    };

    template<typename DType>
    struct GRAD_WEIGHT_POWER2{
      __device__ static void Map(int i,DType *data,DType *gdata,DType *out,DType log2t){
        __shared__ DType quant_unit;

        int tidx=threadIdx.x;
        //compute quantization inside each block
        if(tidx<1){
          quant_unit=(::pow(2.0,::ceil(log2t))*DType(2.0))/DType(QUANT_LEVEL);
        }
        __syncthreads();

        DType int8_val=DType(floor(*(data+i)/quant_unit+0.5));
        DType factor=int8_val>DType(QUANT_LEVEL/2-1)?DType(0.):DType(1.);
        factor=int8_val<-DType(QUANT_LEVEL/2)?DType(0.):factor;
        
        *(out+i)=*(gdata+i)*factor;     
      }
    };
    
    template<typename DType>
    struct UPDATE_LOG2T{
      __device__ static void Map(int i,DType *log2t,DType grad){
        DType norm=grad/DType(::abs(grad)+1e-3);        
        *(log2t)-=1e-3*norm;
      }
    };
    template<typename DType>
    struct REDUCE_POWER2{ 
      __device__ static void Map(int i,DType *grad_src,DType *grad_dst,int pre_num){
        //moving pinters
        int tid = threadIdx.x;

        __shared__ DType sum_grad[THEAD_PER_BLOCK];


        //load data into shared memory
        if(2*i+1<pre_num){
          sum_grad[tid]=*(grad_src+2*i)+*(grad_src+2*i+1);
        } else if(2*i+1==pre_num){
          sum_grad[tid]=*(grad_src+2*i);
        } else{
          sum_grad[tid]=DType(0.);
        }
        
        __syncthreads();
        //call the function
        //compute max/min
        for(int s=blockDim.x/2;s>0;s>>=1){
          if(tid<s&&2*(i+s)<pre_num){
            sum_grad[tid] = sum_grad[tid]+sum_grad[tid+s];
          }
          __syncthreads();
        }   
        if(tid==0){
          grad_dst[blockIdx.x]=sum_grad[0];         
        }
      }
    };
  }
}
namespace mshadow{
  template<typename DType>
  void quantization_int8_weight(std::string qmod,
                                Tensor<gpu, 3, DType> data,Tensor<gpu, 3, DType> &out,
                                Tensor<gpu, 1, DType> aux,
                                Stream<gpu> *s){
    //find min and max
    int num = out.size(0)*out.size(1)*out.size(2);

    //choose quantization path
    if(qmod==std::string("minmax")){
      //declare space for reduction
      int offset = (num+2*THEAD_PER_BLOCK)/(2*THEAD_PER_BLOCK);
      DType *Temp;
      cudaMalloc((void **)&Temp,sizeof(DType)*offset*4);

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
        
        mxnet::op::mxnet_op::Kernel<mxnet::op::REDUCE_MINMAX<DType>,gpu>::Launch(s,current_i,
                                                                                src_max,dst_max,
                                                                                src_min,dst_min,
                                                                                pre_num);
  
        current_num = (current_num+2*THEAD_PER_BLOCK-1)/(THEAD_PER_BLOCK*2);
        //current_num=current_i;
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
      mxnet::op::mxnet_op::Kernel<mxnet::op::QUANT_WEIGHT_GPU_MINMAX<DType>,gpu>::Launch(s,num,
                                                                                         data.dptr_,out.dptr_,
                                                                                         src_max,src_min);
      cudaFree(Temp);
    } else if(qmod==std::string("power2")){
      mxnet::op::mxnet_op::Kernel<mxnet::op::QUANT_WEIGHT_GPU_POWER2<DType>,gpu>::Launch(s,num,
                                                                                         data.dptr_,out.dptr_,
                                                                                         aux[0]);
    }
  }
  template<typename DType>
  void quantization_int8_act(std::string qmod,
                             Tensor<gpu, 3, DType> data,Tensor<gpu, 3, DType> &out,
                             Tensor<gpu, 1, DType> aux,
                             DType decay,Stream<gpu> *s,int quant_countdown,
                             bool init,bool is_train){

    int num = out.size(0)*out.size(1)*out.size(2);
    if(qmod==std::string("minmax")){
      int offset =  (num+2*THEAD_PER_BLOCK)/(2*THEAD_PER_BLOCK);
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
        
        mxnet::op::mxnet_op::Kernel<mxnet::op::REDUCE_MINMAX<DType>,gpu>::Launch(s,current_i,
                                                                                src_max,dst_max,
                                                                                src_min,dst_min,
                                                                                pre_num);
  
        current_num = (current_num+2*THEAD_PER_BLOCK-1)/(THEAD_PER_BLOCK*2);
        //current_num=current_i;
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
      mxnet::op::mxnet_op::Kernel<mxnet::op::UPDATE_MINMAX<DType>,gpu>::Launch(s,1,
                                                                      aux.dptr_,src_max,src_min,
                                                                      decay,init,is_train);
      
      mxnet::op::mxnet_op::Kernel<mxnet::op::QUANT_ACT_GPU_MINMAX<DType>,gpu>::Launch(s,num,
                                                                      data.dptr_,out.dptr_,
                                                                      aux.dptr_,
                                                                      quant_countdown,is_train);
      cudaFree(Temp);
    } else if(qmod==std::string("power2")){
      mxnet::op::mxnet_op::Kernel<mxnet::op::QUANT_WEIGHT_GPU_POWER2<DType>,gpu>::Launch(s,num,
                                                                                         data.dptr_,out.dptr_,
                                                                                         aux[0]);
    }
  }

  template<typename DType>
  void quantization_grad(std::string qmod,
                         Tensor<gpu, 3, DType> gdata,Tensor<gpu, 3, DType> &grad,
                         Tensor<gpu, 3, DType> data,Tensor<gpu, 1, DType> &aux,
                         Stream<gpu> *s){
  
  int num = grad.size(0)*grad.size(1)*grad.size(2);
  int offset = (num+2*THEAD_PER_BLOCK)/(2*THEAD_PER_BLOCK);
  DType *Temp;
  cudaMalloc((void **)&Temp,sizeof(DType)*offset*2);
  //compute gradient for threash hold
  mxnet::op::mxnet_op::Kernel<mxnet::op::GRAD_POWER2<DType>,gpu>::Launch(s,num,
                                                                         data.dptr_,gdata.dptr_,
                                                                         grad.dptr_,aux[0]);
  //reduce gradient for threash hold
  int current_num = num;
  int pre_num;
  int current_i;
  DType *src_grad=grad.dptr_;
  DType *dst_grad=Temp;
  DType *inter_media;
  bool first_iter = true;
  
  while(current_num>1){
    //after this iteration num of ele
    pre_num = current_num;
    current_i = (current_num+1)/2;
    
    mxnet::op::mxnet_op::Kernel<mxnet::op::REDUCE_POWER2<DType>,gpu>::Launch(s,current_i,
                                                                             src_grad,dst_grad,
                                                                             pre_num);

    current_num = (current_num+2*THEAD_PER_BLOCK-1)/(THEAD_PER_BLOCK*2);
    //current_num=current_i;
    if(first_iter){
      src_grad = dst_grad;
      dst_grad = Temp + offset;
      first_iter=false;
    } else {
      inter_media = src_grad;
      src_grad = dst_grad;
      dst_grad = inter_media;
    }
  }
  //compute grad
  mxnet::op::mxnet_op::Kernel<mxnet::op::GRAD_WEIGHT_POWER2<DType>,gpu>::Launch(s,num,
                                                                                data.dptr_,gdata.dptr_,
                                                                                grad.dptr_,aux[0]);
  //update aux
  mxnet::op::mxnet_op::Kernel<mxnet::op::UPDATE_LOG2T<DType>,gpu>::Launch(s,1,
                                                                          aux.dptr_,
                                                                          src_grad[0]);

  cudaFree(Temp);
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

