#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/shape_inference.h"
#include "tensorflow/core/framework/common_shape_fns.h"
#include <cuda_runtime.h>

using namespace tensorflow;

REGISTER_OP("Kmeans")
    .Input("xyz: float32") //(b,n,3)
    .Input("init_xyz: float32")//(b,num,3)
    .Output("result: int32") //(b,n)
    .SetShapeFn([](::tensorflow::shape_inference::InferenceContext* c) {
        ::tensorflow::shape_inference::ShapeHandle dims1; // batch_size * npoint * 3
        c->WithRank(c->input(0), 3, &dims1);
        ::tensorflow::shape_inference::ShapeHandle output1 = c->MakeShape({c->Dim(dims1, 0), c->Dim(dims1, 1)});
        c->set_output(0, output1);
        return Status::OK();
    });
void kmeans(int b, int n,int num, const float *xyz, const float *init_xyz, int *result);  
class kmeansGpuOp: public OpKernel{
  public:
    explicit kmeansGpuOp(OpKernelConstruction* context):OpKernel(context) {   //OpKernelContext是作为OpKernel的核心API Compute函数的参数，所有计算相关的参数都会包含在这个对象中。
    }
    void Compute(OpKernelContext * context)override{
      const Tensor& inp_tensor=context->input(0); //(b,n,3)
      int b=inp_tensor.shape().dim_size(0);  //1
      int n=inp_tensor.shape().dim_size(1);  //2048
      auto inp_flat=inp_tensor.flat<float>();
      const float * xyz=&(inp_flat(0)); 

      const Tensor& inp_tensor1=context->input(1); //(b,num)
      int num=inp_tensor1.shape().dim_size(1);  //nsamples
      auto inp_flat1=inp_tensor1.flat<float>();
      const float * init_xyz=&(inp_flat1(0)); 

      Tensor * out_tensor;
      OP_REQUIRES_OK(context,context->allocate_output(0,TensorShape{b,n},&out_tensor)); //这里申请输出变量output (b,nsamples,3)
      auto out_flat=out_tensor->flat<int>();
      int * out=&(out_flat(0));
      kmeans(b,n,num,xyz,init_xyz,out);  //inp:2048*8*3  out:(1,2048)
    }
};
REGISTER_KERNEL_BUILDER(Name("Kmeans").Device(DEVICE_GPU), kmeansGpuOp);