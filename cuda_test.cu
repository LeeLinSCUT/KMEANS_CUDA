#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include<stdio.h>
__global__ void k_means_gpu(int b, int n, int num, const float *xyz,  const float *init_xyz, int *result) {    //xyz(b,n,3) result(b,n) init_xyz(b,num,3)
	int batch_idx = blockIdx.x;
	xyz += batch_idx*n*3;
	init_xyz += batch_idx*num*3; 
	result += batch_idx*n;
	float x_c,y_c,z_c,x,y,z;
	int inte = 0;
	extern __shared__ float s[];
	float *temp_dist = s;
	int *ct_idx_old = (int*)&temp_dist[n*num];
	float *ct_xyz =(float*)&ct_idx_old[n];
	float *temp_ct = (float*)&ct_xyz[num*3];
	int *ct_cnt = (int*)&temp_ct[num*3];
	int tid = threadIdx.x;   //test:  num=2  n=1024
	float min_dist = 1e8;
	if(tid<num)
	{
		ct_cnt[tid]=0;
		temp_ct[tid*3]=0;
		temp_ct[tid*3+1]=0;
		temp_ct[tid*3+2]=0;
	}
	if(tid<n)
	{
		ct_idx_old[tid]=0;
	}
	__syncthreads();

	while(inte<3)
	{
		if(tid<num)
		{
			temp_ct[tid*3] += xyz[tid*3];
			temp_ct[tid*3+1] += xyz[tid*3+1];
			temp_ct[tid*3+2] += xyz[tid*3+2];
		}
		for (int j=threadIdx.x;j<n;j+=blockDim.x) //一个点一个点处理
		{
			x = xyz[j*3];
			y = xyz[j*3+1];
			z = xyz[j*3+2];
			for(int i=0;i<num;i+=1) //获得第j个点与第i个中心点的距离
			{
				if(inte == 0)
				{
					x_c = init_xyz[i*3];
					y_c = init_xyz[i*3+1];
					z_c = init_xyz[i*3+2];
				}
				else
				{
					x_c = ct_xyz[i*3];
					y_c = ct_xyz[i*3+1];
					z_c = ct_xyz[i*3+2];
				}
				
				temp_dist[j*num+i] = (x-x_c)*(x-x_c)+(y-y_c)*(y-y_c)+(z-z_c)*(z-z_c);
				if(temp_dist[j*num+i]<min_dist)
				{
					result[j] = i;
					min_dist = temp_dist[j*num+i];
				}	
			} //
			min_dist=1e8;
		}
		if(tid==0)
		{
			for(int ct=0;ct<n;ct++)
			{
				ct_cnt[result[ct]]++;
				temp_ct[result[ct]*3] += xyz[ct*3];
				temp_ct[result[ct]*3+1] += xyz[ct*3+1];
				temp_ct[result[ct]*3+2] += xyz[ct*3+2];
			}
		}
		__syncthreads();
		
		if(tid<num)
		{
			ct_xyz[tid*3] = temp_ct[tid*3]/ct_cnt[tid];
			ct_xyz[tid*3+1] = temp_ct[tid*3+1]/ct_cnt[tid];
			ct_xyz[tid*3+2] = temp_ct[tid*3+2]/ct_cnt[tid];
			temp_ct[tid*3] =0;
			temp_ct[tid*3+1] =0;
			temp_ct[tid*3+2] =0;	
			ct_cnt[tid] = 0;
		}

		/*
		for (int j=threadIdx.x;j<n;j+=blockDim.x) //遍历所有的点,检查是否需要继续迭代
		{
			if(result[j]==ct_idx_old[j])
			{
				cnt=cnt++;
				__syncthreads();
				result[j]=1;
			}
			else
			{
				result[j]=0;
			}
		}*/

		for (int j=threadIdx.x;j<n;j+=blockDim.x) //遍历所有的点
		{
			ct_idx_old[j]=result[j];
		}
		inte++;
	}
}



/*

	for(int i=0;i<num;i+=1) //num个类
	{
		while(p_num!=1)
		{
			if(p_num>512)
			{
				if(temp_dist[i*n+j]>temp_dist[i*m+j+cnt*512])
				{
					temp_dist[i*n+j] = temp_dist[i*m+j+cnt*512];
					idx_dist[j] = i*m+j+cnt*512;
					cnt++;
					p_num -=512;
					
				}
			}
			if(p_num<=512)
			{
				if(j<(p_num/2)
				{
					int stride = p_num/2;
					if(temp_dist[i*n+j]>temp_dist[i*m+j+stride])
					{
						idx_dist[j] = i*m+j+stride;
						temp_dist[i*n+j]= = temp_dist[i*m+j+stride];
						p_num = p_num/2;
					} 
				}
			}
		}
	}

*/
void kmeans(int b, int n,int num, const float *xyz, const float *init_xyz, int *result)  
{
    k_means_gpu<<<b,512,n*num*sizeof(float)+n*sizeof(int)+num*3*sizeof(float)+num*3*sizeof(float)+num*sizeof(int)>>>(b,n,num,xyz,init_xyz,result);
} 