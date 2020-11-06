#/bin/bash
nvcc cuda_test.cu -o cuda_test.cu.o -c -arch=sm_50 -O2 -DGOOGLE_CUDA=1 -x cu -Xcompiler -fPIC

# py27 + TF1.2
#g++ -std=c++11 tf_sampling.cpp tf_sampling_g.cu.o -o tf_sampling_so.so -shared -fPIC -I /home/bo/miniconda2/envs/py27tf12/lib/python2.7/site-packages/tensorflow/include -I /usr/local/cuda-8.0/include -lcudart -L /usr/local/cuda-8.0/lib64/ -O2 -D_GLIBCXX_USE_CXX11_ABI=0

# py27 + TF1.4
#g++ -std=c++11 tf_sampling.cpp tf_sampling_g.cu.o -o tf_sampling_so.so -shared -fPIC -I /home/bo/miniconda2/envs/py27tf14/lib/python2.7/site-packages/tensorflow/include -I /usr/local/cuda-8.0/include -I /home/bo/miniconda2/envs/py27tf14/lib/python2.7/site-packages/tensorflow/include/external/nsync/public -lcudart -L /usr/local/cuda-8.0/lib64/ -L/home/bo/miniconda2/envs/py27tf14/lib/python2.7/site-packages/tensorflow -ltensorflow_framework -O2 -D_GLIBCXX_USE_CXX11_ABI=0

# py3.6 + TF1.2
#g++ -std=c++11 tf_sampling.cpp tf_sampling_g.cu.o -o tf_sampling_so.so -shared -fPIC -I /home/bo/miniconda2/envs/py36tf12/lib/python3.6/site-packages/tensorflow/include -I /usr/local/cuda-8.0/include -lcudart -L /usr/local/cuda-8.0/lib64/ -O2 -D_GLIBCXX_USE_CXX11_ABI=0

# py3.6 + TF1.4
g++ -std=c++11 op.cpp cuda_test.cu.o -o kmeans.so -shared -fPIC -I /home/lee/tensorflow/lib/python3.6/site-packages/tensorflow/include -I /usr/local/cuda/include -I /home/lee/tensorflow/lib/python3.6/site-packages/tensorflow/include/external/nsync/public -lcudart -L /usr/local/cuda/lib64/ -L/home/lee/tensorflow/lib/python3.6/site-packages/tensorflow -ltensorflow_framework -O2 -D_GLIBCXX_USE_CXX11_ABI=0
