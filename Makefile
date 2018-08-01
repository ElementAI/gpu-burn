#CUDAPATH=/usr/local/cuda
#
## Have this point to an old enough gcc (for nvcc)
#GCCPATH=/usr
#
#NVCC=nvcc
#CCPATH=${GCCPATH}/bin
#
#drv:
#	PATH=${PATH}:.:${CCPATH}:${PATH} ${NVCC} -I${CUDAPATH}/include -arch=compute_30 -ptx compare.cu -o compare.ptx
#	g++ -O3 -Wno-unused-result -I${CUDAPATH}/include -c gpu_burn-drv.cpp
#	g++ -o gpu_burn gpu_burn-drv.o -O3 -lcuda -L${CUDAPATH}/lib64 -L${CUDAPATH}/lib -Wl,-rpath=${CUDAPATH}/lib64 -Wl,-rpath=${CUDAPATH}/lib -lcublas -lcudart -o gpu_burn

# If the CUDA environment variables are already set, use them
CUDA_PATH?=${CUDA_HOME}

# If the above did not find anything, then use the default path to
# NVIDIA CUDA (typically installed into /usr/local/cuda)
CUDA_PATH?=/usr/local/cuda

# If none of the above worked, try some other common paths
ifneq ("$(wildcard $(CUDA_PATH))","")
# The default PATH is good - nothing else to do
else ifneq ("$(wildcard /usr/local/cuda-9.0)","")
	echo "Cuda9-0 detected"
	CUDA_PATH=/usr/local/cuda-9.0
	NVCCFLAGS=-arch=sm_30 \
	-gencode=arch=compute_30,code=sm_30 \
    -gencode=arch=compute_50,code=sm_50 \
    -gencode=arch=compute_52,code=sm_52 \
    -gencode=arch=compute_60,code=sm_60 \
    -gencode=arch=compute_61,code=sm_61 \
    -gencode=arch=compute_62,code=sm_62 \
    -gencode=arch=compute_70,code=sm_70 \
    -gencode=arch=compute_70,code=compute_70

else ifneq ("$(wildcard /usr/local/cuda-8.0)","")
	CUDA_PATH=/usr/local/cuda-8.0
	NVCCFLAGS=-arch=sm_30 \
	-gencode=arch=compute_30,code=sm_30 \
	-gencode=arch=compute_50,code=sm_50 \
	-gencode=arch=compute_52,code=sm_52 \
	-gencode=arch=compute_60,code=sm_60 \
	-gencode=arch=compute_61,code=sm_61 \
	-gencode=arch=compute_62,code=sm_62

else ifneq ("$(wildcard /usr/local/cuda-7.5)","")
	CUDA_PATH=/usr/local/cuda-7.5
	NVCCFLAGS=-arch=sm_30 \
	-gencode=arch=compute_30,code=sm_30 \
	-gencode=arch=compute_50,code=sm_50 \
	-gencode=arch=compute_52,code=sm_52

else ifneq ("$(wildcard /usr/local/cuda-7.0)","")
	CUDA_PATH=/usr/local/cuda-7.0
	CUDA_PATH=/usr/local/cuda-7.5
	NVCCFLAGS=-arch=sm_30 \
	-gencode=arch=compute_30,code=sm_30 \
	-gencode=arch=compute_50,code=sm_50 \
	-gencode=arch=compute_52,code=sm_52

else ifneq ("$(wildcard /opt/cuda)","")
	CUDA_PATH=/opt/cuda

endif



# Have this point to an old enough gcc (for nvcc)
GCCPATH=/usr


NVCC=${CUDA_PATH}/bin/nvcc
CCPATH=${GCCPATH}/bin

all: gpu_burn

gpu_burn.cuda_kernel: compare.cu Makefile
	PATH=.:${CCPATH}:${PATH} ${NVCC} ${NVCCFLAGS} -I. -I${CUDA_PATH}/include --fatbin compare.cu -o $@

gpu_burn: gpu_burn.cuda_kernel gpu_burn-drv.cpp
	g++ -o gpu_burn gpu_burn-drv.cpp -O3 -I${CUDA_PATH}/include -lcuda -L${CUDA_PATH}/lib64 -L${CUDA_PATH}/lib -Wl,-rpath=${CUDA_PATH}/lib64 -Wl,-rpath=${CUDA_PATH}/lib -lcublas -lcudart


clean:
	rm -f gpu_burn.cuda_kernel gpu_burn
