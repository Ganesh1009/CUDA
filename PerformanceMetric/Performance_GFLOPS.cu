#include <stdio.h>

#define n 1024*1024


__global__ void kernel(int a,int *x, int *y)
{
	int i=threadIdx.x+blockIdx.x*blockDim.x;
		y[i]=a*x[i]+y[i];

}

int main(void)
{
  float elapsedTime = 0.0f;
  cudaEvent_t start, stop;
  cudaError_t err=cudaSuccess;
  int *host_a,*host_b;
  host_b = (int *) malloc(n);
  host_a = (int *) malloc(n);
  int *dev_array_a,*dev_array_b;

  cudaMalloc((void **)&dev_array_a, n);
  cudaMalloc((void **)&dev_array_b, n);

  for (int i = 0; i < n/sizeof(int); i++) {
    host_a[i] = i;
  }
  cudaMemcpy(dev_array_a, host_a, n, cudaMemcpyHostToDevice);

  cudaEventCreate(&start);
  cudaEventCreate(&stop);
  cudaEventRecord(start,0);
  kernel<<<256,1024>>>(2,dev_array_a,dev_array_b);
  cudaEventRecord(stop,0);
  cudaEventSynchronize(stop); //Wait till the event is executed.
  cudaMemcpy(host_b, dev_array_b, n, cudaMemcpyDeviceToHost);
  cudaEventElapsedTime(&elapsedTime,start,stop);
  printf("Time for kernel to exexute:%fms\n",elapsedTime);
  printf("Arithmetic Performance = %5f Gflops/s\n\n", n * 1e-6/elapsedTime);
  cudaEventDestroy(start);
  cudaEventDestroy(stop);

  err=cudaGetLastError();
	if(err!=cudaSuccess) {
		fprintf(stderr,"Error executing the kernel - %s\n",
				 cudaGetErrorString(err));
		exit(EXIT_FAILURE);
	}


}
