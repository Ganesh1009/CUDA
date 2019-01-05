#include <stdio.h>
#include <stdlib.h>

#define ONE_MB 1024*1024
#define SIXTEEN_MB 16*1024*1024
#define FOUR_MB 4*1024*1024



__global__ void testKernel1 (){

}

__global__ void testKernel2 (int *array1, int *resultArray){

  int index = threadIdx.x + blockIdx.x*blockDim.x;
  resultArray[index] = array1[index];
}

__global__ void testKernel3(int *c_array, int *transpose_array, int width){

  int index = threadIdx.x + blockIdx.x * blockDim.x;
  transpose_array[((int)(index/width) + ((index%width)*(width)))] = c_array[index];
}


int main() {
  float time_in_ms = 0.0f;
  cudaEvent_t start, stop;
  cudaError_t err=cudaSuccess;
  //===========TASK 1============================
  cudaEventCreate(&start);
  cudaEventCreate(&stop);
  cudaEventRecord(start,0);
  testKernel1<<<1,1>>>();
  cudaEventRecord(stop,0);
  cudaEventSynchronize(stop); //Wait till the event is executed.
  cudaEventElapsedTime(&time_in_ms,start,stop);
  printf("Time for empty kernel to exexute:%fms\n\n",time_in_ms);
  cudaEventDestroy(start);
  cudaEventDestroy(stop);
  //========TASK 1 end============================

  //=======TASK 2=================================
  int *array1 = (int *) malloc(ONE_MB);
  int *c_array;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);

  for (int i = 0; i < ONE_MB/sizeof(int); i++) {
    array1[i] = i;
  }

  cudaMalloc((void **)&c_array, ONE_MB);
  cudaEventRecord(start,0);
  cudaMemcpy(c_array, array1, ONE_MB, cudaMemcpyHostToDevice);
  cudaEventRecord(stop,0);
  cudaEventSynchronize(stop);
  cudaEventElapsedTime(&time_in_ms,start,stop);
  printf("Time for memory copy of %d bytes is :%fms\n\n",ONE_MB,time_in_ms);
  cudaEventDestroy(start);
  cudaEventDestroy(stop);
  cudaFree(c_array);
  //==============TASK 2 end======================

  //=============TASK 3===========================
  int *arrayCopy = (int *) malloc(ONE_MB),*c_arrayCopy;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);
  cudaMalloc((void **)&c_array, ONE_MB);
  cudaMalloc((void **)&c_arrayCopy, ONE_MB);
  cudaMemcpy(c_array, array1, ONE_MB, cudaMemcpyHostToDevice);
  //testKernel2<<<256,1024>>>(c_array,c_arrayCopy);
  cudaEventRecord(start,0);
  cudaMemcpy(arrayCopy, c_arrayCopy, ONE_MB, cudaMemcpyDeviceToHost);
  cudaEventRecord(stop,0);
  cudaEventSynchronize(stop);
  time_in_ms = 0.0f;
  cudaEventElapsedTime(&time_in_ms,start,stop);
  printf("Time to copy %d bytes from device to host = %fms\n",ONE_MB,time_in_ms);
  printf("Effective bandwidth = %f GB/s \n\n", ONE_MB/time_in_ms/1e6);
  cudaFree(c_array);
  cudaFree(c_arrayCopy);
  free(array1);
  free(arrayCopy);
  cudaEventDestroy(start);
  cudaEventDestroy(stop);
  //============TASK 3 end==================================

  //============TASK 4====================================
  arrayCopy = (int *) malloc(SIXTEEN_MB);
  array1 = (int *) malloc(SIXTEEN_MB);
  for (int i = 0; i < (SIXTEEN_MB)/sizeof(int); i++) {
    array1[i] = i;
  }
  cudaEventCreate(&start);
  cudaEventCreate(&stop);
  cudaMalloc((void **)&c_array, 16*ONE_MB);
  cudaMalloc((void **)&c_arrayCopy, 16*ONE_MB);
  cudaEventRecord(start,0);
  cudaMemcpy(c_array, array1, SIXTEEN_MB, cudaMemcpyHostToDevice);
  //testKernel2<<<4096,1024>>>(c_array,c_arrayCopy);
  cudaMemcpy(arrayCopy, c_arrayCopy, SIXTEEN_MB, cudaMemcpyDeviceToHost);
  cudaEventRecord(stop,0);
  cudaEventSynchronize(stop);
  time_in_ms = 0.0f;
  cudaEventElapsedTime(&time_in_ms,start,stop);
  printf("Time to copy %d bytes from device to host = %fms\n",SIXTEEN_MB,time_in_ms);
  printf("Effective bandwidth = %f GB/s \n\n", (SIXTEEN_MB)/time_in_ms/1e6);
  cudaFree(c_array);
  cudaFree(c_arrayCopy);
  free(array1);
  free(arrayCopy);
  cudaEventDestroy(start);
  cudaEventDestroy(stop);
  //===================TASK 4 end========================

  //===================TASK 5============================
  int *c_array1,*c_array2, number_of_steams = 2;
  int *resultArray1, *resultArray2;

  cudaMallocHost((void **)&resultArray1, SIXTEEN_MB);
  cudaMallocHost((void **)&resultArray2, SIXTEEN_MB);
  cudaMallocHost((void **)&array1, SIXTEEN_MB);

  cudaMalloc((void **)&c_array1, SIXTEEN_MB);
  cudaMalloc((void **)&c_array2, SIXTEEN_MB);

  for (int i = 0; i < (SIXTEEN_MB)/sizeof(int); i++) {
    array1[i] = i;
  }
  cudaEventCreate(&start);
  cudaEventCreate(&stop);

  cudaStream_t stream[number_of_steams];
  for (int i = 0; i < number_of_steams; i++) {
    cudaStreamCreate(&stream[i]);
  }
  cudaEventRecord(start,0);
  //for(int i=0;i<number_of_steams;i++){
    cudaMemcpyAsync(c_array1,array1,SIXTEEN_MB,cudaMemcpyHostToDevice,stream[0]);
    cudaMemcpyAsync(c_array2,array1,SIXTEEN_MB,cudaMemcpyHostToDevice,stream[1]);
  //}
  cudaMemcpyAsync(resultArray1,c_array1,SIXTEEN_MB,cudaMemcpyDeviceToHost,stream[0]);
  cudaMemcpyAsync(resultArray2,c_array2,SIXTEEN_MB,cudaMemcpyDeviceToHost,stream[1]);
  cudaEventRecord(stop,0);
  cudaEventSynchronize(stop);
  time_in_ms = 0.0;
  cudaEventElapsedTime(&time_in_ms,start,stop);
  printf("Time for async data transfer of 2 arrays each of %d bytes multiple arrays is = %fms\n",SIXTEEN_MB, time_in_ms);
  printf("Bandwidth for async data transfer of 2 arrays is : %f \n\n",(SIXTEEN_MB)/time_in_ms/1e6);
  cudaFree(c_array1);
  cudaFree(c_array2);
  cudaFreeHost(resultArray1);
  cudaFreeHost(resultArray2);
  cudaFreeHost(array1);
  cudaEventDestroy(start);
  cudaEventDestroy(stop);

  //==================TASK 5 end=========================


  //=================TASK 6================================
  cudaEventCreate(&start);
  cudaEventCreate(&stop);
  int *array4mb = (int *) malloc(FOUR_MB*sizeof(int));
  int width = 1024;
  for(int i=0; i<FOUR_MB; i++){
    array4mb[i] = i;
  }

  int *cuda_array_4mb, *transpose_4MB;
  cudaMalloc((void **)&cuda_array_4mb,FOUR_MB*(sizeof(int)));
  cudaMalloc((void **)&transpose_4MB,FOUR_MB*(sizeof(int)));
  cudaMemcpy(cuda_array_4mb, array4mb, FOUR_MB*(sizeof(int)), cudaMemcpyHostToDevice);
  cudaEventRecord(start,0);
  testKernel3<<<4096,1024>>>(cuda_array_4mb, transpose_4MB,width);
  cudaEventRecord(stop,0);
  cudaEventSynchronize(stop);
  cudaMemcpy(array4mb, transpose_4MB, FOUR_MB*(sizeof(int)), cudaMemcpyDeviceToHost);
  time_in_ms = 0.0;
  cudaEventElapsedTime(&time_in_ms,start,stop);
  printf("Bandwidth for array of dimension %d is %fGFLOPS\n\n",FOUR_MB,(FOUR_MB)/time_in_ms/1e6);
  //=================Task 6 end============================



  err=cudaGetLastError();
	if(err!=cudaSuccess) {
		fprintf(stderr,"Error executing the kernel - %s\n",
				 cudaGetErrorString(err));
		exit(EXIT_FAILURE);
	}

  cudaDeviceProp deviceProperties;
  cudaGetDeviceProperties(&deviceProperties,0);
  printf("Device name: %s\n", deviceProperties.name);
  printf("Memory Clock Rate (KHz): %d\n",deviceProperties.memoryClockRate);
  printf("Memory Bus Width (bits): %d\n",deviceProperties.memoryBusWidth);
  printf("Peak Memory Bandwidth (GB/s): %f\n",
    2.0*deviceProperties.memoryClockRate*(deviceProperties.memoryBusWidth/8)/1.0e6);
  printf("Registers per multiprocessors: %d\n\n",deviceProperties.regsPerMultiprocessor);
}
