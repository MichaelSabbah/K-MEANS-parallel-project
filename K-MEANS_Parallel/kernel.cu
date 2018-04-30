
#include "structsAndFunctions.h"
#include <string.h>

#define NUM_BLOCKS 1


//the functions for the k-means algorithm

__device__ double calculateDistanceCuda(double* p1Weeks,double* p2Weeks,int numberOfWeeks)
{
	int i;
	double distanceBeforeSqurt=0;
	for(i=0;i<numberOfWeeks;i++)
	{
		distanceBeforeSqurt += ((p2Weeks[i] - p1Weeks[i]) * (p2Weeks[i] - p1Weeks[i]));
	}
	return sqrt(distanceBeforeSqurt);
}

product_t* allocateAndCopyProductsArrayInGpu(product_t* products,int numOfProducts,int numberOfWeeks)
{
	int i;
	product_t* dev_products = 0;
	product_t* host_tempProducts = 0;
	double* tempWeeks;
	cudaError_t cudaStatus;

	cudaStatus = cudaSetDevice(0);
	cudaCheaking(cudaStatus);

	cudaStatus = cudaMalloc((void**)&dev_products, numOfProducts*sizeof(product_t));
	cudaCheaking(cudaStatus);

	cudaStatus = cudaMemcpy(dev_products,products, numOfProducts* sizeof(product_t), cudaMemcpyHostToDevice);
	cudaCheaking(cudaStatus);

	for(i=0;i<numOfProducts;i++)
	{
		cudaStatus = cudaMalloc((void**)&tempWeeks, numberOfWeeks*sizeof(double));
		cudaCheaking(cudaStatus);

		cudaStatus = cudaMemcpy(tempWeeks,products[i].weeks, numberOfWeeks* sizeof(double), cudaMemcpyHostToDevice);
		cudaCheaking(cudaStatus);

		cudaStatus = cudaMemcpy(&dev_products[i].weeks,&tempWeeks, sizeof(double*), cudaMemcpyHostToDevice);
		cudaCheaking(cudaStatus);
	}
	return dev_products;
}

cluster_t* allocateAndCopyClustersArrayInGpu(cluster_t* clusters,int numOfClusters,int numberOfWeeks)
{
	int i;
	cluster_t* dev_clusters = 0;
	cluster_t* host_tempClusters = 0;
	double* tempWeeks;
	cudaError_t cudaStatus;

	cudaStatus = cudaSetDevice(0);
	cudaCheaking(cudaStatus);
	cudaStatus = cudaMalloc((void**)&dev_clusters, numOfClusters*sizeof(cluster_t));
	cudaCheaking(cudaStatus);

	cudaStatus = cudaMemcpy(dev_clusters,clusters, numOfClusters * sizeof(cluster_t), cudaMemcpyHostToDevice);
	cudaCheaking(cudaStatus);

	for(i=0;i<numOfClusters;i++)
	{
		cudaStatus = cudaMalloc((void**)&tempWeeks, numberOfWeeks*sizeof(double));
		cudaCheaking(cudaStatus);

		cudaStatus = cudaMemcpy(tempWeeks,clusters[i].centeroid.weeks, numberOfWeeks* sizeof(double), cudaMemcpyHostToDevice);
		cudaCheaking(cudaStatus);

		cudaStatus = cudaMemcpy(&dev_clusters[i].centeroid.weeks,&tempWeeks, sizeof(double*), cudaMemcpyHostToDevice);
		cudaCheaking(cudaStatus);
	}

	return dev_clusters;
}

void copyClustersArrayFromGpu(cluster_t* clusters,cluster_t* dev_clusters,int numOfClusters,int numOfWeeks)
{
	int i;
	double** temp = (double**)malloc(sizeof(double*) * numOfClusters);
	cudaError_t cudaStatus;

	cudaStatus = cudaSetDevice(0);
	cudaCheaking(cudaStatus);

	for (i = 0; i < numOfClusters; i++)
		temp[i] = clusters[i].centeroid.weeks;

	cudaStatus = cudaMemcpy(clusters, dev_clusters, sizeof(cluster_t) * numOfClusters, cudaMemcpyDeviceToHost);
	cudaCheaking(cudaStatus);

	for(i = 0; i < numOfClusters; i++)
	{
		cudaStatus = cudaMemcpy(temp[i], clusters[i].centeroid.weeks, sizeof(double) * numOfWeeks, cudaMemcpyDeviceToHost);
		cudaCheaking(cudaStatus);

		cudaFree(clusters[i].centeroid.weeks);

		clusters[i].centeroid.weeks = temp[i];
	}

	cudaFree(dev_clusters);
	free(temp);
}

__global__ void classifyProduct(product_t* dev_products,cluster_t* dev_clusters,int numOfProducts,int numOfClusters,int numberOfWeeks)
{
		int i = blockIdx.x * blockDim.x + threadIdx.x;
		int j , currentClusterNum , prevClusterNum;
		double currentMinDistance,distance;
		
		if(i >= numOfProducts)
			return;

		currentClusterNum = 0;
		currentMinDistance = calculateDistanceCuda(dev_clusters[0].centeroid.weeks,dev_products[i].weeks,numberOfWeeks);
		for(j=1;j<numOfClusters;j++)
		{
			distance = calculateDistanceCuda(dev_clusters[j].centeroid.weeks,dev_products[i].weeks,numberOfWeeks);
			if(distance < currentMinDistance)
			{
				currentMinDistance = distance;
				currentClusterNum = j;
			}
		}
		dev_products[i].prevCluster = dev_products[i].currentCluster;
		dev_products[i].currentCluster = currentClusterNum;
}

product_t* classifyProductsToClustersWithCuda(product_t* dev_products,int numOfProducts,int numOfClusters,cluster_t* clusters,int numberOfWeeks)
{
	int i,j;
	int blocksNum,blockSize;
	cluster_t* dev_clusters =0;
	product_t* tempProducts=0;
	cudaError_t cudaStatus;
	cudaDeviceProp prop;

    // Choose which GPU to run on, change this on a multi-GPU system.
    cudaStatus = cudaSetDevice(0);
	cudaCheaking(cudaStatus);

	//get device properties
	cudaStatus = cudaGetDeviceProperties(&prop,0);
    cudaCheaking(cudaStatus);

	blockSize = prop.maxThreadsPerBlock;
	blocksNum = numOfProducts/blockSize;
	if(blocksNum % numOfProducts != 0 || blocksNum == 0)
		blocksNum++;

	dev_clusters = allocateAndCopyClustersArrayInGpu(clusters,numOfClusters,numberOfWeeks);

	classifyProduct<<<blocksNum,blockSize>>>(dev_products,dev_clusters,numOfProducts,numOfClusters,numberOfWeeks);
	checkKernelSucceeded(cudaStatus);

	tempProducts = (product_t*)calloc(numOfProducts,sizeof(product_t));
	cudaStatus = cudaMemcpy(tempProducts,dev_products, numOfProducts* sizeof(product_t), cudaMemcpyDeviceToHost);
	cudaCheaking(cudaStatus);

	copyClustersArrayFromGpu(clusters,dev_clusters,numOfClusters,numberOfWeeks);

	cudaFree(dev_clusters);

	return tempProducts;
}

void cudaCheaking(cudaError_t cudaStatus)
{
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cuda failed.");
	}
}


void checkKernelSucceeded(cudaError_t cudaStatus)
{
	// cudaDeviceSynchronize waits for the kernel to finish, and returns
	// any errors encountered during the launch.
	cudaStatus = cudaDeviceSynchronize();
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaDeviceSynchronize returned error code %d after launching kernel!\n", cudaStatus);
	}
}