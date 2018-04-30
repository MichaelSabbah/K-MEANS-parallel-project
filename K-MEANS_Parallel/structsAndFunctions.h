#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdlib.h>
#include <stdio.h>
#include <mpi.h>
#include <math.h>

#define START_K 2

typedef struct Product
{
	double* weeks;
	int currentCluster;
	int prevCluster;
}product_t;

typedef struct Cluster
{
	product_t centeroid;
	int size;
}cluster_t;

//The Algorithm
void kmeansAlgorithm(int numprocs,int myid,int k,int numberOfProducts,int numberOfWeeks,int maxNumOfIterations,
	cluster_t** clusters,product_t* products,product_t* productsOnGpu,MPI_Datatype* MPI_Cluster,MPI_Datatype* MPI_Product,MPI_Status* status);

//initial functions and files
void readDataFromFile(FILE* inputFile,product_t** products,int* numberOfProducts, int* numberOfWeeks,int* maxNumOfClusters,int* maxNumOfIterations,double* qualityMeasure);
cluster_t* initialFirstClusters(cluster_t* clusters,product_t* products,int k,int numberOfWeeks);
void printDataToFile(FILE* outputFile,cluster_t* clusters,int numOfClusteres,int numberOfWeeks,double quality);

//products organizing functions
int productsMovedBetweenClusters(product_t* products,int numOfProducts);

//calculations functions
double calculateDistance(product_t* p1,product_t* p2,int numberOfWeeks);
void calculateClustersCenters(cluster_t* clusters,int numOfClusters,product_t* products,int numOfProducts,int numberOfWeeks);
product_t calculateAverageOfProducts(int clusterNum,int numOfProductsInCluster,product_t* products,int numOfProducts,int numberOfWeeks);
double calculateQueality(cluster_t* clusters,product_t* products,int numOfProducts,int numOfClusters,int numberOfWeeks);
double calculateDiameter(cluster_t* cluster,int clusterNum,product_t* products,int numOfProducts,int numberOfWeeks);
product_t calculateAverageOfCenteroids(product_t* firstCenteroid,product_t* secondCenteroid,int numberofWeeks);
int calcualteRnageOfProcess(int numprocs,int myid,int numberOfProducts);

//processes sync functions
void processesProductsSync(int prodcessId,int processProductsRange,int numberOfProducts,product_t* allProducts,product_t* partialProducts);
void clustersSizesSync(cluster_t* clusters,product_t* products,int numberOfProducts);

//sync functions
void cudaProductsSync(product_t* tempProducts,product_t* finalProducts,int numOfProducts);

//free functions
void freeAll(cluster_t* clusters,int numOfClusters,product_t* products,int numOfProducts);
void freeClusters(cluster_t* clusters,int size);
void freeProducts(product_t* products,int numOfProducts);
void resetClustersSizes(cluster_t* clusters,int size);

//MPI initialize
void mpiInitialize(int* argc,char *argv[],int* myid,int* numprocs);
void mpiProductTypeDefinition(MPI_Datatype* MPI_Product);
void mpiClusterTypeDefinition(MPI_Datatype* MPI_Cluster,MPI_Datatype* MPI_Product);

//MPI functions
void sendingParametersToProcesses(int numprocs,int* numberofProductsForProcess,int numberOfProducts,int* numberOfWeeks,int* maxNumOfIterations);
void sendingProductsToProcesses(int numprocs,int numberOfWeeks,product_t* products,int processProductsRange,MPI_Datatype* MPI_Product);
product_t* receivingProductsFromRoot(int numberOfProducts,int numberOfWeeks,MPI_Datatype* MPI_Product,MPI_Status* status);
void receivingParametersFromRoot(int* numberOfProducts,int* numberOfWeeks,int* maxNumOfIterations,MPI_Status* status);
void receivingProudctsFromProcesses(int numprocs,int k,MPI_Datatype* MPI_Cluster,MPI_Datatype* MPI_Product,MPI_Status* status,product_t* finalProudcts,cluster_t* finalClusters,int numberOfProducts,int numberOfWeeks);
void sendingNewMissionToProcesses(int numprocs,int k,int numberOfWeeks,cluster_t* finalClusters,MPI_Datatype* MPI_Cluster,int flag);
cluster_t* receivingMissionFromRoot(int k,int numberOfWeeks,MPI_Datatype* MPI_Cluster,MPI_Status* status);

//cuda functions
void cudaCheaking(cudaError_t cudaStatus);
product_t* classifyProductsToClustersWithCuda(product_t* dev_products,int numOfProducts,int numOfClusters,cluster_t* clusters,int numberOfWeeks);
__global__ void classifyProduct(product_t* dev_products,cluster_t* dev_clusters,int numOfProducts,int numOfClusters,int numberOfWeeks);
void copyClustersArrayFromGpu(cluster_t* clusters,cluster_t* dev_clusters,int numOfClusters,int numOfWeeks);
cluster_t* allocateAndCopyClustersArrayInGpu(cluster_t* clusters,int numOfClusters,int numberOfWeeks);
product_t* allocateAndCopyProductsArrayInGpu(product_t* products,int numOfProducts,int numberOfWeeks);
__device__ double calculateDistanceCuda(double* p1Weeks, double* p2Weeks, int numberOfWeeks);
void checkKernelSucceeded(cudaError_t cudaStatus);