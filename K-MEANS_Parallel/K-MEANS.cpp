// claster.cpp : Defines the entry point for the console application.
//
#include "structsAndFunctions.h"
#include <stddef.h>
#include <omp.h>

#define DATA 0
#define TERMINATION 1
#define ROOT 0

int main(int argc,char* argv[])
{
	int numberOfProducts,numberOfWeeks,maxNumOfClusters,maxNumOfIterations;
	int k,iteration,qualityAchived;
	double qualityMeasure,currentQuality;
	int finish = 0;

	//MPI variables
	int myid;
	int numprocs;
	MPI_Datatype MPI_Product;
	MPI_Datatype MPI_Cluster;
	MPI_Status status;

	cluster_t* clusters=NULL;
	product_t* products=0;
	product_t* productsOnGpu = 0;
	// MPI initalize
	mpiInitialize(&argc,argv,&myid,&numprocs);
	mpiProductTypeDefinition(&MPI_Product);
	mpiClusterTypeDefinition(&MPI_Cluster,&MPI_Product);

	if(myid == ROOT)
	{
		int i,j;
		int processProductsRange;
		double t1,t2;
		FILE* inputFile;
		FILE* outputFile;
		
		
		//Read data from file
		inputFile = fopen("C:\\Users\\afeka\\Desktop\\K-MEANS_ParallelProject\\K-MEANS_Parallel\\K-MEANS_Parallel\\Sales_Transactions_Dataset_Weekly.txt", "r");
		readDataFromFile(inputFile,&products,&numberOfProducts,&numberOfWeeks,&maxNumOfClusters,&maxNumOfIterations,&qualityMeasure);
		fclose(inputFile);
		
		t1 = MPI_Wtime();

		//Sending parameters to all processes
		sendingParametersToProcesses(numprocs,&processProductsRange,numberOfProducts,&numberOfWeeks,&maxNumOfIterations);
		//Divide data(products) to each process
		sendingProductsToProcesses(numprocs,numberOfWeeks,products,numberOfProducts,&MPI_Product);
		processProductsRange = numberOfProducts / numprocs; //numberOfProducts for ROOT
		
		productsOnGpu = allocateAndCopyProductsArrayInGpu(products,processProductsRange,numberOfWeeks);

		//Starting the Algorithm
		k = START_K;
		while(k <= maxNumOfClusters)
		{
			if(k>START_K)
				freeClusters(clusters,k-1);
			//sending k to each process
			for(i=1;i<numprocs;i++)
			{
				MPI_Send(&k,1,MPI_INT,i,DATA,MPI_COMM_WORLD);
			}
			
			//The master process runs the algorithm
			clusters = initialFirstClusters(clusters,products,k,numberOfWeeks);
			
			kmeansAlgorithm(numprocs,myid,k,numberOfProducts,numberOfWeeks,maxNumOfIterations,
				&clusters,products,productsOnGpu,&MPI_Cluster,&MPI_Product,&status);
			
			//Quality cheaking
			currentQuality = calculateQueality(clusters,products,numberOfProducts,k,numberOfWeeks);
			if(currentQuality < qualityMeasure || k==maxNumOfClusters)
			{
				for(i=1;i<numprocs;i++)
				{	
					MPI_Send(&k,1,MPI_INT,i,TERMINATION,MPI_COMM_WORLD);
				}
				break;
			}
			else
			{
				k++;
			}			
		}
		if(k>maxNumOfClusters)
			k--;

		t2 = MPI_Wtime();
		printf("\ntime: %lf\n",t2-t1);
		printf("QM: %lf",currentQuality);

		//The algorithm finished, print data to file	
		outputFile = fopen("C:\\Users\\afeka\\Desktop\\K-MEANS_ParallelProject\\K-MEANS_Parallel\\K-MEANS_Parallel\\finalOutput.txt","w");
		printDataToFile(outputFile,clusters,k,numberOfWeeks,currentQuality);
		fclose(outputFile);
		//printFinalData(clusters,k,numberOfWeeks);
		freeClusters(clusters,k);
		freeProducts(products,numberOfProducts);
	}
	else
	{
		int i,j;
		//Get parameters from root 
		receivingParametersFromRoot(&numberOfProducts,&numberOfWeeks,&maxNumOfIterations,&status);	
		//Get products from root 	
		products = receivingProductsFromRoot(numberOfProducts,numberOfWeeks,&MPI_Product,&status);
		//allocate products array in GPU
		productsOnGpu = allocateAndCopyProductsArrayInGpu(products,numberOfProducts,numberOfWeeks);
	
		MPI_Recv(&k,1,MPI_INT,ROOT,MPI_ANY_TAG,MPI_COMM_WORLD,&status);
		while(status.MPI_TAG!=TERMINATION)
		{	
			//wach process runs the algorithm
			kmeansAlgorithm			(numprocs,myid,k,numberOfProducts,numberOfWeeks,maxNumOfIterations,&clusters,products,productsOnGpu,&MPI_Cluster,&MPI_Product,&status);
			MPI_Recv(&k,1,MPI_INT,ROOT,MPI_ANY_TAG,MPI_COMM_WORLD,&status);
		}
		freeProducts(products,numberOfProducts);
	}
	MPI_Finalize();	
	return 0;
}


//functions
void readDataFromFile(FILE* inputFile,product_t** products,int* numberOfProducts, int* numberOfWeeks,int* maxNumOfClusters,int* maxNumOfIterations,double* qualityMeasure)
{
	int i,j;
	char chars[10];
	fscanf(inputFile,"%d %d %d %d %lf",numberOfProducts,numberOfWeeks,maxNumOfClusters,maxNumOfIterations,qualityMeasure);
	*products = (product_t*)calloc(*numberOfProducts,sizeof(product_t));
	for(i=0;i<*numberOfProducts;i++)
	{
		fscanf(inputFile,"%s",chars);
		(*products)[i].weeks = (double*)calloc(*numberOfWeeks,sizeof(double));
		for(j=0;j<*numberOfWeeks;j++)
		{
			fscanf(inputFile,"%lf",&((*products)[i].weeks[j])); 
		}
	}
}

void printDataToFile(FILE* outputFile,cluster_t* clusters,int numOfClusteres,int numberOfWeeks,double quality)
{
	int i,j;
	fprintf(outputFile,"%s","Number of clusters with the best measure\n");
	fprintf(outputFile,"K=%d , QM = %.2f\n",numOfClusteres,quality);
	for(i=0;i<numOfClusteres;i++)
	{
		fprintf(outputFile,"C%d: ",i);
		for(j=0;j<numberOfWeeks;j++)
		{
			fprintf(outputFile,"%.2f,",clusters[i].centeroid.weeks[j]);
		}
		fprintf(outputFile,"\n");
	}
}

cluster_t* initialFirstClusters(cluster_t* clusters,product_t* products,int k,int numberOfWeeks)
{
	int i;
	cluster_t* newClusters = (cluster_t*)calloc(k,sizeof(cluster_t));
	for(i=0;i<k;i++)
	{
		newClusters[i].centeroid.weeks = (double*)calloc(numberOfWeeks,sizeof(double));
		newClusters[i].size=0;
		newClusters[i].centeroid.weeks = products[i].weeks;
	}
	return newClusters;
}

void resetClustersSizes(cluster_t* clusters,int size)
{
	int i;
	for(i=0;i<size;i++)
	{
		clusters[i].size=0;
	}
}

int productsMovedBetweenClusters(product_t* products,int numOfProducts)
{
	int i;
	for(i=0;i<numOfProducts;i++)
	{
		if(products[i].currentCluster != products[i].prevCluster)
			return 1;
	}
	return 0;
}

double calculateDistance(product_t* p1,product_t* p2,int numberOfWeeks)
{
	int i;
	double distanceBeforeSqurt=0;

	for(i=0;i<numberOfWeeks;i++)
	{
		distanceBeforeSqurt += ((p2->weeks[i] - p1->weeks[i])*(p2->weeks[i] - p1->weeks[i]));
	}
	return sqrt(distanceBeforeSqurt);
}

void calculateClustersCenters(cluster_t* clusters,int numOfClusters,product_t* products,int numOfProducts,int numberOfWeeks)
{
	int i;
	//omp
#pragma omp parallel for
	for(i=0;i<numOfClusters;i++)
		clusters[i].centeroid = calculateAverageOfProducts(i,clusters[i].size,products,numOfProducts,numberOfWeeks);
}

product_t calculateAverageOfProducts(int clusterNum,int numOfProductsInCluster,product_t* products,int numOfProducts,int numberOfWeeks)
{
	int i,j;
	product_t* centerProduct = (product_t*)calloc(1,sizeof(product_t));
	centerProduct->weeks = (double*)calloc(numberOfWeeks,sizeof(double));

	for(i=0;i<numberOfWeeks;i++)
	{
		for(j=0;j<numOfProducts;j++)
		{
			if(products[j].currentCluster == clusterNum)
				centerProduct->weeks[i]+=products[j].weeks[i];
		}
		centerProduct->weeks[i]/=numOfProductsInCluster;
	}
	return *centerProduct;
}

double calculateQueality(cluster_t* clusters,product_t* products,int numOfProducts,int numOfClusters,int numberOfWeeks)
{
	int i,j;
	double quality=0;
	double currentClusterDiamameter;
	double distanceToOtherCluster;
	
	//omp
#pragma omp parallel for private(j,currentClusterDiamameter,distanceToOtherCluster) reduction(+:quality)
	for(i=0;i<numOfClusters;i++)
	{
		currentClusterDiamameter = calculateDiameter(&clusters[i],i,products,numOfProducts,numberOfWeeks);
		for(j=0;j<numOfClusters;j++)
		{
			if(i != j)
			{
				distanceToOtherCluster = calculateDistance(&clusters[i].centeroid,&clusters[j].centeroid,numberOfWeeks);
				quality += (currentClusterDiamameter/distanceToOtherCluster);
			}
		}
	}
	return quality/((double)(numOfClusters*(numOfClusters-1)));
}

double calculateDiameter(cluster_t* cluster,int clusterNum,product_t* products,int numOfProducts,int numberOfWeeks)
{
	int i,j;
	double maxDistance=0,finalMaxDistance=0,distance;

	for(i=0;i<numOfProducts;i++)
	{
		if(products[i].currentCluster == clusterNum)
		{
			for(j=0;j<numOfProducts;j++)
			{
				if(i !=j )
				{
					if(products[j].currentCluster == clusterNum)
					{
					distance = calculateDistance(&products[i],&products[j],numberOfWeeks);
					if(distance > maxDistance)
						maxDistance = distance;
					}
				}
			}
		}
	}
	return maxDistance;
}

int calcualteRnageOfProcess(int numprocs,int myid,int numberOfProducts)
{
	if(myid == numprocs-1)//last process
	{
		return (numberOfProducts - (numberOfProducts/numprocs)*(numprocs - 1));
	}
	else
	{
		return numberOfProducts/numprocs;
	}
}

//sync functions
void processesProductsSync(int prodcessId,int processProductsRange,int numberOfProducts,product_t* allProducts,product_t* partialProducts)
{
	int i;
	for(i=0 ; i<processProductsRange; i++)
	{
		allProducts[i+prodcessId*numberOfProducts].currentCluster = partialProducts[i].currentCluster;
		allProducts[i+prodcessId*numberOfProducts].prevCluster = partialProducts[i].prevCluster;
	}
}

void cudaProductsSync(product_t* tempProducts,product_t* finalProducts,int numOfProducts)
{
	int i;
	for(i=0;i<numOfProducts;i++)
	{
		finalProducts[i].currentCluster = tempProducts[i].currentCluster;
		finalProducts[i].prevCluster = tempProducts[i].prevCluster;
	}
}

void clustersSizesSync(cluster_t* clusters,product_t* products,int numberOfProducts)
{
	int i;	
	for(i=0;i<numberOfProducts;i++)
	{
		clusters[products[i].currentCluster].size++;
	}
}

//The Algorithm 
void kmeansAlgorithm(int numprocs,int myid,int k,int numberOfProducts,int numberOfWeeks,int maxNumOfIterations,
	cluster_t** clusters,product_t* products,product_t* productsOnGpu,MPI_Datatype* MPI_Cluster,MPI_Datatype* MPI_Product,MPI_Status* status)
{
	int i,j;
	int iteration=0;
	int rootProductsRange = numberOfProducts/numprocs;
	product_t* tempProducts;
	if(myid == ROOT)
	{
		//Classify products to clusters(if products still moved between clusters)
		do
		{
			resetClustersSizes(*clusters,k);
			sendingNewMissionToProcesses(numprocs,k,numberOfWeeks,*clusters,MPI_Cluster,DATA);
			
			//classify products with cuda 	and sync with original products array			
			tempProducts = classifyProductsToClustersWithCuda(productsOnGpu,rootProductsRange,k,*clusters,numberOfWeeks);			
			cudaProductsSync(tempProducts,products,rootProductsRange);			
			
			//sync clusters sizes
			clustersSizesSync(*clusters,products,rootProductsRange);

			//receiving the rest of the products from the slaves
			receivingProudctsFromProcesses(numprocs,k,MPI_Cluster,MPI_Product,status,products,*clusters,numberOfProducts,numberOfWeeks);
			
			//calculate the new clusters centers
			calculateClustersCenters(*clusters,k,products,numberOfProducts,numberOfWeeks);
			iteration++;	
		}while(productsMovedBetweenClusters(products,numberOfProducts) && iteration<maxNumOfIterations);
		sendingNewMissionToProcesses(numprocs,k,numberOfWeeks,*clusters,MPI_Cluster,TERMINATION);
	}
	else
	{
		int i,j;
		product_t* tempProducts;
		*clusters = receivingMissionFromRoot(k,numberOfWeeks,MPI_Cluster,status);
		while(status->MPI_TAG!=TERMINATION)
		{
			//classify products with cuda and sync with original products array
			tempProducts = classifyProductsToClustersWithCuda(productsOnGpu,numberOfProducts,k,*clusters,numberOfWeeks);
			cudaProductsSync(tempProducts,products,numberOfProducts);
			
			//sending back to the root the temp products
			MPI_Send(products,numberOfProducts,*MPI_Product,ROOT,DATA,MPI_COMM_WORLD);

			//receiving new mission(clusters for classification) from root
			freeClusters(*clusters,k);
			*clusters = receivingMissionFromRoot(k,numberOfWeeks,MPI_Cluster,status);
		}		
	}
}

//MPI initialize
void mpiInitialize(int* argc,char *argv[],int* myid,int* numprocs)
{
	MPI_Init(argc,&argv);
	MPI_Comm_rank(MPI_COMM_WORLD,myid);
	MPI_Comm_size(MPI_COMM_WORLD,numprocs);	
}

void mpiProductTypeDefinition(MPI_Datatype* MPI_Product)
{
	int blocklen[3] = { 1, 1, 1 };
	MPI_Aint offset[3];
	MPI_Datatype type[3] = {MPI_DOUBLE, MPI_INT, MPI_INT};
	offset[0]=offsetof(product_t, weeks);
	offset[1]=offsetof(product_t, currentCluster);
	offset[2]=offsetof(product_t, prevCluster);
	MPI_Type_create_struct(3, blocklen, offset, type, MPI_Product);
	MPI_Type_commit(MPI_Product);
}

void mpiClusterTypeDefinition(MPI_Datatype* MPI_Cluster,MPI_Datatype* MPI_Product)
{
	int blocklen[2] = { 1, 1 };
	MPI_Aint offset[2];
	MPI_Datatype type[2] = {*MPI_Product,MPI_INT};
	offset[0]=offsetof(cluster_t, centeroid);
	offset[1]=offsetof(cluster_t, size);
	MPI_Type_create_struct(2, blocklen, offset, type,MPI_Cluster);
	MPI_Type_commit(MPI_Cluster);
}

//MPI functions
void sendingParametersToProcesses(int numprocs,int* processProductsRange,int numberOfProducts,int* numberOfWeeks,int* maxNumOfIterations)
{
	int i;
	int processProductsRangee;
	for(i=1;i<numprocs;i++)
	{
		processProductsRangee = calcualteRnageOfProcess(numprocs,i,numberOfProducts);
		MPI_Send(&processProductsRangee,1,MPI_INT,i,0,MPI_COMM_WORLD);
		MPI_Send(numberOfWeeks,1,MPI_INT,i,0,MPI_COMM_WORLD);
		MPI_Send(maxNumOfIterations,1,MPI_INT,i,0,MPI_COMM_WORLD);
	}
}

void sendingProductsToProcesses(int numprocs,int numberOfWeeks,product_t* products,int numberOfProducts,MPI_Datatype* MPI_Product)
{
	int i,j;
	int processProductsRange;
	for(i=1;i<numprocs;i++)
	{
		processProductsRange = calcualteRnageOfProcess(numprocs,i,numberOfProducts);
		MPI_Send(&products[i*(numberOfProducts/numprocs)],processProductsRange,*MPI_Product,i,DATA,MPI_COMM_WORLD);

		for(j=0;j<processProductsRange;j++)
		{
			MPI_Send(products[i*(numberOfProducts/numprocs) + j].weeks,numberOfWeeks,MPI_DOUBLE,i,DATA,MPI_COMM_WORLD);
		}
	}
}

product_t* receivingProductsFromRoot(int numberOfProducts,int numberOfWeeks,MPI_Datatype* MPI_Product,MPI_Status* status)
{
	int i;
	product_t* products = (product_t*)malloc((numberOfProducts)*sizeof(product_t));
	MPI_Recv(products,numberOfProducts,*MPI_Product,ROOT,0,MPI_COMM_WORLD,status);
	for(i=0;i<numberOfProducts;i++)
	{
		products[i].weeks = (double*)malloc(numberOfWeeks*sizeof(double));
	}
	for(i=0;i<numberOfProducts;i++)
	{
		MPI_Recv(products[i].weeks,numberOfWeeks,MPI_DOUBLE,ROOT,0,MPI_COMM_WORLD,status);
	}
	return products;
}

void receivingParametersFromRoot(int* numberOfProducts,int* numberOfWeeks,int* maxNumOfIterations,MPI_Status* status)
{
	MPI_Recv(numberOfProducts,1,MPI_INT,ROOT,0,MPI_COMM_WORLD,status);
	MPI_Recv(numberOfWeeks,1,MPI_INT,ROOT,0,MPI_COMM_WORLD,status);
	MPI_Recv(maxNumOfIterations,1,MPI_INT,ROOT,0,MPI_COMM_WORLD,status);
}

void receivingProudctsFromProcesses(int numprocs,int k,MPI_Datatype* MPI_Cluster,MPI_Datatype* MPI_Product,MPI_Status* status,product_t* finalProudcts,cluster_t* finalClusters,int numberOfProducts,int numberOfWeeks)
{
	int i,j;
	int processProductsRange;
	product_t* tempProducts;
	for(i=1;i<numprocs;i++)
	{
		processProductsRange = calcualteRnageOfProcess(numprocs,i,numberOfProducts);

		tempProducts = (product_t*)malloc(sizeof(product_t)*processProductsRange);

		MPI_Recv(tempProducts,processProductsRange,*MPI_Product,i,DATA,MPI_COMM_WORLD,status);

		processesProductsSync(i,processProductsRange,numberOfProducts/numprocs,finalProudcts,tempProducts);

		clustersSizesSync(finalClusters,tempProducts,processProductsRange);

		free(tempProducts);
	}
}

void sendingNewMissionToProcesses(int numprocs,int k,int numberOfWeeks,cluster_t* finalClusters,MPI_Datatype* MPI_Cluster,int flag)
{
	int i,j;
	for(i=1;i<numprocs;i++)
	{
		MPI_Send(finalClusters,k,*MPI_Cluster,i,flag,MPI_COMM_WORLD);
	}
	if (flag == DATA)
	{
		for(i=1;i<numprocs;i++)
	{
			for(j=0;j<k;j++)
			{
				MPI_Send(finalClusters[j].centeroid.weeks,numberOfWeeks,MPI_DOUBLE,i,DATA,MPI_COMM_WORLD);
			}
		}
	}
}

cluster_t* receivingMissionFromRoot(int k,int numberOfWeeks,MPI_Datatype* MPI_Cluster,MPI_Status* status)
{
	int i;
	cluster_t* clusters = (cluster_t*)malloc(k*sizeof(cluster_t));
	MPI_Recv(clusters,k,*MPI_Cluster,ROOT,MPI_ANY_TAG,MPI_COMM_WORLD,status);
	if(status->MPI_TAG!=TERMINATION)
	{
		for(i=0;i<k;i++)
		{
			clusters[i].centeroid.weeks = (double*)malloc(numberOfWeeks*sizeof(double));
			MPI_Recv(clusters[i].centeroid.weeks,numberOfWeeks,MPI_DOUBLE,ROOT,DATA,MPI_COMM_WORLD,status);
		}
		return clusters;
	}
	return NULL;
}

//free functions
void freeClusters(cluster_t* clusters,int size)
{
	int i;
	for(i=0;i<size;i++)
	{
		free(clusters[i].centeroid.weeks);
	}
	free(clusters);
}

void freeAll(cluster_t* clusters,int numOfClusters,product_t* products,int numOfProducts)
{
	freeClusters(clusters,numOfClusters);
	freeProducts(products,numOfProducts);
}

void freeProducts(product_t* products,int numOfProducts)
{
	int i;
	for(i=0;i<numOfProducts;i++)
	{
		free(products[i].weeks);
	}
	free(products);
}
