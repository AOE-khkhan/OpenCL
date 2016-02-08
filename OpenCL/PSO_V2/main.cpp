#include <stdio.h>
#include <stdlib.h>
#include <iostream>
#include <Eigen/Dense>
#include <vector>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <time.h>
#include <ctime>
#include <iostream>
#include <fstream>


#ifdef __APPLE__
#include <OpenCL/opencl.h>
#else
#include <CL/cl.h>
#endif

#define MAX_SOURCE_SIZE (0x100000)

using namespace std;
using namespace Eigen;




int main()
{	
	//  Start Timers
	clock_t begin = clock();
	//Algorithm parameters
	int variableRange = 30;
	int numberOfVariables = 2;
	int numberOfParticles = pow(2,15);
	int alpha = 2;
	int size = numberOfVariables*numberOfParticles;
	float *inertia = (float *)malloc(sizeof(float));
	float in = 1.4;
	inertia[0] = 1.4;
	float beta=0.99;
	float maximumVelocity = 5;
	
	// Initialize random seed
	srand(time(NULL)); rand();
	
	// TODO: NOT USE EIGEN TO INITIALIZE PARAMETERS
	// Intialize particle parameters and other stuff needed for the algorithm
	MatrixXf particlePositions = -variableRange*MatrixXf::Ones(numberOfVariables,numberOfParticles)+ 
					2*variableRange*(MatrixXf::Ones(numberOfVariables,numberOfParticles)+(MatrixXf::Random(numberOfVariables,numberOfParticles)))/2.0;
	MatrixXf particleVelocities = alpha*(-variableRange*MatrixXf::Ones(numberOfVariables,numberOfParticles)+ 
					2*variableRange*(MatrixXf::Ones(numberOfVariables,numberOfParticles)+(MatrixXf::Random(numberOfVariables,numberOfParticles)))/2.0);
	
	MatrixXf bestParticlePositions = MatrixXf::Zero(numberOfVariables,numberOfParticles);
	MatrixXf bestParticleValues = MatrixXf::Ones(1,numberOfParticles)*pow(10,9);
	
	MatrixXf bestGlobalParticleValue = pow(10,9)*MatrixXf::Ones(1,1);
	MatrixXf bestGlobalParticlePositions = MatrixXf::Zero(numberOfVariables,1);
	MatrixXf randArray1 = (MatrixXf::Ones(numberOfVariables,numberOfParticles)+MatrixXf::Random(numberOfVariables,numberOfParticles))/2.0;
	MatrixXf randArray2 = (MatrixXf::Ones(numberOfVariables,numberOfParticles)+MatrixXf::Random(numberOfVariables,numberOfParticles))/2.0;
	MatrixXf c(1,2); c << 2, 2;
	VectorXi indexHolder = VectorXi::LinSpaced(numberOfParticles,0,numberOfParticles-1);
	
	// Load the kernel source code into the array source_str
    FILE *fp;
    char *source_str;
    size_t source_size;

    fp = fopen("ParticleFunctions.cl", "r");
    if (!fp) {
        fprintf(stderr, "Failed to load kernel.\n");
        exit(1);
    }
    source_str = (char*)malloc(MAX_SOURCE_SIZE);
    source_size = fread( source_str, 1, MAX_SOURCE_SIZE, fp);
    fclose( fp );
	
	// TODO: AVOID USING EIGEN?	
	// Variables to hold all of the stuff:
	float *positions = (float*)malloc(sizeof(float)*size);
	Map<MatrixXf>(positions,numberOfVariables,numberOfParticles) = particlePositions;
	
	float *velocities = (float*)malloc(sizeof(float)*size);
	Map<MatrixXf>(velocities,numberOfVariables,numberOfParticles) = particleVelocities;
	
	float *bestPositions = (float*)malloc(sizeof(float)*size);
	Map<MatrixXf>(bestPositions,numberOfVariables,numberOfParticles) = bestParticlePositions;
	
	float *bestValues = (float*)malloc(sizeof(float)*numberOfParticles);
	Map<MatrixXf>(bestValues,1,numberOfParticles) = bestParticleValues;
	
	float *globalBestPositions = (float*)malloc(sizeof(float)*numberOfVariables);
	globalBestPositions[0]= pow(10.0f,9);
	
	float *globalBestValues = (float*)malloc(sizeof(float));
	Map<MatrixXf>(globalBestValues,1,1) = bestGlobalParticleValue;
	
	float *rand1 = (float*)malloc(sizeof(float)*size);
	Map<MatrixXf>(rand1,numberOfVariables,numberOfParticles) = randArray1;
	
	float *rand2 = (float*)malloc(sizeof(float)*size);
	Map<MatrixXf>(rand2,numberOfVariables,numberOfParticles) = randArray2;
	
	float *CC = (float*)malloc(sizeof(float)*2);
	Map<MatrixXf>(CC,1,2) = c;
	
	int *index = (int*)malloc(sizeof(int)*numberOfParticles);
	Map<VectorXi>(index,numberOfParticles) = indexHolder;


    // Get platform and device information
    cl_platform_id platform_id = NULL;
    cl_device_id device_id = NULL;   
    cl_uint ret_num_devices;
    cl_uint ret_num_platforms;
    cl_int ret = clGetPlatformIDs(1, &platform_id, &ret_num_platforms);
    ret = clGetDeviceIDs( platform_id, CL_DEVICE_TYPE_DEFAULT, 1, 
            &device_id, &ret_num_devices);

    // Create an OpenCL context
    cl_context context = clCreateContext( NULL, 1, &device_id, NULL, NULL, &ret);

    // Create a command queue
    cl_command_queue command_queue = clCreateCommandQueue(context, device_id, 0, &ret);
	
    // Create a program from the kernel source
    cl_program program = clCreateProgramWithSource(context, 1, 
            (const char **)&source_str, (const size_t *)&source_size, &ret);

    // Build the program
    ret = clBuildProgram(program, 1, &device_id, NULL, NULL, NULL);

    // Create the OpenCL kernel
    cl_kernel evaluateParticles = clCreateKernel(program, "EvaluateParticles", &ret);
	cl_kernel reduce = clCreateKernel(program, "reduce", &ret);
    cl_kernel updateVelocities = clCreateKernel(program, "UpdateVelocities", &ret);
	

	// Create cl_mem objects
	cl_mem positionsObj = clCreateBuffer(context, CL_MEM_READ_WRITE, sizeof(float)*size, NULL, &ret);
	cl_mem velocitiesObj = clCreateBuffer(context, CL_MEM_READ_WRITE, sizeof(float)*size, NULL, &ret);
	cl_mem bestPositionsObj = clCreateBuffer(context, CL_MEM_READ_WRITE, sizeof(float)*size, NULL, &ret);
	cl_mem bestValuesObj = clCreateBuffer(context, CL_MEM_READ_WRITE, sizeof(float)*numberOfParticles, NULL, &ret);
	cl_mem globalBestPositionsObj = clCreateBuffer(context, CL_MEM_READ_WRITE, sizeof(float)*numberOfVariables, NULL, &ret);
	cl_mem globalBestValuesObj = clCreateBuffer(context, CL_MEM_READ_WRITE, sizeof(float), NULL, &ret);
	cl_mem rand1Obj = clCreateBuffer(context, CL_MEM_READ_ONLY, sizeof(float)*size, NULL, &ret);
	cl_mem rand2Obj = clCreateBuffer(context, CL_MEM_READ_ONLY, sizeof(float)*size, NULL, &ret);
	cl_mem CCObj = clCreateBuffer(context, CL_MEM_READ_ONLY, sizeof(float)*2, NULL, &ret);
	cl_mem indexObj = clCreateBuffer(context, CL_MEM_READ_ONLY, sizeof(int)*numberOfParticles, NULL, &ret);
	cl_mem inertiaObj = clCreateBuffer(context, CL_MEM_READ_WRITE, sizeof(float), NULL, &ret);
	
	// Write to buffer
    ret = clEnqueueWriteBuffer(command_queue, positionsObj, CL_TRUE, 0,size* sizeof(float), positions, 0, NULL, NULL);
	ret = clEnqueueWriteBuffer(command_queue, velocitiesObj, CL_TRUE, 0,size* sizeof(float), velocities, 0, NULL, NULL);
	ret = clEnqueueWriteBuffer(command_queue, bestPositionsObj, CL_TRUE, 0,size* sizeof(float), bestPositions, 0, NULL, NULL);
	ret = clEnqueueWriteBuffer(command_queue, bestValuesObj, CL_TRUE, 0,numberOfParticles * sizeof(float), bestValues, 0, NULL, NULL);
	ret = clEnqueueWriteBuffer(command_queue, globalBestPositionsObj, CL_TRUE, 0,numberOfVariables * sizeof(float), globalBestPositions, 0, NULL, NULL);
	ret = clEnqueueWriteBuffer(command_queue, globalBestValuesObj, CL_TRUE, 0, sizeof(float), globalBestValues, 0, NULL, NULL);
	ret = clEnqueueWriteBuffer(command_queue, rand1Obj, CL_TRUE, 0,size * sizeof(float), rand1, 0, NULL, NULL);
	ret = clEnqueueWriteBuffer(command_queue, rand2Obj, CL_TRUE, 0,size * sizeof(float), rand2, 0, NULL, NULL);
	ret = clEnqueueWriteBuffer(command_queue, CCObj, CL_TRUE, 0,2 * sizeof(float), CC, 0, NULL, NULL);
	ret = clEnqueueWriteBuffer(command_queue, inertiaObj, CL_TRUE, 0, sizeof(float), inertia, 0, NULL, NULL);
	ret = clEnqueueWriteBuffer(command_queue, indexObj, CL_TRUE, 0,numberOfParticles * sizeof(int), index, 0, NULL, NULL);
	

    // Set the arguments for Evaluate particles
    ret = clSetKernelArg(evaluateParticles, 0, sizeof(cl_mem), (void *)&positionsObj);
    ret = clSetKernelArg(evaluateParticles, 1, sizeof(cl_mem), (void *)&bestValuesObj);
    ret = clSetKernelArg(evaluateParticles, 2, sizeof(cl_mem), (void *)&bestPositionsObj);
	ret = clSetKernelArg(evaluateParticles, 3, sizeof(int), &numberOfVariables);
	
	
	// Set the arguments for reduction
    ret = clSetKernelArg(reduce, 0, sizeof(cl_mem), (void *)&bestValuesObj);
    ret = clSetKernelArg(reduce, 1, sizeof(cl_mem), (void *)&bestPositionsObj);
    ret = clSetKernelArg(reduce, 2, sizeof(cl_mem), (void *)&globalBestValuesObj);
	ret = clSetKernelArg(reduce, 3, sizeof(cl_mem), (void *)&globalBestPositionsObj);
	ret = clSetKernelArg(reduce, 4, sizeof(cl_mem), (void *)&indexObj);
	ret = clSetKernelArg(reduce, 5, sizeof(cl_float) * 32, NULL);
	ret = clSetKernelArg(reduce, 6, sizeof(cl_int) * 32, NULL);
	ret = clSetKernelArg(reduce, 7, sizeof(int), &numberOfParticles);

    // Set the arguments for updateVelocities
	ret = clSetKernelArg(updateVelocities,0,sizeof(cl_mem),(void *)&velocitiesObj);
	ret = clSetKernelArg(updateVelocities,1,sizeof(int),&numberOfVariables);
	ret = clSetKernelArg(updateVelocities,2,sizeof(float),&maximumVelocity);
	ret = clSetKernelArg(updateVelocities,3,sizeof(cl_mem),(void *)&inertiaObj);
	ret = clSetKernelArg(updateVelocities,4,sizeof(cl_mem),(void *)&rand1Obj);
	ret = clSetKernelArg(updateVelocities,5,sizeof(cl_mem),(void *)&rand2Obj);
	ret = clSetKernelArg(updateVelocities,6,sizeof(cl_mem),(void *)&CCObj);
	ret = clSetKernelArg(updateVelocities,7,sizeof(cl_mem),(void *)&positionsObj);
	ret = clSetKernelArg(updateVelocities,8,sizeof(cl_mem),(void *)&bestPositionsObj);
	ret = clSetKernelArg(updateVelocities,9,sizeof(cl_mem),(void *)&globalBestPositionsObj);
	
	
    // Execute the OpenCL kernel on the list
    size_t global_item_size[] = {(size_t)size,(size_t)size,(size_t)size};// = size; // Process the entire lists
    size_t local_item_size[] = {1, 32, 1};
	
	
	
	for(int i=0;i<125;i++)
	{
		
		ret = clEnqueueNDRangeKernel(command_queue, evaluateParticles, 1, NULL, &global_item_size[0], &local_item_size[0], 0, NULL, NULL);
		
		ret = clEnqueueNDRangeKernel(command_queue, reduce, 1, NULL, &global_item_size[1], &local_item_size[1], 0, NULL, NULL);
		
		ret = clEnqueueNDRangeKernel(command_queue, updateVelocities, 1, NULL, &global_item_size[2], &local_item_size[2], 0, NULL, NULL);
		
		
		// Rinse and repeat
		// TODO: INCLUDE UPDATE OF THE RANDOM MATRICES. CODE WORKS STILL THOUGH. PERHAPS THROUGH GPU. CALLS TO CPU -> GPU EXPENSIVE
		
	
	}
	
	ret = clEnqueueReadBuffer(command_queue, globalBestValuesObj, CL_TRUE, 0,sizeof(float), globalBestValues, 0, NULL, NULL);
	cout<<"best score :"<<globalBestValues[0]<<endl;
	
	float timeTaken = (float)(clock()-begin) / CLOCKS_PER_SEC;			
	cout<<"Time taken: "<<timeTaken<<endl;
	
   
	
    ret = clFlush(command_queue);
    ret = clFinish(command_queue);
    ret = clReleaseKernel(evaluateParticles);
    ret = clReleaseProgram(program);
	
    ret = clReleaseMemObject(positionsObj);
	ret = clReleaseMemObject(velocitiesObj);
	ret = clReleaseMemObject(bestPositionsObj);
	ret = clReleaseMemObject(globalBestPositionsObj);
	ret = clReleaseMemObject(globalBestValuesObj);
	ret = clReleaseMemObject(rand1Obj);
	ret = clReleaseMemObject(rand2Obj);
	ret = clReleaseMemObject(CCObj);
	ret = clReleaseMemObject(inertiaObj);
	ret = clReleaseMemObject(indexObj);
	
    ret = clReleaseCommandQueue(command_queue);
    ret = clReleaseContext(context);
	
	free(positions);
	free(velocities);
	free(bestPositions);
	free(bestValues);
	free(globalBestPositions);
	free(globalBestValues);
	free(rand1);
	free(rand2);
	free(CC);
	free(inertia);
	free(index);
	
    return 0;
}