#include <stdio.h>
#include <stdlib.h>
#include <iostream>
#include <Eigen/Dense>

#ifdef __APPLE__
#include <OpenCL/opencl.h>
#else
#include <CL/cl.h>
#endif

#define MAX_SOURCE_SIZE (0x100000)

using namespace std;
using namespace Eigen;


#define __CL_ENABLE_EXCEPTIONS

#include "cl.hpp"

#include "util.hpp" // utility library

#include "err_code.h"

#include <vector>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <time.h>
#include <ctime>
#include <iostream>
#include <fstream>


int main()
{	
	
	//Algorithm parameters
	int variableRange = 30;
	int numberOfVariables = 2;
	int numberOfParticles = pow(2,12);
	int alpha = 2;
	float inertia = 1.4;
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
	
	// Create buffers using the external header library to hold the data
	cl::Buffer 	particlePositionsBuffer,
				particleVelocitiesBuffer,
				bestParticlePositionsBuffer, 
				bestParticleValuesBuffer,
				bestGlobalParticlePositionsBuffer,
				bestGlobalParticleValueBuffer,
				randArray1Buffer,
				randArray2Buffer,
				cBuffer,
				indexHolderBuffer;
	
	// Create a context
    cl::Context ctx(CL_DEVICE_TYPE_DEFAULT);

    // Load in kernel source, creating a program object for the context
    cl::Program program(ctx, util::loadProgram("ParticleFunctions.cl"), true);

    // Get the command queue
    cl::CommandQueue queue(ctx);
	
	// Make the kernel functions
    auto EvaluateParticles = cl::make_kernel<cl::Buffer, // Particle positions
								cl::Buffer, // BestParticle value
								cl::Buffer, // Best particle position
								int>(program, "EvaluateParticles");
							
	auto UpdateVelocities = cl::make_kernel<cl::Buffer,//Particle velocities
											int,		// Number of variables
											float,
											float,		// inertia
											cl::Buffer,
											cl::Buffer,
											cl::Buffer, // c
											cl::Buffer, // particle positions
											cl::Buffer, // bestParticlePos & globalBest
											cl::Buffer>(program, "UpdateVelocities");
											
    auto reduceKernel = cl::make_kernel<cl::Buffer, 
										cl::Buffer,
										cl::Buffer,
										cl::Buffer,
										cl::Buffer,
										cl::LocalSpaceArg,
										cl::LocalSpaceArg,
										int,
										cl::Buffer // Results
										>(program,"reduce");
    // Test kernel that includes both EvaluateParticles and Reduce in the same kernel to reduce calling time from CPU side? 
	/*									
    auto TestKernel = cl::make_kernel<cl::Buffer, 
										cl::Buffer, 
										cl::Buffer, 
										int,
										cl::Buffer,
										cl::Buffer,
										cl::Buffer,
										cl::LocalSpaceArg,
										cl::LocalSpaceArg, 
										int, 
										cl::Buffer
										>(program,"Test");
										*/
								
    
    // Write all the buffers 
	particlePositionsBuffer = cl::Buffer(ctx,&particlePositions(0,0),&particlePositions(numberOfVariables-1,numberOfParticles-1)+1,true);
	particleVelocitiesBuffer = cl::Buffer(ctx,&particleVelocities(0,0),&particleVelocities(numberOfVariables-1,numberOfParticles-1)+1,true);
	bestParticlePositionsBuffer = cl::Buffer(ctx,&bestParticlePositions(0,0),&bestParticlePositions(numberOfVariables-1,numberOfParticles-1)+1,true);
	bestParticleValuesBuffer = cl::Buffer(ctx,&bestParticleValues(0,0),&bestParticleValues(0,numberOfParticles-1)+1,true);
	bestGlobalParticlePositionsBuffer = cl::Buffer(ctx,&bestGlobalParticlePositions(0,0),&bestGlobalParticlePositions(numberOfVariables-1,0)+1,true);
    bestGlobalParticleValueBuffer  = cl::Buffer(ctx, &bestGlobalParticleValue(0,0),&bestGlobalParticleValue(0,0)+1,true);
	randArray1Buffer = cl::Buffer(ctx, &randArray1(0,0),&randArray1(numberOfVariables-1,numberOfParticles-1)+1,true);
	randArray2Buffer = cl::Buffer(ctx, &randArray2(0,0),&randArray2(numberOfVariables-1,numberOfParticles-1)+1,true);
	cBuffer = cl::Buffer(ctx, &c(0,0),&c(0,1)+1,true);
	cl::LocalSpaceArg localmemScratch = cl::Local(sizeof(float) * numberOfParticles);
	cl::LocalSpaceArg localmemTmpIndex = cl::Local(sizeof(int) * numberOfParticles);
	cl::Buffer resultsBuffer = cl::Buffer(ctx,CL_MEM_WRITE_ONLY, 2*sizeof(float));
	indexHolderBuffer = cl::Buffer(ctx,&indexHolder(0),&indexHolder(numberOfParticles-1)+1,true);
    
	clock_t begin = clock();
	util::Timer timer;
	int count =0;
	while(inertia > 0.4)
	{
		count ++;
		// Evaluate  each particle
	    EvaluateParticles(
	        cl::EnqueueArgs(
	            queue,
	            cl::NDRange(numberOfParticles)), 
	        particlePositionsBuffer,
			bestParticleValuesBuffer,
			bestParticlePositionsBuffer,
			numberOfVariables);
	    queue.finish();
	
		// Update the global best array using array reduction from the bestParticleValues and bestParticlePositions
	    reduceKernel(
	        cl::EnqueueArgs(
	            queue,
	            cl::NDRange(numberOfParticles)), 
	        bestParticleValuesBuffer,
			bestParticlePositionsBuffer,
			bestGlobalParticleValueBuffer,
			bestGlobalParticlePositionsBuffer,
			indexHolderBuffer,
			localmemTmpIndex,
			localmemScratch,
			numberOfParticles,
			resultsBuffer);
	    queue.finish();
	
		
		//Update particle velocities and positions
	    UpdateVelocities(
	        cl::EnqueueArgs(
	            queue,
	            cl::NDRange(numberOfParticles)), 
	        particleVelocitiesBuffer,
			numberOfVariables,
			maximumVelocity,
			inertia,
			randArray1Buffer,
			randArray2Buffer,
			cBuffer,
			particlePositionsBuffer,
			bestParticlePositionsBuffer,
			bestGlobalParticlePositionsBuffer);
	    queue.finish();
	
		// Rinse and repeat
		// TODO: INCLUDE UPDATE OF THE RANDOM MATRICES. CODE WORKS STILL THOUGH.
	
	
		inertia *= beta;
	}
	cl::copy(queue, bestGlobalParticleValueBuffer, &bestGlobalParticleValue(0,0),&bestGlobalParticleValue(0,0)+1);
	cout<<"BEST SCORE: "<<bestGlobalParticleValue<<endl;
	

    double rtime = static_cast<double>(timer.getTimeMilliseconds()) / 1000.0;
    printf("\nThe kernels ran in %lf seconds\n", rtime);
    return 0;
}