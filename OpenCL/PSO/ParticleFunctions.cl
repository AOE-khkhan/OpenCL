__kernel void EvaluateParticles(                             
   __global float* positions,                                           
   __global float* bestParticleValues,
   __global float* bestParticlePositions,
   const unsigned int nVariables)               
{

   int i = get_global_id(0);            
   float fitness;

   fitness = 1 + pow( -13 + positions[i*nVariables+0] - pow(positions[i*nVariables+1],3) + 5*pow(positions[i*nVariables+1],2.0f) - 2*positions[i*nVariables+1],2.0f) +
	    pow(-29 + positions[i*nVariables+0] +pow(positions[i*nVariables+1],3.0f) + pow(positions[i*nVariables+1],2.0f) - 14*positions[i*nVariables+1],2.0f);

   if ( fitness < bestParticleValues[i])
   {
	   bestParticleValues[i] = fitness;
	   for ( int j = 0; j < nVariables; j++)
	   {
		   bestParticlePositions[i*nVariables+j] = positions[i*nVariables+j];
	   }
   }
  
   
}                                          


__kernel void UpdateVelocities(                             
   __global float* velocities,                                           
   const unsigned int nVariables,
   float maximumVelocity,
   float inertia,
   __global float* randArray1,
   __global float* randArray2,
   __global float* c,
   __global float* positions,
   __global float* bestParticlePostions,
   __global float* bestGlobalParticlePositions)
{
	int i = get_global_id(0);
	float normOfVelocity = 0;
	for (int j = 0; j < nVariables; j++)
	{
		velocities[i*nVariables+j] = inertia*velocities[i*nVariables+j] +
										c[0]*(randArray1[i*nVariables+j])*(bestParticlePostions[i*nVariables+j]-positions[i*nVariables+j]) +
										c[1]*(randArray2[i*nVariables+j])*(bestGlobalParticlePositions[j]-positions[i*nVariables+j]);
		
		normOfVelocity += pow(velocities[i*nVariables+j],2.0f);
	}
	
	normOfVelocity = sqrt(normOfVelocity);
	if (normOfVelocity > maximumVelocity)
	{
		for (int j = 0; j < nVariables; j++)
		{
			velocities[i*nVariables+j] = velocities[i*nVariables+j] * maximumVelocity / normOfVelocity;
		}
	}
	
	for (int j = 0; j < nVariables; j++)
	{
		
		positions[i*nVariables+j] += velocities[i*nVariables+j];
		
		
	}
}   

__kernel void reduce(
            __global float* bestParticleValue,
			__global float* bestParticlePositions,
			__global float* bestGlobalParticleValue,
			__global float* bestGlobalParticlePosition,
			__global int* indexHolder,
			__local int* tmpIndexHolder,
            __local float* scratch,
            __const int length,
            __global float* result) {

				
  int global_index = get_global_id(0);
  int local_index = get_local_id(0);
  
  // Load data into local memory
  if (global_index < length) {
    scratch[local_index] = bestParticleValue[global_index];
	tmpIndexHolder[local_index] = indexHolder[global_index];
  } else {
    // Infinity is the identity element for the min operation
    scratch[local_index] = INFINITY;
	tmpIndexHolder[local_index] = INFINITY;
  }
  barrier(CLK_LOCAL_MEM_FENCE);
  for(int offset = 1;
      offset < get_local_size(0);
      offset <<= 1) {
    int mask = (offset << 1) - 1;
    if ((local_index & mask) == 0) {
      float other = scratch[local_index + offset];
      float mine = scratch[local_index];
      scratch[local_index] = (mine < other) ? mine : other;
	  tmpIndexHolder[local_index] = (mine < other) ? tmpIndexHolder[local_index] : tmpIndexHolder[local_index + offset];
    }
    barrier(CLK_LOCAL_MEM_FENCE);
  }
  if (local_index == 0) {
    result[get_group_id(0)] = scratch[0];
	int minIndex = (int)tmpIndexHolder[0];
	result[1] = minIndex;
	
	if (bestParticleValue[minIndex] < bestGlobalParticleValue[0])
	{
		bestGlobalParticleValue[0] = bestParticleValue[minIndex];
		for( int j = 0; j < 2; j++)
		{
			bestGlobalParticlePosition[j] = bestParticlePositions[minIndex*2+j];
		}
	}
  }
}	
	 
	 
__kernel void Test(                             
	__global float* positions,                                           
	__global float* bestParticleValues,
	__global float* bestParticlePositions,
	const unsigned int nVariables,
	__global float* bestGlobalParticleValue,
	__global float* bestGlobalParticlePosition,
	__global int* indexHolder,
	__local int* tmpIndexHolder,
	__local float* scratch,
	__const int length,
	__global float* result)               
{

   int i = get_global_id(0);  
   int local_index = get_local_id(0);          
   float fitness;

   fitness = 1 + pow( -13 + positions[i*nVariables+0] - pow(positions[i*nVariables+1],3) + 5*pow(positions[i*nVariables+1],2.0f) - 2*positions[i*nVariables+1],2.0f) +
	    pow(-29 + positions[i*nVariables+0] +pow(positions[i*nVariables+1],3.0f) + pow(positions[i*nVariables+1],2.0f) - 14*positions[i*nVariables+1],2.0f);

   if ( fitness < bestParticleValues[i])
   {
	   bestParticleValues[i] = fitness;
	   for ( int j = 0; j < nVariables; j++)
	   {
		   bestParticlePositions[i*nVariables+j] = positions[i*nVariables+j];
	   }
   }
  
   // Load data into local memory
   if (i < length) {
     scratch[local_index] = bestParticleValues[i];
 	tmpIndexHolder[local_index] = indexHolder[i];
   } else {
     // Infinity is the identity element for the min operation
     scratch[local_index] = INFINITY;
 	tmpIndexHolder[local_index] = INFINITY;
   }

   barrier(CLK_LOCAL_MEM_FENCE);
   for(int offset = 1;
       offset < get_local_size(0);
       offset <<= 1) {
     int mask = (offset << 1) - 1;
     if ((local_index & mask) == 0) {
       float other = scratch[local_index + offset];
       float mine = scratch[local_index];
       scratch[local_index] = (mine < other) ? mine : other;
 	  tmpIndexHolder[local_index] = (mine < other) ? tmpIndexHolder[local_index] : tmpIndexHolder[local_index + offset];
     }
     barrier(CLK_LOCAL_MEM_FENCE);
   }
   if (local_index == 0) {
     result[get_group_id(0)] = scratch[0];
 	int minIndex = (int)tmpIndexHolder[0];
 	result[1] = minIndex;
	
 	if (bestParticleValues[minIndex] < bestGlobalParticleValue[0])
 	{
 		bestGlobalParticleValue[0] = bestParticleValues[minIndex];
 		for( int j = 0; j < 2; j++)
 		{
 			bestGlobalParticlePosition[j] = bestParticlePositions[minIndex*2+j];
 		}
 	}
   }
   
   
  
   
}                                          

	               