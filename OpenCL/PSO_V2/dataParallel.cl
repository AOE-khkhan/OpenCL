__kernel void dataParallel(__global float* A, __global float* C)
{
    int base = get_global_id(0);
    
    C[base] = A[base];
}
