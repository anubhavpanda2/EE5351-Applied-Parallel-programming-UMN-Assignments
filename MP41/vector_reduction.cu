#ifdef _WIN32
#  define NOMINMAX 
#endif

#define NUM_ELEMENTS 512

// includes, system
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>
#include <float.h>

// includes, kernels
#include "vector_reduction_kernel.cu"

////////////////////////////////////////////////////////////////////////////////
// declaration, forward
void runTest( int argc, char** argv);

int ReadFile(float*, char* file_name);
float computeOnDevice(float* h_data, int array_mem_size);

extern "C" 
void computeGold( float* reference, float* idata, const unsigned int len);

////////////////////////////////////////////////////////////////////////////////
// Program main
////////////////////////////////////////////////////////////////////////////////
int 
main( int argc, char** argv) 
{
    runTest( argc, argv);
    return EXIT_SUCCESS;
}

////////////////////////////////////////////////////////////////////////////////
//! Run naive scan test
////////////////////////////////////////////////////////////////////////////////
void
runTest( int argc, char** argv) 
{
    int num_elements = NUM_ELEMENTS;
    int errorM = 0;

    const unsigned int array_mem_size = sizeof( float) * num_elements;

    // allocate host memory to store the input data
    float* h_data = (float*) malloc( array_mem_size);

    // * No arguments: Randomly generate input data and compare against the 
    //   host's result.
    // * One argument: Read the input data array from the given file.
    switch(argc-1)
    {      
        case 1:  // One Argument
            errorM = ReadFile(h_data, argv[1]);
            if(errorM != num_elements)
            {
                printf("Error reading input file!\n");
                exit(1);
            }
        break;
        
        default:  // No Arguments or one argument
            // initialize the input data on the host to be integer values
            // between 0 and 1000
            for( unsigned int i = 0; i < num_elements; ++i) 
            {
                h_data[i] = floorf(1000*(rand()/(float)RAND_MAX));
            }
        break;  
    }
    // compute reference solution
    float reference = 0.0f;  
    computeGold(&reference , h_data, num_elements);
    
    // **===-------- Modify the body of this function -----------===**
    float result = computeOnDevice(h_data, num_elements);
    // **===-----------------------------------------------------------===**


    // We can use an epsilon of 0 since values are integral and in a range 
    // that can be exactly represented
    float epsilon = 0.0f;
    unsigned int result_regtest = (abs(result - reference) <= epsilon);
    printf( "Test %s\n", (1 == result_regtest) ? "PASSED" : "FAILED");
    printf( "device: %f  host: %f\n", result, reference);
    // cleanup memory
    free( h_data);
}

// Read a floating point vector into M (already allocated) from file
int ReadFile(float* V, char* file_name)
{
    unsigned int data_read = NUM_ELEMENTS;
    FILE* input = fopen(file_name, "r");
    unsigned i = 0;
    for (i = 0; i < data_read; i++) 
        fscanf(input, "%f", &(V[i]));
    return data_read;
}

// **===----------------- Modify this function ---------------------===**
// Take h_data from host, copies it to device, setup grid and thread 
// dimentions, excutes kernel function, and copy result of reduction back
// to h_data.
// Note: float* h_data is both the input and the output of this function.
float computeOnDevice(float* h_data, int num_elements)
{
int size=num_elements*sizeof(float);
printf("%d",size);
  // placeholder
float* g_data;
int n;
cudaMalloc((void**)&g_data, size);
// cudaMalloc(&n, sizeof(int));
dim3 dimBlock(16,1,1);
    dim3 dimGrid(ceil(num_elements/32.0),1,1);
cudaMemcpy(g_data, h_data, size,cudaMemcpyHostToDevice);
reduction<<<dimGrid,dimBlock>>>(g_data,num_elements);

//cudaMemcpy(n,num_elements, size,cudaMemcpyHostToDevice);
cudaMemcpy(h_data,g_data, size, cudaMemcpyDeviceToHost);
//reduction<<<dimGrid,dimBlock>>>(g_data,num_elements);
printf("%f",h_data[0]);
cudaFree(g_data);
for(int i=1;i<ceil(num_elements/32.0);i++)
h_data[0]+=h_data[i];
//cudaFree(n);
  return h_data[0];
}
