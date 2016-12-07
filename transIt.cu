#include <stdio.h>
#include <cstdlib>
#include <cufftXt.h>
#include <cuda.h>
#include <iostream>

#define RANK 1


#define ERR_CHECK(err_val) {                                    \
    cudaError_t err = err_val;                                  \
    if (err != cudaSuccess) {                                   \
        fprintf(stderr, "Error %s at line %d in file %s\n",     \
                cudaGetErrorString(err), __LINE__, __FILE__);   \
        exit(1);                                                \
    }                                                           \
}

#define FFT_ERR_CHECK(err_val) {                                \
    cufftResult err = err_val;                                  \
    if (err != CUFFT_SUCCESS) {                                 \
        fprintf(stderr, "Error %d at line %d in file %s\n",     \
                err, __LINE__, __FILE__);                       \
        exit(1);                                                \
    }                                                           \
}


/*
__host__ __device__ double retVal(double val) {
    return val;
}

__global__ void copyVal(double *inData, double *outData) {
    outData[threadIdx.x] = inData[threadIdx.x];
}
*/


void transIt(){
    int NX = 4;

    double *data_H, *data_H0, *data_D;

    int sqrtNX = sqrt(NX);
    int dims[] = {sqrtNX,sqrtNX};

    ERR_CHECK( cudaMalloc( (double**) &data_D, sizeof(double)*NX) );
    data_H = (double*) malloc(sizeof(double)*NX);
    data_H0 = (double*) malloc(sizeof(double)*NX);

    // ********************************************************* //
    // Create the input data
    // ********************************************************* //
    std::cout << "INPUT:\n";
    for(int ii=0; ii<sqrtNX; ++ii){
        for(int jj=0; jj<sqrtNX; ++jj){
            data_H0[jj + ii*sqrtNX] = (double) ii + jj*sqrtNX;
            std::cout << data_H0[jj + ii*sqrtNX] << "\t";
        }
        std::cout << "\n";
    }
    std::cout << "\n";

    ERR_CHECK( cudaMemcpy(data_D, data_H0, sizeof(double) * NX, cudaMemcpyHostToDevice));

    // ******************************************************************************** //
    // First, check the transpose on the host
    // ******************************************************************************** //
    double tmp = 0.;
    ERR_CHECK(cudaMemcpy(data_H, data_D, sizeof(double) * NX, cudaMemcpyDeviceToHost));
    std::cout << "OUTPUT 1D:\n";
    for(int ii=0; ii<sqrtNX-1; ++ii){
        for(int jj=ii+1; jj<sqrtNX; ++jj){
            tmp = data_H[jj + ii*sqrtNX];
            data_H[jj + ii*sqrtNX] = data_H[ii + jj*sqrtNX];
            data_H[ii + jj*sqrtNX] = tmp;
            std::cout << data_H[jj + ii*sqrtNX] << "\t";
        }
        std::cout << "\n";
    }

    //Check the inverse FFT for errors
    ERR_CHECK(cudaMemcpy(data_H, data_D, sizeof(double) * NX, cudaMemcpyDeviceToHost));

    std::cout << "OUTPUT 1D Inverse:\n";
    for(int ii=0; ii<sqrtNX; ++ii){
        for(int jj=0; jj<sqrtNX; ++jj){
            std::cout << data_H[jj + ii*sqrtNX] << "\t";
        }
        std::cout << "\n";
    }

    //Overwrite GPU data to original values
    ERR_CHECK( cudaMemcpy(data_D, data_H0, sizeof(double) * NX, cudaMemcpyHostToDevice));

    // ******************************************************************************** //
    // ******************************************************************************** //
    // Next, check the transpose on the device
    // ******************************************************************************** //

    ERR_CHECK(cudaMemcpy(data_H, data_D, sizeof(double) * NX, cudaMemcpyDeviceToHost));

    std::cout << "OUTPUT MANY 1D:\n";
    for(int ii=0; ii<sqrtNX; ++ii){
        for(int jj=0; jj<sqrtNX; ++jj){
            std::cout << data_H[jj + ii*sqrtNX] << "\t";
        }
        std::cout << "\n";
    }

    //Overwrite GPU data to original values
    ERR_CHECK( cudaMemcpy(data_D, data_H0, sizeof(double) * NX, cudaMemcpyHostToDevice));

    // ******************************************************************************** //
    // ******************************************************************************** //
    // Free stuff
    // ******************************************************************************** //

    cudaFree(data_D);
    free(data_H);free(data_H0);
}

int main(){
    transIt();
    return 0;
}
