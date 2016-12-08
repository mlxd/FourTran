//
// Created by Lee James O'Riordan on 12/6/16.
//

#include "nDcuFFT.h"

#include <stdio.h>
#include <cstdlib>
#include <cufftXt.h>
#include <cuda.h>
#include <iostream>
#include <cmath>
#include <vector>

//#####################################################################
//#####################################################################
// These are adapted from macros in the CUDA samples and are exempt 
// from my licensing choices.

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

//#####################################################################
//#####################################################################

void ftParamsInit(int numDims, int* dimSize, FTParams *params){
    int i = 0; //Loop variable, to be used in two separate loops
    int totalElements = 1;
    int *prodVals = (int*) malloc((2*numDims)*sizeof(int)); //2N-1 pairings + 1 for unity at index [0]
    prodVals[0] = 1; //Begin by declaring [0] as unity, to determine higher indices based on previous
    params->numDims = numDims;
    //Determine maximum number of elements in dataset
    for (i = 0; i < numDims; ++i)
        totalElements *= dimSize[i];

    //Calculate the products of adjacent data sizes
    for (i = 1; i <= (2*numDims-1); ++i) {
        if( i <= numDims ){ prodVals[i] = prodVals[i-1] * dimSize[i-1]; }
        else              { prodVals[i] = std::ceil( prodVals[numDims] / prodVals[i - numDims] ); }
    }

    /* Populate the parameter struct array with the appropriate values per dimension.
     * I should write a blog post on how this works.
     * Here's hoping that I remember to do so. Falls out of above though. */
    params->dims = dimSize;
   	params->numElem = totalElements;
    for( i = 0; i < numDims; ++i ){
        params->numTransforms[i] = (i == 0) ? totalElements/dimSize[i] : prodVals[i];
        params->numLoops[i]      = (i == 0) ? prodVals[0] : prodVals[(numDims+1 + i)%(2*numDims)]; // Start/End no trans
        params->stride[i]        = prodVals[i];
        params->dist[i]          = (i == 0) ? prodVals[1] : prodVals[0]; //element 1 for first entry, element 0 otherwise
        params->offset[i]        = ( (i == numDims-1) || (i == 0) ) ? 0 : prodVals[i+1]; //0 for first and last entries

        std::cout << "PARAMS["  << i << "]\n";
        std::cout << params->numTransforms[i]<< "\t"
                  << params->numLoops[i]     << "\t"
                  << params->stride[i]       << "\t"
                  << params->dist[i]         << "\t"
                  << params->offset[i]       << "\t"
                  << params->dims[i]         << "\n";
    }
}

/*
* Here we do the magical transformation
*/
void fft_HDH(FTParams *params, int tDim, double2 *dataIn, double2 *dataOut){
	cufftHandle plan;
	double2 *data_D;
	ERR_CHECK( 
		cudaMalloc((void**) &data_D, sizeof(double2) * params->numElem) );
	ERR_CHECK( 
		cudaMemcpy(data_D, dataIn, sizeof(double2) * params->numElem, cudaMemcpyHostToDevice) );
	FFT_ERR_CHECK(	
		cufftPlanMany(&plan, 1, 
			params->dims, 			params->dims,	params->stride[ tDim ], 
			params->dist[ tDim ], 	params->dims,	params->stride[ tDim ], 
			params->dist[ tDim ], 	CUFFT_Z2Z,		params->numTransforms[ tDim ]
		)
	);
	for ( int i = 0; i < params->numTransforms[ tDim ]; ++i ){
		FFT_ERR_CHECK( 
			cufftExecZ2Z( plan, 
				&data_D[ i * params->offset[tDim] ], 
				&data_D[ i * params->offset[tDim] ], 
				CUFFT_FORWARD
			)
		);
	}
	ERR_CHECK( cudaMemcpy(dataOut, data_D, sizeof(double2) * params->numElem, cudaMemcpyDeviceToHost) );
	cudaFree(data_D);
}

int main(){
    int numDims = 2;
    int dimSize[] = {9,9,3};
    FTParams params;

    params.numTransforms = (int*) malloc(sizeof(int)*numDims);
    params.numLoops = (int*) malloc(sizeof(int)*numDims);
    params.stride = (int*) malloc(sizeof(int)*numDims);
    params.dist = (int*) malloc(sizeof(int)*numDims);
    params.offset = (int*) malloc(sizeof(int)*numDims);

    ftParamsInit(numDims,dimSize, &params);

	double2 *dataIn, *dataOut;

    dataIn = (double2*) malloc(sizeof(double2) * params.numElem);
    dataOut = (double2*) malloc(sizeof(double2) * params.numElem);

    // ******************************************************************************** //
    // Create the input data
    // ******************************************************************************** //
    std::cout << "INPUT:\n";
    for( int ii=0; ii < params.dims[0]; ++ii ){
        std::cout << "C(:,:," << ii+1 << ")=[";
        for( int jj=0; jj < params.dims[1]; ++jj ){
            for( int kk=0; kk < params.dims[2]; ++kk ){
                dataIn[ kk + params.dims[1] * ( jj + ii * params.dims[0] ) ].x = (double) ii;
                dataIn[ kk + params.dims[1] * ( jj + ii * params.dims[0] ) ].y = (double) jj;
                std::cout << dataIn[ kk + params.dims[1] * ( jj + ii * params.dims[0] ) ].x << " + 1i*" << dataIn[ kk + params.dims[1] * ( jj + ii * params.dims[0] ) ].y << "\t";
            }
            std::cout << "\n";
        }
        std::cout << "]\n";
    }
    std::cout << "\n --- \n";

    //ERR_CHECK( cudaMalloc( (double2**) &dataD, sizeof(double2) * params.numElem) );
    //ERR_CHECK( cudaMemcpy(dataD, dataIn, sizeof(double2) * params.numElem, cudaMemcpyHostToDevice) );   	 
	//ERR_CHECK( cudaMemcpy(dataOut, dataD, sizeof(double2) * params.numElem, cudaMemcpyDeviceToHost) );

    // ******************************************************************************** //
    // Perform FFT 
    // ******************************************************************************** //

	fft_HDH(&params, 1, dataIn, dataOut);

    // ******************************************************************************** //
    // Show FFT results
    // ******************************************************************************** //
    std::cout << "OUTPUT:\n";
    for( int ii=0; ii < params.dims[0]; ++ii ){
        std::cout << "C(:,:," << ii+1 << ")=[";
        for( int jj=0; jj < params.dims[1]; ++jj ){
            for( int kk=0; kk < params.dims[2]; ++kk ){
                dataOut[ kk + params.dims[1] * ( jj + ii * params.dims[0] ) ].x = (double) ii;
                dataOut[ kk + params.dims[1] * ( jj + ii * params.dims[0] ) ].y = (double) jj;
                std::cout << dataOut[ kk + params.dims[1] * ( jj + ii * params.dims[0] ) ].x << " + 1i*" << dataOut[ kk + params.dims[1] * ( jj + ii * params.dims[0] ) ].y << "\t";
            }
            std::cout << "\n";
        }
        std::cout << "]\n";
    }
    std::cout << "\n --- \n";
	free (dataIn); free (dataOut);
}
