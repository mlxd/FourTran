#include <stdio.h>
#include <cstdlib>
#include <cufftXt.h>
#include <cuda.h>
#include <iostream>
#include <cassert>


// ********************************************************* //
// Error checking macros
// ********************************************************* //

#define ERR_CHECK(err_val) {                                    \
	cudaError_t err = err_val;                                  \
	if (err != cudaSuccess) {                                   \
		fprintf(stderr, "Error %s at line %d in file %s\n",	    \
				cudaGetErrorString(err), __LINE__, __FILE__);   \
		exit(1);												\
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

// ********************************************************* //
// FT params for ...many plans
// ********************************************************* //

typedef struct FTParams{
        int numTranforms;
        int numLoops;
        int stride;
        int dist;
        int offset;
};

// ********************************************************* //
// Sample cuda kernels
// ********************************************************* //

__host__ __device__ double retVal(double val) {
    return val;
}

__global__ void copyVal(double *inData, double *outData) {
	outData[threadIdx.x] = inData[threadIdx.x];
}

// ********************************************************* //
// FT test functions
// ********************************************************* //

void fftIt2D(){
	cudaDeviceReset();
	int dimSize = 6;
    int NX = dimSize*dimSize;

    int numTransform = std::round(sqrt(NX));
    int sqrtNX = std::round(sqrt(NX));
    int dims1D[] = {(NX)};
    int dims2D[] = {sqrtNX,sqrtNX};
	
	int rank = 1;

    int inembed[] = {NX}; 
    int onembed[] = {NX};
    int istride[] = {1,sqrtNX}; // Consecutive elements, same signal 
    int ostride[] = {1,sqrtNX}; //
    int idist[] = {sqrtNX,1}; // Consecutive signals
    int odist[] = {sqrtNX,1};

    cufftHandle planMany, plan1D, plan2D;
    cufftDoubleComplex *data_H1DFFT, *data_H2DF1DI, *data_HmanyFFT, *data_H0, *data_D;

    ERR_CHECK( cudaMalloc( (cufftDoubleComplex**) &data_D, sizeof(cufftDoubleComplex)*NX) );
    data_H1DFFT = (cufftDoubleComplex*) malloc(sizeof(cufftDoubleComplex)*NX);
    data_H2DF1DI = (cufftDoubleComplex*) malloc(sizeof(cufftDoubleComplex)*NX);
    data_HmanyFFT = (cufftDoubleComplex*) malloc(sizeof(cufftDoubleComplex)*NX);
    data_H0 = (cufftDoubleComplex*) malloc(sizeof(cufftDoubleComplex)*NX);


    // ********************************************************* //
    // Create the input data
    // ********************************************************* //
    std::cout << "INPUT:\n";
    for(int ii=0; ii<sqrtNX; ++ii){
        for(int jj=0; jj<sqrtNX; ++jj){
            data_H0[jj + ii*sqrtNX].x = (double) ii;
            data_H0[jj + ii*sqrtNX].y = (double) jj;
            std::cout << data_H0[jj + ii*sqrtNX].x << " + 1i*" << data_H0[jj + ii*sqrtNX].y << "\t";
        }
        std::cout << "\n";
    }
    std::cout << "\n";

    ERR_CHECK( cudaMemcpy(data_D, data_H0, sizeof(cufftDoubleComplex) * NX, cudaMemcpyHostToDevice));

    // ******************************************************************************** //
    // First, check the 1D FFT along the standard dimension
    // ******************************************************************************** //
    FFT_ERR_CHECK(cufftPlan1d(&plan1D, dims2D[0], CUFFT_Z2Z, numTransform));
    FFT_ERR_CHECK(cufftExecZ2Z(plan1D, data_D, data_D, CUFFT_FORWARD));
    ERR_CHECK(cudaMemcpy(data_H1DFFT, data_D, sizeof(cufftDoubleComplex) * NX, cudaMemcpyDeviceToHost));
    std::cout << "OUTPUT 1D:\n";
    for(int ii=0; ii<sqrtNX; ++ii){
        for(int jj=0; jj<sqrtNX; ++jj){
            std::cout << data_H1DFFT[jj + ii*sqrtNX].x << " + 1i*" << data_H1DFFT[jj + ii*sqrtNX].y << "\t";
        }
        std::cout << "\n";
    }

/*
    //Check the inverse FFT for errors
    FFT_ERR_CHECK(cufftExecZ2Z(plan1D, data_D, data_D, CUFFT_INVERSE));
    ERR_CHECK(cudaMemcpy(data_H1DFFT, data_D, sizeof(cufftDoubleComplex) * NX, cudaMemcpyDeviceToHost));

    std::cout << "OUTPUT 1D Inverse:\n";
    for(int ii=0; ii<sqrtNX; ++ii){
        for(int jj=0; jj<sqrtNX; ++jj){
            std::cout << data_H[jj + ii*sqrtNX].x/sqrtNX << " + 1i*" << data_H[jj + ii*sqrtNX].y/sqrtNX << "\t";
        }
        std::cout << "\n";
    }
*/
    //Overwrite GPU data to original values
    ERR_CHECK( cudaMemcpy(data_D, data_H0, sizeof(cufftDoubleComplex) * NX, cudaMemcpyHostToDevice));

    // ******************************************************************************** //
    // ******************************************************************************** //
    // Next, check the Many FFT along the same expected dimension
    // ******************************************************************************** //

    FFT_ERR_CHECK(cufftPlanMany(&planMany, rank, dims2D, inembed, istride[0], idist[0], onembed, ostride[0], odist[0], CUFFT_Z2Z, numTransform));
    FFT_ERR_CHECK(cufftExecZ2Z(planMany, data_D, data_D, CUFFT_FORWARD));

    //ERR_CHECK(cudaDeviceSynchronize());
    ERR_CHECK(cudaMemcpy(data_HmanyFFT, data_D, sizeof(cufftDoubleComplex) * NX, cudaMemcpyDeviceToHost));

    std::cout << "OUTPUT MANY 1D:\n";
    for(int ii=0; ii<sqrtNX; ++ii){
        for(int jj=0; jj<sqrtNX; ++jj){
            std::cout << data_HmanyFFT[jj + ii*sqrtNX].x << " + 1i*" << data_HmanyFFT[jj + ii*sqrtNX].y << "\t";
        }
        std::cout << "\n";
    }

	try {
		for (int ii=0; ii<NX; ++ii){
			assert( (data_H1DFFT[ii].x - data_HmanyFFT[ii].x) < 1e-7  );
			assert( (data_H1DFFT[ii].y - data_HmanyFFT[ii].y) < 1e-7  );
		}
	} catch (const char* msg) {
		std::cerr << msg << std::endl;
	}
    //Overwrite GPU data to original values
    ERR_CHECK( cudaMemcpy(data_D, data_H0, sizeof(cufftDoubleComplex) * NX, cudaMemcpyHostToDevice));

    // ******************************************************************************** //
    // Check the 2D FFT Forward, 1D FFT back
    // ******************************************************************************** //
    FFT_ERR_CHECK(cufftPlan2d(&plan2D, dims2D[0], dims2D[1], CUFFT_Z2Z));
    FFT_ERR_CHECK(cufftExecZ2Z(plan2D, data_D, data_D, CUFFT_FORWARD));
    FFT_ERR_CHECK(cufftExecZ2Z(plan1D, data_D, data_D, CUFFT_INVERSE));

    ERR_CHECK(cudaMemcpy(data_H2DF1DI, data_D, sizeof(cufftDoubleComplex) * NX, cudaMemcpyDeviceToHost));
    std::cout << "OUTPUT 2DF-1DI:\n";
    for(int ii=0; ii<sqrtNX; ++ii){
        for(int jj=0; jj<sqrtNX; ++jj){
            std::cout << data_H2DF1DI[jj + ii*sqrtNX].x/sqrtNX << " + 1i*" << data_H2DF1DI[jj + ii*sqrtNX].y/sqrtNX << "\t";
        }
        std::cout << "\n";
    }
    //Overwrite GPU data to original values
    ERR_CHECK( cudaMemcpy(data_D, data_H0, sizeof(cufftDoubleComplex) * NX, cudaMemcpyHostToDevice));

    // ******************************************************************************** //
    // ******************************************************************************** //
    // Lastly, check the Many FFT along the other dimension
    // ******************************************************************************** //

    FFT_ERR_CHECK(cufftPlanMany(&planMany, rank, dims2D, inembed, istride[1], idist[1], onembed, ostride[1], odist[1], CUFFT_Z2Z, sqrtNX));
/*	for(int ii=0; ii<sqrtNX; ++ii){
    	FFT_ERR_CHECK(cufftExecZ2Z(planMany, &data_D[0], &data_D[0], CUFFT_FORWARD));
	}*/
    FFT_ERR_CHECK(cufftExecZ2Z(planMany, data_D, data_D, CUFFT_FORWARD));

    //ERR_CHECK(cudaDeviceSynchronize());
    ERR_CHECK(cudaMemcpy(data_HmanyFFT, data_D, sizeof(cufftDoubleComplex) * NX, cudaMemcpyDeviceToHost));

    std::cout << "OUTPUT MANY 1D Other:\n";
    for(int ii=0; ii<sqrtNX; ++ii){
        for(int jj=0; jj<sqrtNX; ++jj){
            std::cout << data_HmanyFFT[jj + ii*sqrtNX].x << " + 1i*" << data_HmanyFFT[jj + ii*sqrtNX].y << "\t";
        }
        std::cout << "\n";
    }
	try {
		for (int ii=0; ii<NX; ++ii){
			//std::cout << ( (data_H2DF1DI[ii].x/sqrtNX - data_HmanyFFT[ii].x) < 1e-7  ) << "\n";
		    assert( (data_H2DF1DI[ii].x/sqrtNX - data_HmanyFFT[ii].x)  < 1e-7 );
			assert( (data_H2DF1DI[ii].y/sqrtNX - data_HmanyFFT[ii].y)  < 1e-7 );
		}
	} catch (const char* msg) {
		std::cerr << msg << std::endl;
	}

    //Overwrite GPU data to original values
    ERR_CHECK( cudaMemcpy(data_D, data_H0, sizeof(cufftDoubleComplex) * NX, cudaMemcpyHostToDevice));

    // ******************************************************************************** //
    // ******************************************************************************** //
    // Free stuff
    // ******************************************************************************** //

    cufftDestroy(planMany);cufftDestroy(plan1D);cufftDestroy(plan2D);
    cudaFree(data_D);
    free(data_HmanyFFT);free(data_H0);
    free(data_H2DF1DI);free(data_H1DFFT);
}


void fftIt3D(){
    int dimLength = 5;
	int NX = dimLength*dimLength*dimLength;

    int cbrtNX = std::cbrt(NX);
    int numTransform = cbrtNX*cbrtNX;

    int paramsMatrix[3][5] = {{cbrtNX*cbrtNX,1,1,cbrtNX,0},{cbrtNX,cbrtNX,cbrtNX,1,cbrtNX*cbrtNX},{cbrtNX*cbrtNX,1,cbrtNX*cbrtNX,1,0}};
    FTParams params[3];

    for(int ii=0; ii<3; ++ii){
        params[ii].numTranforms = paramsMatrix[ii][0];
        params[ii].numLoops = paramsMatrix[ii][1];
        params[ii].stride = paramsMatrix[ii][2];
        params[ii].dist = paramsMatrix[ii][3];
        params[ii].offset = paramsMatrix[ii][4];
    }

    int dims[] = {NX};
    int dims3D[] = {cbrtNX,cbrtNX,cbrtNX};

    int inembed[] = {cbrtNX,cbrtNX,cbrtNX};
    int onembed[] = {cbrtNX,cbrtNX,cbrtNX};
    int istride[] = {1,cbrtNX,cbrtNX*cbrtNX}; // Indexed value is respective dimensionality of the transform along a specific dimension.
    int ostride[] = {1,cbrtNX,cbrtNX*cbrtNX};
    int idist[] = {cbrtNX,1,1}; // [Here][] // The next dimension
    int odist[] = {cbrtNX,1,1};

    cufftHandle planMany, plan1D, plan3D;
    cufftDoubleComplex *data_H1DFFT, *data_HmanyFFT, *data_H0, *data_D;

    ERR_CHECK( cudaMalloc( (cufftDoubleComplex**) &data_D, sizeof(cufftDoubleComplex)*NX) );

    data_H1DFFT = (cufftDoubleComplex*) malloc(sizeof(cufftDoubleComplex)*NX);
    data_HmanyFFT = (cufftDoubleComplex*) malloc(sizeof(cufftDoubleComplex)*NX);
    data_H0 = (cufftDoubleComplex*) malloc(sizeof(cufftDoubleComplex)*NX);


    // ******************************************************************************** //
    // Create the input data
    // ******************************************************************************** //
    std::cout << "INPUT:\n";
    for(int ii=0; ii<cbrtNX; ++ii){
		std::cout << "C(:,:," << ii+1 << ")=[";
        for(int jj=0; jj<cbrtNX; ++jj){
            for(int kk=0; kk<cbrtNX; ++kk){
                data_H0[kk + cbrtNX*(jj + ii*cbrtNX)].x = (double) ii;
                data_H0[kk + cbrtNX*(jj + ii*cbrtNX)].y = (double) jj;//(double) jj;
                std::cout << data_H0[kk + cbrtNX*(jj + ii*cbrtNX)].x << " + 1i*" << data_H0[kk + cbrtNX*(jj + ii*cbrtNX)].y << "\t";
            }
            std::cout << "\n";
        }
        std::cout << "]\n";
    }
    std::cout << "\n --- \n";

    ERR_CHECK( cudaMemcpy(data_D, data_H0, sizeof(cufftDoubleComplex) * NX, cudaMemcpyHostToDevice));

    // ******************************************************************************** //
    // First, check the 1D FFT along the standard dimension
    // ******************************************************************************** //
    FFT_ERR_CHECK(cufftPlan1d(&plan1D, cbrtNX, CUFFT_Z2Z, cbrtNX*cbrtNX));
    FFT_ERR_CHECK(cufftExecZ2Z(plan1D, data_D, data_D, CUFFT_FORWARD));
    ERR_CHECK(cudaMemcpy(data_H1DFFT, data_D, sizeof(cufftDoubleComplex) * NX, cudaMemcpyDeviceToHost));
    std::cout << "OUTPUT 1D_1:\n";
    for(int ii=0; ii<cbrtNX; ++ii){
        for(int jj=0; jj<cbrtNX; ++jj){
            for(int kk=0; kk<cbrtNX; ++kk){
                std::cout << data_H1DFFT[kk + cbrtNX*(jj + ii*cbrtNX)].x << " + 1i*" << data_H1DFFT[kk + cbrtNX*(jj + ii*cbrtNX)].y << "\t";
            }
            std::cout << "\n";
        }
        std::cout << "\n";
    }
    std::cout << "\n --- \n";

    //Overwrite GPU data to original values
    ERR_CHECK( cudaMemcpy(data_D, data_H0, sizeof(cufftDoubleComplex) * NX, cudaMemcpyHostToDevice));

    // ******************************************************************************** //
    // ******************************************************************************** //
    // Next, check the Many FFT along the same expected dimension
    // ******************************************************************************** //
    int tDim = 0; //Transform dimension
	
	int dims2D[] = {cbrtNX,cbrtNX};
   	FFT_ERR_CHECK(cufftPlanMany(&planMany, 1, dims2D, inembed, cbrtNX, 1, onembed, cbrtNX, 1, CUFFT_Z2Z, cbrtNX));
    for (int ii=0; ii<cbrtNX; ++ii){
		FFT_ERR_CHECK(cufftExecZ2Z(planMany, &data_D[ii*cbrtNX*cbrtNX], &data_D[ii*cbrtNX*cbrtNX] , CUFFT_FORWARD));
	}

    /*for (int ft=0; ft < params[tDim].numLoops; ++ft){
        FFT_ERR_CHECK(cufftPlanMany(&planMany, 1, dims, inembed, params[tDim].stride, params[tDim].dist, onembed, params[tDim].stride, params[tDim].dist, CUFFT_Z2Z, params[tDim].numTranforms));
        //FFT_ERR_CHECK(cufftExecZ2Z(planMany, data_D + ((int) pow(cbrtNX,tDim)), data_D + ((int) pow(cbrtNX,tDim)), CUFFT_FORWARD));
        FFT_ERR_CHECK(cufftExecZ2Z(planMany, &data_D[ft*params[tDim].offset], &data_D[ft*params[tDim].offset], CUFFT_FORWARD));
    }*/
    //ERR_CHECK(cudaDeviceSynchronize());
    ERR_CHECK(cudaMemcpy(data_HmanyFFT, data_D, sizeof(cufftDoubleComplex) * NX, cudaMemcpyDeviceToHost));

    std::cout << "OUTPUT MANY 1D:\n";
    for(int ii=0; ii<cbrtNX; ++ii){
            for(int jj=0; jj<cbrtNX; ++jj){
                for(int kk=0; kk<cbrtNX; ++kk){
                    std::cout << data_HmanyFFT[kk + cbrtNX*(jj + ii*cbrtNX)].x << " + 1i*" << data_HmanyFFT[kk + cbrtNX*(jj + ii*cbrtNX)].y << "\t";
                }
                std::cout << "\n";
            }
            std::cout << "\n";
        }
        std::cout << "\n --- \n";

    //Overwrite GPU data to original values
    ERR_CHECK( cudaMemcpy(data_D, data_H0, sizeof(cufftDoubleComplex) * NX, cudaMemcpyHostToDevice));

    // ******************************************************************************** //
    // ******************************************************************************** //
    // Free stuff
    // ******************************************************************************** //

    cufftDestroy(planMany);cufftDestroy(plan1D);cufftDestroy(plan3D);
    cudaFree(data_D);
    free(data_HmanyFFT);free(data_H1DFFT);free(data_H0);
}

void fftMulti3D(){
   	int GPU_N;
    cudaGetDeviceCount(&GPU_N);
	
	std::cout << "Num. Devices = " << GPU_N << "\n";

	cufftHandle plan_input; cufftResult result;
	FFT_ERR_CHECK( cufftCreate(&plan_input) );

	int nGPUs = 2;
	int* whichGPUs = (int*) malloc(sizeof(int)*nGPUs);
	for (int i=0; i<nGPUs; ++i){
		whichGPUs[i] = i;
	}
	FFT_ERR_CHECK( cufftXtSetGPUs(plan_input, nGPUs, whichGPUs) );
    

    //Print the device information to run the code
     for (int i = 0 ; i < nGPUs ; i++)
     {
         cudaDeviceProp deviceProp;
         ERR_CHECK( cudaGetDeviceProperties(&deviceProp, whichGPUs[i]) );
         printf("GPU Device %d: \"%s\" with compute capability %d.%d\n", whichGPUs[i], deviceProp.name, deviceProp.major, deviceProp.minor);

     }

	size_t* worksize = (size_t*) malloc(sizeof(size_t)*nGPUs);
    cufftDoubleComplex *host_data_input, *host_data_output;

    int nx=4,ny,nz;
	ny = nx; nz = ny;

    int size_of_data = sizeof(cufftDoubleComplex) * nx * ny * nz;
    host_data_input = (cufftDoubleComplex*) malloc(size_of_data);
    host_data_output = (cufftDoubleComplex*) malloc(size_of_data);

	for(int ii=0; ii<nx; ++ii){	
		std::cout << "C(:,:," << ii+1 << ")=[";
		for(int jj=0; jj<ny; ++jj){	
			for(int kk=0; kk<nz; ++kk){
				host_data_input[kk + ny*(jj + nx*ii)].x = kk;//(double) kk + ny*(jj + nx*ii);
				host_data_input[kk + ny*(jj + nx*ii)].y = jj;//(double) kk + ny*(jj + nx*ii);
 				std::cout << host_data_input[kk + ny*(jj + nx*ii)].x << "+1i*" << host_data_input[kk + ny*(jj + nx*ii)].y << "  ";
			}	std::cout << "\n";
		}	std::cout << "]\n";
	}

    //FFT_ERR_CHECK( cufftMakePlan3d (plan_input, nz, ny, nx, CUFFT_Z2Z, worksize) );
	int rank = 3; //3D
	long long int dims[3] = {nx,ny,nz};
    FFT_ERR_CHECK( 
		cufftXtMakePlanMany(plan_input, rank, dims, NULL, 0LL, 0LL, CUDA_C_64F, NULL, 0LL, 0LL, CUDA_C_64F, 1LL, worksize, CUDA_C_64F) 
	);//not supported under cufft7.5
    
	cudaLibXtDesc *device_data_input;
    FFT_ERR_CHECK( cufftXtMalloc (plan_input, &device_data_input, CUFFT_XT_FORMAT_INPLACE) );
	FFT_ERR_CHECK( cufftXtMemcpy (plan_input, device_data_input, host_data_input, CUFFT_COPY_HOST_TO_DEVICE) );
	FFT_ERR_CHECK( cufftXtExecDescriptorZ2Z (plan_input, device_data_input, device_data_input, CUFFT_FORWARD) );
	FFT_ERR_CHECK( cufftXtMemcpy (plan_input, host_data_output, device_data_input, CUFFT_COPY_DEVICE_TO_HOST) );
	
	for(int ii=0; ii<nx; ++ii){			
		std::cout << "C(:,:," << ii+1 << ")=[";
		for(int jj=0; jj<ny; ++jj){	
			for(int kk=0; kk<nz; ++kk){
 				std::cout << host_data_output[kk + ny*(jj + nx*ii)].x << "+1i*" << host_data_output[kk + ny*(jj + nx*ii)].y << "  ";
			}	std::cout << "\n";
		}	std::cout << "]\n";
	}

    result = cufftXtFree(device_data_input);
	result = cufftDestroy(plan_input);

	free(host_data_input); free(host_data_output);
}

int main(void) {

    fftMulti3D();
    ERR_CHECK(cudaDeviceReset());

    return (0);
}
