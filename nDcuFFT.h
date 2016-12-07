//
// Created by Lee James O'Riordan on 12/6/16.
//

#ifndef FOURTRAN_H
#define FOURTRAN_H

// ********************************************************* //
// Fourier transform params for ...many plans
// ********************************************************* //

typedef struct FTParams{
    int numDims; 		//Number of dimensions in the data-set
    int numElem;		//Total number of elements in the data-set
    int *dims;			//Array of the dimension sizes
    int *numTransforms;	//Number of FT to perform
    int *numLoops;		//Number of for-loop with which to shift the address pointer
    int *stride;		//Stride for data-set
    int *dist;			//Distance between consecutive realisations
    int *offset;		//Pointer shift offset
};

/* This function generates all the required spacings for the transforms along any of the required dimensions.
 * numDims specfies the number of total dimensions, dimSize is an array with the size of these dimensions, and params is
 * an array of size FTParams*numDims to hold the resulting parameters.
 *
 * To generate the required parameters I will assume they are in the following order:
 * 1. numberTransforms | 2. numberForLoops | 3. dataStride | 4. dataDist | 5. pointerOffset
 *
 * 1. The number of Fourier transforms to perform with the given parameters.
 * 2. The number of loops to perform the transforms with. Useful when shifting the pointer by pointerOffset.
 * 3. The spacing between data in a single dimension.
 * 4. The spacing between consecutive data sets.
 * 5. The value to shift the pointer position of the data to be transformed. When not using first or last dimension.
 *
 * For a 4D dataset XYZW, the transform params are calculated as (while not fully generalised):
 *  X:  prodVals[numDim + 1]   prodVals[0]                          prodVals[0]     prodVals[1]      0
 *  Y:  prodVals[1]            prodVals[(numDim+2) % (2*numDim)]    prodVals[1]     prodVals[0]     [2]
 *  Z:  prodVals[2]            prodVals[(numDim+3) % (2*numDim)]    prodVals[2]     prodVals[0]     [3]
 *  W:  prodVals[numDim-1]     prodVals[(numDim+4) % (2*numDim)]    prodVals[3]     prodVals[0]      0
 *
 *  where prodVals is the list of products between adjacent pairings of dimension sizes (think triangle of products)
 *  prodVals[0] = 1; prodVals[1] = xSize; prodVals[2] = xSize*ySize; prodVals[3] =  xSize*ySize*zSize;
 *  prodVals[4] = xSize*ySize*zSize*wSize; prodVals[5] = ySize*zSize*wSize; prodVals[6] = zSize*wSize;
 *  prodVals[7] =  zSize;
 */
void ftParamsInit(int numDims, int* dimSize, FTParams *params);

/*
*	Send data from host to device, transform, send data back to host, return host pointer
*/
void fft_HDH(FTParams *params, int tDim, const double2 *dataIn, double2 *dataOut);

/*
*	Send data from host to device, transform, return device pointer
*/
void fft_HD(FTParams *params, int tDim, const double2 *dataIn, double2 *dataOut);

/*
*	Transform data on device, return device pointer
*/
void fft_DD(FTParams *params, int tDim, const double2 *dataIn, double2 *dataOut);

#endif //FOURTRAN_H
