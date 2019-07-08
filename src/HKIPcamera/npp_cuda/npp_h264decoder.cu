#include "npp_h264decoder.cuh"
#include <iostream>
#include <npp.h>
#include <nppi.h>
#include <npps.h>

void decodeH264(const unsigned char *input, unsigned char *output, int width,
                int height,int device_id) {

  int curDev = -1;
  cudaGetDevice(&curDev);
  if(curDev!=device_id){
    cudaSetDevice(device_id);
  }

  Npp8u *pNppInput;
  int nppInputStep;
  pNppInput = nppiMalloc_8u_C1(width, height/2*3, &nppInputStep);
  cudaMemcpy(pNppInput, input, sizeof(Npp8u) * width * height/2*3,
             cudaMemcpyHostToDevice);

  Npp8u *pNppInput_V_half = pNppInput + width * height;

  Npp8u *pNppInput_U_half = pNppInput_V_half + width * height /4;

  Npp8u *pNppInputArray[3] = {pNppInput, pNppInput_U_half, pNppInput_V_half};
  int nppInputSteps[3] = {width, width/2, width/2};

  Npp8u *pNppOutput;
  int nppOutputStep;
  pNppOutput = nppiMalloc_8u_C3(width, height, &nppOutputStep);

  NppiSize nppSize;
  nppSize.width = width;
  nppSize.height = height;
  nppiYUV420ToBGR_8u_P3C3R(pNppInputArray, nppInputSteps, pNppOutput, width * 3,
                           nppSize);

  cudaMemcpy(output, pNppOutput, sizeof(unsigned char) * width * height * 3,
             cudaMemcpyDeviceToHost);

  cudaFree(pNppInput);
  cudaFree(pNppOutput);

  if(curDev!=device_id){
    cudaSetDevice(curDev);
  }
}