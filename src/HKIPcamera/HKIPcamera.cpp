// HKIPcamera.cpp
#include "HKIPcamera.h"
#include "HCNetSDK.h"
#include "LinuxPlayM4.h"
#include <cstdio>
#include <cstring>
#include <iostream>
#include <iostream>
#include <list>
#include <opencv2/opencv.hpp>
#include <pthread.h>
#include <time.h>
#include <unistd.h>

#define USECOLOR 1
#define WINAPI

using namespace cv;
using namespace std;

extern "C" void decodeH264(const unsigned char *input, unsigned char *output,
                           int width, int height, int device_id);
//--------------------------------------------

void yv12toYUV(char *outYuv, char *inYv12, int width, int height,
               int widthStep) {
  int col, row;
  unsigned int Y, U, V;
  int tmp;
  int idx;

  for (row = 0; row < height; row++) {
    idx = row * widthStep;
    int rowptr = row * width;

    for (col = 0; col < width; col++) {
      tmp = (row / 2) * (width / 2) + (col / 2);
      Y = (unsigned int)inYv12[row * width + col];
      U = (unsigned int)inYv12[width * height + width * height / 4 + tmp];
      V = (unsigned int)inYv12[width * height + tmp];
      outYuv[idx + col * 3] = Y;
      outYuv[idx + col * 3 + 1] = U;
      outYuv[idx + col * 3 + 2] = V;
    }
  }
}

void yv12toRGBMat(char *inYv12, int width, int height, cv::Mat &outMat,
                  int device_id = 0) {
  if (device_id < 0) {
    int yuvHeight = height * 3 / 2;
    int bufLen = width * yuvHeight;
    cv::Mat yuvImg;
    yuvImg.create(yuvHeight, width, CV_8UC1);
    memcpy(yuvImg.data, inYv12, bufLen * sizeof(unsigned char));
    cv::cvtColor(yuvImg, outMat, CV_YUV2BGR_YV12);
  } else {
    outMat = cv::Mat(height, width, CV_8UC3);
    decodeH264((unsigned char *)(void *)(inYv12), outMat.data, width, height,
               device_id);
  }
}

void CALLBACK DecCBFun(int nPort, char *pBuf, int nSize, FRAME_INFO *pFrameInfo,
                       void *pUser, int nReserved2) {
  long lFrameType = pFrameInfo->nType;
  HKIPcamera *hkipc = static_cast<HKIPcamera *>(pUser);
  bool buffer_full = false;
  pthread_mutex_lock(&(hkipc->frame_list_mutex_));
  if (hkipc->is_buffer_full()) {
    buffer_full = true;
  }
  pthread_mutex_unlock(&(hkipc->frame_list_mutex_));
  if (buffer_full) {
    return;
  }
  int device_id = hkipc->device_id_;
  if (lFrameType == T_YV12) {
#if USECOLOR
    Mat rgbMat;
    yv12toRGBMat(pBuf, pFrameInfo->nWidth, pFrameInfo->nHeight, rgbMat,
                 device_id);
#else
    Mat rgbMat;
    rgbMat.create(pFrameInfo->nHeight, pFrameInfo->nWidth, CV_8UC1);
    memcpy(rgbMat.data, pBuf, pFrameInfo->nWidth * pFrameInfo->nHeight);
#endif
    pthread_mutex_lock(&(hkipc->frame_list_mutex_));
    if (!hkipc->is_buffer_full()) {
      // Mat mat = cvarrToMat(pImg);
      hkipc->push_frame(rgbMat);
    }
    // cvReleaseImage(&pImg);
    pthread_mutex_unlock(&(hkipc->frame_list_mutex_));
  }
}

/// real time stream
void CALLBACK fRealDataCallBack(LONG lRealHandle, DWORD dwDataType,
                                BYTE *pBuffer, DWORD dwBufSize, void *pUser) {
  DWORD dRet;
  HKIPcamera *hkipc = static_cast<HKIPcamera *>(pUser);
  int &nPort = hkipc->nPort_;
  switch (dwDataType) {
  case NET_DVR_SYSHEAD:
    if (!PlayM4_GetPort(&nPort)) {
      break;
    }
    if (dwBufSize > 0) {

      if (!PlayM4_OpenStream(nPort, pBuffer, dwBufSize, 1024 * 1024)) {
        dRet = PlayM4_GetLastError(nPort);
        break;
      }

      if (!PlayM4_SetDecCallBackMend(nPort, DecCBFun, pUser)) {
        dRet = PlayM4_GetLastError(nPort);
        break;
      }

      if (!PlayM4_Play(nPort, NULL)) {
        dRet = PlayM4_GetLastError(nPort);
        break;
      }
    }
    break;

  case NET_DVR_STREAMDATA:

    if (dwBufSize > 0 && nPort != -1) {
      BOOL inData = PlayM4_InputData(nPort, pBuffer, dwBufSize);
      while (!inData) {
        sleep(10);
        inData = PlayM4_InputData(nPort, pBuffer, dwBufSize);
        cout << (L"PlayM4_InputData failed \n") << endl;
      }
    }
    break;
  }
}

void CALLBACK g_ExceptionCallBack(DWORD dwType, LONG user_id_, LONG lHandle,
                                  void *pUser) {
  char tempbuf[256] = {0};
  switch (dwType) {
  case EXCEPTION_RECONNECT:
    printf("----------reconnect--------%d\n", time(NULL));
    break;
  default:
    cout << "Exception : " << dwType << endl;
    break;
  }
}

bool HKIPcamera::OpenCamera(char *ip, long port, char *usr, char *password) {
  user_id_ = NET_DVR_Login_V30(ip, port, usr, password, &struDeviceInfo_);
  if (user_id_ >= 0) {

    // cout << "Log in success!" << endl;
    return TRUE;
  } else {
    printf("Login error, %d\n", NET_DVR_GetLastError());
    printf(" error code %ld\n", user_id_);
    NET_DVR_Cleanup();
    return FALSE;
  }
}
void *ReadCamera(void *inParam) {
  HKIPcamera *hkipc = static_cast<HKIPcamera *>(inParam);
  NET_DVR_SetExceptionCallBack_V30(0, NULL, g_ExceptionCallBack, NULL);

  LONG &user_id_ = hkipc->user_id_;
  NET_DVR_DEVICEINFO_V30 &struDeviceInfo = hkipc->struDeviceInfo_;
  long &channel = hkipc->channel_;
  long &streamtype = hkipc->streamtype_;
  long &linkmode = hkipc->linkmode_;
  LONG &lRealPlayHandle = hkipc->lRealPlayHandle;

  NET_DVR_PREVIEWINFO previewInfo = {0};
  previewInfo.hPlayWnd = NULL;
  previewInfo.lChannel = long(struDeviceInfo.byStartDChan) + channel;
  previewInfo.dwStreamType = streamtype;
  previewInfo.dwLinkMode = linkmode;

  NET_DVR_CLIENTINFO ClientInfo;
  ClientInfo.lChannel = long(struDeviceInfo.byStartDChan) + channel;
  ClientInfo.hPlayWnd = NULL;
  ClientInfo.lLinkMode = linkmode;
  if (streamtype != 0)
    ClientInfo.lLinkMode += 1 << 31;
  ClientInfo.sMultiCastIP = NULL;

  lRealPlayHandle =
      NET_DVR_RealPlay_V40(user_id_, &previewInfo, fRealDataCallBack, inParam);
  // lRealPlayHandle = NET_DVR_RealPlay_V30(user_id_, &ClientInfo,
  // fRealDataCallBack, NULL, TRUE);
  if (lRealPlayHandle < 0) {
    printf("NET_DVR_RealPlay_V30 failed! Error number: %d\n",
           NET_DVR_GetLastError());
    // return -1;
  } else {
    cout << "success！" << endl;
  }
  // return 0;
}

HKIPcamera::HKIPcamera() {
  hWnd = NULL;
  lRealPlayHandle = -1;
  nPort_ = -1;
  channel_ = 1;
  streamtype_ = 0;
  buffersize_ = 10;
  linkmode_ = 0;
}

HKIPcamera::~HKIPcamera() { release(); }

bool HKIPcamera::init(char *ip, char *usr, char *password, long port,
                      long channel, long streamtype, long link_mode,
                      int device_id, long buffer_size) {
  pthread_t hThread;
  // cout << "IP:" << ip << "    UserName:" << usr << "    PassWord:" <<
  // password << endl;
  NET_DVR_Init();
  NET_DVR_SetConnectTime(2000, 1);
  NET_DVR_SetReconnect(10000, true);
  bool login_success = OpenCamera(ip, port, usr, password);
  if (!login_success) {
    return login_success;
  }
  channel_ = channel;
  streamtype_ = streamtype;
  buffersize_ = buffer_size;
  linkmode_ = link_mode;
  device_id_ = device_id;
  pthread_mutex_init(&frame_list_mutex_, NULL);
  // hThread = ::CreateThread(NULL, 0, ReadCamera, NULL, 0, 0);
  ReadCamera(this);
  // pthread_create(&hThread, NULL, ReadCamera, this);
  return true;
}

Mat HKIPcamera::getframe() {
  Mat frame1;
  pthread_mutex_lock(&frame_list_mutex_);
  while (!frame_list_.size()) {
    pthread_mutex_unlock(&frame_list_mutex_);
    pthread_mutex_lock(&frame_list_mutex_);
  }
  list<Mat>::iterator it;
  it = frame_list_.end();
  it--;
  Mat dbgframe = (*(it));
  // (*frame_list_.begin()).copyTo(frame1);
  frame1 = dbgframe;
  frame_list_.pop_front();

  frame_list_.clear();
  pthread_mutex_unlock(&frame_list_mutex_);
  return (frame1);
}

void HKIPcamera::release() {
  // close(hThread);
  if (lRealPlayHandle != -1)
    NET_DVR_StopRealPlay(lRealPlayHandle);
  if (user_id_ > 0)
    NET_DVR_Logout(user_id_);
  NET_DVR_Cleanup();
}

void HKIPcamera::push_frame(const cv::Mat &frame) {
  frame_list_.push_back(frame);
}

long HKIPcamera::get_buffer_size() { return frame_list_.size(); }

bool HKIPcamera::is_buffer_full() { return get_buffer_size() >= buffersize_; }
