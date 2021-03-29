/*
 * @Author: your name
 * @Date: 2020-11-09 14:24:31
 * @LastEditTime: 2020-11-27 10:23:52
 * @LastEditors: Please set LastEditors
 * @Description: In User Settings Edit
 * @FilePath: /cudaDecodeGL(另一个复件)/NvDecode.h
 */
#pragma once

#include <string>
#include <mutex>
#include "nvEncodeAPI.h"
#include "dynlink_nvcuvid.h" // <nvcuvid.h>
#include "dynlink_cuda.h"    // <cuda.h>
#include "FrameQueue.h"
#include "dynlink_builtin_types.h"
#include "cudaProcessFrame.h"
#include "cudaModuleMgr.h"
#include <thread>
extern "C"
{/*
#include <libavcodec/avcodec.h>
#include <libavformat/avformat.h>
#include <libavfilter/avfilter.h>
#include <libavutil/frame.h>
#include <libswscale/swscale.h>
#include <libavutil/pixfmt.h>
#include <libavutil/imgutils.h>
*/


#include  <libavutil/avstring.h>
#include  <libavutil/mathematics.h>
#include "libavutil/pixdesc.h"
#include "libavutil/imgutils.h"
#include "libavutil/dict.h"
#include "libavutil/parseutils.h"
#include "libavutil/samplefmt.h"
#include "libavutil/avassert.h"
#include "libavutil/time.h"
#include "libavformat/avformat.h"
#include "libavdevice/avdevice.h"
#include "libswscale/swscale.h"
#include "libavutil/opt.h"
#include "libavcodec/avfft.h"
#include "libswresample/swresample.h"

#include "libavfilter/buffersink.h"
#include "libavfilter/buffersrc.h"
#include "libavutil/avutil.h"
}
#if defined(WIN32) || defined(_WIN32) || defined(WIN64) || defined(_WIN64)
typedef HMODULE CUDADRIVER;
#else
typedef void *CUDADRIVER;
#endif


class NvDecode
{
public:
	NvDecode();
	~NvDecode();

	CUvideodecoder m_videoDecoder = NULL;
	FrameQueue*    m_pFrameQueue;
	CUVIDDECODECREATEINFO m_oVideoDecodeCreateInfo;
	virtual int start(const std::string  rtspUrl);
	virtual void end();
	virtual bool deQueueFrame(unsigned char **rgbaP, int *width, int *height, unsigned long long *timestamp);
	virtual void quit();

private:
	typedef enum
	{
		ITU601 = 1,
		ITU709 = 2
	} eColorSpace;
	std::string _rtspUrl;
	bool _quitFlag = false;
	std::mutex mtx;
	CUDADRIVER hHandleDriver = 0;
	CUcontext g_oContext;
	CUdevice device;

	CUvideoctxlock ctxLock;
	CUvideoparser  m_videoParser = NULL;
	CUdeviceptr g_pInteropFrame[2]={0,0};
	unsigned char *rgbaBuf = NULL;
	std::thread *decodeThread = NULL;

	CUfunction g_kernelNV12toARGB = NULL;
	bool isFirstFrame = true;
	CUstream         g_KernelSID = 0;
	CUstream           g_ReadbackSID = 0;
      CUfunction         g_kernelPassThru      = 0;
	
	CUmoduleManager   *g_pCudaModule;

	int videoindex = -1;
	int targetWidth, targetHeight;
	AVBitStreamFilterContext *h264bsfc = NULL;
	AVFormatContext* pFormatCtx = NULL;
	AVCodecContext* pCodecCtx = NULL;
    	AVDictionary *pOptions = NULL;

	void startDecode();
	void setColorSpaceMatrix(eColorSpace CSC, float *hueCSC, float hue);
	CUresult updateConstantMemory_drvapi(CUmodule module, float *hueCSC);

	static int CUDAAPI HandleVideoSequence(void* pUserData, CUVIDEOFORMAT* pFormat);
	static int CUDAAPI HandlePictureDecode(void* pUserData, CUVIDPICPARAMS* pPicParams);
	static int CUDAAPI HandlePictureDisplay(void* pUserData, CUVIDPARSERDISPINFO* pPicParams);
};

