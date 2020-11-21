
#include "NvDecode.h"
//#include "NvHWEncoder.h"
#include <thread>
#include <iostream>
#include <cassert>

#include "helper_cuda_drvapi.h"

#define PAD_ALIGN(x,mask) ( (x + mask) & ~mask )

#define __cu(a) do { \
    CUresult  ret; \
    if ((ret = (a)) != CUDA_SUCCESS) { \
        fprintf(stderr, "%s has returned CUDA error %d\n", #a, ret); \
        return NV_ENC_ERR_GENERIC;\
    }} while(0)

NvDecode::NvDecode()
{
}


NvDecode::~NvDecode()
{
	quit();
	if(decodeThread) decodeThread->join();
	delete decodeThread;
	delete m_pFrameQueue;
	if (m_videoDecoder) cuvidDestroyDecoder(m_videoDecoder);
	if (m_videoParser)  cuvidDestroyVideoParser(m_videoParser);

   	if (ctxLock) checkCudaErrors(cuvidCtxLockDestroy(ctxLock));
	if (g_KernelSID) checkCudaErrors(cuStreamDestroy(g_KernelSID));
	av_dict_free(&pOptions);
	avcodec_close(pCodecCtx);
	avformat_close_input(&pFormatCtx);
	cuMemFree(g_pInteropFrame);
	cuMemFreeHost(rgbaBuf);

    
}

int NvDecode::start(const std::string rtspUrl)
{
	__cu(cuInit(0, __CUDA_API_VERSION, hHandleDriver));
	__cu(cuvidInit(0));

	__cu(cuDeviceGet(&device, 0)); //ʹ��0���Կ�
	__cu(cuCtxCreate(&cudaCtx, CU_CTX_SCHED_AUTO, device));
	__cu(cuvidCtxLockCreate(&ctxLock, cudaCtx));
	m_pFrameQueue = new CUVIDFrameQueue(ctxLock);

	CUresult oResult;

    // Initialize CUDA related Driver API
    // Determine if we are running on a 32-bit or 64-bit OS and choose the right PTX file
	try
	{
		if (sizeof(void *) == 4)
		{
			g_pCudaModule = new CUmoduleManager("NV12ToARGB_drvapi32.ptx", "hard_decode", 2, 2, 2);
		}
		else
		{
			g_pCudaModule = new CUmoduleManager("NV12ToARGB_drvapi64.ptx", "hard_decode", 2, 2, 2);
		}
	}
	catch (char const *p_file)
	{
		// If the CUmoduleManager constructor fails to load the PTX file, it will throw an exception
		printf("\n>> CUmoduleManager::Exception!  %s not found!\n", p_file);
		printf(">> Please rebuild NV12ToARGB_drvapi.cu or re-install this sample.\n");
		return -1;
	}
	g_pCudaModule->GetCudaFunction("NV12ToARGB_drvapi",   &g_kernelNV12toARGB);
	g_pCudaModule->GetCudaFunction("Passthru_drvapi",     &g_kernelPassThru);

    // Create a Stream ID for handling Readback
	checkCudaErrors(cuStreamCreate(&g_ReadbackSID, 0));
	checkCudaErrors(cuStreamCreate(&g_KernelSID,   0));
	printf(">> initCudaVideo()\n");
	printf("   CUDA Streams (%s) <g_ReadbackSID = %p>\n", ((g_ReadbackSID == 0) ? "Disabled" : "Enabled"), g_ReadbackSID);
	printf("   CUDA Streams (%s) <g_KernelSID   = %p>\n", ((g_KernelSID   == 0) ? "Disabled" : "Enabled"), g_KernelSID);

	av_register_all();
	avformat_network_init();
	pFormatCtx = avformat_alloc_context();

    	//设置rtsp以tcp方式传输
	av_dict_set(&pOptions, "rtsp_transport", "tcp", 0);
	av_dict_set(&pOptions, "stimeout", "2000000", 0); //设置超时断开连接时间，单位微秒

	if (avformat_open_input(&pFormatCtx, rtspUrl.c_str(), NULL, &pOptions) != 0) {
		printf("Couldn't open input stream.\n");
		return -1;
	}
	if (avformat_find_stream_info(pFormatCtx, NULL)<0) {
		printf("Couldn't find stream information.\n");
		return -1;
	}

	for (int i = 0; i < pFormatCtx->nb_streams; i++) {
		if (pFormatCtx->streams[i]->codec->codec_type == AVMEDIA_TYPE_VIDEO) {
			videoindex = i;
			break;
		}
	}

	if (videoindex == -1) {
		printf("Didn't find a video stream.\n");
		return -1;
	}

	pCodecCtx = pFormatCtx->streams[videoindex]->codec;

	AVCodec* pCodec = avcodec_find_decoder(pCodecCtx->codec_id);
	if (pCodec == NULL) {
		printf("Codec not found.\n");
		return -1;
	}

	//Output Info-----------------------------
	printf("--------------- File Information ----------------\n");
	av_dump_format(pFormatCtx, 0, rtspUrl.c_str(), 0);
	printf("-------------------------------------------------\n");

	CUVIDEOFORMAT oFormat;
	memset(&oFormat, 0, sizeof(CUVIDEOFORMAT));

	switch (pCodecCtx->codec_id) {
	case AV_CODEC_ID_H263:
		oFormat.codec = cudaVideoCodec_MPEG4;
		break;

	case AV_CODEC_ID_H264:
		oFormat.codec = cudaVideoCodec_H264;
		break;

	case AV_CODEC_ID_HEVC:
		oFormat.codec = cudaVideoCodec_HEVC;
		break;

	case AV_CODEC_ID_MJPEG:
		oFormat.codec = cudaVideoCodec_JPEG;
		break;

	case AV_CODEC_ID_MPEG1VIDEO:
		oFormat.codec = cudaVideoCodec_MPEG1;
		break;

	case AV_CODEC_ID_MPEG2VIDEO:
		oFormat.codec = cudaVideoCodec_MPEG2;
		break;

	case AV_CODEC_ID_MPEG4:
		oFormat.codec = cudaVideoCodec_MPEG4;
		break;

	case AV_CODEC_ID_VC1:
		oFormat.codec = cudaVideoCodec_VC1;
		break;
	default:
		break;
	}

	//����ط���FFmoeg��cuvid�Ķ�Ӧ��ϵ���Ǻ�ȷ������������������ƺ����
	switch (pCodecCtx->sw_pix_fmt)
	{
	case AV_PIX_FMT_YUV420P:
		oFormat.chroma_format = cudaVideoChromaFormat_420;
		break;
	case AV_PIX_FMT_YUV422P:
		oFormat.chroma_format = cudaVideoChromaFormat_422;
		break;
	case AV_PIX_FMT_YUV444P:
		oFormat.chroma_format = cudaVideoChromaFormat_444;
		break;
	default:
		oFormat.chroma_format = cudaVideoChromaFormat_420;
		break;
	}

	//���˺þã��������ҵ���FFmpeg�б�ʶ����ʽ��֡��ʽ�ı�ʶλ
	//����ʽ�Ǹ���ɨ��ģ���Ҫ��ȥ���д���
	switch (pCodecCtx->field_order)
	{
	case AV_FIELD_PROGRESSIVE:
	case AV_FIELD_UNKNOWN:
		oFormat.progressive_sequence = true;
		break;
	default:
		oFormat.progressive_sequence = false;
		break;
	}

	pCodecCtx->thread_safe_callbacks = 1;

	oFormat.coded_width = pCodecCtx->coded_width;
	oFormat.coded_height = pCodecCtx->coded_height;

	oFormat.display_area.right = pCodecCtx->width;
	oFormat.display_area.left = 0;
	oFormat.display_area.bottom = pCodecCtx->height;
	oFormat.display_area.top = 0;

	h264bsfc = nullptr;
	if (pCodecCtx->codec_id == AV_CODEC_ID_H264 || pCodecCtx->codec_id == AV_CODEC_ID_HEVC) {
		if (pCodecCtx->codec_id == AV_CODEC_ID_H264)
			h264bsfc = av_bitstream_filter_init("h264_mp4toannexb");
		else
			h264bsfc = av_bitstream_filter_init("hevc_mp4toannexb");
	}


	//cuvidCreateVideoSource : 该接口的作用是创建source，主要参数是设置视频文件路径和回调函数。
	//source会去解析指定视频文件，并通过回调函数实现对视频数据的自定义处理。
	//源码中在视频数据回调函数中，调用了cuvidParseVideoData，即向parser中传递数据;
	// 这里使用ffmpeg解析,未使用；


	//init video decoder
	CUVIDDECODECREATEINFO oVideoDecodeCreateInfo;
	memset(&oVideoDecodeCreateInfo, 0, sizeof(CUVIDDECODECREATEINFO));
	oVideoDecodeCreateInfo.CodecType = oFormat.codec;
	oVideoDecodeCreateInfo.ulWidth = oFormat.coded_width;
	oVideoDecodeCreateInfo.ulHeight = oFormat.coded_height;
	oVideoDecodeCreateInfo.ulNumDecodeSurfaces = FrameQueue::cnMaximumSize;

	// Limit decode memory to 24MB (16M pixels at 4:2:0 = 24M bytes)
	// Keep atleast 6 DecodeSurfaces
	while (oVideoDecodeCreateInfo.ulNumDecodeSurfaces > 6 &&
		oVideoDecodeCreateInfo.ulNumDecodeSurfaces * oFormat.coded_width * oFormat.coded_height > 16 * 1024 * 1024)
	{
		oVideoDecodeCreateInfo.ulNumDecodeSurfaces--;
	}

	oVideoDecodeCreateInfo.ChromaFormat = oFormat.chroma_format;
	oVideoDecodeCreateInfo.OutputFormat = cudaVideoSurfaceFormat_NV12;
	oVideoDecodeCreateInfo.DeinterlaceMode = cudaVideoDeinterlaceMode_Weave;

	targetWidth = PAD_ALIGN(pCodecCtx->width, 0x3f);
	targetHeight = PAD_ALIGN(pCodecCtx->height, 0x0f);
	if (targetWidth <= 0 || targetHeight <= 0) {
		oVideoDecodeCreateInfo.ulTargetWidth = oVideoDecodeCreateInfo.ulWidth;
		oVideoDecodeCreateInfo.ulTargetHeight = oVideoDecodeCreateInfo.ulHeight;
	}
	else {
		oVideoDecodeCreateInfo.ulTargetWidth = targetWidth;
		oVideoDecodeCreateInfo.ulTargetHeight = targetHeight;
	}

	oVideoDecodeCreateInfo.ulNumOutputSurfaces = 2;
	oVideoDecodeCreateInfo.ulCreationFlags = cudaVideoCreate_PreferCUVID;
	oVideoDecodeCreateInfo.vidLock = ctxLock;

	//cuvidCreateDecoder : 该接口是用来创建decoder，通过设置一些解码参数，会返回一个decoder的句柄。
	//这个句柄会在之后的解码接口中被使用。
	oResult = cuvidCreateDecoder(&m_videoDecoder, &oVideoDecodeCreateInfo);
	if (oResult != CUDA_SUCCESS) {
		fprintf(stderr, "cuvidCreateDecoder() failed, error code: %d\n", oResult);
		exit(-1);
	}

	m_oVideoDecodeCreateInfo = oVideoDecodeCreateInfo;

	//init video parser
	CUVIDPARSERPARAMS oVideoParserParameters;
	CUVIDEOFORMATEX pFormat;
	memset(&pFormat, 0, sizeof (CUVIDEOFORMATEX));
	pFormat.format = oFormat;
	memset(&oVideoParserParameters, 0, sizeof(CUVIDPARSERPARAMS));
	oVideoParserParameters.CodecType = oVideoDecodeCreateInfo.CodecType;
	oVideoParserParameters.ulMaxNumDecodeSurfaces = oVideoDecodeCreateInfo.ulNumDecodeSurfaces;
	oVideoParserParameters.ulMaxDisplayDelay = 1;
	oVideoParserParameters.pUserData = this;
	oVideoParserParameters.pExtVideoInfo = &pFormat;
	oVideoParserParameters.pfnSequenceCallback = NvDecode::HandleVideoSequence;
	oVideoParserParameters.pfnDecodePicture = NvDecode::HandlePictureDecode;
	oVideoParserParameters.pfnDisplayPicture = NvDecode::HandlePictureDisplay;

	//cuvidCreateVideoParser : 该接口是用来创建video parser，主要参数是设置三个回调函数，实现对解析出来的数据的处理。
	oResult = cuvidCreateVideoParser(&m_videoParser, &oVideoParserParameters);
	if (oResult != CUDA_SUCCESS) {
		fprintf(stderr, "cuvidCreateVideoParser failed, error code: %d\n", oResult);
		exit(-1);
	}

	if (oResult == CUDA_SUCCESS) {
		checkCudaErrors(cuMemAlloc(&g_pInteropFrame, targetWidth * targetHeight * 4));
		//cuMemAllocHost用来cpu及显卡都可访问的系统内存。
		checkCudaErrors(cuMemAllocHost((void**)&rgbaBuf, targetWidth * targetHeight *4));
		decodeThread = new std::thread(std::bind(&NvDecode::startDecode, this));
	}

	return CUDA_SUCCESS;
}


bool NvDecode::deQueueFrame(unsigned char ** ptr, int *width, int *height, unsigned long long *timestamp)
{
	CUVIDPARSERDISPINFO pInfo;
	if (!(m_pFrameQueue->isEndOfDecode() && m_pFrameQueue->isEmpty())) {
		if (m_pFrameQueue->dequeue(&pInfo)) {
			CCtxAutoLock lck(ctxLock);
			checkCudaErrors(cuCtxPushCurrent(cudaCtx));
			CUdeviceptr pDecodedFrame[2] = { 0,0 };
			CUdeviceptr pInteropFrame[2] = { 0,0 };
			int distinct_fields = 1;
			if (!pInfo.progressive_frame && pInfo.repeat_first_field <= 1) {
				distinct_fields = 2;
			}

			for (int active_field = 0; active_field < distinct_fields; active_field++) {
				CUVIDPROCPARAMS oVPP = { 0 };
				oVPP.progressive_frame = pInfo.progressive_frame;
				oVPP.top_field_first = pInfo.top_field_first;
				oVPP.unpaired_field = (distinct_fields == 1);
				oVPP.second_field = active_field;
				unsigned int nDecodedPitch = 0;
				//cuvidMapVideoFrame可以获取到设备内存中指定的YUV数据地址。
				if (cuvidMapVideoFrame(m_videoDecoder, pInfo.picture_index, &pDecodedFrame[active_field], &nDecodedPitch, &oVPP) != CUDA_SUCCESS) {
					m_pFrameQueue->releaseFrame(&pInfo);
					checkCudaErrors(cuCtxPopCurrent(NULL));
					return false;
				}

				if (isFirstFrame) {
					*ptr = rgbaBuf;
					*width = targetWidth;
					*height = targetHeight;
					float hueColorSpaceMat[9];
					setColorSpaceMatrix(ITU601, hueColorSpaceMat, 0.0f);
					updateConstantMemory_drvapi(g_pCudaModule->getModule(), hueColorSpaceMat);
					isFirstFrame = false;
				}


				pInteropFrame[active_field] = g_pInteropFrame;
				size_t nTexturePitch = 0;
                		nTexturePitch = targetWidth*4;


				// Final Stage: NV12toARGB color space conversion
				CUresult  oRes = cudaLaunchNV12toARGBDrv(pDecodedFrame[active_field], nDecodedPitch,
										pInteropFrame[active_field], nTexturePitch,
										targetWidth, targetHeight, g_kernelNV12toARGB, g_KernelSID);
				if (oRes != CUDA_SUCCESS) {
					std::cout << "launch Kernel failed,status" << oRes << std::endl;
					return false;
				}

				int dstPictch = targetWidth * 4;
				//最后通过cuMemcpyDtoHAsync将设备内存中指定的数据copy到系统内存中。
				checkCudaErrors(cuMemcpyDtoHAsync(rgbaBuf, pInteropFrame[active_field], dstPictch * targetHeight , g_ReadbackSID));
				cuvidUnmapVideoFrame(m_videoDecoder, pDecodedFrame[active_field]);

			}
			checkCudaErrors(cuCtxPopCurrent(NULL));
		
			*timestamp = pInfo.timestamp;
			m_pFrameQueue->releaseFrame(&pInfo);
			return true;
		}
	}
	return false;
}

void NvDecode::quit()
{
	std::lock_guard<std::mutex> lck(mtx);
	_quitFlag = true;
}

void NvDecode::startDecode()
{
	AVPacket *avpkt;
	avpkt = (AVPacket *)av_malloc(sizeof(AVPacket));
	CUVIDSOURCEDATAPACKET cupkt;
	int iPkt = 0;
	CUresult oResult;
	while (int errorCode = av_read_frame(pFormatCtx, avpkt) >= 0)
	 {
	LOOP0:
		{
			std::lock_guard<std::mutex> lck(mtx);
			if (_quitFlag) {
				_quitFlag = false;
				break;
			}
		}

		if (avpkt->stream_index == videoindex)
		{
#if 1
			AVPacket new_pkt = *avpkt;

			if (avpkt && avpkt->size)
			{
				if (h264bsfc) {

					int a = av_bitstream_filter_filter(h264bsfc, pFormatCtx->streams[videoindex]->codec, NULL,
						&new_pkt.data, &new_pkt.size,
						avpkt->data, avpkt->size,
						avpkt->flags & AV_PKT_FLAG_KEY);

					if (a>0) {
						if (new_pkt.data != avpkt->data)//-added this
						{
							av_free_packet(avpkt);

							avpkt->data = new_pkt.data;
							avpkt->size = new_pkt.size;
						}
					}
					else if (a<0) {
						goto LOOP0;
					}

					*avpkt = new_pkt;
				}

				cupkt.payload_size = (unsigned long)avpkt->size;
				cupkt.payload = (const unsigned char*)avpkt->data;

				if (avpkt->pts != AV_NOPTS_VALUE)
				{
					cupkt.flags = CUVID_PKT_TIMESTAMP;
					if (pCodecCtx->pkt_timebase.num && pCodecCtx->pkt_timebase.den)
					{
						AVRational tb;
						tb.num = 1;
						tb.den = AV_TIME_BASE;
						cupkt.timestamp = av_rescale_q(avpkt->pts, pCodecCtx->pkt_timebase, tb);
					}
					else
						cupkt.timestamp = avpkt->pts;
				}
			}
			else
			{
				cupkt.flags = CUVID_PKT_ENDOFSTREAM;
			}
			//cuvidParseVideoData : 该接口是用来向parser塞数据，通过不断地塞h.264数据，parser会通过回调接口对解析出来的数据进行处理。
			oResult = cuvidParseVideoData(m_videoParser, &cupkt);  //这里调用HandlePictureDecode等回调函数。
			if ((cupkt.flags & CUVID_PKT_ENDOFSTREAM) || (oResult != CUDA_SUCCESS))
			{
				break;
			}

			av_free(new_pkt.data);
#else
			cupkt.flags = 0;
			cupkt.payload_size = (unsigned long)avpkt->size;
			cupkt.payload = avpkt->data;
			cupkt.timestamp = 0;
			cuvidParseVideoData(m_videoParser, &cupkt);
#endif
		}
		else
			av_free_packet(avpkt);

	}
	sleep(1);
	if (pCodecCtx->codec_id == AV_CODEC_ID_H264 || pCodecCtx->codec_id == AV_CODEC_ID_HEVC) {
		av_bitstream_filter_close(h264bsfc);
	}
	m_pFrameQueue->endDecode();
	cuMemFree(g_pInteropFrame);
}

void NvDecode::setColorSpaceMatrix(eColorSpace CSC, float * hueCSC, float hue)
{
	float hueSin = sin(hue);
	float hueCos = cos(hue);

	if (CSC == ITU601)
	{
		//CCIR 601
		hueCSC[0] = 1.1644f;
		hueCSC[1] = hueSin * 1.5960f;
		hueCSC[2] = hueCos * 1.5960f;
		hueCSC[3] = 1.1644f;
		hueCSC[4] = (hueCos * -0.3918f) - (hueSin * 0.8130f);
		hueCSC[5] = (hueSin *  0.3918f) - (hueCos * 0.8130f);
		hueCSC[6] = 1.1644f;
		hueCSC[7] = hueCos *  2.0172f;
		hueCSC[8] = hueSin * -2.0172f;
	}
	else if (CSC == ITU709)
	{
		//CCIR 709
		hueCSC[0] = 1.0f;
		hueCSC[1] = hueSin * 1.57480f;
		hueCSC[2] = hueCos * 1.57480f;
		hueCSC[3] = 1.0;
		hueCSC[4] = (hueCos * -0.18732f) - (hueSin * 0.46812f);
		hueCSC[5] = (hueSin *  0.18732f) - (hueCos * 0.46812f);
		hueCSC[6] = 1.0f;
		hueCSC[7] = hueCos *  1.85560f;
		hueCSC[8] = hueSin * -1.85560f;
	}
}

CUresult NvDecode::updateConstantMemory_drvapi(CUmodule module, float * hueCSC)
{
	CUdeviceptr  d_constHueCSC, d_constAlpha;
	size_t       d_cscBytes, d_alphaBytes;

	// First grab the global device pointers from the CUBIN
	cuModuleGetGlobal(&d_constHueCSC, &d_cscBytes, module, "constHueColorSpaceMat");
	cuModuleGetGlobal(&d_constAlpha, &d_alphaBytes, module, "constAlpha");

	CUresult error = CUDA_SUCCESS;

	// Copy the constants to video memory
	cuMemcpyHtoD(d_constHueCSC,
		reinterpret_cast<const void *>(hueCSC),
		d_cscBytes);
	getLastCudaDrvErrorMsg("cuMemcpyHtoD (d_constHueCSC) copy to Constant Memory failed");


 	unsigned int cudaAlpha = ((unsigned int)0xff << 24);

	cuMemcpyHtoD(d_constAlpha,
		reinterpret_cast<const void *>(&cudaAlpha),
		d_alphaBytes);
	getLastCudaDrvErrorMsg("cuMemcpyHtoD (constAlpha) copy to Constant Memory failed");

	return error;
}

int CUDAAPI NvDecode::HandleVideoSequence(void * pUserData, CUVIDEOFORMAT * pFormat)
{
	assert(pUserData);
	NvDecode* pDecoder = (NvDecode*)pUserData;

	if ((pFormat->codec != pDecoder->m_oVideoDecodeCreateInfo.CodecType) ||         // codec-type
		(pFormat->coded_width != pDecoder->m_oVideoDecodeCreateInfo.ulWidth) ||
		(pFormat->coded_height != pDecoder->m_oVideoDecodeCreateInfo.ulHeight) ||
		(pFormat->chroma_format != pDecoder->m_oVideoDecodeCreateInfo.ChromaFormat))
	{
		fprintf(stderr, "NvTranscoder doesn't deal with dynamic video format changing\n");
		return 0;
	}

	return 1;
}

int CUDAAPI NvDecode::HandlePictureDecode(void * pUserData, CUVIDPICPARAMS * pPicParams)
{
	assert(pUserData);
	NvDecode* pDecoder = (NvDecode*)pUserData;
	pDecoder->m_pFrameQueue->waitUntilFrameAvailable(pPicParams->CurrPicIdx);
	
	//cuvidDecodePicture : 该接口就是向解码器传递待解码的数据。
	//需要说明一下，该接口是异步解码，不能通过该接口得到解码后的视频数据，它只是向解码器传数据而已。解码后的数据，是通过parser的HandlePictureDisplay回调得到。
	assert(CUDA_SUCCESS == cuvidDecodePicture(pDecoder->m_videoDecoder, pPicParams));
	return 1;
}

int CUDAAPI NvDecode::HandlePictureDisplay(void * pUserData, CUVIDPARSERDISPINFO * pPicParams)
{
	assert(pUserData);
	NvDecode* pDecoder = (NvDecode*)pUserData;
	pDecoder->m_pFrameQueue->enqueue(pPicParams);

	return 1;
}
