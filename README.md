# GPU_Hard_Decode_RTSP
为解决多路监控视频流cpu高占用率并且cuvid不支持直接解码rtsp流的问题，利用ffmpeg解析rtsp流，在gpu端利用cuda 的视频解码器cuvid来解码，并将NV12转化为RGBA格式;相比opencv软解码，cpu占用率大幅降低。
# 参考
1.NVIDIA_CUDA-9.0_Samples/3_Imaging/cudaDecodeGL
2.https://blog.csdn.net/wanghualin033/article/details/79829069?utm_medium=distribute.pc_relevant.none-task-blog-title-3&spm=1001.2101.3001.4242
3.https://www.jianshu.com/p/9d880ed5bdc4
