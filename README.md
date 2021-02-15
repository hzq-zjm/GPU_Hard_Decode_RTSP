# GPU_Hard_Decode_RTSP
多路监控视频流cpu占用率高，并且cuvid不支持直接解码rtsp流。利用ffmpeg解析rtsp流，在gpu端利用cuda 的视频解码器cuvid来解码，并将NV12转化为RGBA格式。
# Ref
  1.NVIDIA_CUDA-9.0_Samples/3_Imaging/cudaDecodeGL  
  2.https://blog.csdn.net/wanghualin033/article/details/79829069?utm_medium=distribute.pc_relevant.none-task-blog-title-3&spm=1001.2101.3001.4242  
  3.https://www.jianshu.com/p/9d880ed5bdc4
# 2021/2/7
  今天发现opencv4.0+已经支持了，无语！
