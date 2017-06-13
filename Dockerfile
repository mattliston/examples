# This is a comment
FROM nvidia/cuda:8.0-cudnn5-devel
ENV DEBIAN_FRONTEND noninteractive
RUN apt-get update
RUN apt-get install -y build-essential cmake git yasm pkg-config vim wget python-pip python-dev
# NUMPY
RUN pip install numpy
RUN pip install scipy
RUN apt-get update
RUN apt-get install -y python-matplotlib
# TENSORFLOW
RUN pip install --upgrade https://storage.googleapis.com/tensorflow/linux/gpu/tensorflow_gpu-1.2.0rc2-cp27-none-linux_x86_64.whl
# FFMPEG
RUN git clone -b n3.1.2 https://github.com/ffmpeg/ffmpeg /tmp/ffmpeg
RUN apt-get update
RUN apt-get install -y libvorbisenc2 libsdl1.2-dev zlib1g-dev libfaad-dev libgsm1-dev libtheora-dev libvorbis-dev libspeex-dev libopencore-amrwb-dev libopencore-amrnb-dev libxvidcore-dev libxvidcore4 libmp3lame-dev libjpeg62 libjpeg62-dev 
RUN git clone http://git.videolan.org/git/x264.git /tmp/x264
RUN cd /tmp/x264/ && ./configure --disable-asm --enable-shared --enable-pic && make && make install
RUN git clone https://chromium.googlesource.com/webm/libvpx /tmp/libvpx
RUN cd /tmp/libvpx && ./configure --enable-shared --enable-pic && make && make install
RUN cd /tmp/ffmpeg && ./configure --enable-gpl --enable-libvorbis --enable-libvpx --enable-libx264 --enable-nonfree --enable-shared && make install -j && make clean
# OPENCV
RUN apt-get update
RUN apt-get install -y --fix-missing libgtk2.0-dev pkg-config
RUN git clone -b 3.1.0 https://github.com/Itseez/opencv /tmp/opencv
RUN git clone -b 3.1.0 https://github.com/Itseez/opencv_contrib /tmp/opencv/opencv_contrib
RUN cd /tmp/opencv && mkdir build && cd build && cmake -DWITH_CUDA=OFF -DWITH_IPP=ON -DOPENCV_EXTRA_MODULES_PATH=../opencv_contrib/modules .. && make install -j && make clean
# IGRAPH
RUN apt-get update
RUN apt-get install -y libxml2 libxml2-dev
RUN pip install python-igraph
# MISC
RUN apt-get update
RUN apt-get install -y eog bc mplayer mplayer2
RUN apt-get install -y apt-utils sudo
RUN apt-get install -y libcanberra-gtk-module

RUN ldconfig
