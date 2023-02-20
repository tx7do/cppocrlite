#!/usr/bin/env bash

yum update && yum upgrade

# Install Building Tool
yum groupinstall "Development Tools" -y
yum install cmake3 gcc gtk2-devel numpy pkconfig -y

# install image lib
yum install \
          libjpeg-devel libtiff-devel libpng-devel\
          libjpeg libtiff libpng

# install OpenCV
wget https://github.com/opencv/opencv/archive/4.2.0.zip
unzip 4.2.0.zip
cd opencv-4.2.0
mkdir build
cd build
cmake3 -D CMAKE_BUILD_TYPE=DEBUG -D CMAKE_INSTALL_PREFIX=/usr/local ..
make
make install
export PKG_CONFIG_PATH=$PKG_CONFIG_PATH:/usr/local/lib/pkgconfig/
echo '/usr/local/lib/' >> /etc/ld.so.conf.d/opencv.conf
ldconfig

# install OnnxRuntime
REPO="microsoft/onnxruntime"
latest_tag=$(curl -s https://api.github.com/repos/$REPO/releases/latest | grep '"tag_name":' | sed -E 's/.*"([^"]+)".*/\1/')
echo "Using OnnxRuntime version $latest_tag"
wget https://github.com/$REPO/releases/download/$latest_tag/onnxruntime-linux-x64-${latest_tag:1}.tgz
tar -zxvf onnxruntime-linux-x64-${latest_tag:1}.tgz
sudo mv onnxruntime-linux-x64-${latest_tag:1} /usr/local/onnxruntime
rm onnxruntime-linux-x64-${latest_tag:1}.tgz
