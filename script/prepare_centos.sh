#!/usr/bin/env bash

yum update && yum upgrade

# install CMake
yum install cmake

# install gcc
yum install \
            autoconf automake binutils \
            bison flex gcc gcc-c++ gettext \
            libtool make patch pkgconfig \
            redhat-rpm-config rpm-build rpm-sign \
            ctags elfutils indent patchutils

# install image lib
yum install \
          libjpeg-devel libtiff-devel libpng-devel\
          libjpeg libtiff libpng

# install OpenCV
yum install opencv opencv-devel

# install OnnxRuntime
REPO="microsoft/onnxruntime"
latest_tag=$(curl -s https://api.github.com/repos/$REPO/releases/latest | grep '"tag_name":' | sed -E 's/.*"([^"]+)".*/\1/')
echo "Using OnnxRuntime version $latest_tag"
wget https://github.com/$REPO/releases/download/$latest_tag/onnxruntime-linux-x64-${latest_tag:1}.tgz
tar -zxvf onnxruntime-linux-x64-${latest_tag:1}.tgz
sudo mv onnxruntime-linux-x64-${latest_tag:1} /usr/local/onnxruntime
rm onnxruntime-linux-x64-${latest_tag:1}.tgz
