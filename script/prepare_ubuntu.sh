#!/usr/bin/env bash

sudo apt update && sudo apt upgrade

# install CMake
sudo apt install cmake

# install gcc
sudo apt install build-essential

# install image lib
sudo apt install libjpeg-dev libtiff-dev libpng-dev

# install OpenCV
sudo apt install libopencv-dev

# install OpenMP
sudo apt-get install libomp-dev

# install OnnxRuntime
REPO="microsoft/onnxruntime"
latest_tag=$(curl -s https://api.github.com/repos/$REPO/releases/latest | grep '"tag_name":' | sed -E 's/.*"([^"]+)".*/\1/')
echo "Using OnnxRuntime version $latest_tag"
wget https://github.com/$REPO/releases/download/$latest_tag/onnxruntime-linux-x64-${latest_tag:1}.tgz
tar -zxvf onnxruntime-linux-x64-${latest_tag:1}.tgz
sudo mv onnxruntime-linux-x64-${latest_tag:1} /usr/local/onnxruntime
rm onnxruntime-linux-x64-${latest_tag:1}.tgz
