# cppocrlite

[chineseocr_lite](https://github.com/DayBreak-u/chineseocr_lite) cpp library

## Install

### Install OpenCV

```shell
sudo apt install libopencv-dev
```

### Install OpenMP

```shell
sudo apt-get install libomp-dev
```
### Install OnnxRuntime

Download Page: <https://github.com/microsoft/onnxruntime/releases>

```shell
wget https://github.com/microsoft/onnxruntime/releases/download/v1.14.0/onnxruntime-linux-x64-1.14.0.tgz

tar -zxvf onnxruntime-linux-x64-1.14.0.tgz

sudo mv onnxruntime-linux-x64-1.14.0 /opt/onnxruntime

sudo chown -R `whoami` /opt/onnxruntime
```

## Build

```shell
# build Release version
mkdir cmake-build-release
cd cmake-build-release && cmake .. -DCMAKE_BUILD_TYPE=Release
cmake --build . --config Release --target cppocrlite -- -j 12

# build Debug version
mkdir cmake-build-debug
cd cmake-build-debug && cmake .. -DCMAKE_BUILD_TYPE=Debug
cmake --build . --config Debug --target cppocrlite -- -j 12

# install
sudo chmod 757 /usr/local
sudo chmod 757 /usr/local/lib
sudo chmod 757 /usr/local/lib/pkgconfig
cmake --install .
```

or execute shell script:

```shell
cd script && ./build.sh
```

check installation state:

```shell
pkg-config --cflags --libs onnxruntime
pkg-config --cflags --libs opencv4
pkg-config --cflags --libs cppocrlite
```
