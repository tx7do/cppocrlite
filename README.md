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
# generate make files
cmake .

# build Release version
cmake --build . --config Release --target cppocrlite -- -j 12

# build Debug version
cmake --build . --config Debug --target cppocrlite -- -j 12
```
