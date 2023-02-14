#!/usr/bin/env bash

rm -fr cmake-build-release
mkdir cmake-build-release

cd cmake-build-release

cmake .. -DCMAKE_BUILD_TYPE=Release

cmake --build . --config Release --target cppocrlite -- -j 12

sudo cmake --install .
