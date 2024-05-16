cmake_minimum_required(VERSION 3.27)
project(PA_Pycha)

set(CMAKE_CXX_STANDARD 17)
set(RUNTIME_OUTPUT_DIRECTORY ./)
add_library(MyMLP_DLL_TEST SHARED library.cpp)

