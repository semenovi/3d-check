cmake_minimum_required(VERSION 3.8)

# Project name
project(gpu_sort LANGUAGES CSharp)

add_subdirectory(OpenCL)
add_subdirectory(OpenCLlib)
set_property(DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR} PROPERTY VS_STARTUP_PROJECT OpenCL)