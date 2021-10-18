cmake_minimum_required(VERSION 3.20)
project(LinearPartition)

set(CMAKE_CXX_STANDARD 14)

include_directories(src)
include_directories(src/Utils)

add_executable(LinearPartition
        src/Utils/energy_parameter.h
        src/Utils/feature_weight.h
        src/Utils/intl11.h
        src/Utils/intl21.h
        src/Utils/intl22.h
        src/Utils/utility.h
        src/Utils/utility_v.h
        src/bpp.cpp
        src/LinearPartition.cpp
        src/LinearPartition.h)
