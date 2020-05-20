cmake_minimum_required(VERSION 2.8)
project(zernikec2)

add_definitions("-std=c++11 -O3 -ffast-math")

# set numpy and python directories
set(numpydir "/home/hi76tor/python_compiled/lib/python3.6/site-packages/numpy/core/include")
set(PYTHON_LIBRARIES "/home/hi76tor/python_compiled_2/lib/libpython3.6m.so")
set(PYTHON_INCLUDE_DIRS "/home/hi76tor/python_compiled_2/include/python3.6m")
find_package( GSL REQUIRED )
find_package(MPI REQUIRED)

include_directories(${PYTHON_INCLUDE_DIRS} ${numpydir})

# set(CMAKE_BUILD_TYPE Debug)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

add_executable(
	a.out
	src/main.cpp
	)
add_executable(
	b.out
	src/testprop.cpp
	)
target_link_libraries(a.out fftw3 ${PYTHON_LIBRARIES} GSL::gsl ${MPI_LIBRARIES} MPI::MPI_CXX)
target_link_libraries(b.out fftw3 ${PYTHON_LIBRARIES} GSL::gsl ${MPI_LIBRARIES} MPI::MPI_CXX)


add_library(MeasuredImageFormatter SHARED src/MeasuredImageFormatter.cpp)
add_library(MeasuredImageFormatterWrapper SHARED src/MeasuredImageFormatterWrapper.cpp)
target_link_libraries(MeasuredImageFormatterWrapper MeasuredImageFormatter GSL::gsl)
# PYTHON_CONFIGURE_OPTS="--enable-shared"
