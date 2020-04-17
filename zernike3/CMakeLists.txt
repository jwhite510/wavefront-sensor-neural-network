cmake_minimum_required(VERSION 2.8)
project(zernikec2)

add_definitions("-std=c++11 -O3 -ffast-math")

# set(numpydir "/home/zom/Projects/python_cpp/venv/lib/python3.6/site-packages/numpy/core/include/")
set(numpydir "/home/hi76tor/python_compiled/lib/python3.6/site-packages/numpy/core/include")
find_package( PythonLibs REQUIRED )
# find_package( GSL REQUIRED )

# message(${PYTHON_INCLUDE_DIRS})
# message(${PYTHON_LIBRARIES})
set(PYTHON_LIBRARIES "/home/hi76tor/python_compiled_2/lib/libpython3.6m.so")
set(PYTHON_INCLUDE_DIRS "/home/hi76tor/python_compiled_2/include/python3.6m")
message(${PYTHON_INCLUDE_DIRS})
message(${PYTHON_LIBRARIES})

set(gsl_lib "/cluster/spack/opt/spack/linux-centos7-x86_64/intel-18.0.1/gsl-2.4-bp5w6qe4dksrqnubftxxjcyti7jurqkm/lib/libgsl.so")
set(gsl_lib2 "/cluster/spack/opt/spack/linux-centos7-x86_64/intel-18.0.1/gsl-2.4-bp5w6qe4dksrqnubftxxjcyti7jurqkm/lib/libgslcblas.so")
set(gsl_include "/cluster/spack/opt/spack/linux-centos7-x86_64/intel-18.0.1/gsl-2.4-bp5w6qe4dksrqnubftxxjcyti7jurqkm/include")

include_directories(${PYTHON_INCLUDE_DIRS} ${numpydir})

set(CMAKE_BUILD_TYPE Debug)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

add_executable(
	a.out
	src/main.cpp
	src/utility.h
	)
target_link_libraries(a.out fftw3 ${PYTHON_LIBRARIES} -ldl ${gsl_lib} ${gsl_lib2})
# target_link_libraries(a.out fftw3 -ldl ${gsl_lib} ${gsl_lib2})

# PYTHON_CONFIGURE_OPTS="--enable-shared"
