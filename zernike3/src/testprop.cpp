#include<iostream>
#include "zernikedatagen.h"

using namespace std;

int main(){
  PythonInterp Python("/home/jonathon/Projects/diffraction_net/venv/", "testprop");
  array2d<float> object(1024,1024);
  array1d<float> f(1024);

  // get object
  PyObject* wfs = Python.get("get_object_sensor"); vector<int> size_wfs;
  Python.get_returned_numpy_arr(wfs, object.data, size_wfs);

  // get the object sensor frequency axis
  PyObject* wfs_f = Python.get("get_object_sensor_f"); vector<int> size_wfs_f;
  Python.get_returned_numpy_arr(wfs_f, f.data, size_wfs_f);

  // wavefront
  array2d<complex<float>> wave(1024,1024);
  for(int i=0; i < wave.length; i++)
    wave.data[i]=complex<float>(1,0);

  // make material slice
  array2d<complex<float>> slice_cu(1024,1024);
  array2d<complex<float>> slice_Si(1024,1024);
  Parameters params_cu;
  // Parameters params_Si;

  params_cu.beta_Ta =  0.0646711215;
  params_cu.delta_Ta = 0.103724159;

  // params_Si.beta_Ta = 0.00926;
  // params_Si.delta_Ta = 0.02661;

  create_slice(slice_cu, object, params_cu);
  // create_slice(slice_Si, object, params_cu);

  // float Si_distance = 50e-9;
  float cu_distance = 100e-9;

  int steps_cu = cu_distance / params_cu.dz;
  // int steps_Si = Si_distance / params_Si.dz;

  Fft2 fft2(1024);

  // for(int i=0; i<steps_Si; i++) // 50 nm & dz: 10 nm
  //   forward_propagate(wave, slice_Si, f, params_Si, fft2);
  // Python.call_function_np("plot", object.data, vector<int>{object.size_0,object.size_1}, PyArray_FLOAT32);
  // Python.call_function_np("plot_complex", slice_cu.data, vector<int>{slice_cu.size_0,slice_cu.size_1}, PyArray_COMPLEX64);
  // Python.call_function_np("plot_complex", wave.data, vector<int>{wave.size_0,wave.size_1}, PyArray_COMPLEX64);

  Python.call_function_np("plot_complex", wave.data, vector<int>{wave.size_0,wave.size_1}, PyArray_COMPLEX64);
  for(int i=0; i<steps_cu; i++) {
    forward_propagate(wave, slice_cu, f, params_cu, fft2);
  }
  Python.call_function_np("plot_complex", wave.data, vector<int>{wave.size_0,wave.size_1}, PyArray_COMPLEX64);
  Python.call("show");

}
