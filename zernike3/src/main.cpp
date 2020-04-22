#include <iostream>
#include <chrono>
#include "pythonarrays.h"
#include "c_arrays.h"
#include "utility.h"
#include <assert.h>
#include <fftw3.h>
#include <ctime>
#include <gsl/gsl_interp2d.h>
#include <mpi.h>

using namespace std::chrono;

class ZernikeGenerator
{
  public:
  array1d<float>* y;
  array1d<float>* x;
  array2d<float>* rho;
  array2d<float>* phi;
  int N_computational;
  ZernikeGenerator(int N_computational_in)
  {
    N_computational = N_computational_in;
    // define memory
    x = new array1d<float>(N_computational);
    Linspace(7,-7, *x);
    y = new array1d<float>(N_computational);
    Linspace(7,-7, *y);

    // define rho array
    rho = new array2d<float>(N_computational, N_computational);
    for(int i=0; i < rho->size_0; i++)
      for(int j=0; j < rho->size_1; j++)
        (*rho)(i,j) = sqrt(pow((*x)(i), 2) + pow((*y)(j), 2));

    // define phi array
    phi = new array2d<float>(N_computational,N_computational);
  }
  void makezernike(int m, int n, array2d<float>& zernike_polynom)
  {
    int positive_m = 0;
    if( m >= 0)
      positive_m = 1;
    m = abs(m);
    // initialize array R to zeros
    for(int i=0; i < N_computational; i++)
      for(int j=0; j < N_computational; j++)
        zernike_polynom(i,j) = 0;

    for(int k=0; k <= (n-m)/2; k++){
      // calculate numerator
      float numerator;
      numerator = pow(-1, k);
      numerator *= Factorial(n - k);

      // calculate the denominator
      float denominator;
      denominator = Factorial(k);
      denominator *= Factorial(((n+m)/2)-k);
      denominator *= Factorial(((n-m)/2)-k);

      float scalar = numerator / denominator;
      for(int i=0; i < N_computational; i++)
        for(int j=0; j < N_computational; j++)
          zernike_polynom(i,j) += scalar * pow((*rho)(i,j), n - 2 * k);
    }
    for(int i=0; i < N_computational; i++)
      for(int j=0; j < N_computational; j++){
        (*phi)(i,j) = atan2((*y)(i) , (*x)(j));

        if(positive_m) {
          float value = cos(m * (*phi)(i,j));
          zernike_polynom(i,j)*= value;
        }
        else {
          float value = sin(m * (*phi)(i,j));
          zernike_polynom(i,j)*= value;
        }
      }
    // set valus outside unit circle to 0
    for(int i=0; i < N_computational; i++)
      for(int j=0; j < N_computational; j++) {
        if(sqrt(pow((*x)(i), 2) + pow((*y)(j), 2)) > 1)
          zernike_polynom(i,j) = 0;
      }
  }
  ~ZernikeGenerator()
  {
    delete y;
    delete x;
    delete rho;
    delete phi;
  }
};
struct Parameters
{
  float beta_Ta;
  float delta_Ta;
  const float dz = 1e-9;
  const float lam = 13.5e-9;
  const float k = 2 * M_PI / lam;
};

void zernike(const int &m, const int &n, array2d<float> & zernike_polynom)
{

  // define linspace
  array1d<float> x(zernike_polynom.size_0);
  Linspace(7,-7, x);

  array1d<float> y(zernike_polynom.size_0);
  Linspace(7,-7, y);


  // define rho array
  array2d<float> rho(zernike_polynom.size_0, zernike_polynom.size_0);
  for(int i=0; i < rho.size_0; i++)
    for(int j=0; j < rho.size_1; j++)
      rho(i,j) = sqrt(pow(x(i), 2) + pow(y(j), 2));


  // initialize array R to zeros
  array2d<float> R(zernike_polynom.size_0,zernike_polynom.size_0);
  for(int i=0; i < R.size_0; i++)
    for(int j=0; j < R.size_1; j++)
      R(i,j) = 0;

  for(int k=0; k <= (n-m)/2; k++){
    cout << "k => " << k << endl;
    // calculate numerator
    float numerator;
    numerator = pow(-1, k);
    numerator *= Factorial(n - k);

    // calculate the denominator
    float denominator;
    denominator = Factorial(k);
    denominator *= Factorial(((n+m)/2)-k);
    denominator *= Factorial(((n-m)/2)-k);

    float scalar = numerator / denominator;
    for(int i=0; i < R.size_0; i++)
      for(int j=0; j < R.size_1; j++)
        R(i,j) += scalar * pow(rho(i,j), n - 2 * k);
  }
  array2d<float> phi(zernike_polynom.size_0,zernike_polynom.size_0);
  for(int i=0; i < phi.size_0; i++)
    for(int j=0; j < phi.size_1; j++){
      phi(i,j) = atan2(y(i) , x(j));
      phi(i,j) = cos(-m *phi(i,j));
      R(i,j)*=phi(i,j);
    }
  // set valus outside unit circle to 0
  for(int i=0; i < R.size_0; i++)
    for(int j=0; j < R.size_1; j++) {
      if(sqrt(pow(x(i), 2) + pow(y(j), 2)) > 1)
        R(i,j) = 0;
    }
  for(int i=0; i < R.size_0; i++)
    for(int j=0; j < R.size_1; j++){
      zernike_polynom(i,j) = R(i,j);
    }


  return;

}
void fft2shift(array2d<complex<float>> & arrayin)
{
  complex<float> placeholder(0,0);
  for(int i=0; i< arrayin.size_0; i++)
    for(int j=0; j<arrayin.size_1/2; j++) {
      placeholder = arrayin(i,j);
      arrayin(i,j) = arrayin(i,j+arrayin.size_1/2);
      arrayin(i,j+arrayin.size_1/2) = placeholder;
  }
  for(int i=0; i< arrayin.size_0/2; i++)
    for(int j=0; j<arrayin.size_1; j++) {
      placeholder = arrayin(i,j);
      arrayin(i,j) = arrayin(i+arrayin.size_0/2,j);
      arrayin(i+arrayin.size_0/2,j) = placeholder;
  }
}

class Fft2
{
  public:
  int N;
  fftw_complex *in, *out;
  fftw_plan plan_forward;
  fftw_plan plan_backward;

  Fft2(int N_in)
  {
    N = N_in;
    // make a small array of fftw_complex, and compare with numpy
    in = (fftw_complex*) fftw_malloc(sizeof(fftw_complex) * N*N);
    out = (fftw_complex*) fftw_malloc(sizeof(fftw_complex) * N*N);
    plan_forward =  fftw_plan_dft_2d(N, N, in, out, FFTW_FORWARD, FFTW_ESTIMATE);
    plan_backward =  fftw_plan_dft_2d(N, N, in, out, FFTW_BACKWARD, FFTW_ESTIMATE);
  }
  void execute_fft(array2d<complex<float>> & complex_arr)
  {
    for(int i=0; i < N; i++)
      for(int j=0; j < N; j++) {
        in[i*N + j][0] = complex_arr(i,j).real(); // real part
        in[i*N + j][1] = complex_arr(i,j).imag(); // imag part
      }

    fftw_execute(plan_forward); /* repeat as needed */
    for(int i=0; i < N; i++)
      for(int j=0; j < N; j++) {
        complex_arr(i,j) = complex<float>(out[i*N+j][0], out[i*N+j][1]);
      }
  }
  void execute_ifft(array2d<complex<float>> & complex_arr)
  {
    for(int i=0; i < N; i++)
      for(int j=0; j < N; j++) {
        in[i*N + j][0] = complex_arr(i,j).real(); // real part
        in[i*N + j][1] = complex_arr(i,j).imag(); // imag part
      }

    fftw_execute(plan_backward); /* repeat as needed */
    for(int i=0; i < N; i++)
      for(int j=0; j < N; j++) {
        complex_arr(i,j) = complex<float>(out[i*N+j][0], out[i*N+j][1]);
      }
  }
  ~Fft2()
  {
    fftw_destroy_plan(plan_forward);
  }

};

class Interp2d
{
  public:
  gsl_interp2d* Interp;
  gsl_interp_accel* ya;
  gsl_interp_accel* xa;
  double* x;
  double* y;
  double* z;
  int size;

  Interp2d(double* x_in, double* y_in, double* z_in, int size_in)
  {
    x = x_in;
    y = y_in;
    z = z_in;
    size = size_in;

    Interp = gsl_interp2d_alloc(gsl_interp2d_bilinear, size, size);
    gsl_interp2d_init(Interp, x, y, z, size, size);
    // init accelerators
    xa = gsl_interp_accel_alloc();
    ya = gsl_interp_accel_alloc();
  }
  double GetValue(double x_value, double y_value)
  {
      return gsl_interp2d_eval(Interp, x, y, z, x_value, y_value, xa, ya);
  }
  ~Interp2d()
  {
    gsl_interp2d_free(Interp);
  }
};

void forward_propagate(array2d<complex<float>> &E, array2d<complex<float>> &slice, array1d<float> &f, const Parameters & p, Fft2 &fft_2)
{

  for(int i=0;i<E.length; i++)
    E.data[i] *= slice.data[i];

  // fourier transform E
  fft2shift(E);
  fft_2.execute_fft(E);
  fft2shift(E);

  for(int i=0; i<E.size_0;i++)
    for(int j=0; j<E.size_1;j++) {
      float gamma = sqrt(1 - pow((p.lam * f(i)),2) - pow((p.lam * f(j)),2));
      float k_sq = 2 * M_PI * p.dz / p.lam;
      complex<float> H = exp(complex<float>(0,gamma*k_sq));
      // complex<float> H = exp(complex<float>(0,2 * M_PI * p.dz / p.lam));
      E(i,j) *= H;
    }

  // inverse fourier transform E
  fft2shift(E);
  fft_2.execute_ifft(E);
  fft2shift(E);
  // (and shift)

  // divide by scale factor
  // this makes the fft and ifft operations result in the same array values
  for(int i=0; i< E.length; i++)
    E.data[i] /= E.length;

}
void create_slice(array2d<complex<float>> & slice, const array2d<float> & wavefrontsensor, const Parameters & p)
{
  for(int i=0;i<slice.length; i++) {
    // TODO create this properly
    if(wavefrontsensor.data[i] < 0.5) {
      slice.data[i] = exp(complex<float>(-1 * p.k * p.beta_Ta * p.dz, 0));
      slice.data[i] *= exp(complex<float>(0,-1 * p.k * p.delta_Ta * p.dz));
    }
    else{
      slice.data[i] = complex<float>(1,0);
    }
  }
}

void crop_object(array2d<complex<float>> & complex_object, array2d<double> & cropped_real, array2d<double> & cropped_imag)
{
  int N_computational = complex_object.size_0;
  int crop_size = cropped_real.size_0;
  // get top left corner of complex_object
  int c1 = (N_computational/2) - (crop_size/2);
  int c2 = c1;

  for(int i=0; i < crop_size; i++)
    for(int j=0; j < crop_size; j++) {
      cropped_real(i,j) = real(complex_object(i+c1,j+c2));
      cropped_imag(i,j) = imag(complex_object(i+c1,j+c2));
    }
}
class GaussianPropagator
{
  array1d<float>* x;
  array1d<float>* y;
  array2d<float>* gaussian_amp;
  int N_computational;
  Fft2* fft_2;
  public:
  GaussianPropagator(int N_computational_in)
  {
    N_computational = N_computational_in;

    // initialize the FFT
    fft_2 = new Fft2(N_computational);
    // define linspace
    x = new array1d<float>(N_computational);
    Linspace(1,-1, *x);

    y = new array1d<float>(N_computational);
    Linspace(1,-1, *y);

    float width = 0.05;
    gaussian_amp = new array2d<float>(N_computational, N_computational);
    for(int i=0; i< gaussian_amp->size_0; i++)
      for(int j=0; j< gaussian_amp->size_0; j++)
        (*gaussian_amp)(i,j) = exp( - pow((*x)(i), 2) / pow(width,2))*exp( - pow((*y)(j), 2) / pow(width,2));
  }
  void propagate(array2d<complex<float>> &field, array2d<float>& phase)
  {
    for(int i=0; i< field.size_0; i++)
      for(int j=0; j< field.size_1; j++) {
        field(i,j) = complex<float>((*gaussian_amp)(i,j), 0); // gaussian amplitude
        field(i,j) *= exp(complex<float>(0,phase(i,j))); // apply phase
      }
    fft2shift(field);
    // fft complex_object
    fft_2->execute_fft(field);
    fft2shift(field);
  }
  ~GaussianPropagator()
  {
    delete x;
    delete y;
    delete gaussian_amp;
    delete fft_2;
  }
};
class CropAndInterpolateComplex
{
  public:
  array2d<double>* cropped_propagated_beam_real;
  array2d<double>* cropped_propagated_beam_imag;
  array1d<double>* x2;
  array1d<double>* y2;
  array1d<double>* y3;
  array1d<double>* x3;
  int n_interp;
  int crop_size;

  CropAndInterpolateComplex(int n_interp_in, int crop_size_in)
  {
    // n_interp : interpolate to this grid
    // crop_size : start with an image this size
    n_interp = n_interp_in;
    crop_size = crop_size_in;
    // crop the image size:
    cropped_propagated_beam_real = new array2d<double>(crop_size, crop_size);
    cropped_propagated_beam_imag = new array2d<double>(crop_size, crop_size);
    // axes for interpolation
    y2 = new array1d<double>(crop_size);
    Linspace(-1.1,1.1, *y2);
    x2 = new array1d<double>(crop_size);
    Linspace(-1.1,1.1, *x2);

    y3 = new array1d<double>(n_interp);
    x3 = new array1d<double>(n_interp);
  }
  void crop_interp(array2d<complex<float>> &arr, array2d<complex<float>> &interped_arr, float random_size_min)
  {

    // crop the image into the smaller arrays
    crop_object(arr,
        *cropped_propagated_beam_real, // OUT
        *cropped_propagated_beam_imag  // OUT
        );

    // Python.call_function_np("plot_complex", arr.data, vector<int>{arr.size_0,arr.size_1}, PyArray_COMPLEX64);
    // Python.call_function_np("plot_complex", cropped_propagated_beam_real->data, vector<int>{cropped_propagated_beam_real->size_0,cropped_propagated_beam_real->size_1}, PyArray_FLOAT64);

    // create interpolation object with data
    Interp2d interp2d_real(x2->data, y2->data, cropped_propagated_beam_real->data, crop_size);
    Interp2d interp2d_imag(x2->data, y2->data, cropped_propagated_beam_imag->data, crop_size);

    // determine size of image (relative)
    float image_relative_size = RandomF(random_size_min, 1.0); // % size of the original image
    // max x, y location:
    float max_xy_location = 1.0 - image_relative_size;
    // x, y start is between these two values
    float x_tl = RandomF(0, max_xy_location);
    float x_br = x_tl + image_relative_size;
    float y_tl = RandomF(0, max_xy_location);
    float y_br = y_tl + image_relative_size;

    x_tl *=2; x_tl -=1; // shift to -1. 1 coordinates
    x_br *=2; x_br -=1; // shift to -1. 1 coordinates
    y_tl *=2; y_tl -=1; // shift to -1. 1 coordinates
    y_br *=2; y_br -=1; // shift to -1. 1 coordinates

    // interpolate at a random offset and size
    Linspace(x_tl,x_br, *x3);
    Linspace(y_tl,y_br, *y3);
    // interplate onto new image
    for(int i=0; i < n_interp; i++)
      for(int j=0; j < n_interp; j++) {
        interped_arr(i,j) = complex<float>(interp2d_real.GetValue((*x3)(j), (*y3)(i)),
            interp2d_imag.GetValue((*x3)(j), (*y3)(i)));
      }
  }
  ~CropAndInterpolateComplex()
  {
    delete y2;
    delete y3;
    delete x2;
    delete x3;
  }
};
class WaveFrontSensor
{
public:
  array2d<float>* wavefrontsensor;
  array1d<float>* f;
  WaveFrontSensor(int n_interp, PythonInterp &Python)
  {
    wavefrontsensor = new array2d<float>(n_interp, n_interp);
    f = new array1d<float>(n_interp);

    PyObject* wfs = Python.get("get_wavefront_sensor");
    vector<int> size_wfs;
    Python.get_returned_numpy_arr(wfs, wavefrontsensor->data, size_wfs);

    // get the wavefront sensor frequency axis
    PyObject* wfs_f = Python.get("get_wavefront_sensor_f");
    vector<int> size_wfs_f;
    Python.get_returned_numpy_arr(wfs_f, f->data, size_wfs_f);

  }
  ~WaveFrontSensor()
  {
    delete wavefrontsensor;
    delete f;
  }
};

struct zernike_c
{
  int m;
  int n;
};

struct RunParameters
{
  string RunName;
  int Samples;
  int BufferSize;
};
RunParameters parseargs(int argc, char *argv[])
{
  RunParameters runParameters;
  runParameters.Samples = 0;
  runParameters.RunName = "NONE";
  runParameters.BufferSize =0;
  for(int i=0; i < argc; i++)
    if(string(argv[i]) == "--count")
      runParameters.Samples = atoi(argv[i+1]);

  for(int i=0; i < argc; i++)
    if(string(argv[i]) == "--name")
      runParameters.RunName = argv[i+1];

  for(int i=0; i < argc; i++)
    if(string(argv[i]) == "--buffersize")
      runParameters.BufferSize = atoi(argv[i+1]);

  return runParameters;
}

struct DataGenerator
{
  PythonInterp Python;
  int N_computational;
  array2d<float> zernike_polynom;
  ZernikeGenerator zernikegenerator;
  array2d<complex<float>> complex_object;
  GaussianPropagator gaussianp;
  int crop_size;
  int n_interp;
  CropAndInterpolateComplex cropinterp;
  WaveFrontSensor wavefonts;
  Parameters params_cu;
  Parameters params_Si;
  array2d<complex<float>> slice_cu;
  array2d<complex<float>> slice_Si;
  Fft2 fft_2_interp;
  float Si_distance;
  float cu_distance;
  int steps_Si;
  int steps_cu;
  int start_n;
  int max_n;
  array3d<float>* mn_polynomials;
  vector<zernike_c> zernike_cvector;

  DataGenerator(const char* pythonhomedir,
      int N_computational,
      int crop_size,
      int n_interp)

    : Python(pythonhomedir, "utility"),
      zernike_polynom(N_computational,N_computational),
      zernikegenerator(N_computational),
      complex_object(N_computational, N_computational),
      gaussianp(N_computational),
      cropinterp(n_interp, crop_size),
      wavefonts(n_interp, Python),
      slice_cu(n_interp, n_interp),
      slice_Si(n_interp, n_interp),
      fft_2_interp(n_interp)
  {
    this->N_computational = N_computational;
    this->crop_size = crop_size;
    this->n_interp = n_interp;

    // seed random
    srand(time(0));
    // use a square grid
    // crop the image size:

    // define materials
    params_cu.beta_Ta = 0.0612;
    params_cu.delta_Ta = 0.03748;
    params_Si.beta_Ta = 0.00926;
    params_Si.delta_Ta = 0.02661;

    // single slice of the material
    create_slice(slice_cu, *wavefonts.wavefrontsensor, params_cu);
    create_slice(slice_Si, *wavefonts.wavefrontsensor, params_Si);

    // initialize FFT
    // lambda: 13.5 nm
    // forward propagate thorugh 50 nm Si3N4 -> delta:0.02661 , beta:0.00926
    // forward_propagate through 150 nm Cu -> delta:0.03748 , beta:0.0612
    Si_distance = 50e-9;
    cu_distance = 150e-9;
    steps_Si = Si_distance / params_Si.dz;
    steps_cu = cu_distance / params_cu.dz;

    // std::cout << "steps_cu" << " => " << steps_cu << std::endl;
    // std::cout << "steps_Si" << " => " << steps_Si << std::endl;
    start_n = 2;
    max_n = 4;
    for(int n=start_n; n <= max_n; n++)
      for(int m=n; m >=-n ; m-=2)
        zernike_cvector.push_back({m,n});

    // 3d array to hold the zernike polynomials
    mn_polynomials = new array3d<float>(zernike_cvector.size(),N_computational,N_computational);

    int mn_polynomials_index = 0;
    for(zernike_c z : zernike_cvector) {

      // to print all the zernike m and n values
      // cout << "i:" << mn_polynomials_index << " m:" << z.m << "n:" << z.n << endl;

      // generate the zernike coefficient and add it to the matrix
      zernikegenerator.makezernike(z.m, z.n, zernike_polynom);
      for(int i=0; i < N_computational; i++)
        for(int j=0; j < N_computational; j++)
          (*mn_polynomials)(mn_polynomials_index, i, j) = zernike_polynom(i, j);
      mn_polynomials_index ++;
    }


  }
  void makesample(array2d<complex<float>>  &interped_arr)
  {
    // auto time1 = high_resolution_clock::now();
    // get a random value for each coefficient
    for(int i=0; i < zernike_polynom.length; i++)
      zernike_polynom.data[i] = 0.0;

    // for each zernike coeffieicent
    for(int i=0; i < zernike_cvector.size(); i++) {
      // make random scalar
      float r1 = RandomF();
      r1 *= 9; // scalar
      if(RandomF() > 0.5)
        r1 *= -1;

      // float r1 = 2.0; // TODO return this to normal, its disabled to show the cropping

      // std::cout << "r" << " => " << r1 << std::endl;
      for(int j=0; j < N_computational; j++)
        for(int k=0; k < N_computational; k++)
          zernike_polynom(j,k) +=  r1 * (*mn_polynomials)(i, j, k);
    }

    // apply this phase and propagate it
    gaussianp.propagate(complex_object, zernike_polynom);
    cropinterp.crop_interp(complex_object,
        interped_arr, // OUT
        0.1 // between 0 and 1 : the minimum image scale after interpolation
        );

    // TODO: do not set the electric field normalized after multiplying by the wavefront mask
    // !!! -- make it between something and 1, not 0 and 1

    // Python.call_function_np("plot_complex_diffraction", interped_arr.data, vector<int>{interped_arr.size_0,interped_arr.size_1}, PyArray_COMPLEX64);
    // multiplby wavefront sensor
    for(int i=0; i < interped_arr.length; i++)
      interped_arr.data[i] *= wavefonts.wavefrontsensor->data[i];
    // normalize
    normalize(interped_arr);

    // Python.call_function_np("plot_complex_diffraction", interped_arr.data, vector<int>{interped_arr.size_0,interped_arr.size_1}, PyArray_COMPLEX64);
    // propagate through materials
    for(int i=0; i<steps_Si; i++) // 50 nm & dz: 10 nm
      forward_propagate(interped_arr, slice_Si, *wavefonts.f, params_Si, fft_2_interp);
    for(int i=0; i<steps_cu; i++)
      forward_propagate(interped_arr, slice_cu, *wavefonts.f, params_cu, fft_2_interp);

  }

  ~DataGenerator()
  {
    delete mn_polynomials;
  }

};

int main(int argc, char *argv[])
{

  // MPI parameters
  int process_Rank, size_Of_Cluster;
  MPI_Init(NULL, NULL);
  MPI_Comm_size(MPI_COMM_WORLD, &size_Of_Cluster);
  MPI_Comm_rank(MPI_COMM_WORLD, &process_Rank);
  // std::cout << "process_Rank" << " => " << process_Rank << std::endl;
  // std::cout << "size_Of_Cluster" << " => " << size_Of_Cluster << std::endl;

  RunParameters runParameters = parseargs(argc, argv);

  if(runParameters.RunName == "NONE" || runParameters.Samples == 0 || runParameters.BufferSize == 0) {
    if(process_Rank==0)
      cout << "--count --buffersize or --name not set" << endl;
    MPI_Finalize();
    return 1; // parameter not set
  }

  // data generation parameters
  int buffer_size = runParameters.BufferSize; // the amount of samples to store before saving them to hdf5

  // physical parameters
  int n_interp = 128;
  int crop_size = 200;
  int N_computational = 1024;

  // MPI parameters
  int samples_per_process = runParameters.Samples / size_Of_Cluster; // samples for each process to generate
  if(process_Rank == 0) {
    cout << "generating " << runParameters.Samples << " samples" << endl;
    cout << "samples_per_process => " << samples_per_process << endl;
    cout << "buffer_size => " << buffer_size << endl;
  }
  if(samples_per_process % buffer_size != 0) {
    if(process_Rank == 0)
      cout << "ERROR: choose a sample size that is divisible by the buffer size" << endl;
    MPI_Finalize();
    return 0;
  }

  // create data generator and buffers
  array2d<complex<float>> interped_arr(n_interp, n_interp);
  array3d<complex<float>> samples_buffer(buffer_size,n_interp,n_interp);
  DataGenerator datagenerator("/home/zom/Projects/diffraction_net/venv/",
      N_computational, // N_computational
      crop_size, // crop_size
      n_interp // n_interp
      );

  // process 0 initialize the data set
  if(process_Rank == 0)
    datagenerator.Python.call("create_dataset",runParameters.RunName.c_str());
  MPI_Barrier(MPI_COMM_WORLD);


  // each process fill buffer
  int current_buffer_index = 0;
  for(int i=0; i < samples_per_process; i++) {
    cout << "process" << process_Rank << "generating sample: " << i << endl;
    datagenerator.makesample(interped_arr);

    // add to buffer
    for(int i=0; i < n_interp; i++)
      for(int j=0; j < n_interp; j++)
        samples_buffer(current_buffer_index,i,j) = interped_arr(i,j);
    current_buffer_index++;

    if(current_buffer_index == buffer_size) {
      // save the data to hdf5, reset buffer index
      // synchronize threads here in for loop
      current_buffer_index = 0;
      for(int i=0; i < size_Of_Cluster; i++) {
        MPI_Barrier(MPI_COMM_WORLD);
        if(process_Rank == i) {
          cout << "process " << process_Rank << " save to hdf5 " << endl;
          datagenerator.Python.call_function_np("save_to_hdf5",runParameters.RunName.c_str(), samples_buffer.data, vector<int>{samples_buffer.size_0,samples_buffer.size_1,samples_buffer.size_2}, PyArray_COMPLEX64);
        }
      }
      MPI_Barrier(MPI_COMM_WORLD);
    }
  }

  MPI_Finalize();
  return 0;




}
