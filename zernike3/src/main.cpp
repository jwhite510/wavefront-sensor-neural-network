#include "zernikedatagen.h"
#include <mpi.h>
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

  // seed random number generators for all the processes
  uint randomseed = time(0) + process_Rank;
  // uint randomseed = time(0);
  srand(randomseed);
  cout << "process:" << process_Rank << " random seed:" << randomseed << endl;

  // create data generator and buffers
  array2d<complex<float>> interped_arr(n_interp, n_interp);
  array2d<complex<float>> interped_arr_wavefront(n_interp, n_interp);
  array3d<complex<float>> samples_buffer(buffer_size,n_interp,n_interp);
  array3d<complex<float>> interped_arr_wavefront_buffer(buffer_size,n_interp,n_interp);
  DataGenerator datagenerator("/home/jonathon/Projects/diffraction_net/venv/",
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
    if(i%20==0)
      cout << "process" << process_Rank << "generating sample: " << i << endl;
    datagenerator.makesample(interped_arr, interped_arr_wavefront);

    // add to buffer
    for(int i=0; i < n_interp; i++)
      for(int j=0; j < n_interp; j++) {
        samples_buffer(current_buffer_index,i,j) = interped_arr(i,j);
        interped_arr_wavefront_buffer(current_buffer_index,i,j) = interped_arr_wavefront(i,j);
      }
    current_buffer_index++;

    if(current_buffer_index == buffer_size) {
      // save the data to hdf5, reset buffer index
      // synchronize threads here in for loop
      current_buffer_index = 0;
      for(int i=0; i < size_Of_Cluster; i++) {
        MPI_Barrier(MPI_COMM_WORLD);
        if(process_Rank == i) {
          // TODO: make this python function accept both the wavefront and the wavefront multiplied by sensor
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
