#include"c_arrays.h"

class MeasuredImageFormatter
{
  public:
  int value;
  array2d<double> Adif_in;
  array2d<double> Adif_out;
  MeasuredImageFormatter(int i,
      double*dif_in, int dif_in_s0, int dif_in_s1,
      double*dif_out, int dif_out_s0, int dif_out_s1
      );
  void PrintInt();
  void Format();
};

