#include"c_arrays.h"

class MeasuredImageFormatter
{
  public:
  double df_ratio;
  array2d<double> Adif_in;
  array2d<double> Adif_out;
  MeasuredImageFormatter(double df_ratio,
      double*dif_in, int dif_in_s0, int dif_in_s1,
      double*dif_out, int dif_out_s0, int dif_out_s1
      );
  void PrintInt();
  void Format();
};

