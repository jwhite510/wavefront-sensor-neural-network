#include"c_arrays.h"
#include <gsl/gsl_interp2d.h>

class MeasuredImageFormatter
{
  public:
  double df_ratio;
  double rot_angle;
  array2d<double> Adif_in;
  array2d<double> Adif_out;
  array2d<double> Adif_in_scaled;
  array2d<double> Adif_in_scaled_rot;

  // 2d interpolation
  gsl_interp2d* Interp;
  gsl_interp_accel* interp_ya;
  gsl_interp_accel* interp_xa;
  double* interp_y1;
  double* interp_x1;
  double* interp_y2;
  double* interp_x2;

  MeasuredImageFormatter(double df_ratio,double rot_angle,
      double*dif_in, int dif_in_s0, int dif_in_s1,
      double*dif_out, int dif_out_s0, int dif_out_s1
      );
  void PrintInt();
  void Format();
};

void Linspace(double* data, int size, double min, double max);

