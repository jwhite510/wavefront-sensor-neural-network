class MeasuredImageFormatter
{
  public:
  int value;
  double*dif_in;
  int dif_in_s0;
  int dif_in_s1;
  double*dif_out;
  int dif_out_s0;
  int dif_out_s1;
  MeasuredImageFormatter(int i,
      double*dif_in, int dif_in_s0, int dif_in_s1,
      double*dif_out, int dif_out_s0, int dif_out_s1
      );
  void PrintInt();
};

