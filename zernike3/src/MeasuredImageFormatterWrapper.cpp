#include"MeasuredImageFormatterWrapper.h"
#include"MeasuredImageFormatter.h"

using namespace std;

extern "C"{
  void* MeasuredImageFormatter_new(int i,
    double* dif_in, int dif_in_s0, int dif_in_s1,
    double* dif_out, int dif_out_s0, int dif_out_s1)
  {
    MeasuredImageFormatter*m = new MeasuredImageFormatter(i,
        dif_in, dif_in_s0, dif_in_s1,
        dif_out, dif_out_s0, dif_out_s1
        );
    return (void*)m;
  }
  void MeasuredImageFormatter_PrintInt(void*p){
    MeasuredImageFormatter*m=(MeasuredImageFormatter*)p;
    m->PrintInt();
  }
}


