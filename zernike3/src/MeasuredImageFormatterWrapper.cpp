#include"MeasuredImageFormatterWrapper.h"
#include"MeasuredImageFormatter.h"

using namespace std;

extern "C"{
  void* MeasuredImageFormatter_new(int i){
    MeasuredImageFormatter*m = new MeasuredImageFormatter(i);
    return (void*)m;
  }
  void MeasuredImageFormatter_PrintInt(void*p){
    MeasuredImageFormatter*m=(MeasuredImageFormatter*)p;
    m->PrintInt();
  }
}


