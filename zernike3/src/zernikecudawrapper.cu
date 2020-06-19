#include"zernikecuda.h"
#include<iostream>


using namespace std;
extern "C"{
  void* zcuda_new()
  {
    zcuda*m = new zcuda();
    return (void*)m;
  }
  // void zcuda_run(void*p,int N){
    // zcuda*m=(zcuda*)p;
    // m->run(N);
  // }
  // void zcuda_delete(void*p){
    // zcuda*m=(zcuda*)p;
    // delete m;
  // }
}
