#include"MeasuredImageFormatter.h"
#include<iostream>

using namespace std;

MeasuredImageFormatter::MeasuredImageFormatter(int i,
      double*dif_in, int dif_in_s0, int dif_in_s1,
      double*dif_out, int dif_out_s0, int dif_out_s1){
  this->value=i;
  this->dif_in=dif_in;
  this->dif_in_s0=dif_in_s0;
  this->dif_in_s1=dif_in_s1;
  this->dif_out=dif_out;
  this->dif_out_s0=dif_out_s0;
  this->dif_out_s1=dif_out_s1;
}
void MeasuredImageFormatter::PrintInt(){
  cout<<"printInt called"<<endl;
  cout << "this->value => " << this->value << endl;
  cout << "dif_in[0] => " << dif_in[0] << endl;
  for(int i=0; i < dif_in_s0; i++){
    for(int j=0; j < dif_in_s1; j++){
      cout<<dif_in[i*dif_in_s1+j]<<"  ";
    }cout<<endl;
  }

}

