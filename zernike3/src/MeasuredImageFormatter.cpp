#include"MeasuredImageFormatter.h"
#include<iostream>

using namespace std;

MeasuredImageFormatter::MeasuredImageFormatter(int i,
      double*dif_in, int dif_in_s0, int dif_in_s1,
      double*dif_out, int dif_out_s0, int dif_out_s1):
      Adif_in(dif_in,dif_in_s0,dif_in_s1),
      Adif_out(dif_out,dif_out_s0,dif_out_s1)
{
  this->value=i;
}
void MeasuredImageFormatter::PrintInt(){
  cout<<"printInt called"<<endl;
  cout << "this->value => " << this->value << endl;
  for(int i=0; i < Adif_in.size_0; i++){
    for(int j=0; j < Adif_in.size_1; j++){
      cout<<Adif_in(i,j)<<" ";
    }cout<<endl;
  }
}
void MeasuredImageFormatter::Format(){
  cout<<"formatting diffraction pattern c++"<<endl;
  cout<<"Adif_in:"<<endl;
  for(int i=0; i < Adif_in.size_0; i++){
      for(int j=0; j < Adif_in.size_1; j++){
        cout<<Adif_in(i,j)<<" ";
      }cout<<endl;
    }
  // print out
  cout<<"Adif_out:"<<endl;
  for(int i=0; i < Adif_out.size_0; i++){
      for(int j=0; j < Adif_out.size_1; j++){
        cout<<Adif_out(i,j)<<" ";
      }cout<<endl;
    }

  for(int i=0; i < Adif_out.size_0; i++){
    for(int j=0; j < Adif_out.size_1; j++){
      Adif_out(i,j)=2*Adif_in(i,j);
    }
  }




}

