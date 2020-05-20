#include"MeasuredImageFormatter.h"
#include<iostream>
#include<fstream>

using namespace std;

void Linspace(double* data, int size, double min, double max){
  data[0]=min;
  double dx=(max-min)/(double)(size-1);
  for(int i=1; i < size; i++){
    data[i]=data[i-1]+dx;
  }
}

MeasuredImageFormatter::MeasuredImageFormatter(double df_ratio,
      double*dif_in, int dif_in_s0, int dif_in_s1,
      double*dif_out, int dif_out_s0, int dif_out_s1):
      Adif_in(dif_in,dif_in_s0,dif_in_s1),
      Adif_out(dif_out,dif_out_s0,dif_out_s1),
      Adif_in_scaled(dif_in_s0*df_ratio,dif_in_s1*df_ratio)
{
  this->df_ratio=df_ratio;

  // initialize interpolator
  Interp = gsl_interp2d_alloc(gsl_interp2d_bilinear, Adif_in.size_0, Adif_in.size_1);

  interp_x1=new double[Adif_in.size_0];
  interp_y1=new double[Adif_in.size_1];
  interp_x2=new double[Adif_in_scaled.size_0];
  interp_y2=new double[Adif_in_scaled.size_1];

  // linspace between -1 and 1
  Linspace(interp_x1,Adif_in.size_0,-1,1);
  Linspace(interp_y1,Adif_in.size_1,-1,1);
  Linspace(interp_x2,Adif_in_scaled.size_0,-1,1);
  Linspace(interp_y2,Adif_in_scaled.size_1,-1,1);

  gsl_interp2d_init(Interp, interp_x1, interp_y1, Adif_in.data, Adif_in.size_0, Adif_in.size_1);
  interp_xa = gsl_interp_accel_alloc();
  interp_ya = gsl_interp_accel_alloc();

}
void MeasuredImageFormatter::PrintInt(){
  cout<<"printInt called"<<endl;
  cout << "this->df_ratio => " << this->df_ratio << endl;
  for(int i=0; i < Adif_in.size_0; i++){
    for(int j=0; j < Adif_in.size_1; j++){
      cout<<Adif_in(i,j)<<" ";
    }cout<<endl;
  }
}
void MeasuredImageFormatter::Format(){
  cout<<"formatting diffraction pattern c++"<<endl;
  cout<<"Adif_in:"<<endl;
  cout << "this->df_ratio => " << this->df_ratio << endl;
  std::cout << "Adif_in.size_0" << " => " << Adif_in.size_0 << std::endl;
  std::cout << "Adif_in.size_1" << " => " << Adif_in.size_1 << std::endl;
  std::cout << "Adif_in_scaled.size_0" << " => " << Adif_in_scaled.size_0 << std::endl;
  std::cout << "Adif_in_scaled.size_1" << " => " << Adif_in_scaled.size_1 << std::endl;

  // interpolate input onto Adif_in_scaled
  for(int i=0; i < Adif_in_scaled.size_0; i++)
    for(int j=0; j < Adif_in_scaled.size_1; j++)
      Adif_in_scaled(i,j)=gsl_interp2d_eval(
          Interp,interp_x1,interp_y1,Adif_in.data,
          interp_x2[j],interp_y2[i],
          interp_xa,interp_ya
          );

  ofstream f;

  f.open("Adif_in.dat");

  for(int i=0; i < Adif_in.size_0; i++){
    for(int j=0; j < Adif_in.size_1; j++){
      f<<Adif_in(i,j)<<"  ";
    }f<<endl;
  }
  f.close();

  f.open("Adif_in_scaled.dat");
  for(int i=0; i < Adif_in_scaled.size_0; i++){
    for(int j=0; j < Adif_in_scaled.size_1; j++){
      f<<Adif_in_scaled(i,j)<<"  ";
    }f<<endl;
  }
  f.close();

  // for(int i=0; i < Adif_in.size_0; i++){
  //     for(int j=0; j < Adif_in.size_1; j++){
  //       cout<<Adif_in(i,j)<<" ";
  //     }cout<<endl;
  //   }
  // // print out
  // cout<<"Adif_out:"<<endl;
  // for(int i=0; i < Adif_out.size_0; i++){
  //     for(int j=0; j < Adif_out.size_1; j++){
  //       cout<<Adif_out(i,j)<<" ";
  //     }cout<<endl;
  //   }

  // for(int i=0; i < Adif_out.size_0; i++){
  //   for(int j=0; j < Adif_out.size_1; j++){
  //     Adif_out(i,j)=2*Adif_in(i,j);
  //   }
  // }




}

