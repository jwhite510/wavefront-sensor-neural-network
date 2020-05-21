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

MeasuredImageFormatter::MeasuredImageFormatter(double df_ratio,double rot_angle,
      double*dif_in, int dif_in_s0, int dif_in_s1,
      double*dif_out, int dif_out_s0, int dif_out_s1):
      Adif_in(dif_in,dif_in_s0,dif_in_s1),
      Adif_out(dif_out,dif_out_s0,dif_out_s1),
      Adif_in_scaled(dif_in_s0*df_ratio,dif_in_s1*df_ratio),
      Adif_in_scaled_rot(dif_in_s0*df_ratio,dif_in_s1*df_ratio)
{
  this->df_ratio=df_ratio;
  this->rot_angle=rot_angle;

  // initialize interpolator
  Interp = gsl_interp2d_alloc(gsl_interp2d_bilinear, Adif_in.size_0, Adif_in.size_1);

  interp_x1=new double[Adif_in.size_0];
  interp_y1=new double[Adif_in.size_1];
  interp_x2=new double[Adif_in_scaled.size_0];
  interp_y2=new double[Adif_in_scaled.size_1];
  cout<<"initializing MeasuredImageFormatter"<<endl;
  // TODO fix might need to fix linspace
  // Linspace(interp_x1,Adif_in.size_0,nextafter((double)-1,-10),nextafter((double)1,10));
  // Linspace(interp_y1,Adif_in.size_1,nextafter((double)-1,-10),nextafter((double)1,10));

  // Linspace(interp_x1,Adif_in.size_0,-1.1,1.1);
  // Linspace(interp_y1,Adif_in.size_1,-1.1,1.1);

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

  for(int i=0; i < Adif_in_scaled.size_0; i++)
    for(int j=0; j < Adif_in_scaled.size_1; j++)
    {

      bool a=(interp_x2[j]>=interp_x1[Adif_in.size_0-1]);
      bool b=(interp_x2[j]<=interp_x1[0]);
      bool c=(interp_y2[i]>=interp_y1[Adif_in.size_1-1]);
      bool d=(interp_y2[i]<=interp_y1[0]);
      if(a||b||c||d){
        // cout<<"interpolation is out of range"<<endl;
        Adif_in_scaled(i,j)=0;
      }else{
        Adif_in_scaled(i,j)=gsl_interp2d_eval(
            Interp,interp_x1,interp_y1,Adif_in.data,
            interp_x2[j],interp_y2[i],
            interp_xa,interp_ya
            );
      }
    }
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

  // rotate the image
  cout<<"rotate the image by "<<this->rot_angle<<"  degrees to the rotated matrix"<<endl;
  double rot_angle_rad=rot_angle * M_PI / 180.0;
  for(int i=0; i <Adif_in_scaled.size_0; i++){
    for(int j=0; j <Adif_in_scaled.size_1; j++){
      // centered
      int x = j - Adif_in_scaled.size_1/2;
      int y = i - Adif_in_scaled.size_0/2;

      int xnew=x*cos(rot_angle_rad) - y*sin(rot_angle_rad);
      int ynew=x*sin(rot_angle_rad) + y*cos(rot_angle_rad);

      int inew = ynew+Adif_in_scaled.size_0/2;
      int jnew = xnew+Adif_in_scaled.size_1/2;
      if(inew>=0&&inew<Adif_in_scaled_rot.size_0&&jnew>=0&&jnew<Adif_in_scaled_rot.size_1)
        Adif_in_scaled_rot(inew,jnew) = Adif_in_scaled(i,j);
    }
  }
  f.open("Adif_in_scaled_rot.dat");
  for(int i=0; i < Adif_in_scaled_rot.size_0; i++){
      for(int j=0; j < Adif_in_scaled_rot.size_1; j++){
        f<<Adif_in_scaled_rot(i,j)<<"  ";
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

