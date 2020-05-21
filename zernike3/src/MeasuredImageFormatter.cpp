#include"MeasuredImageFormatter.h"
#include<iostream>
#include<fstream>
#include<opencv2/core/core.hpp>
#include<opencv2/highgui/highgui.hpp>
#include<opencv2/imgproc/imgproc.hpp>
#include<algorithm>

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
      Adif_in_scaled_rot(dif_in_s0*df_ratio,dif_in_s1*df_ratio),
      opencvm1(Adif_in_scaled.size_0,Adif_in_scaled.size_1,CV_64F)

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


  // copy Adif_in_scaled to opencv matrix
  // open in opencv
  for(int i=0; i < Adif_in_scaled.size_0; i++)
    for(int j=0; j < Adif_in_scaled.size_1; j++)
      opencvm1.at<double>(cv::Point(i,j))=Adif_in_scaled(i,j);

  f.open("opencvm1.dat");
  for(int i=0; i < opencvm1.rows; i++){
      for(int j=0; j < opencvm1.cols; j++){
        f<<opencvm1.at<double>(cv::Point(i,j))<<"  ";
      }f<<endl;
    }
    f.close();

  // rotate matrix
  cv::Point2f pc(opencvm1.cols/2,opencvm1.rows/2);
  cv::Mat rotationmatrix=cv::getRotationMatrix2D(pc,rot_angle,1.0);
  cv::warpAffine(opencvm1,opencvm1,rotationmatrix,opencvm1.size());


  f.open("opencvm1_rot.dat");
  for(int i=0; i < opencvm1.rows; i++){
      for(int j=0; j < opencvm1.cols; j++){
        f<<opencvm1.at<double>(cv::Point(i,j))<<"  ";
      }f<<endl;
    }
    f.close();


  // // normalize to use imshow
  // auto maxp=max_element(opencvm1.ptr<double>(),opencvm1.ptr<double>()+Adif_in_scaled.length);
  // double maxv=*maxp;
  // for(int i=0; i < Adif_in_scaled.size_0; i++)
    // for(int j=0; j < Adif_in_scaled.size_1; j++)
      // opencvm1.at<double>(cv::Point(i,j))/=maxv;

  // cv::imshow("opencvm1",opencvm1);
  // cv::waitKey();
  // write the image to the output
  for(int i=0; i < Adif_out.size_0; i++){
    for(int j=0; j < Adif_out.size_1; j++){
      int _i=opencvm1.rows/2-Adif_out.size_0/2+i;
      int _j=opencvm1.cols/2-Adif_out.size_1/2+j;
      Adif_out(i,j)=opencvm1.at<double>(cv::Point(_i,_j));
    }
  }
  f.open("Adif_out.dat");
  for(int i=0; i < Adif_out.size_0; i++){
      for(int j=0; j < Adif_out.size_1; j++){
        f<<Adif_out(i,j)<<"  ";
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

