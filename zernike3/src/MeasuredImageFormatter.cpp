#include"MeasuredImageFormatter.h"
#include<iostream>

using namespace std;

MeasuredImageFormatter::MeasuredImageFormatter(int i){
  this->value=i;
}
void MeasuredImageFormatter::PrintInt(){
  cout<<"printInt called"<<endl;
  cout << "this->value => " << this->value << endl;
}

