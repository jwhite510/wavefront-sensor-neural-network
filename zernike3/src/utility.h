#include<iostream>
#include <fftw3.h>
#include <complex>
#include <string>
#include <fstream>

using namespace std;
template<class T>
void Linspace(float min, float max, T & vector)
{
  // float vector[ncount];
  float dx = (max - min) / vector.size_0;
  for(int i=0; i < vector.size_0; i++)
  {
    vector(i) = min+dx*i;
  }
}
float RandomF()
{
  return static_cast <float> (rand()) / static_cast <float> (RAND_MAX);
}
float RandomF(float min, float max)
{
  // generate random number between min and max
  float randomnumber = min;
  float range = max - min;
  randomnumber += RandomF()*range;
  return randomnumber;
}
float Factorial(int number)
{
  return tgamma(number+1);
}
