#include "libLogoRecog.h"
#include <vector>
#include <string>
#include <fstream>
#include <iostream>
#include <opencv2/opencv.hpp>
#include <ratio>    
#include <chrono>

using namespace std;
using namespace cv;
using namespace std::chrono;    

int main(int argc, char *argv[])
{
  ifstream INF(argv[1]);
  int nt,ni;
  INF >> nt >> ni;
  LogoRecog rec;
  rec.init();
  vector<string> lu;
  for (int i = 0; i < nt; i++)
  {
    string ss;
    INF >> ss;
    lu.push_back(ss);
  }
  int nlg = rec.loadLogos(lu);
  cout<<"load logos "<<nlg<<endl;
  int nf = 0,ng = 0;
  for (int i = 0; i < ni; i++)
  {
    string ss;
    int lab;
    INF >> ss >> lab;
    Mat im = imread(ss);
    if (im.rows > im.cols)
    {
      if (im.rows != 960 || im.cols != 720)
        resize(im,im,Size(720,960),0,0,INTER_AREA);
    }
    else
    {
      if (im.rows != 720 || im.cols != 960)
        resize(im,im,Size(960,720),0,0,INTER_AREA);
    }
    high_resolution_clock::time_point t1 = high_resolution_clock::now();    
    int r = rec.recognize(im);
    high_resolution_clock::time_point t2 = high_resolution_clock::now();
    duration<double,std::ratio<1,1>> duration_s(t2-t1);    
    std::cout<<duration_s.count()<<" seconds"<<std::endl;  
    if (r == lab)
    {
      ng++;
    }
    else if (r < 0)
    {
      nf++;
      cout<<ss<<" "<<r<<endl;
    }
  }
  cout << ng <<' '<<nf<<' '<<ni-nf-ng<<' '<<ni<<endl;
  return 0;
}
