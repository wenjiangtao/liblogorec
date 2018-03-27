#include "libLogoRecog.h"
#include <vector>
#include <string>
#include <fstream>
#include <iostream>

using namespace std;

int main(int argc, char *argv[])
{
  ifstream INF(argv[1]);
  int nl;
  INF >> nl;
  for (int i = 0; i < nl; i++)
  {
    string opt;
    INF >> opt;
    int nim;
    INF >> nim;
    string tp,ss;
    vector<string> ims;
    INF >> tp;
    for (int j = 0; j < nim; j++)
    {
      INF >> ss;
      ims.push_back(ss);
    }
    trainOneLogo(tp,ims,opt);
  } 
  return 0;
}
