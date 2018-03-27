#ifndef LIBLOGORECOG_H
#define LIBLOGORECOG_H

#include <stdio.h>
#include <iostream>
#include <vector>
#include <string>
#include <fstream>

#include "opencv2/core.hpp"
#include "opencv2/core/ocl.hpp"
#include "opencv2/features2d.hpp"
#include "opencv2/flann.hpp"
#include "opencv2/imgproc.hpp"
#include "opencv2/xfeatures2d.hpp"
#include "opencv2/highgui.hpp"
#include <opencv2/calib3d.hpp>


const int LR_MAX_FT = 500; 
const int LR_LOW_SUP = 10;
const int LR_LOW_HOMO = 7;
const int LR_DET_SZ = 240;
const float LR_DR = 0.75;
const float LR_BLV = 100.0f;
const int LR_MARGIN = 1;

int trainOneLogo(const std::string &turl, const std::vector<std::string>  &iurls, const std::string &optFile,  const bool use_inverse = true);

struct OneLogo
{
  int minSup;
  bool inv;
  std::vector<cv::KeyPoint> kp, kpi;
  cv::Mat des,desi;
};

#define LR_FLANN 1

class LogoRecog
{
 public:
  int loadLogos(const std::vector<std::string> &urls, const bool clear = true);
  int recognize(const cv::Mat &image);
  void init(const bool _blur_det = false, const float _blv = -1, const int _det_sz = -1, const int _max_ft = -1, const int _low_sup = -1, const int _low_homo = -1, const float _dr = -1.0f);
  int match(const std::vector<cv::KeyPoint> &kp1, const std::vector<cv::KeyPoint> &kp2, const cv::Mat &des1, const cv::Mat &des2, int thr = -1);

  cv::Ptr<cv::Feature2D> f2d;
  int det_sz;
  int margin;

 
 private:
  std::vector<OneLogo> logos;
  cv::FlannBasedMatcher matcher;
  cv::BFMatcher bmatcher;
  int max_ft,low_sup,low_homo;
  float dr;
  bool blur_det;
  float blv;
  bool is_init;
};

#endif
