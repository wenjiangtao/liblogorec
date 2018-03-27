#include "libLogoRecog.h"
#include <unordered_set>
#include <cmath>
#include <cstdio>

using namespace std;
using namespace cv;

int trainOneLogo(const std::string &turl, const std::vector<std::string>  &iurls, const std::string &optFile, const bool use_inverse)
{
  LogoRecog rec;
  rec.init();
  cv::Ptr<cv::Feature2D> f2d(rec.f2d);
  Mat tpl = imread(turl,0);
  if (tpl.empty())
    return -1;
  Mat des,desi,ndes;
  std::vector<cv::KeyPoint> kp, kpi,nkp;
  f2d -> detectAndCompute(tpl,cv::noArray(),kp,des);
  if (use_inverse)
  {
    Mat tpli(255-tpl);
    f2d -> detectAndCompute(tpli,cv::noArray(),kpi,desi);
  }
  int sup = 100000;
  Mat im,imr,imc,imf;
  Rect r;
  cout<<turl<<" "<<iurls.size()<<endl;
  for (int i = 0; i < iurls.size(); i++)
  {
    im = imread(iurls[i],0);
    if (im.empty())
      continue;
    if (im.rows > im.cols)
    {
      resize(im,imr,Size(720,960),0,0,INTER_AREA);
      r = Rect(360-240,480-240,480,480);
    }
    else
    {
      resize(im,imr,Size(960,720),0,0,INTER_AREA);
      r = Rect(480-240,360-240,480,480);
    }
    imc = imr(r);
    resize(imc,imf,Size(rec.det_sz,rec.det_sz),0,0,INTER_AREA);
    f2d -> detectAndCompute(imf,cv::noArray(),nkp,ndes);
    int l = rec.match(kp,nkp,des,ndes,-1);
    int l1 = -1;
    if (use_inverse && l < 0)
      l1 = rec.match(kpi,nkp,desi,ndes,-1);
    int ll = max(l,l1);
    cout<<iurls[i]<<" "<<l<<" "<<l1<<" "<<ll<<endl;
    if (ll > 0)
      sup = min(sup,ll);
  }
  cout<<sup<<endl;

  if (sup < 100000)
  {
    sup -= rec.margin;
    ofstream OF(optFile.c_str());
    if (use_inverse)
      OF << sup << ' ' << 1 <<endl;
    else
      OF << sup << ' ' << 0 <<endl;
    OF << kp.size()<<endl;
    for (int i = 0; i < kp.size(); i++)
      OF << kp[i].pt.x << ' ' << kp[i].pt.y<< ' ' <<kp[i].size<<' ' <<kp[i].angle << ' '<<kp[i].response<<endl;
    for (int i = 0; i < kp.size(); i++)
    {
      for (int j = 0; j < 128; j++)
        OF << des.at<float>(i,j) <<' ';
      OF << endl;
    }
    if (use_inverse)
    {
      OF << kpi.size() <<endl;
      for (int i = 0; i < kpi.size(); i++)
        OF << kpi[i].pt.x << ' ' << kpi[i].pt.y<< ' ' <<kpi[i].size<<' ' <<kpi[i].angle << ' '<<kpi[i].response<<endl;
      for (int i = 0; i < kpi.size(); i++)
      {
        for (int j = 0; j < 128; j++)
          OF << desi.at<float>(i,j) <<' ';
        OF << endl;
      }
    } 
  
    return sup;
  }
  return -1;
}

void LogoRecog::init(const bool _blur_det, const float _blv, const int _det_sz, const int _max_ft, const int _low_sup, const int _low_homo, const float _dr)
{
  //omp_set_num_threads(1);
  cv::setNumThreads(1);
  cv::ocl::setUseOpenCL(false);
  if (_max_ft < 0)
    max_ft = LR_MAX_FT;
  else
    max_ft = _max_ft;
  if (_low_sup < 0)
    low_sup = LR_LOW_SUP;
  else
    low_sup = _low_sup;
  if (_low_homo < 0)
    low_homo = LR_LOW_HOMO;
  else
    low_homo = _low_homo;
  if (!(_det_sz > 0 && _det_sz <= 360))
    det_sz = LR_DET_SZ;
  else
    det_sz = _det_sz;
  if (!(_dr > 0 && _dr < 1))
    dr = LR_DR;
  else
    dr = _dr;
  blur_det = _blur_det;
  if (_blv > 0)
    blv = _blv;
  else
    blv = LR_BLV;
  if (max_ft < 0)
    f2d = xfeatures2d::SIFT::create();
  else
    f2d = xfeatures2d::SIFT::create(max_ft);
  margin = LR_MARGIN;
  is_init = true;
}

int LogoRecog::loadLogos(const std::vector<std::string> &urls, const bool clear)
{
  if (clear)
    logos.clear();
  logos.reserve(logos.size() + urls.size());
  for (int i = 0; i < urls.size(); i++)
  {
    ifstream INF(urls[i].c_str());
    OneLogo logo;
    INF >> logo.minSup;
    int t;
    INF >> t;
    int n;
    INF >> n;
    logo.kp.reserve(n);
    for (int j = 0; j < n; j++)
    {
      float x,y,sz,ang,res;
      INF >> x>>y>>sz>>ang>>res;
      logo.kp.push_back(KeyPoint(x,y,sz,ang,res));
    }
    logo.des = Mat(n,128,CV_32F);
    for (int j = 0; j < n; j++)
    {
      for (int k = 0; k < 128; k++)
      {
        INF >> logo.des.at<float>(j,k);
      }
    }
    if (t)
    {
      logo.inv = true;
      INF >> n;
      logo.kpi.reserve(n);
      for (int j = 0; j < n; j++)
      {
        float x,y,sz,ang,res;
        INF >> x>>y>>sz>>ang>>res;
        logo.kpi.push_back(KeyPoint(x,y,sz,ang,res));
      }
      logo.desi = Mat(n,128,CV_32F);
      for (int j = 0; j < n; j++)
      {
        for (int k = 0; k < 128; k++)
        {
          INF >> logo.desi.at<float>(j,k);
        }
      }
    }
    else
    {
      logo.inv = false;
    }
    logos.push_back(logo);
  }
  return logos.size();
}

int LogoRecog::match(const std::vector<cv::KeyPoint> &kp1, const std::vector<cv::KeyPoint> &kp2, const cv::Mat &des1, const cv::Mat &des2, int thr)
{
  if (!is_init)
  {
    cout<<"need init before use"<<endl;
    return -2;
  }
  vector<vector<DMatch> > matches;
  if (thr < 0)
  {
    bmatcher.knnMatch(des1,des2,matches,2);
  }
  else
    matcher.knnMatch(des1,des2,matches,2);
  unordered_set<int> ps1,ps2;
  vector<DMatch> good;
  for (int i = 0; i < matches.size(); i++)
  {
    const DMatch &m = matches[i][0];
    const DMatch &n = matches[i][1];
    if (m.distance < dr * n.distance)
      if ((ps1.find(m.trainIdx) == ps1.end()) &&  (ps2.find(m.queryIdx) == ps2.end()))
      {
        ps1.insert(m.trainIdx);
        ps2.insert(m.queryIdx);
        good.push_back(m);
      }
  }
  int ll = good.size();
  int hsup = thr;
  if (thr < 0)
  {
    thr = low_sup;
    hsup = low_homo;
  }
  if (ll >= thr)
  {
    std::vector<Point2f> obj;
    std::vector<Point2f> scene;
    for( int i = 0; i < good.size(); i++ )
    {
      //-- Get the keypoints from the good matches
      obj.push_back( kp1[ good[i].queryIdx ].pt );
      scene.push_back( kp2[ good[i].trainIdx ].pt );
    }
    std::vector<unsigned char> mask;
    Mat H = findHomography( obj, scene, RANSAC,5.0, mask );
    ll = countNonZero(mask);
    if (ll >= hsup)
      return ll;
    else
      return -1;
  }
  else
  {
    return -1;
  }
}

int LogoRecog::recognize(const cv::Mat &im)
{
  if (!is_init)
  {
    cout<<"need init before use"<<endl;
    return -3;
  }
  if (im.empty())
  {
    cout<<"image mat empty"<<endl;
    return -4;
  }    
  Mat ndes;
  std::vector<cv::KeyPoint> nkp;
  Mat img,imr;
  if (im.channels() > 1)
    cvtColor(im,img,COLOR_BGR2GRAY);
  else
    img = im.clone();
  Rect r;
  if (img.rows > img.cols)
  {
    if (img.rows != 960 || img.cols != 720)
      resize(img,imr,Size(720,960),0,0,INTER_AREA);
    else
      imr = img;
    r = Rect(360-240,480-240,480,480);
  }
  else
  {
    if (img.rows != 720 || img.cols != 960)
      resize(img,imr,Size(960,720),0,0,INTER_AREA);
    else
      imr = img;
    r = Rect(480-240,360-240,480,480);
  }
  Mat imc = imr(r);
  Mat imf;
  resize(imc,imf,Size(det_sz,det_sz),0,0,INTER_AREA);
  if (blur_det)
  {
    Mat lap;
    Laplacian(imf,lap,CV_32F);
    cv::Scalar mean,stddev;
    meanStdDev(lap,mean,stddev);
    float var = stddev.val[0] * stddev.val[0];
    if (var < blv)
      return -2;
  }
  f2d -> detectAndCompute(imf,cv::noArray(),nkp,ndes);
  int goodM = -1;
  float goodS = -1.0f;
  for (int i = 0; i < logos.size(); i++)
  {
    int l = match(logos[i].kp,nkp,logos[i].des,ndes,logos[i].minSup);
    int l1 = -1;
    if (logos[i].inv && l < 0)
    {
      l1 = match(logos[i].kpi,nkp,logos[i].desi,ndes,logos[i].minSup);
    }
    int ll = max(l,l1);
    if (ll > 0)
    {
      float bb = float(ll) / logos[i].minSup;
      if (bb > goodS)
      {
        goodS = bb;
        goodM = i;
      }
    }
  }
  if (goodS > 0)
    return goodM;
  return -1;
}


