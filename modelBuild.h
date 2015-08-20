
#ifndef modelBuild
#define modelBuild

#include <boost/filesystem.hpp>
using namespace boost::filesystem;


cv::Mat reshapeCol1(cv::Mat in);
void segmentImg1(std::vector<cv::Mat>& out, cv::Mat in, int cropsize);
void textonFind1(cv::Mat& clus, cv::Mat dictionary);
void vecToArr1(std::vector<float> v, float* m);
void modelBuildHandle(int cropsize);

#endif
