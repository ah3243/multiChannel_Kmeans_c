
#ifndef dictCreation
#define dictCreation

void dictCreateHandler(int cropsize);
void binLimits(std::vector<float>& tex);
void removeDups(std::vector<float>& v);
std::vector<float> matToVec(cv::Mat m);
std::vector<float> createBins(cv::Mat texDic);

#endif
