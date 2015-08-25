
#ifndef dictCreation
#define dictCreation

void binLimits(std::vector<float>& tex);
void removeDups(std::vector<float>& v);
std::vector<float> matToVec(cv::Mat m);
std::vector<float> createBins(cv::Mat texDic);
void dictCreateHandler(int cropsize, double scale, int numClusters);

#endif
