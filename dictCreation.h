
#ifndef dictCreation
#define dictCreation

void dicDEBUG(std::string msg, double in);
void binLimits(std::vector<float>& tex);
void removeDups(std::vector<float>& v);
std::vector<float> matToVec(cv::Mat m);
std::vector<float> createBins(cv::Mat texDic);
void dictCreateHandler(int cropsize, int scale, int numClusters, int flags, int attempts, int kmeansIteration, double kmeansEpsilon, int overlap);

#endif
