
#ifndef imgFunctions
#define imgFunctions

void segmentImg(std::vector<cv::Mat>& out, cv::Mat in, int cropsize);
cv::Mat reshapeCol(cv::Mat in);
void textonFind(cv::Mat& clus, cv::Mat dictionary);
void vecToArr(std::vector<float> v, float* m);

#endif
