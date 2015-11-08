
#ifndef imgFunctions
#define imgFunctions

void imgFDEBUG(std::string msg, double in);
int createDir(std::string pathNme);
void segmentImg(std::vector<cv::Mat>& out, cv::Mat in, int cropsize, int overlap, int MISSTOPLEFT_RIGHT);
cv::Mat reshapeCol(cv::Mat in);
void textonFind(cv::Mat& clus, cv::Mat dictionary, std::vector<double>& disVec);
void vecToArr(std::vector<float> v, float* m);
void getDictionary(cv::Mat &dictionary, std::vector<float> &m);
int getClassHist(std::map<std::string, std::vector<cv::Mat> >& savedClassHist);

#endif
