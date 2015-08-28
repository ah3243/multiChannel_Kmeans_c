//Novel Image Testing Header

#ifndef novelImgTest
#define novelImgTest

void addTrueNegatives(std::string exp1, std::string exp2, std::map<std::string, std::vector<double> >& res);
void initROCcnt(std::vector<std::map<std::string, std::vector<double> > >& r, std::vector<std::string> clsNames);
int getClsNames(std::map<std::string, std::vector<double> > &r, std::vector<std::string> &nme);
void cacheTestdata(std::string correct, std::string prediction, std::map<std::string, std::vector<double> >& results);
void organiseResultByClass(std::vector<std::map<std::string, std::vector<double> > >in, std::map<std::string, std::vector<std::vector<double> > > &out, std::vector<std::string> clsNmes);
void calcROCVals(std::map<std::string, std::vector<std::vector<double> > > in, std::map<std::string, std::vector<std::vector<double> > >& out, std::vector<std::string> clssNmes);
void saveTestData(std::vector<std::map<std::string, std::vector<double> > > r, int serial);
int getClassHist(std::map<std::string, std::vector<cv::Mat> >& savedClassHist);
void getDictionary(cv::Mat &dictionary, std::vector<float> &m);
void testNovelImg(int clsAttempts, int numClusters, std::map<std::string, std::vector<double> >& results, std::map<std::string, std::vector<cv::Mat> > classImgs, std::map<std::string, std::vector<cv::Mat> > savedClassHist, std::map<std::string, cv::Scalar> Colors, int cropsize);
void printRAWResults(std::map<std::string, std::vector<double> > r);
void loadVideo(boost::filesystem::path p, std::map<std::string, std::vector<cv::Mat> > &classImages, int scale);
void novelImgHandle(boost::filesystem::path testPath,boost::filesystem::path clsPath, int scale, int cropsize, int numClusters, int DictSize);


#endif
