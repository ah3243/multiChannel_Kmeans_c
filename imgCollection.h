// Image collection header File

#ifndef imgCollection
#define imgCollection

#include <boost/filesystem.hpp>
using namespace boost::filesystem;

void errorFunc(std::string input);
void warnFunc(std::string input);
void menuPrint();
void extractClsNme(std::string &nme);
void extractFullNme(std::string &nme);
int getSuffix(std::string p);
int getHighestSuffix(boost::filesystem::path p, std::string cls);
void getUniqueClassNme(boost::filesystem::path p, std::vector<std::string>& classes);
void printClasses(std::vector<std::string> s);
void getTexImgs(const char *inPath, std::vector<cv::Mat>& textDict);
void generateDirs();
void  cropImage(cv::Mat img, cv::Mat& out);
void saveImage(std::string path, std::string cls , int num, std::vector<cv::Mat>& img);
void clearDir(std::string a);
int clearClass(std::string cls);
void printgetImageMenu();
void getImages(std::vector<cv::Mat>& matArr);
void retnFileNmes(boost::filesystem::path p, std::string name, std::map<std::string, std::vector<std::string> >& matches);
void scaleImg(cv::Mat in, cv::Mat &out, int scale);
void loadClassImgs(boost::filesystem::path p, std::map<std::string, std::vector<cv::Mat> > &classImgs, int scale);
void printImgDis(std::map<std::string ,std::vector<std::string> > s);
void printNovelImgMenu();
void novelImgHandler();
void printTextonMenu();
void textonHandler();
void printClassMenu();
void classHandler();
void imgCollectionHandle();

#endif
