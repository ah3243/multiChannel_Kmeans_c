//Goals//
// 1. import and save 200x200pixel image in Mat
// 2. Segment image into 20 10x10pixel images and store in Mat's
// 3. apply basic gaussian blur
// 4. apply kmeans to blured Segments, store in vector
// 5. repeat 1-4 with image from second class

#include "opencv2/highgui/highgui.hpp"
#include <opencv2/opencv.hpp>
#include "opencv2/core/core.hpp"
//#include "features2d.hpp" // For feature2d (typedef DescriptorExtractor)
#include <iostream>

#include <stdio.h>
#include <stdlib.h>

#include <fstream>
#include <string>

#include <dirent.h>
#include <unistd.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <algorithm> // Maybe fix DescriptorExtractor doesn't have a member 'create'


using std::vector;
using namespace cv;
using namespace std;


void listDir(const char *inPath, vector<Mat>& dirFiles){
  DIR *pdir = NULL;
  cout << "inpath : " << inPath << endl;
  pdir = opendir(inPath);
  // Check that dir was initialised correctly
  if(pdir == NULL){
    cout << "ERROR! unable to open directory, exiting." << endl;
    exit(1);
  }
  struct dirent *pent = NULL;

  // Continue as long as there are still values in the dir list
  while (pent = readdir(pdir)){
    if(pdir==NULL){
      cout << "ERROR! dir was not initialised correctly, exiting." << endl;
      exit(3);
    }
    stringstream ss;
    ss << inPath;
    ss << pent->d_name;
    string a = ss.str();
    Mat tmp = imread(a, CV_BGR2GRAY);
    dirFiles.push_back(tmp);
    if(!tmp.data){
      cout << "unable to read image.." << endl;
    }
  }
  closedir(pdir);
  cout << "finished Reading Successfully.." << endl;
}

int main( int /*argc*/, char** /*argv*/ )
{
  string basePath = "../../../TEST_IMAGES/kth-tips/";
  vector<string> extPath = {"bread/", "cotton/", "cork/"};
  typedef vector<Mat> mV;
  mV txtons(100, Mat(200,200, CV_32FC1));
  mV bread (10,Mat(200,200,CV_32FC1));
  mV cotton (10, Mat(200,200,CV_32FC1));
  mV cork (10, Mat(200,200,CV_32FC1));

  int count = 0;
  for(int i=0;i<4;i++){
    stringstream ss;
    ss << basePath;
    ss << extPath[count];
    ss << "train/";
    string b = ss.str();
    const char* a = b.c_str();
    cout << "this is the path.. " << a << endl;

    switch(i){
      case 0:
      cout << "number: " << i << endl;
      listDir("../../../TEST_IMAGES/kth-tips/textons/",txtons);
      count--;
      break;

      case 1:
      cout << "number: " << i << endl;
      listDir(a,bread);
      break;

      case 2:
      cout << "number: " << i << endl;
      listDir(a,cotton);
      break;

      case 3:
      cout << "number: " << i << endl;
      listDir(a,cork);
      break;
    }
    count ++;
  }
  cout << "This is the size of textons: " << txtons.size() << ", bread: " << bread.size() << ", cotton: " << cotton.size() << ", cork: " << cork.size() << endl;


  Mat in = imread("../../bread.png", CV_BGR2GRAY);
  Mat points(in.rows*in.cols, 1,CV_32F);

  int cnt = 0;
  for(int i =0;i<in.rows;i++){
    for(int j=0;j<in.cols;j++){
      points.at<float>(cnt,0) = in.at<Vec3b>(i,j)[0];
      cnt++;
    }
  }

  int dictSize = 100;
  int attempts = 1;
  int flags = KMEANS_PP_CENTERS;
  TermCriteria tc(TermCriteria::MAX_ITER + TermCriteria::EPS, 10, 0.001);


  BOWKMeansTrainer bowTrainer(dictSize, tc, attempts, flags);

  bowTrainer.add(points);
  bowTrainer.add(points);
  bowTrainer.add(points);
//  cout << "this is the size of the local Feature set: " << bowTrainer.descriptorsCount() << endl;
  Mat dictionary = bowTrainer.cluster();
  cout << "This is the dictionary size: " << dictionary.size() << endl;


  // Create Models/Train Classifier
  vector<KeyPoint> keypoints;
  Mat response_hist;
  Mat img;
  string filepath;
  map<string,Mat> classes_training_data;

  Ptr<FeatureDetector> detector = FeatureDetector::create("SURF");
  Ptr<DescriptorExtractor> discriptor = DescriptorExtractor::create("SIFT");
  Ptr<cv::DescriptorMatcher> matcher = DescriptorMatcher::create("FlannBased");
  BOWImgDescriptorExtractor bowDE(discriptor, matcher);

  bowDE.setVocabulary(dictionary);

  img = imread("../../bread.png", CV_BGR2GRAY);
  //detector.detect(img, keypoints);
  bowDE.compute(img, keypoints, response_hist);



  return 0;
}

//  cout << "\n Usage in C++ API:\n double kmeans(const Mat& samples, int clusterCount, Mat& labels, TermCriteria termcrit, int attempts, int flags, Mat* centers) \n\n\n" << endl;
//
//  int clusterCount = 2;
//  int dimensions = 2;
//  int sampleCount = 20*20;
//
//  Mat points(sampleCount,dimensions, CV_32FC1,Scalar(10));
//  Mat in =imread("../../10pixsq.png", CV_BGR2GRAY);
//  Mat labels;
//  Mat centers(clusterCount, 1, points.type());
//
//
// // // values of 1st half of data set is set to 10
// // // change the values of 2nd half of the data set; i.e. set it to 20
//
// int cnt = 0; // To make sure img vals are stored from 0
//  for(int i =0;i<in.rows;i++)
//  {
//   for(int j=0;j<in.cols;j++)
//   {
//       points.at<float>(cnt,0) = in.at<Vec3b>(i,j)[0];
// //      points.at<float>(cnt,1) = in.at<Vec3b>(i,j)[0];
//       points.at<float>(cnt,1)=rand()%100;
//     cnt++;
//   }
//  }
//
//
// //Mat out = points.reshape(1, points.rows*points.cols);
//
// kmeans(points, clusterCount, labels, TermCriteria( CV_TERMCRIT_EPS+CV_TERMCRIT_ITER, 10, 1.0), 3, KMEANS_PP_CENTERS, centers);
//     // we can print the matrix directly.
//  cout<<"Data: \n"<<points<<endl;
//  cout << "Size: " << points.size() << endl;
//  cout << "This has .channels(): " << points.channels() << endl;
// cout<<"Center: \n"<<centers<<endl;
// //cout<<"Labels: \n"<<labels<<endl;
//  return 0;
// }

/// ---------------------------------------------------------------------------------------------------- ///

// #include "opencv2/imgproc/imgproc.hpp"
// #include "opencv2/highgui/highgui.hpp"
// #include <iostream>
// #include <boost/math/special_functions/round.hpp>
// #include <boost/filesystem.hpp>
// #include <boost/range/algorithm.hpp>
//
// using namespace boost::filesystem;
// using namespace cv;
// using namespace std;
//
// void printCluster(Mat centers){
//     int r = centers.rows;
//     int c = centers.cols;
//
//     cout << "This is the cluster size inside printing: " << centers.size() << endl;
//     for(int i=0;i<c;i++){
//       cout << "This is the col: " << i << endl;
//       for(int j=0;j<r;j++){
//         cout << centers.at<float>(i+j*c) << ", i:" << j << endl;
//       }
//       cout << "Looping.. " << endl;
//     }
//     cout << "Leaving.. " << endl;
// }
//
// void fillImg(Mat in, Mat& out, int channel){
//   int r = in.rows;
//   int c = in.cols;
//
//   for(int i=0;i<c;i++){
//     for(int j=0;j<r;j++){
//       out.at<float>((j+i*c),channel) = in.at<float>(i,j);
// //      cout << "value: " << (j+i*c) << endl;
//     }
//   }
// }
//
// void segmentImg(Mat in, Mat& segImg){
//   int chan = 0;
//   int rmax =8;
//   int cmax =1;
//
//   for(int c=0; c < cmax; c++){
//     for(int r=0 ;r < rmax; r++){
//       // Select ROI and copy to tmp Matrix
//       Mat tmp;
//       in(Rect(r,c,20,20)).copyTo(tmp);
//
//       fillImg(tmp,segImg,chan);
//       cout << "Val: " << segImg.at<Vec3f>(r,c)[0]  << " chan: " << chan<< endl;
//       chan++;
//     }
//   }
// }
//
// int main( int argc, char** argv )
// {
//   string basePath = "../../../TEST_IMAGES/kth-tips/bread/train/";
//   path p(basePath);
// //    getImages(p);
//   stringstream ss;
//   ss << basePath;
//   ss << "52a-scale_2_im_1_col.png";
//   Mat input1 = imread(ss.str(), CV_BGR2GRAY);
//   cout << "This is the img input size: " << input1.size() << endl;

//  int segSide = 20;
//  Mat segImgArr = Mat::zeros(segSide*segSide, 8, CV_32FC2);
//  segmentImg(input1, segImgArr);
  // cout << "This is the size of the Mat going in: " << segImgArr.size() << " rows: " << segImgArr.rows << " cols: " << segImgArr.cols << endl;
  //
  // Mat labels(1000,1000, CV_8U);
  // Mat centers;
  // int attempts = 5, clustercnt = 2;
  // kmeans(segImgArr, clustercnt, labels, TermCriteria(CV_TERMCRIT_ITER|CV_TERMCRIT_EPS, 10000, 0.0001), attempts, KMEANS_PP_CENTERS, centers);
  //
  //   cout << "This is the size of the labels: " << labels.size() << " Rows: " << labels.rows << " Cols: " << labels.cols << endl;
  //   cout << "labels: " << labels << endl;
  //   cout << "CEntroids: " << centers << endl;

/// -------------------------------------------------------------------------------------------------------- ///
  // Mat src = imread( argv[1], 1 );
  // Mat samples(src.rows * src.cols, 1, CV_32FC(1));
  // for( int y = 0; y < src.rows; y++ )
  //   for( int x = 0; x < src.cols; x++ )
  //     for( int z = 0; z < 3; z++)
  //       samples.at<Vec3f>(y + x*src.rows, 1)[0] = src.at<Vec3b>(y,x)[z]; // Multichannel array going into a single channel Mat
  //
  // int clusterCount = 5;
  // Mat labels;
  // int attempts = 5;
  // Mat centers;
  // kmeans(samples, clusterCount, labels, TermCriteria(CV_TERMCRIT_ITER|CV_TERMCRIT_EPS, 10000, 0.0001), attempts, KMEANS_PP_CENTERS, centers );
  //
  // cout << "centers size: " << centers.size() << endl;
  // cout << "centers: " << endl;
  // cout << centers ;
  // cout << endl;
  //
  //  cout << "labels size: " << labels.size() << endl;
  // waitKey( 0 );
//}
