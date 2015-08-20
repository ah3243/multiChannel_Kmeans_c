#include "opencv2/highgui/highgui.hpp" // Needed for HistCalc
#include "opencv2/imgproc/imgproc.hpp" // Needed for HistCalc
#include <opencv2/opencv.hpp>
#include "opencv2/core/core.hpp"
#include <opencv2/nonfree/features2d.hpp>
#include <opencv2/legacy/legacy.hpp>
#include <iostream> // General io
#include <stdio.h> // General io
#include <stdlib.h>
#include <fstream>
#include <string>
#include <unistd.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <algorithm> // Maybe fix DescriptorExtractor doesn't have a member 'create'
#include <boost/filesystem.hpp>


using namespace std;
using namespace cv;

#define ERR(msg) printf("\n\nERROR!: %s Line %d\nExiting.\n\n", msg, __LINE__);

Mat reshapeCol(Mat in){
  Mat points(in.rows*in.cols, 1,CV_32F);
  int cnt = 0;
  for(int i =0;i<in.rows;i++){
    for(int j=0;j<in.cols;j++){
      points.at<float>(cnt, 0) = in.at<Vec3b>(i,j)[0];
      cnt++;
    }
  }
  return points;
}

void segmentImg(vector<Mat>& out, Mat in, int cropsize){
  int size = 200;
  if(in.rows!=200 || in.cols!=200){
    cout << "The input image was not 200x200 pixels.\nExiting.\n";
    exit(-1);
  }
  for(int i=0;i<size;i+=cropsize){
    for(int j=0;j<size;j+=cropsize){
     Mat tmp = Mat::zeros(cropsize,cropsize,CV_32FC1);
     tmp = reshapeCol(in(Rect(i, j, cropsize, cropsize)));
     out.push_back(tmp);
    }
  }
  cout << "This is the size: " << out.size() << " and the average cols: " << out[0].rows << endl;
}

void textonFind(Mat& clus, Mat dictionary){
  if(clus.empty() || dictionary.empty()){
    ERR("Texton Find inputs were empty");
    exit(-1);
  }
  // Loop through input centers
  for(int h=0;h<clus.rows;h++){
    float distance = 0.0, nearest = 0.0;

    distance = abs(dictionary.at<float>(0,0) - clus.at<float>(h,0));
    nearest = dictionary.at<float>(0,0);

    // Compare current centre with all values in texton dictionary
    for(int k = 0; k < dictionary.rows; k++){
      if(abs(dictionary.at<float>(k,0) - clus.at<float>(h,0)) < distance){
        nearest = dictionary.at<float>(k,0);
        distance = abs(dictionary.at<float>(k,0) - clus.at<float>(h,0));
      }
    }
    // Replace input Center with closest Texton Center
    clus.at<float>(h,0) = nearest;
  }
}

void vecToArr(vector<float> v, float* m){
  int size = v.size();
  for(int i=0;i<size;i++){
    m[i] = v[i];
  }
}
