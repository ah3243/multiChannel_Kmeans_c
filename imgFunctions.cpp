#include "opencv2/highgui/highgui.hpp" // Needed for HistCalc
#include "opencv2/imgproc/imgproc.hpp" // Needed for HistCalc
#include <opencv2/opencv.hpp>
#include "opencv2/core/core.hpp"
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

// For offsetting segment, MUST === IMGFUNCTION VALUES!!
#define COLSTART 0
#define ROWSTART 0

#define imgFuncDEBUG 0
#define ERR(msg) printf("\n\nERROR!: %s Line %d\nExiting.\n\n", msg, __LINE__);

void imgFDEBUG(string msg, double in){
  if(imgFuncDEBUG){
    cout << msg;
  if(in!=0){
    cout << in;
  }
  cout << "\n";
  }
}

Mat reshapeCol(Mat in){
  Mat points(in.rows*in.cols, 1,CV_32F);
  int cnt = 0;
//  cout << "inside. These are the rows: " <<  in.rows << " and cols: " << in.cols  << endl;
  for(int i =0;i<in.cols;i++){
//    cout << "outer loop: " << i << endl;
    for(int j=0;j<in.rows;j++){
      points.at<float>(cnt, 0) = in.at<Vec3b>(i,j)[0];
      cnt++;
    }
  }
  return points;
}

// Segment input image and return in vector
void segmentImg(vector<Mat>& out, Mat in, int cropsize, int overlap){
//  int colstart =0, rowstart=0;

  // find the number of possible segments, then calculate gap around these
  int colspace = (in.cols -((in.cols/cropsize)*cropsize))/2;
  int rowspace = (in.rows -((in.rows/cropsize)*cropsize))/2;

  if((cropsize+ROWSTART)>in.rows || (cropsize+COLSTART)>in.cols){
    ERR("cropsize larger than input image. Exiting");
    exit(1);
  }

  stringstream ss;
  ss << "entering segmentImg this is the img rows: ";
  ss << in.rows << " cols: " << in.cols;
  imgFDEBUG(ss.str(), 0);

  // // if no overlap and unable to make >1 segment, place segment in center of screen
  // if(overlap==0&&colspace>0){
  //   colstart= colspace;
  // }
  // if(overlap==0&&rowspace>0){
  //   rowstart= rowspace;
  // }

  for(int i=COLSTART;i<(in.cols-cropsize);i+=(cropsize-overlap)){
    for(int j=ROWSTART;j<(in.rows-cropsize);j+=cropsize){
      Mat tmp = Mat::zeros(cropsize,cropsize,CV_32FC1);
      tmp = reshapeCol(in(Rect(i, j, cropsize, cropsize)));
      out.push_back(tmp);
    }
  }
  // Mat out1 = Mat::zeros(in.cols, in.rows, CV_32FC1);
  // out1 = reshapeCol(in);
  // out.push_back(out1);
  ss.str("");
  ss << "This is the number of segments: " << out.size() << " and the average cols: " << out[0].cols;
  imgFDEBUG(ss.str(), 0);
}

double textonFind(Mat& clus, Mat dictionary, vector<double>& disVec){
  if(clus.empty() || dictionary.empty()){
    ERR("Texton Find inputs were empty.");
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
    disVec.push_back(distance);
  }
}

void vecToArr(vector<float> v, float* m){
  int size = v.size();
  for(int i=0;i<size;i++){
    m[i] = v[i];
  }
}
