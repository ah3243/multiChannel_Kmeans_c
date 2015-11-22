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

#define imgFuncDEBUG 0
#define SHOWSEGMENTS 0 // Display source image and cropped segments

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

int createDir(string pathNme){
  const char* path = pathNme.c_str();
  boost::filesystem::path dir(path);
  if(boost::filesystem::create_directory(dir)){
    cerr << "Directory Created: " << pathNme << endl;
    return 1;
  }else{
    return 1;
  }
  return 0;
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

void getDictionary(Mat &dictionary, vector<float> &m){
  // Load TextonDictionary
  FileStorage fs("dictionary.xml",FileStorage::READ);
  if(!fs.isOpened()){
    ERR("Unable to open Texton Dictionary.");
    exit(-1);
  }
  fs["vocabulary"] >> dictionary;
  fs["bins"] >> m;
  fs.release();
}

int getClassHist(map<string, vector<Mat> >& savedClassHist){
  int serial;
  // Load in Class Histograms(Models)
  FileStorage fs3("models.xml", FileStorage::READ);
  fs3["Serial"] >> serial;
  FileNode fn = fs3["classes"];
  if(fn.type() == FileNode::MAP){

    // Create iterator to go through all the classes
    for(FileNodeIterator it = fn.begin();it != fn.end();it++){
      string clsNme = (string)(*it)["Name"];
      savedClassHist[clsNme];

      // Create node of current Class
      FileNode clss = (*it)["Models"];
      // Iterate through each model inside class, saving to map
      for(FileNodeIterator it1  = clss.begin();it1 != clss.end();it1++){
        FileNode k = *it1;
        Mat tmp;
        k >> tmp;
        savedClassHist[clsNme].push_back(tmp);
      }
    }
    fs3.release();
  }else{
    ERR("Class file was not map.");
    exit(-1);
  }
  return serial;
}

// Segment input image and return in vector
void segmentImg(vector<Mat>& out, Mat in, int cropsize, int overlap, int MISSTOPLEFT_RIGHT){

  // Calculate the maximum number of segments for the given img+cropsize
  int NumColSegments = (in.cols/cropsize);
  int NumRowSegments = (in.rows/cropsize);
  int NumSegmentsTotal = (NumColSegments*NumRowSegments);

  if(NumSegmentsTotal!=6){
    MISSTOPLEFT_RIGHT=0;
  }

  // Calculate gap around the combined segments
  int colspace = (in.cols -(NumColSegments*cropsize))/2;
  int rowspace = (in.rows -(NumRowSegments*cropsize))/2;

  // Make sure manual offset and cropsize are compatible with imagesize
  if((cropsize+rowspace)>in.rows || (cropsize+colspace)>in.cols){
    ERR("cropsize larger than input image. Exiting");
    exit(1);
  }

  stringstream ss;
  ss << "entering segmentImg this is the img rows: ";
  ss << in.rows << " cols: " << in.cols;
  imgFDEBUG(ss.str(), 0);

  if(SHOWSEGMENTS){
    imshow("Whole Image", in);
  }

  int segmentCounter = 0; // Track current segment number(position)

  // Extract the maximum Number of full Segments from the image
  for(int i=0;i<(in.cols-cropsize);i+=(cropsize-overlap)){
    for(int j=rowspace;j<(in.rows-cropsize);j+=cropsize){

      // If MISSTOPLEFT_RIGHT flag == true and segment is top left or top right continue
      if(MISSTOPLEFT_RIGHT==1 && (segmentCounter==0 || segmentCounter==4) && NumSegmentsTotal==6){
        segmentCounter++; // Iterate segment counter
        continue;
      }

      Mat tmp = Mat::zeros(cropsize,cropsize,CV_32FC1);
      Mat normImg = in(Rect(i, j, cropsize, cropsize));
      if(SHOWSEGMENTS){
       imshow("SegImg", normImg);
       cout << "Press any key to change segment." << endl;
       waitKey(0);
      }
      tmp = reshapeCol(normImg);
      out.push_back(tmp);
      segmentCounter++; // Iterate segment counter
    }

  }
  ss.str("");
  ss << "This is the number of segments: " << out.size() << " and the average cols: " << out[0].cols;
  imgFDEBUG(ss.str(), 0);
}

void textonFind(Mat& clus, Mat dictionary, vector<double>& disVec){
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
