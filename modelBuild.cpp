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
#include <assert.h>

#include "filterbank.h" // Filterbank Handling Functions
#include "imgCollection.h" // Img Handling Functions


using namespace boost::filesystem;
using namespace cv;
using namespace std;

#define cropsize  200

#define ERR(msg) printf("\n\nERROR!: %s Line %d\nExiting.\n\n", msg, __LINE__);

Mat reshapeCol1(Mat in){
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

void segmentImg1(vector<Mat>& out, Mat in){
  int size = 200;
  if(in.rows!=200 || in.cols!=200){
    cout << "The input image was not 200x200 pixels.\nExiting.\n";
    exit(-1);
  }
  for(int i=0;i<size;i+=cropsize){
    for(int j=0;j<size;j+=cropsize){
     Mat tmp = Mat::zeros(cropsize,cropsize,CV_32FC1);
     tmp = reshapeCol1(in(Rect(i, j, cropsize, cropsize)));
     out.push_back(tmp);
    }
  }
  cout << "This is the size: " << out.size() << " and the average cols: " << out[0].rows << endl;
}

void textonFind1(Mat& clus, Mat dictionary){
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

void vecToArr1(vector<float> v, float* m){
  int size = v.size();
  for(int i=0;i<size;i++){
    m[i] = v[i];
  }
}

void modelBuildHandle(){
  // Load TextonDictionary
  Mat dictionary;
  vector<float> m;
    FileStorage fs("dictionary.xml",FileStorage::READ);
    fs["vocabulary"] >> dictionary;
    fs["bins"] >> m;
    if(!fs.isOpened()){
      ERR("Unable to open Texton Dictionary.");
      exit(-1);
    }
    fs.release();

    // Load Class imgs and store in classImgs map
    map<string, vector<Mat> > classImgs;
    path p = "../../../TEST_IMAGES/kth-tips/classes";
    loadClassImgs(p, classImgs);

    float bins[m.size()];
    vecToArr1(m, bins);

    // Initilse Histogram parameters
    int histSize = m.size()-1;
    const float* histRange = {bins};
    bool uniform = false;
    bool accumulate = false;


    int clsNumClusters = 10;
    int clsAttempts = 5;
    int clsFlags = KMEANS_PP_CENTERS;
    TermCriteria clsTc(TermCriteria::MAX_ITER + TermCriteria::EPS, 1000, 0.0001);
    BOWKMeansTrainer classTrainer(clsNumClusters, clsTc, clsAttempts, clsFlags);

    cout << "\n\n.......Generating Models...... \n" ;

    map<string, vector<Mat> > classHist;

  // Cycle through Classes
  for(auto const ent1 : classImgs){
    // Cycle through each classes images
    cout << "\nClass: " << ent1.first << endl;
    for(int j=0;j < ent1.second.size();j++){
      Mat in, hold;

      // Send img to be filtered, and responses aggregated with addWeighted
      in = ent1.second[j];
       if(!in.empty())
          filterHandle(in, hold);

      // Segment the 200x200pixel image into 400x1 Mats(20x20)
      vector<Mat> test;
      segmentImg1(test, hold);

      // Push each saved Mat to classTrainer
      for(int k = 0; k < test.size(); k++){
        if(!test[k].empty()){
          classTrainer.add(test[k]);
        }
      }
    }
    // Generate 10 clusters per class and store in Mat
    Mat clus = Mat::zeros(clsNumClusters,1, CV_32FC1);
    clus = classTrainer.cluster();

    // Replace Cluster Centers with the closest matching texton
    textonFind1(clus, dictionary);

    Mat out;
    calcHist(&clus, 1, 0, Mat(), out, 1, &histSize, &histRange, uniform, accumulate);
    classHist[ent1.first].push_back(out);

    classTrainer.clear();
  }

  FileStorage fs2("models.xml",FileStorage::WRITE);
  // int numClasses = classHist.size();
  // fs2 << "Num_Models" << numClasses;
  int cont=0;
  for(auto const ent1 : classHist){
    stringstream ss;
    ss << "class_" << cont;
    fs2 << ss.str() << "{";
      fs2 << "Name" << ent1.first;
      fs2 << "Models" << "{";
        for(int i=0;i<ent1.second.size();i++){
          stringstream ss1;
          ss1 << "m_" << i;
          fs2 << ss1.str() << ent1.second[i];
        }
      fs2 << "}";
    fs2 << "}";
    cont++;
  }
  fs2.release();
}
