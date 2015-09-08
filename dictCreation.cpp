
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
#include <assert.h>
#include <map>

#include "filterbank.h"
#include "imgCollection.h"
#include "imgFunctions.h"

#define DicDEBUG 0

using namespace cv;
using namespace std;

void dicDEBUG(string msg, double in){
  if(DicDEBUG){
    cout << msg;
    if(in!=0){
      cout << in;
    }
    cout << "\n";
  }
}

// Create bins for each textonDictionary Value
void binLimits(vector<float>& tex){
  dicDEBUG("inside binLimits", 0);
  vector<float> bins;
  bins.push_back(0);
  for(int i = 0;i <= tex.size()-1;i++){
      bins.push_back(tex[i] + 0.00001);
  }
  bins.push_back(256);

  for(int i=0;i<bins.size();i++)
    cout << "texDict: " << i << ": "<< tex[i] << " becomes: " << bins[i+1] << endl;
  tex.clear();
  tex = bins;
}

// Assign vector to Set to remove duplicates
void removeDups(vector<float>& v){
  dicDEBUG("inside removeDups", 0);
  sort(v.begin(), v.end());
  auto last = unique(v.begin(), v.end());
  v.erase(last, v.end());
}

vector<float> matToVec(Mat m){
  vector<float> v;
  for(int i=0;i<m.rows;i++){
    v.push_back(m.at<float>(i,0));
  }
  return v;
}

vector<float> createBins(Mat texDic){
  vector<float> v = matToVec(texDic);
  dicDEBUG("\n\nThis is the bin vector size BEFORE: ", v.size());
  binLimits(v);
  dicDEBUG("\n\nThis is the bin vector size AFTER: ", v.size());
  return v;
}

void dictCreateHandler(int cropsize, int scale, int numClusters, int flags, int attempts, int kmeansIteration, double kmeansEpsilon){
  // TermCriteria tc(TermCriteria::MAX_ITER + TermCriteria::EPS, kmeansIteration, kmeansEpsilon);
  TermCriteria tc(TermCriteria::MAX_ITER, kmeansIteration, kmeansEpsilon);
  BOWKMeansTrainer bowTrainer(numClusters, tc, attempts, flags);

  map<string, vector<Mat> > textonImgs;
  path p = "../../../TEST_IMAGES/CapturedImgs/textons";
  loadClassImgs(p, textonImgs, scale);

  vector<vector<Mat> > filterbank;
  int n_sigmas, n_orientations;
  createFilterbank(filterbank, n_sigmas, n_orientations);

  Mat dictionary;
  // Cycle through Classes
  for(auto const ent1 : textonImgs){
    // Cycle through all images in Class
    for(int j=0;j<ent1.second.size();j++){
      Mat in = Mat::zeros(ent1.second[j].cols, ent1.second[j].rows,CV_32FC1);
      Mat hold = Mat::zeros(ent1.second[j].cols, ent1.second[j].rows,CV_32FC1);
      // Send img to be filtered, and responses aggregated with addWeighted
      in = ent1.second[j];
      dicDEBUG("before of filterbank handle texton dict..\n", 0);
      if(!in.empty())
        filterHandle(in, hold, filterbank, n_sigmas, n_orientations);
      dicDEBUG("outside of filterbank handle texton dict..\n", 0);
      // Segment the 200x200pixel image
      vector<Mat> test;
      segmentImg(test, hold, cropsize);
      dicDEBUG("after segmenation: ", test.size());
      // Push each saved Mat to bowTrainer
      for(int k = 0; k < test.size(); k++){
        if(!test[k].empty()){
          bowTrainer.add(test[k]);
        }
      }
    dicDEBUG("This is the bowTrainer.size(): ", bowTrainer.descripotorsCount());
    }
    // Generate specified num of clusters per class and store in Mat
    dictionary.push_back(bowTrainer.cluster());
    bowTrainer.clear();
  }

  vector<float> bins = createBins(dictionary);

  removeDups(bins);

  //Save to file
  dicDEBUG("Saving Dictionary..", 0);
  FileStorage fs("dictionary.xml",FileStorage::WRITE);
  fs << "cropSize" << cropsize;
  fs << "clustersPerClass" << numClusters;
  fs << "totalDictSize" << dictionary.size();
  fs << "flagType" << flags;
  fs << "attempts" << attempts;
  stringstream ss;
  for(auto const ent1 : textonImgs){
    ss << ent1.first << " ";
  }
  fs << "classes" << ss.str();
  fs << "Kmeans" << "{";
    fs << "Iterations" << kmeansIteration;
    fs << "Epsilon" << kmeansEpsilon;
  fs << "}";
  fs << "vocabulary" << dictionary;
  fs << "bins" << bins;
  fs.release();
}
