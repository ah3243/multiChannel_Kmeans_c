
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

#include "filterbank.h"
#include "imgCollection.h"
#include "imgFunctions.h"

using namespace cv;
using namespace std;

// Create bins for each textonDictionary Value
void binLimits(vector<float>& tex){
  cout << "inside binLimits" << endl;

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
  cout << "inside.." << endl;
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
  cout << "\n\nThis is the bin vector size BEFORE: " << v.size() << endl;
  binLimits(v);
  cout << "\n\nThis is the bin vector size AFTER: " << v.size() << endl;
  return v;
}

void dictCreateHandler(int cropsize, int scale, int numClusters){
  int dictSize = numClusters;
  int attempts = 5;
  int flags = KMEANS_PP_CENTERS;
  int kmeansIteration = 1000;
  double kmeansEpsilon = 0.0001;
  TermCriteria tc(TermCriteria::MAX_ITER + TermCriteria::EPS, kmeansIteration, kmeansEpsilon);
  BOWKMeansTrainer bowTrainer(dictSize, tc, attempts, flags);

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
      cout << "before of filterbank handle texton dict..\n\n" << endl;
      if(!in.empty())
        filterHandle(in, hold, filterbank, n_sigmas, n_orientations);
      cout << "outside of filterbank handle texton dict..\n\n" << endl;
      // Segment the 200x200pixel image
      vector<Mat> test;
      segmentImg(test, hold, cropsize);
      cout << "after segmenation: " << test.size() << endl;
      // Push each saved Mat to bowTrainer
      for(int k = 0; k < test.size(); k++){
        if(!test[k].empty()){
          bowTrainer.add(test[k]);
        }
      }
    cout << "This is the bowTrainer.size(): " << bowTrainer.descripotorsCount() << endl;
    }
    // Generate 10 clusters per class and store in Mat
    dictionary.push_back(bowTrainer.cluster());
    bowTrainer.clear();
  }

  vector<float> bins = createBins(dictionary);

  removeDups(bins);

  //Save to file
  cout << "Saving Dictionary.." << endl;
  FileStorage fs("dictionary.xml",FileStorage::WRITE);
  fs << "cropSize" << cropsize;
  fs << "clustersPerClass" << dictSize;
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
