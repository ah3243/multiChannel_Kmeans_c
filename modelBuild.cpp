#include "opencv2/highgui/highgui.hpp" // Needed for HistCalc
#include "opencv2/imgproc/imgproc.hpp" // Needed for HistCalc
#include <opencv2/opencv.hpp>
#include "opencv2/core/core.hpp"
#include <iostream> // General io
#include <stdio.h> // General io
#include <stdlib.h> // rand
#include <fstream>
#include <string>
#include <unistd.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <algorithm> // Maybe fix DescriptorExtractor doesn't have a member 'create'
#include <boost/filesystem.hpp>
#include <assert.h>
#include <time.h> // randTimeSeed
#include <map>

#include "filterbank.h" // Filterbank Handling Functions
#include "imgCollection.h" // Img Import Functions
#include "imgFunctions.h" // img cropping and handling functions

using namespace boost::filesystem;
using namespace cv;
using namespace std;

#define ERR(msg) printf("\n\nERROR!: %s Line %d\nExiting.\n\n", msg, __LINE__);

int modelSerialNum(){
  int serial;
  srand(time(NULL));
  serial = rand();
  cout << "This is the randomNumber" << serial << endl;
  return serial;
}

void modelBuildHandle(int cropsize, int scale, int numClusters, int flags, int attempts, int kmeansIteration, double kmeansEpsilon){
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
    path p = "../../../TEST_IMAGES/CapturedImgs/classes";
    loadClassImgs(p, classImgs, scale);
    float bins[m.size()];
    vecToArr(m, bins);

    vector<vector<Mat> > filterbank;
    int n_sigmas, n_orientations;
    createFilterbank(filterbank, n_sigmas, n_orientations);

    // Initilse Histogram parameters
    int histSize = m.size()-1;
   const float* histRange = {bins};
    // int histSize = 255;
    // float testRange[] = {0,255};
    // const float* histRange = {testRange};
    bool uniform = false;
    bool accumulate = false;

    TermCriteria clsTc(TermCriteria::MAX_ITER, kmeansIteration, kmeansEpsilon);
    BOWKMeansTrainer classTrainer(numClusters, clsTc, attempts, flags);

    cout << "\n\n.......Generating Models...... \n" ;

    map<string, vector<Mat> > classHist;

  // Cycle through Classes
  for(auto const ent1 : classImgs){
    // Cycle through each classes images
    cout << "\nClass: " << ent1.first << endl;
    for(int j=0;j < ent1.second.size();j++){
      Mat in, hold;
      cout << "Cycle ent1.second.size(): " << ent1.second[j].size() << " J: " << j << endl;

      // Send img to be filtered, and responses aggregated with addWeighted
      in = ent1.second[j];
       if(!in.empty())
          filterHandle(in, hold, filterbank, n_sigmas, n_orientations);

      // Segment and flatten the image then push each single column Mat onto a vector
      vector<Mat> test;
      segmentImg(test, hold, cropsize);

      // Push each saved Mat to classTrainer
      for(int k = 0; k < test.size(); k++){
        if(!test[k].empty()){
          classTrainer.add(test[k]);
        }
      }
      // Generate the given number of clusters per Image and store in Mat
      Mat clus = Mat::zeros(numClusters,1, CV_32FC1);
      clus = classTrainer.cluster();

      // Replace Cluster Centers with the closest matching texton
      textonFind(clus, dictionary);
      Mat out;
      cout << "\n\nhistRange before : " << histRange << endl;
      calcHist(&clus, 1, 0, Mat(), out, 1, &histSize, &histRange, uniform, accumulate);
      cout << "histRange after  : " << histRange << endl;
      classHist[ent1.first].push_back(out);
      classTrainer.clear();
    }
  }
  int serial = modelSerialNum();
  stringstream fileName;
//  fileName << serial << "_";
  fileName<< "models.xml";
  FileStorage fs2(fileName.str(),FileStorage::WRITE);
  int clsHist =  classHist.size();
  //Save to file
  cout << "Saving Dictionary.." << endl;
  fs2 << "Serial" << serial;
  fs2 << "modelsInfo" << "{";
    fs2 << "Num_Models" << clsHist;
    fs2 << "cropSize" << cropsize;
    fs2 << "modelsNumOfClusters" << numClusters;
    fs2 << "modelsFlagType" << flags;
    fs2 << "modelsAttempts" << attempts;
    fs2 << "Kmeans" << "{";
      fs2 << "Iterations" << kmeansIteration;
      fs2 << "Epsilon" << kmeansEpsilon;
    fs2 << "}";
  fs2 << "}";

  fs2 << "textonDictInfo" << "{";
    fs2 << "totalDictSize" << dictionary.size();
    fs2 << "vocabulary" << dictionary;
    fs2 << "bins" << m;
  fs2 << "}";

  fs2 << "classes" << "{";
  int cont=0;
  for(auto const ent1 : classHist){
    stringstream ss1;
    ss1 << "class_" << cont;
    fs2 << ss1.str() << "{";
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
  fs2 << "}";
  fs2.release();
}
