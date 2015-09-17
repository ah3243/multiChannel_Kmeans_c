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
#include <numeric> // For getting average texton distance

#include "filterbank.h" // Filterbank Handling Functions
#include "imgCollection.h" // Img Import Functions
#include "imgFunctions.h" // img cropping and handling functions

using namespace boost::filesystem;
using namespace cv;
using namespace std;

#define MdlDEBUG 0
#define printModels 1
#define ERR(msg) printf("\n\nERROR!: %s Line %d\nExiting.\n\n", msg, __LINE__);

int modelSerialNum(){
  int serial;
  srand(time(NULL));
  serial = rand();
  return serial;
}

void saveModels(map<string, vector<Mat> > classSave){
  FileStorage fs4("rawModels.xml",FileStorage::WRITE);
  for(auto const ent0:classSave){
    fs4 << ent0.first << "{";
    for(int k=0;k<ent0.second.size();k++){
      stringstream ss0;
      ss0 << "m_" << k;
      fs4 << ss0.str() << ent0.second[k];
    }
    fs4 << "}";
  }
  fs4.release();
}

vector<double> convMatToVec(Mat m){
  vector<double> v;
  for(int i=0;i<m.rows;i++){
    v.push_back(m.at<float>(i,0));
  }
  return v;
}

void printMdls(map<string, vector<Mat> > classSave){
  for(auto const tnt: classSave){
    cout << "\n" << tnt.first << endl;
    for(int h=0;h<tnt.second.size();h++){
      vector<double> v;
      v = convMatToVec(tnt.second[h]);
      for(int w=0;w<v.size();w++){
        cout << v[w] << ":";
      }
      cout << "\n";
    }
  }
}

void getMeanColor(vector<Mat> in, vector<vector<double> > &classColor){
  for(int p=0;p<in.size();p++){
    Scalar m;
    m = mean(in[p]);
    classColor[0].push_back(m[0]); // get Blue
    classColor[1].push_back(m[1]); // get Green
    classColor[2].push_back(m[2]); // get Red
  }
}

void quickSegment(Mat in, vector<Mat> &out, int cropsize){
  int colstart =0, rowstart=0;
  for(int i=colstart;i<(in.cols-cropsize);i+=cropsize){
    for(int j=rowstart;j<(in.rows-cropsize);j+=cropsize){
      Mat tmp = Mat::zeros(cropsize,cropsize,CV_32FC1);
      tmp = in(Rect(i, j, cropsize, cropsize));
      out.push_back(tmp);
    }
  }
}

void modelBuildHandle(int cropsize, int scale, int numClusters, int flags, int attempts, int kmeansIteration, double kmeansEpsilon, int overlap){
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
    map<string, double> avgDistance; // Store the aggregated distance to class textons
    map<string, vector<Mat> > classSave;
    map<string, vector<double> > clssColor;

  // Cycle through Classes
  for(auto const ent1 : classImgs){
    vector<double> distances; // for holding textondict to model distances
    vector<vector<double> > classColor; // for storing each classes aggregated color

    // push back blue green and red holding vectors
    vector<double> a;
    classColor.push_back(a);
    classColor.push_back(a);
    classColor.push_back(a);

    // Cycle through each classes images
    if(MdlDEBUG){
      cout << "\nClass: " << ent1.first << endl;}
    for(int j=0;j < ent1.second.size();j++){
      Mat in, hold;
      if(MdlDEBUG){
        cout << "Cycle ent1.second.size(): " << ent1.second[j].size() << " J: " << j << endl;}

      // Send img to be filtered, and responses aggregated with addWeighted
      in = ent1.second[j];

      Mat colorSeg;
      in.copyTo(colorSeg);
      vector<Mat> CSegs;
      quickSegment(colorSeg, CSegs, cropsize);
      getMeanColor(CSegs, classColor);

       if(!in.empty())
          filterHandle(in, hold, filterbank, n_sigmas, n_orientations);

      // Segment and flatten the image then push each single column Mat onto a vector
      vector<Mat> test;
      segmentImg(test, hold, cropsize, overlap);

      // Push each saved Mat to classTrainer
      for(int k = 0; k < test.size(); k++){
        if(!test[k].empty()){
          classTrainer.add(test[k]);
        }
      }
      // Generate the given number of clusters per Image and store in Mat
      Mat clus = Mat::zeros(numClusters,1, CV_32FC1);
      clus = classTrainer.cluster();

      classSave[ent1.first].push_back(clus);
      // Replace Cluster Centers with the closest matching texton
      distances.push_back(textonFind(clus, dictionary, distances)); // substitue textons for cluster centres, store agg distance

      Mat out;
      if(MdlDEBUG){
        cout << "\n\nhistRange before : " << histRange << endl;}
      calcHist(&clus, 1, 0, Mat(), out, 1, &histSize, &histRange, uniform, accumulate);
      if(MdlDEBUG){
        cout << "histRange after  : " << histRange << endl;}
      classHist[ent1.first].push_back(out);
      classTrainer.clear();
    }
    // Calculate and store the average distance from texton dictionary centers to model centers
    double tmpdist;
    for(int mm;mm<distances.size();mm++){
      tmpdist += distances[mm];
    }
    avgDistance[ent1.first]=(tmpdist/distances.size());
    cout << "This is the distance: " << ent1.first << ": " << avgDistance[ent1.first] << "\n";

    // Aggregate the segment colors and store
    double b=0, g=0, r=0;
    for(int h=0;h<classColor[0].size();h++){
      b += classColor[0][h];
      g += classColor[1][h];
      r += classColor[2][h];
    }
    // cout << "this is bgr: " << b << " : " << g << " : " << r << "\n";

    b = (b/classColor[0].size());
    g = (g/classColor[1].size());
    r = (r/classColor[2].size());

    // cout << "this is bgr after: " << b << " : " << g << " : " << r << "\n";

    clssColor[ent1.first].push_back(b);
    clssColor[ent1.first].push_back(g);
    clssColor[ent1.first].push_back(r);
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
        fs2 << "blue" << clssColor[ent1.first][0];
        fs2 << "green" << clssColor[ent1.first][1];
        fs2 << "red" << clssColor[ent1.first][2];
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
  if(printModels==1){
    printMdls(classSave);
  }
//  saveModels(classSave);
}
