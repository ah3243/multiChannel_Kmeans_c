/////////////////////////////////////////////////////////////
//              SINGLE IMAGE TESTING MODULE                //
/////////////////////////////////////////////////////////////
// This module will:
// 1. take in a single image.
// 2. Remap to remove distortion.
// 3. process it (normalise, filter it).
// 4. Crop it into 6 segments.
// 5. Classifiy each segment against stored models.
// 6. determine from the output what action should be taken
//     - Move forward
//     - Turn Left
//     - Turn Right
//     - Stop (to collect second image/validate target with higher resolution)
//     - Drop Ordinance (If target Validated)
// 7. Return chosen option.
/////////////////////////////////////////////////////////////

#include <opencv2/highgui/highgui.hpp> // Needed for HistCalc
#include <opencv2/imgproc/imgproc.hpp> // Needed for HistCalc
#include <opencv2/opencv.hpp>
#include <opencv2/core/core.hpp>
#include <iostream> // General io
#include <stdio.h> // General io
#include <stdlib.h>
#include <boost/filesystem.hpp>
#include <assert.h>
#include <chrono>  // time measurement
#include <thread>  // time measurement
#include <map>

#include "filterbank.h" // Filterbank Handling Functions
#include "imgCollection.h" // Img Handling Functions
#include "imgFunctions.h" // Img Processing Functions

using namespace cv;
using namespace std;

#define CHISQU_MAX_threshold 6
#define CHISQU_DIS_threshold 0

#define showImgs 1

// Get Distortion Coefficients and Camera Matrix from file
void getCalVals(Mat &cameraMatrix, Mat &distCoeffs, double &image_Width, double &image_Height){
  FileStorage fs("../../imageRemapping/calFile.xml", FileStorage::READ); // Get config Data from other dir
  fs["Camera_Matrix"] >> cameraMatrix;
  fs["Distortion_Coefficients"] >> distCoeffs;
  fs["image_Width"] >> image_Width;
  fs["image_Height"] >> image_Height;
  fs.release();
}

// Remap input image to remove distortion in line with saved calibration values
Mat remapImg(Mat input){
  // Create variables
  Mat map1, map2;
  Mat cameraMatrix, distCoeffs;
  double image_Width, image_Height;

  getCalVals(cameraMatrix, distCoeffs, image_Width, image_Height);
  // Set size
  Size imageSize(image_Width, image_Height);

  // Create maps
  initUndistortRectifyMap(cameraMatrix, distCoeffs, Mat(),
    getOptimalNewCameraMatrix(cameraMatrix, distCoeffs, imageSize, 0, imageSize, 0),
    imageSize, CV_16SC2, map1, map2);

  // Create remap variables
  Mat view, rview;

  // Create display Windows
  namedWindow("before", 0);
  namedWindow("after", 0);

  view = input;
  // Remap image
  remap(view, rview, map1, map2, INTER_LINEAR);
  if(showImgs){
   imshow("before", view);
   imshow("after", rview);
  }
  return rview;
}

Mat clusterImg(Mat in, map<string, int> params, map<string, double> paramsDB){
  // Extract Parameters
  double kmeansEpsilon = paramsDB["kmeansEpsilon"];
  double kmeansIteration = paramsDB["kmeansIteration"];
  int numClusters = params["numClusters"];
  int clsAttempts = params["kmeansAttempts"];
  int flags = params["flags"];

  TermCriteria clsTc(TermCriteria::MAX_ITER, kmeansIteration, kmeansEpsilon);

  // Create Kmeans Trainier
  BOWKMeansTrainer novelTrainer(numClusters, clsTc, clsAttempts, flags);
  novelTrainer.add(in);
  return novelTrainer.cluster();
}

void avgIterResults(map<string, vector<double> > results){
  // Loop through each class
  for(auto const itrRes:results){
    double tmp;
    // Add all results and divide by number of results to get the mean
    for(int v=0;v<itrRes.second.size(); v++){
      tmp+=itrRes.second[v];
    }
    double avgResult = tmp/itrRes.second.size();
    cout << itrRes.first << " : " << avgResult << endl;
  }
}

int directionHandle(string imgPath, map<string, int> params, map<string, double> paramsDB){
  int scale = params["scale"];
  int cropSize = params["cropSize"];

  // Read in image
  Mat inImg = imread(imgPath,1);

  // Remap input to remove distortion
  Mat rectImg = remapImg(inImg);

  // Scale image
  Mat scaledImg;
  scaleImg(rectImg, scaledImg, scale);

  // Generate and Store Filterbank
  vector<vector<Mat> > filterbank;
  int n_sigmas, n_orientations;
  createFilterbank(filterbank, n_sigmas, n_orientations);

  // Filter input Image
  Mat filteredImg;
  filterHandle(scaledImg, filteredImg, filterbank, n_sigmas, n_orientations);
  imshow("filtered", filteredImg);

  // Crop Image
  vector<Mat> segments;
  segmentImg(segments, filteredImg, cropSize, 0);

  // Import Saved Texton Dictionary
  Mat texDict;
  vector<float> texDictbins;
  getDictionary(texDict, texDictbins);
  // Prepare Bins for histogram
  float bins[texDictbins.size()];
  vecToArr(texDictbins, bins);

  // Import Models
  map<string, vector<Mat> > savedClassModels;
  int serial = getClassHist(savedClassModels);

  // Iterate through each segment
  for(int i=0;i<segments.size();i++){
    map<string, vector<double> > tmpVals; // For storing all best matches between classes over several iterations
    int repeats = params["testRepeats"];

    // Take repeat readings of segments taking the mean of the best results for each class
    for(int k=0;k<repeats;k++){
      // Cluster Segment
      Mat imgClusters = clusterImg(segments[i], params, paramsDB);

      // Supplant cluster centers with nearest textons
      vector<double> textonDistance;
      textonFind(imgClusters, texDict, textonDistance);

      // Calculate Histogram of textons
      Mat out1;
      int histSize = texDictbins.size()-1;
      const float* histRange = {bins};
      calcHist(&imgClusters, 1, 0, Mat(), out1, 1, &histSize, &histRange, false, false);

      // Compare all saved histograms against novelimg
      double high = DBL_MAX, secHigh = DBL_MAX, clsHigh = DBL_MAX;
      string match, secMatch;

      // Loop through all stored Models and compare
      for(auto const ent2 : savedClassModels){
        vector<double> tmpVec;
        // Compare saved histgrams from each class
        for(int j=0;j < ent2.second.size();j++){
          Mat tmpHist = ent2.second[j].clone();
          double val = compareHist(out1,tmpHist,CV_COMP_CHISQR);

          tmpVec.push_back(val); // Push back all results (all of same class)
        }
       sort(tmpVec.begin(), tmpVec.end()); // Sort from best to worse match (for a single class)
       tmpVals[ent2.first].push_back(tmpVec[0]); // Store the best match for a class to the test image
      }
    }
    // Average results and return first and return first and second
    avgIterResults(tmpVals);
  }

  return 0;
}
