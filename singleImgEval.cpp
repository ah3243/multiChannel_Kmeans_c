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
   waitKey(1000);
  }
  return rview;
}


int directionHandle(Mat inImg){
  // Remap input to remove distortion
  Mat rectImg = remapImg(inImg);

  // Generate and Store Filterbank
  vector<vector<Mat> > filterbank;
  int n_sigmas, n_orientations;
  createFilterbank(filterbank, n_sigmas, n_orientations);
  // Filter input Image
  Mat filteredImg;
  filterHandle(inImg, filteredImg, filterbank, n_sigmas, n_orientations);

  // Import Models
  return 0;
}
