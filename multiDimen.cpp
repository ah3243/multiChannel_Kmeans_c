///////////////////////////////////////
// Opencv BoW Texture classification //
///////////////////////////////////////

#include "opencv2/highgui/highgui.hpp" // Needed for HistCalc
#include <opencv2/opencv.hpp>
#include "opencv2/core/core.hpp"
#include <iostream> // General io
#include <stdio.h> // General io
#include <string>
#include <boost/filesystem.hpp>
#include <assert.h>
#include <chrono>  // time measurement
#include <thread>  // time measurement

#include "imgFunctions.h"
#include "filterbank.h"
#include "dictCreation.h" // Generate and store Texton Dictionary
#include "modelBuild.h" // Generate models from class images
#include "novelImgTest.h" // Novel Image Testing module

#define INTERFACE 0
#define DICTIONARY_BUILD 0
#define MODEL_BUILD 0
#define NOVELIMG_TEST 1

#define ERR(msg) printf("\n\nERROR!: %s Line %d\nExiting.\n\n", msg, __LINE__);

using namespace boost::filesystem;
using namespace cv;
using namespace std;


// int main(int argc, char** argv){
//     if(argc<3){
//       cout << "not enough inputs entered, exiting." << endl;
//       exit(1);
//     }
//     Mat img1 = imread(argv[1], CV_LOAD_IMAGE_GRAYSCALE);
//     Mat img2 = imread(argv[2], CV_LOAD_IMAGE_GRAYSCALE);
//     if(img1.empty() || img2.empty())
//     {
//         printf("Can't read one of the images\n");
//         return -1;
//     }system
//
//     // detecting keypoints
//     SurfFeatureDetector detector(400);
//     vector<KeyPoint> keypoints1, keypoints2;
//     detector.detect(img1, keypoints1);
//     detector.detect(img2, keypoints2);
//
//     // computing descriptors
//     SurfDescriptorExtractor extractor;
//     Mat descriptors1, descriptors2;
//     extractor.compute(img1, keypoints1, descriptors1);
//     extractor.compute(img2, keypoints2, descriptors2);
//
//     // matching descriptors
//     BruteForceMatcher<L2<float> > matcher;
//     vector<DMatch> matches;
//     matcher.match(descriptors1, descriptors2, matches);
//     //drawing the results
//     for(int i=0;i<matches.size();i++)
//       cout << "This is the level of match..: " << matches[i].distance << " This is size: " << matches.size() << endl;
//   //  namedWindow("matches", 1);
//   //  Mat img_matches;
//   //  drawMatches(img1, keypoints1, img2, keypoints2, matches, img_matches);
//   //  imshow("matches", img_matches);
//    waitKey(0);
//   }


int main( int argc, char** argv ){
  cout << "\n\n.......Starting Program...... \n\n" ;

 // Available Scales
  ////////////////////////////////////////
  // These are the possible resolutions //
  // assign scale to the corresponding  //
  // number                             //
  //                                    //
  // 0   1280 x 720                     //
  // 1   1152 x 648                     //
  // 2   1024 x 576                     //
  // 3   896 x 504                      //
  // 4   768 x 432                      //
  // 5   640 x 360                      //
  // 6   512 x 288                      //
  // 7   384 x 216                      //
  // 8   256 x 144                      //
  // 9   128 x 72                       //
  ////////////////////////////////////////

  int scale = 8;

  // Adjust the cropSize depending on chosen scale
  double cropScale[]={1, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1};

// int cropsize = (335*cropScale[scale]); // Cropsize is 100Pixels at 384 x 216
  int cropsize = 140;
  int dictDur, modDur, novDur;
  int numClusters = 10;
  int DictSize = 10;

  path textonPath = "../../../TEST_IMAGES/CapturedImgs/textons";
  path clsPath = "../../../TEST_IMAGES/CapturedImgs/classes/";
  path testPath = "../../../TEST_IMAGES/CapturedImgs/novelImgs";

  #if INTERFACE == 1
    ///////////////////////
    // Collecting images //
    ///////////////////////

    cout << "\n\n............... Passing to Img Collection Module ................\n";

    imgCollectionHandle();

    cout << "\n\n............... Returning to main program ..................... \n";

  #endif
  #if DICTIONARY_BUILD == 1
    ////////////////////////////////
    // Creating Texton vocabulary //
    ////////////////////////////////

    cout << "\n\n.......Generating Texton Dictionary...... \n" ;
    // Measure start time
    auto t1 = std::chrono::high_resolution_clock::now();

    dictCreateHandler(cropsize, scale, DictSize);

    // Measure time efficiency
    auto t2 = std::chrono::high_resolution_clock::now();
    dictDur = std::chrono::duration_cast<std::chrono::milliseconds>(t2 - t1).count();

  #endif
  #if MODEL_BUILD == 1
    ///////////////////////////////////////////////////////////
    // Get histogram responses using vocabulary from Classes //
    ///////////////////////////////////////////////////////////

    cout << "\n\n........Generating Class Models from Imgs.........\n";
    // Measure start time
    auto t3 = std::chrono::high_resolution_clock::now();

    modelBuildHandle(cropsize, scale, numClusters);

    // Measure time efficiency
    auto t4 = std::chrono::high_resolution_clock::now();
    modDur = std::chrono::duration_cast<std::chrono::milliseconds>(t4 - t3).count();

  #endif
  #if NOVELIMG_TEST == 1
    //////////////////////////////
    // Test Against Novel Image //
    //////////////////////////////

    cout << "\n\n.......Testing Against Novel Images...... \n" ;
    // Measure time efficiency
    auto t5 = std::chrono::high_resolution_clock::now();

    novelImgHandle(testPath, clsPath, scale, cropsize, numClusters, DictSize);

    // Measure time efficiency
    auto t6 = std::chrono::high_resolution_clock::now();
    novDur = std::chrono::duration_cast<std::chrono::milliseconds>(t6 - t5).count();
  #endif


// /////////////////////////////////// TEST ////////////////////////////////////////////////////////
//
//       vector<vector<Mat> > filterbank;
//       int n_sigmas, n_orientations;
//       createFilterbank(filterbank, n_sigmas, n_orientations);
//
//       Mat imgIn, imgOut;
//       int Flags = KMEANS_PP_CENTERS;
//       TermCriteria Tc(TermCriteria::MAX_ITER + TermCriteria::EPS, 1000, 0.0001);
//       BOWKMeansTrainer tstTrain(30, Tc, 5, Flags);
//
//
//       float blah[] = {0,255};
//       const float* histblah= {blah};
//       int histSize[] = {10};
//       int channels[] = {0};
//       Mat ou1;
//       namedWindow("testWin", CV_WINDOW_AUTOSIZE);
//
//       vector<Mat> compareMe;
//       for(int j=0;j<1;j++){
//         for(int i=0;i<2;i++){
//           // if(i==0){
//           //   img1 = imread("../../bread.png", CV_LOAD_IMAGE_GRAYSCALE);
//           // }else{
//           Mat img1 = imread("../../wool.png", CV_LOAD_IMAGE_GRAYSCALE);
//           imshow("testWin", img1);
//           // waitKey(1000);
//           // }
//
// //         filterHandle(img1, imgOut, filterbank, n_sigmas, n_orientations);
//           cout << "This is the size.." << img1.rows << " cols: " << img1.cols << endl;
//           Mat imgFlat = reshapeCol(img1);
//           cout << "This is the size.." << imgFlat.rows << " cols: " << imgFlat.cols << endl;
//
//           tstTrain.add(imgFlat);
//           Mat clusters = Mat::zeros(10,1, CV_32FC1);
//           clusters = tstTrain.cluster();
//
//          calcHist(&clusters, 1, channels, Mat(), ou1, 1, histSize, &histblah, true, false);
//          cout << "after the calc hist" << endl;
//           compareMe.push_back(ou1);
//          tstTrain.clear();
//         //   cout << "done one loop.. this is the size: " << compareMe.size() << endl;
//         // cout << "This is the cluster1 : " << compareMe[i] << endl;
//         //  cout << "This is the tstTrain.size(): " << tstTrain.descripotorsCount() << endl;
//         }
//
//         double value =  compareHist(compareMe[0], compareMe[1], CV_COMP_CHISQR);
//         cout << "This is the Chisqr comparison.." << value << endl;
//         compareMe.clear();
//       }

/////////////////////////////////// TEST ////////////////////////////////////////////////////////



  int totalTime =0;
  cout << "\n";
  if(DICTIONARY_BUILD == 1){
//    cout << "Texton Dictionary Creation took: "
          cout  << dictDur << "\n";
//            << " milliseconds\n";
            totalTime +=dictDur;
  }
  if(MODEL_BUILD == 1){
//    cout << "Model Creation took: "
        cout << modDur << "\n";
          //  << " milliseconds\n";
            totalTime+=modDur;
  }
  if(NOVELIMG_TEST == 1){
//    cout << "Novel Image Testing took: "
          cout << novDur << "\n";
  //          << " milliseconds\n";
            totalTime+=novDur;
  }
//  cout << "The Total Time was: "
  cout<< totalTime << "\n\n";
  return 0;
}
