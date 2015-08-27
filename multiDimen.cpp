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
//     }
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
  int cropsize = 100;
  double scale = 0.5;
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
