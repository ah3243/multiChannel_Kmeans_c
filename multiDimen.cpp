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

#include "singleImgEval.h"
#include "imgFunctions.h"
#include "filterbank.h"
#include "dictCreation.h" // Generate and store Texton Dictionary
#include "modelBuild.h" // Generate models from class images
#include "novelImgTest.h" // Novel Image Testing module

#define VERBOSE 0

#define INTERFACE 0
#define DICTIONARY_BUILD 1
#define MODEL_BUILD 1
#define NOVELIMG_TEST 1

#define kmeansIteration 100000
#define kmeansEpsilon 0.000001
#define numClusters 10 // For model and test images
#define flags KMEANS_PP_CENTERS

#define ERR(msg) printf("\n\nERROR!: %s Line %d\nExiting.\n\n", msg, __LINE__);

using namespace boost::filesystem;
using namespace cv;
using namespace std;

int main( int argc, char** argv ){
  cout << "number of inputs: " << argc << " 0: " << argv[0] << " 1: " << argv[1] << " 2: " << argv[2] << endl;


  // Int to pass in parameter
  int testInput = atoi(argv[1]);
  // String to detail the type of test
  string testType(argv[2]);

  // Bool to denote the first iteration or not
  int tmp = atoi(argv[3]);
  assert(tmp==1|| tmp==0);
  bool firstGo =(bool)tmp;

  // // ATTEMPTS VARIABLES
  // vector<string> folderName = {"5", "10", "15", "20", "25", "30","35", "40", "45", "50"};
  // vector<int> attemptVec = {5,10,15,20,25,30,35,40,45,50};

  // // TEXTDICT VARIABLES
  // vector<string> folderName = {"3","5","8","10","12","15","20","25","30","40","50"};
  // vector<int> dictClustersVec = {3,5,8,10,12,15,20,25,30,40,50};

  // ---- INITIAL VARIABLES ------ //
  vector<string> folderName = {"TEST"}; // Where to save prediction Visualisations
  vector<int> dictClustersVec = {10};
  vector<int> scaleVec = {8};
  vector<int> attemptVec = {35};

    int programLoop = 1; // Number of time the whole program will repeat
    int testLoop = 1; // Number of results sets which will be gathered
  for(int xx=0;xx<programLoop;xx++){
    cout << "\n.......Starting Program...... \n" ;
    cout << "This is iteration " << xx << endl;
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
    int scale, attempts;

// SCALE
    if(scaleVec.size()==1){
      if(testType.compare("scale")==0){
        scale = testInput;
        cout << "Testing Scale: " << scale << endl;
      }else{
        scale = scaleVec[0];
      }
      if(VERBOSE){
        cout << "USING FIXED SCALE..\n";
      }
    }else{
      scale = scaleVec[xx];
    }

// ATTEMPTS //
    // if multiple vals in 'attemptVec' then run all otherwise check input
    if(attemptVec.size()==1){
      if(testType.compare("attempts")==0){
        attempts = testInput;
      }else{
        attempts = attemptVec[0];
      }
      if(VERBOSE){
        cout << "USING FIXED ATTEMPTS\n";
      }
    }else{
      attempts = attemptVec[xx];
    }

  // Adjust the cropSize depending on chosen scale
  double cropScale[]={1, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1};
  int cropsize;

  if(testType.compare("cropping")==0){
      if(VERBOSE){
        cout << "USING FIXED CROPSIZE\n";
      }
      cropsize = testInput;
      scale=atoi(argv[4]);
  }else{
    cropsize = (700*cropScale[scale]); // Cropsize is 140Pixels at 256 x 144
  }

    int DictSize;
    if(dictClustersVec.size()==1){
      if(VERBOSE){
        cout << "USING FIXED TEXTON CLUSTER NUMBER\n";
      }
      DictSize = dictClustersVec[0];
    }else{
      DictSize = dictClustersVec[xx];
    }

    double modOverlap = 0; // Percentage of crop which will overlap horizontally
    double modelOverlapDb = (modOverlap/100)*cropsize; // Calculate percentage of cropsize
    int modelOverlap = modelOverlapDb; // To convert to int for transfer to function
    // cout << "This is the modelOverlap: " << modelOverlap << " and modelOverlapDb: " << modelOverlapDb << endl;
    // exit(1);


    int testimgOverlap =modelOverlap; // Have the same test and model overlap

    int dictDur, modDur, novDur;

    cout << "\nDictionary Size: " << DictSize << "\nNumber of Clusters: " << numClusters << "\nAttempts: " << attempts << "\nIterations: "
    << kmeansIteration << "\nKmeans Epsilon: " << kmeansEpsilon << endl;
    cout << "This is the cropsize: " << cropsize << "\n";
    cout << "This is the scalesize: " << scale << endl;

    path textonPath = "../../../TEST_IMAGES/CapturedImgs/textons";
    path clsPath = "../../../TEST_IMAGES/CapturedImgs/classes/";
    path testPath = "../../../TEST_IMAGES/CapturedImgs/novelImgs";

    #if INTERFACE == 1
      ///////////////////////
      // Collecting images //
      ///////////////////////

      cout << "\n............... Passing to Img Collection Module ................\n";

      imgCollectionHandle();

      cout << "\n............... Returning to main program ..................... \n";

    #endif
    #if DICTIONARY_BUILD == 1
      ////////////////////////////////
      // Creating Texton vocabulary //
      ////////////////////////////////

      cout << "\n.......Generating Texton Dictionary...... \n" ;
      // Measure start time
      auto t1 = std::chrono::high_resolution_clock::now();

      dictCreateHandler(cropsize, scale, DictSize, flags, attempts, kmeansIteration, kmeansEpsilon, modelOverlap);

      // Measure time efficiency
      auto t2 = std::chrono::high_resolution_clock::now();
      dictDur = std::chrono::duration_cast<std::chrono::milliseconds>(t2 - t1).count();
      waitKey(500); // a a precausion to ensure all processes are finished
    #endif
    #if MODEL_BUILD == 1
      ///////////////////////////////////////////////////////////
      // Get histogram responses using vocabulary from Classes //
      ///////////////////////////////////////////////////////////

      cout << "\n........Generating Class Models from Imgs.........\n";
      // Measure start time
      auto t3 = std::chrono::high_resolution_clock::now();

      modelBuildHandle(cropsize, scale, numClusters, flags, attempts, kmeansIteration, kmeansEpsilon, modelOverlap);

      // Measure time efficiency
      auto t4 = std::chrono::high_resolution_clock::now();
      modDur = std::chrono::duration_cast<std::chrono::milliseconds>(t4 - t3).count();
      waitKey(500); // a a precausion to ensure all processes are finished
    #endif
    #if NOVELIMG_TEST == 1
      //////////////////////////////
      // Test Against Novel Image //
      //////////////////////////////

      // Take however many sets of test results
      for(int r=0;r<testLoop;r++){
      cout << "\n.......Testing Against Novel Images...... \n" ;
        // Measure time efficiency
        auto t5 = std::chrono::high_resolution_clock::now();

        novelImgHandle(testPath, clsPath, scale, cropsize, numClusters, DictSize, flags,
          attempts, kmeansIteration, kmeansEpsilon, testimgOverlap, folderName[xx], firstGo);

        // Measure time efficiency
        auto t6 = std::chrono::high_resolution_clock::now();
        novDur = std::chrono::duration_cast<std::chrono::milliseconds>(t6 - t5).count();
      }
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
    cout << "This is the cropsize: " << cropsize << endl;
    cout << "This is the scalesize: " << scale << endl;
    cout << "Dictionary Size: " << DictSize << "\nNumber of Clusters: " << numClusters << "\nAttempts: " << attempts << "\nIterations: "
    << kmeansIteration << "\nKmeans Epsilon: " << kmeansEpsilon << endl;
    cout << "\n\n..........ENDING ITERATION NUMBER " << xx << ".............\n";
  }
  destroyAllWindows();
  namedWindow("finished", CV_WINDOW_AUTOSIZE);
  moveWindow("finished", 500,200);
  Mat finished = Mat(300,500,CV_8UC1, Scalar(255,255,255));
  string fini = "...FINISHED...";
  Point org(100, 100);
  putText(finished, fini, org, CV_FONT_HERSHEY_COMPLEX, 1, Scalar(0, 0, 0), 2, 8 );
  imshow("finished", finished);

  cout << "..........ENDING PROGRAM.............\n";
  return 0;
}
