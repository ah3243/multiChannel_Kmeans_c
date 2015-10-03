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
#define DICTIONARY_BUILD 1
#define MODEL_BUILD 1
#define NOVELIMG_TEST 1

#define ERR(msg) printf("\n\nERROR!: %s Line %d\nExiting.\n\n", msg, __LINE__);

using namespace boost::filesystem;
using namespace cv;
using namespace std;

int main( int argc, char** argv ){

  // // ATTEMPTS VARIABLES
  // vector<string> folderName = {"5", "10", "15", "20", "25", "30","35", "40", "45", "50"};
  // vector<int> attemptVec = {5,10,15,20,25,30,35,40,45,50};
  // vector<int> scaleVec = {9};
  // vector<int> cropsizeVec = {70};
  // vector<int> dictClustersVec = {30};

  // // TEXTDICT VARIABLES
  // vector<string> folderName = {"3","5","8","10","12","15","20","25","30","40","50"};
  // vector<int> dictClustersVec = {3,5,8,10,12,15,20,25,30,40,50};
  // vector<int> attemptVec = {35};
  // vector<int> scaleVec = {9};
  // vector<int> cropsizeVec = {210};

  // // SCALE VARIABLES
  // vector<string> folderName = {"_9", "_8", "_7", "_6", "_5", "_4","_3", "_2", "_1", "_0"};
  // vector<int> dictClustersVec = {10};
  //  vector<int> scaleVec = {9,8,7,6,5,4,3,2,1,0};
  //  vector<int> attemptVec = {35};

  // // CROPPING SIZE 9
    // vector<string> folderName = {"5","10","20","30","40","50","60","70"};
    // vector<int> dictClustersVec = {10};
    // vector<int> scaleVec = {9};
    // vector<int> attemptVec = {35};
    // vector<int> cropsizeVec = {5,10,20,30,40,50,60,70};

  // // CROPPING SIZE 8
    // vector<string> folderName = {"10","20","40","60","80","100","120","140"};
    // vector<int> dictClustersVec = {10};
    // vector<int> scaleVec = {8};
    // vector<int> attemptVec = {35};
    // vector<int> cropsizeVec = {10,20,40,60,80,100,120,140};

 // CROPPING SIZE 7
  //   vector<string> folderName = {"25","50","80","100","125","150","175", "200"};
  //   vector<int> dictClustersVec = {5};
  //   vector<int> scaleVec = {7};
  //   vector<int> attemptVec = {35};
  //   vector<int> cropsizeVec = {25,50,80,100,125,150,175, 200};

   // // // CROPPING SIZE 6
    //   vector<string> folderName = {"40","80","120","140","160","180","200","240","280"};
    //   vector<int> dictClustersVec = {10};
    //   vector<int> scaleVec = {6};
    //   vector<int> attemptVec = {35};
    //   vector<int> cropsizeVec = {40,80,120,140,160,180,200,240,280};

    // // INDIVIDUAL VARIABLES
    vector<string> folderName = {"TEST"};
    vector<int> cropsizeVec = {70};
    vector<int> dictClustersVec = {10};
    vector<int> scaleVec = {9};
    vector<int> attemptVec = {35};

    int programLoop = 1; // Number of time the whole program will repeat
    int testLoop = 1; // Number of results sets which will be gathered
  for(int xx=0;xx<programLoop;xx++){
    cout << "\n\n.......Starting Program...... \n\n" ;
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

    if(scaleVec.size()==1){
      cout << "USING FIXED SCALE..\n";
      scale = scaleVec[0];
    }else{
      scale = scaleVec[xx];
    }
    if(attemptVec.size()==1){
      cout << "USING FIXED ATTEMPTS\n";
      attempts = attemptVec[0];
    }else{
      attempts = attemptVec[xx];
    }

    // Adjust the cropSize depending on chosen scale
    double cropScale[]={1, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1};

    int cropsize;
  //  cropsize = (700*cropScale[scale]); // Cropsize is 140Pixels at 256 x 144
    if(cropsizeVec.size()==1){
      cout << "USING FIXED CROPSIZE\n";
      cropsize = cropsizeVec[0];
    }else{
      cropsize = cropsizeVec[xx];
    }

    int DictSize;
    if(dictClustersVec.size()==1){
      cout << "USING FIXED TEXTON CLUSTER NUMBER\n";
      DictSize = dictClustersVec[0];
    }else{
      DictSize = dictClustersVec[xx];
    }

    cout << "This is the cropsize: " << cropsize << endl;
    // exit(1);
    cout << "This is the scalesize: " << scale << endl;
    double modOverlap = 0; // Percentage of crop which will overlap horizontally
    double modelOverlapDb = (modOverlap/100)*cropsize; // Calculate percentage of cropsize
    int modelOverlap = modelOverlapDb; // To convert to int for transfer to function
    // cout << "This is the modelOverlap: " << modelOverlap << " and modelOverlapDb: " << modelOverlapDb << endl;
    // exit(1);


    int testimgOverlap =modelOverlap; // Have the same test and model overlap

    int dictDur, modDur, novDur;
    int numClusters = 10;
    int flags = KMEANS_PP_CENTERS;
    int kmeansIteration = 100000;
    double kmeansEpsilon = 0.000001;
    cout << "\nDictionary Size: " << DictSize << "\nNumber of Clusters: " << numClusters << "\nAttempts: " << attempts << "\nIterations: "
    << kmeansIteration << "\nKmeans Epsilon: " << kmeansEpsilon << endl;

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
      cout << "\n\n.......Testing Against Novel Images...... \n" ;
        // Measure time efficiency
        auto t5 = std::chrono::high_resolution_clock::now();

        novelImgHandle(testPath, clsPath, scale, cropsize, numClusters, DictSize, flags,
          attempts, kmeansIteration, kmeansEpsilon, testimgOverlap, folderName[xx]);

        // Measure time efficiency
        auto t6 = std::chrono::high_resolution_clock::now();
        novDur = std::chrono::duration_cast<std::chrono::milliseconds>(t6 - t5).count();
      }
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
//  waitKey(0);
  return 0;
}
