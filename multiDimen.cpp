///////////////////////////////////////
// Opencv BoW Texture classification //
///////////////////////////////////////

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

#include "filterbank.h" // Filterbank Handling Functions
#include "imgCollection.h" // Img Handling Functions
#include "dictCreation.h" // Generate and store Texton Dictionary
#include "modelBuild.h" // Generate models from class images
#include "imgFunctions.h" // Img Processing Functions

#define INTERFACE 0
#define DICTIONARY_BUILD 0
#define MODEL_BUILD 0
#define NOVELIMG_TEST 1

#define ERR(msg) printf("\n\nERROR!: %s Line %d\nExiting.\n\n", msg, __LINE__);

using namespace boost::filesystem;
using namespace cv;
using namespace std;

//#define cropsize  200
#define CHISQU_threshold 1

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

////////////////////////
// Key:               //
// 0 == TruePositive  //
// 1 == FalsePositive //
 // 2 == TrueNegative  //
// 3 == FalseNegative //
////////////////////////
void addTrueNegatives(string exp1, string exp2, map<string, vector<int> >& res){
  for(auto ent7 : res){
    if(exp1.compare(ent7.first)!=0 && exp2.compare(ent7.first)!=0){
      string holder = ent7.first;
      res[holder][2] += 1;
    }
  }
}

void initROCcnt(vector<map<string, vector<int> > >& r, map<string, vector<Mat> > classImgs){
  cout << "initialising.. " << endl;
  int possResults = 4; // allow space for TP, FP, TN, FN

  map<string, vector<int> > a;
  for(auto const ent5 : classImgs){
    for(int i=0;i<possResults;i++){
      a[ent5.first].push_back(0);
    }
    cout << "Initilsing..." << ent5.first << endl;
  }
  r.push_back(a);
  cout << "\n";
}

void printResults(map<string, vector<vector<int> > > r, vector<string> clsNmes){
  assert(r.size()==clsNmes.size());
  string testtype[] = {"TruePositive", "FalsePositive", "TrueNegative", "FalseNegative"};
  cout << "\n\n----------------------------------------------------------\n\n";
  cout << "                    These are the test results                  \n";

  for(int i = 0;i<clsNmes.size();i++){
    string clss = clsNmes[i];
    cout << "\nClass: " << clss << "\n\n";
    // Loop through both TPR and FPR results
    for(int j = 0 ;j<4;j++){
      cout << testtype[j] << " : " ;
      // Loop through all test iterations
      for(int k = 0;k<r[clss][j].size();k++){
        cout << r[clss][j][k] << ", ";
      }
      cout << "\n";
    }
    cout << "\n\n";
  }
}

int getClsNames(map<string, vector<int> > &r, vector<string> &nme){
  for(auto const ent5 : r){
    if(ent5.second.size()<=0){
      return 0;
    }
    nme.push_back(ent5.first);
  }
}

void cacheTestdata(string correct, string prediction, map<string, vector<int> >& results){
  //  cout << " This was the prediction: " << prediction << " ACTUAL: " << correct << endl;
  if(correct.compare(prediction)==0){
    // If correct
    // add 1 to True Positive
    results[correct][0] += 1;
    // add 1 to True Negative to all other classes
    addTrueNegatives(correct, "", results);
  }else if(correct.compare(prediction)!=0){
    // If incorrect
    // add 1 to False Positive for Predicted
    results[prediction][1] += 1;
    // add 1 to False Negative to Correct
    results[correct][3] += 1;
    // add 1 to True Negative to all other classes
    addTrueNegatives(correct, prediction, results);
  }
}

void organiseResultByClass(vector<map<string, vector<int> > >in, map<string, vector<vector<int> > > &out, vector<string> clsNmes){
  int numTests = in.size();

  vector<int> a;
  cout << "\n\nNumber of tests: " << in.size() << endl;
  // Go through each class
  for(int i=0;i<clsNmes.size();i++){
    // Go through the 4 types of result for each class
    cout << "\nnumber of classes: " << clsNmes.size() << endl;
    for(int j=0;j<4;j++){
      cout << "going through test.." << endl;
      // Go through each test
      for(int k=0;k<in.size();k++){
        out[clsNmes[i]].push_back(a); // Initilse with vector
        out[clsNmes[i]][j].push_back(in[k][clsNmes[i]][j]);
      }
    }
  }
}

///////////////////////////
// 0 == TPR == (TP/TP+FN)//
// 1 == FPR == (FP/FP+TN)//
///////////////////////////
void calcROCVals(map<string, vector<vector<int> > > in, map<string, vector<vector<double> > >& out, vector<string> clssNmes){
  assert(clssNmes.size() == in.size());
  vector<double> a;

  // Go through each Class
  for(int i=0;i<clssNmes.size();i++){
    string curCls = clssNmes[i];
    cout << "Class: " << curCls << "\n\n";

    // Go through each test iteration
    for(int j=0;j<in[curCls][0].size();j++){
      // Calculate TPR
      double TPR, TP, FN;
      TP = in[curCls][0][j];
      FN = in[curCls][3][j];
      TPR = (TP/(TP+FN));
      cout << " " << j << ":-  TP: " << TP << " FN: " << FN << " TPR: " << TPR;

      // Calculate FPR
      double FPR, FP, TN;
      FP = in[curCls][1][j];
      TN = in[curCls][2][j];
      FPR = (FP/(FP+TN));
      cout << " FP: " << FP << " TN: " << TN << " FPR: " << FPR;

      // Pushback vector for each test
      out[curCls].push_back(a);
      // Push back results
      out[curCls][j].push_back(TPR);
      out[curCls][j].push_back(FPR);
      // double TPd = TP, TNd = TN, FPd = FP, FNd = FN;

      cout << "\naccuracy: " << ((TP+TN)/(TP+TN+FP+FN))*100 << "\n\n";
    }
  }
}

// void saveTestData(vector<map<string, vector<int> > > r){
  //   cout << "inside test data.." << endl;
  //   vector<string> nme;
  //   assert(getClsNames(r[0], nme));
  //
  //   FileStorage fs("results.xml", FileStorage::WRITE);
  //   for(int i=0;i<r.size();i++){
  //     cout << "Test: " << i << endl;
  //     stringstream ss;
  //     ss << "TEST:" << i;
  //     fs << ss.str() << "{";
  //     for(const auto ent : r){
  //       cout << "Class: " << ent.first << endl;
  //       fs
  //       for(int j=0;j<ent.second.size();j++){
  //
  //       }
  //     }
  //   }
  // }

void getClassHist(map<string, vector<Mat> >& savedClassHist){
  // Load in Class Histograms(Models)
  FileStorage fs3("models.xml", FileStorage::READ);
  FileNode fn = fs3.root();
  if(fn.type() == FileNode::MAP){

    // Create iterator to go through all the classes
    for(FileNodeIterator it = fn.begin();it != fn.end();it++){
      string clsNme = (string)(*it)["Name"];
      savedClassHist[clsNme];

      // Create node of current Class
      FileNode clss = (*it)["Models"];
      // Iterate through each model inside class, saving to map
      for(FileNodeIterator it1  = clss.begin();it1 != clss.end();it1++){
        FileNode k = *it1;
        Mat tmp;
        k >> tmp;
        savedClassHist[clsNme].push_back(tmp);
      }
    }
    fs3.release();
  }else{
    ERR("Class file was not map. Exiting");
    exit(-1);
  }
}

void getDictionary(Mat &dictionary, vector<float> &m){
  // Load TextonDictionary
  FileStorage fs("dictionary.xml",FileStorage::READ);
  if(!fs.isOpened()){
    ERR("Unable to open Texton Dictionary.");
    exit(-1);
  }

  fs["vocabulary"] >> dictionary;
  fs["bins"] >> m;
  fs.release();
}

void testNovelImgHandle(int clsAttempts, int numClusters, map<string, vector<int> >& results, map<string, vector<Mat> > classImgs, map<string, vector<Mat> > savedClassHist, map<string, Scalar> Colors, int cropsize){
  int clsFlags = KMEANS_PP_CENTERS;
  TermCriteria clsTc(TermCriteria::MAX_ITER + TermCriteria::EPS, 1000, 0.0001);
  BOWKMeansTrainer novelTrainer(numClusters, clsTc, clsAttempts, clsFlags);

  // Window for segment prediction display
  namedWindow("mywindow", CV_WINDOW_AUTOSIZE);

  // Import Texton Dictionary
  Mat dictionary;
  vector<float> m;
  getDictionary(dictionary, m);
  float bins[m.size()];
  vecToArr(m, bins);

  // Initilse Histogram parameters
  int histSize = m.size()-1;
  const float* histRange = {bins};
  bool uniform = false;
  bool accumulate = false;

    // Store aggregated correct, incorrect and unknown results for segment prediction display
    int Correct = 0, Incorrect =0, Unknown =0;
    // Loop through All Classes
    for(auto const ent : classImgs){
      // Loop through all images in Class
      cout << "\n\nEntering Class: " << ent.first << endl;
      for(int h=0;h<ent.second.size();h++){
        if(ent.second[h].rows != ent.second[h].cols){
          ERR("Novel input image was now square. Exiting");
          exit(-1);
        }

        Mat in, hold;
        in = ent.second[h];
         if(in.empty()){
          ERR("Novel image was not able to be imported. Exiting.");
          exit(-1);
        }

        // Send img to be filtered, and responses aggregated with addWeighted
        filterHandle(in, hold);

        // Divide the 200x200pixel image into 100 segments of 400x1 (20x20)
        vector<Mat> test;
        segmentImg(test, hold, cropsize);

        int imgSize = ent.second[h].rows;
        Mat disVals = Mat(200,200,CV_8UC3);

        // Counters for putting 'pixels' on display image
        int disrows = 0, discols = 0;
        // Loop through and classify all image segments
        for(int x=0;x<test.size();x++){
          // handle segment prediction printing
          discols = (discols + cropsize)%imgSize;
          if(discols==0){
            disrows += 20;
          }
          if(!test[x].empty()){
             novelTrainer.add(test[x]);
           }

           // Generate 10 clusters per segment and store in Mat
           Mat clus = Mat::zeros(numClusters,1, CV_32FC1);
           clus = novelTrainer.cluster();

           // Replace Cluster Centers with the closest matching texton
           textonFind(clus, dictionary);

          // Calculate the histogram
           Mat out1;
           calcHist(&clus, 1, 0, Mat(), out1, 1, &histSize, &histRange, uniform, accumulate);
           novelTrainer.clear();

           double high = DBL_MAX, secHigh = DBL_MAX;
           string match, secMatch;
           for(auto const ent2 : savedClassHist){
             for(int j=0;j < ent2.second.size();j++){
               double val = compareHist(out1,ent2.second[j],CV_COMP_CHISQR);
               // Save best match value and name
               if(val < high){
                 high = val;
                 match = ent2.first;
               }
               // save second best match and name
               else if(val < secHigh && val > high && match.compare(ent2.first) != 0){
                 secHigh = val;
                 secMatch = ent2.first;
               }
             }
           }

           string prediction = "";
           // If the match is above threshold or nearest other match is to similar, return unknown
          //  if(high>CHISQU_threshold || secHigh<CHISQU_threshold){
          //    prediction = "Unknown";
          //  }else{
             prediction = match;
          //  }
           // Populate Window with predictions
           if(prediction.compare(ent.first)==0){
             Correct += 1;
            rectangle(disVals, Rect(discols, disrows,cropsize,cropsize), Colors["Correct"], -1, 8, 0);
           }else if(prediction.compare("Unknown")==0){
            //  Unknown +=1;
            // rectangle(disVals, Rect(discols, disrows,cropsize,cropsize), Colors[prediction], -1, 8, 0);
           }else{
             Incorrect += 1;
//            rectangle(disVals, Rect(discols, disrows,cropsize,cropsize), Colors["Incorrect"], -1, 8, 0);
           }
          // Save ROC data to results, clsAttempts starts at 0 so is -1
          cacheTestdata(ent.first, match, results);
      }
       imshow("mywindow", disVals);
       waitKey(500);
    }
    // END OF CLASS, CONTINUING TO NEXT CLASS //
  }
}

void printRAWResults(map<string, vector<int> > r){
  cout << "\n\n----------------------------------------------------------\n\n";
  cout << "                    These are the test results                  \n";
  for(auto const ent6 : r){
    cout << ent6.first;
    cout << "\n      TruePositive:  " << ent6.second[0];
    cout << "\n      TrueNegative:  " << ent6.second[1];
    cout << "\n      FalsePositive: " << ent6.second[2];
    cout << "\n      FalseNegative: " << ent6.second[3];
    cout << "\n";
  }
  cout << "\n\n";
}

int main( int argc, char** argv ){
  cout << "\n\n.......Starting Program...... \n\n" ;
  int cropsize = 50;

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
  dictCreateHandler(cropsize);
  #endif

  #if MODEL_BUILD == 1
    ///////////////////////////////////////////////////////////
    // Get histogram responses using vocabulary from Classes //
    ///////////////////////////////////////////////////////////
    cout << "\n\n........Generating Class Models from Imgs.........\n";
    modelBuildHandle(cropsize);
  #endif



  #if NOVELIMG_TEST == 1
  cout << "\n\n.......Testing Against Novel Images...... \n" ;
    //////////////////////////////
    // Test Against Novel Image //
    //////////////////////////////

    // Load Images to be tested
    path p = "../../../TEST_IMAGES/CapturedImgs/classes";
    map<string, vector<Mat> > classImages;
    loadClassImgs(p, classImages);

    map<string, vector<Mat> > savedClassHist;

    getClassHist(savedClassHist);

    // Stock Scalar Colors
    map<string, Scalar> Colors;
      vector<Scalar> clsColor;
        clsColor.push_back(Scalar(0,0,250)); // Blue
        clsColor.push_back(Scalar(255,128,0)); // Orange
        clsColor.push_back(Scalar(255,255,0)); // Yellow
        clsColor.push_back(Scalar(0,255,255)); // Turquoise
        clsColor.push_back(Scalar(127,0,255)); // DarkPurple
        clsColor.push_back(Scalar(255,0,255)); // Purple
        clsColor.push_back(Scalar(255,0,127)); // Pink/Red

      Colors["Correct"] = Scalar(0,255,0); // Green
      Colors["Incorrect"] = Scalar(255,0,0); // Red
      Colors["Unknown"] = Scalar(100,100,100);

      int count =0;
      double txtWidth =0, txtHeight =0;
      for(auto const ent : savedClassHist){
        Colors[ent.first] = clsColor[count];
        int bseline;
        Size s = getTextSize(ent.first, CV_FONT_HERSHEY_SIMPLEX, 0.5, 1, &bseline);
        // Get txtLength
        if(s.width > txtWidth){
          txtWidth = s.width;
        }
        // Get txtHeight
        txtHeight += s.height;

        count++;
      }

    // Holds Class names, each holding a count for TP, FP, FN, FP Values
    vector<map<string, vector<int> > > results;

    // Create Img Legend
    Mat Key = Mat::zeros(txtHeight+120,txtWidth+80,CV_8UC3);
      int cnt=0;
      for(auto const ent1 : Colors){
        putText(Key, ent1.first, Point(10, 20+ cnt*20), CV_FONT_HERSHEY_SIMPLEX, 0.5, Scalar(255,0,100), 1, 8, false);
        rectangle(Key, Rect(txtWidth+50, 10 + cnt*20, 10,10), ent1.second, -1, 8, 0 );
        cnt++;
      }

    // Window for Legend display
    namedWindow("legendWin", CV_WINDOW_AUTOSIZE);
    imshow("legendWin", Key);


  int counter = 0;
  for(int numClusters=5;numClusters<6;numClusters++){
    cout << "\nCount: " << counter << endl;
    initROCcnt(results, classImages); // Initilse map
    int clsAttempts = 5;
    testNovelImgHandle(clsAttempts, numClusters, results[counter], classImages, savedClassHist, Colors, cropsize);
    counter++;
  }

  printRAWResults(results[0]);

  map<string, vector<vector<int> > > resByCls;
  map<string, vector<vector<double> > > ROCVals;
  vector<string> clsNmes;
  getClsNames(results[0], clsNmes); // Get class Names
  organiseResultByClass(results, resByCls, clsNmes);

  // print results, clsAttempts starts at 1 so is -1
  printResults(resByCls, clsNmes);

  calcROCVals(resByCls, ROCVals, clsNmes);


  if(results.size()<2){
    cout << "\n\nThere are not enough iterations to produce a ROC graph. Exiting." << endl;
    return 0;
  }

  int wW = 400, wH = 400, buffer =50, border = buffer-10;
  namedWindow("ROC_Curve", CV_WINDOW_AUTOSIZE);
  Mat rocCurve = Mat(wH,wW, CV_8UC3, Scalar(0,0,0));

  line(rocCurve, Point(border,wH-border), Point(wW-border,wH-border), Scalar(255,255,255), 2, 8, 0); // X axis border
  // putText(rocCurve, "FPR", Point(10, 20+ cnt*20), CV_FONT_HERSHEY_SIMPLEX, 0.5, Scalar(255,0,100), 1, 8, false);
  line(rocCurve, Point(border,border), Point(40,wH-border), Scalar(255,255,255), 2, 8, 0); // Y axis border
  // putText(rocCurve, "TPR", Point(10, 20+ cnt*20), CV_FONT_HERSHEY_SIMPLEX, 0.5, Scalar(255,0,100), 1, 8, false);


  // Cycle through all classes
  for(auto const ent : ROCVals){
    cout << "Class: " << ent.first << endl;
    // Cycle through all iterations to produce graph
    for(int i=0;i<ent.second.size()-1;i++){
      line(rocCurve, Point(((wW-buffer)*ent.second[i][1]+50),((wW-buffer)*ent.second[i][0]+50)),
      Point(((wW-buffer)*ent.second[i+1][1]+50),((wW-buffer)*ent.second[i+1][0])+50),
      Colors[ent.first], 1, 8, 0);
      waitKey(500);
      imshow("ROC_Curve", rocCurve);
      }
  }
waitKey(0);


  #endif

  return 0;
}

void drawROC(){

}
