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
#include <chrono>  // time measurement
#include <thread>  // time measurement

#include "filterbank.h" // Filterbank Handling Functions
#include "imgCollection.h" // Img Handling Functions
#include "imgFunctions.h" // Img Processing Functions

#define ERR(msg) printf("\n\nERROR!: %s Line %d\nExiting.\n\n", msg, __LINE__);

using namespace boost::filesystem;
using namespace cv;
using namespace std;

#define CHISQU_MAX_threshold 3
#define CHISQU_DIS_threshold 0


////////////////////////
// Key:               //
// 0 == TruePositive  //
// 1 == FalsePositive //
// 2 == TrueNegative  //
// 3 == FalseNegative //
////////////////////////
void addTrueNegatives(string exp1, string exp2, map<string, vector<double> >& res){
  for(auto ent7 : res){
    if(exp1.compare(ent7.first)!=0 && exp2.compare(ent7.first)!=0){
      string holder = ent7.first;
      res[holder][2] += 1;
    }
  }
}

void initROCcnt(vector<map<string, vector<double> > >& r, vector<string> clsNames){
  cout << "initialising.. " << endl;
  int possResults = 4; // allow space for TP, FP, TN, FN

  map<string, vector<double> > a;
  for(int i=0;i<clsNames.size();i++){
    for(int j=0;j<possResults;j++){
      a[clsNames[i]].push_back(0.0);
    }
    cout << "Initilsing..." << clsNames[i] << endl;
  }
  r.push_back(a);
  cout << "\n";
}

int getClsNames(map<string, vector<double> > &r, vector<string> &nme){
  for(auto const ent5 : r){
    if(ent5.second.size()<=0){
      return 0;
    }
    nme.push_back(ent5.first);
  }
}

void cacheTestdata(string correct, string prediction, map<string, vector<double> >& results){
  // cout << " This was the prediction: " << prediction << " ACTUAL: " << correct << endl;
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
    // cout << "number 3." << endl;
    // add 1 to True Negative to all other classes
    addTrueNegatives(correct, prediction, results);
  }
}

void organiseResultByClass(vector<map<string, vector<double> > >in, map<string, vector<vector<double> > > &out, vector<string> clsNmes){
  int numTests = in.size();

  vector<double> a;
  cout << "\n\nNumber of tests: " << in.size() << endl;
  // Go through each class
  for(int i=0;i<clsNmes.size();i++){
    // Go through the 4 types of result for each class
    for(int j=0;j<4;j++){
      // Go through each test
      for(int k=0;k<in.size();k++){
        out[clsNmes[i]].push_back(a); // Initilse with vector
        out[clsNmes[i]][j].push_back(in[k][clsNmes[i]][j]);
      }
    }
  }
}

//////////////////////////////////////
// 0 == TPR(Sensitity) == (TP/TP+FN)//
// 1 == PPV(Precision) == (TP/TP+FP)//
//////////////////////////////////////
void calcROCVals(map<string, vector<vector<double> > > in, map<string, vector<vector<double> > >& out, vector<string> clssNmes){
  assert(clssNmes.size() == in.size());
  vector<double> a;

  // Go through each Class
  for(int i=0;i<clssNmes.size();i++){
    string curCls = clssNmes[i];
    cout << "Class: " << curCls << "\n\n";

    // Go through each test iteration
    for(int j=0;j<in[curCls][0].size();j++){
      double TPR, PPV, TP, FN, TN, FP;
      TP = in[curCls][0][j];
      FP = in[curCls][1][j];
      TN = in[curCls][2][j];
      FN = in[curCls][3][j];

      // Calculate TPR
      TPR = (TP/(TP+FN));
      cout << " " << j << ":-  TP: " << TP << " FN: " << FN << " TPR: " << TPR;

      // Calculate FPR
      PPV = (TP/(TP+FP));
      cout << " TP: " << TP << " FP: " << FP << " PPV: " << PPV;

      // Pushback vector for each test
      out[curCls].push_back(a);
      // Push back results
      out[curCls][j].push_back(TPR);
      out[curCls][j].push_back(PPV);
      cout << "\naccuracy: " << ((TP+TN)/(TP+TN+FP+FN))*100 << "\n\n";
    }
  }
}

void saveTestData(vector<map<string, vector<double> > > r, int serial){
  cout << "Saving test data" << endl;
  vector<string> nme;
  assert(getClsNames(r[0], nme));

  stringstream fileName;
  //  fileName << serial << "_";
  fileName << "results.xml";
  FileStorage fs(fileName.str(), FileStorage::WRITE);
  fs << "modelSerial" << serial;
  for(int i=0;i<r.size();i++){
    cout << "Iteration " << i << endl;

    stringstream ss;
    ss << "Iteration" << i;
    fs << ss.str() << "{";
    for(const auto ent : r[i]){
      fs << ent.first << "{";
        double TP, FP, TN, FN, PPV, TPR;
        TP = ent.second[0];
        FP = ent.second[1];
        TN = ent.second[2];
        FN = ent.second[3];
        PPV = (TP/(TP+FP));
        TPR = (TP/(TP+FN));

        fs << "TP" << TP;
        fs << "FP" << FP;
        fs << "TN" << TN;
        fs << "FN" << FN;
        fs << "PPV" << PPV;
        fs << "TPR" << TPR;
      fs << "}";
    }
    fs << "}";
  }
}

int getClassHist(map<string, vector<Mat> >& savedClassHist){
  int serial;
  // Load in Class Histograms(Models)
  FileStorage fs3("models.xml", FileStorage::READ);
  fs3["Serial"] >> serial;
  cout << "This is the serial: " << serial << endl;
  FileNode fn = fs3["classes"];
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
  return serial;
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

void testNovelImg(int clsAttempts, int numClusters, map<string, vector<double> >& results, map<string, vector<Mat> > classImgs, map<string, vector<Mat> > savedClassHist, map<string, Scalar> Colors, int cropsize){
  int clsFlags = KMEANS_PP_CENTERS;
  TermCriteria clsTc(TermCriteria::MAX_ITER + TermCriteria::EPS, 1000, 0.0001);
  BOWKMeansTrainer novelTrainer(numClusters, clsTc, clsAttempts, clsFlags);

  // Window for segment prediction display
  namedWindow("segmentPredictions", CV_WINDOW_AUTOSIZE);
  namedWindow("novelImg", CV_WINDOW_AUTOSIZE);
  namedWindow("correct", CV_WINDOW_AUTOSIZE);

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
        Mat in = Mat(ent.second[h].cols, ent.second[h].rows,CV_32FC1, Scalar(0));
        Mat hold = Mat(ent.second[h].cols, ent.second[h].rows,CV_32FC1,Scalar(0));
        in = ent.second[h];
         if(in.empty()){
          ERR("Novel image was not able to be imported. Exiting.");
          exit(-1);
        }

        // Send img to be filtered, and responses aggregated with addWeighted
        filterHandle(in, hold);

        // Divide the image into segments specified in 'cropsize' and flatten for clustering
        vector<Mat> test;
        segmentImg(test, hold, cropsize);

        int imgSize = ent.second[h].rows;
        Mat disVals = Mat(ent.second[h].cols, ent.second[h].rows,CV_8UC3, Scalar(0,0,0));
        Mat matchDisplay = Mat(50,50,CV_8UC3, Scalar(0,0,0));

        // Counters for putting 'pixels' on display image
        int disrows = 0, discols = 0;
        // Loop through and classify all image segments
        for(int x=0;x<test.size();x++){
          // handle segment prediction printing
          if(discols>imgSize){
            discols=0;
          }
          if(discols==0&& x>0){
            disrows += cropsize;
          }

          if(!test[x].empty()){
             novelTrainer.add(test[x]);
           }
           // Generate specified number of clusters per segment and store in Mat
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
           map<string, double> matchResults;
           for(auto const ent2 : savedClassHist){
             matchResults[ent2.first] = DBL_MAX;

             for(int j=0;j < ent2.second.size();j++){
               double val = compareHist(out1,ent2.second[j],CV_COMP_CHISQR);
               if(val < matchResults[ent2.first]){
                 matchResults[ent2.first] = val;
               }
               // Save best match value and name
               if(val < high||val==high){
                 //Save high value as second high
                 secHigh = high;
                 secMatch = match;
                 // Replace high with new value
                 high = val;
                 match = ent2.first;
               }
               // save second best match in case only one activation of high if()
               if(val < secHigh && val > high && match.compare(ent2.first) != 0){
                 secHigh = val;
                 secMatch = ent2.first;
               }
             }
           }
           string prediction = "";
           // If the match is above threshold or nearest other match is to similar, return unknown
           cout << "high: " << high << " secHigh: " << secHigh << endl;

           // If match above threshold or to close to other match or all values are identical Class as 'UnDefined'
           if(high>CHISQU_MAX_threshold || (secHigh - high)<CHISQU_DIS_threshold || secHigh==DBL_MAX){
             prediction = "UnDefined";
           }
          else{
            prediction = match;
          }
          cout << "ACT: " << ent.first << " PD: " << prediction;
          for(auto const ag : matchResults){
            cout << ", " << ag.first << ": " << ag.second;
           }

          cout << "\n";

          //cout << "First Prediction: " << match << " Actual: " << ent.first << " First Distance: " << high << " Second Class: " << secMatch << " Distance: " << secHigh << endl;
          rectangle(disVals, Rect(discols, disrows,cropsize,cropsize), Colors[prediction], -1, 8, 0);
          // Populate Window with predictions
           if(prediction.compare(ent.first)==0){
             Correct += 1;
            rectangle(disVals, Rect(discols, disrows,cropsize,cropsize), Colors["Correct"], -1, 8, 0);
           }else if(prediction.compare("UnDefined")==0){
             Unknown +=1;
            rectangle(disVals, Rect(discols, disrows,cropsize,cropsize), Colors[prediction], -1, 8, 0);
           }else{
             Incorrect += 1;
           rectangle(disVals, Rect(discols, disrows,cropsize,cropsize), Colors["Incorrect"], -1, 8, 0);
           }

          // Save ROC data to results, clsAttempts starts at 0 so is -1
          cacheTestdata(ent.first, prediction, results);
          //          cout << "This was the high: " << high << " and second high: " << secHigh << "\n";
          discols +=cropsize;
        }
         rectangle(matchDisplay, Rect(0, 0,50,50), Colors[ent.first], -1, 8, 0);
         imshow("correct", matchDisplay);
         //         imshow("novelImg", ent.second[h]);
         imshow("segmentPredictions", disVals);
         waitKey(500);
      }

      // END OF CLASS, CONTINUING TO NEXT CLASS //
    }
}

void printRAWResults(map<string, vector<double> > r){
  cout << "\n\n----------------------------------------------------------\n\n";
  cout << "                    These are the test results                  \n";
  for(auto const ent6 : r){
    double TP=0, TN=0, FP=0, FN=0, PPV = 0, TPR = 0;
    TP = ent6.second[0];
    FP = ent6.second[1];
    TN = ent6.second[2];
    FN = ent6.second[3];
    PPV = (TP/(TP+FP));
    TPR = (TP/(TP+FN));
    cout << ent6.first;
    cout << "\n      TruePositive:  " << TP;
    cout << "\n      FalsePositive: " << FP;
    cout << "\n      TrueNegative:  " << TN;
    cout << "\n      FalseNegative: " << FN;
    cout << "\n      PPV:           " << PPV;
    cout << "\n      TPR:           " << TPR;
    cout << "\n";
  }
  cout << "\n\n";
}

void novelImgHandle(path testPath, path clsPath, double scale, int cropsize, int numClusters, int DictSize){
    // Load Images to be tested
    map<string, vector<Mat> > classImages;
    loadClassImgs(testPath, classImages, scale);

    map<string, vector<Mat> > savedClassHist;
    int serial;
    serial = getClassHist(savedClassHist);

    // Stock Scalar Colors
    map<string, Scalar> Colors;
      vector<Scalar> clsColor;
        clsColor.push_back(Scalar(255,0,0)); // Blue
        clsColor.push_back(Scalar(255,128,0)); // Orange
        clsColor.push_back(Scalar(255,255,0)); // Yellow
        clsColor.push_back(Scalar(0,255,255)); // Turquoise
        clsColor.push_back(Scalar(127,0,255)); // DarkPurple
        clsColor.push_back(Scalar(255,0,255)); // Purple
        clsColor.push_back(Scalar(255,0,127)); // Pink/Red

      Colors["Correct"] = Scalar(0,255,0); // Green
      Colors["Incorrect"] = Scalar(0,0,255); // Red
      Colors["Unknown"] = Scalar(100,100,100); // Gray

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
    vector<map<string, vector<double> > > results;

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

  vector<string> clsNames;
  clsNames.push_back("UnDefined");
  getUniqueClassNme(clsPath, clsNames);
  printClasses(clsNames);
  int counter =0;
  // For loop to get data while varying an input parameter stored as a for condition
  // for(int numClusters=7;numClusters<8;numClusters++){
    initROCcnt(results, clsNames); // Initilse map
    int clsAttempts = 5;
    testNovelImg(clsAttempts, numClusters, results[counter], classImages, savedClassHist, Colors, cropsize);
  counter++;
  //  }
  printRAWResults(results[0]);
  saveTestData(results, serial);

  map<string, vector<vector<double> > > resByCls;
  map<string, vector<vector<double> > > ROCVals;
  vector<string> clsNmes;
  getClsNames(results[0], clsNmes); // Get class Names
  organiseResultByClass(results, resByCls, clsNmes);

  calcROCVals(resByCls, ROCVals, clsNmes);


  if(results.size()>1){
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
        if(i==0){
          line(rocCurve, Point(((wW-buffer)*ent.second[i][1]+50),((wW-buffer)*ent.second[i][0]+50)),
          Point(((wW-buffer)*ent.second[i+1][1]+50),((wW-buffer)*ent.second[i+1][0])+50),
          Scalar(255,255,255), 1, 8, 0);
        }else{
          line(rocCurve, Point(((wW-buffer)*ent.second[i][1]+50),((wW-buffer)*ent.second[i][0]+50)),
          Point(((wW-buffer)*ent.second[i+1][1]+50),((wW-buffer)*ent.second[i+1][0])+50),
          Colors[ent.first], 1, 8, 0);
        }
        waitKey(500);
        imshow("ROC_Curve", rocCurve);
        }
    }
  }else{
    cout << "\n\nThere are not enough iterations to produce a ROC graph. Exiting." << endl;
  }
}
