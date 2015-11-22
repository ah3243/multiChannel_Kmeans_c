#include "opencv2/highgui/highgui.hpp" // Needed for HistCalc
#include "opencv2/imgproc/imgproc.hpp" // Needed for HistCalc
#include <opencv2/opencv.hpp>
#include "opencv2/core/core.hpp"
#include <iostream> // General io
#include <stdio.h> // General io
#include <stdlib.h>
#include <fstream>
#include <string>
#include <unistd.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <boost/algorithm/string.hpp> // Maybe fix DescriptorExtractor doesn't have a member 'create'
#include <boost/filesystem.hpp>
#include <assert.h>
#include <chrono>  // time measurement
#include <thread>  // time measurement
#include <map>
#include <numeric> // For getting average texton distance

#include "filterbank.h" // Filterbank Handling Functions
#include "imgCollection.h" // Img Handling Functions
#include "imgFunctions.h" // Img Processing Functions

#define ERR(msg) printf("\n\nERROR!: %s Line %d\nExiting.\n\n", msg, __LINE__);

using namespace boost::filesystem;
using namespace cv;
using namespace std;

#define CHISQU_MAX_threshold 6
#define CHISQU_DIS_threshold 0

// For offsetting segment, MUST === IMGFUNCTION VALUES!!
#define COLSTART 0
#define ROWSTART 0

// Results Display Flags
#define VERBOSE 0
#define SAVEIMGS 0
#define SAVEPREDICTIONS 1
#define PRINT_HISTRESULTS 0
#define SHOW_PREDICTIONS 1
#define PRINT_RAWRESULTS 0
#define PRINT_CONFUSIONMATRIX 0
#define PRINT_CONFMATAVG 0
#define PRINT_TPRPPV 0
#define PRINT_AVG 0
#define PRINT_COLOR 0

#define MULTITHREAD 0

#define a1 map<string, int>
#define a2 map<string, vector<double> >

////////////////////////
// Key:               //
// 0 == TruePositive  //
// 1 == FalsePositive //
// 2 == TrueNegative  //
// 3 == FalseNegative //
////////////////////////
void addTrueNegatives(string exp1, string exp2, map<string, vector<double> >& res){
  for(auto ent7 : res){
    if(exp1.compare(ent7.first)!=0 && exp2.compare(ent7.first)!=0 && ent7.first.compare("UnDefined")!=0){
      string holder = ent7.first;
      res[holder][2] += 1;
    }
  }
}

void initROCcnt(vector<map<string, vector<double> > >& r, vector<string> clsNames){
  if(VERBOSE){
    cout << "initialising.. " << endl;
  }

  int possResults = 4; // allow space for TP, FP, TN, FN

  map<string, vector<double> > a;
  for(int i=0;i<clsNames.size();i++){
    for(int j=0;j<possResults;j++){
      a[clsNames[i]].push_back(0.0);
    }
    if(VERBOSE){
    cout << "Initilsing..." << clsNames[i] << endl;
    }
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
  }else if(prediction.compare("UnDefined")==0){
    // Add one false positive for to UnDefined and one false negative to actual class
    // No true negatives
    results[prediction][1]+=1;
    results[correct][3]+=1;
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

// Go through input vector from multiple tests and output concatenated results for each class
void organiseResultByClass(vector<map<string, vector<double> > >in, map<string, vector<vector<double> > > &out, vector<string> clsNmes){
  int numTests = in.size();
  vector<double> a;
  cout << "Number of tests: " << in.size() << endl;
  // Go through each class
  for(int i=0;i<clsNmes.size();i++){
    // Go through the 4 types of result for each class
    for(int j=0;j<4;j++){
      // Go through each test and push back results to relevant class
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
void calcROCVals(map<string, vector<vector<double> > > in, map<string, vector<vector<double> > >& out, vector<string> clssNmes, vector<string> testClsNmes){
  assert(clssNmes.size() == in.size());
  vector<double> a;
  double tTP=0, tTN=0, tFP=0, tFN=0;

  // Go through each Class
  for(int i=0;i<clssNmes.size();i++){
    string curCls = clssNmes[i];
    if(PRINT_TPRPPV){
      cout << "Class: " << curCls << "\n\n";
    }

    // Go through each test iteration
    for(int j=0;j<in[curCls][0].size();j++){
      double TPR, PPV, TP, FN, TN, FP;
      TP = in[curCls][0][j];
      FP = in[curCls][1][j];
      TN = in[curCls][2][j];
      FN = in[curCls][3][j];

      // aggregate results in totals
      tTP += TP;
      tFP += FP;
      tFN += FN;

      // Add one to aggregated true negative if class had test image and True positive was 1
      for(int z=0;z<testClsNmes.size();z++){
        if(curCls.compare(testClsNmes[z])==0){
          tTN += TN;

          // Calculate TPR
          TPR = (TP/(TP+FN));
          if(PRINT_TPRPPV){
            cout << " " << j << ":-  TP: " << TP << " FN: " << FN << " TPR: " << TPR;
          }

          // Calculate FPR
          PPV = (TP/(TP+FP));
          if(PRINT_TPRPPV){
            cout << " TP: " << TP << " FP: " << FP << " PPV: " << PPV;
          }

          // Pushback vector for each test
          out[curCls].push_back(a);
          // Push back results
          out[curCls][j].push_back(TPR);
          out[curCls][j].push_back(PPV);
          if(PRINT_TPRPPV){
            cout << "\naccuracy: " << ((TP)/(TP+FP))*100 << "\%" << "\n\n";
          }
        }
       }
      }
  }
  double tPPV, tTPR, tFScore;
  tPPV = (tTP/(tTP+tFP));
  tTPR = (tTP/(tTP+tFN));
  tFScore = 2*((tPPV*tTPR)/(tPPV+tTPR));
  cout << "These are the values: tTp:" << tTP << " tTN: " << tTN << " tFP: " << tFP << " tFN: " << tFN << endl;
  cout << "PPV: " << tPPV << " TPR: " << tTPR << endl;
  cout << "The Combined accuracy is: " << tFScore << endl;

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
    for(const auto entS : r[i]){
      fs << entS.first << "{";
        double TP, FP, TN, FN, PPV, TPR;
        TP = entS.second[0];
        FP = entS.second[1];
        TN = entS.second[2];
        FN = entS.second[3];
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

void getColorData(map<string, vector<double> >&saveColors){
  // Load in Class Histograms(Models)
  FileStorage fs3("models.xml", FileStorage::READ);
  FileNode fn = fs3["classes"];
  if(fn.type() == FileNode::MAP){

    // Create iterator to go through all the classes
    for(FileNodeIterator it = fn.begin();it != fn.end();it++){
      string clsNme = (string)(*it)["Name"];
      double blue = (double)(*it)["blue"];
      double green = (double)(*it)["green"];
      double red = (double)(*it)["red"];
      saveColors[clsNme].push_back(blue);
      saveColors[clsNme].push_back(green);
      saveColors[clsNme].push_back(red);
    }
    fs3.release();
  }else{
    ERR("Class file was not map.");
    exit(-1);
  }
}

void printConfMat(std::map<std::string, std::map<std::string, int> > in){
  cout << "\nprinting confusion matrix:\n\n";

  // Print out names for exporting to excel
  cout << "classes ";
  for(auto const ent1:in){
    for(auto const ent2 : ent1.second){
      cout << " : " << ent2.first;
    }
    cout << "\n";
    break;
  }
  // Pring out number of hits per class
  for(auto const entP : in){
    cout << entP.first << " Hits";
    for(auto const entP1 : entP.second){
      cout << " : " << entP1.second;
    }
    cout << "\n";
  }
  cout << "\n";
}

bool pairCompare(const pair<string, double>& firstElem, const pair<string, double>& secondElem) {
  return firstElem.second < secondElem.second;
}

void qSegment(Mat in, vector<Mat> &out, int cropsize, int MISSTOPLEFT_RIGHT){

  // Calculate the maximum number of segments for the given img+cropsize
  int NumColSegments = (in.cols/cropsize);
  int NumRowSegments = (in.rows/cropsize);
  int NumSegmentsTotal = (NumColSegments*NumRowSegments);

  // Calculate the gap around the combined segments
  int colspace = (in.cols -(NumColSegments*cropsize))/2;
  int rowspace = (in.rows -(NumRowSegments*cropsize))/2;

  int segmentCounter=0;
  for(int i=colspace;i<(in.cols-cropsize);i+=cropsize){
    for(int j=rowspace;j<(in.rows-cropsize);j+=cropsize){
      if(MISSTOPLEFT_RIGHT&&(segmentCounter==0||segmentCounter==4)&&NumSegmentsTotal==6){
        segmentCounter++;
        continue;
      }
      Mat tmp = Mat::zeros(cropsize,cropsize,CV_32FC1);
      tmp = in(Rect(i, j, cropsize, cropsize));
      out.push_back(tmp);
      segmentCounter++;
    }
  }
}

double testNovelImg(int clsAttempts, int numClusters, map<string, vector<double> >& results, map<string, vector<Mat> > testImgs,
                  map<string, vector<Mat> > savedClassHist, map<string, Scalar> Colors, int cropsize, int flags, int kmeansIteration,
                  double kmeansEpsilon, int overlap, string folderName, int testRepeats){

  auto novelStart = std::chrono::high_resolution_clock::now();
  double fpsTotal = 0, frameCount = 0, fpsAvg=0;

  double acc;// Accuracy
  map<string, vector<double> > avgResults;// Used to collate histogram distance results for easy printing/exporting

  // Extract and store saved color data from model.xml
  map<string, vector<double> > saveColors;
  getColorData(saveColors);
  vector<string> Clsnmes;
  for(auto const la : savedClassHist){
    Clsnmes.push_back(la.first);
  }
  map<string, a1 > confMat;

  TermCriteria clsTc(TermCriteria::MAX_ITER, kmeansIteration, kmeansEpsilon);

  if(SHOW_PREDICTIONS){
    // Window for segment prediction display
    namedWindow("segmentPredictions", CV_WINDOW_AUTOSIZE);
    namedWindow("novelImg", CV_WINDOW_AUTOSIZE);
    namedWindow("correct", CV_WINDOW_AUTOSIZE);
    namedWindow("frameCounter", CV_WINDOW_AUTOSIZE);

    // Roughly place windows
    moveWindow("segmentPredictions", 100,400);
    moveWindow("novelImg", 400,400);
    moveWindow("correct", 300,250);
    moveWindow("frameCounter", 350,100);
  }

  // Import Texton Dictionary
  Mat dictionary;
  vector<float> m;
  getDictionary(dictionary, m);
  float bins[m.size()];
  vecToArr(m, bins);

  // Initilse Histogram parameters
  bool uniform = false;
  bool accumulate = false;

  // Store aggregated correct, incorrect and unknown results for segment prediction display
  double Correct = 0, Incorrect =0, Unknown =0, trueNegative=0;
  map<string, double> clsTextonDistances;
    // Loop through All Classes
    for(auto const entx : testImgs){
      confMat[entx.first];
      vector<double> textonDistance; // Stores aggregated distance between textons and original centres

      // Initilse confusion matrix with all classes and 'Unknown'
      for(int q =0;q<Clsnmes.size();q++){
        confMat[entx.first][Clsnmes[q]]=0;
      } confMat[entx.first]["UnDefined"] = 0;

      // Loop through all images in Class
      if(VERBOSE)
        cout << "\n\nEntering Class: " << entx.first << endl;

      for(int h=0;h < entx.second.size();h++){
        vector<double> texDistance;

      // Display current Frame/Image Count
        stringstream ss;
        ss << "Frame: " << h;
        Size textsize = getTextSize(ss.str(), FONT_HERSHEY_SCRIPT_SIMPLEX, 2, 1, 0);
        double frameW_H = textsize.height+20;
        double frameW_W = textsize.width+30;
        Mat frame = Mat(frameW_H, frameW_W, CV_8UC3, Scalar(255,255,255));
        Point org(((frameW_W -textsize.width)), (frameW_H - textsize.height)+20);
        putText(frame, ss.str(), org, CV_FONT_HERSHEY_COMPLEX, 1, Scalar(0, 0, 0), 2, 8 );
        imshow("frameCounter", frame);

        auto fpsStart = std::chrono::high_resolution_clock::now();
        Mat in = Mat(entx.second[h].cols, entx.second[h].rows,CV_32F, Scalar(0));
        Mat hold = Mat(entx.second[h].cols, entx.second[h].rows,CV_32FC1,Scalar(0));
        in = entx.second[h].clone();
         if(in.empty()){
          ERR("Novel image was not able to be imported.");
          exit(-1);
        }
        // segment color image and store in vector
        vector<Mat> colorTest;

        // Disregard top right and top left segments from a 6 segment crop
        int MISSTOPLEFT_RIGHT = 1;

        vector<vector<Mat> > filterbank;
        int n_sigmas, n_orientations;

        createFilterbank(filterbank, n_sigmas, n_orientations);
        // Send img to be filtered, and responses aggregated with addWeighted
        filterHandle(in, hold, filterbank, n_sigmas, n_orientations);
        // Divide the image into segments specified in 'cropsize' and flatten for clustering
        qSegment(in, colorTest, cropsize, MISSTOPLEFT_RIGHT);
        vector<Mat> test;
        segmentImg(test, hold, cropsize, overlap, MISSTOPLEFT_RIGHT);

        // Calculate the maximum number of segments for the given img+cropsize
        int NumColSegments = (in.cols/cropsize);
        int NumRowSegments = (in.rows/cropsize);
        int NumSegmentsTotal = (NumColSegments*NumRowSegments);

        // Calculate the gap around the combined segments
        int colspace = (in.cols -(NumColSegments*cropsize))/2;
        int rowspace = (in.rows -(NumRowSegments*cropsize))/2;
        //
        // // find the number of possible segments, then calculate gap around these
        // int NumColSegments
        // int colspace = (in.cols -((in.cols/cropsize)*cropsize))/2;
        // int rowspace = (in.rows -((in.rows/cropsize)*cropsize))/2;

        // Calculate the space around the cropped segments
        Mat disVals = Mat(in.size(),CV_8UC3, Scalar(255,255,255));
        Mat matchDisplay = Mat(50,50,CV_8UC3, Scalar(0,0,0));

        // Counters for putting 'pixels' on display image
        int disrows = rowspace, discols = colspace;

        // Loop through and classify all image segments
        for(int x=0;x<test.size();x++){

          // -- VISULAISATION VARS -- //
          int imgSize = in.cols;
          // If using reduced segments, skip top left and top right
          if(MISSTOPLEFT_RIGHT && (x==0||x==1)&& NumSegmentsTotal==6){
            discols += cropsize;
          }
          // If discols is larger than the image then move to next row
          if(discols>imgSize-cropsize){
            discols=colspace;
            disrows += cropsize;
          }

          // ----- TEST REPEATS ----- //

          map<string, double> matchResults;
          map<string, vector<double> > testAvgs; // map to hold each classes best matches for single segment over several repeat clusterings

          // Re cluster image segment this number of times averaging the best results
          int numTstRepeats =testRepeats;

          if(MULTITHREAD){
            // #pragma omp parallel for num_threads(2) shared(testAvgs, matchResults, test, numClusters, clsTc, clsAttempts, flags, numTstRepeats, dictionary, m, bins, savedClassHist)
          }
          for(int tstAVG=0;tstAVG<numTstRepeats;tstAVG++){
            // // verify the same number of color and normal segmented images
            // assert(test.size() == colorTest.size());

            // -- CLUSTERING/TEXMATCHING -- //
            BOWKMeansTrainer novelTrainer(numClusters, clsTc, clsAttempts, flags);
            // If segment is not empty add to novelimgTrainer
            if(!test[x].empty()){
               novelTrainer.add(test[x]);
            }
             // Generate specified number of clusters per segment and store in Mat
            Mat tmpClus = Mat::zeros(numClusters,1, CV_32FC1);
            tmpClus = novelTrainer.cluster();
            Mat clus = tmpClus.clone();

            vector<double> textonDistances;
             // Replace Cluster Centers with the closest matching texton
             textonFind(clus, dictionary, textonDistances);

             // -- HIST CALC -- //
             Mat out1;
             int histSize = m.size()-1;
             const float* histRange = {bins};
             calcHist(&clus, 1, 0, Mat(), out1, 1, &histSize, &histRange, false, false);

             // -- HIST MATCH -- //
             double high = DBL_MAX, secHigh = DBL_MAX, clsHigh = DBL_MAX;
             string match, secMatch;
             // Compare against each classes saved Histogram
             for(auto const ent2 : savedClassHist){
               matchResults[ent2.first] = DBL_MAX;
               vector<double> tmpVec;
               // Compare against all saved Histograms from a single class
               for(int j=0;j < ent2.second.size();j++){
                 Mat tmpHist = ent2.second[j].clone();
                 double val = compareHist(out1,tmpHist,CV_COMP_CHISQR);
                 tmpVec.push_back(val);
               }
              sort(tmpVec.begin(), tmpVec.end());
             testAvgs[ent2.first].push_back(tmpVec[0]); // Push back the top match for each class
             }
           } // End of test image reclustering

           ///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
          // The depth of averaging for values
          ///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

          double avgDepth =numTstRepeats;
          vector<pair<string, double>> avg;
          // Get the average of the top n values for each class store in sorted vector
          for(auto const ent9:testAvgs){
            double avgTot=0.0;
            for(int w=0;w<avgDepth;w++){
              avgTot += ent9.second[w];
            }
            pair<string, double> tmp(ent9.first,avgTot/avgDepth);
            avg.push_back(tmp);
            matchResults[ent9.first]=tmp.second;
          }
          // Sort averaged values
          sort(avg.begin(), avg.end(), pairCompare);
          if(PRINT_AVG){
            cout << entx.first << " averages: ";
            for(int z=0;z<avg.size();z++){
              avgResults[avg[z].first].push_back(avg[z].second);  // save all results by class for easy printing later and exporting
              cout << avg[z].first << " : " << avg[z].second << endl;
            }
          }
           string prediction = "";

          //  // -- COLOR -- //
          //  // Take the average bgr values for croppsed patch
          //  Scalar segColor = mean(colorTest[x]);
          //  double tb=0, tg=0, tr=0;
          //  tb = segColor[0];
          //  tg = segColor[1];
          //  tr = segColor[2];
           //
          //  string colorMatch;
          //  double distance = DBL_MAX, Bclear =0, Gclear = 0, Rclear =0;
          //  for(auto const amc:saveColors){
          //    double tmpBDis =0, tmpGDis=0, tmpRDis=0;
          //     tmpBDis =  abs(tb - amc.second[0]); // Blue
          //     tmpGDis =  abs(tg - amc.second[1]); // Green
          //     tmpRDis =  abs(tr - amc.second[2]); // Red
          //     double tmpDis = abs(tmpRDis+tmpGDis+tmpBDis);
          //     if(tmpDis<distance){
          //       Bclear = abs(distance-tmpBDis);
          //       Gclear = abs(distance-tmpGDis);
          //       Rclear = abs(distance-tmpRDis);
          //       distance=tmpDis;
          //       colorMatch = amc.first;
          //     }
          //  }
          //  if(PRINT_COLOR){
          //   cout << "This is the distance: " << distance << " and match: " << colorMatch << " with a clearance of Bclear: " << Bclear << " Gclear: " << Gclear << " Rclear: " << Rclear << endl;
          //  }

           // If match above threshold or to close to other match or all values are identical Class as 'UnDefined'
           if((avg[0].second>CHISQU_MAX_threshold || (avg[1].second - avg[0].second)<CHISQU_DIS_threshold || avg[1].second==DBL_MAX) && avg[0].second>0){
             prediction = "UnDefined";
           }
          else{
            prediction = avg[0].first;
          }
          confMat[entx.first][prediction]+=1;

          rectangle(disVals, Rect(discols, disrows,cropsize,cropsize), Colors[prediction], -1, 8, 0);

          // Save ROC data to results, clsAttempts starts at 0 so is -1
          cacheTestdata(entx.first, prediction, results);

          // Increment segment for visualusation
          discols +=cropsize;
        }
        // Blend predictions with origianl image
        Mat disVals1;
        addWeighted(in, 0.7, disVals, 0.3, 0.0, disVals1, -1);
        rectangle(matchDisplay, Rect(0, 0,50,50), Colors[entx.first], -1, 8, 0);

         if(SAVEPREDICTIONS){
           stringstream ss;

           // Ensure correct Dirs Exist
           ss << "./Predictions/" << folderName;
           assert(createDir("Predictions"));
           assert(createDir(ss.str()));

           ss.str(""); // Clear stringstream
           // Save prediction images
           ss  << "./Predictions/" << folderName << "/" << entx.first << "_" << frameCount << ".png";
           cout << "Saving: " << ss.str() << endl;
           imwrite(ss.str(),disVals1);
           ss.str(""); // Clear stringstream
         }
         if(SHOW_PREDICTIONS){
           imshow("correct", matchDisplay);
           imshow("novelImg", entx.second[h]);
           imshow("segmentPredictions", disVals1);
           waitKey(30);
          }
         auto fpsEnd = std::chrono::high_resolution_clock::now();
         fpsTotal+= std::chrono::duration_cast<std::chrono::milliseconds>(fpsEnd - fpsStart).count();
         frameCount++;
      }
  //     cout << "Correct Was: " << Correct << " Incorrect was: " << Incorrect << " Unknown was: " << Unknown << endl;
  //     cout << "This is trueNegative: " << trueNegative <<" and total: " << (Incorrect+Correct+trueNegative+Unknown) << " average is: " << ((Correct+trueNegative)/(Incorrect+Correct+trueNegative+Unknown))*100 <<  endl;
      // END OF CLASS, CONTINUING TO NEXT CLASS //

      // Aggregate texton distances and store with class name
      double tmpDistance=0;
      int tmpcounter=0;
      for(tmpcounter=0;tmpcounter<textonDistance.size();tmpcounter++){
        tmpDistance+=textonDistance[tmpcounter];
      }
      clsTextonDistances[entx.first]=(tmpDistance/tmpcounter);
    }


    // -- TIME -- //
    int novelTime=0;
    auto novelEnd = std::chrono::high_resolution_clock::now();
    novelTime = std::chrono::duration_cast<std::chrono::milliseconds>(novelEnd - novelStart).count();
    cout << "\nTotal Frames: " << frameCount << " total Time: " << fpsTotal << " Averaged frames per second: " << frameCount/(fpsTotal/1000) << "\n";
    cout << "novelTime: " << novelTime << endl;



    if(PRINT_CONFUSIONMATRIX){
      printConfMat(confMat);
      //acc = ((Correct+trueNegative)/(Correct+trueNegative+Unknown+));
    }

    cout << "\n";

    // Print out collated Histogram distances
    if(PRINT_AVG){
      // Go through all classes
      for(auto const avgRes : avgResults){
        cout << avgRes.first << ":"; // Print current class
        // Go through all class results
        for(int avgResVec=0;avgResVec<avgRes.second.size();avgResVec++){
          cout << avgRes.second[avgResVec];
          // If there will be another result add semi colon for easy exporting
          if(avgResVec<avgRes.second.size()-1){
            cout << ":";
          }
        }
        cout << "\n";
      }
    }

    return acc;
}

void printRAWResults(vector<map<string, vector<double> > > r){
  cout << "\n\n----------------------------------------------------------\n\n";
  cout << "                    These are the test results                  \n";
  cout << "Number of tests: " << r.size() << endl;

  for(int numTests =0;numTests<r.size();numTests++){
    cout << ".....Iteration Number.... " << numTests << endl;
    for(auto const ent6 : r[numTests]){
      double TP=0, TN=0, FP=0, FN=0, PPV = 0, TPR = 0;
      TP = ent6.second[0];
      FP = ent6.second[1];
      TN = ent6.second[2];
      FN = ent6.second[3];
      PPV = (TP/(TP+FP));
      TPR = (TP/(TP+FN));
     cout << "0 : "<< ent6.first;
      cout << "\n      TruePositive:  " << TP;
      cout << "\n      FalsePositive: " << FP;
      cout << "\n      TrueNegative:  " << TN;
      cout << "\n      FalseNegative: " << FN;
      cout << "\n      PPV:           " << PPV;
      cout << "\n      TPR:           " << TPR;
    }
    cout << "\n";
  }
}

void loadVideo(string p, map<string, vector<Mat> > &testImages, int scale){
  cout << "Loading test video" << endl;
  VideoCapture stream;
  stream.open(p);
  if(!stream.isOpened()){
    ERR("Video Stream unable to be opened.");
    exit(1);
  }

  // Validate video input size/ratio
  int vH = stream.get(CV_CAP_PROP_FRAME_HEIGHT);
  int vW = stream.get(CV_CAP_PROP_FRAME_WIDTH);

  if((vW/vH)!= 1280/720){
    ERR("The imported video does not have an aspect ratio of 16:9.");
    exit(-1);
  }
  else if(vH<720|| vW<1280){
    ERR("The imported video was below the minimum resolution of 1280X720.");
    exit(-1);
  }

  vector<string> name;
  extractClsNme(p);

  for(int i=0;i<stream.get(CV_CAP_PROP_FRAME_COUNT);i++){
    Mat in = Mat::zeros(vW, vH, CV_8UC3);
    Mat gray, tmp1, tmp2;
    stream >> in;
    cvtColor(in, gray, CV_BGR2GRAY);
    equalizeHist(gray, tmp1);
    scaleImg(tmp1, tmp2, scale);
    testImages[p].push_back(tmp2);
  }
  cout << "Video Loaded, this is the frame count: " << stream.get(CV_CAP_PROP_FRAME_COUNT) << endl;
  cout << "This is the width of each frame: " << stream.get(CV_CAP_PROP_FRAME_WIDTH) << " and height: " << stream.get(CV_CAP_PROP_FRAME_HEIGHT) << endl;
}

void printFiles(map<string, vector<string> > s, vector<string> &fileNmes, path p){
  cout << "These are the current video files in the directory.\n";
  int count=0;
  for(auto const ent1 : s){
    for(int i=0;i<ent1.second.size();i++){
      stringstream ss;
      cout << count << " " << ent1.second[i] << "\n";
      ss << p.string() << ent1.second[i] << ".mp4";
      fileNmes.push_back(ss.str());
      count++;
    }
  }
}

string getfileNme(vector<string> s){
  cout << "\nPlease enter the number next to the video you want to load.\n";
  int num;
  char b[256];
  cin >> b;
  num = atoi(b);
  cout << "here.." << num << endl;
  if(num>s.size()){
    cout << "That number was out of range please try again.\n";
  }else{
    cout << "This is the name: " << s[num] << endl;
    return s[num];
  }
}

  // // Calculate the average per bin per class
  // void textonDictMetric(map<string, vector<Mat> > models){
  //   cout << "\n\nBelow are the average per bin per class models: \n";
  //   for(auto const ent0 : models){
  //     cout << ent0.first << " "
  //   }
  // }
  double vecAccumulator(vector<double> in){
    double out;
    for(vector<double>::iterator j=in.begin() ;j!=in.end();++j){
      out += *j;
    }
    return out;
  }
void printStart(string filler, int iterations, bool flag){
  if(flag){
    cout << "0 :";
  }
  for(int o=0;o<iterations;o++){
    cout << filler;
    if(o<iterations-1){
      cout << ":";
    }
  }
}
void printVector(vector<double> hh, bool flag){
  for(int jq=0;jq<hh.size();jq++){
    if(flag){
      cout << ":";
    }
    cout << hh[jq];
  }
}
void printPPVTPR(string filler, vector<double> TP, vector<double> H, bool flag){
  for(int jq=0;jq<H.size();jq++){
    // Add seperating ':' if not first value
    if(jq!=0){
        cout << ":";
    }
    cout << (TP[jq]/(TP[jq]+H[jq]));
  }
}
void getclssPPVTPR(string filler, vector<double> TP, vector<double> FP, vector<double> FN, vector<double> &PPVvec, vector<double> &TPRvec){
  assert(TP.size()==FP.size()&& TP.size()==FN.size());
  double aggTP, aggFP, aggFN;

  for(int jq=0;jq<TP.size();jq++){
    aggTP+=TP[jq];
    aggFP+=FP[jq];
    aggFN+=FN[jq];
  }
    double PPV, TPR;
    PPV = (aggTP/(aggTP+aggFP));
    TPR = (aggTP/(aggTP+aggFN));

    PPVvec.push_back(PPV);
    TPRvec.push_back(TPR);
}
double printmicroFScore(double TP, double FP, double FN){
  double PPV, TPR, FScore;
  PPV = (TP/(TP+FP));
  TPR = (TP/(TP+FN));
  FScore=(2*((PPV*TPR)/(PPV+TPR)));
  return FScore;
}
// Take in PPV and TPR results for all classes, take the mean and calc macro FScore
double printmacroFScore(vector<double> PPV, vector<double> TPR){
  int PPVSize, TPRSize;
  double avgPPV, avgTPR, totPPV, totTPR, FScore;

  //Get the number of values
  PPVSize = PPV.size();
  TPRSize = TPR.size();
  // Make sure same number for TPR and PPV
  assert(PPVSize==TPRSize);
  // Sum all TPR and PPV values
  totPPV = vecAccumulator(PPV);
  totTPR = vecAccumulator(TPR);
  avgPPV = (totPPV/PPVSize);
  avgTPR = (totTPR/TPRSize);
  FScore=(2*((avgPPV*avgTPR)/(avgPPV+avgTPR)));

  // cout << "This is the average PPV: " << avgPPV << " and TPR: " << avgTPR << endl;
  return FScore;
}

// Print out all test results
void printResByClss(map<string, vector<vector<double> > > in, bool firstGo){
  // For holding all class PPV and TPR's
  vector<double> avgPPV, avgTPR;
  vector<string> clsNames;
  // Calculate and store PPV and TPR values for each class, store corresponding class Name
  for(auto iter:in){
      if(iter.first.compare("UnDefined")!=0){
        vector<double> TP(iter.second[0]);
        vector<double> FP(iter.second[1]);
        vector<double> FN(iter.second[3]);
        clsNames.push_back(iter.first);
        getclssPPVTPR("FScore",TP, FP, FN, avgPPV, avgTPR);
      }
  }
  // assert the number of classes matches the number of PPV and TPR results
  assert(clsNames.size() == avgPPV.size() && clsNames.size()==avgTPR.size());

  // The aggregated TP, FP and FN for micro FScore
  double microTP, microFP, microFN;
  // Aggregate TP, FP and FN Values across all classes
  for(auto iter:in){
      if(iter.first.compare("UnDefined")!=0){
        microTP += vecAccumulator(iter.second[0]);
        microFP += vecAccumulator(iter.second[1]);
        microFN += vecAccumulator(iter.second[3]);
      }
  }
  // If first overall iteration print labels, else not
  if(firstGo){
    cout << "micro FScore: ";
  }
  cout << printmicroFScore(microTP, microFP, microFN);
  cout << "\n";

  // Print out Agg macroFScore
  // If first overall iteration print labels, else not
  if(firstGo){
    cout << "Macro FScore: ";
  }
  cout << printmacroFScore(avgPPV, avgTPR);
  cout << "\n\n";

  // Loop through all classes results
  for(auto iter:in){
        vector<double> TP(iter.second[0]);
        vector<double> FP(iter.second[1]);
        vector<double> TN(iter.second[2]);
        vector<double> FN(iter.second[3]);
        printStart(iter.first, iter.second[0].size(), firstGo);
        // Print out if first loop if not dont
        cout << "\n";
        if(firstGo){
          cout << "TruePositive";
        }
        printVector(TP, firstGo);
        cout << "\n";
        if(firstGo){
          cout << "FalsePositive";
        }
        printVector(FP, firstGo);
        cout << "\n";
        if(firstGo){
          cout << "TrueNegative";
        }
        printVector(TN, firstGo);
        cout << "\n";
        if(firstGo){
          cout << "FalseNegative";
        }
        printVector(FN, firstGo);
        cout << "\n";
  }
  cout << "\n";
  // Print out PPV
  for(auto iter:in){
      if(iter.first.compare("UnDefined")!=0){
        vector<double> TP(iter.second[0]);
        vector<double> FP(iter.second[1]);
        if(firstGo){
          cout << iter.first << ":";
        }
        printPPVTPR("PPV",TP, FP, firstGo);
        cout << "\n";
      }
  }
  // Print out TPR
  for(auto iter:in){
      if(iter.first.compare("UnDefined")!=0){
        vector<double> TP(iter.second[0]);
        vector<double> FN(iter.second[3]);
        if(firstGo){
          cout << iter.first << ":";
        }
        printPPVTPR("TPR",TP, FN, firstGo);
        cout << "\n";
      }
  }
  // Print out FScore for each class
  if(firstGo){
    cout << "FScore:";
  }
  cout << "\n";
  for(int f=0;f<clsNames.size();f++){
    if(firstGo){
      cout << clsNames[f] << ":";
    }
    cout << 2*((avgPPV[f]*avgTPR[f])/(avgPPV[f]+avgTPR[f])) << "\n";
  }
}
void novelImgHandle(path testPath, path clsPath, int scale, int cropsize, int numClusters,
  int DictSize, int flags, int attempts, int kmeansIteration, double kmeansEpsilon, int overlap, string folderName, bool firstGo, int testRepeats){
    auto novelHandleStart = std::chrono::high_resolution_clock::now();
    // Load Images to be tested
    path vPath = "../../../TEST_IMAGES/CapturedImgs/novelVideo/";
    map<string, vector<Mat> > testImages;

    // string s;
    // cout << "Would you like to analyse a video instead or Imgs? (enter Y or N).\n";
    // cin >> s;
    // boost::algorithm::to_lower(s);
    // if(s.compare("y")==0){

    // If no testRepeats assigned set to 1
    if(testRepeats==0){
      testRepeats=1;
    }
    cout << "\nNumber of TEST Repeats\n";

    // Imports and processes test video if used
    if(false){
      map<string, vector<string> > s1;
      vector<string> fileNmes;
      retnFileNmes(vPath,"", s1);
      printFiles(s1, fileNmes, vPath);
      string path = getfileNme(fileNmes);
      cout << "\nLoading Video.\n";
      loadVideo(path, testImages, scale);
    }else{
      cout << "\nLoading images.\n";
      loadClassImgs(testPath, testImages, scale);
    }
    vector<string> testClsNmes;
    // Load only test image classes into vector
    for(auto const a : testImages){
      testClsNmes.push_back(a.first);
    }

    map<string, vector<Mat> > savedClassHist;
    int serial;
    serial = getClassHist(savedClassHist);

    map<string, Scalar> Colors;
    vector<Scalar> clsColor;
    if(SHOW_PREDICTIONS||SAVEPREDICTIONS){
      // Stock Scalar Colors
          clsColor.push_back(Scalar(255,0,0)); // Blue
          clsColor.push_back(Scalar(0,128,255)); // Orange
          clsColor.push_back(Scalar(0,255,255)); // Yellow
          clsColor.push_back(Scalar(255,255,0)); // Turquoise
          clsColor.push_back(Scalar(127,20,255)); // Pink/Red
          clsColor.push_back(Scalar(255,0,127)); // DarkPurple
          clsColor.push_back(Scalar(255,0,255)); // Purple
          clsColor.push_back(Scalar(0,102,0)); // Dark Green
          clsColor.push_back(Scalar(102,0,102)); // Dark Purple

        Colors["Correct"] = Scalar(0,255,0); // Green
        Colors["Incorrect"] = Scalar(0,0,255); // Red
        Colors["Unknown"] = Scalar(100,100,100); // Gray

        int count =0;
        double txtWidth =0, txtHeight =0;
        for(auto const cols : savedClassHist){
          Colors[cols.first] = clsColor[count];
          int bseline;
          Size s = getTextSize(cols.first, CV_FONT_HERSHEY_SIMPLEX, 0.5, 1, &bseline);
          // Get txtLength
          if(s.width > txtWidth){
            txtWidth = s.width;
          }
          // Get txtHeight
          txtHeight += s.height;
          count++;
        }


      // Create Img Legend
      Mat Key = Mat::zeros(txtHeight+180,txtWidth+80,CV_8UC3);
        int cnt=0;
        for(auto const ent1 : Colors){
          putText(Key, ent1.first, Point(10, 20+ cnt*20), CV_FONT_HERSHEY_SIMPLEX, 0.5, Scalar(255,0,100), 1, 8, false);
          rectangle(Key, Rect(txtWidth+50, 10 + cnt*20, 10,10), ent1.second, -1, 8, 0 );
          cnt++;
        }

      // Window for Legend display
      namedWindow("legendWin", CV_WINDOW_AUTOSIZE);
      imshow("legendWin", Key);
      imwrite("./Predictions/Key.png", Key);
    }
  // Holds Class names, each holding a count for TP, FP, FN, FP Values
  vector<map<string, vector<double> > > results;

  vector<string> clsNames;
  clsNames.push_back("UnDefined");
  getUniqueClassNme(clsPath, clsNames);
  if(VERBOSE){
    printClasses(clsNames);
  }

  vector<double> acc;
  int numofRuns = 1;
  // For loop to get data while varying an input parameter stored as a for condition
  for(int kk=0;kk<numofRuns;kk++){
    cout << "This is iteration: " << kk+1 << " out of: " << numofRuns << endl;
    initROCcnt(results, clsNames); // Initilse map
    cout << "number of test Classes.." << testImages.size() << endl;
    acc.push_back(testNovelImg(attempts, numClusters, results[kk], testImages, savedClassHist, Colors, cropsize,
     flags, kmeansIteration, kmeansEpsilon, overlap, folderName, testRepeats));

   }

  if(PRINT_RAWRESULTS){
    printRAWResults(results);
  }
//  saveTestData(results, serial);

  map<string, vector<vector<double> > > resByCls;
  map<string, vector<vector<double> > > ROCVals;
  vector<string> clsNmes;
  getClsNames(results[0], clsNmes); // Get class Names
  organiseResultByClass(results, resByCls, clsNmes);

  // Print out all iterations results by class
  printResByClss(resByCls, firstGo);
//  calcROCVals(resByCls, ROCVals, clsNmes, testClsNmes);

  int novelHandleTime=0;
  auto novelHandleEnd = std::chrono::high_resolution_clock::now();
  novelHandleTime = std::chrono::duration_cast<std::chrono::milliseconds>(novelHandleEnd - novelHandleStart).count();
}
