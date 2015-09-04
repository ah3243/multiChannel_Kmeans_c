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
#include <boost/algorithm/string.hpp> // Maybe fix DescriptorExtractor doesn't have a member 'create'
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

#define CHISQU_MAX_threshold 6
#define CHISQU_DIS_threshold 1

// Results Display Flags
#define PRINT_HISTRESULTS 1
#define SHOW_PREDICTIONS 1
#define PRINT_RAWRESULTS 1
#define PRINT_CONFUSIONMATRIX 1
#define PRINT_TPRPPV 0
#define PRINT_AVG 1

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
    if(PRINT_TPRPPV)
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
      if(PRINT_TPRPPV)
        cout << " " << j << ":-  TP: " << TP << " FN: " << FN << " TPR: " << TPR;

      // Calculate FPR
      PPV = (TP/(TP+FP));
      if(PRINT_TPRPPV)
        cout << " TP: " << TP << " FP: " << FP << " PPV: " << PPV;

      // Pushback vector for each test
      out[curCls].push_back(a);
      // Push back results
      out[curCls][j].push_back(TPR);
      out[curCls][j].push_back(PPV);
      if(PRINT_TPRPPV)
        cout << "\naccuracy: " << ((TP)/(TP+FP+FN))*100 << "\n\n";
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
    ERR("Class file was not map.");
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

void printConfMat(std::map<std::string, std::map<std::string, int> > in){
  cout << "\nprinting confusion matrix:\n\n";
  cout << "0:0 ";
  for(auto const ent1:in){
    for(auto const ent2 : ent1.second){
      cout << " : " << ent2.first;
    }
    cout << "\n";
    break;
  }

  for(auto const entP : in){
    cout << "class :" << entP.first;
    for(auto const entP1 : entP.second){
      cout << " : " << entP1.second;
    }
    cout << "\n\n";
  }
  cout << "\n";
}

void calcNearestClasses(map<string, vector<map<string, vector<double> > > > results){
  cout << "This is the results size: " << results.size() << endl;
  cout << "these are the average scores for all classes across all segments:\n";

  map<string, map<string, vector<double> > >  avgs;

  vector<string> clsnames;

  for(auto const nmes : results){
    clsnames.push_back(nmes.first);
  }
  cout << "These are the names: " << endl;
  for(int i=0;i<clsnames.size();i++){
    cout << clsnames[i] << endl;
  }


  // Go through each classes segments
  for(auto const a:results){
    // Go through all segments from that class
    for(int i=0;i<a.second.size();i++){
      // Go through all classes which recorded distances from the image
      for(auto const b : a.second[i]){
        for(int j=0;j<b.second.size();j++)
          avgs[a.first][b.first].push_back(b.second[j]);
      }
    }
    cout << "\n";
  }

  for(auto const ae:avgs){
    cout << "Class: " << ae.first << " :: ";
    for(auto const b1:ae.second){
      int vecsize = b1.second.size();
      double q = 0.0;
      for(int i=0;i<vecsize;i++){
        q+= b1.second[i];
      }
 //     b1.second.clear();
      cout << b1.first << " : " << q/vecsize << ": ";
    }
    cout << "\n\n";
  }
  cout << "\n\nleaving calcNearestClasses" << endl;
}

double testNovelImg(int clsAttempts, int numClusters, map<string, vector<double> >& results, map<string, vector<Mat> > testImgs,
                  map<string, vector<Mat> > savedClassHist, map<string, Scalar> Colors, int cropsize, map<string, vector<map<string, vector<double> > > >& fullSegResults){
  auto novelStart = std::chrono::high_resolution_clock::now();
  double fpsTotal = 0, frameCount = 0, fpsAvg=0;

  double acc;// Accuracy

  vector<string> Clsnmes;
  for(auto const la : savedClassHist){
    Clsnmes.push_back(la.first);
  }

  map<string, a1 > confMat;

  int clsFlags = KMEANS_PP_CENTERS;
  TermCriteria clsTc(TermCriteria::MAX_ITER + TermCriteria::EPS, 1000000, 0.0000001);

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
  double Correct = 0, Incorrect =0, Unknown =0;
    // Loop through All Classes
    for(auto const entx : testImgs){
      confMat[entx.first];

      // Initilse confusion matrix with all classes and 'Unknown'
      for(int q =0;q<Clsnmes.size();q++){
        confMat[entx.first][Clsnmes[q]]=0;
      } confMat[entx.first]["UnDefined"] = 0;

      // Loop through all images in Class
      cout << "\n\nEntering Class: " << entx.first << endl;
      for(int h=0;h < entx.second.size();h++){
        BOWKMeansTrainer novelTrainer(numClusters, clsTc, clsAttempts, clsFlags);
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
        Mat in = Mat(entx.second[h].cols, entx.second[h].rows,CV_32FC1, Scalar(0));
        Mat hold = Mat(entx.second[h].cols, entx.second[h].rows,CV_32FC1,Scalar(0));
        in = entx.second[h].clone();
         if(in.empty()){
          ERR("Novel image was not able to be imported.");
          exit(-1);
        }

        vector<vector<Mat> > filterbank;
        int n_sigmas, n_orientations;
        createFilterbank(filterbank, n_sigmas, n_orientations);
        // Send img to be filtered, and responses aggregated with addWeighted
        filterHandle(in, hold, filterbank, n_sigmas, n_orientations);
        // Divide the image into segments specified in 'cropsize' and flatten for clustering
        vector<Mat> test;
        segmentImg(test, hold, cropsize);

        int imgSize = hold.cols;
        Mat disVals = Mat(hold.rows, hold.cols,CV_8UC3, Scalar(0,0,0));
        Mat matchDisplay = Mat(50,50,CV_8UC3, Scalar(0,0,0));

        // Counters for putting 'pixels' on display image
        int disrows = 0, discols = 0;
        // Loop through and classify all image segments
        for(int x=0;x<test.size();x++){
          // handle segment prediction printing
          if(discols>imgSize-cropsize){
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
           Mat out1, out2, lone, lone2;
           lone = clus.clone();
           int histSize = m.size()-1;
           const float* histRange = {bins};
           calcHist(&lone, 1, 0, Mat(), out1, 1, &histSize, &histRange, false, false);
           novelTrainer.clear();
           double high = DBL_MAX, secHigh = DBL_MAX, clsHigh = DBL_MAX;
           string match, secMatch;
           map<string, double> matchResults;

           a2 tmpVals;
           // Compare all saved histograms against novelimg
           for(auto const ent2 : savedClassHist){
             matchResults[ent2.first] = DBL_MAX;
             // Compare saved histgrams from each class
             vector<double> tmpVec;
             for(int j=0;j < ent2.second.size();j++){
               Mat tmpHist = ent2.second[j].clone();
              // DEBUGGING Loop
              //  if(ent2.first.compare("UnevenBrickWall")==0 && j ==0){
              //    cout << "this is hist Number: " << j << " and size: " << tmpHist.size() <<  endl;
              //     for(int ww=0;ww<tmpHist.rows;ww++){
              //       cout << "number: " << ww << ": " << tmpHist.at<float>(1, ww) << endl;
              //     }
              //     cout << "done\n";
              //     exit(1);
              //  }

               double val = compareHist(out1,tmpHist,CV_COMP_CHISQR);
               tmpVec.push_back(val);
             }
            sort(tmpVec.begin(), tmpVec.end());
            tmpVals[ent2.first] = tmpVec;
           }
          double avgDepth =3;
          double avg = DBL_MAX;
          for(auto const ent9:tmpVals){
            double avgTot=0.0, tmpavg=0.0;
            for(int w=0;w<avgDepth;w++){
              avgTot += ent9.second[w];
            }
            tmpavg  = avgTot/avgDepth;
            if(tmpavg < avg){
              avg=tmpavg;
              match = ent9.first;
            }
            matchResults[ent9.first]=tmpavg;
          }

          fullSegResults[entx.first].push_back(tmpVals);
           string prediction = "";

           // If match above threshold or to close to other match or all values are identical Class as 'UnDefined'
          //  if((high>CHISQU_MAX_threshold || (secHigh - high)<CHISQU_DIS_threshold || secHigh==DBL_MAX) && high>0){
          //    prediction = "UnDefined";
          //  }
          // else{
            prediction = match;
          // }
          confMat[entx.first][prediction]+=1;
          if(PRINT_HISTRESULTS){
            cout << "ACT: " << entx.first << " PD: " << prediction;
            for(auto const ag : matchResults){
              cout << ", " << ag.first << ": " << ag.second;
             }
            cout << "\n";
          }
          //cout << "First Prediction: " << match << " Actual: " << entx.first << " First Distance: " << high << " Second Class: " << secMatch << " Distance: " << secHigh << endl;
         rectangle(disVals, Rect(discols, disrows,cropsize,cropsize), Colors[prediction], -1, 8, 0);
          // Populate Window with predictions
           if(prediction.compare(entx.first)==0){
             Correct += 1;
            // rectangle(disVals, Rect(discols, disrows,cropsize,cropsize), Colors["Correct"], -1, 8, 0);
           }else if(prediction.compare("UnDefined")==0){
             Unknown +=1;
            // rectangle(disVals, Rect(discols, disrows,cropsize,cropsize), Colors[prediction], -1, 8, 0);
           }else{
             Incorrect += 1;
          //  rectangle(disVals, Rect(discols, disrows,cropsize,cropsize), Colors["Incorrect"], -1, 8, 0);
           }

          // Save ROC data to results, clsAttempts starts at 0 so is -1
          cacheTestdata(entx.first, prediction, results);
          discols +=cropsize;
        }
        if(SHOW_PREDICTIONS){
         rectangle(matchDisplay, Rect(0, 0,50,50), Colors[entx.first], -1, 8, 0);
         imshow("correct", matchDisplay);
         imshow("novelImg", entx.second[h]);
         imshow("segmentPredictions", disVals);
         waitKey(50);
       }
         auto fpsEnd = std::chrono::high_resolution_clock::now();
         fpsTotal+= std::chrono::duration_cast<std::chrono::milliseconds>(fpsEnd - fpsStart).count();
         frameCount++;
      }
       cout << "Correct Was: " << Correct << " Incorrect was: " << Incorrect << " Unknown was: " << Unknown << endl;
      acc = (Correct/(Incorrect+Unknown+Correct))*100;
      // END OF CLASS, CONTINUING TO NEXT CLASS //
    }
    int novelTime=0;
    auto novelEnd = std::chrono::high_resolution_clock::now();
    novelTime = std::chrono::duration_cast<std::chrono::milliseconds>(novelEnd - novelStart).count();
    cout << "\nTotal Frames: " << frameCount << " total Time: " << fpsTotal << " Averaged frames per second: " << frameCount/(fpsTotal/1000) << "\n";
    cout << "\n\n\nnovelTime: " << novelTime << endl;
    if(PRINT_CONFUSIONMATRIX){
      printConfMat(confMat);
    }
    return acc;
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
};

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

void novelImgHandle(path testPath, path clsPath, int scale, int cropsize, int numClusters, int DictSize){
    auto novelHandleStart = std::chrono::high_resolution_clock::now();
    // Load Images to be tested
    path vPath = "../../../TEST_IMAGES/CapturedImgs/novelVideo/";
    map<string, vector<Mat> > testImages;
    map<string, vector<map<string, vector<double> > > > fullSegResults;

    string s;
    cout << "Would you like to analyse a video instead or Imgs? (enter Y or N).\n";
    cin >> s;
    boost::algorithm::to_lower(s);
    if(s.compare("y")==0){
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

    map<string, vector<Mat> > savedClassHist;
    int serial;
    serial = getClassHist(savedClassHist);

    map<string, Scalar> Colors;
    vector<Scalar> clsColor;
    if(SHOW_PREDICTIONS){
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
    }
  // Holds Class names, each holding a count for TP, FP, FN, FP Values
  vector<map<string, vector<double> > > results;

  vector<string> clsNames;
  clsNames.push_back("UnDefined");
  getUniqueClassNme(clsPath, clsNames);
  printClasses(clsNames);

  vector<double> acc;
  int counter =0;
  // For loop to get data while varying an input parameter stored as a for condition
  // for(int numClusters=7;numClusters<8;numClusters++){
    initROCcnt(results, clsNames); // Initilse map
    int clsAttempts = 20;
    cout << "number of test images.." << testImages.size() << endl;
    acc.push_back(testNovelImg(clsAttempts, numClusters, results[counter], testImages, savedClassHist, Colors, cropsize, fullSegResults));
    cout << "this is the accuracy: " << acc[counter] << endl;
    counter++;
  //  }
  if(PRINT_RAWRESULTS){
    printRAWResults(results[0]);
  }
  saveTestData(results, serial);

  map<string, vector<vector<double> > > resByCls;
  map<string, vector<vector<double> > > ROCVals;
  vector<string> clsNmes;
  getClsNames(results[0], clsNmes); // Get class Names
  organiseResultByClass(results, resByCls, clsNmes);

  if(PRINT_AVG){
    calcNearestClasses(fullSegResults);
  }

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
    for(auto const entCls : ROCVals){
      cout << "Class: " << entCls.first << endl;
      // Cycle through all iterations to produce graph
      for(int i=0;i<entCls.second.size()-1;i++){
        if(i==0){
          line(rocCurve, Point(((wW-buffer)*entCls.second[i][1]+50),((wW-buffer)*entCls.second[i][0]+50)),
          Point(((wW-buffer)*entCls.second[i+1][1]+50),((wW-buffer)*entCls.second[i+1][0])+50),
          Scalar(255,255,255), 1, 8, 0);
        }else{
          line(rocCurve, Point(((wW-buffer)*entCls.second[i][1]+50),((wW-buffer)*entCls.second[i][0]+50)),
          Point(((wW-buffer)*entCls.second[i+1][1]+50),((wW-buffer)*entCls.second[i+1][0])+50),
          Colors[entCls.first], 1, 8, 0);
        }
        waitKey(500);
        imshow("ROC_Curve", rocCurve);
        }
    }
  }else{
    cout << "\n\nThere are not enough iterations to produce a ROC graph. Exiting." << endl;
  }
  cout << "\n\nThis is the accuracy.\n" << endl;
  for(int k=0;k<acc.size();k++){
    cout << "Iteration " << k << " Accuracy: " << acc[k] << "\%\n";
  }
  int novelHandleTime=0;
  auto novelHandleEnd = std::chrono::high_resolution_clock::now();
  novelHandleTime = std::chrono::duration_cast<std::chrono::milliseconds>(novelHandleEnd - novelHandleStart).count();
  cout << "\nnovelHandleTime: " << novelHandleTime << endl;
}
