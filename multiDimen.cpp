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

#define DICTIONARY_BUILD 0
#define MODEL_BUILD 0
#define NOVELIMG_TEST 1
#define ERR(msg) printf("\n\nERROR!: %s Line %d\nExiting.\n\n", msg, __LINE__);

using namespace boost::filesystem;
using namespace cv;
using namespace std;

#define cropsize  20
#define CHISQU_threshold 10


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

Mat reshapeCol(Mat in){
  Mat points(in.rows*in.cols, 1,CV_32F);
  int cnt = 0;
  for(int i =0;i<in.rows;i++){
    for(int j=0;j<in.cols;j++){
      points.at<float>(cnt, 0) = in.at<Vec3b>(i,j)[0];
      cnt++;
    }
  }
  return points;
}

void segmentImg(vector<Mat>& out, Mat in){
  int size = 200;
  if(in.rows!=200 || in.cols!=200){
    cout << "The input image was not 200x200 pixels.\nExiting.\n";
    exit(-1);
  }
  for(int i=0;i<size;i+=cropsize){
    for(int j=0;j<size;j+=cropsize){
     Mat tmp = Mat::zeros(cropsize,cropsize,CV_32FC1);
     tmp = reshapeCol(in(Rect(i, j, cropsize, cropsize)));
     out.push_back(tmp);
    }
  }
  cout << "This is the size: " << out.size() << " and the average cols: " << out[0].rows << endl;
}

void textonFind(Mat& clus, Mat dictionary){
  if(clus.empty() || dictionary.empty()){
    ERR("Texton Find inputs were empty");
    exit(-1);
  }
  // Loop through input centers
  for(int h=0;h<clus.rows;h++){
    float distance = 0.0, nearest = 0.0;

    distance = abs(dictionary.at<float>(0,0) - clus.at<float>(h,0));
    nearest = dictionary.at<float>(0,0);

    // Compare current centre with all values in texton dictionary
    for(int k = 0; k < dictionary.rows; k++){
      if(abs(dictionary.at<float>(k,0) - clus.at<float>(h,0)) < distance){
        nearest = dictionary.at<float>(k,0);
        distance = abs(dictionary.at<float>(k,0) - clus.at<float>(h,0));
      }
    }
    // Replace input Center with closest Texton Center
    clus.at<float>(h,0) = nearest;
  }
}

// Create bins for each textonDictionary Value
void binLimits(vector<float>& tex){
  cout << "inside binLimits" << endl;

  vector<float> bins;
  bins.push_back(0);
  for(int i = 0;i <= tex.size()-1;i++){
      bins.push_back(tex[i] + 0.00001);
  }
  bins.push_back(256);

  for(int i=0;i<bins.size();i++)
    cout << "texDict: " << i << ": "<< tex[i] << " becomes: " << bins[i+1] << endl;
  tex.clear();
  tex = bins;
}

// Assign vector to Set to remove duplicates
void removeDups(vector<float>& v){
  cout << "inside.." << endl;
  sort(v.begin(), v.end());
  auto last = unique(v.begin(), v.end());
  v.erase(last, v.end());
}

vector<float> matToVec(Mat m){
  vector<float> v;
  for(int i=0;i<m.rows;i++){
    v.push_back(m.at<float>(i,0));
  }
  return v;
}
void vecToArr(vector<float> v, float* m){
  int size = v.size();
  for(int i=0;i<size;i++){
    m[i] = v[i];
  }
}

vector<float> createBins(Mat texDic){
  vector<float> v = matToVec(texDic);
  cout << "\n\nThis is the bin vector size BEFORE: " << v.size() << endl;
  binLimits(v);
  cout << "\n\nThis is the bin vector size AFTER: " << v.size() << endl;
  return v;
}

////////////////////////
// Key:               //
// 0 == TruePositive  //
// 1 == FalsePositive //
// 2 == TrueNegative  //
// 3 == FalseNegative //
////////////////////////
void addOneToAllButOne(string exp1, string exp2, map<string, vector<int> >& results){
  for(auto ent4 : results){
    if(exp1.compare(ent4.first)!=0 && exp2.compare(ent4.first)!=0){
      ent4.second[2] = 3;
    }
  }
}

void initROCcnt(vector<map<string, vector<int> > >& r, map<string, vector<Mat> > classImgs){
  cout << "initialising.. " << endl;
  map<string, vector<int> > a;
  for(auto const ent5 : classImgs){
    for(int i=0;i<4;i++){
      a[ent5.first].push_back(1);
    }
    cout << "Initilseing..." << ent5.first << endl;
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
  //cout << "Answer is: " << correct << " prediction is: " << prediction <<endl;
  if(correct.compare(prediction)==0){
    // If correct
    // add 1 to True Positive
    results[correct][0] += 1;
    // add 1 to True Positive to all other classes
    addOneToAllButOne(correct, "", results);
  }else if(correct.compare(prediction)!=0){
    // If incorrect
    // add 1 to False Positive for Predicted
    results[prediction][1] += 1;
    // add 1 to False Negative to Correct
    results[correct][3] += 1;
    // add 1 to True Positive to all other classes
    addOneToAllButOne(correct, prediction, results);
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
void calcROCVals(map<string, vector<vector<int> > > in, map<string, vector<vector<int> > >& out, vector<string> clssNmes){
  assert(clssNmes.size() == in.size());
  vector<int> a;

  // Go through each Class
  for(int i=0;i<clssNmes.size();i++){
    string curCls = clssNmes[i];
    cout << "Class: " << curCls << "\n\n";

    // Go through each test iteration
    for(int j=0;j<in[curCls][0].size();j++){
      // Calculate TPR
      int TPR, TP, FN;
      TP = in[curCls][0][j];
      FN = in[curCls][3][j];
      TPR = (TP/TP+FN);
      cout << " TP: " << TP << " FN: " << FN << " TPR: " << TPR;

      // Calculate FPR
      int FPR, FP, TN;
      FP = in[curCls][1][j];
      TN = in[curCls][2][j];
      FPR = (FP/FP+TN);
      cout << " FP: " << FP << " TN: " << TN << " FPR: " << FPR;

      // Pushback vector for each test
      out[curCls].push_back(a);
      // Push back results
      out[curCls][j].push_back(TPR);
      out[curCls][j].push_back(FPR);
      double TPd = TP, TNd = TN, FPd = FP, FNd = FN;

      cout << "   accuracy: " << ((TPd+TNd)/(TPd+TNd+FPd+FNd))*100 << endl;
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

void testNovelImgHandle(int clsAttempts, int numClusters, map<string, vector<int> >& results, map<string, vector<Mat> > classImgs, map<string, vector<Mat> > savedClassHist, map<string, Scalar> Colors){
  int clsFlags = KMEANS_PP_CENTERS;
  TermCriteria clsTc(TermCriteria::MAX_ITER + TermCriteria::EPS, 1000, 0.0001);
  BOWKMeansTrainer novelTrainer(numClusters, clsTc, clsAttempts, clsFlags);

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
      for(int h=0;h<ent.second.size();h++){
        if(ent.second[h].rows != ent.second[h].cols){
          ERR("Novel input image was now square. Exiting");
          exit(-1);
        }
        int imgSize = ent.second[h].rows;
        Mat disVals = Mat(imgSize,imgSize,CV_8UC3);

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
        segmentImg(test, hold);

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
           if(high>CHISQU_threshold || secHigh<CHISQU_threshold){
             prediction = "Unknown";
           }else{
             prediction = match;
           }

           // Populate Window with predictions
           if(prediction.compare(ent.first)==0){
             Correct += 1;
             rectangle(disVals, Rect(discols, disrows,cropsize,cropsize), Colors["Correct"], -1, 8, 0);
           }else if(prediction.compare("Unknown")==0){
             Unknown +=1;
             rectangle(disVals, Rect(discols, disrows,cropsize,cropsize), Colors[prediction], -1, 8, 0);
           }else{
             Incorrect += 1;
             rectangle(disVals, Rect(discols, disrows,cropsize,cropsize), Colors["Incorrect"], -1, 8, 0);
           }
          // Save ROC data to results, clsAttempts starts at 0 so is -1
          cacheTestdata(ent.first, match, results);

      }
      //  imshow("mywindow", disVals);
      //  waitKey(500);
    }
    // END OF CLASS, CONTINUING TO NEXT CLASS //
  }

  // namedWindow("ROC", CV_WINDOW_AUTOSIZE);
  // Mat roc = Mat(400,400,CV_8UC3, Scalar(255,255,255));
  // line(roc, Point(0,400), Point(400,0), Scalar(0,0,255), 1, 8, 0);
  // imshow("ROC", roc);
}

int main( int argc, char** argv ){
  cout << "\n\n.......Starting Program...... \n\n" ;

  //////////////////////////////
  // Create Texton vocabulary //
  //////////////////////////////
  #if DICTIONARY_BUILD == 1
    cout << "\n\n.......Generating Texton Dictionary...... \n" ;
    int dictSize = 10;
    int attempts = 5;
    int flags = KMEANS_PP_CENTERS;
    TermCriteria tc(TermCriteria::MAX_ITER + TermCriteria::EPS, 1000, 0.0001);
    BOWKMeansTrainer bowTrainer(dictSize, tc, attempts, flags);

    map<string, vector<Mat> > textonImgs;
    path p = "../../../TEST_IMAGES/kth-tips/classes";
    loadClassImgs(p, textonImgs);

    Mat dictionary;
    for(auto const ent1 : textonImgs){
      for(int j=0;j<ent1.second.size();j++){
        Mat in = Mat::zeros(200,200,CV_32FC1);
        Mat hold = Mat::zeros(200,200,CV_32FC1);
        // Send img to be filtered, and responses aggregated with addWeighted
        in = ent1.second[j];
        if(!in.empty())
          filterHandle(in, hold);

        // Segment the 200x200pixel image into 400x1 Mats(20x20)
        vector<Mat> test;
        segmentImg(test, hold);

        // Push each saved Mat to bowTrainer
        for(int k = 0; k < test.size(); k++){
          if(!test[k].empty()){
            bowTrainer.add(test[k]);
          }
        }
      }
      cout << "This is the bowTrainer.size(): " << bowTrainer.descripotorsCount() << endl;
      // Generate 10 clusters per class and store in Mat
      dictionary.push_back(bowTrainer.cluster());
      bowTrainer.clear();
    }

    vector<float> bins = createBins(dictionary);

    removeDups(bins);

    //Save to file
    cout << "Saving Dictionary.." << endl;
    FileStorage fs("dictionary.xml",FileStorage::WRITE);
    fs << "vocabulary" << dictionary;
    fs << "bins" << bins;
    fs.release();
  #endif


  #if MODEL_BUILD == 1
  cout << "\n\n........Generating Class Models from Imgs.........\n";
    ///////////////////////////////////////////////////////////
    // Get histogram responses using vocabulary from Classes //
    ///////////////////////////////////////////////////////////

    cout << "\n\n........Loading Texton Dictionary.........\n";

    // Load TextonDictionary
    Mat dictionary;
    vector<float> m;
      FileStorage fs("dictionary.xml",FileStorage::READ);
      fs["vocabulary"] >> dictionary;
      fs["bins"] >> m;
      if(!fs.isOpened()){
        ERR("Unable to open Texton Dictionary.");
        exit(-1);
      }
      fs.release();

      // Load Class imgs and store in classImgs map
      map<string, vector<Mat> > classImgs;
      path p = "../../../TEST_IMAGES/kth-tips/classes";
      loadClassImgs(p, classImgs);

      float bins[m.size()];
      vecToArr(m, bins);

      // Initilse Histogram parameters
      int histSize = m.size()-1;
      const float* histRange = {bins};
      bool uniform = false;
      bool accumulate = false;


      int clsNumClusters = 10;
      int clsAttempts = 5;
      int clsFlags = KMEANS_PP_CENTERS;
      TermCriteria clsTc(TermCriteria::MAX_ITER + TermCriteria::EPS, 1000, 0.0001);
      BOWKMeansTrainer classTrainer(clsNumClusters, clsTc, clsAttempts, clsFlags);

      cout << "\n\n.......Generating Models...... \n" ;

      map<string, vector<Mat> > classHist;

    // Cycle through Classes
    for(auto const ent1 : classImgs){
      // Cycle through each classes images
      cout << "\nClass: " << ent1.first << endl;
      for(int j=0;j < ent1.second.size();j++){
        Mat in, hold;

        // Send img to be filtered, and responses aggregated with addWeighted
        in = ent1.second[j];
         if(!in.empty())
            filterHandle(in, hold);

        // Segment the 200x200pixel image into 400x1 Mats(20x20)
        vector<Mat> test;
        segmentImg(test, hold);

        // Push each saved Mat to classTrainer
        for(int k = 0; k < test.size(); k++){
          if(!test[k].empty()){
            classTrainer.add(test[k]);
          }
        }

        // Generate 10 clusters per class and store in Mat
        Mat clus = Mat::zeros(clsNumClusters,1, CV_32FC1);
        clus = classTrainer.cluster();

        // Replace Cluster Centers with the closest matching texton
        textonFind(clus, dictionary);

        Mat out;
        calcHist(&clus, 1, 0, Mat(), out, 1, &histSize, &histRange, uniform, accumulate);
        classHist[ent1.first].push_back(out);

        classTrainer.clear();
      }
    }

    FileStorage fs2("models.xml",FileStorage::WRITE);
    // int numClasses = classHist.size();
    // fs2 << "Num_Models" << numClasses;
    int cont=0;
    for(auto const ent1 : classHist){
      stringstream ss;
      ss << "class_" << cont;
      fs2 << ss.str() << "{";
        fs2 << "Name" << ent1.first;
        fs2 << "Models" << "{";
          for(int i=0;i<ent1.second.size();i++){
            stringstream ss1;
            ss1 << "m_" << i;
            fs2 << ss1.str() << ent1.second[i];
          }
        fs2 << "}";
      fs2 << "}";
      cont++;
    }
    fs2.release();

  #endif


  #if NOVELIMG_TEST == 1
  cout << "\n\n.......Testing Against Novel Images...... \n" ;
    //////////////////////////////
    // Test Against Novel Image //
    //////////////////////////////


    // Load Images to be tested
    map<string, vector<Mat> > classImgs;
      path p = "../../../TEST_IMAGES/kth-tips/classes";
      loadClassImgs(p, classImgs);

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

      Colors["Correct"] = Scalar(0,255,0);
      Colors["Incorrect"] = Scalar(255,0,0);
      Colors["Unknown"] = Scalar(100,100,100);

      int count =0;
      for(auto const ent : savedClassHist){
        Colors[ent.first] = clsColor[count];
        count++;
      }

    // Create Img Legend
    Mat Key = Mat::zeros(400,200,CV_8UC3);
      int cnt=0;
      for(auto const ent1 : Colors){
        putText(Key, ent1.first, Point(10, 20+ cnt*20), CV_FONT_HERSHEY_SIMPLEX, 0.5, Scalar(255,0,100), 1, 8, false);
        rectangle(Key, Rect(100, 10 + cnt*20, 10,10), ent1.second, -1, 8, 0 );
        cnt++;
      }

    // Window for Legend display
    namedWindow("legendWin", CV_WINDOW_AUTOSIZE);
    imshow("legendWin", Key);
    // Window for segment prediction display
    namedWindow("mywindow", CV_WINDOW_AUTOSIZE);

    // Holds Class names, each holding a count for TP, FP, FN, FP Values
    vector<map<string, vector<int> > > results;

  int counter = 0;
  for(int numClusters=1;numClusters<10;numClusters++){
    initROCcnt(results, classImgs); // Initilse map
    cout << "This is the size of the results.." << results.size() << endl;
    int clsAttempts = 5;
    testNovelImgHandle(clsAttempts, numClusters, results[counter], classImgs, savedClassHist, Colors);
    counter++;
  }
  map<string, vector<vector<int> > > resByCls;
  map<string, vector<vector<int> > > ROCVals;
  vector<string> clsNmes;
  getClsNames(results[0], clsNmes); // Get class Names
  organiseResultByClass(results, resByCls, clsNmes);


  // print results, clsAttempts starts at 1 so is -1
  printResults(resByCls, clsNmes);

  calcROCVals(resByCls, ROCVals, clsNmes);

  #endif

  return 0;
}
