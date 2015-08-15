///////////////////////////////////////
// Opencv BoW Texture classification //
///////////////////////////////////////

#include "opencv2/highgui/highgui.hpp"
#include <opencv2/opencv.hpp>
#include "opencv2/core/core.hpp"
#include <opencv2/nonfree/features2d.hpp>
#include <opencv2/legacy/legacy.hpp>
//#include "features2d.hpp" // For feature2d (typedef DescriptorExtractor)
#include <iostream>
#include <stdio.h>
#include <stdlib.h>
#include <fstream>
#include <string>
#include <dirent.h>
#include <unistd.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <algorithm> // Maybe fix DescriptorExtractor doesn't have a member 'create'

#include "filterbank.h" // Filterbank Handling Functions

#define DICTIONARY_BUILD 1
#define ERR(msg) printf("\n\nERROR!: %s Line %d\nExiting.\n", msg, __LINE__);

using std::vector;
using namespace cv;
using namespace std;

typedef vector<Mat> m;
typedef vector<m> mV;

void errorFunc(string input){
  cerr << "\n\nERROR!: " << input << "\nExiting.\n\n";
  exit(-1);
}

void warnFunc(string input){
  cerr << "\nWARNING!: " << input << endl;
};



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



void getNovelImgs(const char *inPath, map<string, vector<Mat> >& novelImgs){
  DIR *pdir = NULL;
  cout << "inpath : " << inPath << endl;
  pdir = opendir(inPath);
  // Check that dir was initialised correctly
  if(pdir == NULL){
    errorFunc("Unable to open directory.");
  }
  struct dirent *pent = NULL;

  // Continue as long as there are still values in the dir list
  while (pent = readdir(pdir)){
    if(pdir==NULL){
      errorFunc("Dir was not initialised correctly.");
    }

    // Extract and save img filename without extension
    stringstream ss;
    ss << pent->d_name;
    string fileNme =  ss.str();

    // If not file then continue iteration
    string dot[] = {".", ".."};
    if(fileNme.compare(dot[0]) == 0 || fileNme.compare(dot[1]) == 0){
      continue;
    }
    string cls;
    int lastIdx = fileNme.find_last_of(".");
    int classmk = fileNme.find_last_of("_");
    if(classmk>0){
      cls = fileNme.substr(0, classmk);
    } else{
      cls = fileNme.substr(0, lastIdx);
    }

    ss.str(""); // Clear stringstream

    // Read in image
    ss << inPath;
    ss << pent->d_name;
    string a = ss.str();
    Mat tmp = imread(a, CV_BGR2GRAY);

    if(tmp.data){
      // if(!novelImgs.count(cls))
      //   novelImgs[cls][0].create(tmp.rows, tmp.cols, tmp.type()); // if key not yet created initialise
      novelImgs[cls].push_back(tmp);
      cout << "pushing back: " << cls << endl;
    }else{
      warnFunc("Unable to read Image.");
    }
  }
  closedir(pdir);
  cout << "finished Reading Successfully.." << endl;
}

void listDir(const char *inPath, vector<m >& dirFiles, vector<Mat>& textDict, int flag){
  DIR *pdir = NULL;
  cout << "inpath : " << inPath << endl;
  pdir = opendir(inPath);
  // Check that dir was initialised correctly
  if(pdir == NULL){
    errorFunc("Unable to open directory.");
  }
  struct dirent *pent = NULL;

  // Continue as long as there are still values in the dir list

  m local;
  while (pent = readdir(pdir)){
    if(pdir==NULL){
      errorFunc("Dir was not initialised correctly.");
    }
    stringstream ss;
    ss << inPath << "/";
    ss << pent->d_name;
    string a = ss.str();
    Mat tmp = imread(a, CV_BGR2GRAY);
    if(tmp.data){
      switch(flag){
        case 1:
          local.push_back(tmp);
          break;
        case 2:
          textDict.push_back(tmp);
          break;
      }
    }else{
      warnFunc("Unable to read image.");
    }
  }
  dirFiles.push_back(local);
  closedir(pdir);
  cout << "finished Reading Successfully.." << endl;
}


void importImgs(mV &modelImg, vector<string> classes){
  string basePath = "../../../TEST_IMAGES/kth-tips/";

  int count = 0;
  for(int i=0;i<classes.size();i++){
    stringstream ss;
    ss << basePath;
    ss << classes[count];
    string b = ss.str();
    const char* a = b.c_str();
    cout << "this is the path.. " << a << endl;

    cout << "number: " << i << endl;
    // -----CHANGE!-----! //
    vector<Mat> plcHolder;
    // -----CHANGE!-----! //
    listDir(a, modelImg, plcHolder, 1);

    count ++;
  }
  cout << "This is the size of bread: " << modelImg[0].size() << ", cotton: " << modelImg[1].size() << ", cork: " << modelImg[2].size() << ", wood: " << modelImg[3].size() << ", AFoil: " << modelImg[4].size() << endl;
}

// void loadClassImgs(path p, map<string, vector<Mat> > &classImgs){
//     if(!exists(p) || !is_directory(p)){
//       cout << "\nClass loading path was not valid.\nExiting.\n";
//       exit(-1);
//     }
//     directory_iterator itr_end;
//     for(directory_iterator itr(p); itr != itr_end; ++itr){
//       string nme = itr -> path().string();
//       Mat img = imread(nme, CV_BGR2GRAY);
//       extractClsNme(nme);
//       cout << "Pushing back: " << nme << " img.size(): " << img.size() << endl;
//       classImgs[nme].push_back(img);
//     }
//     cout << "This is the total for each class: " << endl;
//     for(auto const & ent1 : classImgs){
//       cout << "Class: " << ent1.first;
//       cout << " Number: " << ent1.second.size();
//       cout << "\n";
//     }
//     cout << "\n";
// }

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
  int cropsize = 20, size = 200;
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

int main( int argc, char** argv ){
  cout << "\n\n.......Loading Model Images...... \n" ;

  mV modelImg;
  vector<string> classes = {"bread", "cotton", "cork", "wood", "alumniniumFoil"};

//  map<string, vector<Mat> > classImgs;
//  loadClassImgs(, classImgs);

  importImgs(modelImg, classes);

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


  vector<Mat> txtons;

  // -----CHANGE!-----! //
  vector<m> plcHolder;
  // -----CHANGE!-----! //

  listDir("../../../TEST_IMAGES/kth-tips/textons/",plcHolder ,txtons, 2);
  Mat dictionary;

  for(int i=0;i<modelImg.size();i++){
    for(int j=0;j<modelImg[i].size();j++){
      Mat hold;
      // Send img to be filtered, and responses aggregated with addWeighted
      if(!modelImg[i][j].empty())
        filterHandle(modelImg[i][j], hold);

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

  // Save to file
  cout << "Saving Dictionary.." << endl;
  FileStorage fs("dictionary.xml",FileStorage::WRITE);
  fs << "vocabulary" << dictionary;
  fs.release();

  #else
  ///////////////////////////////////////////////////////////
  // Get histogram responses using vocabulary from Classes //
  ///////////////////////////////////////////////////////////

  // Load TextonDictionary
  Mat dictionary;
  FileStorage fs("dictionary.xml",FileStorage::READ);
  fs["vocabulary"] >> dictionary;
  if(!fs.isOpened()){
    ERR("Unable to open Texton Dictionary.");
    exit(-1);
  }
  fs.release();

  vector<KeyPoint> keypoints;
  Mat response_hist;
  vector<string> class_names;
  Mat img;
  string filepath;
  map<string,Mat> classes_training_data;


  Ptr<FeatureDetector> detector(new SiftFeatureDetector());
  Ptr<DescriptorMatcher> matcher(new FlannBasedMatcher);
  Ptr<DescriptorExtractor> extractor(new SiftDescriptorExtractor);

  BOWImgDescriptorExtractor bowDE(extractor, matcher);
  bowDE.setVocabulary(dictionary);

  cout << "\n\n.......Generating Models...... \n" ;

  for(int i = 0; i < modelImg.size(); i++){
    for(int j = 0; j < modelImg[i].size(); j++){
      Mat bob = imread("../../bread.png", CV_BGR2GRAY);;
      detector->detect(bob, keypoints);

      bowDE.compute(modelImg[i][j], keypoints, response_hist);

      if(!classes_training_data.count(classes[i])){
        cout << "Creating new class.." << endl;
        classes_training_data[classes[i]].create(0,response_hist.cols,response_hist.type());
        class_names.push_back(classes[i]);
      }
      classes_training_data[classes[i]].push_back(response_hist);
    }
  }



  //////////////////////////////////////////
  // Create 1 against all SVM classifiers //
  //////////////////////////////////////////

  cout << "\n\n.......Generating Classifiers...... \n" ;

  map<string,CvSVM> classifiers;

  // Iterate through class models
  for(int i=0;i<classes.size();i++){
    string class_ = classes[i];
    cout << "Currently training class: " << class_ << endl;

    Mat samples(0, response_hist.cols, response_hist.type());
    Mat labels(0, 1, CV_32FC1);

    // Copy class samples and labels
    cout << "adding " << classes_training_data[class_].rows << " positive" << endl;
    samples.push_back(classes_training_data[class_]);

    // Set the label to 1 for positive match
    Mat class_label = Mat::ones(classes_training_data[class_].rows, 1, CV_32FC1);
    labels.push_back(class_label);

    for(map<string,Mat>::iterator it1 = classes_training_data.begin(); it1 != classes_training_data.end(); ++it1) {
      string not_class_ = (*it1).first;
      if(not_class_.compare(class_)==0)
        continue; //skip if not_class == currentclass

      samples.push_back(classes_training_data[not_class_]);

      // Set the label as zero for negative match
      class_label = Mat::zeros(classes_training_data[not_class_].rows, 1, CV_32FC1);
      labels.push_back(class_label);
    }

    cout << "Train.." << endl;
    Mat samples_32f; samples.convertTo(samples_32f, CV_32F);

    if(samples.rows == 0)
      continue; //phantom class?!

    // Train and store classifiers in Map
    classifiers[class_].train(samples_32f, labels);
  }



    //////////////////////////////
    // Test Against Novel Image //
    //////////////////////////////

  cout << "\n\n.......Testing Against Novel Images...... \n" ;

  map<string, map<string, int> > confusionMatrix;

 // Mat novelImage1 = imread(argv[1], CV_BGR2GRAY);
 //
 //  if(!novelImage1.data){
 //    cout << "novelImage unable to be loaded.\nExiting." << endl;
 //    exit(0);
 //  }
  map<string, vector<Mat> > novelImgs;

  getNovelImgs("../../../TEST_IMAGES/kth-tips/NovelTest/", novelImgs);

  double y,n, total;
  // for(int i=0;i<modelImg.size();i++){
  //   for(int j=0;j<modelImg[i].size();j++){
  for(map<string, vector<Mat> >::iterator it = novelImgs.begin(); it != novelImgs.end(); ++it){
   cout << "\nThe class is: " << it->first << endl;
    for(int i=0;i<it->second.size();i++){
      if(it->second[i].rows == 0){
        errorFunc("NovelImage map contains blank Mat.");
      };

      Mat responseNovel_hist;
      detector->detect(it->second[i], keypoints);
      bowDE.compute(it->second[i], keypoints, responseNovel_hist);

      float minf = FLT_MAX;
      string min_class;
      for(map<string,CvSVM>::iterator it = classifiers.begin(); it != classifiers.end(); ++it ){
        float res = (*it).second.predict(responseNovel_hist, true);
        if(res<minf){
          minf = res;
          min_class = (*it).first;
        }
      }
      if(min_class == it->first){
        cout << "YES, Predicted: " << min_class << " Actual: " << it->first << endl;
        y++;
      }else {
        cout << "NO, Predicted: " << min_class << " Actual: " << it->first << endl;
        n++;
      }
      total++;
    }
  }

  cout << "\nThe total ratio was:\nCorrect: " << y << "\nIncorrect: " << n << "\n\nPercent correct: " << (y/total)*100 << "\%\n\n";

// //      Add 1 to the class with the closest match
//     confusionMatrix[min_class][classes[i]]++;
  //   }
  // }

  #endif

  return 0;
}
