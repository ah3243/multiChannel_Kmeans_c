///////////////////////////////////////
// Opencv BoW Texture classification //
///////////////////////////////////////

#include "opencv2/highgui/highgui.hpp" // Needed for HistCalc
#include "opencv2/imgproc/imgproc.hpp" // Needed for HistCalc
#include <opencv2/opencv.hpp>
#include "opencv2/core/core.hpp"
#include <opencv2/nonfree/features2d.hpp>
#include <opencv2/legacy/legacy.hpp>
//#include "features2d.hpp" // For feature2d (typedef DescriptorExtractor)
#include <iostream> // General io
#include <stdio.h> // General io
#include <stdlib.h>
#include <fstream>
#include <string>
//#include <dirent.h>
#include <unistd.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <algorithm> // Maybe fix DescriptorExtractor doesn't have a member 'create'
#include <boost/filesystem.hpp>

#include "filterbank.h" // Filterbank Handling Functions
#include "imgCollection.h" // Img Handling Functions

#define DICTIONARY_BUILD 1

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
vector<float> binLimits(vector<float> tex){
  cout << "inside binLimits" << endl;

  vector<float> bins;
  bins.push_back(0);

  for(int i = 0;i <= tex.size()-1;i++){
      bins.push_back(tex[i] + 0.00001);
  }
  bins.push_back(256);
  bins.pop_back(); // To remove residual value
  for(int i=0;i<bins.size();i++)
    cout << "texDict: " << i << ": "<< tex[i] << " becomes: " << bins[i+1] << endl;
  return bins;
}

// Assign vector to Set to remove duplicates
vector<float> removeDups(vector<float> v){
  cout << "inside.." << endl;
  auto last = unique(v.begin(), v.end());
  v.erase(last, v.end());
  return v;
}

vector<float> matToVec(Mat m){
  vector<float> v;
  for(int i=0;i<m.rows;i++){
    v.push_back(m.at<float>(i,0));
  }
  sort(v.begin(), v.end());
  vector<float> o =  removeDups(v);
  // for(int i=0;i<o.size();i++){
  //   cout << "Number: " << i << ": " << o[i] << endl;
  // }
  return o;
}
void vecToArr(vector<float> v, float* m){
  int size = v.size();
  for(int i=0;i<size;i++){
    m[i] = v[i];
  }
}

void createBins(Mat texDic, float* m){
  vector<float> v = matToVec(texDic);
  vector<float> bins = binLimits(v);
  vecToArr(bins, m);
}

int main( int argc, char** argv ){
  cout << "\n\n.......Loading Model Images...... \n" ;

 map<string, vector<Mat> > classImgs;
 path p = "../../../TEST_IMAGES/kth-tips/classes";
 loadClassImgs(p, classImgs);

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


    Mat dictionary;
    for(auto const ent1 : classImgs){
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

    //Save to file
    cout << "Saving Dictionary.." << endl;
    FileStorage fs("dictionary.xml",FileStorage::WRITE);
    fs << "vocabulary" << dictionary;
    fs.release();

    float m[40];
    createBins(dictionary, m);
    cout << "Create hist.." << endl;
    int histSize = 20;
    float range[] = {0,100};
    const float* histRange = {range};
    bool uniform = true;
    bool accumulate = false;

    Mat out;
    calcHist(&dictionary, 1, 0, Mat(), out, 1, &histSize, &histRange, uniform, accumulate);
    cout << "This is the size: " << out << endl;
    return 0;
  #else
  ///////////////////////////////////////////////////////////
  // Get histogram responses using vocabulary from Classes //
  ///////////////////////////////////////////////////////////

  cout << "\n\n........Loading Texton Dictionary.........\n";

  // Load TextonDictionary
  Mat dictionary;
    FileStorage fs("dictionary.xml",FileStorage::READ);
    fs["vocabulary"] >> dictionary;
    if(!fs.isOpened()){
      ERR("Unable to open Texton Dictionary.");
      exit(-1);
    }
    fs.release();


    int clsDictSize = 10;
    int clsAttempts = 5;
    int clsFlags = KMEANS_PP_CENTERS;
    TermCriteria clsTc(TermCriteria::MAX_ITER + TermCriteria::EPS, 1000, 0.0001);
    BOWKMeansTrainer classTrainer(clsDictSize, clsTc, clsAttempts, clsFlags);


    cout << "\n\n.......Generating Models...... \n" ;

    map<string, vector<Mat> > classClusters;

  // Cycle through Classes
  for(auto const ent1 : classImgs){
    // Cycle through each classes images
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


      Mat clus = Mat::zeros(10,1, CV_32FC1);
      clus = classTrainer.cluster();
      cout << "These are the cluster centeres BEFORE: " << clus << endl;
      textonFind(clus, dictionary);
      cout << "These are the cluster centeres AFTER: " << clus << endl;

      return 0;
      classClusters[ent1.first].push_back(clus);

      classTrainer.clear();
    }
  }



//     //////////////////////////////
//     // Test Against Novel Image //
//     //////////////////////////////
//
//   cout << "\n\n.......Testing Against Novel Images...... \n" ;
//
//   map<string, map<string, int> > confusionMatrix;
//
//  // Mat novelImage1 = imread(argv[1], CV_BGR2GRAY);
//  //
//  //  if(!novelImage1.data){
//  //    cout << "novelImage unable to be loaded.\nExiting." << endl;
//  //    exit(0);
//  //  }
//   map<string, vector<Mat> > novelImgs;
//
//   getNovelImgs("../../../TEST_IMAGES/kth-tips/NovelTest/", novelImgs);
//
//   double y,n, total;
//   // for(int i=0;i<modelImg.size();i++){
//   //   for(int j=0;j<modelImg[i].size();j++){
//   for(map<string, vector<Mat> >::iterator it = novelImgs.begin(); it != novelImgs.end(); ++it){
//    cout << "\nThe class is: " << it->first << endl;
//     for(int i=0;i<it->second.size();i++){
//       if(it->second[i].rows == 0){
//         errorFunc("NovelImage map contains blank Mat.");
//       };
//
//       Mat responseNovel_hist;
//       detector->detect(it->second[i], keypoints);
//       bowDE.compute(it->second[i], keypoints, responseNovel_hist);
//
//       float minf = FLT_MAX;
//       string min_class;
//       for(map<string,CvSVM>::iterator it = classifiers.begin(); it != classifiers.end(); ++it ){
//         float res = (*it).second.predict(responseNovel_hist, true);
//         if(res<minf){
//           minf = res;
//           min_class = (*it).first;
//         }
//       }
//       if(min_class == it->first){
//         cout << "YES, Predicted: " << min_class << " Actual: " << it->first << endl;
//         y++;
//       }else {
//         cout << "NO, Predicted: " << min_class << " Actual: " << it->first << endl;
//         n++;
//       }
//       total++;
//     }
//   }
//
//   cout << "\nThe total ratio was:\nCorrect: " << y << "\nIncorrect: " << n << "\n\nPercent correct: " << (y/total)*100 << "\%\n\n";
//
// // //      Add 1 to the class with the closest match
// //     confusionMatrix[min_class][classes[i]]++;
//   //   }
//   // }

  #endif

  return 0;
}
