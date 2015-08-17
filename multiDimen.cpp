///////////////////////////////////////
// Opencv BoW Texture classification //
///////////////////////////////////////

#include "opencv2/highgui/highgui.hpp" // Needed for HistCalc
#include "opencv2/imgproc/imgproc.hpp" // Needed for HistCalc
//#include "opencv2/imgproc/rectange.hpp" // Needed for Rectanges
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


    int clsNumClusters = 50;
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
    //////////////////////////////
    // Test Against Novel Image //
    //////////////////////////////

  cout << "\n\n.......Testing Against Novel Images...... \n" ;

  // Load TextonDictionary
  Mat dictionary;
  vector<float> m;
    FileStorage fs("dictionary.xml",FileStorage::READ);
    if(!fs.isOpened()){
      ERR("Unable to open Texton Dictionary.");
      exit(-1);
    }

    fs["vocabulary"] >> dictionary;
    fs["bins"] >> m;
    float bins[m.size()];
    vecToArr(m, bins);
    fs.release();

    // Load Class imgs and store in classImgs map
    map<string, vector<Mat> > classImgs;
    path p = "../../../TEST_IMAGES/kth-tips/classes";
    loadClassImgs(p, classImgs);

  Mat in, hold;
  map<string, vector<Mat> > savedClassHist;

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

  // Initilse Histogram parameters
  int histSize = m.size()-1;
  const float* histRange = {bins};
  bool uniform = false;
  bool accumulate = false;

  // Initialise Clustering Parameters
  int clsNumClusters = 50;
  int clsAttempts = 5;
  int clsFlags = KMEANS_PP_CENTERS;
  TermCriteria clsTc(TermCriteria::MAX_ITER + TermCriteria::EPS, 1000, 0.0001);
  BOWKMeansTrainer novelTrainer(clsNumClusters, clsTc, clsAttempts, clsFlags);

map<string, Scalar> Colors;
  // Block Colors
  vector<Scalar> clsColor;
    clsColor.push_back(Scalar(255,0,0)); // Red
    clsColor.push_back(Scalar(0,255,0)); // Green
    clsColor.push_back(Scalar(0,0,250)); // Blue
    clsColor.push_back(Scalar(255,255,0));
    clsColor.push_back(Scalar(0,255,100));
    clsColor.push_back(Scalar(0,255,255));
    clsColor.push_back(Scalar(100,100,100));

  Colors["Unknown"] = Scalar(100,100,100);
  int count =0;
  for(auto const ent : savedClassHist){
    Colors[ent.first] = clsColor[count];
    cout << "and again.." << endl;
    count++;
  }

  Mat Key = Mat::zeros(400,200,CV_8UC3);
  int cnt=0;
  for(auto const ent1 : Colors){
    putText(Key, ent1.first, Point(10, 20+ cnt*20), CV_FONT_HERSHEY_SIMPLEX, 0.5, Scalar(255,0,100), 1, 8, false);
    rectangle(Key, Rect(100, 10 + cnt*20, 10,10), ent1.second, -1, 8, 0 );
    cnt++;
    //  rectangle(disVals, Rect(i,j,cropsize,cropsize), Colors[prediction], -1, 8, 0);
  }


  namedWindow("legendWin", CV_WINDOW_AUTOSIZE);
  imshow("legendWin", Key);


  Mat disVals = Mat(100,200,CV_8UC3);


  in = classImgs["cork"][0];

  // Send img to be filtered, and responses aggregated with addWeighted
   if(!in.empty()){
    filterHandle(in, hold);
   }

    // Segment the 200x200pixel image into 400x1 Mats(20x20)
    vector<Mat> test;
    segmentImg(test, hold);

//    void segmentNovelImg(vector<Mat>& out, Mat in){
      int cropsize = 20, size = 200;
      if(in.rows!=200 || in.cols!=200){
        cout << "The input image was not 200x200 pixels.\nExiting.\n";
        exit(-1);
      }
      for(int i=0;i<size;i+=cropsize){
        for(int j=0;j<size;j+=cropsize){
         Mat tmp = Mat::zeros(cropsize,cropsize,CV_32FC1);
         tmp = reshapeCol(in(Rect(i, j, cropsize, cropsize)));

         if(!tmp.empty()){
           novelTrainer.add(tmp);
         }

         // Generate 10 clusters per class and store in Mat
         Mat clus = Mat::zeros(clsNumClusters,1, CV_32FC1);
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
//             cout << "class: " << ent2.first << " MatchValue:" << val << endl;
             if(val < high){
               high = val;
               match = ent2.first;
             }
             if(val < secHigh && val > high && match.compare(ent2.first) != 0){
               secHigh = val;
               secMatch = ent2.first;
             }
           }
         }
         string prediction = "";
         if(secHigh-high>high){
           prediction = "Unknown";
         }else{
           prediction = match;
         }
         cout << "done..: " << match << endl;
         rectangle(disVals, Rect(i,j,cropsize,cropsize), Colors[prediction], -1, 8, 0);
        }
      }

//    }


//   if(secHigh-high>high){
//   cout << "\n\nThe input sample was matched as: " << match << " with a value of: " << high << endl;
//   cout << "The second best match was: " << secMatch << " with a value of: " << secHigh << endl;
//   cout << "\nThe variation between these two values is: " << secHigh-high << "\n\n";
// }else{
//   cout << "\n\nThis was a very close match with the distance being smaller than the chosen value.\n\n";
// }

namedWindow("mywindow", CV_WINDOW_AUTOSIZE);
imshow("mywindow", disVals);
waitKey(0);


  #endif

  return 0;
}
