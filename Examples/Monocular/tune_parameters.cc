#include <iostream>
#include <fstream>
#include <sstream>
#include <iomanip>
#include <vector>
#include <opencv2/core/core.hpp>
#include <string>
using namespace std;


#include "SuperPointExtractor.h"
#include "SuperPointMatcher.h"

void LoadImages(const string &strSequence, vector<string> &vstrImageFilenames,
                vector<double> &vTimestamps);


const float TH_HIGH = 2.0f;
const float TH_LOW = 1.0f;
const float NN_RATIO = 0.95f;

int main(int argc, char **argv)
{
    if(argc != 2)
    {
        cerr << endl << "Usage: ./mono_kitti path_to_sequence" << endl;
        return 1;
    }

    vector<string> vstrImageFilenames;
    vector<double> vTimestamps;
    LoadImages(string(argv[1]), vstrImageFilenames, vTimestamps);

    int nImages = vstrImageFilenames.size();

    ORB_SLAM3::SuperPointExtractor Extractor(
        "/home/c12888mg/Code/DarkSLAM/SUPERSLAM3/Weights/SuperPointNet.pt",
        1000, 1.2f, 8, 0.015f);

    ORB_SLAM3::SuperPointMatcher Matcher(0.7f);

    cv::Mat first_im = cv::imread(vstrImageFilenames[0], cv::IMREAD_UNCHANGED);
    std::vector<cv::KeyPoint> keypoints1;
    cv::Mat descriptors1;
    std::vector<int> vLappingArea;

    Extractor(first_im, cv::Mat(), keypoints1, descriptors1, vLappingArea);
    // std::cout << "Image 0: extracted " << keypoints1.size() << " keypoints." << std::endl;

    vector<float> overall_dists;
    vector<float> overall_dists2;
    vector<float> dists2;
    bool flag = true;
    vector<int> reject_counter;
    vector<int> nn_counter;
    vector<int> th_counter;

    for(int ni=1; ni<nImages; ni++)
    {
        if(!flag)
            break;
        // Read image from file
        cv::Mat im = cv::imread(vstrImageFilenames[ni],cv::IMREAD_UNCHANGED); //,cv::IMREAD_UNCHANGED);

        if(im.empty())
        {
            cerr << endl << "Failed to load image at: " << vstrImageFilenames[ni] << endl;
            return 1;
        }


        std::vector<cv::KeyPoint> keypoints2;
        cv::Mat descriptors2;

        Extractor(im, cv::Mat(), keypoints2, descriptors2, vLappingArea);
        // std::cout << "Image " << ni << ": extracted " << keypoints2.size() << " keypoints." << std::endl;

        // Calculate distances between descriptors1 and descriptors2
        std::vector<cv::DMatch> th_matches;
        std::vector<cv::DMatch> nn_matches;
        std::vector<float> dists;
        

        int rejected = 0;
        int nn_matches_count = 0;
        int th_matches_count = 0;
        
        // int best_idx2 = -1;
        for(int i=0; i<descriptors1.rows; i++)
        {
            float best_dist = TH_HIGH;
            float best_dist2 = TH_HIGH;
            int best_idx = -1;
            cv::Mat desc1 = descriptors1.row(i);
            for(int j=0; j<descriptors2.rows; j++)
            {
                cv::Mat desc2 = descriptors2.row(j);
                float dist = Matcher.DescriptorDistance(desc1, desc2);
                if(dist < best_dist)
                {
                    best_dist = dist;
                    best_idx = j;
                }
                else if(dist < best_dist2)
                {
                    best_dist2 = dist;
                    // best_idx2 = j;
                }
            }
            // Store match
            // Hard threshold
            dists.push_back(best_dist);
            if(best_dist < TH_LOW)
            {
                th_matches.push_back(cv::DMatch(i, best_idx, best_dist));
                // cout << best_dist << endl;
                th_matches_count++;
            }
            else{
                // cout << "Rejected match with distance: " << best_dist << endl;
                rejected++;
            }
            // Nearest neighbor ratio test
            dists2.push_back(best_dist2);
            if(best_dist < NN_RATIO * best_dist2)
            {
                nn_matches.push_back(cv::DMatch(i, best_idx, best_dist));
                nn_matches_count++;
            }

        }
        reject_counter.push_back(rejected);
        nn_counter.push_back(nn_matches_count);
        th_counter.push_back(th_matches_count);

        std::cout << "Image " << ni << ": TH matches = " << th_matches.size()
                  << ", NN matches = " << nn_matches.size() << std::endl;
        // Average distance
        float avg_dist = 0.f;
        for(float d : dists){
            avg_dist += d;
        }
        avg_dist /= dists.size();
        // std::cout << "Average descriptor distance: " << avg_dist << std::endl;
        overall_dists.push_back(avg_dist);

        float avg_dist2 = 0.f;
        for(float d : dists2){
            avg_dist2 += d;
        }
        avg_dist2 /= dists2.size();
        // std::cout << "Average second nearest neighbor distance: " << avg_dist2 << std::endl;
        overall_dists2.push_back(avg_dist2);

        // Draw matches
        // TH matches
        // cv::Mat img_matches_th;
        // cv::drawMatches(first_im, keypoints1, im, keypoints2, th_matches, img_matches_th);
        // // NN matches
        // cv::Mat img_matches_nn;
        // cv::drawMatches(first_im, keypoints1, im, keypoints2, nn_matches, img_matches_nn);
        // // Show images (vstack)
        // cv::Mat img_combined;
        // cv::vconcat(img_matches_th, img_matches_nn, img_combined);
        // cv::imshow("Matches (Top: TH, Bottom: NN)", img_combined);
        // while(true)
        // {
        //     char c = (char)cv::waitKey(0);
        //     if(c == 27 || c == 'q' || c == 'Q')
        //         flag = false;
        //     else if(c == 'n' || c == 'N')
        //         break;
        // }
    }

    // Overall average distance
    float overall_avg_dist = 0.f;
    for(float d : overall_dists){
        overall_avg_dist += d;
    }
    overall_avg_dist /= overall_dists.size();
    cout << "Overall average descriptor distance: " << overall_avg_dist << std::endl;

    float avg_dists2 = 0.f;
    for(float d : overall_dists2){
        avg_dists2 += d;
    }
    avg_dists2 /= overall_dists2.size();
    cout << "Overall average second nearest neighbor distance: " << avg_dists2 << endl;

    cout << "Ratio: " << overall_avg_dist / avg_dists2 << std::endl;

    // Rejects per image
    float average_rejects = 0.f;
    for(int i : reject_counter){
        average_rejects += i;
    }
    average_rejects /= reject_counter.size();
    cout << "Average rejected matches per image: " << average_rejects << std::endl;
    // NN matches per image
    float average_nn = 0.f;
    for(int i : nn_counter){
        average_nn += i;
    }
    average_nn /= nn_counter.size();
    cout << "Average NN matches per image: " << average_nn << std::endl;
    // TH matches per image
    float average_th = 0.f;
    for(int i : th_counter){
        average_th += i;
    }
    average_th /= th_counter.size();
    cout << "Average TH matches per image: " << average_th << std::endl;

}

void LoadImages(const string &strPathToSequence, vector<string> &vstrImageFilenames, vector<double> &vTimestamps)
{
    ifstream fTimes;
    string strPathTimeFile = strPathToSequence + "/times.txt";
    fTimes.open(strPathTimeFile.c_str());
    while(!fTimes.eof())
    {
        string s;
        getline(fTimes,s);
        if(!s.empty())
        {
            stringstream ss;
            ss << s;
            double t;
            ss >> t;
            vTimestamps.push_back(t);
        }
    }

    string strPrefixLeft = strPathToSequence + "/image_0/";

    const int nTimes = vTimestamps.size();
    vstrImageFilenames.resize(nTimes);

    for(int i=0; i<nTimes; i++)
    {
        stringstream ss;
        ss << setfill('0') << setw(6) << i;
        vstrImageFilenames[i] = strPrefixLeft + ss.str() + ".png";
    }
}