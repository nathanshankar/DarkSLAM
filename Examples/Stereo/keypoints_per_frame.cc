#include<opencv2/core/core.hpp>

#include<System.h>

#include <filesystem>
#include<fstream>

using namespace std;

void LoadImages(const string &strPathToSequence, vector<string> &vstrImageLeft,
                vector<string> &vstrImageRight, vector<double> &vTimestamps);

int main(int argc, char **argv)
{
    if(argc != 2)
    {
        cerr << endl << "Usage: ./keypoints_per_frame path_to_sequence" << endl;
        return 1;
    }

    // Retrieve paths to images
    vector<string> vstrImageLeft;
    vector<string> vstrImageRight;
    vector<double> vTimestamps;
    LoadImages(string(argv[1]), vstrImageLeft, vstrImageRight, vTimestamps);

    const int nImages = vstrImageLeft.size();

    cout << "Images in left sequence: " << vstrImageLeft.size() << endl;
    cout << "Images in right sequence: " << vstrImageRight.size() << endl;

    vector<int> vKpsLeft, vKpsRight;

    // Feature extractor
    ORB_SLAM3::SuperPointExtractor Extractor(
    "/home/c12888mg/Code/DarkSLAM/SUPERSLAM3/Weights/SuperPointNet.pt",
    2000, 1.2f, 8, 0.015f);

    cv::Mat imLeft, imRight;

    for (int ni=1; ni<nImages; ni++)
    {
        // Read left and right images from file
        imLeft = cv::imread(vstrImageLeft[ni],cv::IMREAD_UNCHANGED); //,cv::IMREAD_UNCHANGED);
        imRight = cv::imread(vstrImageRight[ni],cv::IMREAD_UNCHANGED); //,cv::IMREAD_UNCHANGED);

        if(imLeft.empty() || imRight.empty())
        {
            cerr << endl << "Failed to load image at: "
                 << string(vstrImageLeft[ni]) << " or "
                 << string(vstrImageRight[ni]) << endl;
            continue;
        }

        std::vector<cv::KeyPoint> keypointsLeft, keypointsRight;
        cv::Mat descriptorsLeft, descriptorsRight;
        std::vector<int> vLappingArea;

        Extractor(imLeft, cv::Mat(), keypointsLeft, descriptorsLeft, vLappingArea);
        Extractor(imRight, cv::Mat(), keypointsRight, descriptorsRight, vLappingArea);

        // visualise keypoints left image
        cv::Mat vis_img_left;
        cv::drawKeypoints(imLeft, keypointsLeft, vis_img_left, cv::Scalar(0,255,0), cv::DrawMatchesFlags::DEFAULT);
        cv::imshow("Keypoints Left Image", vis_img_left);
        cv::waitKey(1);


        vKpsLeft.push_back(keypointsLeft.size());
        vKpsRight.push_back(keypointsRight.size()); 
    }

    double avgKeypointsL = 0.0;
    for(size_t i = 0; i < vKpsLeft.size(); i++)
    {
            avgKeypointsL += vKpsLeft[i];
    }
    avgKeypointsL /= vKpsLeft.size();
    cout << "average number of keypoints tracked per frame: " << avgKeypointsL << endl;

    double avgKeypointsR = 0.0;
    for(size_t i = 0; i < vKpsRight.size(); i++)
    {
            avgKeypointsR += vKpsRight[i];
    }
    avgKeypointsR /= vKpsRight.size();
    cout << "average number of keypoints tracked per frame: " << avgKeypointsR << endl;

    // save to text file
    ofstream keypointsFile("KeypointsPerFrame.txt");
    if(keypointsFile.is_open())
    {
            for(size_t i = 0; i < vKpsLeft.size(); i++)
            {
                    keypointsFile << vKpsLeft[i] << "," << vKpsRight[i] << '\n';
            }
            keypointsFile.close();
    }

}

void LoadImages(const string &strPathToSequence, vector<string> &vstrImageLeft,
                vector<string> &vstrImageRight, vector<double> &vTimestamps)
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
    string strPrefixRight = strPathToSequence + "/image_1/";

    const int nTimes = vTimestamps.size();
    vstrImageLeft.resize(nTimes);
    vstrImageRight.resize(nTimes);

    for(int i=1; i<nTimes; i++)
    {
        stringstream ss;
        ss << setfill('0') << setw(6) << i;
        vstrImageLeft[i] = strPrefixLeft + ss.str() + ".png";
        vstrImageRight[i] = strPrefixRight + ss.str() + ".png";
    }
}