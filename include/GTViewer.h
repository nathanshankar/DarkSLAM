#pragma once

#include <sophus/se3.hpp>
#include <vector>
#include <string>
#include <atomic>

#include <mutex>
#include <condition_variable>
#include <deque>

namespace ORB_SLAM3
{

struct TimedPoseGT
{
    double t = 0.0;
    Sophus::SE3d T_WC;
};

std::vector<TimedPoseGT> LoadTUMGTPoseFile(const std::string& tum_file);

class GTViewer
{
public:
    GTViewer(std::vector<TimedPoseGT> gt_traj,
             bool use_sim3,
             double max_dt = 0.02,
             int min_pairs_for_align = 30);

    void PushEstimateSample(double t, const Sophus::SE3f& T_WC_f);
    void RunGTvsEstTrajectoryViewer();
    void RequestFinish();
    void SetKeyFramePoseProvider(std::function<std::vector<Sophus::SE3f>()> f);

private:
    // ✅ must be a complete type because we store it in std containers
    struct TimedEst
    {
        double t = 0.0;
        Sophus::SE3d T_WC;
    };

    std::vector<TimedPoseGT> mGT;
    bool mUseSim3 = false;
    double mMaxDt = 0.02;
    int mMinPairs = 30;

    std::mutex mMutexQ;
    std::condition_variable mCv;
    std::deque<TimedEst> mQueue;

    std::atomic_bool mFinishRequested{false};

    std::vector<TimedEst> PopSamples(size_t max);
    void WaitForSamplesOrFinish(int ms);
    std::function<std::vector<Sophus::SE3f>()> mGetKFs;
};

} // namespace ORB_SLAM3