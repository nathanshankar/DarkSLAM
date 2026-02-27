#include "GTViewer.h"

#include <pangolin/pangolin.h>

#include <Eigen/Core>
#include <Eigen/Geometry>
#include <Eigen/Dense>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <fstream>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <thread>

namespace ORB_SLAM3
{

// -------------------- Internal helpers (file-local) --------------------

static const TimedPoseGT* FindNearestGT(const std::vector<TimedPoseGT>& gt, double t)
{
    if(gt.empty()) return nullptr;

    auto it = std::lower_bound(gt.begin(), gt.end(), t,
        [](const TimedPoseGT& a, double time){ return a.t < time; });

    if(it == gt.begin()) return &(*it);
    if(it == gt.end())   return &gt.back();

    const TimedPoseGT& after  = *it;
    const TimedPoseGT& before = *(it - 1);
    return (std::abs(after.t - t) < std::abs(t - before.t)) ? &after : &before;
}

static Eigen::Matrix4d ComputeAlignmentUmeyama(
    const std::vector<Eigen::Vector3d>& src,  // GT points
    const std::vector<Eigen::Vector3d>& dst,  // EST points
    bool with_scaling)
{
    const int N = static_cast<int>(src.size());
    if (N < 3 || dst.size() != src.size())
        return Eigen::Matrix4d::Identity();

    Eigen::MatrixXd src_mat(3, N), dst_mat(3, N);
    for (int i = 0; i < N; ++i)
    {
        src_mat.col(i) = src[i];
        dst_mat.col(i) = dst[i];
    }

    return Eigen::umeyama(src_mat, dst_mat, with_scaling);
}

static Eigen::Vector3d TransformPoint(const Eigen::Matrix4d& T, const Eigen::Vector3d& p)
{
    Eigen::Vector4d ph(p.x(), p.y(), p.z(), 1.0);
    return (T * ph).head<3>();
}

static void DrawPolyline(const std::vector<Eigen::Vector3d>& pts)
{
    glBegin(GL_LINE_STRIP);
    for(const auto& p : pts)
        glVertex3d(p.x(), p.y(), p.z());
    glEnd();
}

// -------------------- Public loader --------------------

std::vector<TimedPoseGT> LoadTUMGTPoseFile(const std::string& tum_file)
{
    std::ifstream f(tum_file);
    if(!f.is_open())
        throw std::runtime_error("Cannot open GT file: " + tum_file);

    std::vector<TimedPoseGT> traj;
    traj.reserve(200000);

    std::string line;
    while(std::getline(f, line))
    {
        if(line.empty() || line[0] == '#') continue;

        std::istringstream iss(line);
        double t, tx, ty, tz, qx, qy, qz, qw;
        if(!(iss >> t >> tx >> ty >> tz >> qx >> qy >> qz >> qw))
            continue;

        Eigen::Quaterniond q(qw, qx, qy, qz);
        q.normalize();

        Sophus::SE3d T_WC(q, Eigen::Vector3d(tx, ty, tz));
        traj.push_back({t, T_WC});
    }

    std::sort(traj.begin(), traj.end(),
              [](const TimedPoseGT& a, const TimedPoseGT& b){ return a.t < b.t; });

    std::cerr << "[GTViewer] Loaded GT poses: " << traj.size() << std::endl;
    return traj;
}

// -------------------- GTViewer implementation --------------------

// struct GTViewer::TimedEst
// {
//     double t = 0.0;
//     Sophus::SE3d T_WC;
// };

GTViewer::GTViewer(std::vector<TimedPoseGT> gt_traj,
                   bool use_sim3,
                   double max_dt,
                   int min_pairs_for_align)
    : mGT(std::move(gt_traj))
    , mUseSim3(use_sim3)
    , mMaxDt(max_dt)
    , mMinPairs(min_pairs_for_align)
{}

void GTViewer::PushEstimateSample(double t, const Sophus::SE3f& T_WC_f)
{
    TimedEst s;
    s.t = t;
    s.T_WC = T_WC_f.cast<double>();

    {
        std::lock_guard<std::mutex> lk(mMutexQ);
        mQueue.push_back(std::move(s));
        if(mQueue.size() > 5000) mQueue.pop_front();
    }
    mCv.notify_one();
}

void GTViewer::RequestFinish()
{
    mFinishRequested.store(true);
    mCv.notify_one();
}

std::vector<GTViewer::TimedEst> GTViewer::PopSamples(size_t max)
{
    std::vector<TimedEst> out;
    out.reserve(max);

    std::lock_guard<std::mutex> lk(mMutexQ);
    while(!mQueue.empty() && out.size() < max)
    {
        out.push_back(std::move(mQueue.front()));
        mQueue.pop_front();
    }
    return out;
}

void GTViewer::WaitForSamplesOrFinish(int ms)
{
    std::unique_lock<std::mutex> lk(mMutexQ);
    mCv.wait_for(lk, std::chrono::milliseconds(ms), [&]{
        return !mQueue.empty() || mFinishRequested.load();
    });
}

void GTViewer::SetKeyFramePoseProvider(std::function<std::vector<Sophus::SE3f>()> f)
{
    mGetKFs = std::move(f);
}

void GTViewer::RunGTvsEstTrajectoryViewer()
{
    pangolin::CreateWindowAndBind("GT vs Estimated Trajectory", 1280, 1600);

    // Set background color to white
    glClearColor(1.0f, 1.0f, 1.0f, 1.0f);

    glEnable(GL_DEPTH_TEST);
    glEnable(GL_BLEND);
    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);

    pangolin::OpenGlRenderState s_cam(
        pangolin::ProjectionMatrix(1024, 768, 500, 500, 512, 389, 0.1, 10000),
        pangolin::ModelViewLookAt(0, -30, -30, 0, 0, 0, 0, -1, 0) // was (0, -3, -3, 0, 0, 0, 0, -1, 0)
    );

    pangolin::Handler3D handler(s_cam);
    pangolin::View& d_cam = pangolin::CreateDisplay()
        .SetBounds(0.0, 1.0, 0.0, 1.0, -1024.0f/768.0f)
        .SetHandler(&handler);

    // pangolin::CreatePanel("menu").SetBounds(0.0, 1.0, 0.0, 0.2);
    // pangolin::Var<bool> menuFollowCam("menu.Follow Camera", true, true);
    // pangolin::Var<bool> menuShowAxes("menu.Show Axes", true, true);
    // pangolin::Var<bool> menuAutoAlign("menu.Auto Align", true, true);
    // pangolin::Var<bool> menuStop("menu.Stop", false, true);
    // pangolin::Var<bool> menuReset("menu.Reset", false, false);

    // We keep these for GT plotting + alignment caching
    std::vector<Eigen::Vector3d> gt_pts;
    std::vector<Eigen::Vector3d> est_pairs;
    std::vector<Eigen::Vector3d> gt_pairs;

    gt_pts.reserve(200000);
    est_pairs.reserve(200000);
    gt_pairs.reserve(200000);

    bool align_ready = false;
    Eigen::Matrix4d T_gt_to_est = Eigen::Matrix4d::Identity();
    Eigen::Vector3d last_cam_pos(0,0,0);

    while(!pangolin::ShouldQuit() && !mFinishRequested.load())
    {
        // Update from new samples (used for GT association + alignment)
        auto samples = PopSamples(/*max=*/10);
        for(const auto& s : samples)
        {
            const Eigen::Vector3d p_est = s.T_WC.translation();
            last_cam_pos = p_est;

            const TimedPoseGT* g = FindNearestGT(mGT, s.t);
            if(g && std::abs(g->t - s.t) <= mMaxDt)
            {
                const Eigen::Vector3d p_gt = g->T_WC.translation();
                gt_pts.push_back(p_gt);

                // Use frame samples for alignment pairs (OK for visualization)
                gt_pairs.push_back(p_gt);
                est_pairs.push_back(p_est);

                if((int)gt_pairs.size() >= mMinPairs)
                {
                    T_gt_to_est = ComputeAlignmentUmeyama(gt_pairs, est_pairs, mUseSim3);
                    align_ready = true;
                }
            }
        }

        // If we have a keyframe provider, use it to follow the latest KF
        if(mGetKFs)
        {
            auto kf_poses = mGetKFs(); // Twc
            if(!kf_poses.empty())
                last_cam_pos = kf_poses.back().translation().cast<double>();

            s_cam.SetModelViewMatrix(
                pangolin::ModelViewLookAt(
                    last_cam_pos.x(), last_cam_pos.y() - 3.0, last_cam_pos.z() - 3.0,
                    last_cam_pos.x(), last_cam_pos.y(),       last_cam_pos.z(),
                    0, -1, 0
                )
            );
        }
        // else if(menuFollowCam)
        // {
        //     // Fallback: follow latest frame sample
        //     s_cam.SetModelViewMatrix(
        //         pangolin::ModelViewLookAt(
        //             last_cam_pos.x(), last_cam_pos.y() - 3.0, last_cam_pos.z() - 3.0,
        //             last_cam_pos.x(), last_cam_pos.y(),       last_cam_pos.z(),
        //             0, -1, 0
        //         )
        //     );
        // }

        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
        d_cam.Activate(s_cam);

        // if(pangolin::Pushed(menuReset))
        // {
        //     gt_pts.clear();
        //     est_pairs.clear();
        //     gt_pairs.clear();
        //     align_ready = false;
        //     T_gt_to_est.setIdentity();
        // }

        // if(menuStop)
        // {
        //     pangolin::FinishFrame();
        //     continue;
        // }

        // if(menuShowAxes)
        // {
        //     glLineWidth(2.0f);
        //     glBegin(GL_LINES);

        //     glColor3f(1.0f, 0.0f, 0.0f); glVertex3f(0,0,0); glVertex3f(1,0,0);
        //     glColor3f(0.0f, 1.0f, 0.0f); glVertex3f(0,0,0); glVertex3f(0,1,0);
        //     glColor3f(0.0f, 0.0f, 1.0f); glVertex3f(0,0,0); glVertex3f(0,0,1);

        //     glEnd();
        //     glLineWidth(1.0f);
        // }

        glLineWidth(3.0f);

        // ---------------- Estimated (red): draw from KeyFrames (loop-closure corrected) ----------------
        std::vector<Eigen::Vector3d> est_pts_draw;
        if(mGetKFs)
        {
            auto kf_poses = mGetKFs(); // vector<Sophus::SE3f> Twc
            est_pts_draw.reserve(kf_poses.size());
            for(const auto& Twc : kf_poses)
                est_pts_draw.push_back(Twc.translation().cast<double>());
        }
        // If provider isn't set, you could draw nothing, or keep a separate frame trajectory buffer.
        // Here we draw nothing if no provider.
        glColor3f(1.0f, 0.0f, 0.0f);
        if(!est_pts_draw.empty())
            DrawPolyline(est_pts_draw);

        // ---------------- GT (green), aligned into estimate frame ----------------
        glColor3f(0.0f, 1.0f, 0.0f);
        if(align_ready)
        {
            glBegin(GL_LINE_STRIP);
            for(const auto& p_gt : gt_pts)
            {
                Eigen::Vector3d p = TransformPoint(T_gt_to_est, p_gt);
                glVertex3d(p.x(), p.y(), p.z());
            }
            glEnd();
        }

        glLineWidth(1.0f);
        pangolin::FinishFrame();

        if(samples.empty())
            WaitForSamplesOrFinish(/*ms=*/10);
    }
}
} // namespace ORB_SLAM3